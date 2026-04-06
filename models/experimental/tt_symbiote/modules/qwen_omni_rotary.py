# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN-backed rotary embeddings for Qwen3-Omni-MoE (replaces HF ``nn.Module`` RoPE layers).

Frequency layout matches Hugging Face ``modeling_qwen3_omni_moe`` (MRoPE / 1D / vision freqs).
``inv_freq`` matmuls and MRoPE interleaving run in **PyTorch** (same numerics as HF). The
embedding ``cos`` / ``sin`` (and vision ``freqs`` table when applicable) are computed with
``ttnn`` on device where a mesh is available.

For ``rope_type != "default"`` (dynamic / longrope / etc.), forwards delegate to the original
HF layer stored in ``_fallback_torch_layer``.
"""

from __future__ import annotations

import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor


def _x_dtype(x) -> torch.dtype:
    if isinstance(x, TorchTTNNTensor):
        if x.elem is not None:
            return x.elem.dtype
        return torch.bfloat16
    if isinstance(x, torch.Tensor):
        return x.dtype
    return torch.bfloat16


def _ttnn_replicated_to_torch(mesh_device, tensor: ttnn.Tensor, *, leading_dim: int) -> torch.Tensor:
    """Host readback for tensors uploaded with ``ReplicateTensorToMesh`` (see ``attention.py`` / ``linear.py``)."""
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        return ttnn.to_torch(tensor)
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    out = ttnn.to_torch(tensor, mesh_composer=mesh_composer)
    # Replicated logical tensor is stacked once per device along dim 0; keep a single copy.
    if out.shape[0] != leading_dim and out.shape[0] >= leading_dim:
        out = out[:leading_dim]
    return out


def _position_ids_torch(position_ids, mesh_device=None) -> torch.Tensor:
    """Return ``position_ids`` as a host ``torch.long`` tensor (HF MRoPE layout).

    Symbiote may pass a raw ``ttnn.Tensor`` on the mesh, a ``TorchTTNNTensor``, or PyTorch tensors.
    ``torch.as_tensor`` cannot ingest ``ttnn.Tensor`` — convert with ``ttnn.to_torch`` + mesh composer.
    """

    def _ttnn_to_torch_long(raw: ttnn.Tensor) -> torch.Tensor:
        ld = int(raw.shape[0])
        if mesh_device is not None and mesh_device.get_num_devices() > 1 and ld > 0:
            t = _ttnn_replicated_to_torch(mesh_device, raw, leading_dim=ld)
        else:
            t = ttnn.to_torch(raw)
        return t.long()

    # Unwrapped device tensor (see module_run / tree_map)
    if isinstance(position_ids, ttnn.Tensor):
        return _ttnn_to_torch_long(position_ids)

    if isinstance(position_ids, TorchTTNNTensor):
        t = position_ids.elem if position_ids.elem is not None else None
        if t is None and position_ids.ttnn_tensor is not None:
            t = _ttnn_to_torch_long(position_ids.ttnn_tensor)
        if isinstance(t, ttnn.Tensor):
            t = _ttnn_to_torch_long(t)
        position_ids = t

    if isinstance(position_ids, torch.Tensor):
        return position_ids.long()
    return torch.as_tensor(position_ids, dtype=torch.long)


def _apply_interleaved_mrope_torch(
    freqs: torch.Tensor,
    mrope_section: list[int],
) -> torch.Tensor:
    """Match HF ``Qwen3OmniMoeThinkerTextRotaryEmbedding.apply_interleaved_mrope``."""
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def _cos_sin_ttnn(emb: torch.Tensor, attention_scaling: float, device, out_dtype: torch.dtype):
    """Compute ``cos(emb)``, ``sin(emb)`` on device via ``ttnn``, return torch tensors."""
    mesh_mapper = ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None
    emb_tt = ttnn.from_torch(
        emb.float().contiguous(),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_mapper,
    )
    cos = ttnn.cos(emb_tt) * attention_scaling
    sin = ttnn.sin(emb_tt) * attention_scaling
    if out_dtype == torch.bfloat16:
        cos = ttnn.typecast(cos, ttnn.bfloat16)
        sin = ttnn.typecast(sin, ttnn.bfloat16)
    elif out_dtype == torch.float16:
        cos = ttnn.typecast(cos, ttnn.float16)
        sin = ttnn.typecast(sin, ttnn.float16)
    lead = int(emb.shape[0])
    cos_t = _ttnn_replicated_to_torch(device, cos, leading_dim=lead)
    sin_t = _ttnn_replicated_to_torch(device, sin, leading_dim=lead)
    return cos_t.to(dtype=out_dtype), sin_t.to(dtype=out_dtype)


class TTNNQwen3OmniMoeThinkerTextRotaryEmbedding(TTNNModule):
    """MRoPE for thinker text / talker text (same logic as HF ``Qwen3OmniMoeThinkerTextRotaryEmbedding``)."""

    def __init__(self):
        super().__init__()
        self._inv_freq_cpu: torch.Tensor | None = None
        self.attention_scaling = 1.0
        self.mrope_section: list[int] = [24, 20, 20]
        self.rope_type: str = "default"
        self.config = None

    @classmethod
    def from_torch(cls, torch_layer):
        m = cls()
        m._fallback_torch_layer = torch_layer
        m._inv_freq_cpu = torch_layer.inv_freq.detach().float().contiguous().clone()
        m.attention_scaling = float(getattr(torch_layer, "attention_scaling", 1.0))
        m.mrope_section = list(getattr(torch_layer, "mrope_section", [24, 20, 20]))
        m.rope_type = getattr(torch_layer, "rope_type", "default")
        m.config = getattr(torch_layer, "config", None)
        return m

    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def deallocate_weights_impl(self):
        return self

    def forward(self, x, position_ids):
        if self._fallback_torch_layer is not None and self.rope_type != "default":
            return self._fallback_torch_layer(x, position_ids)

        position_ids = _position_ids_torch(position_ids, self.device)
        out_dtype = _x_dtype(x)

        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        inv_freq = self._inv_freq_cpu.to(device=position_ids.device, dtype=torch.float32)
        bs = int(position_ids.shape[1])
        seq = int(position_ids.shape[2])
        d_half = int(inv_freq.shape[0])

        inv_freq_expanded = inv_freq.view(1, 1, d_half, 1).expand(3, bs, d_half, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(2, 3)
        freqs_t = _apply_interleaved_mrope_torch(freqs, self.mrope_section)
        emb = torch.cat((freqs_t, freqs_t), dim=-1)

        if self.device is None:
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            return cos.to(dtype=out_dtype), sin.to(dtype=out_dtype)

        cos, sin = _cos_sin_ttnn(emb, self.attention_scaling, self.device, out_dtype)
        return cos, sin


class TTNNQwen3OmniMoeTalkerRotaryEmbedding(TTNNQwen3OmniMoeThinkerTextRotaryEmbedding):
    """Same implementation as thinker text RoPE (HF uses an empty subclass)."""


class TTNNQwen3OmniMoeRotaryEmbedding(TTNNModule):
    """Standard 1D RoPE ``(cos, sin)`` for ``talker.code_predictor`` (HF ``Qwen3OmniMoeRotaryEmbedding``)."""

    def __init__(self):
        super().__init__()
        self._inv_freq_cpu: torch.Tensor | None = None
        self.attention_scaling = 1.0
        self.rope_type: str = "default"
        self.config = None

    @classmethod
    def from_torch(cls, torch_layer):
        m = cls()
        m._fallback_torch_layer = torch_layer
        m._inv_freq_cpu = torch_layer.inv_freq.detach().float().contiguous().clone()
        m.attention_scaling = float(getattr(torch_layer, "attention_scaling", 1.0))
        m.rope_type = getattr(torch_layer, "rope_type", "default")
        m.config = getattr(torch_layer, "config", None)
        return m

    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def deallocate_weights_impl(self):
        return self

    def forward(self, x, position_ids):
        if self._fallback_torch_layer is not None and self.rope_type != "default":
            return self._fallback_torch_layer(x, position_ids)

        position_ids = _position_ids_torch(position_ids, self.device)
        out_dtype = _x_dtype(x)

        inv_freq = self._inv_freq_cpu.to(device=position_ids.device, dtype=torch.float32)
        batch = int(position_ids.shape[0])
        seq = int(position_ids.shape[1])
        d_half = int(inv_freq.shape[0])

        inv_freq_expanded = inv_freq[None, :, None].expand(batch, -1, 1).float()
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        if self.device is None:
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            return cos.to(dtype=out_dtype), sin.to(dtype=out_dtype)

        cos, sin = _cos_sin_ttnn(emb, self.attention_scaling, self.device, out_dtype)
        return cos, sin


class TTNNQwen3OmniMoeVisionRotaryEmbedding(TTNNModule):
    """Vision freq table ``(seq_len, dim//2)`` matching HF ``Qwen3OmniMoeVisionRotaryEmbedding.forward``."""

    def __init__(self):
        super().__init__()
        self.dim = 0
        self.theta = 10000.0
        self._inv_freq_cpu: torch.Tensor | None = None

    @classmethod
    def from_torch(cls, torch_layer):
        m = cls()
        m._fallback_torch_layer = torch_layer
        m.dim = int(torch_layer.dim)
        m.theta = float(torch_layer.theta)
        m._inv_freq_cpu = torch_layer.inv_freq.detach().float().contiguous().clone()
        return m

    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def deallocate_weights_impl(self):
        return self

    def forward(self, seqlen: int):
        inv_freq = self._inv_freq_cpu
        d_half = int(inv_freq.shape[0])
        seq = torch.arange(seqlen, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(seq, inv_freq)

        if self.device is None:
            return freqs.to(dtype=inv_freq.dtype)

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        seq_col = ttnn.from_torch(
            seq.reshape(seqlen, 1).contiguous(),
            device=self.device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        inv_row = ttnn.from_torch(
            inv_freq.reshape(1, d_half).contiguous(),
            device=self.device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        freqs_tt = ttnn.matmul(seq_col, inv_row)
        freqs_host = _ttnn_replicated_to_torch(self.device, freqs_tt, leading_dim=seqlen)
        freqs_out = freqs_host.reshape(seqlen, d_half).to(dtype=inv_freq.dtype)
        return freqs_out
