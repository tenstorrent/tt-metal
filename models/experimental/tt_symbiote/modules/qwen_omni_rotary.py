# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN-backed rotary embeddings for Qwen3-Omni-MoE (replaces HF ``nn.Module`` RoPE layers).

Frequency layout matches Hugging Face ``modeling_qwen3_omni_moe`` (MRoPE / 1D / vision freqs).
``inv_freq`` matmuls and MRoPE interleaving run in **PyTorch** (same numerics as HF).
Embedding ``cos`` / ``sin`` are computed with ``ttnn.cos`` / ``ttnn.sin``. On a multi-device mesh,
unaries may leave ``[B, S, H]`` **sequence-sharded** (``S/num_devices`` per chip); we then
``ttnn.all_gather`` along whichever dim shards ``S`` (often dim 1 on ``[B,S,H]``; mesh
unaries can shard another axis) so host readback matches full ``S`` before wrapping for attention.

For ``rope_type != "default"`` (dynamic / longrope / etc.), forwards delegate to the original
HF layer stored in ``_fallback_torch_layer``.
"""

from __future__ import annotations

import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import DistributedTensorConfig
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import tree_map


def _x_dtype(x) -> torch.dtype:
    if isinstance(x, TorchTTNNTensor):
        if x.elem is not None:
            return x.elem.dtype
        return torch.bfloat16
    if isinstance(x, torch.Tensor):
        return x.dtype
    return torch.bfloat16


def _is_host_ttnn_tensor_obj(x) -> bool:
    """True for mesh-backed ``ttnn`` tensors that may not pass ``isinstance(..., ttnn.Tensor)``."""
    if x is None or isinstance(x, (torch.Tensor, TorchTTNNTensor)):
        return False
    if isinstance(x, ttnn.Tensor):
        return True
    cls = type(x)
    mod = getattr(cls, "__module__", "") or ""
    return cls.__name__ == "Tensor" and ("ttnn" in mod or "_ttnn" in mod)


def _replicated_mesh_config(mesh_device):
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        return None
    return DistributedTensorConfig(
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )


def _set_rotary_outputs_replicated(module: TTNNModule, output_tensors):
    """RoPE outputs are fully replicated; avoid ``get_tensor_config_for_tensor`` shard heuristics + log spam."""
    cfg = _replicated_mesh_config(module.device)
    if cfg is None:
        return TTNNModule.set_output_tensors_config_impl(module, output_tensors)

    def apply(e):
        if isinstance(e, TorchTTNNTensor):
            e.set_distributed_tensor_config(cfg)
        return e

    return tree_map(apply, output_tensors)


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

    def _ttnn_to_torch_long(raw) -> torch.Tensor:
        ld = int(raw.shape[0])
        if mesh_device is not None and mesh_device.get_num_devices() > 1 and ld > 0:
            t = _ttnn_replicated_to_torch(mesh_device, raw, leading_dim=ld)
        else:
            t = ttnn.to_torch(raw)
        return t.long()

    if isinstance(position_ids, ttnn.Tensor) or _is_host_ttnn_tensor_obj(position_ids):
        return _ttnn_to_torch_long(position_ids)

    if isinstance(position_ids, TorchTTNNTensor):
        t = position_ids.elem if position_ids.elem is not None else None
        if t is None and position_ids.ttnn_tensor is not None:
            t = _ttnn_to_torch_long(position_ids.ttnn_tensor)
        elif isinstance(t, ttnn.Tensor) or _is_host_ttnn_tensor_obj(t):
            t = _ttnn_to_torch_long(t)
        position_ids = t

    if isinstance(position_ids, torch.Tensor):
        return position_ids.long()
    if _is_host_ttnn_tensor_obj(position_ids):
        return _ttnn_to_torch_long(position_ids)
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
    """Compute ``cos(emb)``, ``sin(emb)`` on device with ``ttnn``; stitch seq shards on mesh if needed."""
    if emb.ndim != 3:
        raise ValueError(f"_cos_sin_ttnn expects [batch, seq, dim], got {tuple(emb.shape)}")

    b, s, h = int(emb.shape[0]), int(emb.shape[1]), int(emb.shape[2])
    target_shape = (b, s, h)
    nd = int(device.get_num_devices())
    mesh_mapper = ttnn.ReplicateTensorToMesh(device) if nd > 1 else None

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

    if nd > 1:
        # Mesh cos/sin may shard sequence on any dim, not only dim=1; stitch before host readback.
        while True:
            gather_dim = None
            for d in range(len(cos.shape)):
                s_local = int(cos.shape[d])
                if s_local != s and s_local * nd == s:
                    gather_dim = d
                    break
            if gather_dim is None:
                break
            cos = ttnn.all_gather(cos, dim=gather_dim, num_links=1, topology=ttnn.Topology.Linear)
            sin = ttnn.all_gather(sin, dim=gather_dim, num_links=1, topology=ttnn.Topology.Linear)
            ttnn.synchronize_device(device)

    if nd <= 1:
        cos_t = ttnn.to_torch(cos)
        sin_t = ttnn.to_torch(sin)
    else:
        composer = ttnn.ConcatMeshToTensor(device, dim=0)
        cos_t = ttnn.to_torch(cos, mesh_composer=composer)
        sin_t = ttnn.to_torch(sin, mesh_composer=composer)
        if cos_t.shape != target_shape:
            if cos_t.numel() == b * s * h:
                cos_t = cos_t.reshape(target_shape)
                sin_t = sin_t.reshape(target_shape)
            elif cos_t.ndim == 3 and cos_t.shape[0] == nd * b and cos_t.shape[1] == s and cos_t.shape[2] == h:
                cos_t = cos_t[:b].contiguous()
                sin_t = sin_t[:b].contiguous()

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

    def set_output_tensors_config_impl(self, output_tensors):
        return _set_rotary_outputs_replicated(self, output_tensors)

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

    def set_output_tensors_config_impl(self, output_tensors):
        return _set_rotary_outputs_replicated(self, output_tensors)

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

    def set_output_tensors_config_impl(self, output_tensors):
        return _set_rotary_outputs_replicated(self, output_tensors)

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
