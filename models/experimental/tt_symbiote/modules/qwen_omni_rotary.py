# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni-MoE RoPE: HF-compatible freqs; inv_freq/MRoPE interleave + cos/sin computed in torch on host. Outputs are returned as host tensors so downstream attention modules decide their own upload/layout (avoids redundant per-layer host-device round-trips). Non-default rope_type delegates to _fallback_torch_layer."""

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
    """Compute ``cos(emb)``, ``sin(emb)`` in torch and return host tensors.

    Audio/text output is mathematically unchanged: this matches HF's reference path
    (``emb.cos() * attention_scaling``, ``emb.sin() * attention_scaling``) and ``emb`` is
    already float32 from the caller. The previous implementation uploaded ``emb`` to the
    mesh, ran ``ttnn.cos/sin`` in float32, and read back to host, after which downstream
    attention modules re-uploaded ``cos/sin`` with their own layout/mesh strategy. That
    round-trip only added latency / PCIe pressure on every layer's RoPE call.
    ``device`` is intentionally unused (kept for signature compatibility with callers).
    """
    if emb.ndim != 3:
        raise ValueError(f"_cos_sin_ttnn expects [batch, seq, dim], got {tuple(emb.shape)}")
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos.to(dtype=out_dtype), sin.to(dtype=out_dtype)


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
        # Same math as HF reference: torch.outer(arange, inv_freq). The previous TTNN matmul
        # path uploaded both 1D operands and read the tiny result back, so it only added
        # host-device round-trips with no kernel benefit.
        inv_freq = self._inv_freq_cpu
        seq = torch.arange(seqlen, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(seq, inv_freq)
        return freqs.to(dtype=inv_freq.dtype)
