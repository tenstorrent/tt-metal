# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Omni specific TTNN modules."""

from __future__ import annotations

import math
import os
import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn
from models.tt_cnn.tt.builder import Conv2dConfiguration
from models.experimental.tt_symbiote.core.module import (
    TTNNModule,
    run_on_devices,
    DeviceArch,
    set_distributed_tensor_config,
)
from models.experimental.tt_symbiote.core.run_config import (
    DistributedConfig,
    DistributedTensorConfig,
    TracedRun,
    trace_enabled,
)
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import torch_dtype_to_ttnn_dtype, tree_map
from models.experimental.tt_symbiote.modules.attention import (
    TTNNBailingMoEAttention,
    TTNNPagedAttentionKVCache,
    TTNNSDPAAttention,
)
from models.experimental.tt_symbiote.modules.conv import NHWCConvPytorch, TTNNConv2dNHWC
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinear,
    TTNNLinearSilu,
    TTNNLinearIColShardedWRowSharded,
    TTNNLinearIColShardedWAllReduced,
    TTNNLinearIReplicatedWColSharded,
)
from models.experimental.tt_symbiote.modules.moe import (
    TTNNMoE,
    TTNNMoERouterDecode,
    TTNNExperts,
    TTNNGlm4MoeTopkRouter,
    TTNNGlm4MoeMLP,
    even_int_div,
    _safe_repeat,
    _to_torch_any,
)
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.modules.rope import TTNNRotaryPositionEmbedding
from models.experimental.tt_symbiote.modules.embedding import TTNNEmbedding
from models.experimental.tt_symbiote.modules.tensor import TTNNReshape
from models.experimental.tt_symbiote.utils.device_management import DeviceInit

logger = logging.getLogger(__name__)

# ``DistributedConfig`` / ``DistributedTensorConfig`` come from ``run_config`` above for Omni callers.


def _tensor_shardable_for_default_mesh_config(mesh_device, tensor) -> bool:
    """True when tensor dims divide cleanly along mesh batch/channel axes (default 2-D shard layout)."""
    if tensor is None:
        return True
    if len(tensor.shape) < 2:
        return False
    ms = mesh_device.shape
    return tensor.shape[-1] % ms[-1] == 0 and tensor.shape[0] % ms[0] == 0


def distributed_config_col_sharded_last_dim(mesh_device) -> DistributedTensorConfig:
    """Build metadata for last-dim column-sharded activations on ``mesh_device``."""

    def logical_shape_for_col_sharded(sharded_shape):
        shape_list = list(sharded_shape)
        n = int(mesh_device.get_num_devices())
        shape_list[-1] = int(shape_list[-1]) * int(n)
        return tuple(shape_list)

    return DistributedTensorConfig(
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
        logical_shape_fn=logical_shape_for_col_sharded,
    )


@dataclass
class QwenOmniReplicatedMeshTensorConfig(DistributedTensorConfig):
    """Replicated readback via ``ConcatMeshToTensor(dim=0)``; sets ``replicate_compose_slice_dim0_to_leading``."""

    replicate_compose_slice_dim0_to_leading: bool = True


def qwen_omni_replicated_concat_dim0_tensor_config(mesh_device) -> Optional[QwenOmniReplicatedMeshTensorConfig]:
    """Replicate + concat on dim 0, then slice when Omni ``to_torch`` patch is active."""
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        return None
    return QwenOmniReplicatedMeshTensorConfig(
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )


@dataclass
class QwenOmniDistributedConfig(DistributedConfig):
    """Qwen3-Omni: ambiguous replicated tensors use ``ConcatMeshToTensor(dim=0)`` + dim-0 slice on readback."""

    def get_tensor_config_for_tensor(self, module_name, tensor):
        if tensor is not None and not _tensor_shardable_for_default_mesh_config(self.mesh_device, tensor):
            logger.warning(
                "Could not determine tensor config for %s with shape %s. Assuming replication to all devices. "
                "Override set_output_tensors_config_impl in the module to set the correct config for this tensor.",
                module_name,
                getattr(tensor, "shape", tensor),
            )
            return QwenOmniReplicatedMeshTensorConfig(
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
            )
        return self.tensor_config


def qwen_omni_maybe_slice_replicated_mesh_compose(cfg, ttnn_tensor, result: torch.Tensor) -> torch.Tensor:
    """After mesh ``to_torch``, slice stacked replicas on dim 0 when ``cfg`` opts in (Qwen3-Omni)."""
    if getattr(cfg, "replicate_compose_slice_dim0_to_leading", False) and result.dim() >= 1:
        lead = int(ttnn_tensor.shape[0])
        if result.shape[0] > lead:
            return result[:lead].contiguous()
    return result


_QWEN_OMNI_NORMALRUN_TO_TORCH_PATCHED = False


def ensure_qwen_omni_normalrun_to_torch_slice() -> None:
    """Patch :meth:`NormalRun.to_torch` for Omni dim-0 slice after ``ttnn.to_torch`` when configured."""
    global _QWEN_OMNI_NORMALRUN_TO_TORCH_PATCHED
    if _QWEN_OMNI_NORMALRUN_TO_TORCH_PATCHED:
        return
    from models.experimental.tt_symbiote.core.run_config import NormalRun

    def to_torch(self):
        """Convert to PyTorch tensor."""
        if self.elem is not None and self.elem.device.type != "meta" and self.ttnn_tensor is None:
            return self.elem

        def _to_torch(self_inner):
            is_mesh_device = self_inner.ttnn_distributed_tensor_config is not None
            if is_mesh_device:
                result = ttnn.to_torch(
                    self_inner.ttnn_tensor,
                    mesh_composer=self_inner.ttnn_distributed_tensor_config.mesh_composer,
                ).to(self_inner.device, self_inner.dtype)
                result = qwen_omni_maybe_slice_replicated_mesh_compose(
                    self_inner.ttnn_distributed_tensor_config, self_inner.ttnn_tensor, result
                )
            else:
                result = ttnn.to_torch(self_inner.ttnn_tensor).to(self_inner.device, self_inner.dtype)
            return result

        result = self.elem
        if self.ttnn_tensor is not None and self.elem is None:
            result = _to_torch(self)
        assert result is not None, "Both ttnn_tensor and elem are None. This should not happen."
        if result.device.type == "meta" and self.ttnn_tensor is not None:
            result = _to_torch(self)
        self.elem = result if self.elem is None else self.elem
        return self.elem

    NormalRun.to_torch = staticmethod(to_torch)
    _QWEN_OMNI_NORMALRUN_TO_TORCH_PATCHED = True


class QwenOmniDeviceInit(DeviceInit):
    """Use with ``set_device(..., device_init=QwenOmniDeviceInit)`` for Qwen3-Omni on multi-device mesh."""

    @classmethod
    def init_state_impl(cls, device):
        ensure_qwen_omni_normalrun_to_torch_slice()
        return QwenOmniDistributedConfig(device)


def _codec_embedding_aligned_up(x: int, alignment: int) -> int:
    return ((int(x) + alignment - 1) // alignment) * alignment


def _pad_last_dim_row_major_codec_embedding(tensor, pad_amount: int, *, value):
    """Right-pad the last dimension (row-major)."""
    rank = len(tensor.shape)
    last = rank - 1
    padding = tuple((0, pad_amount if i == last else 0) for i in range(rank))
    return ttnn.pad(tensor, padding=padding, value=value)


def _slice_dim_codec_embedding(tensor, dim: int, length: int):
    starts = [0] * len(tensor.shape)
    ends = list(tensor.shape)
    ends[dim] = length
    return ttnn.slice(tensor, starts, ends)


def _prepare_embedding_indices(tt_indices):
    """Return ``(indices_uint32_or_unchanged, orig_seq_len)`` for ``ttnn.embedding``.

    ``ttnn.embedding`` requires indices in **UINT32** or BFLOAT16; symbiote maps ``torch.long`` → INT32.
    ``ttnn.typecast`` to UINT32 on row-major INT32 requires the **last dim** to be a multiple of 32;
    we pad with 0, then the caller must slice the **embedding output** on the sequence dim when
    ``orig_seq_len`` is not ``None``.
    """
    if not isinstance(tt_indices, ttnn.Tensor):
        return tt_indices, None
    dt = tt_indices.dtype
    if dt == ttnn.uint32 or dt == ttnn.bfloat16:
        return tt_indices, None

    rank = len(tt_indices.shape)
    seq_dim = rank - 1
    seq_len = int(tt_indices.shape[seq_dim])
    padded_len = _codec_embedding_aligned_up(seq_len, 32)
    orig_seq_len = None
    if padded_len != seq_len:
        tt_indices = _pad_last_dim_row_major_codec_embedding(tt_indices, padded_len - seq_len, value=0)
        orig_seq_len = seq_len

    tt_indices = ttnn.typecast(tt_indices, ttnn.uint32)
    return tt_indices, orig_seq_len


def _maybe_slice_embedding_output(out, orig_seq_len):
    """Undo sequence padding: embedding output is ``[..., seq, hidden]``."""
    if orig_seq_len is None:
        return out
    rank = len(out.shape)
    seq_dim = rank - 2
    return _slice_dim_codec_embedding(out, seq_dim, orig_seq_len)


@trace_enabled
class TTNNQwen3OmniMoeCodecPredictorEmbedding(TTNNEmbedding):
    """Codec predictor ModuleList embeddings: UINT32/padding_idx prep; hidden-sharded like TTNNEmbedding; mesh outputs col-sharded to match talker norms/attn."""

    @property
    def weight(self):
        return self.torch_layer.weight

    @property
    def padding_idx(self):
        return self.torch_layer.padding_idx

    def set_output_tensors_config_impl(self, output_tensors):
        """Col-shard last dim like decoder norms; fresh DistributedTensorConfig per leaf (reuse truncated long TTS)."""

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if self.device is not None and self.device.get_num_devices() > 1:
                    e.set_distributed_tensor_config(distributed_config_col_sharded_last_dim(self.device))
            return e

        if self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)
        return tree_map(set_col_sharded_config, output_tensors)

    def forward(self, tt_indices):
        tt_indices, orig_seq_len = _prepare_embedding_indices(tt_indices)
        pad = self.torch_layer.padding_idx
        pad_token = int(pad) if pad is not None and int(pad) >= 0 else None
        if pad_token is None:
            out = ttnn.embedding(
                tt_indices,
                self.tt_weight,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            out = ttnn.embedding(
                tt_indices,
                self.tt_weight,
                padding_idx=pad_token,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return _maybe_slice_embedding_output(out, orig_seq_len)


# ---------------------------------------------------------------------------
# Qwen3-Omni RoPE: extends shared rope.TTNNRotaryPositionEmbedding for mesh /
# replicated cos/sin quirks (narrow cos/sin; tile-aligned full rotary). Generic
# attention modules keep importing the baseline class from ``rope.py``.
# ---------------------------------------------------------------------------
class TTNNQwenOmniRotaryPositionEmbedding(TTNNRotaryPositionEmbedding):
    """Same API as ``TTNNRotaryPositionEmbedding`` with Omni-specific last-dim handling."""

    def forward(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ):
        if q.layout != ttnn.TILE_LAYOUT:
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if k.layout != ttnn.TILE_LAYOUT:
            k = ttnn.to_layout(k, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if cos.layout != ttnn.TILE_LAYOUT:
            cos = ttnn.to_layout(cos, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if sin.layout != ttnn.TILE_LAYOUT:
            sin = ttnn.to_layout(sin, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if len(sin.shape) == 3:
            sin = ttnn.unsqueeze(sin, dim=0)
        if len(cos.shape) == 3:
            cos = ttnn.unsqueeze(cos, dim=0)
        batch_size, n_q_heads, seq_len, head_dim = q.shape
        batch_size2, n_k_heads, seq_len2, head_dim2 = k.shape
        assert seq_len == seq_len2, "Query and Key sequence lengths must match."
        assert batch_size == batch_size2, "Query and Key batch sizes must match."
        assert head_dim == head_dim2, "Query and Key head dimensions must match."

        original_head_dim = head_dim
        original_seq_len = seq_len
        rotary_dim = cos.shape[-1]

        # Replicated cos/sin can be errantly all-gathered to head_dim×num_devices; slice to match Q/K.
        if rotary_dim > head_dim:
            cos = cos[:, :, :, :head_dim]
            sin = sin[:, :, :, :head_dim]
            rotary_dim = head_dim

        if rotary_dim < head_dim:
            q_rot = q[:, :, :, :rotary_dim]
            q_pass = q[:, :, :, rotary_dim:]
            k_rot = k[:, :, :, :rotary_dim]
            k_pass = k[:, :, :, rotary_dim:]

            padded_rotary_dim = rotary_dim
            if rotary_dim % 32 != 0:
                padded_rotary_dim = ((rotary_dim + 31) // 32) * 32
                cos = ttnn.pad(cos, [1, 1, cos.shape[-2], padded_rotary_dim], [0, 0, 0, 0], 0.0)
                sin = ttnn.pad(sin, [1, 1, sin.shape[-2], padded_rotary_dim], [0, 0, 0, 0], 0.0)
                q_rot = ttnn.pad(q_rot, [batch_size, n_q_heads, seq_len, padded_rotary_dim], [0, 0, 0, 0], 0.0)
                k_rot = ttnn.pad(k_rot, [batch_size2, n_k_heads, seq_len2, padded_rotary_dim], [0, 0, 0, 0], 0.0)

            q_rot_embedded = ttnn.experimental.rotary_embedding(q_rot, cos, sin)
            k_rot_embedded = ttnn.experimental.rotary_embedding(k_rot, cos, sin)

            if q_rot_embedded.shape[-2] != seq_len:
                q_rot_embedded = q_rot_embedded[:, :, :seq_len, :]
            if k_rot_embedded.shape[-2] != seq_len:
                k_rot_embedded = k_rot_embedded[:, :, :seq_len, :]
            if padded_rotary_dim != rotary_dim:
                q_rot_embedded = q_rot_embedded[:, :, :, :rotary_dim]
                k_rot_embedded = k_rot_embedded[:, :, :, :rotary_dim]

            q_rotated = ttnn.concat([q_rot_embedded, q_pass], dim=-1)
            k_rotated = ttnn.concat([k_rot_embedded, k_pass], dim=-1)
        else:
            padded_dim = rotary_dim
            if rotary_dim % 32 != 0:
                padded_dim = ((rotary_dim + 31) // 32) * 32
                pad_shape = [int(cos.shape[i]) for i in range(len(cos.shape) - 1)] + [padded_dim]
                cos = ttnn.pad(cos, pad_shape, [0, 0, 0, 0], 0.0)
                sin = ttnn.pad(sin, pad_shape, [0, 0, 0, 0], 0.0)
                q = ttnn.pad(q, [batch_size, n_q_heads, seq_len, padded_dim], [0, 0, 0, 0], 0.0)
                k = ttnn.pad(k, [batch_size2, n_k_heads, seq_len2, padded_dim], [0, 0, 0, 0], 0.0)

            q_rotated = ttnn.experimental.rotary_embedding(q, cos, sin)
            k_rotated = ttnn.experimental.rotary_embedding(k, cos, sin)

        if q_rotated.shape[-1] != original_head_dim:
            q_rotated = q_rotated[:, :, :, :original_head_dim]
        if k_rotated.shape[-1] != original_head_dim:
            k_rotated = k_rotated[:, :, :, :original_head_dim]
        if q_rotated.shape[-2] != original_seq_len:
            q_rotated = q_rotated[:, :, :original_seq_len, :]
        if k_rotated.shape[-2] != original_seq_len:
            k_rotated = k_rotated[:, :, :original_seq_len, :]

        return q_rotated, k_rotated


def _mesh_host_stitch_device_shards(tt_tensor: ttnn.Tensor, mesh_device) -> torch.Tensor | None:
    """Concat host tensors when each mesh device holds a shard on one logical dimension."""
    if mesh_device is None or not hasattr(mesh_device, "get_num_devices"):
        return None
    nd = int(mesh_device.get_num_devices())
    if nd <= 1:
        return None
    shards = ttnn.get_device_tensors(tt_tensor)
    if len(shards) != nd:
        return None
    local = tuple(int(x) for x in shards[0].shape)
    for t in (tt_tensor.shape, getattr(tt_tensor, "padded_shape", None)):
        if t is None:
            continue
        logical = tuple(int(x) for x in t)
        if len(logical) != len(local):
            continue
        for d in range(len(logical)):
            if local[d] != logical[d] and local[d] * nd == logical[d]:
                parts = [ttnn.to_torch(s).contiguous() for s in shards]
                return torch.cat(parts, dim=d)
    return None


def _ttnn_mesh_to_torch_one_replica(tt_tensor: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """Host tensor matching one logical replica (avoid ambiguous mesh compose behavior on replicated tensors)."""
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        return ttnn.to_torch(tt_tensor).contiguous()
    stitched = _mesh_host_stitch_device_shards(tt_tensor, mesh_device)
    if stitched is not None:
        return stitched.contiguous()
    shards = ttnn.get_device_tensors(tt_tensor)
    if shards:
        return ttnn.to_torch(shards[0]).contiguous()
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    result = ttnn.to_torch(tt_tensor, mesh_composer=composer)
    lead = int(tt_tensor.shape[0])
    if result.dim() >= 1 and int(result.shape[0]) > lead:
        result = result[:lead].contiguous()
    return result.contiguous()


def _upload_bct_replicated(x_t: torch.Tensor, mesh_device):
    """Upload host tensor with ReplicateTensorToMesh on multi-device meshes."""
    mesh_mapper = None
    if mesh_device is not None and hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() > 1:
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.from_torch(
        x_t.contiguous(),
        dtype=torch_dtype_to_ttnn_dtype(x_t.dtype),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
    )


def _ttnn_linear_mesh_all_gather_input_if_needed(input_tensor, mesh_dev, in_features: int):
    """When activations are width-sharded (``last * num_devices == in_features``), gather before ``ttnn.linear``."""
    if mesh_dev is None or not hasattr(mesh_dev, "get_num_devices") or mesh_dev.get_num_devices() <= 1:
        return input_tensor
    last_dim = int(input_tensor.shape[-1])
    n = int(mesh_dev.get_num_devices())
    if last_dim != in_features and last_dim * n == in_features:
        return ttnn.all_gather(
            input_tensor,
            dim=-1,
            cluster_axis=1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
    return input_tensor


@trace_enabled
class TTNNQwenOmniLinear(TTNNLinear):
    """Qwen3-Omni ``nn.Linear`` replacement: mesh width all-gather, TorchTTNNTensor upload, replicated readback."""

    def set_output_tensors_config_impl(self, output_tensors):
        """On mesh, default readback can concat the last dim (``out_features * N``); materialize logical width."""
        if self.device_state is None or self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        def _materialize_one_replica(e):
            if not isinstance(e, TorchTTNNTensor) or e.ttnn_tensor is None:
                return e
            t = e.ttnn_tensor
            n = int(t.shape[0])
            h = int(self.out_features)
            pt = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            if pt.shape[0] > n:
                pt = pt[:n]
            if pt.shape[-1] > h:
                pt = pt[..., :h]
            e.elem = pt.contiguous()
            e.ttnn_tensor = None
            if getattr(e, "_distributed_tensor_config", None) is not None:
                e._distributed_tensor_config = None
            return e

        return tree_map(_materialize_one_replica, output_tensors)

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through linear layer."""
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        if isinstance(input_tensor, TorchTTNNTensor):
            if input_tensor.ttnn_tensor is not None:
                input_tensor = input_tensor.ttnn_tensor
            elif input_tensor.elem is not None:
                dev = self.device
                mesh_mapper = (
                    ttnn.ReplicateTensorToMesh(dev)
                    if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1
                    else None
                )
                input_tensor = ttnn.from_torch(
                    input_tensor.elem.contiguous().to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=dev,
                    mesh_mapper=mesh_mapper,
                )
            else:
                raise TypeError("TTNNQwenOmniLinear.forward: TorchTTNNTensor has neither ttnn_tensor nor elem")

        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        input_tensor = _ttnn_linear_mesh_all_gather_input_if_needed(input_tensor, self.device, self.in_features)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)  # Add batch dimensions if needed
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        tt_output = ttnn.linear(input_tensor, self.tt_weight, bias=self.tt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [self.out_features])
        return tt_output


class TTNNQwen3OmniMoeAudioEncoderConvOutLinear(TTNNQwenOmniLinear):
    """``audio_tower.conv_out``: same mesh rules as :class:`TTNNQwenOmniLinear`; distinct type for upgrades / tests."""

    @classmethod
    def from_torch(cls, linear: nn.Linear):
        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
        )
        new_linear._fallback_torch_layer = linear
        new_linear.weight = linear.weight
        new_linear.bias = linear.bias
        return new_linear


class TTNNQwenOmniIColShardedWAllReduced(TTNNLinearIColShardedWAllReduced):
    """Omni vision / talker MLP fc2: after RS+AG the activation is full width per device; bias must be replicated.

    Base :class:`~models.experimental.tt_symbiote.modules.linear.TTNNLinearIColShardedWAllReduced` shards bias
    like fc1; adding it to the all-reduced output caused 1152 vs 144 style broadcast errors on mesh.
    """

    def move_weights_to_device_impl(self):
        if isinstance(self.tt_weight_host, torch.Tensor):
            self.tt_weight_host = preprocess_linear_weight(
                self.tt_weight_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),
            )
        if isinstance(self.tt_bias_host, torch.Tensor):
            self.tt_bias_host = preprocess_linear_bias(
                self.tt_bias_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None


class TTNNSnakeBeta(TTNNModule):
    """TTNN SnakeBeta (HF ``SnakeBeta`` in Qwen3-Omni code2wav decoder): x + 1/b * sin^2(x*a)."""

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.no_div_by_zero = 0.000000001
        self.alpha = None
        self.beta = None

    @classmethod
    def from_torch(cls, torch_layer, *args, **kwargs):
        in_features = int(getattr(torch_layer, "in_features", torch_layer.alpha.shape[0]))
        new_layer = cls(in_features)
        new_layer._fallback_torch_layer = torch_layer
        return new_layer

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()
        tl = self.torch_layer
        if tl is None:
            return
        w_alpha = tl.alpha.detach().float().contiguous()
        w_beta = tl.beta.detach().float().contiguous()
        mesh_mapper = None
        if self.device is not None and hasattr(self.device, "get_num_devices") and self.device.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        self.alpha = ttnn.from_torch(
            w_alpha,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.beta = ttnn.from_torch(
            w_beta,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def set_output_tensors_config_impl(self, output_tensors):
        if self.device_state is None or self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        def _materialize_one_replica(t):
            if not isinstance(t, TorchTTNNTensor) or t.ttnn_tensor is None:
                return t
            tt = t.ttnn_tensor
            pt = _ttnn_mesh_to_torch_one_replica(tt, self.device)
            t.elem = pt.contiguous()
            t.ttnn_tensor = None
            if getattr(t, "_distributed_tensor_config", None) is not None:
                t._distributed_tensor_config = None
            return t

        return tree_map(_materialize_one_replica, output_tensors)

    @staticmethod
    def _snake_beta_chunk_t() -> int:
        raw = os.environ.get("TT_SYMBIOTE_SNAKEBETA_CHUNK_T", "4096")
        try:
            v = int(raw)
        except ValueError:
            v = 4096
        return max(512, v)

    def _forward_fp32_core(
        self,
        input_fp32: ttnn.Tensor,
        alpha_exp: ttnn.Tensor,
        reciprocal_beta: ttnn.Tensor,
    ) -> ttnn.Tensor:
        x_times_alpha = ttnn.multiply(input_fp32, alpha_exp)
        sin_result = ttnn.sin(x_times_alpha)
        sin_squared = ttnn.pow(sin_result, 2.0)
        scaled_sin = ttnn.multiply(reciprocal_beta, sin_squared)
        result = ttnn.add(input_fp32, scaled_sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in (x_times_alpha, sin_result, sin_squared, scaled_sin):
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
        return result

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        out_dtype = input_tensor.dtype
        shape = tuple(int(s) for s in input_tensor.shape)
        if len(shape) != 3:
            raise ValueError(f"TTNNSnakeBeta expects rank-3 [B,C,T], got shape {shape}")
        b, c, t_len = shape

        chunk_t = self._snake_beta_chunk_t()
        alpha_expanded = ttnn.unsqueeze(self.alpha, 0)
        alpha_expanded = ttnn.unsqueeze(alpha_expanded, -1)
        beta_expanded = ttnn.unsqueeze(self.beta, 0)
        beta_expanded = ttnn.unsqueeze(beta_expanded, -1)
        alpha_exp = ttnn.exp(alpha_expanded)
        try:
            ttnn.deallocate(alpha_expanded)
        except Exception:
            pass
        beta_exp = ttnn.exp(beta_expanded)
        beta_plus_eps = ttnn.add(beta_exp, self.no_div_by_zero)
        reciprocal_beta = ttnn.reciprocal(beta_plus_eps)
        try:
            ttnn.deallocate(beta_expanded)
            ttnn.deallocate(beta_exp)
            ttnn.deallocate(beta_plus_eps)
        except Exception:
            pass

        if t_len <= chunk_t:
            if out_dtype != ttnn.float32:
                input_fp32 = ttnn.typecast(input_tensor, ttnn.float32)
            else:
                input_fp32 = input_tensor
            result = self._forward_fp32_core(input_fp32, alpha_exp, reciprocal_beta)
            if out_dtype != ttnn.float32:
                if input_fp32 is not input_tensor:
                    try:
                        ttnn.deallocate(input_fp32)
                    except Exception:
                        pass
                result = ttnn.typecast(result, out_dtype)
            try:
                ttnn.deallocate(alpha_exp)
                ttnn.deallocate(reciprocal_beta)
            except Exception:
                pass
            return result

        out_chunks = []
        for t0 in range(0, t_len, chunk_t):
            t1 = min(t0 + chunk_t, t_len)
            sl = ttnn.slice(input_tensor, (0, 0, t0), (b, c, t1))
            if out_dtype != ttnn.float32:
                sl_fp32 = ttnn.typecast(sl, ttnn.float32)
            else:
                sl_fp32 = sl
            res_fp32 = self._forward_fp32_core(sl_fp32, alpha_exp, reciprocal_beta)
            if out_dtype != ttnn.float32:
                out_chunks.append(ttnn.typecast(res_fp32, out_dtype))
            else:
                out_chunks.append(res_fp32)

        torch_parts = []
        mesh_dev = self.device
        for ch in out_chunks:
            torch_parts.append(_ttnn_mesh_to_torch_one_replica(ch, mesh_dev))
            try:
                ttnn.deallocate(ch)
            except Exception:
                pass
        merged_torch = torch.cat(torch_parts, dim=2)
        try:
            ttnn.deallocate(alpha_exp)
            ttnn.deallocate(reciprocal_beta)
        except Exception:
            pass
        return _upload_bct_replicated(merged_torch, mesh_dev)


class TTNNBailingMoEAttentionHostCachePositionRoPE(TTNNBailingMoEAttention):
    """BailingMoE decode variant — RoPE cos/sin from host ``cache_position_tensor``.

    Base :class:`~models.experimental.tt_symbiote.modules.attention.TTNNBailingMoEAttention`
    uses :meth:`~models.experimental.tt_symbiote.modules.attention.TTNNBailingMoEAttention._get_cur_pos_device_tensor`
    so decode is trace-friendly. This subclass keeps the older path where
    :meth:`BailingRotarySetup.get_cos_sin_for_decode` is driven by CPU indices
    while paged KV still receives a device ``cur_pos_tt`` built via
    :func:`ttnn.from_torch`.
    """

    def _forward_decode_paged(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[ttnn.Tensor],
        past_key_values: "TTNNPagedAttentionKVCache",
        cache_position: Optional[torch.LongTensor],
    ) -> tuple:
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        qkv_states = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query_states = ttnn.slice(qkv_states, [0, 0, 0], [batch_size, 1, q_size])
        key_states = ttnn.slice(qkv_states, [0, 0, q_size], [batch_size, 1, q_size + kv_size])
        value_states = ttnn.slice(qkv_states, [0, 0, q_size + kv_size], [batch_size, 1, q_size + 2 * kv_size])
        ttnn.deallocate(qkv_states)

        query_states = ttnn.reshape(query_states, (1, batch_size, self.num_heads, self.head_dim))
        key_states = ttnn.reshape(key_states, (1, batch_size, self.num_kv_heads, self.head_dim))
        value_states = ttnn.reshape(value_states, (1, batch_size, self.num_kv_heads, self.head_dim))

        if query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)
        if value_states.dtype != ttnn.bfloat16:
            value_states = ttnn.typecast(value_states, ttnn.bfloat16)

        query_states = ttnn.to_memory_config(query_states, ttnn.L1_MEMORY_CONFIG)
        key_states = ttnn.to_memory_config(key_states, ttnn.L1_MEMORY_CONFIG)

        query_states, key_states = self._apply_qk_norm(query_states, key_states)

        layer_idx = self._fallback_torch_layer.layer_idx

        if cache_position is None:
            cur_pos = past_key_values.get_seq_length(layer_idx)
            cache_position_tensor = torch.tensor([cur_pos], dtype=torch.int32)
        else:
            cp = cache_position
            if isinstance(cp, TorchTTNNTensor):
                cp = cp.to_torch
            if isinstance(cp, ttnn.Tensor):
                mesh_composer = None
                if hasattr(cp, "device") and cp.device() is not None and cp.device().get_num_devices() > 1:
                    mesh_composer = ttnn.ConcatMeshToTensor(cp.device(), dim=0)
                cp = ttnn.to_torch(cp, mesh_composer=mesh_composer)
            cache_position_tensor = cp.flatten()[:batch_size].to(torch.int32)

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        cur_pos_tt = ttnn.from_torch(
            cache_position_tensor,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if isinstance(query_states, ttnn.Tensor) and query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if isinstance(key_states, ttnn.Tensor) and key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)

        cos_ttnn, sin_ttnn = self._rotary_setup.get_cos_sin_for_decode(cache_position_tensor)

        batch_grid = ttnn.num_cores_to_corerangeset(batch_size, self.device.compute_with_storage_grid_size(), True)

        rope_shard_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        cos_ttnn = ttnn.to_memory_config(cos_ttnn, rope_shard_mem)
        sin_ttnn = ttnn.to_memory_config(sin_ttnn, rope_shard_mem)

        q_shard_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        query_states = ttnn.to_memory_config(query_states, q_shard_mem)

        k_shard_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, key_states.shape[-1]),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        key_states = ttnn.to_memory_config(key_states, k_shard_mem)

        trans_mat = self._rotary_setup.get_trans_mat_decode_sharded(batch_size)

        query_states = ttnn.experimental.rotary_embedding_llama(
            query_states, cos_ttnn, sin_ttnn, trans_mat, is_decode_mode=True
        )
        key_states = ttnn.experimental.rotary_embedding_llama(
            key_states, cos_ttnn, sin_ttnn, trans_mat, is_decode_mode=True
        )

        num_cores = batch_size
        compute_grid = self.device.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
        kv_vol = key_states.volume() // key_states.padded_shape[-1] // num_cores
        kv_shard = ttnn.ShardSpec(
            shard_grid,
            [kv_vol, key_states.padded_shape[-1]],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        kv_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            kv_shard,
        )
        key_states = ttnn.to_memory_config(key_states, kv_mem)
        value_states = ttnn.to_memory_config(value_states, kv_mem)

        past_key_values.paged_update_on_device(
            key_states,
            value_states,
            layer_idx=layer_idx,
            current_pos=cur_pos_tt,
        )
        ttnn.deallocate(key_states)
        ttnn.deallocate(value_states)

        attn_output = past_key_values.paged_sdpa_decode(
            query_states,
            layer_idx,
            current_pos=cur_pos_tt,
            scale=self.scaling,
            program_config=self.sdpa.decode_program_config,
            compute_kernel_config=self.sdpa.compute_kernel_config,
        )

        sdpa_output_memcfg = ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),
            core_grid=ttnn.CoreGrid(y=1, x=batch_size),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        attn_output = ttnn.to_memory_config(attn_output, sdpa_output_memcfg)

        attn_output = ttnn.experimental.nlp_concat_heads_decode(
            attn_output,
            num_heads=self.num_heads,
        )
        attn_output = self.dense(attn_output)

        if batch_size < 32:
            attn_output = ttnn.slice(attn_output, [0, 0, 0, 0], [1, 1, batch_size, attn_output.shape[-1]])

        attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, -1))

        return attn_output, None, past_key_values


class NHWCConvTransposePytorch(nn.Module):
    """A wrapper around nn.ConvTranspose2d to handle NHWC input/output."""

    def __init__(self, conv: nn.ConvTranspose2d) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class NCDHWConv3dPytorch(nn.Module):
    """PyTorch fallback for :class:`TTNNConv3d` (NCDHW in/out)."""

    def __init__(self, conv: nn.Conv3d) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _conv1d_to_height1_conv2d(conv1d: nn.Conv1d) -> nn.Conv2d:
    """Map ``nn.Conv1d`` to ``nn.Conv2d`` with ``kernel_size=(1, k)`` for TTNN 2D conv."""

    def _one(x):
        if isinstance(x, int):
            return x
        return int(x[0])

    k = _one(conv1d.kernel_size)
    s = _one(conv1d.stride)
    p = _one(conv1d.padding)
    d = _one(conv1d.dilation)

    conv2d = nn.Conv2d(
        in_channels=conv1d.in_channels,
        out_channels=conv1d.out_channels,
        kernel_size=(1, k),
        stride=(1, s),
        padding=(0, p),
        dilation=(1, d),
        groups=conv1d.groups,
        bias=conv1d.bias is not None,
        device=conv1d.weight.device,
        dtype=conv1d.weight.dtype,
    )
    with torch.no_grad():
        conv2d.weight.copy_(conv1d.weight.unsqueeze(2))
        if conv1d.bias is not None:
            conv2d.bias.copy_(conv1d.bias)
    return conv2d


class TTNNConv1d(TTNNModule):
    """1D convolution via TTNN ``conv2d`` with height 1 (activations ``[B, C, T]`` → NHWC ``[B, 1, T, C]``).

    Pairs with :class:`TTNNQwen3OmniMoeCausalConvNet` in ``qwen_omni_activation`` for HF ``Qwen3OmniMoeCausalConvNet``.
    """

    def __init__(self):
        super().__init__()
        self.conv2d = None

    @classmethod
    def from_torch(cls, conv1d: nn.Conv1d, slice_config=None):
        new = cls()
        new._fallback_torch_layer = conv1d
        equiv = _conv1d_to_height1_conv2d(conv1d)
        # Qwen Omni NHWC permute + mesh width-shard gather live on TTNNQwenOmniConv2dNHWC (defined below).
        new.conv2d = TTNNQwenOmniConv2dNHWC.from_torch(equiv, slice_config=slice_config)
        return new

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _qwen_omni_conv2d_mesh_output_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, input_tensor, reshape_output=True):
        """``input_tensor``: ``ttnn.Tensor`` or ``torch.Tensor`` ``[B, C, T]``. Returns ``[B, C_out, T_out]``.

        TTNN children of TTNN parents use ``_bypass_tensor_wrapping`` (see ``device_management``), so
        :class:`TTNNQwen3OmniMoeCausalConvNet` (``qwen_omni_activation``) passes host torch after ``F.pad``; convert here.
        """
        if isinstance(input_tensor, TorchTTNNTensor):
            input_tensor = input_tensor.ttnn_tensor if input_tensor.ttnn_tensor is not None else input_tensor.elem
        if isinstance(input_tensor, torch.Tensor):
            mesh_mapper = None
            dev = self.device
            if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
                mesh_mapper = ttnn.ReplicateTensorToMesh(dev)
            input_tensor = ttnn.from_torch(
                input_tensor,
                dtype=torch_dtype_to_ttnn_dtype(input_tensor.dtype),
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=mesh_mapper,
            )
        x = ttnn.permute(input_tensor, (0, 2, 1))
        x = ttnn.unsqueeze(x, 1)
        out = self.conv2d(x, reshape_output=reshape_output)
        out = ttnn.squeeze(out, 1)
        return ttnn.permute(out, (0, 2, 1))


def _int_pair_2d(x) -> tuple:
    if isinstance(x, int):
        return (x, x)
    t = tuple(int(v) for v in x)
    if len(t) == 1:
        return (t[0], t[0])
    return (t[0], t[1])


def _transpose_conv_output_w(
    input_width: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    out_pad_w: int,
) -> int:
    """Output width for conv transpose (symmetric padding), matches PyTorch / TTNN sliding window."""
    return (input_width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + out_pad_w + 1


def _conv_transpose2d_dram_width_slice_config(w_out: int):
    """Explicit DRAM width slicing for large transpose convs.

    Without this, TTNN auto-slice can pick a marginal slice count (e.g. 12 for ~5.4k-wide
    output) that still exhausts L1 vs circular buffers on Wormhole (TT_THROW CB clash).
    """
    max_slices = 172
    if w_out <= 512:
        return None
    # Smaller slices than auto (~output/450) to keep per-slice L1 well under free DRAM/L1 budget.
    num_slices = max(48, min(max_slices, (w_out + 79) // 80))
    return ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceWidth)


def _conv_transpose1d_to_height1_conv_transpose2d(c1: nn.ConvTranspose1d) -> nn.ConvTranspose2d:
    """Map ``nn.ConvTranspose1d`` to ``nn.ConvTranspose2d`` with ``kernel_size=(1, k)`` for TTNN 2D transpose conv."""

    k = _int_pair_2d(c1.kernel_size)[0]
    s = _int_pair_2d(c1.stride)[0]
    p = _int_pair_2d(c1.padding)[0]
    op = _int_pair_2d(c1.output_padding)[0]
    d = _int_pair_2d(c1.dilation)[0]

    c2 = nn.ConvTranspose2d(
        in_channels=c1.in_channels,
        out_channels=c1.out_channels,
        kernel_size=(1, k),
        stride=(1, s),
        padding=(0, p),
        output_padding=(0, op),
        dilation=(1, d),
        groups=c1.groups,
        bias=c1.bias is not None,
        device=c1.weight.device,
        dtype=c1.weight.dtype,
    )
    with torch.no_grad():
        c2.weight.copy_(c1.weight.unsqueeze(2))
        if c1.bias is not None:
            c2.bias.copy_(c1.bias)
    return c2


@trace_enabled
class TTNNConvTranspose2dNHWC(TTNNModule):
    """TTNN ``conv_transpose2d`` on NHWC activations (``[B, H, W, C]``)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        groups: int = 1,
        slice_config=None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.slice_config = slice_config
        self.reshape = TTNNReshape()

    @classmethod
    def from_torch(cls, conv: nn.ConvTranspose2d, slice_config=None) -> "TTNNConvTranspose2dNHWC":
        new_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=_int_pair_2d(conv.kernel_size),
            stride=_int_pair_2d(conv.stride),
            padding=_int_pair_2d(conv.padding),
            output_padding=_int_pair_2d(conv.output_padding),
            dilation=_int_pair_2d(conv.dilation),
            groups=conv.groups,
            slice_config=slice_config,
        )
        new_conv._fallback_torch_layer = NHWCConvTransposePytorch(conv)
        return new_conv

    def preprocess_weights_impl(self):
        if self.torch_layer is None:
            self._fallback_torch_layer = NHWCConvTransposePytorch(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.output_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )
        inner = self.torch_layer.conv
        self.tt_weight, self.tt_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
            inner.weight, inner.bias
        )
        super().preprocess_weights_impl()

    def deallocate_weights_impl(self):
        ttnn.deallocate(self.tt_weight)
        if self.tt_bias is not None:
            ttnn.deallocate(self.tt_bias)
        super().deallocate_weights_impl()

    def forward(self, input_tensor: ttnn.Tensor, reshape_output=True) -> ttnn.Tensor:
        batch_size, input_height, input_width, _ = input_tensor.shape
        batch_size, input_height, input_width = int(batch_size), int(input_height), int(input_width)

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=True,
            config_tensors_in_dram=True,
        )
        if self.slice_config is not None:
            conv_config.shard_layout = self.slice_config

        dev = input_tensor.device()
        w_out_est = _transpose_conv_output_w(
            input_width,
            int(self.kernel_size[1]),
            int(self.stride[1]),
            int(self.padding[1]),
            int(self.dilation[1]),
            int(self.output_padding[1]),
        )
        dram_slice_config = _conv_transpose2d_dram_width_slice_config(w_out_est)
        # LoFi reduces circular-buffer pressure for very wide transpose paths (still DRAM-sliced).
        math_fidelity = ttnn.MathFidelity.LoFi if w_out_est > 4096 else ttnn.MathFidelity.HiFi4
        compute_config = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        if input_tensor.memory_config().is_sharded():
            input_tensor = ttnn.sharded_to_interleaved(input_tensor)

        kwargs = dict(
            input_tensor=input_tensor,
            weight_tensor=self.tt_weight,
            device=dev,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            bias_tensor=self.tt_bias,
            conv_config=conv_config,
            compute_config=compute_config,
            mirror_kernel=True,
            dtype=ttnn.bfloat16,
        )
        if dram_slice_config is not None:
            kwargs["dram_slice_config"] = dram_slice_config
        if reshape_output:
            out, (oh, ow) = ttnn.conv_transpose2d(**kwargs, return_output_dim=True)
            return self.reshape(out, [batch_size, oh, ow, -1])
        return ttnn.conv_transpose2d(**kwargs, return_output_dim=False)


class TTNNConvTranspose1d(TTNNModule):
    """1D transposed convolution via TTNN ``conv_transpose2d`` with height 1 (``[B,C,T]`` ↔ NHWC ``[B,1,T,C]``)."""

    def __init__(self):
        super().__init__()
        self.conv2d_t = None
        self.in_channels = None
        self.out_channels = None
        self.kernel_size = None
        self.stride = None
        self.padding = None
        self.output_padding = None
        self.dilation = None
        self.groups = None

    @classmethod
    def from_torch(cls, conv1d: nn.ConvTranspose1d, slice_config=None):
        new = cls()
        new._fallback_torch_layer = conv1d
        new.in_channels = conv1d.in_channels
        new.out_channels = conv1d.out_channels
        new.kernel_size = conv1d.kernel_size
        new.stride = conv1d.stride
        new.padding = conv1d.padding
        new.output_padding = conv1d.output_padding
        new.dilation = conv1d.dilation
        new.groups = conv1d.groups
        equiv = _conv_transpose1d_to_height1_conv_transpose2d(conv1d)
        new.conv2d_t = TTNNConvTranspose2dNHWC.from_torch(equiv, slice_config=slice_config)
        return new

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _qwen_omni_conv2d_mesh_output_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, input_tensor, reshape_output=True):
        if isinstance(input_tensor, TorchTTNNTensor):
            input_tensor = input_tensor.ttnn_tensor if input_tensor.ttnn_tensor is not None else input_tensor.elem
        if isinstance(input_tensor, torch.Tensor):
            mesh_mapper = None
            dev = self.device
            if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
                mesh_mapper = ttnn.ReplicateTensorToMesh(dev)
            input_tensor = ttnn.from_torch(
                input_tensor,
                dtype=torch_dtype_to_ttnn_dtype(input_tensor.dtype),
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=mesh_mapper,
            )
        x = ttnn.permute(input_tensor, (0, 2, 1))
        x = ttnn.unsqueeze(x, 1)
        out = self.conv2d_t(x, reshape_output=reshape_output)
        out = ttnn.squeeze(out, 1)
        return ttnn.permute(out, (0, 2, 1))


# TTNN ``experimental.conv3d`` layout (see ``tests/ttnn/unit_tests/operations/conv/test_conv3d.py``).
_CONV3D_CHANNEL_ALIGNMENT = 32


def _conv3d_chunk_n_from_env() -> int:
    """Chunk size along conv3d / patch-embed batch dim (env ``TT_SYMBIOTE_CONV3D_CHUNK_N``, default 1024)."""
    import os

    return max(1, int(os.environ.get("TT_SYMBIOTE_CONV3D_CHUNK_N", "1024")))


def _conv3d_mesh_max_n_per_chunk(dev) -> int:
    """Upper bound on leading batch N for one conv3d compute slice on a mesh (patch embed).

    Without chunking, ``from_torch`` + ``experimental.conv3d`` + output readback can request 100+ GiB DRAM
    for large ``N``.  Set ``TT_SYMBIOTE_CONV3D_CHUNK_N`` to tune (default 1024).
    """
    if dev is None or not hasattr(dev, "get_num_devices") or dev.get_num_devices() <= 1:
        return 0
    return _conv3d_chunk_n_from_env()


def _int_triplet_3d(x) -> tuple:
    if isinstance(x, int):
        return (x, x, x)
    t = tuple(int(v) for v in x)
    if len(t) == 1:
        return (t[0], t[0], t[0])
    if len(t) == 3:
        return (t[0], t[1], t[2])
    raise ValueError(f"Expected int or length-1/3 tuple for 3D conv hyperparameter, got {x!r}")


def _conv3d_out_dim(size_in: int, pad: int, stride: int, kernel: int, dilation: int = 1) -> int:
    eff_k = dilation * (kernel - 1) + 1
    return (size_in + 2 * pad - eff_k) // stride + 1


def _conv3d_padding_triple_for_ttnn(padding) -> tuple:
    """Map ``nn.Conv3d.padding`` to a ``(pd, ph, pw)`` triple for TTNN (symmetric triple only)."""
    if isinstance(padding, int):
        return (padding, padding, padding)
    p = tuple(int(x) for x in padding)
    if len(p) == 3:
        return (p[0], p[1], p[2])
    if len(p) == 6:
        raise ValueError("TTNNConv3d does not support 6-tuple asymmetric padding; use PyTorch fallback.")
    raise ValueError(f"Unsupported Conv3d padding: {padding!r}")


def _conv3d_torch_weights_to_ttnn_layout(
    conv: nn.Conv3d,
    *,
    alignment: int = _CONV3D_CHANNEL_ALIGNMENT,
    c_in_block: int = 0,
) -> tuple:
    """Layout ``nn.Conv3d`` weights/bias like ``prepare_weights`` in ``test_conv3d`` (host torch)."""
    w = conv.weight.data  # [out, in, kD, kH, kW]
    c_in = int(conv.in_channels)
    out_ch = int(conv.out_channels)
    w = w.permute(2, 3, 4, 1, 0)  # kD, kH, kW, C, out
    align_pad = (alignment - c_in % alignment) % alignment
    if align_pad:
        w = torch.nn.functional.pad(w, (0, 0, 0, align_pad))
    kD, kH, kW, c_pad, _ = w.shape
    c_in_block_eff = c_pad if c_in_block == 0 else int(c_in_block)
    num_c_in_blocks = c_pad // c_in_block_eff
    assert num_c_in_blocks * c_in_block_eff == c_pad
    w = w.reshape(kD, kH, kW, num_c_in_blocks, c_in_block_eff, out_ch)
    w = w.permute(3, 0, 1, 2, 4, 5)
    w = w.reshape(-1, out_ch).contiguous()
    b = conv.bias.data.reshape(1, -1).contiguous() if conv.bias is not None else None
    return w, b


def _conv3d_torch_input_to_ndhwc_row_major(
    x: torch.Tensor, *, alignment: int = _CONV3D_CHANNEL_ALIGNMENT
) -> torch.Tensor:
    """NCDHW → NDHWC and pad channel to ``alignment`` (TTNN conv3d input)."""
    if x.dim() != 5:
        raise RuntimeError(f"TTNNConv3d expects 5D NCDHW input, got shape {tuple(x.shape)}")
    c_in = int(x.shape[1])
    tt_input = x.permute(0, 2, 3, 4, 1).contiguous()
    align_pad = (alignment - c_in % alignment) % alignment
    if align_pad:
        tt_input = torch.nn.functional.pad(tt_input, (0, align_pad))
    return tt_input


def _ttnn_conv3d_output_to_torch_ncdhw(
    tt_out: ttnn.Tensor,
    *,
    n: int,
    d_out: int,
    h_out: int,
    w_out: int,
    out_channels: int,
    mesh_device,
) -> torch.Tensor:
    """Match ``reshape_output`` in ``test_conv3d``; host ``[N,C,D,H,W]``."""
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        t = ttnn.to_torch(tt_out)
    else:
        shards = ttnn.get_device_tensors(tt_out)
        if shards:
            t = ttnn.to_torch(shards[0])
        else:
            composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
            t = ttnn.to_torch(tt_out, mesh_composer=composer)
            lead = int(tt_out.shape[0])
            if t.dim() >= 1 and int(t.shape[0]) > lead:
                t = t[:lead].contiguous()
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    t = t.reshape(n, d_out, h_out, w_out, out_channels)
    return t.permute(0, 4, 1, 2, 3).contiguous()


def _ttnn_tensor_to_torch_batch_leading(tt: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """Device/mesh tensor → host torch; for multi-device concat on dim 0, slice to logical batch."""
    n0 = int(tt.shape[0])
    if mesh_device is None or not hasattr(mesh_device, "get_num_devices") or mesh_device.get_num_devices() <= 1:
        return ttnn.to_torch(tt)
    shards = ttnn.get_device_tensors(tt)
    if shards:
        return ttnn.to_torch(shards[0])
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    t = ttnn.to_torch(tt, mesh_composer=composer)
    if t.dim() >= 1 and int(t.shape[0]) > n0:
        t = t[:n0].contiguous()
    return t


@trace_enabled
class TTNNConv3d(TTNNModule):
    """3D convolution via ``ttnn.experimental.conv3d`` (NDHWC activations, Qwen3-Omni vision patch embed).

    Matches the weight/input layout used in ``tests/ttnn/unit_tests/operations/conv/test_conv3d.py``.
    For large patch kernels (e.g. ``kernel_size == stride`` with spatial product ≥ 196) and
    ``out_channels % 32 == 0``, uses Qwen-VL-style L1 blocking (``C_in_block=16``, ``C_out_block=32``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
        dilation: tuple,
        groups: int,
        padding_mode: str,
        conv3d_blocking=None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.conv3d_blocking = conv3d_blocking
        self.tt_weight = None
        self.tt_bias = None

    @property
    def weight(self):
        """HF ``Qwen3OmniMoeVisionPatchEmbed`` reads ``proj.weight.dtype``; mirror ``nn.Conv3d``."""
        if self.torch_layer is None:
            raise AttributeError("weight")
        return self.torch_layer.conv.weight

    @property
    def bias(self):
        if self.torch_layer is None:
            raise AttributeError("bias")
        return self.torch_layer.conv.bias

    @classmethod
    def from_torch(cls, conv: nn.Conv3d, conv3d_blocking=None) -> "TTNNConv3d":
        ks = _int_triplet_3d(conv.kernel_size)
        st = _int_triplet_3d(conv.stride)
        try:
            pad_triple = _conv3d_padding_triple_for_ttnn(conv.padding)
        except ValueError:
            pad_triple = (0, 0, 0)
        blocking = conv3d_blocking
        if blocking is None:
            if ks == st and ks[0] * ks[1] * ks[2] >= 196 and conv.out_channels % 32 == 0 and conv.groups == 1:
                blocking = (16, 32, 1, 1, 1)
        new_mod = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=ks,
            stride=st,
            padding=pad_triple,
            dilation=_int_triplet_3d(conv.dilation),
            groups=conv.groups,
            padding_mode=conv.padding_mode,
            conv3d_blocking=blocking,
        )
        new_mod._fallback_torch_layer = NCDHWConv3dPytorch(conv)
        return new_mod

    def preprocess_weights_impl(self):
        if self.torch_layer is None:
            self._fallback_torch_layer = NCDHWConv3dPytorch(
                nn.Conv3d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    bias=True,
                )
            )
        conv = self.torch_layer.conv
        c_in_block = 0
        if self.conv3d_blocking is not None:
            c_in_block = int(self.conv3d_blocking[0])
        w_torch, b_torch = _conv3d_torch_weights_to_ttnn_layout(conv, c_in_block=c_in_block)
        self.tt_weight = ttnn.from_torch(w_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0)
        if b_torch is not None:
            self.tt_bias = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0)
        else:
            self.tt_bias = None
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        if self.tt_weight is not None:
            self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
        if self.tt_bias is not None:
            self.tt_bias = ttnn.to_device(self.tt_bias, self.device)
        super().move_weights_to_device_impl()

    def deallocate_weights_impl(self):
        if self.tt_weight is not None:
            ttnn.deallocate(self.tt_weight)
        if self.tt_bias is not None:
            ttnn.deallocate(self.tt_bias)
        super().deallocate_weights_impl()

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _qwen_omni_conv2d_mesh_output_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def __call__(self, *args, **kwds):
        """Same as ``NormalRun.module_run`` but skip ``to_ttnn_wrap`` on inputs.

        Default mesh ``to_ttnn`` for 5D NCDHW activations can size buffers incorrectly; we keep
        ``TorchTTNNTensor.elem`` as torch until :meth:`forward` uploads NDHWC via ``from_torch``.
        """
        bypass = getattr(self, "_bypass_tensor_wrapping", False)
        if bypass:
            from models.experimental.tt_symbiote.core.run_config import NormalRun

            return NormalRun.module_run(self, *args, **kwds)

        import time

        from tracy import signpost

        from models.experimental.tt_symbiote.core.run_config import (
            DispatchManager,
            NormalRun,
            compose_transforms,
            post_process_ttnn_module_output,
            set_device_wrap,
            wrap_to_torch_ttnn_tensor,
        )

        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        assert self.device is not None, "Device must be set for TTNN module execution."
        begin_full = time.time()
        transform = compose_transforms(wrap_to_torch_ttnn_tensor, set_device_wrap(self.device))
        func_args = tree_map(transform, args)
        other_kwargs = {k: v for k, v in kwds.items() if "past_key_value" not in k}
        func_kwargs = tree_map(transform, other_kwargs)
        func_kwargs.update({k: v for k, v in kwds.items() if "past_key_value" in k})
        begin = time.time()
        self.preprocess_weights()
        end = time.time()
        DispatchManager.set_current_module_name(self.module_name)
        DispatchManager.record_timing(
            "TTNN", self.module_name, self.__class__.__name__ + "_preprocess_weights", {}, end - begin
        )
        begin = time.time()
        self.move_weights_to_device()
        end = time.time()
        DispatchManager.record_timing(
            "TTNN", self.module_name, self.__class__.__name__ + "_move_weights_to_device", {}, end - begin
        )
        if NormalRun.signpost_mode is not None:
            signpost(f"{self.module_name}", f"{self.__class__.__name__}")
        begin = time.time()
        result = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
        end = time.time()
        DispatchManager.record_timing("TTNN", self.module_name, self.__class__.__name__ + "_forward", {}, end - begin)
        DispatchManager.set_current_module_name(None)
        end_full = time.time()
        DispatchManager.record_timing(
            "TorchModules", self.module_name, self.__class__.__name__, {}, end_full - begin_full
        )
        return result

    def _experimental_conv3d_ncdhw_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``ttnn.experimental.conv3d`` for one CPU ``[N,C,D,H,W]`` batch; return host ``[N,C_out,D_out,H_out,W_out]``."""
        n, _, d_in, h_in, w_in = x.shape
        pd, ph, pw = self.padding
        sd, sh, sw = self.stride
        dd, dh, dw = self.dilation
        kd, kh, kw = self.kernel_size
        d_out = _conv3d_out_dim(d_in, pd, sd, kd, dd)
        h_out = _conv3d_out_dim(h_in, ph, sh, kh, dh)
        w_out = _conv3d_out_dim(w_in, pw, sw, kw, dw)

        dev = self.device
        mesh_mapper = None
        if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(dev)
        ndhwc = _conv3d_torch_input_to_ndhwc_row_major(x)
        tt_in = ttnn.from_torch(
            ndhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mesh_mapper,
        )
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        grid = dev.compute_with_storage_grid_size()
        grid_xy = (int(grid.x), int(grid.y))

        kwargs = dict(
            input_tensor=tt_in,
            weight_tensor=self.tt_weight,
            bias_tensor=self.tt_bias,
            dtype=ttnn.bfloat16,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            padding_mode=self.padding_mode,
            compute_kernel_config=compute_kernel_config,
        )
        if self.conv3d_blocking is not None:
            cin_b, cout_b, tb, hb, wb = self.conv3d_blocking
            kwargs["config"] = ttnn.Conv3dConfig(
                weights_dtype=ttnn.bfloat16,
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
                T_out_block=int(tb),
                W_out_block=int(wb),
                H_out_block=int(hb),
                C_out_block=int(cout_b),
                C_in_block=int(cin_b),
                dilation=self.dilation,
                compute_with_storage_grid_size=grid_xy,
            )
        tt_out = ttnn.experimental.conv3d(**kwargs)
        out_torch = _ttnn_conv3d_output_to_torch_ncdhw(
            tt_out,
            n=n,
            d_out=d_out,
            h_out=h_out,
            w_out=w_out,
            out_channels=self.out_channels,
            mesh_device=dev,
        )
        ttnn.deallocate(tt_in)
        ttnn.deallocate(tt_out)
        return out_torch

    def forward(self, x):
        if isinstance(x, TorchTTNNTensor):
            x = x.ttnn_tensor if x.ttnn_tensor is not None else x.elem
        dev = self.device
        conv_pad = self.torch_layer.conv.padding
        if isinstance(conv_pad, tuple) and len(conv_pad) == 6:
            if isinstance(x, ttnn.Tensor):
                x = _ttnn_tensor_to_torch_batch_leading(x, dev)
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"TTNNConv3d expected torch or ttnn tensor, got {type(x)}")
            return self.torch_layer.conv(x)

        if self.groups != 1 or self.padding_mode != "zeros":
            if isinstance(x, ttnn.Tensor):
                x = _ttnn_tensor_to_torch_batch_leading(x, dev)
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"TTNNConv3d expected torch or ttnn tensor, got {type(x)}")
            x = x.to(dtype=self.torch_layer.conv.weight.dtype)
            return self.torch_layer.conv(x)

        if isinstance(x, ttnn.Tensor):
            x = _ttnn_tensor_to_torch_batch_leading(x, dev)
            x = x.to(dtype=torch.bfloat16).contiguous()

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"TTNNConv3d expected torch or ttnn tensor, got {type(x)}")
        x = x.to(dtype=torch.bfloat16).contiguous()
        n = int(x.shape[0])
        max_chunk = _conv3d_mesh_max_n_per_chunk(dev)
        if max_chunk > 0 and n > max_chunk:
            parts = []
            for s in range(0, n, max_chunk):
                e = min(s + max_chunk, n)
                parts.append(self._experimental_conv3d_ncdhw_torch(x[s:e].contiguous()))
            out_torch = torch.cat(parts, dim=0)
        else:
            out_torch = self._experimental_conv3d_ncdhw_torch(x)

        self.deallocate_weights()

        c_out = int(out_torch.shape[1])
        out_2d = out_torch.reshape(-1, c_out).to(dtype=torch.bfloat16).contiguous()

        mesh_mapper = None
        if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(dev)
        return ttnn.from_torch(
            out_2d,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=mesh_mapper,
        )


def _qwen_omni_conv2d_mesh_output_config(mesh_device):
    """Replicated conv readback: Replicate + ConcatMesh(dim=0) slice (avoid ConcatMesh2d inflation on audio). TTNNQwenOmniConv2dNHWC AGs width when sharded."""
    return qwen_omni_replicated_concat_dim0_tensor_config(mesh_device)


def _pair_int2_conv(x) -> tuple:
    if isinstance(x, int):
        return (x, x)
    t = tuple(int(v) for v in x)
    if len(t) == 1:
        return (t[0], t[0])
    return (t[0], t[1])


def _expected_nhwc_conv2d_out_width(w_in: int, conv_mod: TTNNConv2dNHWC) -> int:
    _, kw = _pair_int2_conv(conv_mod.kernel_size)
    _, sw = _pair_int2_conv(conv_mod.stride)
    _, pw = _pair_int2_conv(conv_mod.padding)
    _, dw = _pair_int2_conv(conv_mod.dilation)
    return (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1


def _nhwc_spatial_w_per_device_shard(out: ttnn.Tensor, mesh_num_devices: int) -> int | None:
    """If ``out`` is one width shard per mesh device, return local W; else None.

    ``out.shape[2]`` can still reflect the **logical** full width while each device's tensor
    only holds ``W_full / nd`` (e.g. 48 vs 384). Using shard geometry avoids skipping
    ``all_gather`` and fixes residual adds that saw ``T=48`` vs ``T=384``.
    """
    if mesh_num_devices <= 1:
        return None
    shards = ttnn.get_device_tensors(out)
    if len(shards) != mesh_num_devices:
        return None
    return int(shards[0].shape[2])


def _maybe_all_gather_nhwc_width_across_mesh(
    conv_mod: TTNNConv2dNHWC, out: ttnn.Tensor, input_nhwc: ttnn.Tensor
) -> ttnn.Tensor:
    """Stitch per-device width shards (``W_local * num_devices == W_full``) before NCHW permute."""
    dev = conv_mod.device
    if dev is None or not hasattr(dev, "get_num_devices") or dev.get_num_devices() <= 1:
        return out
    nd = int(dev.get_num_devices())
    w_in = int(input_nhwc.shape[2])
    w_exp = _expected_nhwc_conv2d_out_width(w_in, conv_mod)
    w_meta = int(out.shape[2])
    w_shard = _nhwc_spatial_w_per_device_shard(out, nd)
    # Prefer per-device width when we see a full mesh of spatial shards (avoids false w_meta == w_exp).
    w_cur = w_shard if w_shard is not None else w_meta
    if w_cur == w_exp:
        return out
    if w_cur * nd == w_exp:
        ds = getattr(conv_mod, "device_state", None)
        if ds is not None and getattr(ds, "ccl_manager", None) is not None:
            out = ttnn.experimental.all_gather_async(
                out,
                dim=2,
                multi_device_global_semaphore=ds.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=ds.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
                num_links=1,
                topology=ttnn.Topology.Linear,
            )
        else:
            out = ttnn.all_gather(out, dim=2, num_links=1, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(dev)
    return out


@trace_enabled
class TTNNQwenOmniConv2dNHWC(TTNNConv2dNHWC):
    """Qwen3-Omni Conv2d: permute audio NCHW [B,C,F,T] vs vision NHWC for TTNN conv2d; restore layout for HF. from_torch always returns this subclass."""

    def _logical_in_channels(self) -> int:
        tl = self.torch_layer
        if tl is None:
            return int(self.in_channels)
        inner = getattr(tl, "conv", None)
        if inner is not None:
            return int(inner.in_channels)
        return int(self.in_channels)

    @classmethod
    def from_torch(cls, conv: nn.Conv2d, slice_config=None) -> "TTNNQwenOmniConv2dNHWC":
        # Mirror :class:`TTNNConv2dNHWCInputMultipleOf16` but always construct ``cls`` so
        # :meth:`forward` below is used (``InputMultipleOf16.from_torch`` returns ``TTNNConv2dNHWC``).
        if conv.in_channels > 16 or conv.in_channels % 16 == 0:
            new_mod = cls(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                slice_config=slice_config,
            )
            new_mod._fallback_torch_layer = NHWCConvPytorch(conv)
            return new_mod
        new_mod = cls(
            in_channels=16,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            slice_config=slice_config,
        )
        conv.weight = nn.Parameter(
            torch.nn.functional.pad(conv.weight, (0, 0, 0, 0, 0, (16 - conv.in_channels % 16) % 16))
        )
        new_mod._fallback_torch_layer = NHWCConvPytorch(conv)
        return new_mod

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _qwen_omni_conv2d_mesh_output_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, input_tensor: ttnn.Tensor, reshape_output=True) -> ttnn.Tensor:
        logical_in = self._logical_in_channels()
        shape = input_tensor.shape
        nchw_in = False
        if len(shape) != 4:
            raise RuntimeError(f"TTNNQwenOmniConv2dNHWC: expected 4D input, got shape={list(shape)}")
        if int(shape[1]) == logical_in and int(shape[-1]) != logical_in:
            nchw_in = True
            input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
        elif int(shape[-1]) != logical_in and int(shape[1]) != logical_in:
            raise RuntimeError(
                f"TTNNQwenOmniConv2dNHWC: cannot match in_channels={logical_in} to layout " f"shape={list(shape)}"
            )

        if int(self.in_channels) > logical_in and int(input_tensor.shape[-1]) == logical_in:
            pad_c = int(self.in_channels) - logical_in
            layout_in = input_tensor.layout
            if layout_in == ttnn.TILE_LAYOUT:
                input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
            input_tensor = ttnn.pad(
                input_tensor,
                ((0, 0), (0, 0), (0, 0), (0, pad_c)),
                value=0.0,
            )
            if layout_in == ttnn.TILE_LAYOUT:
                input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)

        out = TTNNConv2dNHWC.forward(self, input_tensor, reshape_output=reshape_output)
        out = _maybe_all_gather_nhwc_width_across_mesh(self, out, input_tensor)
        if nchw_in:
            out = ttnn.permute(out, (0, 3, 1, 2))
        return out


# ========================================================================
# Merged from modules/qwen_omni_rotary.py
# ========================================================================


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
    return qwen_omni_replicated_concat_dim0_tensor_config(mesh_device)


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


# ========================================================================
# Merged from modules/qwen_omni_normalization.py
# ========================================================================


class TTNNQwenLayerNorm(TTNNModule):
    """HF ``nn.LayerNorm`` → ``ttnn.layer_norm`` (single device) or distributed pre/all_gather/post (mesh)."""

    @staticmethod
    def _normalized_numel(normalized_shape) -> int:
        if isinstance(normalized_shape, int):
            return int(normalized_shape)
        n = 1
        for d in normalized_shape:
            n *= int(d)
        return n

    @property
    def _is_distributed(self) -> bool:
        return self.device is not None and self.device.get_num_devices() > 1

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        use_row_major_workaround: bool = False,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        self.embedding_dim = self._normalized_numel(self.normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = elementwise_affine and bias
        self.use_row_major_workaround = use_row_major_workaround
        self.compute_kernel_config = None
        self.tt_weight = None
        self.tt_bias = None
        self.weight_distributed = None
        self.bias_distributed = None
        # Mesh: True when (emb//32) % mesh_width != 0 — use all_gather + full-width replicated LN (vision 1152 on 1×8).
        self._distributed_gather_layernorm = False
        if self.embedding_dim % 32 != 0:
            raise ValueError(
                f"TTNNQwenLayerNorm: embedding_dim ({self.embedding_dim}) must be divisible by 32 for TTNN tile ops"
            )
        if self.elementwise_affine:
            self.torch_weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.torch_bias = nn.Parameter(torch.zeros(self.normalized_shape)) if self.use_bias else None
        else:
            self.torch_weight = None
            self.torch_bias = None

    def set_output_tensors_config_impl(self, output_tensors):
        """Col-sharded last dim on mesh, or replicated full hidden when using gather + full ``layer_norm``."""

        def set_gather_output_config(e):
            """Replicated mesh: ConcatMeshToTensor(dim=0) for host unwrap."""
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None and self.device is not None:
                e.set_distributed_tensor_config(
                    DistributedTensorConfig(
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                        mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                    )
                )
            return e

        def materialize_merger_ln_q_one_replica(e):
            """Merger ln_q: one replica torch elem for HF view (same slice-after-concat idea as TTNNQwen3OmniVisionMLP)."""
            if not isinstance(e, TorchTTNNTensor) or e.ttnn_tensor is None:
                return e
            t = e.ttnn_tensor
            n = int(t.shape[0])
            pt = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            if pt.shape[0] > n:
                pt = pt[:n]
            e.elem = pt.contiguous()
            e.ttnn_tensor = None
            if getattr(e, "_distributed_tensor_config", None) is not None:
                e._distributed_tensor_config = None
            return e

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if self._is_distributed and self.device is not None:
                    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
                    mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=-1)

                    def logical_shape_for_col_sharded(shape):
                        shape_list = list(shape)
                        num_devices = self.device.get_num_devices()
                        shape_list[-1] = shape_list[-1] * num_devices
                        return tuple(shape_list)

                    e.set_distributed_tensor_config(
                        DistributedTensorConfig(
                            mesh_mapper=mesh_mapper,
                            mesh_composer=mesh_composer,
                            logical_shape_fn=logical_shape_for_col_sharded,
                        )
                    )
            return e

        if not self._is_distributed:
            return super().set_output_tensors_config_impl(output_tensors)
        if getattr(self, "_distributed_gather_layernorm", False) or getattr(
            self, "_force_replicated_input_layernorm", False
        ):
            name = self.module_name or ""
            # Do not dim-0-shard LN out: vision rotary needs full seq; sharding breaks q*cos broadcast.
            if "merger" in name and "ln_q" in name:
                return tree_map(materialize_merger_ln_q_one_replica, output_tensors)
            return tree_map(set_gather_output_config, output_tensors)
        return tree_map(set_col_sharded_config, output_tensors)

    @classmethod
    def from_torch(cls, layer_norm: nn.LayerNorm, use_row_major_workaround: bool = False):
        """Symbiote calls ``from_torch(hf_module)`` only — use ``set_device`` for the mesh."""
        if not layer_norm.elementwise_affine:
            return layer_norm
        emb = cls._normalized_numel(layer_norm.normalized_shape)
        if emb % 32 != 0:
            return layer_norm
        new_layer = cls(
            normalized_shape=layer_norm.normalized_shape,
            eps=layer_norm.eps,
            elementwise_affine=layer_norm.elementwise_affine,
            bias=layer_norm.bias is not None,
            use_row_major_workaround=use_row_major_workaround,
        )
        if layer_norm.weight is not None:
            new_layer.torch_weight = nn.Parameter(layer_norm.weight.data.clone())
        if layer_norm.bias is not None:
            new_layer.torch_bias = nn.Parameter(layer_norm.bias.data.clone())
        new_layer._fallback_torch_layer = layer_norm
        return new_layer

    def preprocess_weights_impl(self):
        if not self.elementwise_affine:
            self.tt_weight = None
            self.tt_bias = None
            return
        # Mesh: sharded ROW_MAJOR gamma in ``move_weights_to_device_impl``, or TILE + replicate when ntiles % width != 0.
        if self.device is not None and self.device.get_num_devices() > 1:
            ncol = int(list(self.device.shape)[-1])
            ntiles = self.embedding_dim // 32
            # Audio tower: replicated activations — avoid layer_norm_post_all_gather + per-shard gamma (wrong local dim).
            if getattr(self, "_force_replicated_input_layernorm", False) or ntiles % ncol != 0:
                self._distributed_gather_layernorm = True
                # Host TT tensors; move_weights uses from_torch(..., mesh_mapper=...) (to_device has no mesh_mapper).
                self.tt_weight = None
                self.tt_bias = None
                self.weight_distributed = None
                self.bias_distributed = None
                return
            self._distributed_gather_layernorm = False
            self.tt_weight = None
            self.tt_bias = None
            return
        weight = self.torch_weight
        bias = self.torch_bias
        if self.use_row_major_workaround:
            layout = ttnn.ROW_MAJOR_LAYOUT
            weight_reshaped = weight.reshape(-1, 32)
            bias_reshaped = bias.reshape(-1, 32) if bias is not None else None
        else:
            layout = ttnn.TILE_LAYOUT
            weight_reshaped = weight.reshape(1, -1)
            bias_reshaped = bias.reshape(1, -1) if bias is not None else None
        self.tt_weight = ttnn.from_torch(weight_reshaped, dtype=ttnn.bfloat16, layout=layout)
        if bias_reshaped is not None:
            self.tt_bias = ttnn.from_torch(bias_reshaped, dtype=ttnn.bfloat16, layout=layout)
        else:
            self.tt_bias = None

    def _build_sharded_gamma_beta_row_major(self):
        """``[1,1,ntiles,32]`` + ``ShardTensor2dMesh`` on tile dim — matches ``TTNNDistributedRMSNorm``."""
        emb = self.embedding_dim
        mesh_shape = list(self.device.shape)
        ncol = int(mesh_shape[-1])
        ntiles = emb // 32
        assert ntiles % ncol == 0, "gather path should not call _build_sharded_gamma_beta_row_major"
        w_bf16 = self.torch_weight.reshape(-1).to(torch.bfloat16)
        relayout_w = w_bf16.view(1, 1, ntiles, 32)
        mesh_mapper = ttnn.ShardTensor2dMesh(self.device, dims=(None, 2), mesh_shape=mesh_shape)
        self.weight_distributed = ttnn.as_tensor(
            relayout_w,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)
        if self.use_bias and self.torch_bias is not None:
            b_bf16 = self.torch_bias.reshape(-1).to(torch.bfloat16)
            relayout_b = b_bf16.view(1, 1, ntiles, 32)
            self.bias_distributed = ttnn.as_tensor(
                relayout_b,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
            )
            self.bias_distributed = ttnn.to_device(self.bias_distributed, self.device)
        else:
            z = torch.zeros(emb, dtype=torch.bfloat16)
            relayout_z = z.view(1, 1, ntiles, 32)
            self.bias_distributed = ttnn.as_tensor(
                relayout_z,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
            )
            self.bias_distributed = ttnn.to_device(self.bias_distributed, self.device)

    def move_weights_to_device_impl(self):
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        if not self.elementwise_affine:
            return
        if self.device.get_num_devices() > 1:
            if getattr(self, "_distributed_gather_layernorm", False) or getattr(
                self, "_force_replicated_input_layernorm", False
            ):
                self._distributed_gather_layernorm = True
                rep = ttnn.ReplicateTensorToMesh(self.device)
                w = self.torch_weight.reshape(1, -1).to(torch.bfloat16)
                self.tt_weight = ttnn.from_torch(
                    w,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    mesh_mapper=rep,
                )
                if self.use_bias and self.torch_bias is not None:
                    b = self.torch_bias.reshape(1, -1).to(torch.bfloat16)
                    self.tt_bias = ttnn.from_torch(
                        b,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        mesh_mapper=rep,
                    )
                else:
                    self.tt_bias = None
                return
            self._build_sharded_gamma_beta_row_major()
            return
        if self.tt_weight is not None:
            self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
        if self.tt_bias is not None:
            self.tt_bias = ttnn.to_device(self.tt_bias, self.device)

    def _forward_distributed(self, x: ttnn.Tensor, original_shape: tuple) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rank = len(original_shape)
        if rank == 2:
            x = ttnn.unsqueeze(x, 0)
            x = ttnn.unsqueeze(x, 0)
        elif rank == 3:
            x = ttnn.unsqueeze(x, 1)
        elif rank != 4:
            raise RuntimeError(f"TTNNQwenLayerNorm: expected rank 2–4 activations, got rank {rank}")

        tt_stats = ttnn.layer_norm_pre_all_gather(
            x,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        tt_stats = ttnn.all_gather(
            tt_stats,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        tt_out = ttnn.layer_norm_post_all_gather(
            x,
            tt_stats,
            epsilon=self.eps,
            weight=self.weight_distributed,
            bias=self.bias_distributed,
            compute_kernel_config=self.compute_kernel_config,
        )
        tt_stats.deallocate(True)

        if rank == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
        elif rank == 2 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [int(tt_out.shape[2]), int(tt_out.shape[3])])
        return tt_out

    def _forward_distributed_gather_ln(self, x: ttnn.Tensor, original_shape: tuple) -> ttnn.Tensor:
        """Col-shard width does not tile-shard evenly on mesh (e.g. vision 1152 on 8 devices): gather, full LN, replicate out."""
        emb = self.embedding_dim
        n_dev = int(self.device.get_num_devices())
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        wloc = int(x.shape[-1])
        if wloc * n_dev == emb:
            x = ttnn.all_gather(
                x,
                dim=-1,
                num_links=1,
                topology=ttnn.Topology.Linear,
            )
        elif wloc != emb:
            raise RuntimeError(
                f"TTNNQwenLayerNorm gather path: need col-shard {emb}/{n_dev} or full width {emb}, got last dim {wloc}"
            )

        rank = len(original_shape)
        if rank == 2:
            x = ttnn.unsqueeze(ttnn.unsqueeze(x, 0), 0)
        elif rank == 3:
            x = ttnn.unsqueeze(x, 1)
        elif rank != 4:
            raise RuntimeError(f"TTNNQwenLayerNorm: gather path expected rank 2–4 activations, got rank {rank}")

        tt_out = ttnn.layer_norm(
            x,
            weight=self.tt_weight,
            bias=self.tt_bias,
            epsilon=self.eps,
            compute_kernel_config=self.compute_kernel_config,
        )

        if rank == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
        elif rank == 2 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [int(tt_out.shape[2]), int(tt_out.shape[3])])
        return tt_out

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        original_shape = tuple(int(d) for d in x.shape)
        if self._is_distributed and self.elementwise_affine:
            if getattr(self, "_distributed_gather_layernorm", False) or getattr(
                self, "_force_replicated_input_layernorm", False
            ):
                return self._forward_distributed_gather_ln(x, original_shape)
            return self._forward_distributed(x, original_shape)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.layer_norm(
            x,
            weight=self.tt_weight,
            bias=self.tt_bias,
            epsilon=self.eps,
            compute_kernel_config=self.compute_kernel_config,
        )


# ========================================================================
# Merged from modules/qwen_omni_attention.py
# ========================================================================

_VISION_ATTN_SEQ_CHUNK = 4096


class TTNNQwen3VLMoeVisionAttention(TTNNModule):
    """TTNN implementation of Qwen3VLMoeVisionAttention (image / video frames in the vision tower)."""

    def __init__(self):
        super().__init__()

        self.hidden_size = None
        self.num_heads = None
        self.head_dim = None
        self.scaling = None

        self.qkv = None
        self.proj = None

        self.sdpa = TTNNSDPAAttention()
        self.is_causal = False
        self.core_grid = ttnn.CoreGrid(y=8, x=8)

    @classmethod
    def from_torch(cls, torch_attn):
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn

        config = torch_attn.config

        new_attn.hidden_size = config.hidden_size
        new_attn.num_heads = config.num_heads
        new_attn.head_dim = config.hidden_size // config.num_heads
        new_attn.scaling = new_attn.head_dim**-0.5

        new_attn.qkv = TTNNLinear.from_torch(torch_attn.qkv)
        new_attn.proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_attn.proj)

        return new_attn

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()

        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )

            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _to_ttnn(self, tensor):
        return tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor

    def _maybe_all_gather(self, tensor):
        t = self._to_ttnn(tensor)
        if not self._is_distributed:
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Ring,
        )

    @staticmethod
    def _last_hidden_dim(hidden_states) -> int:
        return int(hidden_states.shape[-1])

    def _leading_seq_len(self, hidden_states) -> int:
        t = self._to_ttnn(hidden_states)
        shape = t.shape
        hs = self.hidden_size
        if len(shape) == 2:
            return int(shape[0])
        if len(shape) == 3:
            last = int(shape[-1])
            if hs is not None and last == hs:
                return int(shape[1])
            if int(shape[0]) == 1:
                return int(shape[1])
        return int(shape[0])

    def rotate_half(self, x):
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        return ttnn.concat([ttnn.neg(x2), x1], dim=-1)

    def apply_rotary_pos_emb_vision(self, q, k, cos, sin):
        cos = ttnn.unsqueeze(cos, -2)
        sin = ttnn.unsqueeze(sin, -2)

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed

    @staticmethod
    def _to_raw_ttnn(tensor, max_unwrap=5):
        for _ in range(max_unwrap):
            if type(tensor).__name__ != "TorchTTNNTensor":
                return tensor
            raw = getattr(tensor, "ttnn_tensor", None)
            if raw is None:
                raw = tensor.to_ttnn
            tensor = raw
        return tensor

    def _slice_along_seq(self, tensor, s: int, e: int):
        t = self._to_ttnn(tensor)
        rank = len(t.shape)
        last = int(t.shape[-1])
        if rank == 2:
            return ttnn.slice(t, (s, 0), (e, last))
        if rank == 3 and int(t.shape[0]) == 1:
            return ttnn.slice(t, (0, s, 0), (1, e, last))
        if rank == 4:
            d1, d2, d3 = int(t.shape[1]), int(t.shape[2]), int(t.shape[3])
            return ttnn.slice(t, (s, 0, 0, 0), (e, d1, d2, d3))
        return ttnn.slice(t, (s, 0), (e, last))

    def _slice_rotary(self, cos_or_sin, s: int, e: int):
        t = self._to_ttnn(cos_or_sin)
        last = int(t.shape[-1])
        return ttnn.slice(t, (s, 0), (e, last))

    def _forward_chunk(
        self,
        hidden_states,
        position_embeddings,
        seq_len: int,
    ):
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(
                hidden_states,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        qkv = self.qkv(hidden_states)
        qkv = ttnn.reshape(
            self._to_raw_ttnn(qkv),
            (seq_len, 3, self.num_heads, self.head_dim),
        )

        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]

        cos, sin = position_embeddings

        q, k = self.apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = ttnn.permute(q, (1, 0, 2))
        k = ttnn.permute(k, (1, 0, 2))
        v = ttnn.permute(v, (1, 0, 2))

        q = ttnn.unsqueeze(q, 0)
        k = ttnn.unsqueeze(k, 0)
        v = ttnn.unsqueeze(v, 0)

        head_dim_padded = ((self.head_dim + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if head_dim_padded != self.head_dim:
            pad_size = head_dim_padded - self.head_dim
            q = ttnn.pad(q, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)
            k = ttnn.pad(k, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)
            v = ttnn.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)

        attn_output = self.sdpa(
            self,
            q,
            k,
            v,
            attention_mask=None,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=False,
            transpose_output=True,
        )

        attn_output = self._to_raw_ttnn(attn_output)
        if head_dim_padded != self.head_dim:
            attn_output = attn_output[:, :, :, : self.head_dim]

        attn_output = self._to_ttnn(attn_output)
        if self._is_distributed and int(attn_output.shape[-1]) != self.head_dim:
            attn_output = self._maybe_all_gather(attn_output)

        attn_output = ttnn.reshape(
            self._to_raw_ttnn(attn_output),
            (seq_len, self.hidden_size),
        )

        return self.proj(attn_output)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        expected_hidden = self.hidden_size
        if self._last_hidden_dim(self._to_ttnn(hidden_states)) != expected_hidden and self._is_distributed:
            hidden_states = self._maybe_all_gather(hidden_states)

        hidden_states = self._to_ttnn(hidden_states)
        if len(hidden_states.shape) == 3 and int(hidden_states.shape[0]) == 1:
            hidden_states = ttnn.squeeze(hidden_states, 0)

        seq_len = self._leading_seq_len(hidden_states)

        cos, sin = position_embeddings

        if seq_len <= _VISION_ATTN_SEQ_CHUNK:
            cos_sin = (self._maybe_all_gather(cos), self._maybe_all_gather(sin))
            return self._forward_chunk(hidden_states, cos_sin, seq_len)

        out_chunks = []
        for s in range(0, seq_len, _VISION_ATTN_SEQ_CHUNK):
            e = min(s + _VISION_ATTN_SEQ_CHUNK, seq_len)
            chunk_len = e - s
            hs_chunk = self._slice_along_seq(hidden_states, s, e)
            cos_chunk = self._slice_rotary(cos, s, e)
            sin_chunk = self._slice_rotary(sin, s, e)
            cos_sin = (self._maybe_all_gather(cos_chunk), self._maybe_all_gather(sin_chunk))
            out_chunks.append(self._forward_chunk(hs_chunk, cos_sin, chunk_len))

        if len(out_chunks) == 1:
            return out_chunks[0]
        raw_chunks = [self._to_raw_ttnn(c) for c in out_chunks]
        return ttnn.concat(raw_chunks, dim=0)


class TTNNQwen3OmniAttention(TTNNModule):
    """Qwen3-Omni thinker attention (no sliding window) on TTNN."""

    def __init__(self):
        super().__init__()
        self.sdpa = TTNNSDPAAttention()
        self.rope = TTNNQwenOmniRotaryPositionEmbedding()
        self.core_grid = ttnn.CoreGrid(y=8, x=8)

    def init_parameters(self):
        self.q_proj = TTNNLinear.from_torch(self.torch_layer.q_proj)
        self.k_proj = TTNNLinear.from_torch(self.torch_layer.k_proj)
        self.v_proj = TTNNLinear.from_torch(self.torch_layer.v_proj)
        self.o_proj = TTNNLinearIReplicatedWColSharded.from_torch(self.torch_layer.o_proj)

    @classmethod
    def from_torch(cls, torch_layer):
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_layer
        new_attn.head_dim = torch_layer.head_dim
        new_attn.scaling = torch_layer.scaling
        new_attn.is_causal = torch_layer.is_causal
        new_attn.num_key_value_groups = getattr(torch_layer, "num_key_value_groups", 1)
        new_attn.init_parameters()
        return new_attn

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()

        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )
            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        q_norm = self.torch_layer.q_norm
        self._q_norm_weight = ttnn.from_torch(
            q_norm.weight.unsqueeze(0).expand(32, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._q_norm_eps = q_norm.variance_epsilon

        k_norm = self.torch_layer.k_norm
        self._k_norm_weight = ttnn.from_torch(
            k_norm.weight.unsqueeze(0).expand(32, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._k_norm_eps = k_norm.variance_epsilon

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _is_symbiote_replicated(self, tensor) -> bool:
        """True if TorchTTNNTensor uses ReplicateTensorToMesh (skip dim=-1 AG on cos/sin or RoPE breaks)."""
        if isinstance(tensor, TorchTTNNTensor):
            cfg = tensor.ttnn_distributed_tensor_config
            if cfg is not None and cfg.mesh_mapper is not None:
                return "Replicate" in type(cfg.mesh_mapper).__name__
        return False

    def _to_ttnn(self, tensor):
        return tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor

    def _maybe_all_gather(self, tensor):
        t = self._to_ttnn(tensor)
        if not self._is_distributed:
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Ring,
        )

    def _maybe_all_gather_if_col_sharded(self, tensor):
        """All-gather column-sharded activations; pass through replicated tensors unchanged."""
        t = self._to_ttnn(tensor)
        if not self._is_distributed:
            return t
        if self._is_symbiote_replicated(tensor):
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Ring,
        )

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        expected_hidden = self.q_proj.in_features
        if hidden_states.shape[-1] != expected_hidden and self._is_distributed:
            hidden_states = self._maybe_all_gather(hidden_states)

        query_states = self._to_ttnn(self.q_proj(hidden_states))
        key_states = self._to_ttnn(self.k_proj(hidden_states))
        value_states = self._to_ttnn(self.v_proj(hidden_states))

        batch_size = query_states.shape[0]
        seq_length = query_states.shape[1]
        num_q_heads = query_states.shape[-1] // self.head_dim
        num_kv_heads = key_states.shape[-1] // self.head_dim

        query_states = ttnn.reshape(query_states, (batch_size, seq_length, num_q_heads, self.head_dim))
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))

        key_states = ttnn.reshape(key_states, (batch_size, seq_length, num_kv_heads, self.head_dim))
        key_states = ttnn.permute(key_states, (0, 2, 1, 3))

        value_states = ttnn.reshape(value_states, (batch_size, seq_length, num_kv_heads, self.head_dim))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        query_states = ttnn.rms_norm(query_states, weight=self._q_norm_weight, epsilon=self._q_norm_eps)
        key_states = ttnn.rms_norm(key_states, weight=self._k_norm_weight, epsilon=self._k_norm_eps)

        cos, sin = position_embeddings
        cos = self._maybe_all_gather_if_col_sharded(cos)
        sin = self._maybe_all_gather_if_col_sharded(sin)

        query_states, key_states = self.rope(query_states, key_states, cos, sin)
        query_states = self._to_ttnn(query_states)
        key_states = self._to_ttnn(key_states)

        if past_key_values is not None:
            mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if self._is_distributed else None
            k_torch = ttnn.to_torch(key_states, mesh_composer=mesh_composer)
            v_torch = ttnn.to_torch(value_states, mesh_composer=mesh_composer)
            if self._is_distributed:
                k_torch = k_torch[:1]
                v_torch = v_torch[:1]

            k_torch, v_torch = past_key_values.update(
                k_torch,
                v_torch,
                self.torch_layer.layer_idx,
            )

            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self._is_distributed else None
            key_states = ttnn.from_torch(
                k_torch.contiguous(),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            value_states = ttnn.from_torch(
                v_torch.contiguous(),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        attn_output = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            transpose_output=False,
        )

        attn_output = ttnn.experimental.nlp_concat_heads(self._to_ttnn(attn_output))
        attn_output = ttnn.squeeze(attn_output, 1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class TTNNQwen3Attention(TTNNModule):
    """
    TTNN implementation of Qwen3 Attention with sliding-window support
    """

    def __init__(self):
        super().__init__()

        self.sdpa = TTNNSDPAAttention()
        self.rope = TTNNQwenOmniRotaryPositionEmbedding()

        self.core_grid = ttnn.CoreGrid(y=8, x=8)

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()

        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )

            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self._is_distributed else None
        q_norm = self.torch_layer.q_norm
        self._q_norm_weight = ttnn.from_torch(
            q_norm.weight.unsqueeze(0).expand(32, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._q_norm_eps = q_norm.variance_epsilon

        k_norm = self.torch_layer.k_norm
        self._k_norm_weight = ttnn.from_torch(
            k_norm.weight.unsqueeze(0).expand(32, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._k_norm_eps = k_norm.variance_epsilon

    def init_parameters(self):
        self.q_proj = TTNNLinear.from_torch(self.torch_layer.q_proj)
        self.k_proj = TTNNLinear.from_torch(self.torch_layer.k_proj)
        self.v_proj = TTNNLinear.from_torch(self.torch_layer.v_proj)
        # Use mesh-safe output projection (same pattern as thinker attention).
        self.o_proj = TTNNLinearIReplicatedWColSharded.from_torch(self.torch_layer.o_proj)

    @classmethod
    def from_torch(cls, torch_layer):
        new_attn = cls()

        new_attn._fallback_torch_layer = torch_layer

        new_attn.num_key_value_groups = getattr(torch_layer, "num_key_value_groups", 1)

        new_attn.head_dim = torch_layer.head_dim
        new_attn.scaling = torch_layer.scaling
        new_attn.is_causal = torch_layer.is_causal

        new_attn.sliding_window = getattr(torch_layer, "sliding_window", None)

        new_attn.init_parameters()

        return new_attn

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _to_ttnn(self, tensor):
        return tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor

    def _maybe_all_gather(self, tensor):
        t = self._to_ttnn(tensor)
        if not self._is_distributed:
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Ring,
        )

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        expected_hidden = self.q_proj.in_features
        if hidden_states.shape[-1] != expected_hidden and self._is_distributed:
            hidden_states = self._maybe_all_gather(hidden_states)

        query_states = self._to_ttnn(self.q_proj(hidden_states))
        key_states = self._to_ttnn(self.k_proj(hidden_states))
        value_states = self._to_ttnn(self.v_proj(hidden_states))

        batch_size = query_states.shape[0]
        seq_length = query_states.shape[1]
        num_q_heads = query_states.shape[-1] // self.head_dim
        num_kv_heads = key_states.shape[-1] // self.head_dim

        query_states = ttnn.reshape(query_states, (batch_size, seq_length, num_q_heads, self.head_dim))
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))

        key_states = ttnn.reshape(key_states, (batch_size, seq_length, num_kv_heads, self.head_dim))
        key_states = ttnn.permute(key_states, (0, 2, 1, 3))

        value_states = ttnn.reshape(value_states, (batch_size, seq_length, num_kv_heads, self.head_dim))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        query_states = ttnn.rms_norm(query_states, weight=self._q_norm_weight, epsilon=self._q_norm_eps)
        key_states = ttnn.rms_norm(key_states, weight=self._k_norm_weight, epsilon=self._k_norm_eps)

        cos, sin = position_embeddings
        cos = self._maybe_all_gather(cos)
        sin = self._maybe_all_gather(sin)

        query_states, key_states = self.rope(
            query_states,
            key_states,
            cos,
            sin,
        )
        query_states = self._to_ttnn(query_states)
        key_states = self._to_ttnn(key_states)

        if past_key_values is not None:
            mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if self._is_distributed else None
            k_torch = ttnn.to_torch(key_states, mesh_composer=mesh_composer)
            v_torch = ttnn.to_torch(value_states, mesh_composer=mesh_composer)
            if self._is_distributed:
                k_torch = k_torch[:1]
                v_torch = v_torch[:1]

            k_torch, v_torch = past_key_values.update(
                k_torch,
                v_torch,
                self.torch_layer.layer_idx,
            )

            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self._is_distributed else None
            key_states = ttnn.from_torch(
                k_torch.contiguous(),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            value_states = ttnn.from_torch(
                v_torch.contiguous(),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        attn_output = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            transpose_output=False,
        )

        attn_output_tt = getattr(attn_output, "to_ttnn", attn_output)
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output_tt)
        attn_output = ttnn.squeeze(attn_output, 1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class TTNNQwen3OmniMoeCode2WavAttention(TTNNModule):
    """TTNN implementation of Qwen3OmniMoeCode2WavAttention (code2wav in the code2wav tower)."""

    def __init__(self):
        super().__init__()

        self.sdpa = TTNNSDPAAttention()
        self.rope = TTNNQwenOmniRotaryPositionEmbedding()
        self.core_grid = ttnn.CoreGrid(y=8, x=8)

        self.num_heads = None
        self.num_kv_heads = None
        self.num_kv_groups = None
        self.head_dim = None
        self.hidden_size = None

        self.scaling = None
        self.sliding_window = None

        self.use_windowed_attention = False

    @classmethod
    def from_torch(cls, torch_attn):
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn

        config = torch_attn.config

        new_attn.hidden_size = config.hidden_size
        new_attn.num_heads = config.num_attention_heads
        new_attn.num_kv_heads = config.num_key_value_heads
        new_attn.num_kv_groups = new_attn.num_heads // new_attn.num_kv_heads
        new_attn.head_dim = torch_attn.head_dim

        new_attn.scaling = torch_attn.scaling
        new_attn.sliding_window = config.sliding_window

        # projections
        new_attn.q_proj = TTNNLinear.from_torch(torch_attn.q_proj)
        new_attn.k_proj = TTNNLinear.from_torch(torch_attn.k_proj)
        new_attn.v_proj = TTNNLinear.from_torch(torch_attn.v_proj)
        new_attn.o_proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_attn.o_proj)

        return new_attn

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()

        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )
            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _to_ttnn(self, tensor):
        """Extract raw ttnn tensor (bypass TorchTTNNTensor shard metadata)."""
        return tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor

    def _maybe_all_gather(self, tensor):
        t = self._to_ttnn(tensor)
        if not self._is_distributed:
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Ring,
        )

    def _is_symbiote_replicated(self, tensor) -> bool:
        """Replicated cos/sin must not be all-gathered on dim=-1 (concat breaks RoPE last dim)."""
        if isinstance(tensor, TorchTTNNTensor):
            cfg = tensor.ttnn_distributed_tensor_config
            if cfg is not None and cfg.mesh_mapper is not None:
                return "Replicate" in type(cfg.mesh_mapper).__name__
        return False

    def _maybe_all_gather_if_col_sharded(self, tensor):
        t = self._to_ttnn(tensor)
        if not self._is_distributed:
            return t
        if self._is_symbiote_replicated(tensor):
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Ring,
        )

    def _prepare_code2wav_rotary_cos_sin(self, cos, sin, seq_len: int):
        """Full sequence on every chip for RoPE; skip dim=-1 gather on replicated cos (breaks head dim)."""
        if not self._is_distributed or self.device.get_num_devices() <= 1:
            return self._maybe_all_gather_if_col_sharded(cos), self._maybe_all_gather_if_col_sharded(sin)

        nd = int(self.device.get_num_devices())
        cos_t = self._to_ttnn(cos)
        sin_t = self._to_ttnn(sin)
        gather_dim = None
        for d in range(len(cos_t.shape)):
            sl = int(cos_t.shape[d])
            if sl != seq_len and sl * nd == seq_len:
                gather_dim = d
                break
        if gather_dim is not None:
            cos_t = ttnn.experimental.all_gather_async(
                cos_t,
                dim=gather_dim,
                multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
                num_links=1,
                topology=ttnn.Topology.Ring,
            )
            sin_t = ttnn.experimental.all_gather_async(
                sin_t,
                dim=gather_dim,
                multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
                num_links=1,
                topology=ttnn.Topology.Ring,
            )
            return cos_t, sin_t

        if self._is_symbiote_replicated(cos):
            return cos_t, sin_t
        return self._maybe_all_gather_if_col_sharded(cos), self._maybe_all_gather_if_col_sharded(sin)

    @staticmethod
    def _to_raw_ttnn(tensor):
        """Unwrap TorchTTNNTensor to raw ttnn.Tensor for ttnn ops that don't accept the wrapper."""
        if hasattr(tensor, "ttnn_tensor"):
            return tensor.ttnn_tensor
        return tensor

    def repeat_kv(self, x):
        if self.num_kv_groups == 1:
            return x

        x = self._to_raw_ttnn(x)
        B, H, S, D = x.shape

        x = ttnn.reshape(x, (B, H, 1, S, D))
        x = ttnn.repeat(x, (1, 1, self.num_kv_groups, 1, 1))
        x = ttnn.reshape(x, (B, H * self.num_kv_groups, S, D))

        return x

    def build_sliding_mask(self, seq_len):
        W = self.sliding_window

        mask = torch.full((seq_len, seq_len), float("-inf"))
        for i in range(seq_len):
            start = max(0, i - W)
            mask[i, start : i + 1] = 0

        mask = mask.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)  # [1,1,S,S]

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self._is_distributed else None
        return ttnn.from_torch(
            mask,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mesh_replicate_full_attention_mask(self, attention_mask, seq_len: int):
        t = self._to_ttnn(attention_mask)
        if not self._is_distributed:
            if t.dtype != ttnn.bfloat16:
                return ttnn.typecast(t, ttnn.bfloat16)
            return t

        if int(t.shape[-1]) == int(t.shape[-2]) == seq_len:
            if t.dtype != ttnn.bfloat16:
                return ttnn.typecast(t, ttnn.bfloat16)
            return attention_mask

        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
        torch_m = ttnn.to_torch(t, mesh_composer=mesh_composer)
        if torch_m.dtype in (torch.uint8, torch.bool):
            tf = torch_m.to(torch.float32)
            torch_m = torch.where(tf > 0, 0.0, float("-inf")).to(torch.bfloat16)
        else:
            torch_m = torch_m.to(torch.bfloat16)

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(
            torch_m.contiguous(),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        expected_hidden = self.q_proj.in_features
        if hidden_states.shape[-1] != expected_hidden and self._is_distributed:
            hidden_states = self._maybe_all_gather(hidden_states)

        hs = self._to_ttnn(hidden_states)
        B, S, _ = hs.shape

        query = self._to_ttnn(self.q_proj(hidden_states))
        key = self._to_ttnn(self.k_proj(hidden_states))
        value = self._to_ttnn(self.v_proj(hidden_states))

        query = ttnn.reshape(query, (B, S, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (B, S, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(value, (B, S, self.num_kv_heads, self.head_dim))

        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.permute(value, (0, 2, 1, 3))

        cos, sin = position_embeddings
        cos, sin = self._prepare_code2wav_rotary_cos_sin(cos, sin, S)
        query, key = self.rope(query, key, cos, sin)
        query = self._to_ttnn(query)
        key = self._to_ttnn(key)

        if past_key_values is not None:
            mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if self._is_distributed else None
            k_torch = ttnn.to_torch(key, mesh_composer=mesh_composer)
            v_torch = ttnn.to_torch(value, mesh_composer=mesh_composer)
            if self._is_distributed:
                k_torch = k_torch[:1]
                v_torch = v_torch[:1]

            k_torch, v_torch = past_key_values.update(
                k_torch,
                v_torch,
                self._fallback_torch_layer.layer_idx,
            )

            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self._is_distributed else None
            key = ttnn.from_torch(
                k_torch.contiguous(),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            value = ttnn.from_torch(
                v_torch.contiguous(),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        key = self.repeat_kv(key)
        value = self.repeat_kv(value)

        if self.use_windowed_attention and past_key_values is None and S % self.sliding_window == 0:
            W = self.sliding_window
            num_windows = S // W

            query_w = ttnn.view(query, (B, self.num_heads, num_windows, W, self.head_dim))
            key_w = ttnn.view(key, (B, self.num_heads, num_windows, W, self.head_dim))
            value_w = ttnn.view(value, (B, self.num_heads, num_windows, W, self.head_dim))

            query_w = ttnn.reshape(query_w, (B * num_windows, self.num_heads, W, self.head_dim))
            key_w = ttnn.reshape(key_w, (B * num_windows, self.num_heads, W, self.head_dim))
            value_w = ttnn.reshape(value_w, (B * num_windows, self.num_heads, W, self.head_dim))

            attn_output = self.sdpa(
                self,
                query_w,
                key_w,
                value_w,
                None,
                dropout=0.0,
                scaling=self.scaling,
                is_causal=True,
                transpose_output=True,
            )

            attn_output = ttnn.reshape(
                self._to_raw_ttnn(attn_output),
                (B, num_windows, W, self.num_heads, self.head_dim),
            )
            attn_output = ttnn.permute(attn_output, (0, 3, 1, 2, 4))
            attn_output = ttnn.reshape(attn_output, (B, self.num_heads, S, self.head_dim))

        else:
            if attention_mask is None:
                attention_mask = self.build_sliding_mask(S)
            am = self._to_ttnn(attention_mask)
            if self._is_distributed and int(am.shape[-1]) != int(am.shape[-2]):
                attention_mask = self._mesh_replicate_full_attention_mask(attention_mask, S)

            attn_output = self.sdpa(
                self,
                query,
                key,
                value,
                attention_mask,
                dropout=0.0,
                scaling=self.scaling,
                is_causal=False,
                transpose_output=True,
            )

        attn_output = ttnn.reshape(self._to_raw_ttnn(attn_output), (B, S, self.hidden_size))
        output = self.o_proj(attn_output)

        return output, None


class TTNNQwenAudioAttention(TTNNModule):
    """TTNN implementation of Qwen3AudioAttentionOptimized (audio attention in the audio tower)."""

    def __init__(self):
        super().__init__()

        self.embed_dim = None
        self.num_heads = None
        self.head_dim = None
        self.scaling = None

        self.qkv_proj = None
        self.out_proj = None

        self.sdpa = TTNNSDPAAttention()
        self.core_grid = ttnn.CoreGrid(y=8, x=8)

        self.is_causal = False

    @classmethod
    def from_torch(cls, torch_attn):
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn

        new_attn.embed_dim = torch_attn.embed_dim
        new_attn.num_heads = torch_attn.num_heads
        new_attn.head_dim = torch_attn.head_dim
        new_attn.scaling = torch_attn.scaling

        # ---- fuse QKV weights ----

        qkv_weight = torch.cat(
            [
                torch_attn.q_proj.weight,
                torch_attn.k_proj.weight,
                torch_attn.v_proj.weight,
            ],
            dim=0,
        )

        q_bias = torch_attn.q_proj.bias
        k_bias = torch_attn.k_proj.bias
        v_bias = torch_attn.v_proj.bias

        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

        fused_qkv = torch.nn.Linear(
            torch_attn.embed_dim,
            torch_attn.embed_dim * 3,
            bias=True,
        )

        fused_qkv.weight = torch.nn.Parameter(qkv_weight)
        fused_qkv.bias = torch.nn.Parameter(qkv_bias)

        new_attn.qkv_proj = TTNNLinear.from_torch(fused_qkv)
        new_attn.out_proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_attn.out_proj)

        return new_attn

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()

        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )

            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _to_ttnn(self, tensor):
        return tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor

    def _maybe_all_gather(self, tensor):
        t = self._to_ttnn(tensor)
        if not self._is_distributed:
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Ring,
        )

    @staticmethod
    def _last_hidden_dim(hidden_states) -> int:
        return int(hidden_states.shape[-1])

    def _leading_seq_len(self, hidden_states) -> int:
        shape = hidden_states.shape
        ed = self.embed_dim
        if len(shape) == 2:
            return int(shape[0])
        if len(shape) == 3:
            last = int(shape[-1])
            if ed is not None and last == ed:
                return int(shape[1])
            if int(shape[0]) == 1:
                return int(shape[1])
        return int(shape[0])

    def _to_torch_mesh_concat(self, tensor):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        if isinstance(tensor, torch.Tensor) and not isinstance(tensor, TorchTTNNTensor):
            return tensor
        if isinstance(tensor, TorchTTNNTensor):
            return tensor.to_torch
        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if self._is_distributed else None
        return ttnn.to_torch(self._to_ttnn(tensor), mesh_composer=mesh_composer)

    def _cu_seqlens_to_torch_int64(self, cu_seqlens):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        if cu_seqlens is None:
            return None
        if isinstance(cu_seqlens, TorchTTNNTensor):
            cu = self._to_torch_mesh_concat(cu_seqlens)
        elif isinstance(cu_seqlens, torch.Tensor):
            return cu_seqlens.detach().cpu().flatten().to(torch.int64)
        else:
            cu = self._to_torch_mesh_concat(cu_seqlens)
        if self._is_distributed:
            n = self.device.get_num_devices()
            if cu.dim() == 2 and int(cu.shape[0]) == n:
                cu = cu[0]
        return cu.flatten().to(torch.int64)

    @staticmethod
    def _cu_seqlens_allows_ttnn_from_flat(cu_flat: torch.Tensor | None, seq_len: int) -> bool:
        if cu_flat is None:
            return True
        if cu_flat.numel() < 2:
            return True
        n_seg = cu_flat.numel() - 1
        if n_seg > 1:
            return False
        return int(cu_flat[0].item()) == 0 and int(cu_flat[-1].item()) == seq_len

    @staticmethod
    def _varlen_additive_mask_torch(cu_flat: torch.Tensor, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
        cu = cu_flat.flatten().to(torch.int64)
        for i in range(1, cu.numel()):
            a = int(cu[i - 1].item())
            b = int(cu[i].item())
            if a < b and b <= seq_len:
                mask[..., a:b, a:b] = 0
        return mask

    def _additive_mask_torch_to_ttnn(self, mask_torch: torch.Tensor) -> ttnn.Tensor:
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self._is_distributed else None
        return ttnn.from_torch(
            mask_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
        cu_seqlens=None,
        **kwargs,
    ):
        cu_flat = self._cu_seqlens_to_torch_int64(cu_seqlens)
        if cu_flat is not None and cu_flat.numel() >= 2:
            seq_len = int(cu_flat[-1].item())
        else:
            seq_len = self._leading_seq_len(hidden_states)

        allows_single_segment = self._cu_seqlens_allows_ttnn_from_flat(cu_flat, seq_len)

        sdpa_attn_mask = attention_mask
        if cu_flat is not None and cu_flat.numel() >= 2 and not allows_single_segment:
            sdpa_attn_mask = self._additive_mask_torch_to_ttnn(
                self._varlen_additive_mask_torch(cu_flat, seq_len, torch.bfloat16)
            )

        expected_hidden = self.embed_dim
        if self._last_hidden_dim(hidden_states) != expected_hidden and self._is_distributed:
            hidden_states = self._maybe_all_gather(hidden_states)

        if len(hidden_states.shape) == 3 and int(hidden_states.shape[0]) == 1:
            hidden_states = ttnn.squeeze(hidden_states, 0)

        hidden_states = self._to_ttnn(hidden_states)

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(
                hidden_states,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        hidden_states = ttnn.unsqueeze(hidden_states, 0)
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.unsqueeze(hidden_states, 1)

        qkv_out = self.qkv_proj(hidden_states)
        if hasattr(qkv_out, "to_ttnn"):
            qkv = qkv_out.to_ttnn
        elif getattr(qkv_out, "ttnn_tensor", None) is not None:
            qkv = qkv_out.ttnn_tensor
        else:
            qkv = qkv_out

        qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)

        query, key, value = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
        )

        ttnn.deallocate(qkv)

        attn_output = self.sdpa(
            self,
            query,
            key,
            value,
            sdpa_attn_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=False,
            transpose_output=False,
        )

        attn_output_tt = getattr(attn_output, "to_ttnn", attn_output)
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output_tt)

        attn_output = ttnn.squeeze(attn_output, 0)

        attn_output = self.out_proj(attn_output)

        return attn_output


# ========================================================================
# Merged from modules/qwen_omni_mlp.py
# ========================================================================


def _normalize_qwen_omni_vision_act(torch_mlp) -> str:
    """Map HF ACT2FN to gelu vs silu (use class name for nn.Module; __name__ alone was wrong for GELUTanh)."""
    act_fn = getattr(torch_mlp, "act_fn", None)
    if act_fn is None:
        return "silu"
    if isinstance(act_fn, nn.Module):
        cn = act_fn.__class__.__name__.lower()
        if "gelu" in cn:
            return "gelu"
        if "silu" in cn or "swish" in cn:
            return "silu"
    name = (getattr(act_fn, "__name__", None) or type(act_fn).__name__ or "").lower()
    if "gelu" in name:
        return "gelu"
    if "silu" in name or "swish" in name:
        return "silu"
    return "silu"


class TTNNQwen3OmniVisionMLP(TTNNModule):
    """TTNN implementation of Qwen3OmniMoeVisionMLP (fc1 -> act -> fc2)."""

    def __init__(self):
        super().__init__()
        self.hidden_size = None
        self.intermediate_size = None

        self.linear_fc1 = None
        self.linear_fc2 = None

        # Normalized: "gelu" | "silu" (see _normalize_qwen_omni_vision_act).
        self.act_fn = None

    @classmethod
    def from_torch(cls, torch_mlp):
        module = cls()
        module._fallback_torch_layer = torch_mlp

        module.hidden_size = torch_mlp.hidden_size
        module.intermediate_size = torch_mlp.intermediate_size

        # TP MLP: fc1 col-shard intermediate; fc2 all-reduce to replicated output.
        module.linear_fc1 = TTNNLinearIReplicatedWColSharded.from_torch(torch_mlp.linear_fc1)
        module.linear_fc2 = TTNNQwenOmniIColShardedWAllReduced.from_torch(torch_mlp.linear_fc2)

        module.act_fn = _normalize_qwen_omni_vision_act(torch_mlp)

        return module

    def preprocess_weights_impl(self):
        self.linear_fc1.preprocess_weights()
        self.linear_fc2.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.linear_fc1.move_weights_to_device()
        self.linear_fc2.move_weights_to_device()

    def deallocate_weights_impl(self):
        self.linear_fc1.deallocate_weights()
        self.linear_fc2.deallocate_weights()

    def set_output_tensors_config_impl(self, output_tensors):
        """After FC2: materialize one [N,hidden] replica on elem (avoid dim=-1 concat vs residual); see post_process_ttnn_module_output."""
        if self.device_state is None or self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        def _materialize_one_replica(e):
            from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

            if not isinstance(e, TorchTTNNTensor) or e.ttnn_tensor is None:
                return e
            t = e.ttnn_tensor
            n = int(t.shape[0])
            h = int(self.hidden_size)
            # Replicated per device: concat on batch dim, then take first replica (MoE-style).
            pt = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            if pt.shape[0] > n:
                pt = pt[:n]
            if pt.shape[-1] > h:
                pt = pt[..., :h]
            e.elem = pt.contiguous()
            e.ttnn_tensor = None
            if getattr(e, "_distributed_tensor_config", None) is not None:
                e._distributed_tensor_config = None
            return e

        return tree_map(_materialize_one_replica, output_tensors)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        # Most TTNN paths already provide a ttnn.Tensor; keep this conversion as a safety net.
        if not isinstance(hidden_states, ttnn.Tensor):
            hidden_states = ttnn.from_torch(
                hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )

        # Full hidden width: AG if sharded; slice if concat width > hidden (mesh artifacts).
        in_width = int(hidden_states.shape[-1])
        if in_width > int(self.hidden_size):
            rank = len(hidden_states.shape)
            starts = [0] * rank
            ends = [int(s) for s in hidden_states.shape]
            ends[-1] = int(self.hidden_size)
            hidden_states = ttnn.slice(hidden_states, starts, ends)
        elif in_width < int(self.hidden_size):
            hidden_states = ttnn.all_gather(
                hidden_states,
                dim=-1,
                cluster_axis=1,
                num_links=1,
                topology=ttnn.Topology.Linear,
            )

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden_states = self.linear_fc1(hidden_states)

        # HF vision ACT2FN (often GELUTanh).
        if self.act_fn == "gelu":
            hidden_states = ttnn.gelu(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            hidden_states = ttnn.silu(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden_states = self.linear_fc2(hidden_states)

        # Slice FC2 output if width > hidden (concat semantics).
        out_width = int(hidden_states.shape[-1])
        if out_width > int(self.hidden_size):
            rank = len(hidden_states.shape)
            starts = [0] * rank
            ends = [int(s) for s in hidden_states.shape]
            ends[-1] = int(self.hidden_size)
            hidden_states = ttnn.slice(hidden_states, starts, ends)

        return hidden_states


class TTNNQwen3OmniTalkerResizeMLP(TTNNModule):
    """TalkerResizeMLP: same TP as TTNNQwen3OmniVisionMLP (fc1 col-shard, fc2 all-reduce); thinker_hidden → text hidden."""

    def __init__(self):
        super().__init__()
        self.input_hidden_size = None
        self.intermediate_size = None
        self.output_hidden_size = None
        self.linear_fc1 = None
        self.linear_fc2 = None
        self.act_fn = None
        self._released_prefill_traces = False

    @classmethod
    def from_torch(cls, torch_mlp):
        module = cls()
        module._fallback_torch_layer = torch_mlp
        module.input_hidden_size = int(torch_mlp.linear_fc1.in_features)
        module.intermediate_size = int(torch_mlp.linear_fc1.out_features)
        module.output_hidden_size = int(torch_mlp.linear_fc2.out_features)
        module.linear_fc1 = TTNNLinearIReplicatedWColSharded.from_torch(torch_mlp.linear_fc1)
        module.linear_fc2 = TTNNQwenOmniIColShardedWAllReduced.from_torch(torch_mlp.linear_fc2)
        module.act_fn = _normalize_qwen_omni_vision_act(torch_mlp)
        return module

    def preprocess_weights_impl(self):
        self.linear_fc1.preprocess_weights()
        self.linear_fc2.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.linear_fc1.move_weights_to_device()
        self.linear_fc2.move_weights_to_device()

    def deallocate_weights_impl(self):
        self.linear_fc1.deallocate_weights()
        self.linear_fc2.deallocate_weights()

    def set_output_tensors_config_impl(self, output_tensors):
        if self.device_state is None or self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        def _materialize_one_replica(e):
            from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

            if not isinstance(e, TorchTTNNTensor) or e.ttnn_tensor is None:
                return e
            t = e.ttnn_tensor
            n = int(t.shape[0])
            h = int(self.output_hidden_size)
            pt = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            if pt.shape[0] > n:
                pt = pt[:n]
            if pt.shape[-1] > h:
                pt = pt[..., :h]
            e.elem = pt.contiguous()
            e.ttnn_tensor = None
            if getattr(e, "_distributed_tensor_config", None) is not None:
                e._distributed_tensor_config = None
            return e

        return tree_map(_materialize_one_replica, output_tensors)

    @staticmethod
    def _safe_deallocate(tensor):
        try:
            ttnn.deallocate(tensor)
        except Exception:
            pass

    def _materialize_one_replica_torch(self, tensor):
        mesh_composer = None
        if self.device is not None and self.device.get_num_devices() > 1:
            mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        pt = ttnn.to_torch(tensor, mesh_composer=mesh_composer)
        n = int(tensor.shape[0])
        h = int(self.output_hidden_size)
        if pt.shape[0] > n:
            pt = pt[:n]
        if pt.shape[-1] > h:
            pt = pt[..., :h]
        return pt.contiguous()

    def _forward_device_impl(self, hidden_states):
        hidden_states = self.linear_fc1(hidden_states)
        if self.act_fn == "gelu":
            activated = ttnn.gelu(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            activated = ttnn.silu(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._safe_deallocate(hidden_states)

        hidden_states = self.linear_fc2(activated)
        self._safe_deallocate(activated)

        out_w = int(hidden_states.shape[-1])
        if out_w > int(self.output_hidden_size):
            rank = len(hidden_states.shape)
            starts = [0] * rank
            ends = [int(s) for s in hidden_states.shape]
            ends[-1] = int(self.output_hidden_size)
            sliced = ttnn.slice(hidden_states, starts, ends)
            self._safe_deallocate(hidden_states)
            hidden_states = sliced

        return hidden_states

    def _forward_device(self, hidden_states, *, disable_child_trace: bool = False):
        if not disable_child_trace:
            return self._forward_device_impl(hidden_states)

        from models.experimental.tt_symbiote.core import run_config as run_config_module

        was_tracing = run_config_module._TRACE_RUNNING
        run_config_module._TRACE_RUNNING = True
        try:
            return self._forward_device_impl(hidden_states)
        finally:
            run_config_module._TRACE_RUNNING = was_tracing

    def _forward_chunked_to_torch(self, hidden_states, chunk_size: int):
        shape = [int(s) for s in hidden_states.shape]
        seq_dim = len(shape) - 2
        seq_len = shape[seq_dim]
        chunks = []

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            starts = [0] * len(shape)
            ends = list(shape)
            starts[seq_dim] = start
            ends[seq_dim] = end

            chunk = ttnn.slice(hidden_states, starts, ends)
            out = self._forward_device(chunk, disable_child_trace=True)
            chunks.append(self._materialize_one_replica_torch(out))
            self._safe_deallocate(out)
            self._safe_deallocate(chunk)

        return torch.cat(chunks, dim=seq_dim)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        if os.environ.get("TT_SYMBIOTE_RUN_MODE") == "TRACED" and not self._released_prefill_traces:
            if TracedRun.cache_size() > 0:
                ttnn.synchronize_device(self.device)
                TracedRun.release_all()
            self._released_prefill_traces = True

        if not isinstance(hidden_states, ttnn.Tensor):
            hidden_states = ttnn.from_torch(
                hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )

        in_w = int(hidden_states.shape[-1])
        if in_w > int(self.input_hidden_size):
            rank = len(hidden_states.shape)
            starts = [0] * rank
            ends = [int(s) for s in hidden_states.shape]
            ends[-1] = int(self.input_hidden_size)
            hidden_states = ttnn.slice(hidden_states, starts, ends)
        elif in_w < int(self.input_hidden_size):
            hidden_states = ttnn.all_gather(
                hidden_states,
                dim=-1,
                cluster_axis=1,
                num_links=1,
                topology=ttnn.Topology.Linear,
            )

        traced_mode = os.environ.get("TT_SYMBIOTE_RUN_MODE") == "TRACED"
        chunk_size = int(
            os.environ.get(
                "TT_SYMBIOTE_QWEN_OMNI_RESIZE_MLP_SEQ_CHUNK",
                os.environ.get("TT_SYMBIOTE_MOE_SEQ_CHUNK", "512"),
            )
        )
        if traced_mode and len(hidden_states.shape) >= 2:
            seq_len = int(hidden_states.shape[-2])
            if chunk_size > 0 and seq_len > chunk_size:
                return self._forward_chunked_to_torch(hidden_states, chunk_size)

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return self._forward_device(hidden_states, disable_child_trace=traced_mode)


# ========================================================================
# Merged from modules/qwen_omni_decoder.py (code2wav conv helpers + classes;
# shared mesh/readback/upload helpers are defined once at top of this file.)
# ========================================================================


def _code2wav_bct_replicated_mesh_config(mesh_device):
    """code2wav ``[B, C, T]`` activations are replicated per mesh device (see ``TTNNQwenOmniConv2dNHWC``)."""
    return qwen_omni_replicated_concat_dim0_tensor_config(mesh_device)


def _ensure_code2wav_bct_full_t(out, mesh_device, expected_t: int):
    """Stitch width-sharded BCT conv shards on time and re-upload with ReplicateTensorToMesh; no-op if each device already has full T."""
    if mesh_device is None or not hasattr(mesh_device, "get_num_devices"):
        return out
    nd = int(mesh_device.get_num_devices())
    if nd <= 1:
        return out

    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    tt = None
    if isinstance(out, TorchTTNNTensor):
        tt = out.ttnn_tensor if out.ttnn_tensor is not None else None
    elif isinstance(out, ttnn.Tensor):
        tt = out
    if tt is None:
        return out

    shards = ttnn.get_device_tensors(tt)
    if len(shards) != nd:
        return out
    shard_t = int(shards[0].shape[-1])
    if shard_t == expected_t:
        return out
    if shard_t * nd != expected_t:
        return out

    parts = [ttnn.to_torch(s).contiguous() for s in shards]
    full = torch.cat(parts, dim=-1)
    return ttnn.from_torch(
        full,
        dtype=torch_dtype_to_ttnn_dtype(full.dtype),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _upload_bct_replicated(x_t: torch.Tensor, mesh_device):
    """Upload a host ``[B, C, T]`` torch tensor to TTNN with ``ReplicateTensorToMesh``."""
    mesh_mapper = None
    if mesh_device is not None and hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() > 1:
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.from_torch(
        x_t.contiguous(),
        dtype=torch_dtype_to_ttnn_dtype(x_t.dtype),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
    )


def _materialize_code2wav_bct_from_ttnn(tt_tensor: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """One logical [B,C,T] on host; prefer one replica readback (dim=0 compose can truncate T on mismatched mesh metadata)."""
    return _ttnn_mesh_to_torch_one_replica(tt_tensor, mesh_device)


def _materialize_code2wav_chain_output(x, mesh_device) -> torch.Tensor:
    """Convert symbiote / TTNN activations after conv to plain ``torch`` for ``+ residual``."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(x, TorchTTNNTensor):
        if x.ttnn_tensor is not None:
            return _materialize_code2wav_bct_from_ttnn(x.ttnn_tensor, mesh_device)
        return x.elem.contiguous()
    if isinstance(x, ttnn.Tensor):
        return _materialize_code2wav_bct_from_ttnn(x, mesh_device)
    return x.contiguous()


class TTNNQwen3OmniMoeCausalConvNet(TTNNModule):
    """HF ``Qwen3OmniMoeCausalConvNet``: causal padding on host ``torch``, convolution via :class:`TTNNConv1d`."""

    def __init__(self):
        super().__init__()
        self.conv = None
        self.stride = None
        self.kernel_size = None
        self.dilation = None
        self.padding = None

    @classmethod
    def from_torch(cls, m, *args, **kwargs):
        from models.experimental.tt_symbiote.models.qwen_omni.qwen_omni_modules import TTNNConv1d

        new = cls()
        new._fallback_torch_layer = m
        new.stride = m.stride
        new.kernel_size = m.kernel_size
        new.dilation = m.dilation
        new.padding = m.padding
        new.conv = TTNNConv1d.from_torch(m.conv)
        return new

    @staticmethod
    def _causal_conv_host_time_threshold() -> int:
        """Above this padded T, run HF nn.Conv1d on host (TTNN conv OOMs on very long code2wav). Env: TT_SYMBIOTE_CODE2WAV_CAUSAL_CONV_HOST_T (default 8192)."""
        raw = os.environ.get("TT_SYMBIOTE_CODE2WAV_CAUSAL_CONV_HOST_T", "8192")
        try:
            v = int(raw)
        except ValueError:
            v = 8192
        return max(512, v)

    def _get_extra_padding_for_conv1d(self, hidden_state: torch.Tensor) -> int:
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def set_output_tensors_config_impl(self, output_tensors):
        if self.conv is not None:
            return self.conv.set_output_tensors_config_impl(output_tensors)
        return super().set_output_tensors_config_impl(output_tensors)

    def forward(self, hidden_state):
        x_t = _materialize_code2wav_chain_output(hidden_state, self.device)
        extra_padding = self._get_extra_padding_for_conv1d(x_t)
        t_padded = int(x_t.shape[-1]) + self.padding + int(extra_padding)
        expected_t = (t_padded - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        x_t = F.pad(x_t, (self.padding, extra_padding), mode="constant", value=0)
        t_pad = int(x_t.shape[-1])
        # Long streams: TTNN conv can OOM (e.g. T~5e5); host path uses same padded tensor and weights as _fallback_torch_layer.conv.
        if t_pad > self._causal_conv_host_time_threshold():
            hf = self._fallback_torch_layer
            with torch.no_grad():
                out_t = hf.conv(x_t).contiguous()
            out = _upload_bct_replicated(out_t, self.device)
            return _ensure_code2wav_bct_full_t(out, self.device, expected_t)
        out = self.conv(x_t)
        return _ensure_code2wav_bct_full_t(out, self.device, expected_t)


class TTNNQwen3OmniMoeCausalTransConvNet(TTNNModule):
    """HF ``Qwen3OmniMoeCausalTransConvNet``: TTNN transpose conv, then host crop to match HF time trimming."""

    def __init__(self):
        super().__init__()
        self.conv = None
        self.left_pad = None
        self.right_pad = None

    @classmethod
    def from_torch(cls, m, *args, **kwargs):
        from models.experimental.tt_symbiote.models.qwen_omni.qwen_omni_modules import TTNNConvTranspose1d

        new = cls()
        new._fallback_torch_layer = m
        new.left_pad = int(m.left_pad)
        new.right_pad = int(m.right_pad)
        new.conv = TTNNConvTranspose1d.from_torch(m.conv)
        return new

    @staticmethod
    def _causal_trans_conv_host_time_threshold():
        """If T exceeds threshold, run HF transposed conv on host (DRAM/clicks vs TTNN). Env TT_SYMBIOTE_CODE2WAV_TRANS_CONV_HOST_T (default 1; 0 = TTNN-only)."""
        raw = os.environ.get("TT_SYMBIOTE_CODE2WAV_TRANS_CONV_HOST_T", "1")
        try:
            v = int(raw)
        except ValueError:
            v = 1
        if v <= 0:
            return None
        return max(1, v)

    def set_output_tensors_config_impl(self, output_tensors):
        if self.conv is not None:
            return self.conv.set_output_tensors_config_impl(output_tensors)
        return super().set_output_tensors_config_impl(output_tensors)

    def forward(self, hidden_state):
        x_t = _materialize_code2wav_chain_output(hidden_state, self.device)
        t_in = int(x_t.shape[-1])
        th = self._causal_trans_conv_host_time_threshold()
        dev = self.device
        mesh_mapper = None
        if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(dev)

        if th is not None and t_in > th:
            hf = self._fallback_torch_layer
            with torch.no_grad():
                y_t = hf(x_t).contiguous()
            return ttnn.from_torch(
                y_t,
                dtype=torch_dtype_to_ttnn_dtype(y_t.dtype),
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=mesh_mapper,
            )

        out = self.conv(x_t)
        if isinstance(out, ttnn.Tensor):
            y_t = _materialize_code2wav_bct_from_ttnn(out, self.device)
        else:
            y_t = out
        t_out = int(y_t.shape[-1])
        end = t_out - self.right_pad
        y_t = y_t[..., self.left_pad : end].contiguous()
        return ttnn.from_torch(
            y_t,
            dtype=torch_dtype_to_ttnn_dtype(y_t.dtype),
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=mesh_mapper,
        )


class TTNNQwen3OmniMoeConvNeXtBlock(TTNNModule):
    """ConvNeXtBlock: TTNN depthwise (TTNNQwen3OmniMoeCausalConvNet) + host LayerNorm/pwconv/GELU/residual so branch and shortcut match T."""

    def __init__(self):
        super().__init__()
        self.dwconv = None

    @classmethod
    def from_torch(cls, m, *args, **kwargs):
        new = cls()
        new._fallback_torch_layer = m
        new.dwconv = TTNNQwen3OmniMoeCausalConvNet.from_torch(m.dwconv)
        return new

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _code2wav_bct_replicated_mesh_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, hidden_states):
        dev = self.device
        residual = _materialize_code2wav_chain_output(hidden_states, dev)

        hidden_states = self.dwconv(hidden_states)
        x = _materialize_code2wav_chain_output(hidden_states, dev)

        hf = self._fallback_torch_layer
        x = x.permute(0, 2, 1)
        x = hf.norm(x)
        x = hf.pwconv1(x)
        x = hf.act(x)
        x = hf.pwconv2(x)
        x = hf.gamma * x
        x = x.permute(0, 2, 1)

        result = x + residual
        return _upload_bct_replicated(result, dev)


class TTNNQwen3OmniMoeCode2WavDecoderResidualUnit(TTNNModule):
    """code2wav residual unit: TTNN SnakeBeta + CausalConvNet; host add (conv stitches T for branch vs shortcut)."""

    def __init__(self):
        super().__init__()
        self.act1 = None
        self.conv1 = None
        self.act2 = None
        self.conv2 = None

    @classmethod
    def from_torch(cls, m, *args, **kwargs):
        from models.experimental.tt_symbiote.models.qwen_omni.qwen_omni_modules import TTNNSnakeBeta

        new = cls()
        new._fallback_torch_layer = m
        new.act1 = TTNNSnakeBeta.from_torch(m.act1)
        new.conv1 = TTNNQwen3OmniMoeCausalConvNet.from_torch(m.conv1)
        new.act2 = TTNNSnakeBeta.from_torch(m.act2)
        new.conv2 = TTNNQwen3OmniMoeCausalConvNet.from_torch(m.conv2)
        return new

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _code2wav_bct_replicated_mesh_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, hidden_state):
        dev = self.device
        if self.act1 is not None:
            self.act1._bypass_tensor_wrapping = False
        if self.act2 is not None:
            self.act2._bypass_tensor_wrapping = False

        residual = _materialize_code2wav_chain_output(hidden_state, dev)

        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)

        branch = _materialize_code2wav_chain_output(hidden_state, dev)
        result = branch + residual
        return _upload_bct_replicated(result, dev)


# ========================================================================
# Merged from modules/qwen_omni_lm_head.py
# ========================================================================


def _lm_head_logits_dtensor_config(mesh_device):
    """Replicated logits: compose then slice dim 0 so HF sampling sees ``[batch, …]`` not ``[batch*n_dev, …]``."""
    return qwen_omni_replicated_concat_dim0_tensor_config(mesh_device)


class TTNNQwenOmniThinkerLmHead(TTNNModule):
    """Thinker logits: all-gather hidden if sharded (like TTNNQwen3OmniAttention), linear, replicated readback with dim0 slice for generate. Chunked matmul when env byte caps hit."""

    @classmethod
    def from_torch(cls, linear: nn.Linear):
        m = cls()
        m._fallback_torch_layer = linear
        m.in_features = int(linear.in_features)
        m.out_features = int(linear.out_features)
        m.weight = linear.weight
        m.bias = linear.bias
        return m

    def preprocess_weights_impl(self):
        self.tt_weight_host = preprocess_linear_weight(self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(self.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def move_weights_to_device_impl(self):
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None

    def deallocate_weights_impl(self):
        if getattr(self, "tt_weight", None) is not None:
            ttnn.deallocate(self.tt_weight)
            self.tt_weight = None
        if getattr(self, "tt_bias", None) is not None:
            ttnn.deallocate(self.tt_bias)
            self.tt_bias = None
        super().deallocate_weights_impl()

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _to_ttnn(self, tensor):
        return tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor

    def _maybe_all_gather_hidden(self, tensor):
        """All-gather last dim when activations are column-sharded (``last * num_devices == in_features``).

        Do **not** skip based on ``TorchTTNNTensor`` "replicated" metadata alone: symbiote can tag
        tensors replicated while the physical ttnn width is still a shard (e.g. 256 of 2048), which
        broke chunked LM-head readback (ragged ``vocab`` dims across chunks).
        """
        t = self._to_ttnn(tensor)
        last = int(t.shape[-1])
        if last == self.in_features:
            return t
        mesh = self.device
        if mesh is None or not hasattr(mesh, "get_num_devices") or mesh.get_num_devices() <= 1:
            return t
        n = int(mesh.get_num_devices())
        if last * n != self.in_features:
            return t
        if self._is_distributed:
            return ttnn.experimental.all_gather_async(
                t,
                dim=-1,
                multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
                num_links=1,
                topology=ttnn.Topology.Ring,
            )
        return ttnn.all_gather(
            t,
            dim=-1,
            cluster_axis=1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _lm_head_logits_dtensor_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def _readback_logits_bf16(self, tt_out: ttnn.Tensor, expected_token_rows: int) -> torch.Tensor:
        """Device logits → host ``(expected_token_rows, out_features)`` bf16 for chunk concat."""
        n = 1 if self.device is None else int(self.device.get_num_devices())
        wid = int(tt_out.shape[-1])
        if n > 1 and wid * n == self.out_features:
            pt = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=-1))
        elif n > 1:
            pt = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
        else:
            pt = ttnn.to_torch(tt_out)
        pt = pt.to(torch.bfloat16)
        while pt.dim() > 2:
            pt = pt.reshape(-1, pt.shape[-1])
        if pt.shape[0] > expected_token_rows:
            pt = pt[:expected_token_rows]
        if pt.shape[-1] > self.out_features:
            pt = pt[..., : self.out_features]
        return pt.contiguous()

    def _forward_linear_4d(self, x: ttnn.Tensor) -> ttnn.Tensor:
        input_tensor_shape = list(x.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)
        x4 = ttnn.reshape(x, input_shape)
        out = ttnn.linear(x4, self.tt_weight, bias=self.tt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(out, input_tensor_shape[:-1] + [self.out_features])

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except ValueError:
            return default

    def _effective_chunk_rows(self, n_inner: int, chunk_cap: int, hidden_last: int) -> int:
        max_chunk_out = self._env_int("TT_SYMBIOTE_LM_HEAD_MAX_CHUNK_OUTPUT_BYTES", 64 * 1024 * 1024)
        max_chunk_in = self._env_int("TT_SYMBIOTE_LM_HEAD_MAX_CHUNK_INPUT_BYTES", 32 * 1024 * 1024)
        r = min(max(1, chunk_cap), max(1, n_inner))
        while r > 1:
            if r * self.out_features * 2 <= max_chunk_out and r * hidden_last * 2 <= max_chunk_in:
                return r
            r = max(1, r // 2)
        return 1

    def forward(self, hidden_states):
        x = self._maybe_all_gather_hidden(hidden_states)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(x.shape)
        n_inner = int(math.prod(int(d) for d in input_tensor_shape[:-1]))
        hidden = int(input_tensor_shape[-1])
        if n_inner == 0:
            return ttnn.reshape(x, input_tensor_shape[:-1] + [self.out_features])

        max_out = self._env_int("TT_SYMBIOTE_LM_HEAD_MAX_OUTPUT_BYTES", 32 * 1024 * 1024)
        max_in = self._env_int("TT_SYMBIOTE_LM_HEAD_MAX_INPUT_BYTES", 48 * 1024 * 1024)
        chunk_cap = self._env_int("TT_SYMBIOTE_LM_HEAD_CHUNK_TOKENS", 16)
        est_out = n_inner * self.out_features * 2
        est_in = n_inner * hidden * 2
        # ``chunk_cap <= 0`` disables chunking (single device-sized matmul; can OOM on long prefill).
        if chunk_cap <= 0 or (est_out <= max_out and est_in <= max_in):
            return self._forward_linear_4d(x)

        row_step = self._effective_chunk_rows(n_inner, chunk_cap, hidden)
        x2 = ttnn.reshape(x, (n_inner, hidden))
        parts: list[torch.Tensor] = []
        for r0 in range(0, n_inner, row_step):
            r1 = min(r0 + row_step, n_inner)
            sub = ttnn.slice(x2, (r0, 0), (r1, hidden))
            sub4 = ttnn.reshape(sub, (1, 1, r1 - r0, hidden))
            tt_chunk = self._forward_linear_4d(sub4)
            parts.append(self._readback_logits_bf16(tt_chunk, r1 - r0))
        logits_cpu = torch.cat(parts, dim=0).reshape(*input_tensor_shape[:-1], self.out_features)
        return TorchTTNNTensor(logits_cpu)


def replace_thinker_lm_head_with_ttnn(thinker: nn.Module) -> None:
    """Replace ``thinker.lm_head`` with :class:`TTNNQwenOmniThinkerLmHead` when it is ``nn.Linear``.

    If :data:`torch.nn.Linear` was already swapped to :class:`~models.experimental.tt_symbiote.modules.linear.TTNNLinear`
    by a thinker-wide op map, recover weights from ``_fallback_torch_layer`` and install this head instead.
    """
    if "lm_head" not in getattr(thinker, "_modules", {}):
        return
    old = thinker._modules.get("lm_head")
    if old is None or isinstance(old, TTNNQwenOmniThinkerLmHead):
        return
    if isinstance(old, nn.Linear):
        thinker._modules["lm_head"] = TTNNQwenOmniThinkerLmHead.from_torch(old)
        return
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear

    if isinstance(old, TTNNLinear):
        tl = getattr(old, "_fallback_torch_layer", None)
        if isinstance(tl, nn.Linear):
            new_head = TTNNQwenOmniThinkerLmHead.from_torch(tl)
            if getattr(old, "_unique_name", None) is not None:
                new_head._unique_name = old._unique_name
            thinker._modules["lm_head"] = new_head


def _lm_head_from_linear_or_ttnn_linear(m):
    if isinstance(m, TTNNQwenOmniThinkerLmHead):
        return m
    if isinstance(m, nn.Linear):
        return TTNNQwenOmniThinkerLmHead.from_torch(m)
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear

    if isinstance(m, TTNNLinear):
        tl = getattr(m, "_fallback_torch_layer", None)
        if isinstance(tl, nn.Linear):
            nh = TTNNQwenOmniThinkerLmHead.from_torch(tl)
            if getattr(m, "_unique_name", None) is not None:
                nh._unique_name = m._unique_name
            return nh
    return m


def replace_code_predictor_lm_head_with_ttnn(talker: nn.Module) -> None:
    """Replace ``talker.code_predictor.lm_head`` (``nn.ModuleList[nn.Linear]``) with TTNN heads.

    Uses a plain ``list`` for multiple heads: ``nn.ModuleList`` only accepts ``nn.Module`` children,
    while :class:`TTNNQwenOmniThinkerLmHead` is a ``TTNNModule`` (not ``nn.Module``).
    HF only does ``self.lm_head[i](hidden_states)``, so a list is sufficient.
    """
    cp = getattr(talker, "code_predictor", None)
    if cp is None:
        return
    old = getattr(cp, "lm_head", None)
    if old is None:
        return
    if isinstance(old, nn.ModuleList):
        new_heads = [_lm_head_from_linear_or_ttnn_linear(m) for m in old]
        if "lm_head" in cp._modules:
            del cp._modules["lm_head"]
        cp.lm_head = new_heads
        return
    if isinstance(old, nn.Linear):
        cp._modules["lm_head"] = TTNNQwenOmniThinkerLmHead.from_torch(old)
        return
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear

    if isinstance(old, TTNNLinear):
        tl = getattr(old, "_fallback_torch_layer", None)
        if isinstance(tl, nn.Linear):
            nh = TTNNQwenOmniThinkerLmHead.from_torch(tl)
            if getattr(old, "_unique_name", None) is not None:
                nh._unique_name = old._unique_name
            cp._modules["lm_head"] = nh


def replace_talker_codec_head_with_ttnn(talker: nn.Module) -> None:
    """Replace ``talker.codec_head`` (``nn.Linear``) with :class:`TTNNQwenOmniThinkerLmHead`."""
    if "codec_head" not in getattr(talker, "_modules", {}):
        return
    old = talker._modules.get("codec_head")
    if old is None or isinstance(old, TTNNQwenOmniThinkerLmHead):
        return
    if isinstance(old, nn.Linear):
        talker._modules["codec_head"] = TTNNQwenOmniThinkerLmHead.from_torch(old)
        return
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear

    if isinstance(old, TTNNLinear):
        tl = getattr(old, "_fallback_torch_layer", None)
        if isinstance(tl, nn.Linear):
            new_head = TTNNQwenOmniThinkerLmHead.from_torch(tl)
            if getattr(old, "_unique_name", None) is not None:
                new_head._unique_name = old._unique_name
            talker._modules["codec_head"] = new_head


def _to_ttnn_raw(tensor):
    """Return raw ttnn.Tensor from TorchTTNNTensor or ttnn.Tensor for use in ttnn ops."""
    if tensor is None:
        raise ValueError("Expected a tensor; got None.")
    if isinstance(tensor, TorchTTNNTensor):
        if not hasattr(tensor, "to_ttnn"):
            raise AttributeError("TorchTTNNTensor has no to_ttnn property.")
        return tensor.to_ttnn
    if hasattr(tensor, "shape") and hasattr(tensor, "layout"):
        return tensor
    raise TypeError(f"Expected TorchTTNNTensor or ttnn.Tensor; got {type(tensor).__name__}.")


# ========================================================================
# Merged from modules/qwen_omni_moe.py (thinker HF MoE adapter)
# ========================================================================


def _thinker_experts_adapter(thinker_mlp):
    """Adapt HF thinker experts for TTNNExperts (needs config + gate_up/down tensors)."""
    hf_experts = thinker_mlp.experts
    cfg = getattr(hf_experts, "config", None)
    if cfg is None:
        cfg = type("ThinkerExpertsConfig", (), {})()
    cfg.hidden_size = getattr(cfg, "hidden_size", hf_experts.gate_up_proj.shape[2])
    cfg.moe_intermediate_size = getattr(cfg, "moe_intermediate_size", hf_experts.gate_up_proj.shape[1] // 2)
    cfg.n_routed_experts = getattr(cfg, "n_routed_experts", hf_experts.gate_up_proj.shape[0])
    cfg.num_experts_per_tok = getattr(cfg, "num_experts_per_tok", None) or getattr(thinker_mlp.gate, "top_k", 8)

    adapter = type("ThinkerExpertsAdapter", (), {})()
    adapter.gate_up_proj = hf_experts.gate_up_proj
    adapter.down_proj = hf_experts.down_proj
    adapter.config = cfg
    return adapter


class TTNNQwen3OmniThinkerMoE(TTNNModule):
    """Thinker MoE: HF-style linear→softmax→top-k router on device; TTNNExperts dispatch/combine. Returns torch tensor for decoder."""

    @classmethod
    def from_torch(cls, thinker_mlp):
        module = cls()
        module._fallback_torch_layer = thinker_mlp
        g = thinker_mlp.gate
        module._gate_w_torch = g.weight.data.clone()
        module.top_k = int(g.top_k)
        module.norm_topk_prob = bool(g.norm_topk_prob)
        module.num_experts = int(g.num_experts)
        experts_for_tt = _thinker_experts_adapter(thinker_mlp)
        module.experts = TTNNExperts.from_torch(experts_for_tt)
        return module

    def preprocess_weights_impl(self):
        self._gate_tt_host = preprocess_linear_weight(self._gate_w_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        del self._gate_w_torch
        self.experts.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.gate_weight_tt = ttnn.to_device(self._gate_tt_host, self.device)
        self.experts.move_weights_to_device()

    def deallocate_weights_impl(self):
        gw = getattr(self, "gate_weight_tt", None)
        if gw is not None:
            ttnn.deallocate(gw)
            self.gate_weight_tt = None
        self.experts.deallocate_weights()

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        if not self._is_distributed:
            return tensor
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

    def _moe_from_tiled_4d(self, hidden_states_tile, b, s, h, orig_batch, out_dtype):
        """Run gate + experts on TILE activations (b, 1, s, h). Returns torch (b, s, hidden_size)."""
        t = b * s
        x_2d = ttnn.reshape(hidden_states_tile, ttnn.Shape((t, h)))
        gate_logits = ttnn.linear(x_2d, self.gate_weight_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        probs = ttnn.softmax(gate_logits, dim=-1)
        ttnn.deallocate(gate_logits)

        topk_vals, topk_idx = ttnn.topk(probs, k=self.top_k, dim=-1)
        ttnn.deallocate(probs)

        if self.norm_topk_prob:
            denom = ttnn.sum(topk_vals, dim=-1, keepdim=True)
            topk_vals = ttnn.div(topk_vals, denom)
            ttnn.deallocate(denom)

        topk_idx = ttnn.to_layout(topk_idx, ttnn.ROW_MAJOR_LAYOUT)
        topk_vals = ttnn.to_layout(topk_vals, ttnn.ROW_MAJOR_LAYOUT)
        topk_idx = ttnn.reshape(topk_idx, ttnn.Shape((t, self.top_k)))
        topk_vals = ttnn.reshape(topk_vals, ttnn.Shape((t, self.top_k)))

        expert_out = self.experts.forward(hidden_states_tile, topk_idx, topk_vals)
        try:
            from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

            if isinstance(expert_out, TorchTTNNTensor):
                expert_out = expert_out.to_ttnn
        except Exception:
            pass
        expert_out = _to_ttnn_raw(expert_out)
        h_out = int(self.experts.hidden_size)
        expert_out = ttnn.reshape(expert_out, ttnn.Shape((b, s, h_out)))

        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if self.device.get_num_devices() > 1 else None
        out_torch = ttnn.to_torch(expert_out, mesh_composer=mesh_composer).to(out_dtype)
        ttnn.deallocate(expert_out)
        if mesh_composer is not None:
            out_torch = out_torch.narrow(0, 0, int(orig_batch))
        return out_torch

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        hidden_states_torch = _to_torch_any(hidden_states)
        orig_shape = hidden_states_torch.shape
        out_dtype = hidden_states_torch.dtype
        orig_batch = int(orig_shape[0])

        hidden_states_tt = _to_ttnn_raw(hidden_states)
        hidden_states_tt = self._maybe_all_gather(hidden_states_tt)
        if len(hidden_states_tt.shape) == 3:
            b, s, h = (int(hidden_states_tt.shape[0]), int(hidden_states_tt.shape[1]), int(hidden_states_tt.shape[2]))
            hidden_states_tt = ttnn.reshape(hidden_states_tt, ttnn.Shape((b, 1, s, h)))
        else:
            b, s, h = (
                int(hidden_states_tt.shape[0]),
                int(hidden_states_tt.shape[2]),
                int(hidden_states_tt.shape[3]),
            )

        seq_chunk = int(os.environ.get("TT_SYMBIOTE_MOE_SEQ_CHUNK", "1024"))
        if seq_chunk <= 0:
            seq_chunk = s + 1

        # Long prefill: drop full-sequence TILE if present (dense RM is usually smaller), then tile only each chunk.
        # Set TT_SYMBIOTE_MOE_SEQ_CHUNK=0 to force the single-shot path (legacy behavior).
        if s > seq_chunk:
            if hidden_states_tt.layout == ttnn.TILE_LAYOUT:
                hidden_rm = ttnn.to_layout(
                    hidden_states_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                try:
                    ttnn.deallocate(hidden_states_tt)
                except Exception:
                    pass
                hidden_states_tt = hidden_rm
            parts = []
            for s0 in range(0, s, seq_chunk):
                s1 = min(s0 + seq_chunk, s)
                sc = s1 - s0
                h_rm = ttnn.slice(hidden_states_tt, (0, 0, s0, 0), (b, 1, s1, h))
                h_tile = ttnn.to_layout(h_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                try:
                    ttnn.deallocate(h_rm)
                except Exception:
                    pass
                parts.append(self._moe_from_tiled_4d(h_tile, b, sc, h, orig_batch, out_dtype))
                try:
                    ttnn.deallocate(h_tile)
                except Exception:
                    pass
            out_torch = torch.cat(parts, dim=1)
            return out_torch.reshape(orig_shape)

        if hidden_states_tt.layout != ttnn.TILE_LAYOUT:
            hidden_states_tt = ttnn.to_layout(hidden_states_tt, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        out_torch = self._moe_from_tiled_4d(hidden_states_tt, b, s, h, orig_batch, out_dtype)
        return out_torch.reshape(orig_shape)


def _make_fitted_sparse_matmul_program_config(
    device,
    out_features: int,
    in0_block_w: int,
    per_core_M: int = 1,
):
    """sparse_matmul config that fits the grid to the number of output tiles."""
    grid = device.compute_with_storage_grid_size()
    max_x = int(getattr(grid, "x"))
    max_y = int(getattr(grid, "y"))
    n_tiles = (int(out_features) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE

    best = None
    for pcn in range(1, n_tiles + 1):
        n_cores = math.ceil(n_tiles / pcn)
        if n_cores > max_x * max_y:
            continue
        if n_cores * pcn != n_tiles:
            continue
        for gy in range(1, min(n_cores, max_y) + 1):
            if n_cores % gy == 0:
                gx = n_cores // gy
                if gx <= max_x:
                    best = (gx, gy, pcn)
                    break
        if best is not None:
            break

    if best is None:
        core_x, core_y = max_x, max_y
        pcn = max(1, math.ceil(n_tiles / (core_x * core_y)))
    else:
        core_x, core_y, pcn = best

    out_subblock_w = min(pcn, 4)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=int(in0_block_w),
        out_subblock_h=1,
        out_subblock_w=int(out_subblock_w),
        out_block_h=1,
        out_block_w=int(pcn),
        per_core_M=int(per_core_M),
        per_core_N=int(pcn),
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


class Qwen3RouteTokenToExperts(nn.Module):
    """Softmax-based routing for Qwen3-Coder-Next / Qwen3-Omni (vs sigmoid+bias for GLM/DeepSeek)."""

    def __init__(self, top_k, norm_topk_prob, routed_scaling_factor, n_routed_experts):
        super().__init__()
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.n_routed_experts = n_routed_experts
        self.use_softmax = True
        self.n_group = 1
        self.topk_group = 1
        self.register_buffer("e_score_correction_bias", torch.zeros(n_routed_experts, dtype=torch.float32))

    def forward(self, router_logits):
        probs = F.softmax(router_logits.to(torch.float32), dim=-1).to(router_logits.dtype)
        _, topk_indices = torch.topk(probs, k=self.top_k, dim=-1, sorted=False)
        topk_weights = probs.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class TTNNQwenOmniMoERouterDecode(TTNNMoERouterDecode):
    """MoE decode router that adds Qwen3 softmax routing (removed from generic :class:`~TTNNMoERouterDecode`)."""

    def forward(self, logits: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        r = self._fallback_torch_layer
        if getattr(r, "use_softmax", False):
            if logits.layout != ttnn.TILE_LAYOUT:
                logits = ttnn.to_layout(logits, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            logits = ttnn.reshape(logits, ttnn.Shape((1, 1, logits.shape[0], logits.shape[1])))
            if logits.dtype != ttnn.float32:
                logits_f32 = ttnn.typecast(logits, ttnn.float32)
                ttnn.deallocate(logits)
            else:
                logits_f32 = logits

            probs_f32 = ttnn.softmax(logits_f32, dim=-1)
            ttnn.deallocate(logits_f32)
            T = probs_f32.shape[2]
            probs_bf16 = ttnn.typecast(probs_f32, ttnn.bfloat16)
            _, topk_expert_idx = ttnn.topk(probs_bf16, k=r.top_k, dim=3, largest=True, sorted=False)
            ttnn.deallocate(probs_bf16)
            topk_weights = ttnn.gather(probs_f32, dim=3, index=topk_expert_idx)
            ttnn.deallocate(probs_f32)
            denom = ttnn.sum(topk_weights, dim=3, keepdim=True) + 1e-20
            topk_weights = ttnn.div(topk_weights, denom)
            ttnn.deallocate(denom)
            scale_rep = _safe_repeat(self._scale_dev, ttnn.Shape((1, 1, T, 1)))
            scale_bf16 = ttnn.to_layout(scale_rep, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            scale_f32 = ttnn.typecast(scale_bf16, ttnn.float32)
            ttnn.deallocate(scale_bf16)
            topk_weights = ttnn.mul(topk_weights, scale_f32)
            ttnn.deallocate(scale_f32)
            topk_weights = ttnn.typecast(topk_weights, ttnn.bfloat16)
            topk_expert_idx = ttnn.reshape(topk_expert_idx, ttnn.Shape((T, r.top_k)))
            topk_weights = ttnn.reshape(topk_weights, ttnn.Shape((T, r.top_k)))
            return topk_expert_idx, topk_weights

        return super().forward(logits)

    def set_output_tensors_config_impl(self, output_tensors):
        if self.device_state is None or self.device is None or self.device.get_num_devices() <= 1:
            return output_tensors

        num_devices = self.device.get_num_devices()

        def _set_config(tensor):
            try:
                shape = list(tensor.shape)
            except Exception:
                return tensor

            if len(shape) >= 1 and shape[0] % num_devices == 0:
                config = DistributedTensorConfig(
                    mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
                    mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                )
            else:
                config = DistributedTensorConfig(
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                    mesh_composer=ttnn.create_mesh_composer(self.device, ttnn.MeshComposerConfig([0, len(shape)])),
                )
            return set_distributed_tensor_config(config)(tensor)

        return tree_map(_set_config, output_tensors)


class TTNNQwen3OmniSparseMoE(TTNNMoE):
    """Qwen3 Coder-Next / Qwen3-Omni MoE (softmax route + gated shared expert). Prefer :class:`TTNNQwen3TalkerMoE` for talker HF blocks."""

    @classmethod
    def from_torch(cls, torch_moe):
        adapted_config = cls._adapt_config(torch_moe.gate, torch_moe.experts)
        module = cls(adapted_config)
        module._fallback_torch_layer = torch_moe

        zero_bias = torch.zeros(
            torch_moe.gate.weight.shape[0],
            device=torch_moe.gate.weight.device,
            dtype=torch_moe.gate.weight.dtype,
        )
        module.gate = TTNNGlm4MoeTopkRouter.from_parameters(torch_moe.gate.weight, zero_bias)
        module._gate_weight_torch = torch_moe.gate.weight.detach().T.contiguous().to(torch.bfloat16)
        module.route_tokens_to_experts = TTNNQwenOmniMoERouterDecode.from_torch(
            Qwen3RouteTokenToExperts(
                top_k=adapted_config.num_experts_per_tok,
                norm_topk_prob=adapted_config.norm_topk_prob,
                routed_scaling_factor=adapted_config.routed_scaling_factor,
                n_routed_experts=adapted_config.n_routed_experts,
            )
        )
        experts_wrapper = cls._wrap_experts(torch_moe.experts, adapted_config)
        module.experts = TTNNExperts.from_torch(experts_wrapper)
        module.shared_experts = TTNNGlm4MoeMLP.from_torch(torch_moe.shared_expert)
        module.shared_expert_gate = TTNNLinear.from_torch(torch_moe.shared_expert_gate)
        return module

    def preprocess_weights_impl(self):
        pass

    def move_weights_to_device_impl(self):
        self._gate_weight_tt = ttnn.from_torch(
            self._gate_weight_torch,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self._gate_weight_torch = None

    @run_on_devices(DeviceArch.T3K)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.num_devices = self.device.get_num_devices()
        self.num_dispatch_devices = self.device.shape[0]
        self.num_experts_per_device = even_int_div(self.config.n_routed_experts, self.num_devices)
        residual = x

        x = ttnn.experimental.all_gather_async(
            x,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_shape = list(x.shape)
        T = 1
        for d in x_shape[:-1]:
            T *= d
        H = x_shape[-1]
        if len(x_shape) == 3:
            b, s, h = int(x_shape[0]), int(x_shape[1]), int(x_shape[2])
        else:
            b, s, h = int(x_shape[0]), int(x_shape[2]), int(x_shape[3])

        seq_chunk = int(os.environ.get("TT_SYMBIOTE_MOE_SEQ_CHUNK", "1024"))
        if seq_chunk <= 0:
            seq_chunk = s + 1

        x_full = x

        if s > seq_chunk:
            x4d = ttnn.reshape(x, ttnn.Shape((b, 1, s, h))) if len(x_shape) == 3 else x
            routed_parts = []
            for s0 in range(0, s, seq_chunk):
                s1 = min(s0 + seq_chunk, s)
                x_c = ttnn.slice(x4d, (0, 0, s0, 0), (b, 1, s1, h))
                sc = s1 - s0
                Tc = b * sc
                x_2d = ttnn.reshape(x_c, ttnn.Shape((Tc, H)))
                router_logits = ttnn.linear(
                    x_2d,
                    self._gate_weight_tt,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                topk_experts_indices, topk_experts_weights = self.route_tokens_to_experts(router_logits)
                routed_parts.append(self.experts(x_c, topk_experts_indices, topk_experts_weights))
                try:
                    ttnn.deallocate(x_c)
                except Exception:
                    pass
            if len(x_shape) == 3:
                try:
                    ttnn.deallocate(x4d)
                except Exception:
                    pass
            routed_ttnn_parts = []
            for p in routed_parts:
                u = p.to_ttnn if hasattr(p, "to_ttnn") else p
                routed_ttnn_parts.append(_to_ttnn_raw(u))
            routed_output = (
                routed_ttnn_parts[0]
                if len(routed_ttnn_parts) == 1
                else ttnn.concat(routed_ttnn_parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            )
            if len(routed_ttnn_parts) > 1:
                for p in routed_ttnn_parts:
                    try:
                        ttnn.deallocate(p)
                    except Exception:
                        pass
        else:
            x_2d = ttnn.reshape(x, ttnn.Shape((T, H)))
            router_logits = ttnn.linear(
                x_2d,
                self._gate_weight_tt,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            topk_experts_indices, topk_experts_weights = self.route_tokens_to_experts(router_logits)

            if len(x.shape) == 3:
                x = ttnn.reshape(x, ttnn.Shape((x.shape[0], 1, x.shape[1], x.shape[2])))
            routed_output = self.experts(x, topk_experts_indices, topk_experts_weights)

        routed_out = routed_output.to_ttnn if hasattr(routed_output, "to_ttnn") else routed_output
        n_rs = self.device.shape[1]
        if n_rs > 1:
            routed_out = ttnn.mul(routed_out, 1.0 / float(n_rs))
        routed_output = ttnn.experimental.reduce_scatter_minimal_async(
            routed_out,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        shared_output = self.shared_experts(residual)
        gate_raw = self.shared_expert_gate(x_full)
        gate_raw = gate_raw.to_ttnn if hasattr(gate_raw, "to_ttnn") else gate_raw
        gate_val = ttnn.sigmoid(gate_raw)
        shared_raw = shared_output.to_ttnn if hasattr(shared_output, "to_ttnn") else shared_output
        gated_shared = ttnn.mul(gate_val, shared_raw)

        output = ttnn.add(routed_output, gated_shared)
        output = ttnn.squeeze(output, 1)
        return output

    @staticmethod
    def _adapt_config(gate, experts):
        class AdaptedConfig:
            pass

        config = AdaptedConfig()
        config.hidden_size = getattr(gate, "hidden_dim", gate.weight.shape[1])
        config.moe_intermediate_size = (
            getattr(experts, "intermediate_dim", None)
            or getattr(experts, "config", type("C", (), {"moe_intermediate_size": None})()).moe_intermediate_size
        )
        config.num_experts_per_tok = getattr(gate, "top_k", None)
        config.n_routed_experts = getattr(gate, "num_experts", gate.weight.shape[0])
        config.n_group = 1
        config.topk_group = 1
        config.routed_scaling_factor = 1.0
        config.norm_topk_prob = getattr(gate, "norm_topk_prob", False)
        config.hidden_act = getattr(experts, "act_fn", None) or "silu"
        if config.num_experts_per_tok is None:
            config.num_experts_per_tok = 8
        return config

    @staticmethod
    def _wrap_experts(qwen3_experts, config):
        class ExpertsWrapper:
            pass

        w = ExpertsWrapper()
        w.config = config
        w.gate_up_proj = qwen3_experts.gate_up_proj
        w.down_proj = qwen3_experts.down_proj
        return w


def _consolidate_talker_experts_from_module_list(experts_module_list, config):
    if hasattr(experts_module_list, "gate_up_proj") and hasattr(experts_module_list, "down_proj"):
        consolidated = type("ConsolidatedExperts", (), {})()
        gu = experts_module_list.gate_up_proj
        dp = experts_module_list.down_proj
        consolidated.gate_up_proj = gu.data if isinstance(gu, torch.nn.Parameter) else gu
        consolidated.down_proj = dp.data if isinstance(dp, torch.nn.Parameter) else dp
        consolidated.config = config
        return consolidated

    num_experts = len(experts_module_list)
    interm = config.moe_intermediate_size
    hidden = config.hidden_size
    gate_up_proj = torch.empty(num_experts, 2 * interm, hidden, dtype=experts_module_list[0].gate_proj.weight.dtype)
    down_proj = torch.empty(num_experts, hidden, interm, dtype=experts_module_list[0].down_proj.weight.dtype)
    for i in range(num_experts):
        gate_up_proj[i] = torch.cat(
            [experts_module_list[i].gate_proj.weight, experts_module_list[i].up_proj.weight], dim=0
        )
        down_proj[i] = experts_module_list[i].down_proj.weight
    consolidated = type("ConsolidatedExperts", (), {})()
    consolidated.gate_up_proj = gate_up_proj
    consolidated.down_proj = down_proj
    consolidated.config = config
    return consolidated


class Qwen3OmniMoeTalkerTextExpertsTTNN(TTNNExperts):
    """Qwen3-Omni talker experts: sparse_matmul + TTNNExperts dispatch/combine."""

    @classmethod
    def from_torch(cls, torch_experts, config):
        if hasattr(torch_experts, "gate_up_proj") and hasattr(torch_experts, "down_proj"):
            consolidated = torch_experts
        else:
            consolidated = _consolidate_talker_experts_from_module_list(torch_experts, config)
        module = cls(config)
        module._fallback_torch_layer = consolidated
        intermediate = config.moe_intermediate_size
        gu = consolidated.gate_up_proj
        dp = consolidated.down_proj
        gu_t = gu.data if isinstance(gu, torch.nn.Parameter) else gu
        dp_t = dp.data if isinstance(dp, torch.nn.Parameter) else dp
        module.torch_w1_proj = gu_t[:, :intermediate, :].permute(0, 2, 1).contiguous()
        module.torch_w3_proj = gu_t[:, intermediate:, :].permute(0, 2, 1).contiguous()
        module.torch_w2_proj = dp_t.permute(0, 2, 1).contiguous()
        return module

    def move_weights_to_device_impl(self):
        self.num_experts_per_device = self._get_num_experts_per_device(self.config, self.device)
        self.num_devices = self.device.get_num_devices()
        self.num_dispatch_devices = self.device.shape[1]

        self.tt_w1_proj = ttnn.to_device(self.tt_w1_proj, self.device)
        self.tt_w3_proj = ttnn.to_device(self.tt_w3_proj, self.device)
        self.tt_w2_proj = ttnn.to_device(self.tt_w2_proj, self.device)

        self.expert_mapping_tensors = ttnn.from_torch(
            torch.eye(self.num_devices, dtype=torch.int32)
            .repeat_interleave(self.num_experts_per_device, dim=0)
            .unsqueeze(0)
            .unsqueeze(0),
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.remap_topk_mask = ttnn.from_torch(
            torch.ones((1, self.num_dispatch_devices, 1, self.num_experts), dtype=torch.bfloat16),
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        hidden_tiles = self.hidden_size // ttnn.TILE_SIZE
        intermediate_tiles = self.intermediate_size // ttnn.TILE_SIZE

        self._gate_up_program_config = _make_fitted_sparse_matmul_program_config(
            device=self.device,
            out_features=int(self.intermediate_size),
            in0_block_w=min(4, hidden_tiles),
            per_core_M=1,
        )
        self._down_program_config = _make_fitted_sparse_matmul_program_config(
            device=self.device,
            out_features=int(self.hidden_size),
            in0_block_w=min(4, intermediate_tiles),
            per_core_M=1,
        )
        self._expert_compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )


class Qwen3OmniMoeTalkerTextMLPTTNN(TTNNModule):
    """Talker shared-expert SwiGLU on TTNN (TTNNLinearSilu + col-shard linears)."""

    @classmethod
    def from_torch(cls, torch_mlp, config=None):
        tt_module = cls()
        tt_module._fallback_torch_layer = torch_mlp
        tt_module.config = config
        tt_module.gate_proj = TTNNLinearSilu.from_torch(
            torch_mlp.gate_proj,
            linear_class=TTNNLinearIColShardedWRowSharded,
        )
        tt_module.up_proj = TTNNLinearIColShardedWRowSharded.from_torch(torch_mlp.up_proj)
        tt_module.down_proj = TTNNLinearIColShardedWRowSharded.from_torch(torch_mlp.down_proj)
        return tt_module

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x is None:
            raise ValueError("Qwen3OmniMoeTalkerTextMLPTTNN.forward: input x is None.")
        if not hasattr(x, "shape") or len(x.shape) < 2:
            raise ValueError(
                f"Qwen3OmniMoeTalkerTextMLPTTNN.forward: input must be at least 2D; got shape {getattr(x, 'shape', None)}."
            )
        if self.config is not None and hasattr(self.config, "hidden_size"):
            last_dim = int(x.shape[-1])
            if last_dim <= 0:
                raise ValueError(
                    f"Qwen3OmniMoeTalkerTextMLPTTNN.forward: input last dim must be positive; got {last_dim}."
                )
        x_gate = self.gate_proj(x)
        x_up = self.up_proj(x)
        a = _to_ttnn_raw(x_gate)
        b = _to_ttnn_raw(x_up)
        x = ttnn.mul(a, b)
        x = self.down_proj(x)
        return x


class TTNNQwen3TalkerMoE(TTNNQwen3OmniSparseMoE):
    """Qwen3-Omni talker sparse MoE; :meth:`from_torch` wraps ``Qwen3OmniMoeTalkerTextSparseMoeBlock``."""

    @classmethod
    def from_torch(cls, talker_block):
        qwen_config = getattr(talker_block.shared_expert, "config", None)

        class _Cfg:
            pass

        cfg = _Cfg()
        if qwen_config is not None:
            cfg.hidden_size = qwen_config.hidden_size
            cfg.moe_intermediate_size = qwen_config.moe_intermediate_size
            cfg.num_experts_per_tok = qwen_config.num_experts_per_tok
            cfg.n_routed_experts = qwen_config.num_experts
            cfg.norm_topk_prob = getattr(qwen_config, "norm_topk_prob", False)
            cfg.hidden_act = getattr(qwen_config, "hidden_act", "silu")
        else:
            cfg.hidden_size = talker_block.gate.weight.shape[1]
            ex = talker_block.experts
            if hasattr(ex, "gate_up_proj") and hasattr(ex, "intermediate_dim"):
                cfg.moe_intermediate_size = ex.intermediate_dim
                cfg.n_routed_experts = getattr(ex, "num_experts", ex.gate_up_proj.shape[0])
            elif hasattr(ex, "gate_up_proj"):
                cfg.moe_intermediate_size = ex.gate_up_proj.shape[1] // 2
                cfg.n_routed_experts = ex.gate_up_proj.shape[0]
            else:
                cfg.moe_intermediate_size = getattr(ex[0], "intermediate_size", None) or (
                    ex[0].gate_proj.weight.shape[0] if len(ex) else 0
                )
                cfg.n_routed_experts = talker_block.gate.weight.shape[0]
            cfg.num_experts_per_tok = 8
            cfg.norm_topk_prob = False
            cfg.hidden_act = "silu"

        consolidated = _consolidate_talker_experts_from_module_list(talker_block.experts, cfg)
        consolidated.intermediate_dim = cfg.moe_intermediate_size

        class _GateAdapter:
            pass

        gate_adapter = _GateAdapter()
        gate_adapter.weight = talker_block.gate.weight
        gate_adapter.hidden_dim = cfg.hidden_size
        gate_adapter.num_experts = cfg.n_routed_experts
        gate_adapter.top_k = cfg.num_experts_per_tok
        gate_adapter.norm_topk_prob = cfg.norm_topk_prob

        class _TalkerMoEAdapter:
            pass

        adapter = _TalkerMoEAdapter()
        adapter.gate = gate_adapter
        adapter.experts = consolidated
        adapter.shared_expert = talker_block.shared_expert
        adapter.shared_expert_gate = talker_block.shared_expert_gate

        module = super().from_torch(adapter)
        module._fallback_torch_layer = talker_block
        module.experts = Qwen3OmniMoeTalkerTextExpertsTTNN.from_torch(consolidated, cfg)
        return module


@trace_enabled
class TTNNQwenOmniDistributedRMSNorm(TTNNDistributedRMSNorm):
    """Distributed RMSNorm for Qwen3-Omni on mesh: gather fallback when tile sharding does not align with mesh width, plus HF API compatibility."""

    @property
    def weight(self):
        tl = self.torch_layer
        if tl is not None and hasattr(tl, "weight"):
            return tl.weight
        raise AttributeError(f"{type(self).__name__} has no attribute 'weight'")

    @property
    def variance_epsilon(self):
        tl = self.torch_layer
        if tl is None:
            raise AttributeError(f"{type(self).__name__} has no attribute 'variance_epsilon'")
        if hasattr(tl, "variance_epsilon"):
            return tl.variance_epsilon
        if hasattr(tl, "eps"):
            return tl.eps
        raise AttributeError(f"{type(self).__name__} has no attribute 'variance_epsilon'")

    @property
    def _is_distributed(self):
        return self.device is not None and self.device.get_num_devices() > 1

    def set_output_tensors_config_impl(self, output_tensors):
        """Col-sharded activations, or replicated full width when using gather + local ``ttnn.rms_norm``."""

        def set_gather_output_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None and self.device is not None:
                e.set_distributed_tensor_config(
                    DistributedTensorConfig(
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                        mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                    )
                )
            return e

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if self._is_distributed and self.device is not None:
                    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
                    mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=-1)

                    def logical_shape_for_col_sharded(shape):
                        shape_list = list(shape)
                        num_devices = self.device.get_num_devices()
                        shape_list[-1] = shape_list[-1] * num_devices
                        return tuple(shape_list)

                    e.set_distributed_tensor_config(
                        DistributedTensorConfig(
                            mesh_mapper=mesh_mapper,
                            mesh_composer=mesh_composer,
                            logical_shape_fn=logical_shape_for_col_sharded,
                        )
                    )
            return e

        if not self._is_distributed:
            return super().set_output_tensors_config_impl(output_tensors)
        if getattr(self, "_distributed_gather_rmsnorm", False) and self.device is not None:
            return tree_map(set_gather_output_config, output_tensors)
        return tree_map(set_col_sharded_config, output_tensors)

    def move_weights_to_device_impl(self):
        dim = int(self.torch_layer.weight.shape[0])
        padded_dim = ((dim + 31) // 32) * 32
        weight = self.torch_layer.weight
        if padded_dim != dim:
            weight = torch.nn.functional.pad(weight, (0, padded_dim - dim), value=1.0)
        w_bf16 = weight.to(torch.bfloat16)

        self._distributed_gather_rmsnorm = False
        self.tt_weight_gather = None
        self._gather_full_dim = padded_dim

        if self.device is None or self.device.get_num_devices() <= 1:
            relayout = w_bf16.view(1, 1, padded_dim // 32, 32)
            self.weight_distributed = ttnn.as_tensor(relayout, layout=ttnn.ROW_MAJOR_LAYOUT)
            self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)
            return

        mesh_shape = list(self.device.shape)
        ncol = int(mesh_shape[-1])
        ntiles = padded_dim // 32

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        if ntiles % ncol != 0:
            self._distributed_gather_rmsnorm = True
            rep = ttnn.ReplicateTensorToMesh(self.device)
            w_row = w_bf16.reshape(1, -1)
            self.tt_weight_gather = ttnn.from_torch(
                w_row,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=rep,
            )
            self.weight_distributed = None
            return

        relayout = w_bf16.view(1, 1, ntiles, 32)
        mesh_mapper = ttnn.ShardTensor2dMesh(self.device, dims=(None, 2), mesh_shape=mesh_shape)
        self.weight_distributed = ttnn.as_tensor(
            relayout,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)

    def _forward_distributed_gather_rms(self, inp: ttnn.Tensor, original_shape: tuple) -> ttnn.Tensor:
        emb = int(self._gather_full_dim)
        n_dev = int(self.device.get_num_devices())
        if inp.layout != ttnn.TILE_LAYOUT:
            inp = ttnn.to_layout(inp, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        wloc = int(inp.shape[-1])
        if wloc * n_dev == emb:
            inp = ttnn.all_gather(
                inp,
                dim=-1,
                num_links=1,
                topology=ttnn.Topology.Linear,
            )
        elif wloc != emb:
            raise RuntimeError(
                f"TTNNQwenOmniDistributedRMSNorm gather path: need col-shard {emb}/{n_dev} or full width {emb}, got last dim {wloc}"
            )

        rank = len(original_shape)
        if rank == 2:
            inp = ttnn.unsqueeze(ttnn.unsqueeze(inp, 0), 0)
        elif rank == 3:
            inp = ttnn.unsqueeze(inp, 1)
        elif rank != 4:
            raise RuntimeError(
                f"TTNNQwenOmniDistributedRMSNorm: gather path expected rank 2–4 activations, got rank {rank}"
            )

        eps = getattr(self.torch_layer, "variance_epsilon", getattr(self.torch_layer, "eps", 1e-6))
        tt_out = ttnn.rms_norm(
            inp,
            weight=self.tt_weight_gather,
            epsilon=eps,
            compute_kernel_config=self.compute_kernel_config,
        )

        if rank == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
        elif rank == 2 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [int(tt_out.shape[2]), int(tt_out.shape[3])])
        return tt_out

    @run_on_devices(DeviceArch.T3K)
    def forward(self, inp):
        original_shape = tuple(int(d) for d in inp.shape)
        if self._is_distributed and getattr(self, "_distributed_gather_rmsnorm", False):
            return self._forward_distributed_gather_rms(inp, original_shape)
        if len(original_shape) == 3:
            inp = ttnn.unsqueeze(inp, 1)
        if inp.layout != ttnn.TILE_LAYOUT:
            inp = ttnn.to_layout(inp, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_stats = ttnn.rms_norm_pre_all_gather(inp, dtype=ttnn.bfloat16)
        tt_stats = ttnn.all_gather(
            tt_stats,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Ring,
        )
        eps = getattr(self.torch_layer, "variance_epsilon", getattr(self.torch_layer, "eps", 1e-6))
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=eps,
            weight=self.weight_distributed,
        )
        tt_stats.deallocate(True)

        if len(original_shape) == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])

        return tt_out


__all__ = [
    "Qwen3OmniMoeTalkerTextMLPTTNN",
    "TTNNBailingMoEAttentionHostCachePositionRoPE",
    "TTNNConv1d",
    "TTNNConv3d",
    "TTNNConvTranspose1d",
    "TTNNConvTranspose2dNHWC",
    "TTNNQwen3Attention",
    "TTNNQwen3OmniAttention",
    "TTNNQwen3OmniMoeAudioEncoderConvOutLinear",
    "TTNNQwen3OmniMoeCausalConvNet",
    "TTNNQwen3OmniMoeCausalTransConvNet",
    "TTNNQwen3OmniMoeCode2WavAttention",
    "TTNNQwen3OmniMoeCode2WavDecoderResidualUnit",
    "TTNNQwen3OmniMoeConvNeXtBlock",
    "TTNNQwen3OmniMoeRotaryEmbedding",
    "TTNNQwen3OmniMoeThinkerTextRotaryEmbedding",
    "TTNNQwen3OmniMoeTalkerRotaryEmbedding",
    "TTNNQwen3OmniMoeVisionRotaryEmbedding",
    "TTNNQwen3OmniSparseMoE",
    "TTNNQwen3OmniTalkerResizeMLP",
    "TTNNQwen3OmniThinkerMoE",
    "TTNNQwen3OmniVisionMLP",
    "TTNNQwen3TalkerMoE",
    "TTNNQwen3VLMoeVisionAttention",
    "TTNNQwenAudioAttention",
    "TTNNQwenLayerNorm",
    "TTNNQwenOmniConv2dNHWC",
    "TTNNQwenOmniDistributedRMSNorm",
    "TTNNQwenOmniIColShardedWAllReduced",
    "TTNNQwenOmniLinear",
    "TTNNQwenOmniMoERouterDecode",
    "TTNNQwenOmniRotaryPositionEmbedding",
    "TTNNQwenOmniThinkerLmHead",
    "TTNNSnakeBeta",
    "replace_code_predictor_lm_head_with_ttnn",
    "replace_talker_codec_head_with_ttnn",
    "replace_thinker_lm_head_with_ttnn",
    "TTNNQwen3OmniMoeCodecPredictorEmbedding",
    "DistributedConfig",
    "DistributedTensorConfig",
    "QwenOmniDistributedConfig",
    "QwenOmniReplicatedMeshTensorConfig",
    "distributed_config_col_sharded_last_dim",
    "ensure_qwen_omni_normalrun_to_torch_slice",
    "QwenOmniDeviceInit",
    "qwen_omni_maybe_slice_replicated_mesh_compose",
    "qwen_omni_replicated_concat_dim0_tensor_config",
]
