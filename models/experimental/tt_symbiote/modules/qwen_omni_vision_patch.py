# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni vision patch stem: HF ``Conv3d`` patch embed and ``Qwen3OmniMoeVisionPatchMerger`` on TTNN.

``Qwen3OmniMoeVisionPatchEmbed`` uses a strided ``Conv3d`` whose kernel equals the patch volume; it is
numerically equivalent to ``Linear(in_channels * T * H * W, embed_dim)`` on flattened patches.
"""

from __future__ import annotations

import torch
from torch import nn
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import tree_map
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinear,
    TTNNLinearIColShardedWAllReduced,
    TTNNLinearIReplicatedWColSharded,
)
from models.experimental.tt_symbiote.modules.normalization import TTNNQwenLayerNorm


def _replicate_mapper(device):
    if device is None or device.get_num_devices() <= 1:
        return None
    return ttnn.ReplicateTensorToMesh(device)


def _ensure_ttnn(x, device, *, mesh_mapper=None):
    if isinstance(x, ttnn.Tensor):
        return x
    if isinstance(x, TorchTTNNTensor):
        if x.ttnn_tensor is not None:
            return x.ttnn_tensor
        if x.elem is not None:
            return ttnn.from_torch(
                x.elem.contiguous().to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=mesh_mapper,
            )
    if isinstance(x, torch.Tensor):
        return ttnn.from_torch(
            x.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
        )
    raise TypeError(f"_ensure_ttnn: unsupported type {type(x)}")


class TTNNQwen3OmniVisionPatchEmbed(TTNNModule):
    """TTNN patch embedding: flattened patch volume → ``ttnn.linear`` (Conv3d-equivalent)."""

    @classmethod
    def from_torch(cls, pe):
        m = cls()
        m._fallback_torch_layer = pe
        m.patch_size = int(pe.patch_size)
        m.temporal_patch_size = int(pe.temporal_patch_size)
        m.in_channels = int(pe.in_channels)
        m.embed_dim = int(pe.embed_dim)
        m._flat_dim = m.in_channels * m.temporal_patch_size * m.patch_size * m.patch_size

        w = pe.proj.weight.data.clone().to(torch.bfloat16)
        b = pe.proj.bias.data.clone().to(torch.bfloat16) if pe.proj.bias is not None else None
        w_flat = w.reshape(m.embed_dim, m._flat_dim)
        lin = nn.Linear(m._flat_dim, m.embed_dim, bias=b is not None)
        lin.weight.data.copy_(w_flat)
        if b is not None:
            lin.bias.data.copy_(b)
        m.linear = TTNNLinear.from_torch(lin)
        return m

    def preprocess_weights_impl(self):
        self.linear.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.linear.move_weights_to_device()

    def deallocate_weights_impl(self):
        self.linear.deallocate_weights()

    def set_output_tensors_config_impl(self, output_tensors):
        """Match vision hidden width after mesh readback (same idea as ``TTNNQwen3OmniVisionMLP``)."""
        if self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        def _materialize_one_replica(e):
            if not isinstance(e, TorchTTNNTensor) or e.ttnn_tensor is None:
                return e
            t = e.ttnn_tensor
            n = int(t.shape[0])
            h = int(self.embed_dim)
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
        mapper = _replicate_mapper(self.device)
        x = _ensure_ttnn(hidden_states, self.device, mesh_mapper=mapper)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Match HF ``view(-1, in_ch, T, P, P)`` then flatten patch volume for ``ttnn.linear``.
        x = ttnn.reshape(
            x,
            (-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size),
        )
        x = ttnn.reshape(x, (-1, self._flat_dim))

        return self.linear(x)


class TTNNQwen3OmniVisionPatchMerger(TTNNModule):
    """TTNN ``Qwen3OmniMoeVisionPatchMerger``: LayerNorm + 2× Linear + GELU (TP pattern like vision MLP)."""

    @classmethod
    def from_torch(cls, merger):
        m = cls()
        m._fallback_torch_layer = merger
        m.use_postshuffle_norm = bool(merger.use_postshuffle_norm)
        m.merged_dim = int(merger.hidden_size)
        m.out_hidden_size = int(merger.mlp[2].out_features)

        m.ln_q = TTNNQwenLayerNorm.from_torch(merger.ln_q)
        m.lin1 = TTNNLinearIReplicatedWColSharded.from_torch(merger.mlp[0])
        m.lin2 = TTNNLinearIColShardedWAllReduced.from_torch(merger.mlp[2])
        return m

    def preprocess_weights_impl(self):
        if isinstance(self.ln_q, TTNNQwenLayerNorm):
            self.ln_q.preprocess_weights()
        self.lin1.preprocess_weights()
        self.lin2.preprocess_weights()

    def move_weights_to_device_impl(self):
        if isinstance(self.ln_q, TTNNQwenLayerNorm):
            self.ln_q.move_weights_to_device()
        self.lin1.move_weights_to_device()
        self.lin2.move_weights_to_device()

    def deallocate_weights_impl(self):
        if isinstance(self.ln_q, TTNNQwenLayerNorm):
            self.ln_q.deallocate_weights()
        self.lin1.deallocate_weights()
        self.lin2.deallocate_weights()

    def set_output_tensors_config_impl(self, output_tensors):
        if self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        def _materialize_one_replica(e):
            if not isinstance(e, TorchTTNNTensor) or e.ttnn_tensor is None:
                return e
            t = e.ttnn_tensor
            n = int(t.shape[0])
            h = int(self.out_hidden_size)
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
    def forward(self, hidden):
        mapper = _replicate_mapper(self.device)
        h = _ensure_ttnn(hidden, self.device, mesh_mapper=mapper)

        if self.use_postshuffle_norm:
            h = ttnn.reshape(h, (-1, self.merged_dim))

        if isinstance(self.ln_q, TTNNQwenLayerNorm):
            h = self.ln_q(h)
        else:
            th = ttnn.to_torch(h)
            th = self.ln_q(th)
            h = ttnn.from_torch(
                th.contiguous().to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=mapper,
            )

        h = _ensure_ttnn(h, self.device, mesh_mapper=mapper)
        if h.layout != ttnn.TILE_LAYOUT:
            h = ttnn.to_layout(h, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        h = ttnn.reshape(h, (-1, self.merged_dim))

        in_w = int(h.shape[-1])
        if in_w < self.merged_dim:
            h = ttnn.all_gather(
                h,
                dim=-1,
                cluster_axis=1,
                num_links=1,
                topology=ttnn.Topology.Linear,
            )
        elif in_w > self.merged_dim:
            rank = len(h.shape)
            starts = [0] * rank
            ends = [int(s) for s in h.shape]
            ends[-1] = self.merged_dim
            h = ttnn.slice(h, starts, ends)

        h = self.lin1(h)
        h = ttnn.gelu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = self.lin2(h)

        out_w = int(h.shape[-1])
        if out_w > int(self.out_hidden_size):
            rank = len(h.shape)
            starts = [0] * rank
            ends = [int(s) for s in h.shape]
            ends[-1] = int(self.out_hidden_size)
            h = ttnn.slice(h, starts, ends)

        return h
