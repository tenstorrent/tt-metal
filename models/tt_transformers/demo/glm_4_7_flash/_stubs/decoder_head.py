# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN lm_head for GLM-4.7-Flash (TP-sharded).

A single linear projection from hidden_size (2048) to vocab_size (154880).
Column-parallel: shard the vocab (output) dim across TP chips, then
all_gather to reassemble the full logits.
"""
from __future__ import annotations

import os

import torch

import ttnn


class DecoderHead:
    def __init__(self, device, torch_module) -> None:
        self.device = device

        sd = torch_module.state_dict()
        W = sd["weight"].float().t().contiguous()  # [2048, 154880]

        try:
            _is_mesh = isinstance(device, ttnn.MeshDevice)
        except AttributeError:
            _is_mesh = False
        _mesh_shape = list(device.shape) if _is_mesh else [1, 1]
        shard = bool(os.environ.get("TT_HW_PLANNER_SHARD_RUN")) and _is_mesh
        TP = _mesh_shape[-1] if shard else 1
        shard = shard and TP > 1
        self._shard = shard
        self._TP = TP
        self._tp_axis = len(_mesh_shape) - 1
        self._mesh_shape = _mesh_shape

        # Column-parallel: shard the vocab (output) dimension of the transposed
        # weight, then all_gather in TILE layout to reassemble.
        if shard:
            self._w = ttnn.from_torch(
                W,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=_mesh_shape, dims=(None, 1)),
            )
        else:
            kw = dict(dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
            if _is_mesh:
                kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
            self._w = ttnn.from_torch(W, **kw)

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, x, **kwargs):
        if isinstance(x, torch.Tensor):
            kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            try:
                if isinstance(self.device, ttnn.MeshDevice):
                    kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
            except AttributeError:
                pass
            x = ttnn.from_torch(x.to(torch.bfloat16), **kw)

        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        out = ttnn.matmul(x, self._w)

        if self._shard:
            out = ttnn.all_gather(out, dim=-1, num_links=1, topology=ttnn.Topology.Linear)

        return out


def build(device, torch_module=None):
    return DecoderHead.build(device, torch_module)


def decoder_head(device, torch_module=None):
    return DecoderHead.build(device, torch_module)
