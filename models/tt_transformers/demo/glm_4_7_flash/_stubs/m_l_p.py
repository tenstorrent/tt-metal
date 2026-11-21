# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN SwiGLU MLP for GLM-4.7-Flash (TP-sharded).

SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))

gate_proj / up_proj are column-parallel (shard output dim).
down_proj is row-parallel (shard input dim) + all_reduce.
"""
from __future__ import annotations

import os

import ttnn


class TtMLP:
    def __init__(self, device, torch_module) -> None:
        self.device = device
        sd = torch_module.state_dict()

        Wgate = sd["gate_proj.weight"].float().t().contiguous()  # [2048, 10240]
        Wup = sd["up_proj.weight"].float().t().contiguous()  # [2048, 10240]
        Wdown = sd["down_proj.weight"].float().t().contiguous()  # [10240, 2048]

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

        def rep(t):
            if _is_mesh:
                return ttnn.from_torch(
                    t,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                )
            return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

        def shd(t, dim):
            return ttnn.from_torch(
                t,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=_mesh_shape, dims=(None, dim)),
            )

        def rep_bf8(t, layout=ttnn.TILE_LAYOUT):
            if _is_mesh:
                return ttnn.from_torch(
                    t,
                    dtype=ttnn.bfloat8_b,
                    layout=layout,
                    device=device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                )
            return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=layout, device=device)

        def shd_bf8(t, dim, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat8_b,
                layout=layout,
                device=device,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=_mesh_shape, dims=(None, dim)),
            )

        if shard:
            self._w_gate = shd_bf8(Wgate, 1)  # column-parallel
            self._w_up = shd_bf8(Wup, 1)  # column-parallel
            self._w_down = shd_bf8(Wdown, 0)  # row-parallel
        else:
            self._w_gate = rep_bf8(Wgate)
            self._w_up = rep_bf8(Wup)
            self._w_down = rep_bf8(Wdown)

        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def _fp32(self, t):
        if isinstance(t, ttnn.Tensor):
            return ttnn.typecast(t, ttnn.float32) if t.dtype != ttnn.float32 else t
        kw = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        try:
            if isinstance(self.device, ttnn.MeshDevice):
                kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
        except AttributeError:
            pass
        return ttnn.from_torch(t.float(), **kw)

    def __call__(self, x, **kwargs):
        hs = self._fp32(x)
        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT)

        gate_out = ttnn.matmul(hs, self._w_gate, compute_kernel_config=self.ckc)
        up_out = ttnn.matmul(hs, self._w_up, compute_kernel_config=self.ckc)
        ttnn.deallocate(hs)

        gate_act = ttnn.silu(gate_out)
        ttnn.deallocate(gate_out)

        inter = ttnn.multiply(gate_act, up_out)
        ttnn.deallocate(gate_act)
        ttnn.deallocate(up_out)

        out = ttnn.matmul(inter, self._w_down, compute_kernel_config=self.ckc)
        ttnn.deallocate(inter)

        if self._shard:
            out = ttnn.all_reduce(out, cluster_axis=self._tp_axis, topology=ttnn.Topology.Linear)

        return ttnn.typecast(out, ttnn.bfloat16)


def build(device, torch_module=None):
    return TtMLP.build(device, torch_module)


def m_l_p(*args, **kwargs):
    return TtMLP.build(*args, **kwargs)
