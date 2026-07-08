# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `feed_forward_swi_g_l_u`
(meituan-longcat/LongCat-Video's `dit.blocks.*.ffn`, class `FeedForwardSwiGLU`
in the vendored `longcat_video/modules/blocks.py`):

    w1 = Linear(dim, hidden_dim, bias=False)
    w2 = Linear(hidden_dim, dim, bias=False)
    w3 = Linear(dim, hidden_dim, bias=False)
    forward(x):
        return w2(silu(w1(x)) * w3(x))

This component graduates DIRECTLY tensor-parallel (no single-device phase):
standard Megatron-style TP. `w1`/`w3` are column-parallel -- their OUTPUT
(hidden_dim) is split across the mesh, so the elementwise `silu(w1(x)) *
w3(x)` needs no communication as long as both share the same split. `w2` is
row-parallel -- its INPUT (hidden_dim) is split to match, so each device
computes only a partial sum over its hidden_dim shard; an all_reduce sums the
partials back into the full, replicated output. The input `x` arrives
replicated across the mesh (matches column-parallel's requirement), and the
gathered/replicated output must equal the single-device golden -- placement
changes, math does not.
"""

from __future__ import annotations

import ttnn


class TtFeedForwardSwiGLU:
    def __init__(self, mesh_device: ttnn.MeshDevice, torch_module) -> None:
        self.mesh_device = mesh_device
        self.dtype = ttnn.bfloat16

        state = torch_module.state_dict()

        def _col_parallel_weight(key):
            # nn.Linear.weight is [out_features, in_features]; ttnn.linear's
            # second operand wants [in_features, out_features]. Column-parallel
            # splits out_features (dim=-1 after transpose) across the mesh.
            w = state[key].transpose(0, 1).contiguous()
            return ttnn.from_torch(
                w,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            )

        def _row_parallel_weight(key):
            # Row-parallel splits in_features (dim=0 after transpose) to match
            # the column-parallel output shard that feeds it.
            w = state[key].transpose(0, 1).contiguous()
            return ttnn.from_torch(
                w,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )

        self.w1 = _col_parallel_weight("w1.weight")
        self.w3 = _col_parallel_weight("w3.weight")
        self.w2 = _row_parallel_weight("w2.weight")

        # HiFi4 + fp32 dest-accumulation: this SwiGLU runs once per DiT block (x48), so
        # bf16 matmul rounding compounds; HiFi4 keeps the residual stream faithful.
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.silu(ttnn.linear(x, self.w1, compute_kernel_config=self.ckc))
        up = ttnn.linear(x, self.w3, compute_kernel_config=self.ckc)
        hidden = ttnn.multiply(gate, up)
        out = ttnn.linear(hidden, self.w2, compute_kernel_config=self.ckc)
        return ttnn.all_reduce(out, topology=ttnn.Topology.Linear)


def build(mesh_device: ttnn.MeshDevice, torch_module) -> TtFeedForwardSwiGLU:
    return TtFeedForwardSwiGLU(mesh_device, torch_module)
