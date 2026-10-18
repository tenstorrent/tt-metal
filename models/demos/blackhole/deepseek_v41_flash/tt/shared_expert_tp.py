# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared expert split over the mesh COLUMNS (experimental, not wired into the model; see tests/test_shared_expert_tp.py).

The intermediate dim (2304) is cut in 8 slices of 288 (9 tiles), one per mesh column: gate|up weight column-sharded on the output dim
(per column [w0_c | w1_c], 576 wide), down weight row-sharded on the input dim. Every device of column c computes the PARTIAL
    out_c = (silu(min(x@w0_c, L)) * clamp(x@w1_c, -L, L)) @ w2_c          [1,1,T,dim]
and  sum_c out_c  ==  the replicated shared expert. The partial is meant to be added into the routed partial of the same device BEFORE the
reduce_scatter over mesh axis 1 of the MoE block (tt/moe_block.py), so no extra collective is needed.
Weights are bfp8 like the replicated version: per chip 1/8 of the bytes (4.7 MB instead of 37.6 MB per layer).
"""

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert_v2 import TILE, _cfg1d


class DSV41SharedExpertColTP:
    def __init__(
        self,
        mesh_device,
        w0,
        w1,
        w2,
        limit=10.0,
        dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat16,
        mid_dtype=ttnn.bfloat16,
        g1=(12, 8),
        g2=(12, 8),
        pn1=1,
        pn2=5,  # swept in tests/test_shared_expert_tp.py: bw1=32 bw2=9 pn2=5 = 33.1 us (bw1=8 bw2=3 pn2=2: 44.9 us)
        bw1=32,
        bw2=9,
        fid="HiFi4",
        cfg=True,
    ):
        """w0 (gate), w1 (up): [1, 1, dim, inter]; w2 (down): [1, 1, inter, dim] host tensors ([in, out])."""
        self.md = mesh_device
        rows, cols = tuple(mesh_device.shape)
        self.cols = cols
        self.dim, self.inter = w0.shape[-2], w0.shape[-1]
        assert self.inter % (cols * TILE) == 0
        self.ic = self.inter // cols  # 288
        self.limit = float(limit)
        self.out_dtype, self.mid_dtype = out_dtype, mid_dtype
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, fid),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.act_a = [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MINIMUM, self.limit),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        ]
        self.act_b = [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MINIMUM, self.limit),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MAXIMUM, -self.limit),
        ]
        ic = self.ic
        # per-column [w0_c | w1_c] blocks, concatenated over columns -> sharding dim 3 over the columns gives each column its block
        w01 = torch.cat(
            [torch.cat([w0[..., c * ic : (c + 1) * ic], w1[..., c * ic : (c + 1) * ic]], dim=-1) for c in range(cols)],
            dim=-1,
        )
        col_map = lambda d: ttnn.ShardTensor2dMesh(mesh_device, dims=(None, d), mesh_shape=(rows, cols))
        up = lambda t, d: ttnn.from_torch(
            t.contiguous(),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=col_map(d),
        )
        self.w01, self.w2 = up(w01, 3), up(w2, 2)
        Kt, ict = self.dim // TILE, ic // TILE
        if cfg:
            self.pc1 = _cfg1d(g1, 2 * ict, pn1, Kt, bw1)
            self.pc2 = _cfg1d(g2, self.dim // TILE, pn2, ict, bw2)
        else:
            self.pc1 = self.pc2 = None

    def forward(self, h):
        """h [1, 1, T, dim] bf16 -> partial [1, 1, T, dim] (out_dtype) of THIS column's slice of the intermediate dim."""
        kw = lambda pc, dt: dict(
            dtype=dt,
            compute_kernel_config=self.ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **({"program_config": pc} if pc else {}),
        )
        gu = ttnn.linear(h, self.w01, **kw(self.pc1, self.mid_dtype))
        g, u = gu[:, :, :, : self.ic], gu[:, :, :, self.ic :]
        act = ttnn.multiply(
            g,
            u,
            input_tensor_a_activations=self.act_a,
            input_tensor_b_activations=self.act_b,
            dtype=self.mid_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.linear(act, self.w2, **kw(self.pc2, self.out_dtype))
