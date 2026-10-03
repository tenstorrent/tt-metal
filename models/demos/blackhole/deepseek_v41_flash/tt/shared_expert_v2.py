# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSV41SharedExpertV2: drop-in for DSV41SharedExpert with explicit matmul program configs (see tests/test_shared_expert_v2.py).

Same maths: out = (silu(min(x@w0, 10)) * clamp(x@w1, -10, 10)) @ w2, bfp8 weights, fp32 accumulate and fp32 output.
"""

import math

import torch

import ttnn

TILE = 32


def _cfg1d(grid_xy, n_tiles, per_core_N, k_tiles, in0_block_w, fused_activation=None):
    cores = math.ceil(n_tiles / per_core_N)
    gx = grid_xy[0]
    gy = math.ceil(cores / gx)
    sw = per_core_N
    while sw > 4 or per_core_N % sw:
        sw -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(min(gx, cores), gy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=1,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


class DSV41SharedExpertV2:
    def __init__(self, mesh_device, w0, w1, w2, limit=10.0, dtype=ttnn.bfloat8_b, mode="fused1d", **kw):
        """w0 (gate), w1 (up): [1, 1, dim, inter]; w2 (down): [1, 1, inter, dim] host tensors ([in, out]).
        mode: 'dram' (DRAM-sharded weights, L1 width-sharded activations), 'split1d' (separate gate/up weights, 1D mcast
        configs), 'fused1d' (fused gate|up weight, 1D mcast configs)."""
        self.md, self.mode, self.kw = mesh_device, mode, kw
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        self.inter, self.dim = w0.shape[-1], w0.shape[-2]
        self.limit = float(limit)
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, kw.get("fid", "HiFi4")),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=kw.get("l1acc", False),
        )
        self.mid_dtype = kw.get("mid_dtype", ttnn.bfloat16)
        self.act_a = [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MINIMUM, self.limit),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        ]
        self.act_b = [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MINIMUM, self.limit),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MAXIMUM, -self.limit),
        ]
        K, N = self.dim, self.inter
        Kt, Nt = K // TILE, N // TILE

        def up(t, mem=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(
                t.contiguous(),
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=rep,
            )

        if mode == "dram":
            nb = mesh_device.dram_grid_size().x
            self.nb = nb
            dram_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})

            def dmem(k, n):
                pad = math.ceil(n / (TILE * nb)) * TILE * nb
                return ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(dram_crs, (k, pad // nb), ttnn.ShardOrientation.ROW_MAJOR),
                )

            self.w0, self.w1 = up(w0, dmem(K, N)), up(w1, dmem(K, N))
            self.w2 = up(w2, dmem(N, K))
            nc = kw.get("cores", 8)
            self.nc = nc
            grid = mesh_device.compute_with_storage_grid_size()
            crs = ttnn.num_cores_to_corerangeset(nc, grid, row_wise=True)
            sh = lambda width: ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs, (TILE, width // nc), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.m_in, self.m_mid, self.m_out = sh(K), sh(N), sh(K)
            pc = lambda k, n, bw: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=bw, per_core_M=1, per_core_N=math.ceil(n / TILE / nc), fused_activation=None
            )
            self.pc1 = pc(K, N, kw.get("bw1", Kt // nc))
            self.pc2 = pc(N, K, kw.get("bw2", Nt // nc))
        elif mode == "split1d":
            self.w0, self.w1, self.w2 = up(w0), up(w1), up(w2)
            self.pc1 = _cfg1d(kw.get("g1", (12, 8)), Nt, kw.get("pn1", 4), Kt, kw.get("bw1", 8))
            self.pc2 = _cfg1d(kw.get("g2", (12, 8)), K // TILE, kw.get("pn2", 4), Nt, kw.get("bw2", 4))
        elif mode == "fused1d":
            self.w01, self.w2 = up(torch.cat([w0, w1], dim=-1)), up(w2)
            self.pc1 = _cfg1d(kw.get("g1", (12, 8)), 2 * Nt, kw.get("pn1", 4), Kt, kw.get("bw1", 8))
            self.pc2 = _cfg1d(kw.get("g2", (12, 8)), K // TILE, kw.get("pn2", 4), Nt, kw.get("bw2", 4))
        else:
            raise ValueError(mode)

    def forward(self, h):
        """h [1, 1, T, dim] bf16 -> [1, 1, T, dim] fp32."""
        lin = lambda x, w, pc, mem=ttnn.DRAM_MEMORY_CONFIG, dt=None: ttnn.linear(
            x, w, dtype=dt or self.mid_dtype, compute_kernel_config=self.ckc, program_config=pc, memory_config=mem
        )
        mul = lambda g, u, mem=ttnn.DRAM_MEMORY_CONFIG: ttnn.multiply(
            g,
            u,
            input_tensor_a_activations=self.act_a,
            input_tensor_b_activations=self.act_b,
            dtype=self.mid_dtype,
            memory_config=mem,
        )
        if self.mode == "dram":
            hs = ttnn.to_memory_config(h, self.m_in)
            g = lin(hs, self.w0, self.pc1, self.m_mid)
            u = lin(hs, self.w1, self.pc1, self.m_mid)
            ttnn.deallocate(hs)
            act = mul(g, u, self.m_mid)
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            o = lin(act, self.w2, self.pc2, self.m_out, ttnn.float32)
            ttnn.deallocate(act)
            r = ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(o)
            return r
        if self.mode == "split1d":
            g, u = lin(h, self.w0, self.pc1), lin(h, self.w1, self.pc1)
        else:
            gu = lin(h, self.w01, self.pc1)
            g, u = gu[:, :, :, : self.inter], gu[:, :, :, self.inter :]
        act = mul(g, u)
        return lin(act, self.w2, self.pc2, dt=ttnn.float32)
