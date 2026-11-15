# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 MoE decode block on TP32 experts (HY4_MOE_TP32=1, default OFF, JIT only; kernel tt/tp32_kernels, host class hy4_moe_tp32.Tp32Experts).

Same contract as ``Hy4MoEBlock.forward(tt_x_gate [1,1,T,HID] tile, tt_x_tokens, fold=None, ft_defer=False) -> [1,1,T,HID/8]`` (T tokens per mesh row, 4 rows).
Per layer: router per row (unchanged Hy4Gate / skinny router) -> route words [T x (8 ids, 8 bf16 score bits)] int32 -> row all_gather of (route, x)
-> TP32 kernel on all 4 T tokens (each chip: its 64 intermediate columns of ALL 256 experts; partial [4T, HID]) -> [shared-expert fold placed by a per-row 0/1 mask matmul]
-> reduce_scatter over the 8 columns (hidden) -> row-major -> reduce_scatter over the 4 rows (tokens) -> [1,1,T,HID/8].
No dispatch, no combine, no tail, no expert mapping.
"""
import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import Tp32Experts

HID = 6144


class Hy4MoeTp32Block:
    def __init__(self, md, ccl, gate, tpe: Tp32Experts, T):
        """gate: Hy4Gate; tpe: Tp32Experts on the 4x8 mesh (chip rank = mesh_row * 8 + col); T = tokens per mesh row (T * 4 <= 8)."""
        self.md, self.ccl, self.gate, self.tpe, self.T = md, ccl, gate, tpe, T
        self.rows, self.cols = tuple(md.shape)
        assert T * self.rows <= 8, "v0 kernel: <= 8 tokens on the mesh"
        # mask[r] = [Tt, T] with ones at (r*T + i, i): places a row's own [T, HID] fold into the Tt-token layout (zeros for the other rows)
        Tt = T * self.rows
        m = torch.zeros(self.rows, 1, Tt, T)
        for r in range(self.rows):
            for i in range(T):
                m[r, 0, r * T + i, i] = 1.0
        self.mask = ttnn.from_torch(
            m,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols)),
        )
        self.Tt = Tt
        # sel[r] = [T, Tt]: picks the T tokens of mesh row r out of the Tt gathered tokens (final stage: all-reduce over rows, then select)
        m2 = torch.zeros(self.rows, 1, T, Tt)
        for r in range(self.rows):
            for i in range(T):
                m2[r, 0, i, r * T + i] = 1.0
        self.sel = ttnn.from_torch(
            m2,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols)),
        )
        self.ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    # ------------------------------------------------------------------ pieces
    def _pack_route(self, w, idx):
        """w bf16 RM [T,1,1,8], idx uint16 RM [T,1,1,8] -> int32 RM [1,1,1,16 T]: per token 8 ids then 8 words whose low 16 bits are the bf16 score bits."""
        T = self.T
        # the skinny router writes the dispatch formats (L1 sharded): bring both to interleaved DRAM first
        idx = ttnn.to_memory_config(idx, ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.to_memory_config(w, ttnn.DRAM_MEMORY_CONFIG)

        ids = ttnn.typecast(ttnn.to_layout(ttnn.reshape(idx, [1, 1, T, 8]), ttnn.TILE_LAYOUT), ttnn.int32)
        sc = ttnn.typecast(ttnn.to_layout(ttnn.reshape(w, [1, 1, T, 8]), ttnn.TILE_LAYOUT), ttnn.float32)
        sb = ttnn.bitwise_right_shift(ttnn.bitcast(sc, ttnn.int32), 16)
        ttnn.deallocate(sc)
        r = ttnn.concat([ids, sb], dim=3)
        ttnn.deallocate(ids)
        ttnn.deallocate(sb)
        r = ttnn.to_layout(r, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(r, [1, 1, 1, 16 * T])

    def _ag_rows(self, t, dim):
        return ttnn.experimental.all_gather_async(
            t,
            dim=dim,
            cluster_axis=0,
            mesh_device=self.md,
            topology=ttnn.Topology.Linear,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            num_links=self.ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
        )

    def _rs(self, t, dim, axis, topology):
        return ttnn.experimental.reduce_scatter_minimal_async(
            t,
            dim=dim,
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            num_links=self.ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            cluster_axis=axis,
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
        )

    # ------------------------------------------------------------------ forward
    def forward(self, tt_x_gate, tt_x_tokens=None, forced_routing=None, fold=None, ft_defer=False):
        T, Tt = self.T, self.Tt
        w, idx = forced_routing if forced_routing is not None else self.gate.forward(tt_x_gate)
        self.last_routing = (w, idx)
        route = self._pack_route(w, idx)
        route_all = self._ag_rows(route, 3)  # [1,1,1,16 Tt] int32 RM, rows in mesh-row order
        ttnn.deallocate(route)
        x_all = ttnn.to_layout(
            ttnn.reshape(self._ag_rows(tt_x_tokens, 0), [1, 1, Tt, HID]), ttnn.TILE_LAYOUT
        )  # tokens [T,1,1,HID] RM -> [Tt,1,1,HID] -> [1,1,Tt,HID]
        part = self.tpe(x_all, route_all, Tt)  # [1,1,Tt,HID] bf16 TILE partial
        ttnn.deallocate(x_all)
        ttnn.deallocate(route_all)
        if fold is not None:
            fp = ttnn.matmul(
                self.mask, fold, compute_kernel_config=self.ckc
            )  # [1,1,Tt,T] x fold [1,1,T,HID] -> [1,1,Tt,HID]
            part2 = ttnn.add(part, fp)
            ttnn.deallocate(part)
            ttnn.deallocate(fp)
            part = part2
        y = self._rs(part, 3, 1, ttnn.Topology.Ring)  # [1,1,Tt,HID/8]: summed over the 8 columns
        ttnn.deallocate(part)
        y2 = self._rs(y, 3, 0, ttnn.Topology.Linear)  # [1,1,Tt,HID/32]: summed over the 4 rows too
        ttnn.deallocate(y)
        y3 = self._ag_rows(y2, 3)  # [1,1,Tt,HID/8]: all 32 chips summed, every row holds all Tt tokens
        ttnn.deallocate(y2)
        z = ttnn.matmul(self.sel, y3, compute_kernel_config=self.ckc)  # [1,1,T,HID/8]: this row's own tokens
        ttnn.deallocate(y3)
        return z
