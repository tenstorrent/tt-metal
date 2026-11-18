# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 MoE decode block on TP32 experts (HY4_MOE_TP32=1, default OFF, JIT only; kernel tt/tp32_kernels, host class hy4_moe_tp32.Tp32Experts).

Same contract as ``Hy4MoEBlock.forward(tt_x_gate [1,1,T,HID] tile, tt_x_tokens, fold=None, ft_defer=False) -> [1,1,T,HID/8]`` (T tokens per mesh row, 4 rows).
Per layer: router per row (unchanged Hy4Gate / skinny router) -> route words [T x (8 ids, 8 bf16 score bits)] int32 -> row all_gather of (route, x)
-> TP32 kernel on all 4 T tokens (each chip: its 64 intermediate columns of ALL 256 experts; partial [4T, HID]) -> [shared-expert fold placed by a per-row 0/1 mask matmul]
-> reduce_scatter over the 8 columns (hidden) -> row-major -> reduce_scatter over the 4 rows (tokens) -> [1,1,T,HID/8].
No dispatch, no combine, no tail, no expert mapping.
"""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import Tp32Experts

HID = 6144
PACK_U16 = os.environ.get("HY4_TP32_PACK", "") == "u16"
RED_AG = os.environ.get("HY4_TP32_RED", "") == "ag"
ROUTE_LOCAL = (
    os.environ.get("HY4_TP32_ROUTE", "") == "local"
)  # router on the gathered x (every chip routes all Tt tokens, writes the route words itself): no pack, no route all-gather
_XP = {}
XPUSH = (
    os.environ.get("HY4_B47B_XPUSH", "0") == "1"
)  # B47b: x + route push instead of the row all-gathers (T in 2, 4, ROUTE_ROW path)
ROUTE16 = (
    os.environ.get("HY4_B30_ROUTE16", "0") == "1"
)  # B30: ROUTE_LOCAL at Tt <= 16 (T=16 skinny router: 16-row tiles, two tokens per merge core)
ROUTE_ROW = (
    os.environ.get("HY4_B29_ROUTE_ROW", "0") == "1"
)  # B29 (default OFF): per-mesh-row router (T <= 8) writes the packed route words itself (skinny select merge ROUTE mode): no router_select program, no pack ops; the row all-gather stays


class Hy4MoeTp32Block:
    def __init__(self, md, ccl, gate, tpe: Tp32Experts, T):
        """gate: Hy4Gate; tpe: Tp32Experts on the 4x8 mesh (chip rank = mesh_row * 8 + col); T = tokens per mesh row (T * 4 <= 8)."""
        self.md, self.ccl, self.gate, self.tpe, self.T = md, ccl, gate, tpe, T
        self.shared_fuse = None  # B19 (HY4_MOE_FUSE=1): hy4_skinny.SkinnyShared (1/4-scaled w2) run inside the tp24 program on all Tt tokens
        self.rows, self.cols = tuple(md.shape)
        assert T * self.rows <= (
            16 if type(tpe).__name__ == "Tp24Experts" else 8
        ), "TP32 kernel: v0 <= 8 tokens on the mesh, tp24 <= 16"
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
        if (
            PACK_U16
        ):  # row-major end to end: ids uint16 || bf16 score bits (bitcast) as uint16 [T,1,1,16] -> int32 words (same bytes as the tile path; experiment ab-2)
            idx = ttnn.to_memory_config(idx, ttnn.DRAM_MEMORY_CONFIG)
            w = ttnn.to_memory_config(w, ttnn.DRAM_MEMORY_CONFIG)
            r = ttnn.concat(
                [ttnn.reshape(idx, [T, 1, 1, 8]), ttnn.bitcast(ttnn.reshape(w, [T, 1, 1, 8]), ttnn.uint16)], dim=3
            )
            r = ttnn.typecast(r, ttnn.int32)
            return ttnn.reshape(r, [1, 1, 1, 16 * T])
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
    def forward_partial(self, tt_x_gate, tt_x_tokens=None, forced_routing=None, fold=None, defer_fold=False):
        """router + gathers + TP32 kernel + shared-expert fold -> the un-reduced [1,1,4T,HID] bf16 TILE partial of this chip (rows = the Tt tokens of the 4 mesh rows in row order)."""
        T, Tt = self.T, self.Tt
        route_all = None
        xp_used = None
        if (
            ROUTE_LOCAL
            and forced_routing is None
            and Tt <= (16 if ROUTE16 else 8)
            and hasattr(self.gate, "route_words")
        ):
            x_all = ttnn.to_layout(ttnn.reshape(self._ag_rows(tt_x_tokens, 0), [1, 1, Tt, HID]), ttnn.TILE_LAYOUT)
            self.last_routing = None
            route_all = self.gate.route_words(
                x_all
            )  # [1,1,1,16 Tt] int32 RM, tokens in mesh-row order (same bytes as pack + row all-gather)
        if route_all is None:
            route = None
            if ROUTE_ROW and forced_routing is None and hasattr(self.gate, "route_words"):
                route = self.gate.route_words(tt_x_gate)  # [1,1,1,16 T] int32 RM = the packed words of this mesh row
            if route is not None:
                self.last_routing = None
            else:
                w, idx = forced_routing if forced_routing is not None else self.gate.forward(tt_x_gate)
                self.last_routing = (w, idx)
                route = self._pack_route(w, idx)
            if (
                XPUSH
                and route is not None
                and self.shared_fuse is None
                and tt_x_tokens is not None
                and self.T in (2, 4)
                and tuple(self.md.shape)[0] == 4
            ):  # B47b: push the x rows + route words over the 4-row group, the tp24 kernels wait for the flags (no row all-gathers, no tilize)
                key = (id(self.md), self.T)
                if key not in _XP:
                    from models.demos.blackhole.hy4_preview.tt import hy4_xpush

                    _XP[key] = hy4_xpush.XPush(self.md, self.T)
                xp = _XP[key]
                xp(tt_x_tokens, route)
                ttnn.deallocate(route)
                xp_used = xp
            else:
                route_all = self._ag_rows(route, 3)  # [1,1,1,16 Tt] int32 RM, rows in mesh-row order
                ttnn.deallocate(route)
                x_all = ttnn.to_layout(
                    ttnn.reshape(self._ag_rows(tt_x_tokens, 0), [1, 1, Tt, HID]), ttnn.TILE_LAYOUT
                )  # tokens [T,1,1,HID] RM -> [Tt,1,1,HID] -> [1,1,Tt,HID]
        if self.shared_fuse is not None and fold is None:
            part = self.tpe(
                x_all, route_all, Tt, shared=self.shared_fuse
            )  # routed + shared (w2 scaled 1/4, every mesh row computes all Tt tokens: the row reduction restores 1x)
            ttnn.deallocate(x_all)
            ttnn.deallocate(route_all)
            return part
        if xp_used is not None:
            part = self.tpe(None, None, Tt, xp=xp_used)
        else:
            part = self.tpe(x_all, route_all, Tt)  # [1,1,Tt,HID] bf16 TILE partial
            ttnn.deallocate(x_all)
            ttnn.deallocate(route_all)
        if (
            fold is not None and not defer_fold
        ):  # defer_fold (HY4_BOUNDARY_FUSE=fold): the caller hands `fold` to the rows-reduce program
            fp = ttnn.matmul(
                self.mask, fold, compute_kernel_config=self.ckc
            )  # [1,1,Tt,T] x fold [1,1,T,HID] -> [1,1,Tt,HID]
            part2 = ttnn.add(part, fp)
            ttnn.deallocate(part)
            ttnn.deallocate(fp)
            part = part2
        return part

    def forward(self, tt_x_gate, tt_x_tokens=None, forced_routing=None, fold=None, ft_defer=False):
        part = self.forward_partial(tt_x_gate, tt_x_tokens, forced_routing, fold)
        y = self._rs(part, 3, 1, ttnn.Topology.Ring)  # [1,1,Tt,HID/8]: summed over the 8 columns
        ttnn.deallocate(part)
        if RED_AG:  # one gather + local sum instead of RS rows + AG rows (HY4_TP32_RED=ag, experiment ab-1)
            yg = self._ag_rows(y, 0)  # [4,1,Tt,HID/8]
            ttnn.deallocate(y)
            y3 = ttnn.sum(yg, dim=0, keepdim=True)
            ttnn.deallocate(yg)
        else:
            y2 = self._rs(y, 3, 0, ttnn.Topology.Linear)  # [1,1,Tt,HID/32]: summed over the 4 rows too
            ttnn.deallocate(y)
            y3 = self._ag_rows(y2, 3)  # [1,1,Tt,HID/8]: all 32 chips summed, every row holds all Tt tokens
            ttnn.deallocate(y2)
        z = ttnn.matmul(self.sel, y3, compute_kernel_config=self.ckc)  # [1,1,T,HID/8]: this row's own tokens
        ttnn.deallocate(y3)
        return z
