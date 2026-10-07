# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode-sized MoE for one layer (trace friendly): exact router -> dead-slot routing spread -> dispatch -> unified bf16-activation experts -> combine ->
post_combine_reduce -> all-reduce over mesh columns.

Interface (for the decode agent):
    dec = MiMoMoEDecode(mesh, gate, moe)                 # gate: MiMoGate, moe: MiMoPrefillMoE (bf16 env switches set, see below)
    mask = dec.make_mask_host(live_counts)               # host tensor [4 mesh rows x 64 slots], 1.0 for live slots (first live_counts[r] slots of row r)
    mask_dev = dec.alloc_mask(mask)                      # persistent device tensor; update in place per step: dec.update_mask(mask_dev, mask)
    out = dec.forward(x, mask_dev)                       # x [1,1,32,4096] bf16 TILE per device (32 slots of this mesh row, dead slots zero) -> [1,1,32,4096]
All device-side, no host reads: valid inside ttnn.begin_trace_capture / execute_trace. Dead slots (hidden zero) would all route to the same experts (the
bias top-8) and overflow a chip's dispatch buffer, so their routing is replaced by a round-robin spread (spread_pad_routing) with weight 1/8.
Requires MOE_UNI_BF16_X/INTERMED/OUT=1 in the environment of the process that builds the op (and the patched _ttnncpp.so) for bf16 activations.
"""
import os

import torch

import ttnn
from models.demos.blackhole.mimo_v26.tt.moe_prefill import (  # noqa: F401
    COLS,
    HIDDEN,
    ROWS,
    TOPK,
    free,
    hold_begin,
    hold_end,
    spread_pad_routing,
)

SLOTS = 64  # masked_bincount needs tokens % 64 == 0: 32 decode slots + 32 pad rows per mesh row


def dead_routing(dead):
    """Routing (idx [1,SLOTS,TOPK] int, w [1,SLOTS,TOPK]) given to dead/pad slots. "spread32" (default of the decode path): round-robin over ONE expert per chip (ids col*32+row*8, 32 experts =
    1/8 of the expert weights), weight 0: every chip still gets dead-slot tokens every layer, so fabric traffic stays balanced. "sentinel" (expert id 256, never dispatched, cheapest) HANGS deep traced
    decode (combine multicast barrier, ~per-replay probability at 48 layers: LOG moe track); "spread": all 256 experts (touches every expert's weights every call).
    """
    sidx, sw = spread_pad_routing(1, SLOTS)
    if dead == "sentinel":
        return torch.full_like(sidx, 256), torch.zeros_like(sw)
    if dead == "spread32":
        ids = torch.tensor([c * 32 + r * 8 for c in range(8) for r in range(4)])
        slot = torch.arange(SLOTS).unsqueeze(1) * TOPK + torch.arange(TOPK).unsqueeze(0)
        return ids[slot % 32].reshape(1, SLOTS, TOPK), torch.zeros_like(sw)
    return sidx, sw


class MiMoMoEDecode:
    def __init__(self, mesh, gate, moe, all_reduce=True, num_links=None, router="exact", dead="spread32"):
        """router: \"exact\" (original, 64-row ttnn path) or \"fast\" (router_select kernel on the 32 live slots; identical expert sets on all 47 layers)."""
        self.router = router
        self.dead = dead  # "sentinel": dead slots get expert id 256 (= num experts, dispatch table -1): never dispatched, so cost scales with live tokens; "spread": old round-robin
        self._zpad = {}
        self.mesh, self.gate, self.moe, self.do_ar = mesh, gate, moe, all_reduce
        rep = ttnn.ReplicateTensorToMesh(mesh)
        # spread routing for the 64 slots of one mesh row, expert ids exact in bf16 (<= 256), weights 1/8
        sidx, sw = dead_routing(dead)
        self.sp_idx = ttnn.from_torch(
            sidx.reshape(1, 1, SLOTS, TOPK).bfloat16(),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )
        self.sp_w = ttnn.from_torch(
            sw.reshape(1, 1, SLOTS, TOPK).bfloat16(),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )
        self.zw = ttnn.from_torch(
            torch.zeros(32, 1, 1, TOPK).bfloat16(),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep,
        )
        self.zi = ttnn.from_torch(
            torch.zeros(32, 1, 1, TOPK).to(torch.int32),
            device=mesh,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep,
        )
        self.ones = ttnn.from_torch(
            torch.ones(1, 1, SLOTS, 1).bfloat16(),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )

    # -- mask helpers: [ROWS, 1, SLOTS, 1] sharded over mesh rows, replicated over columns
    def make_mask_host(self, live_counts):
        m = torch.zeros(ROWS, 1, SLOTS, 1)
        for r, n in enumerate(live_counts):
            m[r, 0, :n, 0] = 1.0
        return m.bfloat16()

    def _mapper(self):
        return ttnn.ShardTensor2dMesh(self.mesh, dims=(0, None), mesh_shape=(ROWS, COLS))

    def alloc_mask(self, mask_host):
        return ttnn.from_torch(
            mask_host, device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._mapper()
        )

    # -- DEVICE-RESIDENT mask derivation (no host copy per step): the decode step keeps dead/live flags (EOS, position cap) or per-row live counts as device tensors and derives the
    #    [1,1,64,1] bf16 tile mask the MoE forward consumes with device ops inside the trace. The host API above stays as a fallback.
    def mask_from_flags(self, flags):
        """flags: device tensor whose LAST dim is the slot axis n <= 32 (any shape like [1,1,1,n] / [n] / [1,n]; per mesh row, replicated over columns; dtype uint32/int32/uint16/bf16/float32;
        nonzero = live). Returns the mask [1,1,64,1] bf16 TILE (rows >= n are 0 = dead)."""
        n = flags.shape[-1]
        f = ttnn.reshape(flags, [1, 1, 1, n])
        if f.layout != ttnn.TILE_LAYOUT:
            f = ttnn.to_layout(f, ttnn.TILE_LAYOUT)
        if f.dtype != ttnn.bfloat16:
            f = ttnn.typecast(f, ttnn.bfloat16)
        f = ttnn.gt(f, 0.0)  # any nonzero -> 1.0
        col = ttnn.transpose(f, -2, -1)  # [1,1,n,1]
        return ttnn.pad(col, [(0, 0), (0, 0), (0, SLOTS - n), (0, 0)], 0.0)

    def mask_from_count(self, count):
        """count: device tensor [1,1,1,1] (per mesh row live-slot count, uint32/int32/bf16): mask[i] = 1 for i < count. Returns [1,1,64,1] bf16 TILE."""
        if not hasattr(self, "_arange"):
            self._arange = ttnn.from_torch(
                torch.arange(SLOTS).reshape(1, 1, SLOTS, 1).bfloat16(),
                device=self.mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        c = count
        if c.layout != ttnn.TILE_LAYOUT:
            c = ttnn.to_layout(c, ttnn.TILE_LAYOUT)
        if c.dtype != ttnn.bfloat16:
            c = ttnn.typecast(c, ttnn.bfloat16)
        return ttnn.lt(self._arange, c)

    def update_mask(self, mask_dev, mask_host):
        host = ttnn.from_torch(mask_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._mapper())
        ttnn.copy_host_to_device_tensor(host, mask_dev)

    def _spread(self, wts, idx, mask):
        T = SLOTS
        i = ttnn.to_layout(ttnn.reshape(idx, [1, 1, T, TOPK]), ttnn.TILE_LAYOUT)
        i = ttnn.typecast(i, ttnn.bfloat16)
        w = ttnn.to_layout(ttnn.reshape(wts, [1, 1, T, TOPK]), ttnn.TILE_LAYOUT)
        dead = ttnn.subtract(self.ones, mask)
        i2 = ttnn.add(ttnn.multiply(i, mask), ttnn.multiply(self.sp_idx, dead))
        w2 = ttnn.add(ttnn.multiply(w, mask), ttnn.multiply(self.sp_w, dead))
        i2 = ttnn.reshape(ttnn.to_layout(ttnn.typecast(i2, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT), [T, 1, 1, TOPK])
        w2 = ttnn.reshape(ttnn.to_layout(w2, ttnn.ROW_MAJOR_LAYOUT), [T, 1, 1, TOPK])
        return w2, i2

    def _column_reduce(self, part, B):
        """Sum the 8 per-column partials of the first B rows. MIMO_MOE_AR=ag_sum (default; race agent, commit feabf363015): slice to the B live rows, all_gather over the column axis and
        sum locally (ttnn.all_reduce = reduce_scatter + all_gather left one 32-column tile of one device stale in ~1e-3 of the calls behind dispatch/combine). MIMO_MOE_AR=rs: old all_reduce.
        """
        mode = os.environ.get("MIMO_MOE_AR", "ag_sum")
        if (
            mode == "rs64"
        ):  # the pre-merge behaviour: all_reduce of the full 64-row partial, then slice (hang repro for the sentinel path)
            full = ttnn.all_reduce(part, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            o = ttnn.slice(full, [0, 0, 0, 0], [1, 1, B, HIDDEN])
            free(full)
            return o
        x = ttnn.slice(part, [0, 0, 0, 0], [1, 1, B, HIDDEN])
        if mode == "rs":
            return ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = ttnn.all_gather(x, dim=0, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        free(x)
        o = ttnn.sum(g, dim=0, keepdim=True)
        free(g)
        return o

    def forward(self, x, mask, upto=99):
        """upto (stage-timing hook): 0 = pad+router+spread, 1..4 = moe stages (counts, dispatch, experts, combine), 5 = full moe without all-reduce."""
        B = x.shape[2]
        xp = ttnn.pad(x, [(0, 0), (0, 0), (0, SLOTS - B), (0, 0)], 0.0)
        if self.router == "fast":
            assert B <= 32
            w32, i32 = self.gate.route_v(x, "sel_mm1x8")  # router_select handles any T <= 32 rows
            npad = SLOTS - B
            if npad not in self._zpad:
                rep = ttnn.ReplicateTensorToMesh(self.mesh)
                self._zpad[npad] = (
                    ttnn.from_torch(
                        torch.zeros(npad, 1, 1, TOPK).bfloat16(),
                        device=self.mesh,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=rep,
                    ),
                    ttnn.from_torch(
                        torch.zeros(npad, 1, 1, TOPK).to(torch.int32),
                        device=self.mesh,
                        dtype=ttnn.uint16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=rep,
                    ),
                )
            zw, zi = self._zpad[npad]
            wts = ttnn.concat([w32, zw], dim=0)  # rows B..63: replaced by the dead-slot routing through the mask
            idx = ttnn.concat([i32, zi], dim=0)
        else:
            wts, idx = self.gate.route(xp, "exact")
        wts2, idx2 = self._spread(wts, idx, mask)
        if upto == 0:
            return wts2, idx2
        x_rm = ttnn.reshape(ttnn.to_layout(xp, ttnn.ROW_MAJOR_LAYOUT), [1, SLOTS, HIDDEN])
        part = self.moe.forward(x_rm, wts2, idx2, upto=min(upto, 99) if upto <= 4 else 99)
        if upto <= 5:
            return part
        if self.do_ar:
            o = self._column_reduce(part, B)
            free(part)
            return o
        o = ttnn.slice(part, [0, 0, 0, 0], [1, 1, B, HIDDEN])
        free(part)
        return o
