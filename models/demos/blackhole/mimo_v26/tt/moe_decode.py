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
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tt.moe_prefill import COLS, HIDDEN, ROWS, TOPK, spread_pad_routing

SLOTS = 64  # masked_bincount needs tokens % 64 == 0: 32 decode slots + 32 pad rows per mesh row


class MiMoMoEDecode:
    def __init__(self, mesh, gate, moe, all_reduce=True, num_links=None):
        self.mesh, self.gate, self.moe, self.do_ar = mesh, gate, moe, all_reduce
        rep = ttnn.ReplicateTensorToMesh(mesh)
        # spread routing for the 64 slots of one mesh row, expert ids exact in bf16 (<= 256), weights 1/8
        sidx, sw = spread_pad_routing(1, SLOTS)
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

    def forward(self, x, mask, upto=99):
        """upto (stage-timing hook): 0 = pad+router+spread, 1..4 = moe stages (counts, dispatch, experts, combine), 5 = full moe without all-reduce."""
        B = x.shape[2]
        xp = ttnn.pad(x, [(0, 0), (0, 0), (0, SLOTS - B), (0, 0)], 0.0)
        wts, idx = self.gate.route(xp, "exact")
        wts2, idx2 = self._spread(wts, idx, mask)
        if upto == 0:
            return wts2, idx2
        x_rm = ttnn.reshape(ttnn.to_layout(xp, ttnn.ROW_MAJOR_LAYOUT), [1, SLOTS, HIDDEN])
        part = self.moe.forward(x_rm, wts2, idx2, upto=min(upto, 99) if upto <= 4 else 99)
        if upto <= 5:
            return part
        out = ttnn.all_reduce(part, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG) if self.do_ar else part
        if self.do_ar:
            ttnn.deallocate(part)
        o = ttnn.slice(out, [0, 0, 0, 0], [1, 1, B, HIDDEN])
        ttnn.deallocate(out)
        return o
