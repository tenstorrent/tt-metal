# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Grouped (sparse) MoE for the batch-32 DFlash verify (256 rows per device, TP experts: every device holds all 32
experts, 720 intermediate columns each, in the batched decode's dense layout [hidden, 32 x 720]).

The dense verify MoE computes every expert for every row (8x the routed work: 256 rows x 32 experts vs 256 x 4). Here
each expert's routed rows are grouped into 32-row slots and only those are computed:

  plan    (fused_decode/kernels/gmoe_plan.cpp, one core per slot): the dense router's routed experts (top-4, the
          5th on a BF16 tie at the 4th logit: the router's top-5 ids and the dense weights), expert e's rows fill
          slots base[e] .. base[e] + ceil(count_e / 32) - 1 (at most 72 slots); writes P1T [72 x 32, 256] (one-hot
          gather), Pw [256, 72 x 32] (the dense routing weights, the combine) and the slot table [tiles[E], base[E]];
  gather  X_g = P1T x  (a matmul; one-hot at HiFi2 is an exact copy);
  gate_up (gmoe_gu_*.cpp): per (expert, column tile) the expert's slots x its gate / up columns + bias + SwiGLU ->
          A_g [64 x 32, 23 x 32] (the expert's 23 local column tiles; its 720 columns are 22.5 tiles of the dense
          layout, so the tile it shares with a neighbour carries the neighbour's columns too);
  down    (gmoe_down_*.cpp): per (expert, output column tile) A_g's slots x the expert's 23 down K tiles -> Y_g
          [72 x 32, hidden] BFP8; the shared K tile has the neighbour's rows zeroed, so the neighbour columns of A_g
          add nothing;
  combine out = Pw Y_g (each row's 4-5 expert outputs, weighted; a matmul).

The grouped matmuls read the slot table and spread their (expert, column) units over the cores by cost (weight read +
per-slot compute); their weights are relaid at init so a unit's tiles are one DRAM page (one large read, both NOCs,
several units in flight). Per layer at 256 rows: ~570 us vs ~900 us for the dense MoE.
"""

import os

import torch

import ttnn

from ..fused_decode.boundary import KERNEL_DIR

TILE = 32
E = 32
MAX_SLOTS = 72  # sum over experts of ceil(count / 32) <= 256 x 5 / 32 + 32 (a BF16 tie routes 5 experts)
MAXS = 8  # slots per expert (256 rows)
GROUP = 4  # gate|up in0 block (slots)


def _pc2d(gx, gy, bw, pm, pn, sw=1):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=bw,
        out_subblock_h=1,
        out_subblock_w=sw,
        out_block_h=pm,
        out_block_w=pn,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=None,
    )


class GroupedMoE:
    def __init__(self, mesh, moe, hidden, rows=256):
        """moe: per layer dict(gate, up [hidden, E x 720] BFP4, down [E x 720, hidden] BFP4, gate_bias / up_bias
        [1, E x 720]) on DRAM (BatchedVerify.moe)."""
        assert rows == 256
        self.mesh, self.hidden, self.rows = mesh, hidden, rows
        # weight reads in flight per core (units, split over both NOCs). With real routing tables (all 24 layers of a
        # real run injected, _scratch_b32_gmoe.py GUTAB) fewer outstanding reads are faster (DRAM congestion): gate|up
        # 1 unit (vs 3: 240 -> 224 us at 3 pages), down 2 (vs 6: 92 -> 86 us)
        self.gu_depth = int(os.environ.get("B32_GMOE_GU_DEPTH", "1"))
        self.dn_depth = int(os.environ.get("B32_GMOE_DN_DEPTH", "2"))
        # gate|up unit pages split in gu_split (consecutive pages in consecutive DRAM banks). Real tables, weights-only
        # reads: odd splits 3 / 5 / 9 / 15 158-169 us, even 2 / 4 / 6 / 10 / 12 / 20 192-198 us (not the NOC choice,
        # the start bank or 256-byte page alignment: each tested, no change); full op 3 pages 217 vs 10 pages 268 us
        self.gu_split = int(os.environ.get("B32_GMOE_GU_SPLIT", "3"))
        # gate|up unit cost weights (gmoe_common.hpp unit_cost: wc + slots x cc): real tables 2:1 217 vs 3:1 224 us
        self.gu_wc, self.gu_cc = 2, 1
        # gate|up in0 block (slots): an expert with more slots reloads its rows per block per unit
        self.gu_group = int(os.environ.get("B32_GMOE_GU_GROUP", str(GROUP)))
        # the gate|up bias copied into DST before the matmul accumulates (GMOE_BIAS_DST; no BF16 round trip and add
        # pass: 215.5 -> 212.1 us)
        self.gu_bias_dst = os.environ.get("B32_GMOE_BIAS_DST", "1") == "1"
        self.gu_inflight = 0  # (gate|up units with outstanding reads, GMOE_INFLIGHT; 0: gu_depth)
        self.gu_probe = []  # (unit-test probes: GMOE_PROBE_NOMM / NOX / NOW)
        # relaid weight pages padded to a multiple of page_align bytes (B32_GMOE_PAGE_ALIGN; 0: unpadded)
        self.page_align = int(os.environ.get("B32_GMOE_PAGE_ALIGN", "0"))
        grid = mesh.compute_with_storage_grid_size()
        self.gx = grid.x
        self.cores_all = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )
        self.nc = grid.x * grid.y
        self.kt = hidden // TILE
        self.nt = moe[0]["gate"].shape[-1] // TILE  # 720 column tiles
        self.ku = 23  # local column tiles per expert
        self.nu_down = hidden // TILE
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def zeros(shape, dtype, mc=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(torch.zeros(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                                   memory_config=mc, mesh_mapper=rep)  # fmt: skip

        S = MAX_SLOTS * TILE
        # the gather / combine matrices in L1 (the route op writes them, the matmuls read them): route 54 -> 45 us,
        # gather 39 -> 36 us per layer
        self.p1t = zeros([1, 1, S, rows], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG)
        self.pw = zeros([1, 1, rows, S], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG)
        # the plan table [tiles[E], base[E]] (written by the plan op, read by the grouped matmuls' readers)
        self.table = ttnn.from_torch(torch.zeros(1, 1, 1, 2 * E, dtype=torch.int32), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                     mesh_mapper=rep)  # fmt: skip
        # A_g in L1 (the down reader's single-tile reads skip DRAM: round 23.8 -> 23.3 ms at ISL 256; B32_GMOE_AG_L1=0)
        ag_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("B32_GMOE_AG_L1", "1") == "1" else ttnn.DRAM_MEMORY_CONFIG
        # (height sharded one 32-row slot per core, B32_GMOE_AG_SHARDED: the down reader loads a slot in one read)
        self.ag_sharded = os.environ.get("B32_GMOE_AG_SHARDED", "1") == "1"
        if self.ag_sharded:
            ag_mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                ttnn.ShardSpec(self._first_cores(MAX_SLOTS), [TILE, self.ku * TILE], ttnn.ShardOrientation.ROW_MAJOR),
            )  # fmt: skip
        self.ag = zeros([1, 1, S, self.ku * TILE], ttnn.bfloat8_b, ag_mc)
        # Y_g: the down outputs per slot row (the combine's K side), BFP8: down + combine 28 us faster per layer than
        # BF16, output PCC vs the float reference 0.99991 (BF16 0.99995)
        # (B32_GMOE_YG_L1, with bias_slot: one Y_g in L1 for all layers, the route op copies the layer's down bias rows
        # into its bias slot: the combine reads Y_g from L1)
        self.bias_slot = os.environ.get("B32_GMOE_BIAS", "1") == "1"
        self.yg_l1 = self.bias_slot and os.environ.get("B32_GMOE_YG_L1", "1") == "1"
        self.yg = zeros(
            [1, 1, S, hidden], ttnn.bfloat8_b, ttnn.L1_MEMORY_CONFIG if self.yg_l1 else ttnn.DRAM_MEMORY_CONFIG
        )
        # per layer: 32-row bias tiles, and the down K tile each expert shares with a neighbour, the neighbour's rows
        # zeroed (row tile e of edge)
        keep = torch.zeros(E, 1, 1, TILE, 1)
        keep[0::2, ..., :16, :] = 1  # e even: its columns are the first 16 of its last tile
        keep[1::2, ..., 16:, :] = 1  # e odd: the last 16 of its first tile
        keep_t = [ttnn.from_torch(keep[e], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
                  for e in range(E)]  # fmt: skip
        # the down bias through the combine (gmoe_route.cpp GMOE_BIAS_SLOT): the last slot of each layer's own Y_g holds
        # the experts' down bias rows, Pw's columns of it the routing weights (no bias matmul and add per layer)
        self.layers = []
        for W in moe:

            def rows32(b):
                rm = ttnn.to_layout(ttnn.slice(b, [0, 0, 0, 0], [1, 1, 1, b.shape[-1]]), ttnn.ROW_MAJOR_LAYOUT)
                return ttnn.to_layout(ttnn.repeat(rm, ttnn.Shape([1, 1, TILE, 1])), ttnn.TILE_LAYOUT)

            edges = []
            for e in range(E):
                kt = self.g0(e) + (0 if e % 2 else self.ku - 1)
                part = ttnn.slice(W["down"], [0, 0, kt * TILE, 0], [1, 1, kt * TILE + TILE, hidden])
                edges.append(ttnn.multiply(part, keep_t[e], dtype=ttnn.bfloat4_b))
                part.deallocate(True)
            edge = ttnn.concat(edges, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for t in edges:
                t.deallocate(True)
            gu = self._relayout(W["gate"], W["up"], E * self.ku, 2 * self.kt * 576, down=False, split=self.gu_split)
            dn = self._relayout(W["down"], edge, E * self.nu_down, self.ku * 576, down=True)
            edge.deallocate(True)
            self.layers.append(
                dict(gate_bias=rows32(W["gate_bias"]), up_bias=rows32(W["up_bias"]), gate_up=gu, down=dn)
            )
            if self.bias_slot:
                assert W["down_bias"].shape[-2] == E
                b8 = ttnn.typecast(W["down_bias"], ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if self.yg_l1:
                    self.layers[-1]["down_bias8"] = b8
                else:
                    pad = zeros([1, 1, S - TILE, hidden], ttnn.bfloat8_b)
                    self.layers[-1]["yg"] = ttnn.concat([pad, b8], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    pad.deallocate(True)
                    b8.deallocate(True)
        for t in keep_t:
            t.deallocate(True)
        self.ck_hifi2 = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )  # fmt: skip
        self.ck_combine = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )  # fmt: skip
        # gather [64 x 32, 256] x [256, hidden]: 10 x 8 cores, 8 x 9 output tiles each; combine [256, 64 x 32] x
        # [64 x 32, hidden]: 1 x 9 output tiles per core, 16-tile K blocks
        self.gather_pc = _pc2d(10, 8, rows // TILE, MAX_SLOTS // 8, self.kt // 10, 3)
        # X_g block sharded as the gather's 10 x 8 output blocks (B32_GMOE_XG_SHARDED): a slot's row is 10 contiguous
        # 9-tile runs, the gate|up reader loads it in 10 reads instead of 90 single-tile ones
        self.xg_sharded = os.environ.get("B32_GMOE_XG_SHARDED", "1") == "1"
        if self.xg_sharded:
            self.xg_mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                ttnn.ShardSpec(ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))]),
                               [MAX_SLOTS // 8 * TILE, self.kt // 10 * TILE], ttnn.ShardOrientation.ROW_MAJOR),
            )  # fmt: skip
            self.xg_noc = [mesh.worker_core_from_logical_core(ttnn.CoreCoord(j, 0)).x for j in range(10)]
            self.xg_noc += [mesh.worker_core_from_logical_core(ttnn.CoreCoord(0, i)).y for i in range(8)]
        self.combine_pc = _pc2d(10, 8, 8, 1, self.kt // 10, 3)
        # (B32_GMOE_DUMP: every layer's slot table copied per pass, for the real routing's slot counts)
        self.dump = None
        if os.environ.get("B32_GMOE_DUMP"):
            self.dump = [ttnn.from_torch(torch.zeros(1, 1, 1, 2 * E, dtype=torch.int32), dtype=ttnn.uint32,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                         mesh_mapper=rep) for _ in moe]  # fmt: skip

    def _page_a(self, page):
        """The relaid weights' page stride: page padded to self.page_align bytes (0: unpadded)."""
        a = self.page_align
        return -(-page // a) * a if a else page

    def _relayout(self, src0, src1, units, page, down, split=1):
        """Every unit's weight tiles into one contiguous page (fused_decode/kernels/gmoe_relayout.cpp): a [units, page]
        row-major DRAM tensor (BFP4 bytes); split: as [units x split, page / split]; pages padded per _page_a."""
        pa = self._page_a(page // split)
        dst = ttnn.empty([1, 1, units * split, pa // 4], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        cores = self.cores_all
        nt = src0.shape[-1] // TILE
        ct = [0, self.nc, self.gx, units, self.kt, nt, self.nu_down if down else self.ku, self.ku]
        for t in (src0, src1, dst):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gmoe_relayout.cpp"), core_ranges=cores, compile_time_args=ct,
            runtime_args=[], common_runtime_args=[t.buffer_address() for t in (src0, src1, dst)],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
            defines=([("GMOE_RELAYOUT_DOWN", "1")] if down else []) + ([("GMOE_RELAYOUT_SPLIT", str(split))] if split > 1 else [])
            + ([("GMOE_RELAYOUT_PAGE_A", str(pa))] if pa != page // split else []),
        )  # fmt: skip
        cbs = [self._cb(0, ttnn.uint32, page, 1, cores)]
        ttnn.generic_op([src0, src1, dst], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs))
        return dst

    @staticmethod
    def g0(e):
        return (45 * e) // 2

    def _first_cores(self, n):
        """The first n cores of the grid in row-major order (x fastest: core index y * gx + x)."""
        full, rest = divmod(n, self.gx)
        ranges = []
        if full:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.gx - 1, full - 1)))
        if rest:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rest - 1, full)))
        return ttnn.CoreRangeSet(ranges)

    @staticmethod
    def _cb(i, dtype, page, pages, cores):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=page)],
        )

    def plan(self, ids, w):
        """ids: the router's top-5 expert ids [1, 1, rows, 5] (UINT16 tiles); w: dense routing weights [1, 1, rows, E]
        (BF16 tiles). Writes self.p1t / self.pw / self.table."""
        cores = self._first_cores(MAX_SLOTS)
        ct = [0, self.gx, MAX_SLOTS]
        io = [ids, w, self.p1t, self.pw, self.table]
        for t in io:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gmoe_plan.cpp"),
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=[],
            common_runtime_args=[t.buffer_address() for t in io],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
        scr = 4 * (self.rows // TILE) * 2048 + 8 * self.rows + 8 * E + 64
        cbs = [self._cb(0, ttnn.bfloat16, scr, 1, cores)]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs))

    def route(self, logits, li=None):
        """The routing and the plan in one program (fused_decode/kernels/gmoe_route.cpp): logits [1, 1, rows, E] (BF16
        tiles, the router matmul's) -> the dense routing weights W [1, 1, rows, E] (BF16, L1; the down bias's), and
        self.p1t / self.pw / self.table as plan(). Replaces the router's top-k / softmax ops and the plan op. With
        bias_slot, W goes into Pw's last slot instead (the returned W is not written)."""
        RW, RH = 9, MAX_SLOTS // 9
        c0, c1 = ttnn.CoreCoord(0, 0), ttnn.CoreCoord(RW - 1, RH - 1)
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c0, c1)])
        a_, b_ = self.mesh.worker_core_from_logical_core(c0), self.mesh.worker_core_from_logical_core(c1)
        w = ttnn.empty([1, 1, self.rows, E], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                       memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
        io = [logits, w, self.p1t, self.pw, self.table]
        ct = [0, RW, MAX_SLOTS, RW * RH, 0, 1]
        defines = [("GMOE_BIAS_SLOT", str(MAX_SLOTS - 1))] if self.bias_slot else []
        extra = []
        if self.yg_l1:  # the layer's down bias rows into the shared Y_g's bias slot (cores 64 .. 71)
            b8 = self.layers[li]["down_bias8"]
            defines.append(("GMOE_BIAS_COPY", str(self.kt)))
            extra = [b8.buffer_address(), self.yg.buffer_address()]
            io = io + [b8, self.yg]
        for t in io:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gmoe_route.cpp"), core_ranges=cores, compile_time_args=ct,
            runtime_args=[], common_runtime_args=[t.buffer_address() for t in io[:5]] + [a_.x, a_.y, a_.x, a_.y, b_.x, b_.y] + extra,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
            defines=defines,
        )  # fmt: skip
        scr = self.rows * 32 + 64 + 4 * 32 + 2048 + 4 * 64 + 2 * (self.rows // TILE) * 2048 + 8 * E + 12 * 1088
        cbs = [self._cb(0, ttnn.bfloat16, scr, 1, cores)]
        # 0: arrivals, 1: go, 2 .. 12: the expert counts (atomic increments at the hub, zeroed by every launch)
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, cores, 0) for i in range(2 + (E + 2) // 3)]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[k], semaphores=sems, cbs=cbs))
        return w

    def gather(self, x8):
        """x8: [1, 1, rows, hidden] BFP8 -> X_g [1, 1, 72 x 32, hidden] BFP8 (L1)."""
        return ttnn.matmul(self.p1t, x8, program_config=self.gather_pc, compute_kernel_config=self.ck_hifi2,
                           dtype=ttnn.bfloat8_b, memory_config=self.xg_mc if self.xg_sharded else ttnn.L1_MEMORY_CONFIG)  # fmt: skip

    def _meta_cbs(self, cores, base):
        page = 16 * ((4 * (2 + 2 * E) + 15) // 16)
        return [self._cb(base, ttnn.uint32, page, 1, cores), self._cb(base + 1, ttnn.uint32, page, 1, cores)]

    def gate_up(self, li, xg, wc=None, cc=None):
        """A_g = SwiGLU(X_g's slots x their expert's gate / up columns + bias) (self.ag)."""
        L = self.layers[li]
        wc = self.gu_wc if wc is None else wc
        cc = self.gu_cc if cc is None else cc
        cores = self.cores_all
        cb_scr, cb_mc, cb_mw, cb_x, cb_w, cb_b, cb_out, cb_mm = range(8)
        io = [self.table, xg, L["gate_up"], L["gate_bias"], L["up_bias"]]
        rct = [cb_scr, cb_mc, cb_mw, cb_x, cb_w, cb_b, self.nc, self.gx, self.kt, self.nt, self.ku, wc, cc]
        for t in io:
            rct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wct = [cb_mw, cb_out, self.ku] + ttnn.TensorAccessorArgs(self.ag).get_compile_time_args()
        cct = [cb_mc, cb_x, cb_w, cb_b, cb_out, self.kt, self.ku, cb_mm]
        rca = [t.buffer_address() for t in io]
        xg_arg = len(rca)
        if self.xg_sharded:
            rca += self.xg_noc
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_gu_reader.cpp"), core_ranges=cores, compile_time_args=rct,
                runtime_args=[], common_runtime_args=rca,
                defines=[("GMOE_DEPTH", str(self.gu_depth)), ("GMOE_SPLIT", str(self.gu_split)), ("GMOE_GROUP", str(self.gu_group))]
                + [(d, "1") for d in self.gu_probe]
                + ([("GMOE_INFLIGHT", str(self.gu_inflight))] if self.gu_inflight else [])
                + ([("GMOE_PAGE_A", str(self._page_a(2 * self.kt * 576 // self.gu_split)))] if self.page_align else [])
                + ([("GMOE_XG_SHARDED", str(xg_arg)), ("GMOE_XG_SX", "10"), ("GMOE_XG_RS", str(MAX_SLOTS // 8))] if self.xg_sharded else []),
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_writer.cpp"), core_ranges=cores, compile_time_args=wct,
                runtime_args=[], common_runtime_args=[self.ag.buffer_address()],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_gu_compute.cpp"), core_ranges=cores, compile_time_args=cct,
                runtime_args=[], defines=[("GMOE_GROUP", str(self.gu_group))] + [(d, "1") for d in self.gu_probe]
                + ([("GMOE_BIAS_DST", "1")] if self.gu_bias_dst else []),
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True,
                                                    math_approx_mode=False),
            ),
        ]  # fmt: skip
        cbs = [
            self._cb(cb_scr, ttnn.uint32, 8 * E, 1, cores),
            *self._meta_cbs(cores, cb_mc),
            self._cb(cb_x, ttnn.bfloat8_b, 1088, self.gu_group * self.kt, cores),
            self._cb(cb_w, ttnn.bfloat4_b, 576, self.gu_depth * 2 * self.kt, cores),
            self._cb(cb_b, ttnn.bfloat16, 2048, self.gu_depth * 2, cores),
            self._cb(cb_out, ttnn.bfloat8_b, 1088, 4, cores),
            self._cb(cb_mm, ttnn.bfloat16, 2048, 4, cores),
        ]
        ttnn.generic_op(io + [self.ag], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    def _yg(self, li):
        return self.layers[li]["yg"] if self.bias_slot and not self.yg_l1 else self.yg

    def down(self, li, wc=4, cc=1):
        """Y_g = A_g's slots x their expert's down rows (self.yg; with bias_slot the layer's own, its last slot the
        down biases)."""
        L = self.layers[li]
        yg = self._yg(li)
        cores = self.cores_all
        cb_scr, cb_mc, cb_mw, cb_a, cb_w, cb_out = range(6)
        io = [self.table, self.ag, L["down"]]
        rct = [cb_scr, cb_mc, cb_mw, cb_a, cb_w, self.nc, self.gx, self.ku, self.nu_down, wc, cc, MAXS]
        for t in io:
            rct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wct = [cb_mw, cb_out, self.nu_down] + ttnn.TensorAccessorArgs(yg).get_compile_time_args()
        cct = [cb_mc, cb_a, cb_w, cb_out, self.ku, self.nu_down, MAXS]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_down_reader.cpp"), core_ranges=cores, compile_time_args=rct,
                runtime_args=[], common_runtime_args=[t.buffer_address() for t in io],
                defines=[("GMOE_DEPTH", str(self.dn_depth))] + ([("GMOE_AG_SHARDED", "1")] if self.ag_sharded else [])
                + ([("GMOE_PAGE_A", str(self._page_a(self.ku * 576)))] if self.page_align else []),
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_writer.cpp"), core_ranges=cores, compile_time_args=wct,
                runtime_args=[], common_runtime_args=[yg.buffer_address()],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_down_compute.cpp"), core_ranges=cores, compile_time_args=cct,
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True,
                                                    math_approx_mode=False),
            ),
        ]  # fmt: skip
        cbs = [
            self._cb(cb_scr, ttnn.uint32, 8 * E, 1, cores),
            *self._meta_cbs(cores, cb_mc),
            self._cb(cb_a, ttnn.bfloat8_b, 1088, MAXS * self.ku, cores),
            self._cb(cb_w, ttnn.bfloat4_b, 576, self.dn_depth * self.ku, cores),
            self._cb(cb_out, ttnn.bfloat8_b, 1088, 8, cores),
        ]
        ttnn.generic_op(io + [yg], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    def combine(self, li):
        """[1, 1, rows, hidden] BF16: every row's routed experts' outputs weighted by its routing weights (with
        bias_slot after route(): plus the weighted down biases)."""
        return ttnn.matmul(self.pw, self._yg(li), program_config=self.combine_pc, compute_kernel_config=self.ck_combine,
                           dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip

    def from_x8(self, li, x8):
        """__call__ after route() from the normed rows already in BFP8 (L1 interleaved; dflash/batched.py _ResNorm
        writes them)."""
        if self.dump is not None:
            ttnn.copy(self.table, self.dump[li])
        xg = self.gather(x8)
        self.gate_up(li, xg)
        xg.deallocate(True)
        self.down(li)
        return self.combine(li)

    def __call__(self, li, x, ids=None, w=None):
        """The routed experts' down outputs summed with their routing weights (no down bias, no all-reduce). x: the
        normed rows [1, 1, rows, hidden] BF16 (DRAM); ids / w: the router's top-5 ids and dense weights (None: the
        plan was made by route())."""
        if ids is not None:
            self.plan(ids, w)
        x8 = ttnn.typecast(x, ttnn.bfloat8_b)
        xg = self.gather(x8)
        x8.deallocate(True)
        self.gate_up(li, xg)
        xg.deallocate(True)
        self.down(li)
        return self.combine(li)
