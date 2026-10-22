# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Grouped (sparse) MoE for the batch-32 DFlash verify (256 rows per device, TP experts: every device holds all 32
experts, 720 intermediate columns each, in the batched decode's dense layout [hidden, 32 x 720]).

The dense verify MoE computes every expert for every row (8x the routed work: 256 rows x 32 experts vs 256 x 4). Here
each expert's routed rows are grouped into 32-row slots and only those are computed:

  route   (fused_decode/kernels/gmoe_route.cpp): from the router's logits (dflash/batched.py _ResNorm), every row's
          routed experts (top-4, the 5th on a BF16 tie at the 4th logit, as the dense router), their softmax weights,
          and the plan: expert e's rows fill slots base[e] .. base[e] + ceil(count_e / 32) - 1 (at most 72 slots);
          writes P1T [72 x 32, 256] (one-hot gather), Pw [256, 72 x 32] (the routing weights, the combine; its last
          slot the down bias's) and the slot table [tiles[E], base[E]]; copies the layer's down bias rows into Y_g's
          last slot;
  gather  X_g = P1T x  (a matmul; one-hot at HiFi2 is an exact copy), block sharded as the gate|up reader loads it;
  gate_up (gmoe_gu_*.cpp): per (expert, column tile) the expert's slots x its gate / up columns + bias + SwiGLU ->
          A_g [72 x 32, 23 x 32] (the expert's 23 local column tiles; its 720 columns are 22.5 tiles of the dense
          layout, so the tile it shares with a neighbour carries the neighbour's columns too);
  down    (gmoe_down_*.cpp): per (expert, output column tile) A_g's slots x the expert's 23 down K tiles -> Y_g
          [72 x 32, hidden] BFP8; the shared K tile has the neighbour's rows zeroed, so the neighbour columns of A_g
          add nothing;
  combine out = Pw Y_g (each row's 4-5 expert outputs, weighted, plus the weighted down biases; a matmul).

The grouped matmuls read the slot table and spread their (expert, column) units over the cores by cost (weight read +
per-slot compute); their weights are relaid at init so a unit's tiles are a few large DRAM pages.
"""

import torch

import ttnn

from ..fused_decode.boundary import KERNEL_DIR

TILE = 32
E = 32
MAX_SLOTS = 72  # sum over experts of ceil(count / 32) <= 256 x 5 / 32 + 32 (a BF16 tie routes 5 experts)
MAXS = 8  # slots per expert (256 rows); the down's A_g block (an expert with more: in groups of MAXS, gmoe_down_*)


def max_slots(rows):
    """The slots of a rows-row pass: rows x 5 / 32 + 31 at most routed (each expert's last slot partly full) + the bias
    slot, a multiple of 8 (X_g's 8 core rows); 72 up to 256 rows, 112 at 512, 192 at 1024."""
    return max(MAX_SLOTS, -(-(rows * 5 // TILE + 32) // 8) * 8)


GROUP = 4  # gate|up in0 block (slots)
# Timed with the routing tables of all 24 layers of a real run injected: fewer outstanding DRAM reads are faster
# (congestion): gate|up 1 unit in flight (vs 3: 240 -> 224 us per layer at 3 pages), down 2 (vs 6: 92 -> 86 us)
GU_DEPTH, DN_DEPTH = 1, 2
# gate|up unit pages (consecutive pages in consecutive DRAM banks): weights-only reads with odd splits 3 / 5 / 9 / 15
# took 158-169 us, even 2 / 4 / 6 / 10 / 12 / 20 192-198 us; the full op 217 (3 pages) vs 268 us (10)
GU_SPLIT = 3
# unit cost weights (gmoe_common.hpp unit_cost: wc + slots x cc): gate|up 2:1 (217 vs 224 us at 3:1), down 4:1
GU_WC, GU_CC = 2, 1
DN_WC, DN_CC = 4, 1


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
    def __init__(self, mesh, moe, hidden, rows=256, share=None):
        """moe: per layer dict(gate, up [hidden, E x 720] BFP4, down [E x 720, hidden] BFP4, gate_bias / up_bias
        [1, E x 720], down_bias [E, hidden]) on DRAM (BatchedVerify.moe); gate / up / down are only read here.
        rows: the routed pass's rows (32 .. 256, a multiple of 32). share: another GroupedMoE whose relaid weights and
        rows-independent buffers (table, A_g, Y_g) this one uses (passes of different row counts, run one at a time)."""
        assert rows % TILE == 0 and TILE <= rows <= 1024
        self.mesh, self.hidden, self.rows = mesh, hidden, rows
        self.S = max_slots(rows)
        self.gu_split = GU_SPLIT
        grid = mesh.compute_with_storage_grid_size()
        self.gx = grid.x
        self.cores_all = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )
        self.nc = grid.x * grid.y
        self.kt = hidden // TILE
        self.nt = share.nt if share is not None else moe[0]["gate"].shape[-1] // TILE  # 720 column tiles
        self.ku = 23  # local column tiles per expert
        self.nu_down = hidden // TILE
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def zeros(shape, dtype, mc):
            return ttnn.from_torch(torch.zeros(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                                   memory_config=mc, mesh_mapper=rep)  # fmt: skip

        S = self.S * TILE
        # the gather / combine matrices in L1 (the route op writes them, the matmuls read them): route 54 -> 45 us,
        # gather 39 -> 36 us per layer
        # (a 1024-row pass: in DRAM, 115 KB per core each in L1 left too little L1 for the prefill's programs)
        pmc = ttnn.L1_MEMORY_CONFIG if rows <= 512 else ttnn.DRAM_MEMORY_CONFIG
        self.p1t = zeros([1, 1, S, rows], ttnn.bfloat16, pmc)
        self.pw = zeros([1, 1, rows, S], ttnn.bfloat16, pmc)
        self._pass_configs(rows)
        if share is not None and share.S != self.S:  # (more slots: its own A_g / Y_g; the weights shared)
            self.table, self.layers = share.table, share.layers
            self.ck_hifi2, self.ck_combine = share.ck_hifi2, share.ck_combine
            self._slot_buffers(zeros)
            return
        if share is not None:
            self.table, self.ag, self.yg, self.layers = share.table, share.ag, share.yg, share.layers
            self.ck_hifi2, self.ck_combine = share.ck_hifi2, share.ck_combine
            self.xg_mc, self.xg_noc = share.xg_mc, share.xg_noc
            return
        # the plan table [tiles[E], base[E]] (written by the route op, read by the grouped matmuls' readers)
        self.table = ttnn.from_torch(torch.zeros(1, 1, 1, 2 * E, dtype=torch.int32), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                     mesh_mapper=rep)  # fmt: skip
        self._slot_buffers(zeros)
        # per layer: 32-row bias tiles, and the down K tile each expert shares with a neighbour, the neighbour's rows
        # zeroed (row tile e of edge)
        keep = torch.zeros(E, 1, 1, TILE, 1)
        keep[0::2, ..., :16, :] = 1  # e even: its columns are the first 16 of its last tile
        keep[1::2, ..., 16:, :] = 1  # e odd: the last 16 of its first tile
        keep_t = [ttnn.from_torch(keep[e], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
                  for e in range(E)]  # fmt: skip
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
            gu = self._relayout(W["gate"], W["up"], E * self.ku, 2 * self.kt * 576, down=False, split=GU_SPLIT)
            dn = self._relayout(W["down"], edge, E * self.nu_down, self.ku * 576, down=True)
            edge.deallocate(True)
            assert W["down_bias"].shape[-2] == E
            self.layers.append(
                dict(
                    gate_bias=rows32(W["gate_bias"]),
                    up_bias=rows32(W["up_bias"]),
                    gate_up=gu,
                    down=dn,
                    down_bias8=ttnn.typecast(W["down_bias"], ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                )
            )
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

    def _slot_buffers(self, zeros):
        """The slot-count-dependent buffers: A_g, Y_g and X_g's layout."""
        mesh, S = self.mesh, self.S * TILE
        # A_g in L1, height sharded one 32-row slot per core (the down reader loads a slot in one read; in DRAM: round
        # 23.8 vs 23.3 ms at ISL 256); more slots than cores (512 / 1024-row passes): 2 per core
        spc = -(-self.S // self.nc)
        ag_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(self._first_cores(-(-self.S // spc)), [spc * TILE, self.ku * TILE], ttnn.ShardOrientation.ROW_MAJOR),
        )  # fmt: skip
        self.ag = zeros([1, 1, S, self.ku * TILE], ttnn.bfloat8_b, ag_mc)
        # Y_g: the down outputs per slot row (the combine's K side), BFP8 (down + combine 28 us faster per layer than
        # BF16, output PCC vs the float reference 0.99991 vs 0.99995), one in L1 for all layers; its last slot holds the
        # layer's down bias rows (copied by the route op) and Pw's columns of it the routing weights: the down bias
        # through the combine, no bias matmul and add per layer
        self.yg = zeros([1, 1, S, self.hidden], ttnn.bfloat8_b, ttnn.L1_MEMORY_CONFIG)
        # X_g block sharded as the gather's 10 x 8 output blocks: a slot's row is 10 contiguous 9-tile runs, the gate|up
        # reader loads it in 10 reads instead of 90 single-tile ones
        self.xg_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))]),
                           [self.S // 8 * TILE, self.kt // 10 * TILE], ttnn.ShardOrientation.ROW_MAJOR),
        )  # fmt: skip
        self.xg_noc = [mesh.worker_core_from_logical_core(ttnn.CoreCoord(j, 0)).x for j in range(10)]
        self.xg_noc += [mesh.worker_core_from_logical_core(ttnn.CoreCoord(0, i)).y for i in range(8)]

    def _pass_configs(self, rows):
        """The row-count-dependent matmul configs: gather [S x 32, rows] x [rows, hidden]: 10 x 8 cores, S / 8 x 9 output
        tiles each; combine [rows, S x 32] x [S x 32, hidden]: 10 x gy cores (gy = rows / 32 up to 8), rows / 32 / gy x
        9 output tiles each, 8-tile K blocks."""
        kt_r, pm = rows // TILE, self.S // 8
        # the gather's K blocks: all the rows up to 256; more rows: blocks small enough for the double-buffered in0
        bw = kt_r if kt_r <= 8 else max(d for d in range(1, 9) if kt_r % d == 0 and pm * d <= 56)
        self.gather_pc = _pc2d(10, 8, bw, pm, self.kt // 10, 3)
        gy = min(kt_r, 8)
        self.combine_pc = _pc2d(10, gy, 8, kt_r // gy, self.kt // 10, 3)

    def _relayout(self, src0, src1, units, page, down, split=1):
        """Every unit's weight tiles into one contiguous page (fused_decode/kernels/gmoe_relayout.cpp): a [units, page]
        row-major DRAM tensor (BFP4 bytes); split: as [units x split, page / split]."""
        dst = ttnn.empty([1, 1, units * split, page // 4 // split], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                         device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        cores = self.cores_all
        nt = src0.shape[-1] // TILE
        ct = [0, self.nc, self.gx, units, self.kt, nt, self.nu_down if down else self.ku, self.ku]
        for t in (src0, src1, dst):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gmoe_relayout.cpp"), core_ranges=cores, compile_time_args=ct,
            runtime_args=[], common_runtime_args=[t.buffer_address() for t in (src0, src1, dst)],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
            defines=([("GMOE_RELAYOUT_DOWN", "1")] if down else []) + ([("GMOE_RELAYOUT_SPLIT", str(split))] if split > 1 else []),
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

    def route(self, logits, li):
        """The routing and the plan in one program (fused_decode/kernels/gmoe_route.cpp): logits [1, 1, rows, E] (BF16
        tiles) -> self.p1t / self.pw (the routing weights in Pw's slot columns and its last slot) / self.table, and the
        layer's down bias rows into Y_g's last slot."""
        if self.S <= MAX_SLOTS:
            RW, RH, rpc = 9, MAX_SLOTS // 9, 4
        else:  # (512 / 1024 rows: the whole grid, 8 / 16 rows per phase-1 core, several slots per core)
            g = self.mesh.compute_with_storage_grid_size()
            RW, RH, rpc = g.x, g.y, self.rows // 64
        c0, c1 = ttnn.CoreCoord(0, 0), ttnn.CoreCoord(RW - 1, RH - 1)
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c0, c1)])
        a_, b_ = self.mesh.worker_core_from_logical_core(c0), self.mesh.worker_core_from_logical_core(c1)
        b8 = self.layers[li]["down_bias8"]
        io = [logits, self.p1t, self.pw, self.table, b8, self.yg]
        ct = [0, RW, self.S, RW * RH, 0, 1]
        for t in io:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        addr = [t.buffer_address() for t in io]
        k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gmoe_route.cpp"), core_ranges=cores, compile_time_args=ct,
            runtime_args=[], common_runtime_args=addr[:4] + [a_.x, a_.y, a_.x, a_.y, b_.x, b_.y] + addr[4:],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
            defines=[("GMOE_KTH", str(self.kt)), ("GMOE_ROWS", str(self.rows)), ("GMOE_RPC", str(rpc))]
            + ([("GMOE_CNT16", "1")] if self.rows >= 1024 else []),
        )  # fmt: skip
        scr = (self.rows * 32 + 64 + rpc * 32 + 2048 + rpc * 64 + 2 * (self.rows // TILE) * 2048 + 8 * E + 12 * 1088
               + (self.rows // rpc) * 64)  # fmt: skip
        cbs = [self._cb(0, ttnn.bfloat16, scr, 1, cores)]
        # 0: arrivals, 1: go, 2 .. 12: the expert counts (atomic increments at the hub, zeroed by every launch)
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, cores, 0) for i in range(2 + (E + 2) // 3)]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[k], semaphores=sems, cbs=cbs))

    def gather(self, x8):
        """x8: [1, 1, rows, hidden] BFP8 -> X_g [1, 1, 72 x 32, hidden] BFP8 (L1, block sharded)."""
        return ttnn.matmul(self.p1t, x8, program_config=self.gather_pc, compute_kernel_config=self.ck_hifi2,
                           dtype=ttnn.bfloat8_b, memory_config=self.xg_mc)  # fmt: skip

    def _meta_cbs(self, cores, base):
        page = 16 * ((4 * (2 + 2 * E) + 15) // 16)
        return [self._cb(base, ttnn.uint32, page, 1, cores), self._cb(base + 1, ttnn.uint32, page, 1, cores)]

    def gate_up(self, li, xg):
        """A_g = SwiGLU(X_g's slots x their expert's gate / up columns + bias) (self.ag)."""
        L = self.layers[li]
        cores = self.cores_all
        cb_scr, cb_mc, cb_mw, cb_x, cb_w, cb_b, cb_out = range(7)
        io = [self.table, xg, L["gate_up"], L["gate_bias"], L["up_bias"]]
        rct = [cb_scr, cb_mc, cb_mw, cb_x, cb_w, cb_b, self.nc, self.gx, self.kt, self.nt, self.ku, GU_WC, GU_CC]
        for t in io:
            rct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wct = [cb_mw, cb_out, self.ku] + ttnn.TensorAccessorArgs(self.ag).get_compile_time_args()
        cct = [cb_mc, cb_x, cb_w, cb_b, cb_out, self.kt, self.ku]
        rca = [t.buffer_address() for t in io]
        xg_arg = len(rca)
        rca += self.xg_noc
        # (512 / 1024-row passes: an expert with more than GROUP slots in groups of GROUP, each with its units' weights,
        # instead of its rows of X_g reloaded per unit: gate|up 4.1 -> 8.1 ms per pass at 512 rows otherwise)
        gmajor = [("GMOE_GU_GMAJOR", "1")] if self.rows > 256 else []
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_gu_reader.cpp"), core_ranges=cores, compile_time_args=rct,
                runtime_args=[], common_runtime_args=rca,
                defines=[("GMOE_DEPTH", str(GU_DEPTH)), ("GMOE_SPLIT", str(GU_SPLIT)), ("GMOE_XG_ARG", str(xg_arg)),
                         ("GMOE_XG_SX", "10"), ("GMOE_XG_RS", str(self.S // 8))] + gmajor,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_writer.cpp"), core_ranges=cores, compile_time_args=wct,
                runtime_args=[], common_runtime_args=[self.ag.buffer_address()],
                defines=[("GMOE_CAP", str(GROUP))] if gmajor else [],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_gu_compute.cpp"), core_ranges=cores, compile_time_args=cct,
                runtime_args=[], defines=gmajor,
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True,
                                                    math_approx_mode=False),
            ),
        ]  # fmt: skip
        cbs = [
            self._cb(cb_scr, ttnn.uint32, 8 * E, 1, cores),
            *self._meta_cbs(cores, cb_mc),
            self._cb(cb_x, ttnn.bfloat8_b, 1088, GROUP * self.kt, cores),
            self._cb(cb_w, ttnn.bfloat4_b, 576, GU_DEPTH * 2 * self.kt, cores),
            self._cb(cb_b, ttnn.bfloat16, 2048, GU_DEPTH * 2, cores),
            self._cb(cb_out, ttnn.bfloat8_b, 1088, 4, cores),
        ]
        ttnn.generic_op(io + [self.ag], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    def down(self, li):
        """Y_g = A_g's slots x their expert's down rows (self.yg; its last slot the route op's down bias rows)."""
        L = self.layers[li]
        yg = self.yg
        cores = self.cores_all
        cb_scr, cb_mc, cb_mw, cb_a, cb_w, cb_out = range(6)
        io = [self.table, self.ag, L["down"]]
        rct = [cb_scr, cb_mc, cb_mw, cb_a, cb_w, self.nc, self.gx, self.ku, self.nu_down, DN_WC, DN_CC, MAXS]
        for t in io:
            rct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wct = [cb_mw, cb_out, self.nu_down] + ttnn.TensorAccessorArgs(yg).get_compile_time_args()
        cct = [cb_mc, cb_a, cb_w, cb_out, self.ku, self.nu_down, MAXS]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_down_reader.cpp"), core_ranges=cores, compile_time_args=rct,
                runtime_args=[], common_runtime_args=[t.buffer_address() for t in io],
                defines=[("GMOE_DEPTH", str(DN_DEPTH))],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_writer.cpp"), core_ranges=cores, compile_time_args=wct,
                runtime_args=[], common_runtime_args=[yg.buffer_address()], defines=[("GMOE_CAP", str(MAXS))],
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
            self._cb(cb_w, ttnn.bfloat4_b, 576, DN_DEPTH * self.ku, cores),
            self._cb(cb_out, ttnn.bfloat8_b, 1088, 8, cores),
        ]
        ttnn.generic_op(io + [yg], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    def combine(self):
        """[1, 1, rows, hidden] BF16: every row's routed experts' outputs weighted by its routing weights, plus the
        weighted down biases (Y_g's last slot)."""
        return ttnn.matmul(self.pw, self.yg, program_config=self.combine_pc, compute_kernel_config=self.ck_combine,
                           dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip

    def from_x8(self, li, x8):
        """After route(li): the routed experts' outputs for the normed rows x8 [1, 1, rows, hidden] BFP8 (L1
        interleaved; dflash/batched.py _ResNorm writes them), summed with their routing weights, plus the down bias
        (no all-reduce)."""
        xg = self.gather(x8)
        self.gate_up(li, xg)
        xg.deallocate(True)
        self.down(li)
        return self.combine()
