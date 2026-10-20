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
        # weight reads in flight per core (units, split over both NOCs): gate|up 288 -> 241 us at 3 (vs 1); down: no
        # gain past 6
        self.gu_depth, self.dn_depth = 3, 6
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
        self.ag = zeros([1, 1, S, self.ku * TILE], ttnn.bfloat8_b)
        # Y_g: the down outputs per slot row (the combine's K side), BFP8: down + combine 28 us faster per layer than
        # BF16, output PCC vs the float reference 0.99991 (BF16 0.99995)
        self.yg = zeros([1, 1, S, hidden], ttnn.bfloat8_b)
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
            gu = self._relayout(W["gate"], W["up"], E * self.ku, 2 * self.kt * 576, down=False)
            dn = self._relayout(W["down"], edge, E * self.nu_down, self.ku * 576, down=True)
            edge.deallocate(True)
            self.layers.append(
                dict(gate_bias=rows32(W["gate_bias"]), up_bias=rows32(W["up_bias"]), gate_up=gu, down=dn)
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
        # gather [64 x 32, 256] x [256, hidden]: 10 x 8 cores, 8 x 9 output tiles each; combine [256, 64 x 32] x
        # [64 x 32, hidden]: 1 x 9 output tiles per core, 16-tile K blocks
        self.gather_pc = _pc2d(10, 8, rows // TILE, MAX_SLOTS // 8, self.kt // 10, 3)
        self.combine_pc = _pc2d(10, 8, 8, 1, self.kt // 10, 3)

    def _relayout(self, src0, src1, units, page, down):
        """Every unit's weight tiles into one contiguous page (fused_decode/kernels/gmoe_relayout.cpp): a [units, page]
        row-major DRAM tensor (BFP4 bytes)."""
        dst = ttnn.empty([1, 1, units, page // 4], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
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
            defines=[("GMOE_RELAYOUT_DOWN", "1")] if down else [],
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

    def route(self, logits):
        """The routing and the plan in one program (fused_decode/kernels/gmoe_route.cpp): logits [1, 1, rows, E] (BF16
        tiles, the router matmul's) -> the dense routing weights W [1, 1, rows, E] (BF16, L1; the down bias's), and
        self.p1t / self.pw / self.table as plan(). Replaces the router's top-k / softmax ops and the plan op."""
        RW, RH = 9, MAX_SLOTS // 9
        c0, c1 = ttnn.CoreCoord(0, 0), ttnn.CoreCoord(RW - 1, RH - 1)
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c0, c1)])
        a_, b_ = self.mesh.worker_core_from_logical_core(c0), self.mesh.worker_core_from_logical_core(c1)
        w = ttnn.empty([1, 1, self.rows, E], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                       memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
        io = [logits, w, self.p1t, self.pw, self.table]
        ct = [0, RW, MAX_SLOTS, RW * RH, 0, 1]
        for t in io:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gmoe_route.cpp"), core_ranges=cores, compile_time_args=ct,
            runtime_args=[], common_runtime_args=[t.buffer_address() for t in io] + [a_.x, a_.y, a_.x, a_.y, b_.x, b_.y],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )  # fmt: skip
        scr = self.rows * 32 + 4 * 32 + 2048 + 4 * 64 + 2 * (self.rows // TILE) * 2048 + 8 * E
        cbs = [self._cb(0, ttnn.bfloat16, scr, 1, cores)]
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, cores, 0) for i in (0, 1)]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[k], semaphores=sems, cbs=cbs))
        return w

    def gather(self, x8):
        """x8: [1, 1, rows, hidden] BFP8 -> X_g [1, 1, 72 x 32, hidden] BFP8 (L1)."""
        return ttnn.matmul(self.p1t, x8, program_config=self.gather_pc, compute_kernel_config=self.ck_hifi2,
                           dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip

    def _meta_cbs(self, cores, base):
        page = 16 * ((4 * (2 + 2 * E) + 15) // 16)
        return [self._cb(base, ttnn.uint32, page, 1, cores), self._cb(base + 1, ttnn.uint32, page, 1, cores)]

    def gate_up(self, li, xg, wc=3, cc=1):
        """A_g = SwiGLU(X_g's slots x their expert's gate / up columns + bias) (self.ag)."""
        L = self.layers[li]
        cores = self.cores_all
        cb_scr, cb_mc, cb_mw, cb_x, cb_w, cb_b, cb_out, cb_mm = range(8)
        io = [self.table, xg, L["gate_up"], L["gate_bias"], L["up_bias"]]
        rct = [cb_scr, cb_mc, cb_mw, cb_x, cb_w, cb_b, self.nc, self.gx, self.kt, self.nt, self.ku, wc, cc]
        for t in io:
            rct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wct = [cb_mw, cb_out, self.ku] + ttnn.TensorAccessorArgs(self.ag).get_compile_time_args()
        cct = [cb_mc, cb_x, cb_w, cb_b, cb_out, self.kt, self.ku, cb_mm]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_gu_reader.cpp"), core_ranges=cores, compile_time_args=rct,
                runtime_args=[], common_runtime_args=[t.buffer_address() for t in io], defines=[("GMOE_DEPTH", str(self.gu_depth))],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_writer.cpp"), core_ranges=cores, compile_time_args=wct,
                runtime_args=[], common_runtime_args=[self.ag.buffer_address()],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_gu_compute.cpp"), core_ranges=cores, compile_time_args=cct,
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True,
                                                    math_approx_mode=False),
            ),
        ]  # fmt: skip
        cbs = [
            self._cb(cb_scr, ttnn.uint32, 8 * E, 1, cores),
            *self._meta_cbs(cores, cb_mc),
            self._cb(cb_x, ttnn.bfloat8_b, 1088, GROUP * self.kt, cores),
            self._cb(cb_w, ttnn.bfloat4_b, 576, self.gu_depth * 2 * self.kt, cores),
            self._cb(cb_b, ttnn.bfloat16, 2048, self.gu_depth * 2, cores),
            self._cb(cb_out, ttnn.bfloat8_b, 1088, 4, cores),
            self._cb(cb_mm, ttnn.bfloat16, 2048, 4, cores),
        ]
        ttnn.generic_op(io + [self.ag], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    def down(self, li, wc=4, cc=1):
        """Y_g = A_g's slots x their expert's down rows (self.yg)."""
        L = self.layers[li]
        cores = self.cores_all
        cb_scr, cb_mc, cb_mw, cb_a, cb_w, cb_out = range(6)
        io = [self.table, self.ag, L["down"]]
        rct = [cb_scr, cb_mc, cb_mw, cb_a, cb_w, self.nc, self.gx, self.ku, self.nu_down, wc, cc, MAXS]
        for t in io:
            rct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wct = [cb_mw, cb_out, self.nu_down] + ttnn.TensorAccessorArgs(self.yg).get_compile_time_args()
        cct = [cb_mc, cb_a, cb_w, cb_out, self.ku, self.nu_down, MAXS]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_down_reader.cpp"), core_ranges=cores, compile_time_args=rct,
                runtime_args=[], common_runtime_args=[t.buffer_address() for t in io], defines=[("GMOE_DEPTH", str(self.dn_depth))],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gmoe_writer.cpp"), core_ranges=cores, compile_time_args=wct,
                runtime_args=[], common_runtime_args=[self.yg.buffer_address()],
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
        ttnn.generic_op(io + [self.yg], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    def combine(self):
        """[1, 1, rows, hidden] BF16: every row's routed experts' outputs weighted by its routing weights."""
        return ttnn.matmul(self.pw, self.yg, program_config=self.combine_pc, compute_kernel_config=self.ck_combine,
                           dtype=ttnn.bfloat16)  # fmt: skip

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
        return self.combine()
