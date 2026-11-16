# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 greedy LM head: the vocab projection of the final normed hidden streamed per DRAM bank, with the argmax in
the same program (kernels/lmh_*.cpp), then GreedyB1's per-device merge, all-gather and pick (fused_decode/greedy.py).

Each device holds its vocab shard as moe_decode down_layout columns (one 1x32-output column = kt weight tiles,
contiguous in its bank); ~12 cores per bank on the bank's row each stream ~21 columns through a double-buffered CB,
run one LoFi custom_mm over K per column (the release LM head is BFP8 / LoFi) and keep the logits tiles in L1; their
writer takes the argmax with the softcap tie classes and posts it to the merge core. No logits tensor is written.
"""

from pathlib import Path

import ttnn
from models.demos.gemma4_26b_a4b_qb2.tt.moe_decode.op import TILE_BYTES, bank_split, down_layout, upload_per_chip

KERNELS = Path(__file__).resolve().parent / "kernels"
TILE = 32


class LmHeadGreedyB1:
    def __init__(self, mesh, weights, greedy, dtype=ttnn.bfloat8_b, per_bank=12):
        """weights: per chip [hidden, vd] torch (the device's vocab shard); greedy: the model's GreedyB1 (classes,
        merge core, local tile, pick)."""
        from models.demos.gpt_oss.tt.fused_decode.stream import row_placement

        hidden, vd = weights[0].shape
        assert vd == greedy.vd
        self.mesh, self.greedy, self.dtype = mesh, greedy, dtype
        self.kt, self.n = hidden // TILE, vd // TILE
        self.tile_bytes = TILE_BYTES[dtype]
        banks = mesh.dram_grid_size().x
        self.weight = upload_per_chip(mesh, [down_layout(w.float()[None], banks) for w in weights], dtype)
        merge = (greedy.merge.x, greedy.merge.y)
        while True:
            try:
                placed = row_placement(mesh, per_bank)
            except AssertionError:
                per_bank -= 1
                continue
            if all(merge not in {(c.x, c.y) for c in cs} for cs in placed.values()):
                break
            per_bank -= 1
        self.units = []  # (core, bank, j0 in the bank, cols, first column overall)
        for b, (cpb, c0) in enumerate(bank_split(self.n, banks)):
            for r, (cols, j0) in enumerate(bank_split(cpb, per_bank)):
                self.units.append((placed[b][r], b, j0, cols, c0 + j0))
        self.max_cols = max(u[3] for u in self.units)
        cores = [u[0] for u in self.units]
        self.worker_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        self.all_set = self.worker_set.merge(greedy.merge_set)

    def _cb(self, index, dtype, page, pages, cores):
        fmt = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)]
        tiny = dtype == ttnn.bfloat16 and page == 64
        if tiny:
            fmt = [
                ttnn.CBFormatDescriptor(
                    buffer_index=index,
                    data_format=dtype,
                    page_size=page,
                    tile=ttnn.TileDescriptor(ttnn.Tile([1, TILE])),
                )
            ]
        return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores, format_descriptors=fmt)

    def __call__(self, x, tokens):
        """x: the final normed hidden, row 0 of a [1, 1, 1, hidden] BF16 TILE tensor in L1; tokens: the [1, 1, 1, 32]
        UINT32 decode token input. Writes the greedy token into tokens[..., 0] on every device."""
        g = self.greedy
        cb_x, cb_w, cb_out, cb_scr, cb_slots, sem = 0, 1, 2, 3, 4, 0
        cbs = [
            self._cb(cb_x, ttnn.bfloat16, 64, self.kt, self.worker_set),
            self._cb(cb_w, self.dtype, self.tile_bytes, 2 * self.kt, self.worker_set),
            self._cb(cb_out, ttnn.bfloat16, 64, self.max_cols, self.worker_set),
            self._cb(cb_scr, ttnn.bfloat16, 256, 1, self.worker_set),
            self._cb(cb_slots, ttnn.bfloat16, 16 * len(self.units), 1, self.all_set),
        ]
        sems = [ttnn.SemaphoreDescriptor(sem, ttnn.CoreType.WORKER, self.all_set, 0)]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, (core, bank, j0, cols, first) in enumerate(self.units):
            reader_rt[core.x][core.y] = [x.buffer_address(), self.weight.buffer_address(), bank, j0, cols]
            compute_rt[core.x][core.y] = [cols]
            writer_rt[core.x][core.y] = [
                first * TILE,
                cols,
                g.merge_noc[0],
                g.merge_noc[1],
                i,
                g.classes.buffer_address(),
            ]
        dm = ttnn.DataMovementConfigDescriptor
        kernels_common = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "lmh_reader.cpp"),
                core_ranges=self.worker_set,
                compile_time_args=[cb_x, cb_w, self.kt, self.tile_bytes]
                + ttnn.TensorAccessorArgs(x).get_compile_time_args(),
                runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "lmh_compute.cpp"),
                core_ranges=self.worker_set,
                compile_time_args=[cb_x, cb_w, cb_out, self.kt],
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "lmh_writer.cpp"),
                core_ranges=self.worker_set,
                compile_time_args=[cb_out, cb_scr, cb_slots, sem]
                + ttnn.TensorAccessorArgs(g.classes).get_compile_time_args(),
                runtime_args=writer_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
        ]
        mesh_program = ttnn.MeshProgramDescriptor()
        for d in range(self.mesh.shape[1]):
            coord = ttnn.MeshCoordinate(0, d)
            merge_rt = ttnn.RuntimeArgs()
            merge_rt[g.merge.x][g.merge.y] = [g.local.buffer_address(), d * g.vd]
            merge = ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "greedy_merge.cpp"),
                core_ranges=g.merge_set,
                compile_time_args=[len(self.units), sem, cb_slots],
                runtime_args=merge_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
            )
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(
                kernels=kernels_common + [merge], semaphores=sems, cbs=cbs
            )
        ttnn.generic_op([x, self.weight, g.classes, g.local], mesh_program)
        return g.finish(tokens)
