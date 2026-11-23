# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Two-stage verify LM head: stage 1 streams the BFP4 LM head (half the BF8 head's bytes) keeping each row's K best
columns per core (LinearStream out_mode 6 + LM_TOPK); stage 2 (this op, fused_decode/kernels/lm_rescore_*.cpp)
rescores each row's K best candidates of the device with the BF8 weights: the row cores merge the stage-1 lists into a
table multicast to every core; each distinct candidate column tile is read once and multiplied (every row of x) by the
first entry that has it (the same LoFi custom_mm over the same K tiles as the BF8 LM head, so the logit is the BF8
head's), which writes [key, column] for every entry of that tile into the argmax merge's per-core lists
(dflash/argmax.py). On 1304 captured verify rows the BF8 argmax ranked at most 5th among the BFP4 logits (K = 8)."""

import ttnn

from ..fused_decode.boundary import KERNEL_DIR
from ..fused_decode.stream import BFP8_TILE_BYTES, NBIAS, TILE


class LMRescore:
    def __init__(self, mesh_device, rows, K, n1, hidden):
        """n1: stage-1 cores (lists). The BF8 weights are the padded stream of dflash/lm_head.py padded_lm_head with
        round_robin (column tile n on bank n % banks: the random column reads spread over the banks)."""
        self.mesh, self.M, self.K, self.n1 = mesh_device, rows, K, n1
        self.kx = hidden // TILE
        self.kt = self.kx + 1
        self.col_bytes = self.kt * BFP8_TILE_BYTES
        grid = mesh_device.compute_with_storage_grid_size()
        n = rows * K
        # one K x rows rectangle at (0, 0) (single-core ranges launch measurably slower under trace): candidate j of
        # row r on (j, r)
        assert K <= grid.x and rows <= grid.y
        self.core_list = [ttnn.CoreCoord(i // rows, i % rows) for i in range(n)]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(K - 1, rows - 1))])

    def __call__(self, lists1, x, w8, lists2):
        M, K = self.M, self.K
        cb_x, cb_w, cb_tab, cb_scr, cb_out = 0, 1, 2, 3, 16
        rows_tile = ttnn.TileDescriptor(ttnn.Tile([M, TILE]))

        def cb(i, dtype, page, pages, tile=None):
            fmt = dict(buffer_index=i, data_format=dtype, page_size=page)
            if tile is not None:
                fmt["tile"] = tile
            return ttnn.CBDescriptor(
                total_size=page * pages, core_ranges=self.cores, format_descriptors=[ttnn.CBFormatDescriptor(**fmt)]
            )

        cbs = [
            cb(cb_x, ttnn.bfloat16, 2 * TILE * M, self.kt, rows_tile),
            cb(cb_w, ttnn.bfloat8_b, BFP8_TILE_BYTES, self.kt),
            cb(cb_tab, ttnn.uint32, max(4 * M * K, 64), 1),
            cb(cb_scr, ttnn.uint32, self.n1 * K * 8, 1),
            cb(cb_out, ttnn.bfloat16, 2 * TILE * M, 1, rows_tile),
        ]
        reader_ct = [cb_x, cb_w, cb_tab, cb_scr, self.n1, K, M, self.kx, self.col_bytes, NBIAS]
        for t in (lists1, x):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [cb_out, cb_tab, M, K, K * M * 8] + ttnn.TensorAccessorArgs(lists2).get_compile_time_args()
        c0, c1 = self.core_list[0], self.core_list[-1]
        a_, b_ = self.mesh.worker_core_from_logical_core(c0), self.mesh.worker_core_from_logical_core(c1)
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.core_list):
            r, j = i % M, i // M  # (x0 + j, y0 + r)
            reader_rt[c.x][c.y] = [lists1.buffer_address(), x.buffer_address(), w8.buffer_address(), r, j]
            reader_rt[c.x][c.y] = list(reader_rt[c.x][c.y]) + [a_.x, a_.y, b_.x, b_.y, a_.x, a_.y]
            writer_rt[c.x][c.y] = [lists2.buffer_address(), r, j]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "lm_rescore_reader.cpp"),
                core_ranges=self.cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=[("RS_RR", str(self.mesh.dram_grid_size().x))],
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "lm_rescore_writer.cpp"),
                core_ranges=self.cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "lm_rescore_compute.cpp"),
                core_ranges=self.cores,
                compile_time_args=[cb_x, cb_w, cb_out, self.kt],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, self.cores, 0) for i in (0, 1)]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        ttnn.generic_op([lists1, x, w8, lists2], program)
