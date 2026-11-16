# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 router of the fused decode (kernels/router_*.cpp): the FP32 router projection of the boundary's flat router
input, split over K, and the top-k tail, as one program. Replaces ttnn.linear (FP32 x BF16 on 4 cores, compute bound:
88 HiFi4 tile products per core) + moe_decode.router_topk_b1.

  matmul cores (n, kb), n < E / 32, kb < kb_split: row 0 of logits tile n over K tiles [kb * kt, (kb + 1) * kt)
      (FP32 activation rows built from the flat input, BF16 weight tiles, HiFi4, FP32 DST), row 0 sent to the reducer;
  reducer: the kb_split partials of each logits tile summed on the SFPU in FP32, then the integer top-k / softmax /
      per-expert scale tail (moe_decode/kernels/router_topk_math.hpp) -> the packed routing row [1, 1, 32, 2k] (bf16
      ROW_MAJOR, row 0 = k weights then k ids; RouterTopk.op_packed format).
"""

import os
from pathlib import Path

import ttnn

KERNELS = Path(__file__).resolve().parent / "kernels"
TILE = 32


class RouterB1:
    def __init__(self, mesh, weight, per_expert_scale, k, *, kb_split=8, reducer=None, skip=()):
        """weight: [hidden, E] torch (applied to the prescaled FP32 input: logits = x @ weight); per_expert_scale: the
        [1, 1, 1, E] bf16 ROW_MAJOR device tensor; skip: logical (x, y) cores to leave free (e.g. the boundary core)."""
        hidden, E = weight.shape
        assert hidden % TILE == 0 and E % TILE == 0 and (hidden // TILE) % kb_split == 0
        self.mesh, self.E, self.k, self.kb = mesh, E, k, kb_split
        self.kt = hidden // TILE // kb_split
        self.n_tiles = E // TILE
        grid = mesh.compute_with_storage_grid_size()
        self.reducer = reducer or ttnn.CoreCoord(grid.x - 1, grid.y - 2)
        taken = set(skip) | {(self.reducer.x, self.reducer.y)}
        free = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
        self.cores = free[: self.n_tiles * kb_split]  # core c = (n, kb), c = n * kb_split + kb
        self.mm_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.red_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.reducer, self.reducer)])
        self.all_set = self.mm_set.merge(self.red_set)
        r = mesh.worker_core_from_logical_core(self.reducer)
        self.red_noc = (r.x, r.y)
        # weight tiles in core order: core (n, kb) holds W[(kb kt + t) tiles, n tile] at pages c kt + t
        w = weight.float().reshape(hidden // TILE, TILE, self.n_tiles, TILE).permute(2, 0, 1, 3)  # [n, ktile, 32, 32]
        w = w.reshape(self.n_tiles, kb_split, self.kt, TILE, TILE).reshape(-1, TILE)
        self.weight = ttnn.from_torch(
            w[None, None].bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.pes = per_expert_scale
        self.debug = None  # tests: an L1 [1, 1, 1, E] FP32 row-major tensor receiving the reducer's logits

    def _cb(self, index, dtype, page, pages, cores):
        fmt = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)]
        return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores, format_descriptors=fmt)

    def __call__(self, x):
        """x: the flat FP32 router input (one-core sharded tensor, value h at byte 4 h; fused_decode/boundary.py).
        Returns the packed routing row [1, 1, 32, 2k] bf16 ROW_MAJOR (L1)."""
        device = self.mesh
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, 2 * self.k]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        x_core = x.memory_config().shard_spec.grid.bounding_box().start
        xn = device.worker_core_from_logical_core(x_core)
        cb_x, cb_w, cb_out, cb_part, cb_logits, cb_scr, sem = 0, 1, 2, 3, 4, 5, 0
        n_part = self.n_tiles * self.kb
        cbs = [
            self._cb(cb_x, ttnn.float32, 4096, self.kt, self.mm_set),
            self._cb(cb_w, ttnn.bfloat16, 2048, self.kt, self.mm_set),
            self._cb(cb_out, ttnn.float32, 4096, 1, self.mm_set),
            # partials: on every core (same address on every core), only the reducer's are written
            self._cb(cb_part, ttnn.float32, 4096, n_part, self.all_set),
            self._cb(cb_logits, ttnn.float32, 4096, self.n_tiles, self.red_set),
            self._cb(cb_scr, ttnn.bfloat16, self.E * 6 + 4 * self.k + 64, 1, self.red_set),
        ]
        sems = [ttnn.SemaphoreDescriptor(sem, ttnn.CoreType.WORKER, self.all_set, 0)]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c, core in enumerate(self.cores):
            reader_rt[core.x][core.y] = [self.weight.buffer_address(), x.buffer_address(), xn.x, xn.y, c, c % self.kb]
            writer_rt[core.x][core.y] = [self.red_noc[0], self.red_noc[1], c]
        red_rt = ttnn.RuntimeArgs()
        red_rt[self.reducer.x][self.reducer.y] = [self.pes.buffer_address(), out.buffer_address()]
        if self.debug is not None:
            red_rt[self.reducer.x][self.reducer.y].append(self.debug.buffer_address())
        dm = ttnn.DataMovementConfigDescriptor
        red_ct = [cb_logits, cb_scr, self.E, self.k]
        red_ct += ttnn.TensorAccessorArgs(self.pes).get_compile_time_args()
        red_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
        if self.debug is not None:
            red_ct += ttnn.TensorAccessorArgs(self.debug).get_compile_time_args()
        # the partials unpack straight to DST in FP32 (the SFPU sum keeps full precision)
        red_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[cb_part] = (
            ttnn.UnpackToDestMode.UnpackToDestFp32
            if os.environ.get("RTR_DEFAULT_UNPACK") is None
            else ttnn.UnpackToDestMode.Default
        )
        red_cfg.unpack_to_dest_mode = modes
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "router_mm_reader.cpp"),
                core_ranges=self.mm_set,
                compile_time_args=[cb_x, cb_w, self.kt] + ttnn.TensorAccessorArgs(self.weight).get_compile_time_args(),
                runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "router_mm_compute.cpp"),
                core_ranges=self.mm_set,
                compile_time_args=[cb_x, cb_w, cb_out, self.kt],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "router_mm_writer.cpp"),
                core_ranges=self.mm_set,
                compile_time_args=[cb_out, cb_part, sem],
                runtime_args=writer_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "router_red_reader.cpp"),
                core_ranges=self.red_set,
                compile_time_args=[cb_part, n_part, sem],
                runtime_args=[],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "router_red_compute.cpp"),
                core_ranges=self.red_set,
                compile_time_args=[cb_part, cb_logits, self.n_tiles, self.kb],
                runtime_args=[],
                config=red_cfg,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "router_red_writer.cpp"),
                core_ranges=self.red_set,
                compile_time_args=red_ct,
                runtime_args=red_rt,
                defines=[("DEBUG_LOGITS", "1")] if self.debug is not None else [],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
        ]
        ttnn.generic_op(
            [x, self.weight, self.pes, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        )
        return out
