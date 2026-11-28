# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused routing of the tensor-parallel routed prefill (prefill_routed.py, TP mode: every expert is local on every
device, so no histogram crosses the mesh).

From a chunk's top-k expert ids [C, k] (uint16 TILE) one program (kernels/tp_route.cpp, on E worker cores) writes:
  counts [1, E]  uint32 ROW_MAJOR: tokens of each expert;
  region [1, E]  uint32 ROW_MAJOR: each expert's first dispatch row (the 32-rounded counts' exclusive prefix);
  pos    [C, k]  uint32 ROW_MAJOR: dispatch row of pair (t, j) = region[e] + #pairs of its expert e from tokens < t;
  tok    [1, rows] uint32 ROW_MAJOR: token of every dispatch row (0 on the region padding, past the last region and on
                 the spare tile).
These are exactly what masked_bincount + offset_cumsum (counts, regions) and _pair_positions + the dispatch scatter
(pos, tok) computed in ~22 ops per chunk; one generic op costs the host one dispatch instead (untraced 128 / 1K
prefills were bound by the host).
"""

from pathlib import Path

import ttnn

KERNEL = Path(__file__).resolve().parent / "kernels" / "tp_route.cpp"
TILE = 32
GX = 8  # the E cores: core q at logical (q % GX, q // GX)


def _round(n, m):
    return -(-n // m) * m


class TpRoute:
    def __init__(self, mesh_device, num_experts, top_k):
        assert num_experts % GX == 0 and num_experts <= 64, num_experts
        self.mesh_device = mesh_device
        self.E, self.k = num_experts, top_k
        grid = mesh_device.compute_with_storage_grid_size()
        assert grid.x >= GX and grid.y >= num_experts // GX, f"{grid} grid < {GX} x {num_experts // GX}"
        self.cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, num_experts // GX - 1))]
        )
        nocs = [mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(q % GX, q // GX)) for q in range(num_experts)]
        self._noc = [c.x for c in nocs] + [c.y for c in nocs]
        self._programs = {}  # C -> (kernels, cbs, sems, hash)

    def __call__(self, indices, spare):
        """indices: [C, k] uint16 TILE (C a multiple of 32); spare: the spare tile's first row. Returns (counts,
        region, pos, tok), new DRAM tensors."""
        C = indices.shape[-2]
        E, k, mesh = self.E, self.k, self.mesh_device
        rows = spare + TILE

        def empty(shape):
            return ttnn.empty.function(
                shape, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        counts, region, pos, tok = empty([1, E]), empty([1, E]), empty([C, k]), empty([1, rows])
        prog = self._programs.get((C, spare))
        if prog is None:
            prog = self._programs[(C, spare)] = self._build(indices, counts, region, pos, tok, C, spare)
        kernels, cbs, sems, phash = prog
        kernels[0].common_runtime_args = [
            indices.buffer_address(),
            pos.buffer_address(),
            tok.buffer_address(),
            counts.buffer_address(),
            region.buffer_address(),
        ] + self._noc
        desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        desc.custom_program_hash = phash
        ttnn.generic_op.function([indices, counts, region, pos, tok], desc)
        return counts, region, pos, tok

    def _build(self, indices, counts, region, pos, tok, C, spare):
        E, k = self.E, self.k
        tr = C // TILE
        trc = -(-tr // E)  # tile rows per token block
        P = -(-tr // trc)  # token blocks (cores) in use
        block = trc * TILE
        slot = _round(block * 4, 64)
        pos_page = pos.buffer_aligned_page_size()
        cb_idx, cb_hist, cb_runs, cb_region, cb_pos, cb_stage = range(6)
        sizes = {
            cb_idx: trc * TILE * TILE * 2,
            cb_hist: E * E * 4,
            cb_runs: E * slot,
            cb_region: _round(C, TILE) * 4 + 2 * E * 4 + 1024 * 4,
            cb_pos: block * pos_page,
            cb_stage: E * block * 4,
        }
        cbs = [
            ttnn.CBDescriptor(
                total_size=size,
                core_ranges=self.cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=size)],
            )
            for i, size in sizes.items()
        ]
        sem_hist, sem_runs = 0, 1
        sems = [
            ttnn.SemaphoreDescriptor(sem_hist, ttnn.CoreType.WORKER, self.cores, 0),
            ttnn.SemaphoreDescriptor(sem_runs, ttnn.CoreType.WORKER, self.cores, 0),
        ]
        ct = [C, k, E, P, trc, spare, slot, pos_page, cb_idx, cb_hist, cb_runs, cb_region, cb_pos, cb_stage]
        ct += [sem_hist, sem_runs, GX]
        for t in (indices, pos, tok, counts, region):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL),
                core_ranges=self.cores,
                compile_time_args=ct,
                common_runtime_args=[0] * (5 + 2 * E),
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                ),
            )
        ]
        phash = hash(("TpRoute", C, spare, E, k, P, trc, pos_page)) & ((1 << 64) - 1)
        return kernels, cbs, sems, phash
