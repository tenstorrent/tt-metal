# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ternary decode matmul (A3) on deepseek_v3_b1's DRAM streaming skeleton.

y [1, N] = x [1, K] @ W [K, N], M=1. x is a 1x32 tiny-tile row replicated on every
worker; W is width-sharded over the DRAM banks with each bank's tiles reordered so a
K-stick per output column is contiguous; each worker streams its own columns.

Step (a), this file: codes only, with ``workers_per_bank`` in {1, 2}. With 2 workers a
bank's columns are split in halves and the second worker reads from an address
offset, so the reader is unchanged.
"""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import get_max_page_size_and_num_pages
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

KERNEL = "models/demos/blackhole/bonsai2/native/a3/kernels/ternary_mm_kernel.cpp"


def worker_cores(device, workers_per_bank: int):
    """[(core, bank, sub)] -- the optimal reader per bank, plus a neighbour per extra worker."""
    primary = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    grid = device.compute_with_storage_grid_size()
    used = {(c.x, c.y) for c in primary}
    out = [(ttnn.CoreCoord(c.x, c.y), b, 0) for b, c in enumerate(primary)]
    for sub in range(1, workers_per_bank):
        for b, c in enumerate(primary):
            near = sorted(((dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if dx or dy),
                          key=lambda d: (abs(d[0]) + abs(d[1]), abs(d[0])))
            for dx, dy in near:
                x, y = c.x + dx, c.y + dy
                if 0 <= x < grid.x and 0 <= y < grid.y and (x, y) not in used:
                    used.add((x, y))
                    out.append((ttnn.CoreCoord(x, y), b, sub))
                    break
            else:
                raise RuntimeError(f"no free neighbour for bank {b} at {c}")
    return out


def core_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def shuffle_for_banks(w, banks: int):
    """[K, N] -> same shape, each bank shard's tiles reordered column-major (K-sticks)."""
    from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles

    return shuffle_tensor_tiles(w, 32, banks)


def weight_mem(device, k: int, n: int):
    banks = device.dram_grid_size().x
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
        ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
                       [k, n // banks], ttnn.ShardOrientation.ROW_MAJOR))


def act_mem(cores, k: int):
    """x replicated: one [1, K] shard per worker (the tensor is [workers, K])."""
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(core_set(cores), [1, k], ttnn.ShardOrientation.ROW_MAJOR))


def out_mem(cores, n: int):
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(core_set(cores), (1, n // len(cores)), ttnn.ShardOrientation.ROW_MAJOR))


def ternary_mm(x, w, y, workers, subblock_k: int):
    """x: [len(workers), K] 1x32 tiles, height-sharded (act_mem). w: shuffled, weight_mem.
    y: [1, N] 1x32 tiles, out_mem. ``workers`` from worker_cores()."""
    device = x.device()
    k = x.memory_config().shard_spec.shape[1]
    kt = k // 32
    banks = device.dram_grid_size().x
    wpb = len(workers) // banks
    pcn = w.memory_config().shard_spec.shape[1] // 32 // wpb
    assert kt % subblock_k == 0
    tile_bytes = w.get_tile().get_tile_size(w.dtype)
    page, pages = get_max_page_size_and_num_pages(device, subblock_k, tile_bytes)
    cores = [c for c, _, _ in workers]
    grid = core_set(cores)
    max_dest = 4  # fp32 Dst, half sync
    sbw = max(d for d in range(1, max_dest + 1) if pcn % d == 0)

    cb_in1 = ttnn.CBDescriptor(
        total_size=3 * subblock_k * tile_bytes, core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=w.dtype, page_size=tile_bytes,
                                                    tile=ttnn.TileDescriptor(w.get_tile()))])
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, x), cb_in1, ttnn.cb_descriptor_from_sharded_tensor(4, y)]

    vcs, seen = [], []
    for core, bank, _ in workers:
        vc = bank & 0x3
        if any(c.y == core.y and (b & 0x3) == vc for c, b in seen):
            vc = (vc + 1) & 0x3
        seen.append((core, bank))
        vcs.append((core, vc))
    common = [("tm_cb_in0", 0), ("tm_cb_in1", 1), ("tm_cb_out", 4), ("tm_subblock_k", subblock_k),
              ("tm_per_core_n", pcn), ("tm_num_subblocks_k", kt // subblock_k)]
    uk = UnifiedKernelDescriptor(
        kernel_source=KERNEL,
        core_ranges=grid,
        ncrisc_named_compile_time_args=common + [
            ("tm_in1_page_size", page), ("tm_in1_num_pages", pages),
            ("tm_in1_block_bytes", subblock_k * tile_bytes), ("tm_num_tiles_k", kt)],
        brisc_named_compile_time_args=[],
        trisc_named_compile_time_args=common + [("tm_subblock_w", sbw), ("tm_tile_r_dim", 1)],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False),
        unified_compile_time_core_descriptors=[UnifiedCompileTimeCoreDescriptor(
            named_compile_time_arg="tm_is_active_core", core_range=grid, value=1, other_value=0)],
        per_core_compile_time_descriptors=[
            PerCoreCompileTimeDescriptor(named_compile_time_arg="tm_bank_id",
                                         core_values=[(c, b) for c, b, _ in workers], other_value=0),
            PerCoreCompileTimeDescriptor(named_compile_time_arg="tm_vc", core_values=vcs, other_value=0),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="tm_in1_addr",
                core_values=[(c, w.buffer_address() + s * pcn * kt * tile_bytes) for c, _, s in workers],
                other_value=0),
        ],
    )
    prog = ttnn.ProgramDescriptor(kernels=uk.get_kernel_descriptors().kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([x, w, y], prog)
    return y
