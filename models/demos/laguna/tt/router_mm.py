# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Router scores sigmoid(x @ w) for up to 3 token tiles as one K-split program (Laguna; kernels/rmm_*.cpp).

The DRAM-sharded router matmul gives each of its 8 cores one output column over the whole K (each reads 196 KB of the
1.57 MB bf16 weight; ~16 us per 32 rows against a ~3 us DRAM floor), one call per 32 rows. Here core (n, kc) reads only K chunk kc of column
n (KC = 12 chunks: 96 cores, 16 KB each), the chunks' fp32 partials meet at the column's first core over the NOC, are
summed exactly in fp32 (unpacked straight to DST) and go through the sigmoid: fp32 scores [1, 1, T, E] (3.5e-4 from
an fp32 torch reference).

Device time (p150x4, 3072 x 256): 11 / 19 / 27 us at 32 / 64 / 96 rows (current: ~16 us per 32 rows plus reshards).
Stage breakdown at 32 rows: launch ~1.1, weight reads ~4.2, activation reads ~1.3 (~8 at 96 rows: 8 cores read each
x tile), the 11 partial adds ~2.6 and the sigmoid ~1.3 us (one output tile per column core). The K split count (3 to
12) barely matters. Traced, the 96-core program's dispatch makes it slower than the current path at 32 rows (23 vs
21 us), so the decoder uses it only above 32 rows."""

from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


def _ranges(gx, idxs):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(i % gx, i // gx), ttnn.CoreCoord(i % gx, i // gx)) for i in idxs})


def router_scores(x, w, KC=12, memory_config=ttnn.L1_MEMORY_CONFIG):
    """x [1, 1, T, K] bf16 TILE (T a multiple of 32, at most 96); w [K, E] bf16 TILE (DRAM interleaved)."""
    device = x.device()
    T, K, E = int(x.shape[-2]), int(x.shape[-1]), int(w.shape[-1])
    Mt, Kt, Nt = T // TILE, K // TILE, E // TILE
    assert T % TILE == 0 and 1 <= Mt <= 3 and Kt % KC == 0, (T, K, KC)
    gs = device.compute_with_storage_grid_size()
    cores = Nt * KC
    assert cores <= gs.x * gs.y, cores
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, E]), ttnn.float32, ttnn.TILE_LAYOUT, device, memory_config)
    grid = ttnn.num_cores_to_corerangeset(cores, gs, True)
    reducers = [n * KC for n in range(Nt)]
    red_grid = _ranges(gs.x, reducers)
    work_grid = _ranges(gs.x, [c for c in range(cores) if c % KC])
    noc = []
    for i in reducers:
        p = device.worker_core_from_logical_core(ttnn.CoreCoord(i % gs.x, i // gs.x))
        noc += [p.x, p.y]
    Kc = Kt // KC

    def cb(i, fmt, page, pages):
        return ttnn.CBDescriptor(total_size=page * pages, core_ranges=grid,
                                 format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)])  # fmt: skip

    cbs = [cb(0, ttnn.bfloat16, 2048, Mt * Kc), cb(1, ttnn.bfloat16, 2048, Kc), cb(2, ttnn.float32, 4096, Mt),
           cb(3, ttnn.float32, 4096, (KC - 1) * Mt), cb(16, ttnn.float32, 4096, Mt)]  # fmt: skip
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "rmm_reader.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[gs.x, Mt, Kt, Nt, KC] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(w).get_compile_time_args()),
        common_runtime_args=[x.buffer_address(), w.buffer_address()], config=ttnn.ReaderConfigDescriptor(),
    )  # fmt: skip
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "rmm_writer.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid, compile_time_args=[gs.x, Mt, Nt, KC] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
        common_runtime_args=[out.buffer_address()] + noc, config=ttnn.WriterConfigDescriptor(),
    )  # fmt: skip

    def compute(core_ranges, reduce):
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        cfg.fp32_dest_acc_en = True
        cfg.math_approx_mode = False
        mode = [ttnn.UnpackToDestMode.Default] * 64
        mode[2] = mode[3] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = mode
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "rmm_compute.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_ranges, compile_time_args=[Mt, Kc, KC, int(reduce)], config=cfg,
        )  # fmt: skip

    kernels = [reader, writer, compute(red_grid, True), compute(work_grid, False)]
    sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=grid, initial_value=0)]
    ttnn.generic_op([x, w, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
    return out
