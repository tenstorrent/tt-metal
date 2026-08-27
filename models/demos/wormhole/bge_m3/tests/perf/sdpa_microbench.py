# SPDX-License-Identifier: Apache-2.0
"""Branch C microbenchmark driver. Runs ONE BENCH_MODE (env BGE_BENCH_MODE) of the
score-block compute kernel and reports traced-wall + device-kernel duration.
Whole-kernel timing per build variant => Amdahl split of the SUB_EXP region:
  mode 0 = subtract/broadcast only
  mode 1 = approximate exp only
  mode 2 = pack + reduce only
  mode 3 = full sub+exp+pack sequence
Standalone (no pytest/fabric). Subprocess-isolated; external timeout by caller.
"""
import os
import time

import torch

import ttnn

KROOT = "models/demos/wormhole/bge_m3/tt/custom_ops/encoder_sdpa/kernels"
KERNEL = f"{KROOT}/microbench_compute.cpp"

ROWS, COLS = 4, 64  # Sq_chunk_t x Sk_chunk_t tiles
REPEATS = 384  # 96 Q-work/core * 4 K-chunks
TILE = 32


def main():
    mode = int(os.environ.get("BGE_BENCH_MODE", "3"))
    dev = ttnn.open_mesh_device(ttnn.MeshShape(2, 1), trace_region_size=20_000_000)
    try:
        grid = dev.compute_with_storage_grid_size()
        cr = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])

        def cb(idx, ntiles, dt=ttnn.bfloat16):
            fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=TILE * TILE * 2)
            return ttnn.CBDescriptor(total_size=ntiles * TILE * TILE * 2, core_ranges=cr, format_descriptors=[fmt])

        # scores 4x64, max 4 (col-bcast), out scratch 4
        cb_scores, cb_max, cb_out = 0, 1, 2
        cbs = [cb(cb_scores, ROWS * COLS), cb(cb_max, ROWS), cb(cb_out, 8)]

        ct = [cb_scores, cb_max, cb_out, ROWS, COLS, REPEATS]
        compute = ttnn.KernelDescriptor(
            kernel_source=KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cr,
            compile_time_args=ct,
            runtime_args=[[ttnn.CoreCoord(x, y), []] for x in range(grid.x) for y in range(grid.y)],
            defines=[("BENCH_MODE", str(mode))],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
            ),
        )

        # dummy input tensors sized to the CBs so generic_op has io_tensors
        def mk(ntiles):
            return ttnn.from_torch(
                torch.randn(1, 1, TILE, ntiles * TILE, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        scores_t, max_t, out_t = mk(ROWS * COLS), mk(ROWS), mk(8)
        desc = ttnn.ProgramDescriptor(kernels=[compute], cbs=cbs)
        io = [scores_t, max_t, out_t]

        ttnn.generic_op(io, desc)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        ttnn.generic_op(io, desc)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.release_trace(dev, tid)
        ts.sort()
        print(f"RESULT mode={mode} traced_wall_min={ts[0]:.4f} med={ts[len(ts)//2]:.4f} ms", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
