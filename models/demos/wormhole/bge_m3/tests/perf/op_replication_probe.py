# SPDX-License-Identifier: Apache-2.0
"""Does per-program trace-replay overhead scale with core count / op size?

1-core tiny ops measured ~5us/program (mesh_dispatch_microbench). BGE layer
scaling implied ~0.9ms/program. This probe chains N copies of the REAL 64-core
AttnOut matmul ([M,1024]x[1024,1024], DRAM in/out) in a trace, sweeps N, and fits
the per-op wall slope. Compare slope to the op's profiled DEVICE KERNEL DURATION
(~0.83ms):
  slope ~= kernel_duration  => "880ms" is REAL device time, no hidden overhead
                               (the gap to 1020ms wall is elsewhere / data movement).
  slope  > kernel_duration  => real per-program overhead beyond kernel time exists
                               (reclaimable). Report the delta.
Also chains a 1-tile (1-core) matmul as the low-core control.
"""
import time

import pytest
import torch
from loguru import logger

import ttnn

M = 49152


def _cfg(mb, kb, nb, sbh=4, sbw=2):
    return ttnn.MinimalMatmulConfig(M_block_size=mb, K_block_size=kb, N_block_size=nb,
                                    subblock_h=sbh, subblock_w=sbw,
                                    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8))


@pytest.mark.parametrize("device_params", [{"trace_region_size": 200_000_000, "num_command_queues": 1}], indirect=True)
def test_op_replication(mesh_device):
    dev = mesh_device
    torch.manual_seed(0)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True,
                                           fp32_dest_acc_en=False, packer_l1_acc=True)

    def mk(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # 64-core AttnOut-shaped matmul: [M,1024] x [1024,1024] -> [M,1024], chained (out feeds next in).
    def run_case(name, Mrows, dim, cfg, Ns):
        x0 = mk(torch.randn(1, 1, Mrows, dim, dtype=torch.bfloat16) * 0.1)
        w = mk(torch.randn(1, 1, dim, dim, dtype=torch.bfloat16) * 0.05)

        def build(n):
            y = ttnn.experimental.minimal_matmul(input_tensor=x0, weight_tensor=w, config=cfg,
                                                  memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b,
                                                  compute_kernel_config=ckc)
            for _ in range(n - 1):
                y2 = ttnn.experimental.minimal_matmul(input_tensor=y, weight_tensor=w, config=cfg,
                                                      memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b,
                                                      compute_kernel_config=ckc)
                ttnn.deallocate(y)
                y = y2
            return y

        xs, ys = [], []
        for n in Ns:
            out = build(n); ttnn.synchronize_device(dev); ttnn.deallocate(out)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            out = build(n)
            ttnn.end_trace_capture(dev, tid, cq_id=0); ttnn.synchronize_device(dev)
            for _ in range(4):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            ts = []
            for _ in range(15):
                t0 = time.perf_counter(); ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                ts.append((time.perf_counter() - t0) * 1e3)
            ttnn.release_trace(dev, tid); ttnn.deallocate(out)
            ts.sort()
            xs.append(n); ys.append(ts[0])
            logger.info(f"  [{name}] N={n:3d}: wall min={ts[0]:8.3f} ms  ({ts[0]/n:.3f} ms/op)")
        nn = len(xs); sx, sy = sum(xs), sum(ys); sxx = sum(v*v for v in xs); sxy = sum(a*b for a, b in zip(xs, ys))
        slope = (nn*sxy - sx*sy) / (nn*sxx - sx*sx); intercept = (sy - slope*sx)/nn
        logger.info(f"  [{name}] SLOPE = {slope:.3f} ms/op   intercept = {intercept:.3f} ms")
        return slope

    logger.info("===== 64-core AttnOut matmul (M=49152, K=N=1024, DRAM in/out) =====")
    s64 = run_case("attnout_64core", M, 1024, _cfg(16, 8, 4), [30, 60, 120])
    logger.info("===== 1-core matmul control (M=32, K=N=32) =====")
    s1 = run_case("mm_1core", 32, 32, _cfg(1, 1, 1, 1, 1), [50, 100, 200])
    logger.info("=" * 70)
    logger.info(f"  64-core matmul per-op wall = {s64:.3f} ms/op  (profiled kernel dur ~0.83 ms)")
    logger.info(f"  1-core  matmul per-op wall = {s1*1000:.1f} us/op")
    logger.info(f"  => per-program overhead beyond kernel scales with core count: "
                f"{'YES' if s64 > 1.2 else 'NO (slope ~= kernel time => data-movement bound, no reclaimable dispatch)'}")
    logger.info("=" * 70)
