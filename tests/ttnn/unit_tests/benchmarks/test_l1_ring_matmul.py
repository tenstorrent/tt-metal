# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Benchmark: the L1-resident RING matmul (ttnn.matmul with
MatmulMultiCoreReuseMultiCast1DProgramConfig(gather_in0=True, mcast_in0=False)) at decode-style
M=32, with BOTH the weight (in1) and the activation (in0) L1 WIDTH-SHARDED across the same
compute-core grid, and the output also L1 width-sharded. Timed via trace capture + replay ONLY
(no eager timing, no minimal_matmul or DRAM-sharded comparison variants -- see
generated/gemm_sanity/l1_weights_matmul.py, the "V7-ring" section, for the experiment this was
ported from).

This is NOT a GEMM-from-DRAM number: weights are created once, in L1, before the timed region,
so the numbers below exclude the cost of ever streaming weights from DRAM.

Run:
    TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 pytest tests/ttnn/unit_tests/benchmarks/test_l1_ring_matmul.py --timeout 0

Reference trace times from generated/gemm_sanity/run_l1.log (V7-ring gather_in0 rows):
    BF16 32x2048x2048 @8x4 grid: 7.4 us/op
    BF16 32x4096x4096 @8x8 grid: 13.4 us/op  (the 8x4/32-core shard is 1 MB/core and OOMs on L1
                                               fragmentation for this shape; 8x8/64-core ran)
    FP8  32x2048x2048 @8x4 grid: 7.4 us/op
    FP8  32x4096x4096 @8x4 grid: 9.6 us/op
"""

import csv
import os
import time
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

GEMM_FLOPS_BENCHMARK_ENV = "TTNN_RUN_GEMM_FLOPS_BENCHMARK"

TILE = 32
M = 32


def pcc_of(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors (ported from
    generated/gemm_sanity/decode_gemm_sanity.py's pcc_of, itself a reimplementation of
    ttnn.tt_lib._internal.comparison_funcs.get_atol_rtol_pcc's get_pcc)."""
    golden = golden.clone().double().flatten()
    calculated = calculated.clone().double().flatten()
    golden[torch.isnan(golden) | torch.isinf(golden)] = 0
    calculated[torch.isnan(calculated) | torch.isinf(calculated)] = 0
    if torch.equal(golden, calculated):
        return 1.0
    if golden.std() == 0 or calculated.std() == 0:
        return 1.0 if torch.allclose(golden, calculated) else 0.0
    return torch.corrcoef(torch.stack([golden, calculated]))[0, 1].item()


def bytes_per_elem(dtype):
    if dtype == ttnn.bfloat16:
        return 2.0
    if dtype == ttnn.bfloat8_b:
        return 1.0625
    raise ValueError(f"unhandled dtype {dtype}")


def pick_subblock_w(per_core_n, cap=8):
    """Largest divisor of per_core_n that is <= cap (ported from
    generated/gemm_sanity/l1_weights_matmul.py)."""
    for d in range(min(cap, per_core_n), 0, -1):
        if per_core_n % d == 0:
            return d
    return 1


def to_host(t):
    """Read a (possibly sharded) device tensor back to torch, converting to interleaved first
    (ported from generated/gemm_sanity/decode_gemm_sanity.py)."""
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        buffer_type = t.memory_config().buffer_type
        interleaved = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)
        t2 = ttnn.sharded_to_interleaved(t, interleaved)
        out = ttnn.to_torch(t2)
        ttnn.deallocate(t2)
        return out
    return ttnn.to_torch(t)


# ---------------------------------------------------------------------------------------------
# Ring-matmul reprs actually produced for each of these cases, captured verbatim from
# generated/gemm_sanity/run_l1.log's "V7-ring gather_in0" rows, so this file documents what ran:
#
# BF16 K=2048 N=2048 grid=8x4 (trace=7.42 us, pcc=0.99961):
#   progcfg=MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=8-4,
#     in0_block_w=2,out_subblock_h=1,out_subblock_w=2,out_block_h=1,out_block_w=2,per_core_M=1,
#     per_core_N=2,fuse_batch=1,fused_activation=std::nullopt,mcast_in0=0,gather_in0=1,
#     hop_cores={},num_global_cb_receivers=1,untilize_out=0,allowed_worker_cores=std::nullopt,
#     stream_in1=0)
#   in0_mem=MemoryConfig(WIDTH_SHARDED,L1,shard_spec=ShardSpec{grid=[{0,0}-{7,3}], shape=[32, 64]})
#   in1_mem=MemoryConfig(WIDTH_SHARDED,L1,shard_spec=ShardSpec{grid=[{0,0}-{7,3}], shape=[2048, 64]})
#   out_mem=MemoryConfig(WIDTH_SHARDED,L1,shard_spec=ShardSpec{grid=[{0,0}-{7,3}], shape=[32, 64]})
#
# BF16 K=4096 N=4096 grid=8x8 (trace=13.42 us, pcc=0.99901):
#   progcfg=...(compute_with_storage_grid_size=8-8,in0_block_w=2,out_subblock_w=2,out_block_w=2,
#     per_core_M=1,per_core_N=2,mcast_in0=0,gather_in0=1,hop_cores={},untilize_out=0)
#   in0_mem shape=[32, 64], in1_mem shape=[4096, 64], out_mem shape=[32, 64] (grid {0,0}-{7,7})
#   (the 8x4/32-core shard for this shape is 1 MB/core and OOMs: "Not enough space to allocate
#    33554432 B L1 buffer across 32 banks ... bank size is 1436672 B")
#
# FP8 K=2048 N=2048 grid=8x4 (trace=7.40 us, pcc=0.99971): same shapes as the BF16 2048/8x4 case
#   above (in0_block_w=2, out_subblock_w=2, per_core_N=2).
#
# FP8 K=4096 N=4096 grid=8x4 (trace=9.58 us, pcc=0.99936):
#   progcfg=...(compute_with_storage_grid_size=8-4,in0_block_w=4,out_subblock_w=4,out_block_w=4,
#     per_core_M=1,per_core_N=4,mcast_in0=0,gather_in0=1,hop_cores={},untilize_out=0)
#   in0_mem shape=[32, 128], in1_mem shape=[4096, 128], out_mem shape=[32, 128] (grid {0,0}-{7,3})
# ---------------------------------------------------------------------------------------------

CASES = [
    ("BF16", ttnn.bfloat16, ttnn.MathFidelity.HiFi2, 2048, 2048, 8, 4),
    ("BF16", ttnn.bfloat16, ttnn.MathFidelity.HiFi2, 4096, 4096, 8, 8),
    ("FP8", ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, 2048, 2048, 8, 4),
    ("FP8", ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, 4096, 4096, 8, 4),
]
CASE_IDS = [f"{name}_{k}x{n}_{gx}x{gy}" for name, _, _, k, n, gx, gy in CASES]


@pytest.mark.skipif(
    os.getenv(GEMM_FLOPS_BENCHMARK_ENV) != "1",
    reason=f"Benchmark is manual-only; set {GEMM_FLOPS_BENCHMARK_ENV}=1 to run",
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 8388608}], indirect=True)
@pytest.mark.parametrize("case_name, dtype, math_fidelity, K, N, gx, gy", CASES, ids=CASE_IDS)
def test_l1_ring_matmul(device, case_name, dtype, math_fidelity, K, N, gx, gy):
    K_tiles = K // TILE
    N_tiles = N // TILE
    num_cores = gx * gy

    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x < gx or compute_grid.y < gy:
        pytest.skip(f"requested grid {gx}x{gy} exceeds device compute grid {compute_grid.x}x{compute_grid.y}")
    if K_tiles % num_cores != 0 or N_tiles % num_cores != 0:
        pytest.skip(f"K_tiles={K_tiles} or N_tiles={N_tiles} not divisible by num_cores={num_cores}")

    torch.manual_seed(0)
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    ref = torch.matmul(a.float(), b.float())

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # Ported verbatim from the "V7-ring" section of generated/gemm_sanity/l1_weights_matmul.py:
    # in0 (activation) L1 WIDTH-SHARDED across K, in1 (weight) L1 WIDTH-SHARDED across N (full K
    # height), output L1 WIDTH-SHARDED across N -- all on the SAME core grid.
    core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    in0_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set, [M, K // num_cores], ttnn.ShardOrientation.ROW_MAJOR),
    )
    in1_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set, [K, N // num_cores], ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set, [M, N // num_cores], ttnn.ShardOrientation.ROW_MAJOR),
    )

    in0_block_w = K_tiles // num_cores
    per_core_N = N_tiles // num_cores
    out_subblock_w = pick_subblock_w(per_core_N, cap=8)

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=ttnn.CoreRangeSet([]),
        untilize_out=False,
    )

    weight_shard_kb = K * (N // num_cores) * bytes_per_elem(dtype) / 1024

    # Weights (and activations) are created ONCE, in L1, outside the timed region.
    in0 = in1 = None
    out = None
    tid = None
    try:
        in1 = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_mem)
        in0 = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem)

        def op():
            return ttnn.matmul(
                in0,
                in1,
                program_config=program_config,
                memory_config=out_mem,
                compute_kernel_config=compute_kernel_config,
            )

        # Compile run, then warmup.
        out = op()
        for _ in range(5):
            out = op()
        ttnn.synchronize_device(device)

        # Trace-only timing: capture 20 back-to-back matmul calls into one trace, replay it 3
        # times, keep the MIN wall-clock time, divide by 20 -> us/op.
        trace_capture_ended = False
        try:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(20):
                out = op()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            trace_capture_ended = True

            best_s = None
            for _ in range(3):
                t0 = time.perf_counter()
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(device)
                dt = time.perf_counter() - t0
                best_s = dt if best_s is None else min(best_s, dt)
        finally:
            if tid is not None:
                try:
                    if not trace_capture_ended:
                        ttnn.end_trace_capture(device, tid, cq_id=0)
                finally:
                    ttnn.release_trace(device, tid)
                    tid = None

        us_per_op = best_s / 20 * 1e6

        calc = to_host(out)
        pcc = pcc_of(ref, calc.float())
    finally:
        for t_ in (out, in0, in1):
            if t_ is not None:
                ttnn.deallocate(t_)

    assert pcc > 0.99, f"PCC {pcc:.4f} too low for {case_name} M=32 K={K} N={N} grid={gx}x{gy}"

    tflops = 2 * M * N * K / (us_per_op * 1e-6) / 1e12
    logger.info(
        f"L1-ring {case_name} 32x{K}x{N} grid {gx}x{gy}: {us_per_op:.2f} us/op (trace), "
        f"{tflops:.1f} TFLOP/s, weight shard {weight_shard_kb:.0f} KB/core, pcc {pcc:.4f}"
    )

    csv_path = Path(os.environ["TT_METAL_HOME"]) / "generated" / "l1_ring_matmul.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "dtype",
                    "M",
                    "K",
                    "N",
                    "grid",
                    "num_cores",
                    "in0_block_w",
                    "per_core_N",
                    "out_subblock_w",
                    "trace_us_per_op",
                    "tflops",
                    "weight_shard_kb",
                    "pcc",
                ]
            )
        writer.writerow(
            [
                case_name,
                M,
                K,
                N,
                f"{gx}x{gy}",
                num_cores,
                in0_block_w,
                per_core_N,
                out_subblock_w,
                f"{us_per_op:.4f}",
                f"{tflops:.2f}",
                f"{weight_shard_kb:.1f}",
                f"{pcc:.4f}",
            ]
        )
