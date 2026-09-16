# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Benchmark: the DRAM-sharded decode matmul (ttnn.matmul with
MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig) at decode-style M=32, with the weight (in1)
DRAM WIDTH-SHARDED across the DRAM banks, the activation (in0) L1 WIDTH-SHARDED on an 8x4
(32-core) compute grid, and the output L1 width-sharded on that same grid. Timed via trace
capture + replay ONLY (no eager timing, no comparison variants -- see the "V2 DRAM-sharded
decode" section of generated/gemm_sanity/decode_gemm_sanity.py for the experiment this was
ported from).

This IS a GEMM-from-DRAM number: the weight lives in DRAM and is streamed in every call, so
(unlike tests/ttnn/unit_tests/benchmarks/test_l1_ring_matmul.py) this measures real DRAM
bandwidth, not an L1-resident compute rate.

Run:
    TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 pytest tests/ttnn/unit_tests/benchmarks/test_dram_sharded_decode_matmul.py --timeout 0

Reference trace times from generated/gemm_sanity/run2.log ("V2 DRAM-sharded decode" rows):
    BF16 32x2048x2048: 26.9 us
    BF16 32x4096x4096: 70.8 us
    FP8  32x2048x2048: 25.2 us
    FP8  32x4096x4096: 40.9 us
"""

import csv
import math
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
# DRAM-sharded decode matmul reprs actually produced for each of these cases, captured verbatim
# from generated/gemm_sanity/run2.log's "V2 DRAM-sharded decode" rows, so this file documents
# what ran (num_banks = device.dram_grid_size().x = 8 on this box; grid 8x4 = 32 compute cores):
#
# BF16 K=2048 N=2048 (eager=27.60 us, trace=26.93 us, pcc=0.99995, 313.4 GB/s):
#   progcfg=MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=2,per_core_M=1,
#     per_core_N=2,fused_activation=std::nullopt,num_workers_per_dram_bank=1)
#
# BF16 K=4096 N=4096 (eager=71.70 us, trace=70.79 us, pcc=0.99994, 475.3 GB/s):
#   progcfg=MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=4,per_core_M=1,
#     per_core_N=4,fused_activation=std::nullopt,num_workers_per_dram_bank=1)
#
# FP8  K=2048 N=2048 (eager=25.86 us, trace=25.23 us, pcc=0.99981, 177.7 GB/s):
#   progcfg=MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=2,per_core_M=1,
#     per_core_N=2,fused_activation=std::nullopt,num_workers_per_dram_bank=1)
#
# FP8  K=4096 N=4096 (eager=41.48 us, trace=40.82 us, pcc=0.99979, 436.4 GB/s):
#   progcfg=MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=4,per_core_M=1,
#     per_core_N=4,fused_activation=std::nullopt,num_workers_per_dram_bank=1)
#
# Memory configs (from the V2 helpers in generated/gemm_sanity/decode_gemm_sanity.py, themselves
# ported from models/demos/blackhole/qwen36/tt/tp_common.py's create_dram_sharded_mem_config /
# create_activation_shard_config):
#   in1 (weight) [1,1,K,N]: padded_N = ceil(N / (32*num_banks)) * 32*num_banks; MemoryConfig(
#     WIDTH_SHARDED, DRAM, ShardSpec(CoreRangeSet({CoreRange((0,0),(num_banks-1,0))}),
#     [K, padded_N // num_banks], ROW_MAJOR))
#   in0 (activation) [1,1,32,K] bfloat16: MemoryConfig(WIDTH_SHARDED, L1, ShardSpec(
#     CoreRangeSet({CoreRange((0,0),(7,3))}), [32, K // 32], ROW_MAJOR))
#   out: V2 used the bare ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG (WIDTH_SHARDED, L1, no explicit
#     ShardSpec) -- the DRAM-sharded matmul kernel derives the actual output shard from the
#     program config and the in0 grid at runtime, so that is what this file uses too.
# ---------------------------------------------------------------------------------------------

CASES = [
    ("BF16", ttnn.bfloat16, ttnn.MathFidelity.HiFi2, 2048, 2048),
    ("BF16", ttnn.bfloat16, ttnn.MathFidelity.HiFi2, 4096, 4096),
    ("FP8", ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, 2048, 2048),
    ("FP8", ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, 4096, 4096),
]
CASE_IDS = [f"{name}_{k}x{n}" for name, _, _, k, n in CASES]


@pytest.mark.skipif(
    os.getenv(GEMM_FLOPS_BENCHMARK_ENV) != "1",
    reason=f"Benchmark is manual-only; set {GEMM_FLOPS_BENCHMARK_ENV}=1 to run",
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 8388608}], indirect=True)
@pytest.mark.parametrize("case_name, dtype, math_fidelity, K, N", CASES, ids=CASE_IDS)
def test_dram_sharded_decode_matmul(device, case_name, dtype, math_fidelity, K, N):
    K_tiles = K // TILE
    N_tiles = N // TILE
    gx, gy = 8, 4
    num_cores = gx * gy
    num_workers_per_dram_bank = 1

    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x < gx or compute_grid.y < gy:
        pytest.skip(f"requested grid {gx}x{gy} exceeds device compute grid {compute_grid.x}x{compute_grid.y}")
    if K_tiles % 32 != 0:
        pytest.skip(f"K_tiles={K_tiles} not divisible by 32")

    num_banks = device.dram_grid_size().x

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

    # Ported verbatim (recipe-wise) from the "V2 DRAM-sharded decode" section of
    # generated/gemm_sanity/decode_gemm_sanity.py: in1 (weight) DRAM WIDTH-SHARDED across the
    # DRAM banks, in0 (activation) L1 WIDTH-SHARDED on the 8x4 compute grid, output L1
    # width-sharded via the bare (grid-less) L1_WIDTH_SHARDED_MEMORY_CONFIG.
    padded_N = math.ceil(N / (TILE * num_banks)) * TILE * num_banks
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    in1_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, [K, padded_N // num_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )

    act_core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    in0_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(act_core_range_set, [M, K // num_cores], ttnn.ShardOrientation.ROW_MAJOR),
    )

    out_mem = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    in0_block_w = K_tiles // 32
    per_core_N = math.ceil(N_tiles / 32)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=per_core_N,
        fused_activation=None,
        num_workers_per_dram_bank=num_workers_per_dram_bank,
    )

    weight_shard_kb = K * (padded_N // num_banks) * bytes_per_elem(dtype) / 1024

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

    assert pcc > 0.99, f"PCC {pcc:.4f} too low for {case_name} M=32 K={K} N={N} grid 8x4"

    tflops = 2 * M * N * K / (us_per_op * 1e-6) / 1e12

    # DRAM bandwidth utilization. tile_bytes_in1 is 2048 B/tile for bfloat16, 1088 B/tile for
    # bfloat8_b; the activation and output tiles are always bfloat16 (2048 B/tile) regardless of
    # the weight dtype. M_tiles = 32/32 = 1 since M is fixed at 32 in this benchmark.
    # 512 GB/s = 8 GDDR6 channels x 64 GB/s per Blackhole chip.
    tile_bytes_in1 = 2048 if dtype == ttnn.bfloat16 else 1088
    M_tiles = M // TILE
    bytes_moved = K_tiles * N_tiles * tile_bytes_in1 + M_tiles * K_tiles * 2048 + M_tiles * N_tiles * 2048
    dram_gbps = bytes_moved / (us_per_op * 1e-6) / 1e9
    dram_bw_util_pct = bytes_moved / (us_per_op * 1e-6) / 512e9 * 100

    logger.info(
        f"DRAM-sharded {case_name} 32x{K}x{N} grid 8x4 workers/bank {num_workers_per_dram_bank}: "
        f"{us_per_op:.2f} us/op (trace), {tflops:.1f} TFLOP/s, {dram_gbps:.0f} GB/s = "
        f"{dram_bw_util_pct:.0f}% of 512 GB/s, pcc {pcc:.4f}"
    )

    csv_path = Path(os.environ["TT_METAL_HOME"]) / "generated" / "dram_sharded_decode_matmul.csv"
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
                    "num_workers_per_dram_bank",
                    "in0_block_w",
                    "per_core_N",
                    "trace_us_per_op",
                    "tflops",
                    "weight_shard_kb",
                    "dram_bw_gbps",
                    "dram_bw_util_pct",
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
                num_workers_per_dram_bank,
                in0_block_w,
                per_core_N,
                f"{us_per_op:.4f}",
                f"{tflops:.2f}",
                f"{weight_shard_kb:.1f}",
                f"{dram_gbps:.1f}",
                f"{dram_bw_util_pct:.1f}",
                f"{pcc:.4f}",
            ]
        )
