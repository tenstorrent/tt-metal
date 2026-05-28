# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# This test runs different shapes for matmul_2d, with possibly the best configurations for performance.
#
# The inputs include:
#   - m, k, n: Dimensions of the input tensors.
#   - in0_sharded, out_sharded: Flags indicating whether the in0 (activation) and output tensors are sharded or not.
#   - in0_block_w_div: A parameter to divide an in0 block into multiple chunks, helping to reduce L1 cache usage.
#   - num_out_blocks_h: A parameter to divide an output block into multiple chunks on height dim, helping to reduce L1 cache usage.
#   - num_out_blocks_w: A parameter to divide an output block into multiple chunks on width dim, helping to reduce L1 cache usage.
#
# Test is measuring and calculating following performance metrics:
#   - Inference time: The average time taken for inference in nanoseconds.
#   - TFLOPs: The number of tera floating-point operations per second.
#   - Host based utilization: The ratio of ideal cycles to inference cycles, calculated for both user selected and full available grid size.
#   - Device based utilization: The ratio of ideal cycles to TRISC1 kernel duration, calculated for both user selected and full available grid size.
#
# Important notes regarding performance metrics calculation:
#   - If profiler build is not enabled (TT_METAL_DEVICE_PROFILER environment variable is not set) only host based utilization is calculated.
#   - If profiler build is enabled, additionally device based utilization is calculated using TRISC1 kernel duration that is
#     measured and reported by profiler build.
#   - Inference time is measured by host based on repeated(num_measurement_iterations times) execution of matmul operation.
#   - Measured inference time includes all host and device overheads. If profiler build is enabled it can have impact on
#     runtime performance of device and host operations.
#   - TFLOPs is calculated based on the formula: 2 * m * k * n / 1e12 / inference_time_avg
#       - factor 2 comes from fact that we are doing multiplication and addition in matmul operation.
#   - Host based utilization is calculated using formula: ideal_cycle / inference_cycle
#       - ideal_cycle is the number of cycles required to perform matmul on the input tensors:
#         m * k * n / (tile_h * tile_w * tile_h) * (cycle_per_tile / num_cores)
#           - cycle_per_tile is idealistic number of cycles required for Tensix engine to perform matmul on 2 tiles and it is based on
#             math fidelity: for LoFi it is 16, for HiFi2 it is 32, for HiFi3 it is 48 and for HiFi4 it is 64.
#           - num_cores is number of cores in grid that is used to execute matmul operation, 8x8 grid will have 64 cores.
#       - inference_cycle is calculated as: inference_time_avg * device_freq[Hz]
#           - device_freq is fixed value common for specific architecture, for Wormhole B0 it is 1000MHz, for Blackhole it is 1350MHz
#             This value can change and needs to be updated in code until there is python support to read device frequency from device.
#   - Host based utilization uses fixed device frequency value, in reality device frequency can vary due to frequency throttling. Calculated value
#     is worst case scenario for host based utilization since frequency throttling will reduce device frequency and thus increase utilization.
#   - Device based utilization is calculated using formula: ideal_cycle / trisc1_kernel_duration
#       - trisc1_kernel_duration is read from profiler log and it is average duration of TRISC1 kernel in number of cycles.

import time

from loguru import logger
import csv
import pytest
import torch
import ttnn
from models.common.utility_functions import profiler, is_wormhole_b0, is_blackhole
from pathlib import Path
import os
import numpy as np
from tracy.process_device_log import import_log_run_stats
import tracy.device_post_proc_config as device_post_proc_config
from tracy.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG, rm

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


# FP8 path runs with fp32_dest_acc_en=True + bf16 output, which roughly doubles
# per-core L1 footprint vs bf8 (interm0 CB jumps from Float16_b 2048 B/tile to
# Float32 4096 B/tile, plus output CB / L1 sharded buffer go from bf8 1088 B to
# bf16 2048 B per tile). With two outputs alive at the moment of a new matmul's
# allocation (the previous result + the one being allocated), bigger FP8 shapes
# OOM L1. The deallocates between warmup/measurement iterations keep only one
# output alive at a time. Set FP8_EXTRA_DEALLOCATES=False to recover BF8-style
# measurement windows (no host-side deallocate calls inside the timed region)
# at the cost of OOMing on larger FP8 shapes.
FP8_EXTRA_DEALLOCATES = True
# Only apply the deallocates when the scaled M*K*N reaches this threshold;
# smaller shapes fit L1 without them, so we keep their measurement windows
# clean. 3840*4224*4224 corresponds to the (384, 384, 384) shape on an 11x10
# grid -- the smallest shape that actually needs the dealloc on BH.
FP8_EXTRA_DEALLOCATES_MIN_MKN = 3840 * 4224 * 4224


SUBBLOCK_HW_CHOICES = [
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),  # subblock_hw = 8
    (7, 1),
    (1, 7),  # subblock_hw = 7
    (3, 2),
    (2, 3),
    (6, 1),
    (1, 6),  # subblock_hw = 6
    (5, 1),
    (1, 5),  # subblock_hw = 5
    (2, 2),
    (4, 1),
    (1, 4),  # subblock_hw = 4
    (3, 1),
    (1, 3),  # subblock_hw = 3
    (2, 1),
    (1, 2),  # subblock_hw = 2
    (1, 1),  # subblock_hw = 1
]


def get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, out_sharded=False, fp32_dest_acc_en=False):
    for subblock_hw in SUBBLOCK_HW_CHOICES:
        out_subblock_h = subblock_hw[0]
        out_subblock_w = subblock_hw[1]

        if fp32_dest_acc_en:
            if (out_subblock_h * out_subblock_w) > 4:
                continue

        if out_sharded:
            if n_tiles_per_core % out_subblock_w != 0 or out_subblock_h != 1:
                continue

        if m_tiles_per_core % out_subblock_h == 0 and n_tiles_per_core % out_subblock_w == 0:
            return (out_subblock_h, out_subblock_w)

    return (1, 1)


# Get default value of device frequency[MHz] based on architecture.
# Hardcoded values are used until there is python support to read freq
# value from device. Please note that this needs to be updated if device
# runs at different frequency.
def get_device_frequency():
    if is_wormhole_b0():
        return 1000
    elif is_blackhole():
        return 1350
    else:
        return None


def get_profiler_data():
    # Import profiler log file and run perf related statistic calculation
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    data = {}

    # Add device frequency from profiler log
    data["device_freq"] = deviceData["deviceInfo"]["freq"]
    # Add TRISC1-Math kernel average duration time - TRISC1 kernel zones is always present in profiler log
    data["trisc1_kernel_duration"] = deviceData["devices"][0]["cores"]["DEVICE"]["analysis"][
        "device_trisc1_kernel_duration"
    ]["stats"]["Average"]

    return data


# Check if test is run with profiler build enabled.
def get_profiler_build_enabled():
    return os.getenv("TT_METAL_DEVICE_PROFILER") != None


# These configs are all based on a 1x1 compute grid, and will be scaled by the benchmark according to the max grid size
# M will be scaled by Y (num cols), N and K will be scaled by X (num rows)
# (m, k, n, in0_sharded, out_sharded, in0_block_w_div, num_out_blocks_h, num_out_blocks_w)
matmul_shapes_bfloat16 = [
    (64, 64, 64, True, True, 1, 1, 1),
    (64, 128, 128, True, True, 1, 1, 1),
    (64, 128, 256, True, True, 1, 1, 1),
    (128, 128, 128, True, True, 1, 1, 1),
    (128, 128, 256, True, True, 1, 1, 1),
    (128, 256, 256, True, True, 1, 1, 1),
    (256, 256, 256, True, True, 1, 1, 1),
    (256, 256, 384, True, True, 1, 1, 1),
    (256, 384, 384, True, True, 2, 1, 1),
    (384, 384, 384, True, True, 4, 1, 1),
    (384, 384, 512, False, False, 2, 1, 1),
    (384, 512, 512, False, False, 2, 1, 1),
    (416, 320, 320, False, False, 1, 1, 1),  # P150 square: 4160x4160x4160
    (512, 512, 512, False, False, 1, 2, 2),
    (1024, 1024, 1024, False, False, 2, 4, 4),
    (2048, 2048, 2048, False, False, 4, 8, 8),
]

matmul_shapes_bfloat8_b = [
    (64, 64, 64, True, True, 1, 1, 1),
    (64, 128, 128, True, True, 1, 1, 1),
    (64, 128, 256, True, True, 1, 1, 1),
    (128, 128, 128, True, True, 1, 1, 1),
    (128, 128, 256, True, True, 1, 1, 1),
    (128, 256, 256, True, True, 1, 1, 1),
    (256, 256, 256, True, True, 1, 1, 1),
    (256, 256, 384, True, True, 1, 1, 1),
    (256, 384, 384, True, True, 1, 1, 1),
    (384, 384, 384, True, True, 2, 1, 1),
    (384, 384, 512, True, True, 2, 1, 1),
    (416, 320, 320, False, False, 1, 1, 1),
    (512, 512, 512, False, False, 1, 2, 2),
    (1024, 1024, 1024, False, False, 2, 4, 4),
    # (2048, 2048, 2048, False, False, 4, 8, 8),
]

matmul_shapes_bfloat4_b = [
    (64, 64, 64, True, True, 1, 1, 1),
    (64, 128, 128, True, True, 1, 1, 1),
    (64, 128, 256, True, True, 1, 1, 1),
    (128, 128, 128, True, True, 1, 1, 1),
    (128, 128, 256, True, True, 1, 1, 1),
    (128, 256, 256, True, True, 1, 1, 1),
    (256, 256, 256, True, True, 1, 1, 1),
    (256, 256, 384, True, True, 1, 1, 1),
    (256, 384, 384, True, True, 1, 1, 1),
    (384, 384, 384, True, True, 1, 1, 1),
    (384, 384, 512, True, True, 1, 1, 1),
    (384, 512, 512, True, True, 2, 1, 1),
    (416, 320, 320, False, False, 1, 1, 1),
    (512, 512, 512, True, True, 2, 1, 1),
    (1024, 1024, 1024, False, False, 2, 2, 2),
    (2048, 2048, 2048, False, False, 4, 4, 4),
]

# FP8 shapes: tuned separately from BF8 because the FP8 path runs with
# fp32_dest_acc_en=True (LLK requirement) and packer_l1_acc=False, which forces
# the interm0 CB to Float32 (4096 B/tile) instead of Float16_b (2048 B/tile).
# Combined with the bf16 output CB/L1 buffer (also 2x bf8) this nearly doubles
# the per-core L1 footprint vs BF8. For shapes where the per-core out block
# (per_core_M*per_core_N) is large enough that interm0_CB = out_block_tiles*4096
# would push the static CB region past ~1.0 MB, bump num_out_blocks_h/w to split
# the output block. Per-core interm0 = out_block_h * out_block_w * 4096; we keep
# this under ~400 KB so total CBs + sharded L1 buffers fit BH's ~1.4 MB usable L1.
matmul_shapes_fp8_e4m3 = [
    (64, 64, 64, True, True, 1, 1, 1),
    (64, 128, 128, True, True, 1, 1, 1),
    (64, 128, 256, True, True, 1, 1, 1),
    (128, 128, 128, True, True, 1, 1, 1),
    (128, 128, 256, True, True, 1, 1, 1),
    (128, 256, 256, True, True, 1, 1, 1),
    (256, 256, 256, True, True, 1, 1, 1),
    (256, 256, 384, True, True, 1, 1, 1),
    (256, 384, 384, True, True, 1, 1, 1),
    (384, 384, 384, True, True, 2, 1, 1),
    # FP8 split: per_core 12x16 = 192 tiles -> interm0 fp32 = 768 KB OOMs L1.
    # Split M (not N): matmul_device_operation.cpp:1363 requires
    # out_block_w == per_core_N || out_block_h == 1, so num_out_blocks_w>1 with
    # out_block_h>1 is rejected. num_out_blocks_h=2 -> out_block = 6x16 = 96
    # tiles, interm0 = 384 KB. Also halves in0_CB since in0_block_h = out_block_h.
    (384, 384, 512, True, True, 2, 2, 1),
    (416, 320, 320, False, False, 1, 1, 1),
    (512, 512, 512, False, False, 1, 2, 2),
    (1024, 1024, 1024, False, False, 2, 4, 4),
    # (2048, 2048, 2048, False, False, 4, 8, 8),
]

# (dtype, math_fidelity, use_trace)
# Only BF8 and FP8 are active; other dtypes are commented out per the FP8 vs BF8
# experiment scope. FP8 entries use ttnn.fp8_e4m3 as a sentinel; inside the test
# body inputs are uploaded as ttnn.uint8 (see test_fp8_matmul.py::test_fp8_matmul_perf
# for the same hack) and the matmul code remaps UInt8 input CBs to Fp8_e4m3.
matmul_configs = [
    # (ttnn.bfloat16, ttnn.MathFidelity.HiFi2, False),
    # (ttnn.bfloat16, ttnn.MathFidelity.HiFi4, False),
    # (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, False),
    # (ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, False),
    # (ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, False),
    # (ttnn.bfloat16, ttnn.MathFidelity.HiFi2, True),
    # (ttnn.bfloat16, ttnn.MathFidelity.HiFi4, True),
    # (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True),
    # (ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, True),
    # (ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True),
    (ttnn.fp8_e4m3, ttnn.MathFidelity.LoFi, False),
    (ttnn.fp8_e4m3, ttnn.MathFidelity.LoFi, True),
]


# @pytest.mark.skip(reason="Benchmark is not intended to be run as part of CI and can be manually run locally")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 3855488}], indirect=True)
@pytest.mark.parametrize("tile_h", [32])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("num_warmup_iterations", [5])
@pytest.mark.parametrize("num_measurement_iterations", [50])
def test_matmul_2d_host_perf(
    device,
    grid_size,
    tile_h,
    tile_w,
    num_warmup_iterations,
    num_measurement_iterations,
):
    ENVS = dict(os.environ)
    TT_METAL_HOME = Path(ENVS["TT_METAL_HOME"])
    ARTIFACTS_DIR = TT_METAL_HOME / "generated"
    FILE_NAME = ARTIFACTS_DIR / "matmul_2d_host_perf_report.csv"

    # Calculate device utilization only when profiler build is enabled.
    # If profiler build is enabled, kernel execution time is available in profiler log by default.
    calc_device_utilization = get_profiler_build_enabled()

    # Get maximum available grid size for compute from device
    compute_grid_size = device.compute_with_storage_grid_size()
    # If user did not specify grid size, use maximum available grid size for compute
    if grid_size is None:
        grid_size = (compute_grid_size.x, compute_grid_size.y)
    # Check if requested grid size is within available compute grid size, skip test if not
    if compute_grid_size.y < grid_size[1] or compute_grid_size.x < grid_size[0]:
        pytest.skip(
            f"Skipping test as requested compute grid size {grid_size} exceeds available compute grid {compute_grid_size}"
        )

    LoFi_cycle = 16
    HiFi2_cycle = LoFi_cycle * 2
    HiFi3_cycle = LoFi_cycle * 3
    HiFi4_cycle = LoFi_cycle * 4

    with open(FILE_NAME, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = [
            "m",
            "k",
            "n",
            "use_trace",
            "grid_size",
            "in0_sharded",
            "out_sharded",
            "in0_storage_type",
            "in1_storage_type",
            "out_storage_type",
            "dtype",
            "math_fidelity",
            "inference_time_avg [ns]",
            "TFLOPs (avg)",
            f"Host based utilization[%] (vs user selected grid {grid_size[0]}x{grid_size[1]})",
            f"Host based utilization[%] (vs full available grid {compute_grid_size.x}x{compute_grid_size.y})",
        ]
        if calc_device_utilization:
            header.extend(
                [
                    f"Device based utilization[%] (vs user selected grid {grid_size[0]}x{grid_size[1]})",
                    f"Device based utilization[%] (vs full available grid {compute_grid_size.x}x{compute_grid_size.y})",
                ]
            )
        writer.writerow(header)

        for dtype, math_fidelity, use_trace in matmul_configs:
            if dtype == ttnn.bfloat16:
                matmul_shapes = matmul_shapes_bfloat16
            elif dtype == ttnn.bfloat8_b:
                matmul_shapes = matmul_shapes_bfloat8_b
            elif dtype == ttnn.bfloat4_b:
                matmul_shapes = matmul_shapes_bfloat4_b
            elif dtype == ttnn.fp8_e4m3:
                matmul_shapes = matmul_shapes_fp8_e4m3

            # FP8 hack: ship inputs as UInt8 tensors of raw FP8 e4m3 bytes. The matmul
            # device code remaps UInt8 input CBs to Fp8_e4m3 (see matmul_utilities.hpp)
            # so the LLK runs the real FP8 path. Output stays bf16 because the FP8
            # output writeback isn't on the golden path yet. Compute config also flips
            # to fp32_dest / no packer_l1_acc to match the FP8 LLK gtest config.
            is_fp8 = dtype == ttnn.fp8_e4m3
            input_dtype = ttnn.uint8 if is_fp8 else dtype
            output_dtype = ttnn.bfloat16 if is_fp8 else dtype
            # FP8 forces fp32_dest_acc_en=True (halves dest -> caps subblock H*W at 4).
            # Subblock selection must know this or it picks shapes that overflow dest.
            fp32_dest_acc_en = is_fp8
            packer_l1_acc = not is_fp8
            for m, k, n, in0_sharded, out_sharded, in0_block_w_div, num_out_blocks_h, num_out_blocks_w in matmul_shapes:
                profiler.clear()

                # Scale the input shapes by the grid size
                m = m * grid_size[1]
                k = k * grid_size[0]
                n = n * grid_size[0]

                in0_shape = [1, 1, m, k]
                in1_shape = [1, 1, k, n]

                in0_block_w = k // grid_size[0] // 32 // in0_block_w_div
                per_core_M = m // grid_size[1] // tile_h
                per_core_N = n // grid_size[0] // tile_w
                out_block_h = per_core_M // num_out_blocks_h
                out_block_w = per_core_N // num_out_blocks_w
                out_subblock_h, out_subblock_w = get_subblock_sizes(
                    out_block_h, out_block_w, out_sharded, fp32_dest_acc_en=fp32_dest_acc_en
                )

                logger.info(f"M*K*N = {m}*{k}*{n} out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}")

                logger.info(
                    f"Allocating host torch tensors (in0_shape={in0_shape}, in1_shape={in1_shape}, is_fp8={is_fp8})"
                )
                if is_fp8:
                    # Use uint8 raw bytes as the FP8 e4m3 payload. Numerical output will
                    # be garbage but kernel dispatch and timing are unaffected.
                    in0 = torch.ones(in0_shape, dtype=torch.uint8) * 56
                    in1 = torch.randint(0, 256, in1_shape, dtype=torch.uint8)
                else:
                    in0 = torch.ones(in0_shape).bfloat16()
                    in1 = torch.randn(in1_shape).bfloat16()
                logger.info("Host torch tensors allocated")

                if in0_sharded:
                    in0_storage_type = "L1"
                else:
                    in0_storage_type = "DRAM"

                in1_storage_type = "DRAM"

                if out_sharded:
                    out_storage_type = "L1"
                else:
                    out_storage_type = "DRAM"

                if in0_sharded:
                    in0_memory_config = ttnn.create_sharded_memory_config(
                        (1, 1, m, k),
                        core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
                        strategy=ttnn.ShardStrategy.BLOCK,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    )
                else:
                    in0_memory_config = ttnn.DRAM_MEMORY_CONFIG
                logger.info(
                    f"Uploading in0 to device (dtype={input_dtype}, memory_config={'sharded' if in0_sharded else 'DRAM'})"
                )
                in0_t = ttnn.from_torch(
                    in0,
                    tile=ttnn.Tile((tile_h, 32)),
                    dtype=input_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=in0_memory_config,
                )
                logger.info("in0 uploaded")

                logger.info(f"Uploading in1 to device (dtype={input_dtype}, memory_config=DRAM)")
                in1_t = ttnn.from_torch(
                    in1,
                    tile=ttnn.Tile((32, tile_w)),
                    dtype=input_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                logger.info("in1 uploaded")

                program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=grid_size,
                    in0_block_w=in0_block_w,
                    out_subblock_h=out_subblock_h,
                    out_subblock_w=out_subblock_w,
                    out_block_h=out_block_h,
                    out_block_w=out_block_w,
                    per_core_M=per_core_M,
                    per_core_N=per_core_N,
                    transpose_mcast=False,
                    fused_activation=None,
                )

                compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=math_fidelity,
                    math_approx_mode=True,
                    # FP8 LLK requires FP32 dest accumulator; packer_l1_acc is not wired
                    # up for the FP8 path. Matches the FP8 gtest / test_fp8_matmul.py.
                    fp32_dest_acc_en=fp32_dest_acc_en,
                    packer_l1_acc=packer_l1_acc,
                    throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
                )

                if out_sharded:
                    out_mem_config = ttnn.MemoryConfig(
                        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                        buffer_type=ttnn.BufferType.L1,
                    )
                else:
                    out_mem_config = ttnn.DRAM_MEMORY_CONFIG

                if out_sharded:
                    output_tile = ttnn.Tile([tile_h, 32]) if tile_h <= 16 else ttnn.Tile([tile_h, tile_w])
                else:
                    output_tile = ttnn.Tile([tile_h, tile_w])

                # output_t = ttnn.matmul(
                #     in0_t,
                #     in1_t,
                #     program_config=program_config,
                #     memory_config=out_mem_config,
                #     dtype=output_dtype,
                #     compute_kernel_config=compute_kernel_config,
                #     output_tile=output_tile,
                # )

                # Per-iter deallocates gated on FP8_EXTRA_DEALLOCATES + the M*K*N
                # threshold (top of file). Keeps only one output buffer alive
                # across the next matmul's allocation so larger FP8 shapes fit
                # L1; smaller shapes skip the deallocates so their measurement
                # window stays clean.
                fp8_dealloc = is_fp8 and FP8_EXTRA_DEALLOCATES and m * k * n >= FP8_EXTRA_DEALLOCATES_MIN_MKN
                for iter in range(0, num_warmup_iterations):
                    # logger.info(f"Warmup iteration {iter} started")
                    if fp8_dealloc and iter > 0:
                        ttnn.deallocate(output_t)
                    output_t = ttnn.matmul(
                        in0_t,
                        in1_t,
                        program_config=program_config,
                        memory_config=out_mem_config,
                        dtype=output_dtype,
                        compute_kernel_config=compute_kernel_config,
                        output_tile=output_tile,
                    )
                    # logger.info(f"Warmup iteration {iter} completed")

                if calc_device_utilization:
                    # Clear profiler log and read profiler data after warmup iterations
                    ttnn.ReadDeviceProfiler(device)
                    rm(profiler_log_path)

                if fp8_dealloc:
                    ttnn.deallocate(output_t)

                # Synchronize device to ensure all warmup iterations are completed and device is in clean state
                ttnn.synchronize_device(device)

                if use_trace:
                    tid = ttnn.begin_trace_capture(device, cq_id=0)
                    # logger.info(f"Trace capture started")
                    for iter in range(0, num_measurement_iterations):
                        if fp8_dealloc and iter > 0:
                            ttnn.deallocate(output_t)
                        output_t = ttnn.matmul(
                            in0_t,
                            in1_t,
                            program_config=program_config,
                            memory_config=out_mem_config,
                            dtype=output_dtype,
                            compute_kernel_config=compute_kernel_config,
                            output_tile=output_tile,
                        )
                    ttnn.end_trace_capture(device, tid, cq_id=0)
                    # logger.info(f"Trace capture completed")

                    profiler.start(f"run")
                    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(device)
                    profiler.end(f"run")
                    ttnn.release_trace(device, tid)
                    # logger.info(f"Trace execution completed")
                else:
                    profiler.start(f"run")
                    for iter in range(0, num_measurement_iterations):
                        # logger.info(f"Measurement iteration {iter} started")
                        if fp8_dealloc and iter > 0:
                            ttnn.deallocate(output_t)
                        output_t = ttnn.matmul(
                            in0_t,
                            in1_t,
                            program_config=program_config,
                            memory_config=out_mem_config,
                            dtype=output_dtype,
                            compute_kernel_config=compute_kernel_config,
                            output_tile=output_tile,
                        )
                    ttnn.synchronize_device(device)
                    profiler.end(f"run")

                if calc_device_utilization:
                    # Read profiler log data
                    ttnn.ReadDeviceProfiler(device)
                    profiler_data = get_profiler_data()
                    trisc1_kernel_duration = profiler_data["trisc1_kernel_duration"]

                # Read device frequency
                device_freq = get_device_frequency()

                inference_time_avg = profiler.get("run") / num_measurement_iterations
                tflops = 2 * m * k * n / 1e12 / inference_time_avg
                if math_fidelity == ttnn.MathFidelity.LoFi:
                    cycle_per_tile = LoFi_cycle
                elif math_fidelity == ttnn.MathFidelity.HiFi2:
                    cycle_per_tile = HiFi2_cycle
                elif math_fidelity == ttnn.MathFidelity.HiFi3:
                    cycle_per_tile = HiFi3_cycle
                elif math_fidelity == ttnn.MathFidelity.HiFi4:
                    cycle_per_tile = HiFi4_cycle
                num_cores_user_grid = grid_size[0] * grid_size[1]
                num_cores_full_grid = compute_grid_size.x * compute_grid_size.y
                ideal_cycle_full_grid = m * k * n / tile_h / tile_w / 32 * cycle_per_tile / num_cores_full_grid
                ideal_cycle_user_grid = m * k * n / tile_h / tile_w / 32 * cycle_per_tile / num_cores_user_grid
                inference_cycle = inference_time_avg * device_freq * 1e6
                utilization_full_grid = ideal_cycle_full_grid / inference_cycle
                utilization_user_grid = ideal_cycle_user_grid / inference_cycle
                if calc_device_utilization:
                    # Calculate device utilization based on TRISC1 kernel duration
                    utilization_full_grid_device = ideal_cycle_full_grid / np.mean(trisc1_kernel_duration)
                    utilization_user_grid_device = ideal_cycle_user_grid / np.mean(trisc1_kernel_duration)

                logger.info(
                    f"M*K*N = {m}*{k}*{n} == inference time (avg): {inference_time_avg}, tflops (avg): {tflops}, utilization (vs user selected grid {grid_size[0]}x{grid_size[1]}): {utilization_user_grid * 100:.2f}%, utilization (vs full available grid {compute_grid_size.x}x{compute_grid_size.y}): {utilization_full_grid * 100:.2f}%"
                )

                output_tensor = ttnn.to_torch(output_t)
                ttnn.deallocate(output_t)
                ttnn.deallocate(in0_t)
                ttnn.deallocate(in1_t)
                csv_data = [
                    m,
                    k,
                    n,
                    f"{True}" if use_trace else f"{False}",
                    grid_size,
                    in0_sharded,
                    out_sharded,
                    in0_storage_type,
                    in1_storage_type,
                    out_storage_type,
                    dtype,
                    math_fidelity,
                    f"{inference_time_avg * 1e9:.2f}",
                    f"{tflops:.2f}",
                    f"{utilization_user_grid * 100:.2f}",
                    f"{utilization_full_grid * 100:.2f}",
                ]
                if calc_device_utilization:
                    csv_data.extend(
                        [
                            f"{utilization_user_grid_device * 100:.2f}",
                            f"{utilization_full_grid_device * 100:.2f}",
                        ]
                    )
                writer.writerow(csv_data)
                file.flush()


matmul_shapes_oob = [
    (64, 64, 64),
    (64, 128, 128),
    (64, 128, 256),
    (128, 128, 128),
    (128, 128, 256),
    (128, 256, 256),
    (256, 256, 256),
    (256, 256, 384),
    # Shapes from here down OOM the FP8 OOB path: the default matmul auto-picker
    # doesn't yet account for fp32_dest_acc_en bumping the interm0 CB to Float32
    # (4096 B/tile vs Float16_b 2048 B/tile), so per-core blocks of ~96+ tiles
    # blow the static CB region. Keep them commented out while the FP8 OOB
    # experience is the focus; re-enable for the BF8 baseline if/when that gets
    # turned back on in matmul_configs_oob.
    # (256, 384, 384),
    # (384, 384, 384),
    # (384, 384, 512),
    # (384, 512, 512),
    # (512, 512, 512),
]

# Only BF8 and FP8 are active; other dtypes are commented out per the FP8 vs BF8
# experiment scope. FP8 entries use ttnn.fp8_e4m3 as a sentinel; inside the test
# body inputs are uploaded as ttnn.uint8 (see test_fp8_matmul.py::test_fp8_matmul_perf
# for the same hack) and the matmul code remaps UInt8 input CBs to Fp8_e4m3.
matmul_configs_oob = [
    # (ttnn.bfloat16, False),
    # (ttnn.bfloat16, True),
    (ttnn.bfloat8_b, False),
    (ttnn.bfloat8_b, True),
    # (ttnn.bfloat4_b, False),
    # (ttnn.bfloat4_b, True),
    (ttnn.fp8_e4m3, False),
    (ttnn.fp8_e4m3, True),
]


# @pytest.mark.skip(reason="Benchmark is not intended to be run as part of CI and can be manually run locally")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 3855488}], indirect=True)
@pytest.mark.parametrize("tile_h", [32])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("num_warmup_iterations", [1])
@pytest.mark.parametrize("num_measurement_iterations", [5])
def test_matmul_2d_host_perf_out_of_box(
    device,
    grid_size,
    tile_h,
    tile_w,
    num_warmup_iterations,
    num_measurement_iterations,
):
    ENVS = dict(os.environ)
    TT_METAL_HOME = Path(ENVS["TT_METAL_HOME"])
    ARTIFACTS_DIR = TT_METAL_HOME / "generated"
    FILE_NAME = ARTIFACTS_DIR / "matmul_2d_host_perf_out_of_box_report.csv"

    # Calculate device utilization only when profiler build is enabled.
    # If profiler build is enabled, kernel execution time is available in profiler log by default.
    calc_device_utilization = get_profiler_build_enabled()

    # Get maximum available grid size for compute from device
    compute_grid_size = device.compute_with_storage_grid_size()
    # If user did not specify grid size, use maximum available grid size for compute
    if grid_size is None:
        grid_size = (compute_grid_size.x, compute_grid_size.y)
    # Check if requested grid size is within available compute grid size, skip test if not
    if compute_grid_size.y < grid_size[1] or compute_grid_size.x < grid_size[0]:
        pytest.skip(
            f"Skipping test as requested compute grid size {grid_size} exceeds available compute grid {compute_grid_size}"
        )

    LoFi_cycle = 16
    HiFi2_cycle = LoFi_cycle * 2
    HiFi3_cycle = LoFi_cycle * 3
    HiFi4_cycle = LoFi_cycle * 4

    with open(FILE_NAME, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = [
            "m",
            "k",
            "n",
            "use_trace",
            "grid_size",
            "in0_storage_type",
            "in1_storage_type",
            "out_storage_type",
            "dtype",
            "math_fidelity",
            "inference_time_avg [ns]",
            "TFLOPs (avg)",
            f"Host based utilization[%] (vs user selected grid {grid_size[0]}x{grid_size[1]})",
            f"Host based utilization[%] (vs full available grid {compute_grid_size.x}x{compute_grid_size.y})",
        ]
        if calc_device_utilization:
            header.extend(
                [
                    f"Device based utilization[%] (vs user selected grid {grid_size[0]}x{grid_size[1]})",
                    f"Device based utilization[%] (vs full available grid {compute_grid_size.x}x{compute_grid_size.y})",
                ]
            )
        writer.writerow(header)

        for dtype, use_trace in matmul_configs_oob:
            matmul_shapes = matmul_shapes_oob
            if dtype == ttnn.bfloat16:
                math_fidelity = ttnn.MathFidelity.HiFi2
            elif dtype == ttnn.bfloat8_b:
                math_fidelity = ttnn.MathFidelity.LoFi
            elif dtype == ttnn.bfloat4_b:
                math_fidelity = ttnn.MathFidelity.LoFi
            elif dtype == ttnn.fp8_e4m3:
                math_fidelity = ttnn.MathFidelity.LoFi

            # FP8 hack: ship inputs as UInt8 tensors of raw FP8 e4m3 bytes. The matmul
            # device code remaps UInt8 input CBs to Fp8_e4m3 (see matmul_utilities.hpp)
            # so the LLK runs the real FP8 path. Output stays bf16 because the FP8
            # output writeback isn't on the golden path yet.
            is_fp8 = dtype == ttnn.fp8_e4m3
            input_dtype = ttnn.uint8 if is_fp8 else dtype
            output_dtype = ttnn.bfloat16 if is_fp8 else dtype
            for m, k, n in matmul_shapes:
                profiler.clear()

                # Scale the input shapes by the grid size
                m = m * grid_size[1]
                k = k * grid_size[0]
                n = n * grid_size[0]

                in0_shape = [1, 1, m, k]
                in1_shape = [1, 1, k, n]

                if is_fp8:
                    # uint8 raw bytes as FP8 e4m3 payload; output values are garbage but
                    # dispatch/timing are unaffected.
                    in0 = torch.ones(in0_shape, dtype=torch.uint8) * 56
                    in1 = torch.randint(0, 256, in1_shape, dtype=torch.uint8)
                else:
                    in0 = torch.ones(in0_shape).bfloat16()
                    in1 = torch.randn(in1_shape).bfloat16()

                in0_storage_type = "DRAM"
                in1_storage_type = "DRAM"
                out_storage_type = "DRAM"

                in0_t = ttnn.from_torch(
                    in0,
                    tile=ttnn.Tile((tile_h, 32)),
                    dtype=input_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                in1_t = ttnn.from_torch(
                    in1,
                    tile=ttnn.Tile((32, tile_w)),
                    dtype=input_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                # FP8 LLK requires fp32 dest acc and no packer_l1_acc. For the non-FP8
                # OOB path we pass compute_kernel_config=None to keep the @-operator
                # default behavior (identical to `in0_t @ in1_t`).
                if is_fp8:
                    fp8_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                        math_fidelity=math_fidelity,
                        math_approx_mode=True,
                        fp32_dest_acc_en=True,
                        packer_l1_acc=False,
                        throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
                    )

                    def run_matmul():
                        return ttnn.matmul(
                            in0_t,
                            in1_t,
                            dtype=output_dtype,
                            compute_kernel_config=fp8_compute_kernel_config,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )

                else:

                    def run_matmul():
                        return in0_t @ in1_t

                # Per-iter deallocates gated on FP8_EXTRA_DEALLOCATES + the M*K*N
                # threshold (top of file). Same rationale as in
                # test_matmul_2d_host_perf above.
                fp8_dealloc = is_fp8 and FP8_EXTRA_DEALLOCATES and m * k * n >= FP8_EXTRA_DEALLOCATES_MIN_MKN
                output_t = run_matmul()

                for iter in range(0, num_warmup_iterations):
                    if fp8_dealloc:
                        ttnn.deallocate(output_t)
                    output_t = run_matmul()

                if calc_device_utilization:
                    # Clear profiler log and read profiler data after warmup iterations
                    ttnn.ReadDeviceProfiler(device)
                    rm(profiler_log_path)

                if fp8_dealloc:
                    ttnn.deallocate(output_t)

                # Synchronize device to ensure all warmup iterations are completed and device is in clean state
                ttnn.synchronize_device(device)

                if use_trace:
                    tid = ttnn.begin_trace_capture(device, cq_id=0)
                    for iter in range(0, num_measurement_iterations):
                        if fp8_dealloc and iter > 0:
                            ttnn.deallocate(output_t)
                        output_t = run_matmul()
                    ttnn.end_trace_capture(device, tid, cq_id=0)
                    profiler.start(f"run")
                    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(device)
                    profiler.end(f"run")
                    ttnn.release_trace(device, tid)
                else:
                    profiler.start(f"run")
                    for iter in range(0, num_measurement_iterations):
                        if fp8_dealloc and iter > 0:
                            ttnn.deallocate(output_t)
                        output_t = run_matmul()
                    ttnn.synchronize_device(device)
                    profiler.end(f"run")

                if calc_device_utilization:
                    # Read profiler log data
                    ttnn.ReadDeviceProfiler(device)
                    profiler_data = get_profiler_data()
                    trisc1_kernel_duration = profiler_data["trisc1_kernel_duration"]

                # Read device frequency
                device_freq = get_device_frequency()

                inference_time_avg = profiler.get("run") / num_measurement_iterations
                tflops = 2 * m * k * n / 1e12 / inference_time_avg
                if math_fidelity == ttnn.MathFidelity.LoFi:
                    cycle_per_tile = LoFi_cycle
                elif math_fidelity == ttnn.MathFidelity.HiFi2:
                    cycle_per_tile = HiFi2_cycle
                elif math_fidelity == ttnn.MathFidelity.HiFi3:
                    cycle_per_tile = HiFi3_cycle
                elif math_fidelity == ttnn.MathFidelity.HiFi4:
                    cycle_per_tile = HiFi4_cycle
                num_cores_user_grid = grid_size[0] * grid_size[1]
                num_cores_full_grid = compute_grid_size.x * compute_grid_size.y
                ideal_cycle_full_grid = m * k * n / tile_h / tile_w / 32 * cycle_per_tile / num_cores_full_grid
                ideal_cycle_user_grid = m * k * n / tile_h / tile_w / 32 * cycle_per_tile / num_cores_user_grid
                inference_cycle = inference_time_avg * device_freq * 1e6
                utilization_full_grid = ideal_cycle_full_grid / inference_cycle
                utilization_user_grid = ideal_cycle_user_grid / inference_cycle
                if calc_device_utilization:
                    # Calculate device utilization based on TRISC1 kernel duration
                    utilization_full_grid_device = ideal_cycle_full_grid / np.mean(trisc1_kernel_duration)
                    utilization_user_grid_device = ideal_cycle_user_grid / np.mean(trisc1_kernel_duration)
                logger.info(
                    f"M*K*N = {m}*{k}*{n} == inference time (avg): {inference_time_avg}, tflops (avg): {tflops}, utilization (vs user selected grid {grid_size[0]}x{grid_size[1]}): {utilization_user_grid * 100:.2f}%, utilization (vs full available grid {compute_grid_size.x}x{compute_grid_size.y}): {utilization_full_grid * 100:.2f}%"
                )

                output_tensor = ttnn.to_torch(output_t)
                ttnn.deallocate(output_t)
                ttnn.deallocate(in0_t)
                ttnn.deallocate(in1_t)
                csv_data = [
                    m,
                    k,
                    n,
                    f"{True}" if use_trace else f"{False}",
                    grid_size,
                    in0_storage_type,
                    in1_storage_type,
                    out_storage_type,
                    dtype,
                    math_fidelity,
                    f"{inference_time_avg * 1e9:.2f}",
                    f"{tflops:.2f}",
                    f"{utilization_user_grid * 100:.2f}",
                    f"{utilization_full_grid * 100:.2f}",
                ]
                if calc_device_utilization:
                    csv_data.extend(
                        [
                            f"{utilization_user_grid_device * 100:.2f}",
                            f"{utilization_full_grid_device * 100:.2f}",
                        ]
                    )
                writer.writerow(csv_data)
                file.flush()
