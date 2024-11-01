# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

from loguru import logger
import csv
import pytest
import torch
import ttnn
from models.utility_functions import run_for_wormhole_b0, is_grayskull, profiler
from tests.ttnn.utils_for_testing import assert_with_pcc
from pathlib import Path
import os


@run_for_wormhole_b0()
# fmt: off
@pytest.mark.parametrize("height,width,average_time", [
    (1024, 1024, 1),
])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
# fmt: on
def test_benchmark_ttnn_add(device, use_program_cache, height, width, dtype, average_time):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((height, width))
    torch_input_tensor_b = torch.rand((height, width))

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    ttnn.matmul(input_tensor_a, input_tensor_b)
    total_time = 0
    for i in range(3):
        start = time.time()
        output = ttnn.add(input_tensor_a, input_tensor_b)
        end = time.time()
        duration = end - start
        total_time = total_time + duration
        print(f"ttnn.add: {duration} seconds")
        ttnn.to_torch(output)
    total_time = total_time / 3
    assert total_time <= average_time


@run_for_wormhole_b0()
# fmt: off
@pytest.mark.parametrize("m_size,k_size,n_size,average_time", [
    (384, 1024, 1024, 1),
])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
# fmt: on
def test_benchmark_ttnn_matmul(device, use_program_cache, m_size, k_size, n_size, dtype, average_time):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((m_size, k_size))
    torch_input_tensor_b = torch.rand((k_size, n_size))

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    ttnn.matmul(input_tensor_a, input_tensor_b)
    total_time = 0
    for i in range(3):
        start = time.time()
        output = ttnn.matmul(input_tensor_a, input_tensor_b)
        end = time.time()
        duration = end - start
        total_time = total_time + duration
        print(f"ttnn.matmul: {duration} seconds")
        ttnn.to_torch(output)
    total_time = total_time / 3
    assert total_time <= average_time


def find_max_subblock(out_block_h, out_block_w):
    max_product = 0
    best_h = 1
    best_w = 1

    for h in range(1, out_block_h + 1):
        if out_block_h % h == 0:  # h is a divisor of out_block_h
            for w in range(1, out_block_w + 1):
                if out_block_w % w == 0 and h * w <= 8:  # w is a divisor and product condition met
                    if h * w > max_product:
                        max_product = h * w
                        best_h = h
                        best_w = w
    if out_block_w > best_w:
        best_h = 1
    return best_h, best_w, max_product


# This test runs different shapes for matmul_2d, with possibly the best configurations for performance.
#
# The inputs include:
#   - m, k, n: Dimensions of the input tensors.
#   - in0_sharded, out_sharded: Flags indicating whether the in0 (activation) and output tensors are sharded or not.
#   - in0_block_w_div: A parameter to divide an in0 block into multiple chunks, helping to reduce L1 cache usage.

matmul_shapes_bfloat16 = [
    (512, 512, 512, True, True, 1),
    (512, 1024, 1024, True, True, 1),
    (512, 1024, 2048, True, True, 1),
    (1024, 1024, 1024, True, True, 1),
    (1024, 1024, 2048, True, True, 1),
    (1024, 2048, 2048, True, True, 1),
    (2048, 2048, 2048, True, True, 1),
    (2048, 2048, 3072, True, True, 1),
    (2048, 3072, 3072, True, True, 2),
    (3072, 3072, 3072, True, True, 4),
    (3072, 3072, 4096, False, False, 2),
    (3072, 4096, 4096, False, False, 2),
    (4096, 4096, 4096, False, False, 4),
]

matmul_shapes_bfloat8_b = [
    (512, 512, 512, True, True, 1),
    (512, 1024, 1024, True, True, 1),
    (512, 1024, 2048, True, True, 1),
    (1024, 1024, 1024, True, True, 1),
    (1024, 1024, 2048, True, True, 1),
    (1024, 2048, 2048, True, True, 1),
    (2048, 2048, 2048, True, True, 1),
    (2048, 2048, 3072, True, True, 1),
    (2048, 3072, 3072, True, True, 1),
    (3072, 3072, 3072, True, True, 2),
    (3072, 3072, 4096, True, True, 2),
    (3072, 4096, 4096, True, True, 4),
    (4096, 4096, 4096, False, False, 4),
]


@pytest.mark.skip(reason="WH didt hang, need to skip CI and run locally only")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 800768}], indirect=True)
@pytest.mark.parametrize("grid_size", [(8, 8)])
@pytest.mark.parametrize("tile_h", [32])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize(
    "dtype, math_fidelity", [(ttnn.bfloat16, ttnn.MathFidelity.HiFi2), (ttnn.bfloat8_b, ttnn.MathFidelity.LoFi)]
)
@pytest.mark.parametrize("num_warmup_iterations", [5])
@pytest.mark.parametrize("num_measurement_iterations", [100])
def test_matmul_2d_host_perf(
    device,
    grid_size,
    tile_h,
    tile_w,
    dtype,
    math_fidelity,
    num_warmup_iterations,
    num_measurement_iterations,
    use_program_cache,
):
    ENVS = dict(os.environ)
    TT_METAL_HOME = Path(ENVS["TT_METAL_HOME"])
    ARTIFACTS_DIR = TT_METAL_HOME / "generated"
    FILE_NAME = ARTIFACTS_DIR / "matmul_2d_host_perf_report.csv"

    with open(FILE_NAME, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "m",
                "k",
                "n",
                "in0_sharded",
                "out_sharded",
                "dtype",
                "math_fidelity",
                "inference_time_avg (ns)",
                "TFLOPs (avg)",
            ]
        )

        if dtype == ttnn.bfloat16:
            matmul_shapes = matmul_shapes_bfloat16
        else:
            matmul_shapes = matmul_shapes_bfloat8_b
        for m, k, n, in0_sharded, out_sharded, in0_block_w_div in matmul_shapes:
            profiler.clear()

            in0_shape = [1, 1, m, k]
            in1_shape = [1, 1, k, n]

            in0_block_w = k // grid_size[0] // 32 // in0_block_w_div
            out_block_h = m // grid_size[1] // tile_h
            out_block_w = n // grid_size[0] // tile_w
            out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

            in0 = torch.ones(in0_shape).bfloat16()
            in1 = torch.randn(in1_shape).bfloat16()

            if in0_sharded:
                in0_memory_config = ttnn.create_sharded_memory_config(
                    (1, 1, m, k),
                    core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                )
            else:
                in0_memory_config = ttnn.DRAM_MEMORY_CONFIG
            in0_t = ttnn.from_torch(
                in0,
                tile=ttnn.Tile((tile_h, 32)),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=in0_memory_config,
            )
            in1_t = ttnn.from_torch(
                in1,
                tile=ttnn.Tile((32, tile_w)),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                per_core_M=out_block_h,
                per_core_N=out_block_w,
                transpose_mcast=False,
                fused_activation=None,
            )

            if is_grayskull():
                compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                    math_fidelity=math_fidelity,
                    math_approx_mode=True,
                )
            else:
                compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=math_fidelity,
                    math_approx_mode=True,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
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

            output_t = ttnn.matmul(
                in0_t,
                in1_t,
                program_config=program_config,
                memory_config=out_mem_config,
                dtype=dtype,
                compute_kernel_config=compute_kernel_config,
                output_tile=output_tile,
            )

            tid = ttnn.begin_trace_capture(device, cq_id=0)
            output_t = ttnn.matmul(
                in0_t,
                in1_t,
                program_config=program_config,
                memory_config=out_mem_config,
                dtype=dtype,
                compute_kernel_config=compute_kernel_config,
                output_tile=output_tile,
            )
            ttnn.end_trace_capture(device, tid, cq_id=0)

            for iter in range(0, num_warmup_iterations):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

            profiler.start(f"run")
            for iter in range(0, num_measurement_iterations):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            profiler.end(f"run")
            ttnn.DumpDeviceProfiler(device)
            inference_time_avg = profiler.get("run") / num_measurement_iterations
            tflops = 2 * m * k * n / 1e12 / inference_time_avg
            logger.info(f"M*K*N = {m}*{k}*{n} == inference time (avg): {inference_time_avg}, tflops (avg): {tflops}")

            output_tensor = ttnn.to_torch(output_t)
            ttnn.deallocate(output_t)
            ttnn.deallocate(in0_t)
            ttnn.deallocate(in1_t)
            writer.writerow([m, k, n, in0_sharded, out_sharded, dtype, math_fidelity, inference_time_avg, tflops])
