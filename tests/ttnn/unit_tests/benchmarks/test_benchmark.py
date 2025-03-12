# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
from timeit import default_timer as timer

from loguru import logger
import csv
import pytest
import torch
import ttnn
from models.utility_functions import run_for_wormhole_b0, is_grayskull, profiler
from tests.ttnn.utils_for_testing import assert_with_pcc
from pathlib import Path
import os
import sys


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


# This test runs different shapes for matmul_2d, with possibly the best configurations for performance.
#
# The inputs include:
#   - m, k, n: Dimensions of the input tensors.
#   - in0_sharded, out_sharded: Flags indicating whether the in0 (activation) and output tensors are sharded or not.
#   - in0_block_w_div: A parameter to divide an in0 block into multiple chunks, helping to reduce L1 cache usage.
#   - num_out_blocks_h: A parameter to divide an output block into multiple chunks on height dim, helping to reduce L1 cache usage.
#   - num_out_blocks_w: A parameter to divide an output block into multiple chunks on width dim, helping to reduce L1 cache usage.

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


matmul_shapes_bfloat16 = [
    (512, 512, 512, False, False, 1, 1, 1),
]
matmul_configs = [
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi4, False),
]


# @pytest.mark.skip(reason="WH didt hang, need to skip CI and run locally only")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 3855488}], indirect=True)
@pytest.mark.parametrize("grid_size", [(1, 1)])
@pytest.mark.parametrize("tile_h", [32])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("num_warmup_iterations", [5])
@pytest.mark.parametrize("num_measurement_iterations", [100])
def test_matmul_2d_host_perf(
    device,
    grid_size,
    tile_h,
    tile_w,
    num_warmup_iterations,
    num_measurement_iterations,
):
    ENVS = dict(os.environ)
    ERR_FILE_PATH = Path(ENVS["ERR_FILE_PATH"])

    LoFi_cycle = 16
    HiFi2_cycle = LoFi_cycle * 2
    HiFi3_cycle = LoFi_cycle * 3
    HiFi4_cycle = LoFi_cycle * 4

    for dtype, math_fidelity, use_trace in matmul_configs:
        if dtype == ttnn.bfloat16:
            matmul_shapes = matmul_shapes_bfloat16
        for m, k, n, in0_sharded, out_sharded, in0_block_w_div, num_out_blocks_h, num_out_blocks_w in matmul_shapes:
            # scale input size to match BH grid size
            m = (m // 8) * grid_size[1]
            n = (n // 8) * grid_size[0]
            k = (k // 8) * grid_size[0]

            in0_shape = [1, 1, m, k]
            in1_shape = [1, 1, k, n]

            in0_block_w = k // grid_size[0] // 32 // in0_block_w_div
            per_core_M = m // grid_size[1] // tile_h
            per_core_N = n // grid_size[0] // tile_w
            out_block_h = per_core_M // num_out_blocks_h
            out_block_w = per_core_N // num_out_blocks_w
            out_subblock_h, out_subblock_w = get_subblock_sizes(out_block_h, out_block_w, out_sharded)

            logger.info(f"M*K*N = {m}*{k}*{n} out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}")

            in0 = torch.ones(in0_shape).bfloat16()
            in1 = torch.randn(in1_shape).bfloat16()

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

            ttnn.device.EnablePersistentKernelCache()

            max_nops_unpack = 50
            max_nops_math = 50
            max_nops_pack = 50
            max_reps = 10
            COUNTER = 0
            with open(ERR_FILE_PATH, "r") as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
                # Skip the one after the last successful run
                START_COUNT = int(last_line.split()[0]) + 2
                f.close()
                f = open(ERR_FILE_PATH, "a")
                f.write(f"{START_COUNT-1} : HANGED/ABORT\n")
                f.close()
                print(f"Starting from id {START_COUNT}")
                for x in range(0, max_nops_unpack):
                    for y in range(0, max_nops_math):
                        for z in range(0, max_nops_pack):
                            if COUNTER < START_COUNT:
                                COUNTER += 1
                                continue

                            # ttnn.device.DisablePersistentKernelCache()
                            os.environ["TT_NOP_UNPACK"] = str(x)
                            os.environ["TT_NOP_MATH"] = str(y)
                            os.environ["TT_NOP_PACK"] = str(z)
                            start = timer()
                            for iter in range(0, max_reps):
                                output_t = ttnn.matmul(
                                    in0_t,
                                    in1_t,
                                    program_config=program_config,
                                    memory_config=out_mem_config,
                                    dtype=dtype,
                                    compute_kernel_config=compute_kernel_config,
                                    output_tile=output_tile,
                                )
                                ttnn.deallocate(output_t)
                            end = timer()
                            time_taken = end - start
                            runtime_spike = "True" if (time_taken > 0.1) else "False"
                            with open(ERR_FILE_PATH, "a") as f:
                                f.write(
                                    f"{COUNTER} unpack_nop={x} math_nop={y}, pack_nop={z} time={time_taken} runtime_spike={runtime_spike}\n"
                                )
                                f.close()
                            COUNTER += 1
                ttnn.synchronize_device(device)
                ttnn.deallocate(in0_t)
                ttnn.deallocate(in1_t)
