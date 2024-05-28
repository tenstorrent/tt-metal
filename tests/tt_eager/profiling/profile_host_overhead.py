# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import tt_lib
import time
import statistics
from models.utility_functions import torch2tt_tensor
from tests.tt_eager.profiling import ops_for_profiling

test_sweep_args = [
    (
        (1, 2, 1024, 1024),
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.TILE,
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
    ),
    (
        (1, 4, 1024, 1024),
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.TILE,
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
    ),
]

all_num_call_to_stack = [1, 3]  # For 10 and more test  execution spills to dispatch
num_repeats = 10


def measure_host_overhead(op_func, device, num_call_to_stack):
    start_time = time.time()
    for _ in range(num_call_to_stack):
        op_func()

    tt_lib.device.Synchronize(device)

    duration = 1000 * (time.time() - start_time)
    total_op_time = duration / num_call_to_stack
    logger.info(f"{num_call_to_stack} calls and Synchronize after {duration:.2f}ms ({total_op_time:.2f}ms per call)")

    start_time = time.time()
    for _ in range(num_call_to_stack):
        op_func()

    duration = 1000 * (time.time() - start_time)
    overhead_ms = duration / num_call_to_stack
    logger.info(f"{num_call_to_stack} calls without Synchronize {duration:.2f}ms ({overhead_ms:.2f}ms per call)")

    start_time = time.time()
    tt_lib.device.Synchronize(device)
    duration = 1000 * (time.time() - start_time)
    duration_per_call = duration / num_call_to_stack

    logger.info(f"Synchronize {duration:.2f}ms ({duration_per_call:.2f}ms per call)")
    return overhead_ms, total_op_time


def measure_host_overhead_binary(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    device,
    op,
    num_call_to_stack,
    num_repeats,
    bcast=False,
    bcast_dim=tt_lib.tensor.BcastOpDim.W,
    norm_shapes=False,
    embeddings_shapes=False,
):
    input_shape_2 = input_shape

    if bcast:
        if bcast_dim == tt_lib.tensor.BcastOpDim.W:
            input_shape_2 = [input_shape[-4], input_shape[-3], input_shape[-2], 32]
        elif bcast_dim == tt_lib.tensor.BcastOpDim.H:
            input_shape_2 = [input_shape[-4], input_shape[-3], 32, input_shape[-1]]
        else:
            input_shape_2 = [input_shape[-4], input_shape[-3], 32, 32]

    if embeddings_shapes:
        input_shape = (input_shape[0], 1, 1, input_shape[-1])
        input_shape_2 = (input_shape_2[0], 1, 1, input_shape_2[-1])

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    y = torch.Tensor(size=input_shape_2).uniform_(-100, 100)

    x = torch2tt_tensor(x, device, dlayout, in_mem_config, dtype)
    y = torch2tt_tensor(y, device, dlayout, in_mem_config, dtype)

    def op_func():
        op(x, y)

    result_overhead = []
    result_op = []

    for _ in range(num_repeats):
        overhead_ms, total_op_time = measure_host_overhead(op_func, device, num_call_to_stack)
        result_overhead.append(overhead_ms)
        result_op.append(total_op_time)

    return result_overhead, result_op


def measure_host_overhead_unary(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    device,
    op,
    num_call_to_stack,
    num_repeats,
    bcast=False,
    bcast_dim=tt_lib.tensor.BcastOpDim.W,
    norm_shapes=False,
    embeddings_shapes=False,
):
    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    x = torch2tt_tensor(x, device, dlayout, in_mem_config, dtype)

    def op_func():
        op(x)

    result_overhead = []
    result_op = []

    for _ in range(num_repeats):
        overhead_ms, total_op_time = measure_host_overhead(op_func, device, num_call_to_stack)
        result_overhead.append(overhead_ms)
        result_op.append(total_op_time)

    return result_overhead, result_op


def measure_host_overhead_ternary(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    device,
    op,
    num_call_to_stack,
    num_repeats,
    bcast=False,
    bcast_dim=tt_lib.tensor.BcastOpDim.W,
    norm_shapes=False,
    embeddings_shapes=False,
):
    input_shape_2 = input_shape

    if norm_shapes:
        input_shape_2 = [input_shape[0], input_shape[1], 32, input_shape[3]]

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    y = torch.Tensor(size=input_shape_2).uniform_(-100, 100)
    z = torch.Tensor(size=input_shape_2).uniform_(-100, 100)

    x = torch2tt_tensor(x, device, dlayout, in_mem_config, dtype)
    y = torch2tt_tensor(y, device, dlayout, in_mem_config, dtype)
    z = torch2tt_tensor(z, device, dlayout, in_mem_config, dtype)

    def op_func():
        op(x, y, z)

    result_overhead = []
    result_op = []

    for _ in range(num_repeats):
        overhead_ms, total_op_time = measure_host_overhead(op_func, device, num_call_to_stack)
        result_overhead.append(overhead_ms)
        result_op.append(total_op_time)

    return result_overhead, result_op


def run_measure_host_overhead(op, device, text_file, measuring_func):
    results_overhead = []
    results_op = []

    for input_shape, dtype, dlayout, in_mem_config, out_mem_config in test_sweep_args:
        logger.info("")
        logger.info(f"Profiling op {op['name']} for input shape {input_shape}")

        if "layout" in op and op["layout"] == "ROW_MAJOR":
            dlayout = tt_lib.tensor.Layout.ROW_MAJOR

        norm_shapes = False if "norm_shapes" not in op else op["norm_shapes"]
        embeddings_shapes = False if "embeddings_shapes" not in op else op["embeddings_shapes"]
        bcast = False if "bcast" not in op else op["bcast"]
        bcast_dim = tt_lib.tensor.BcastOpDim.W if "bcast_dim" not in op else op["bcast_dim"]

        # Warmup
        measuring_func(
            input_shape,
            dtype,
            dlayout,
            in_mem_config,
            out_mem_config,
            device,
            op["op"],
            1,
            1,
            bcast=bcast,
            bcast_dim=bcast_dim,
            norm_shapes=norm_shapes,
            embeddings_shapes=embeddings_shapes,
        )

        for num_call_to_stack in all_num_call_to_stack:
            overhead_ms, op_ms = measuring_func(
                input_shape,
                dtype,
                dlayout,
                in_mem_config,
                out_mem_config,
                device,
                op["op"],
                num_call_to_stack,
                num_repeats,
                bcast=bcast,
                bcast_dim=bcast_dim,
                norm_shapes=norm_shapes,
                embeddings_shapes=embeddings_shapes,
            )

            results_overhead += overhead_ms
            results_op += op_ms

    min_val = round(min(results_overhead), 2)
    mean_val = round(statistics.mean(results_overhead), 2)

    min_val_op = round(min(results_op), 2)
    mean_val_op = round(statistics.mean(results_op), 2)

    logger.info(f"Measure overhead of launching {op['name']} is {min_val:.2f}ms (mean {mean_val:.2f}ms)")
    text_file.write(f"{op['name']}, {min_val}, {mean_val}\n")


def test_host_overhead(device):
    """
    Run witout tracy:
    pytest tests/tt_eager/profiling/profile_host_overhead.py

    Run with tracy:
    python -m tracy -v -r -p -o host_overgead_profile -m "pytest tests/tt_eager/profiling/profile_host_overhead.py"
    """
    output_folder_path = "host_overhead_profiler_output.csv"

    # if user_input is not None and len(user_input) > 0:
    #     output_folder_path = user_input[0]

    with open(output_folder_path, "w") as text_file:
        text_file.write(f"op, overhead min(ms), overhead mean(ms)\n")

        for op in ops_for_profiling.all_binary_ops:
            run_measure_host_overhead(op, device, text_file, measure_host_overhead_binary)

        for op in ops_for_profiling.all_unary_ops:
            run_measure_host_overhead(op, device, text_file, measure_host_overhead_unary)

        for op in ops_for_profiling.all_ternary_ops:
            run_measure_host_overhead(op, device, text_file, measure_host_overhead_ternary)
