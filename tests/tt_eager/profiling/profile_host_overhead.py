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

all_num_call_to_stack = [1, 2, 3]  # For 10 and more test  execution spills to dispatch
num_repeats = 10


def measure_host_overhead(op_func, device, num_call_to_stack):
    start_time = time.time()
    for _ in range(num_call_to_stack):
        op_func()

    tt_lib.device.Synchronize(device)

    duration = 1000 * (time.time() - start_time)
    duration_per_call = duration / num_call_to_stack
    logger.info(
        f"{num_call_to_stack} calls and Synchronize after {duration:.2f}ms ({duration_per_call:.2f}ms per call)"
    )

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
    return overhead_ms


def measure_host_overhead_binary(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, device, op, num_call_to_stack
):
    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    y = torch.Tensor(size=input_shape).uniform_(-100, 100)

    x = torch2tt_tensor(x, device, dlayout, in_mem_config, dtype)
    y = torch2tt_tensor(y, device, dlayout, in_mem_config, dtype)

    def op_func():
        op(x, y)

    return measure_host_overhead(op_func, device, num_call_to_stack)


def measure_host_overhead_unary(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, device, op, num_call_to_stack
):
    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    x = torch2tt_tensor(x, device, dlayout, in_mem_config, dtype)

    def op_func():
        op(x)

    return measure_host_overhead(op_func, device, num_call_to_stack)


def test_host_overhead(device):
    """
    Run witout tracy:
    pytest tests/tt_eager/profiling/profile_host_overhead.py

    Run with tracy:
    python -m tracy -v -r -p -o host_overgead_profile -m "pytest tests/tt_eager/profiling/profile_host_overhead.py"
    """
    results = {}
    output_folder_path = "host_overhead_profiler_output.csv"

    # if user_input is not None and len(user_input) > 0:
    #     output_folder_path = user_input[0]

    for op in ops_for_profiling.all_binary_ops:
        for input_shape, dtype, dlayout, in_mem_config, out_mem_config in test_sweep_args:
            logger.info("")
            logger.info(f"Profiling op {op['name']} for input shape {input_shape}")
            results[op["name"]] = []

            # Warmup
            overhead_ms = measure_host_overhead_binary(
                input_shape,
                dtype,
                dlayout,
                in_mem_config,
                out_mem_config,
                device,
                op["op"],
                1,
            )

            for _ in range(num_repeats):
                for num_call_to_stack in all_num_call_to_stack:
                    overhead_ms = measure_host_overhead_binary(
                        input_shape,
                        dtype,
                        dlayout,
                        in_mem_config,
                        out_mem_config,
                        device,
                        op["op"],
                        num_call_to_stack,
                    )

                    results[op["name"]].append(overhead_ms)

    for op in ops_for_profiling.all_unary_ops:
        for input_shape, dtype, dlayout, in_mem_config, out_mem_config in test_sweep_args:
            logger.info("")
            logger.info(f"Profiling op {op['name']} for input shape {input_shape}")
            results[op["name"]] = []

            if "layout" in op and op["layout"] == "ROW_MAJOR":
                dlayout = tt_lib.tensor.Layout.ROW_MAJOR

            # Warmup
            logger.info(f"Warming up")
            overhead_ms = measure_host_overhead_unary(
                input_shape,
                dtype,
                dlayout,
                in_mem_config,
                out_mem_config,
                device,
                op["op"],
                1,
            )

            logger.info(f"Warm up finished")
            for _ in range(num_repeats):
                for num_call_to_stack in all_num_call_to_stack:
                    overhead_ms = measure_host_overhead_unary(
                        input_shape,
                        dtype,
                        dlayout,
                        in_mem_config,
                        out_mem_config,
                        device,
                        op["op"],
                        num_call_to_stack,
                    )

                    results[op["name"]].append(overhead_ms)

    logger.info("")

    with open(output_folder_path, "w") as text_file:
        text_file.write(f"op, min(ms), mean(ms)\n")

        for key in results:
            min_val = round(min(results[key]), 2)
            mean_val = round(statistics.mean(results[key]), 2)

            logger.info(f"Measure overhead of launching {key} is {min_val:.2f}ms (mean {mean_val:.2f}ms)")
            text_file.write(f"{key}, {min_val}, {mean_val}\n")
