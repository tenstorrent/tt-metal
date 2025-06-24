# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import ttnn
import time
import statistics
from loguru import logger

from tests.ttnn.profiling import ops_for_profiling
from tracy import signpost


test_sweep_args = [
    # (
    #     (1, 2, 1024, 1024),
    #     ttnn.bfloat16,
    #     ttnn.TILE_LAYOUT,
    #     ttnn.DRAM_MEMORY_CONFIG,
    #     ttnn.DRAM_MEMORY_CONFIG,
    # ),
    ((1, 4, 1024, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
]

all_num_call_to_stack = [20]  # For 10 and more test  execution spills to dispatch
NUM_REPEATS = 20


def torch2tt_tensor(host_tensor, device, dlayout, in_mem_config, dtype, shard_tensor=False):
    x = ttnn.from_torch(
        host_tensor,
        dtype=dtype,
        layout=dlayout,
        device=device,
        memory_config=in_mem_config,
    )

    if not shard_tensor:
        return x

    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    device_grid_size = ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=x.shape,
        core_grid=device_grid_size,  # ttnn.CoreGrid(y=8, x=5),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )

    return ttnn.to_memory_config(x, block_sharded_mem_config)


def measure_host_overhead(op_func, op_name, device, num_call_to_stack, is_warmup):
    ttnn.synchronize_device(device)

    if not is_warmup:
        signpost(header=f"start {op_name}")

    start_time = time.time()
    for _ in range(num_call_to_stack):
        op_func()

    ttnn.synchronize_device(device)

    duration = 1000 * (time.time() - start_time)
    avg_op_time = duration / num_call_to_stack
    logger.info(f"{num_call_to_stack} calls and Synchronize after {duration:.2f}ms ({avg_op_time:.2f}ms per call)")

    start_time = time.time()
    for _ in range(num_call_to_stack):
        # signpost(header=f"starting {op_name}")
        op_func()
        # signpost(header=f"ending {op_name}")

    dispatch_end_time = time.time()
    ttnn.synchronize_device(device)

    sync_time = 1000 * (time.time() - dispatch_end_time)
    dispatch_time = 1000 * (dispatch_end_time - start_time)
    avg_dispatch_time = dispatch_time / num_call_to_stack

    logger.info(
        f"{num_call_to_stack} calls without Synchronize {dispatch_time:.2f}ms ({avg_dispatch_time:.3f}ms per call)"
    )
    logger.info(f"Synchronize {sync_time:.2f}ms ({(sync_time/num_call_to_stack):.2f}ms per call)")

    if not is_warmup:
        signpost(header=f"end {op_name}")

    return avg_dispatch_time, avg_op_time


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
    shape_func=None,
    is_complex=[False, False],
    need_out_mem_cfg=False,
    is_warmup=False,
    use_sharded_tensors=[False, False],
):
    input_shape_0 = input_shape
    input_shape_1 = input_shape

    if shape_func is not None:
        input_shape_0, input_shape_1 = shape_func(input_shape)

    x = torch.Tensor(size=input_shape_0).uniform_(-100, 100).bfloat16()
    y = torch.Tensor(size=input_shape_1).uniform_(-100, 100).bfloat16()

    x = torch2tt_tensor(x, device, dlayout, in_mem_config, dtype, shard_tensor=use_sharded_tensors[0])
    y = torch2tt_tensor(y, device, dlayout, in_mem_config, dtype, shard_tensor=use_sharded_tensors[1])

    if is_complex[0]:
        x = ttnn.complex_tensor(x, x)

    if is_complex[1]:
        y = ttnn.complex_tensor(y, y)

    def op_func():
        if need_out_mem_cfg:
            op["op"](x, y, memory_config=in_mem_config)
        else:
            op["op"](x, y)

    result_overhead = []
    result_op = []

    for _ in range(num_repeats):
        overhead_ms, total_op_time = measure_host_overhead(op_func, op["name"], device, num_call_to_stack, is_warmup)
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
    shape_func=None,
    is_complex=[False],
    need_out_mem_cfg=False,
    is_warmup=False,
    use_sharded_tensors=[False],
):
    input_shape_0 = input_shape

    if shape_func is not None:
        input_shape_0 = shape_func(input_shape)

    x = torch.Tensor(size=input_shape_0).uniform_(-100, 100).bfloat16()
    x = torch2tt_tensor(x, device, dlayout, in_mem_config, dtype, shard_tensor=use_sharded_tensors[0])

    if is_complex[0]:
        x = ttnn.complex_tensor(x, x)

    def op_func():
        if need_out_mem_cfg:
            op["op"](x, memory_config=in_mem_config)
        else:
            op["op"](x)

    result_overhead = []
    result_op = []

    for _ in range(num_repeats):
        overhead_ms, total_op_time = measure_host_overhead(op_func, op["name"], device, num_call_to_stack, is_warmup)
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
    shape_func=None,
    is_complex=[False, False, False],
    need_out_mem_cfg=False,
    is_warmup=False,
    use_sharded_tensors=[False, False, False],
):
    input_shape_0 = input_shape
    input_shape_1 = input_shape
    input_shape_2 = input_shape

    if shape_func is not None:
        input_shape_0, input_shape_1, input_shape_2 = shape_func(input_shape)

    x = torch.Tensor(size=input_shape_0).uniform_(-100, 100).bfloat16()
    y = torch.Tensor(size=input_shape_1).uniform_(-100, 100).bfloat16()
    z = torch.Tensor(size=input_shape_2).uniform_(-100, 100).bfloat16()

    x = torch2tt_tensor(x, device, dlayout, in_mem_config, dtype, shard_tensor=use_sharded_tensors[0])
    y = torch2tt_tensor(y, device, dlayout, in_mem_config, dtype, shard_tensor=use_sharded_tensors[1])
    z = torch2tt_tensor(z, device, dlayout, in_mem_config, dtype, shard_tensor=use_sharded_tensors[2])

    if is_complex[0]:
        x = ttnn.complex_tensor(x, x)

    if is_complex[1]:
        y = ttnn.complex_tensor(y, y)

    if is_complex[2]:
        z = ttnn.complex_tensor(z, z)

    def op_func():
        if need_out_mem_cfg:
            op["op"](x, y, z, memory_config=in_mem_config)
        else:
            op["op"](x, y, z)

    result_overhead = []
    result_op = []

    for _ in range(num_repeats):
        overhead_ms, total_op_time = measure_host_overhead(op_func, op["name"], device, num_call_to_stack, is_warmup)
        result_overhead.append(overhead_ms)
        result_op.append(total_op_time)

    return result_overhead, result_op


def run_measure_host_overhead(op, device, text_file, measuring_func):
    results_overhead = []
    results_op = []
    op_count = 0

    for input_shape, dtype, dlayout, in_mem_config, out_mem_config in test_sweep_args:
        logger.info("")
        logger.info(f"Profiling op {op['name']} for input shape {input_shape}")

        if "layout" in op and op["layout"] == "ROW_MAJOR":
            dlayout = ttnn.ROW_MAJOR_LAYOUT

        num_repeats = op["num_repeats"] if "num_repeats" in op else NUM_REPEATS
        shape_func = None if "shape_func" not in op else op["shape_func"]
        is_complex = [False, False, False] if "is_complex" not in op else op["is_complex"]
        need_out_mem_cfg = False if "need_out_mem_cfg" not in op else op["need_out_mem_cfg"]
        use_sharded_tensors = [False, False, False] if "use_sharded_tensors" not in op else op["use_sharded_tensors"]

        # Warmup
        measuring_func(
            input_shape,
            dtype,
            dlayout,
            in_mem_config,
            out_mem_config,
            device,
            op,
            num_call_to_stack=1,
            num_repeats=1,
            shape_func=shape_func,
            is_complex=is_complex,
            need_out_mem_cfg=need_out_mem_cfg,
            is_warmup=True,
            use_sharded_tensors=use_sharded_tensors,
        )

        for num_call_to_stack in all_num_call_to_stack:
            overhead_ms, op_ms = measuring_func(
                input_shape,
                dtype,
                dlayout,
                in_mem_config,
                out_mem_config,
                device,
                op,
                num_call_to_stack,
                num_repeats,
                shape_func=shape_func,
                need_out_mem_cfg=need_out_mem_cfg,
                is_complex=is_complex,
                use_sharded_tensors=use_sharded_tensors,
            )

            op_count += len(overhead_ms) * num_call_to_stack * 2
            results_overhead += overhead_ms
            results_op += op_ms

    min_val = round(min(results_overhead), 3)
    mean_val = round(statistics.mean(results_overhead), 3)

    # min_val_op = round(min(results_op), 2)
    mean_val_op = round(statistics.mean(results_op), 3)

    logger.info(f"Measure overhead of launching {op['name']} is {min_val:.2f}ms (mean {mean_val:.2f}ms)")
    text_file.write(f"{op['name']},{op_count},{min_val},{mean_val},{mean_val_op}\n")


def test_host_overhead(device, user_input):
    """
    Run witout tracy:
    pytest tests/ttnn/profiling/profile_host_overhead.py --input-method cli --cli-input host_overhead_profile

    Run only for one op:
    pytest tests/ttnn/profiling/profile_host_overhead.py --input-method cli --cli-input host_overhead_profile::ttnn.add

    Run with tracy:
    python -m tracy -v -r -p -o host_overhead_profile --no-device -m "pytest tests/ttnn/profiling/profile_host_overhead.py --input-method cli --cli-input host_overhead_profile"
    """

    # Enable program cache
    device.enable_program_cache()

    if "::" in user_input[0]:
        splitted = user_input[0].split("::")
        out_directory = splitted[0]
        op_name = splitted[1]
        out_file_path = os.path.join(out_directory, f"host_overhead_{op_name}.csv")
    else:
        out_directory = user_input[0]
        out_file_path = os.path.join(out_directory, f"host_overhead_profiler_output.csv")
        op_name = ""

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    with open(out_file_path, "w") as text_file:
        start_time = time.time()
        text_file.write(
            f"op,count,python min dispatch time (ms),python mean dispatch time(ms),python mean dispatch + sync time (ms)\n"
        )

        for op in ops_for_profiling.all_binary_ops:
            if op_name != "":
                if op["name"] != op_name:
                    continue

            run_measure_host_overhead(op, device, text_file, measure_host_overhead_binary)

        for op in ops_for_profiling.all_unary_ops:
            if op_name != "":
                if op["name"] != op_name:
                    continue

            run_measure_host_overhead(op, device, text_file, measure_host_overhead_unary)

        for op in ops_for_profiling.all_ternary_ops:
            if op_name != "":
                if op["name"] != op_name:
                    continue

            run_measure_host_overhead(op, device, text_file, measure_host_overhead_ternary)

        duration = (time.time() - start_time) / 60
        logger.info(f"Profiling finished in {duration:.2f}min")
