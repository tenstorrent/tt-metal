#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.utility_functions import profiler


# Generate tensor with random data
def random_torch_tensor(dtype, shape):
    torch.manual_seed(1234)
    if dtype == ttnn.uint8:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        return torch.rand(shape, dtype=torch.bfloat16)
    assert False, f"Unsupported dtype {dtype}"


# This assumes all devices in the mesh contain identical output tensors.
# Ex: this is true for all TM ops, not true for ReduceScatter CCL op.
def run_test(mesh_device, run_op_proc, check_op_proc, enable_trace):
    if not enable_trace:
        # Run the op
        tt_outputs = run_op_proc()

        # Every device executes the same op, check that each device returned the
        # same result
        view = mesh_device.get_view() if ttnn.using_distributed_env() else None
        coords = list(tt_outputs.tensor_topology().mesh_coords())
        for coord, tt_output in zip(coords, ttnn.get_device_tensors(tt_outputs)):
            if view is not None and not view.is_local(coord):
                continue
            tt_output = ttnn.to_torch(tt_output)
            check_op_proc(tt_output)
    else:
        profiler.clear()

        # Compile the op
        tt_outputs = run_op_proc()
        tt_outputs.deallocate(True)

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_outputs = run_op_proc()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

        # Execute trace
        profiler.start("op_perf")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        profiler.end("op_perf")
        ttnn.release_trace(mesh_device, trace_id)

        # Get results
        # op_time = profiler.get("op_perf")
        profiler.print(units="us")

        # Every device executes the same op, check that each device returned the
        # same result
        view = mesh_device.get_view() if ttnn.using_distributed_env() else None
        coords = list(tt_outputs.tensor_topology().mesh_coords())
        for coord, tt_output in zip(coords, ttnn.get_device_tensors(tt_outputs)):
            if view is not None and not view.is_local(coord):
                continue
            tt_output = ttnn.to_torch(tt_output)
            check_op_proc(tt_output)
