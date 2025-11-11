#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import profiler
from tests.ttnn.utils_for_testing import assert_with_pcc


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
    # return torch.rand(shape).bfloat16().float()
    assert False, f"Unsupported dtype {dtype}"


def run_test(mesh_device, run_op_proc, check_op_proc):
    ####################### NON-TRACE RUN #######################

    # Run the op
    tt_output = run_op_proc()
    tt_output = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Every device executes the same op, check that each device returned the
    # same result
    N = tt_output.shape[0] // mesh_device.get_num_devices()
    for dev in range(mesh_device.get_num_devices()):
        i = dev * N
        j = (dev + 1) * N
        check_op_proc(tt_output[i:j, ...])

    ####################### TRACE RUN #######################

    # Compile the op
    tt_output = run_op_proc()
    tt_output.deallocate(True)

    # Capture the trace
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_output = run_op_proc()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    # Execute trace
    profiler.start("op_perf")
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    profiler.end("op_perf")
    ttnn.release_trace(mesh_device, trace_id)

    # Get results
    tt_output = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # op_time = profiler.get("op_perf")
    profiler.print(units="us")

    # Every device executes the same op, check that each device returned the
    # same result
    N = tt_output.shape[0] // mesh_device.get_num_devices()
    for dev in range(mesh_device.get_num_devices()):
        i = dev * N
        j = (dev + 1) * N
        check_op_proc(tt_output[i:j, ...])


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape, repeat_shape, dtype, mem_config",
    [
        ([1, 1, 1, 256], [1, 32, 1, 1], ttnn.float32, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 1, 8], [1, 32, 1, 1], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 1, 4], [1, 32, 1, 1], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 256, 1], [1, 1, 1, 32], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 1, 8], [1, 32, 1, 1], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 1, 8], [1, 32, 1, 1], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 128, 7168], [1, 8, 1, 1], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
        ([1, 1, 32, 8], [1, 7168, 1, 1], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_repeat(mesh_device, shape, repeat_shape, dtype, mem_config, layout):
    profiler.clear()

    torch_input = random_torch_tensor(dtype, shape)
    torch_output = torch_input.repeat(repeat_shape)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.repeat(tt_input, repeat_shape)

    def check_op(tt_output):
        assert_with_pcc(torch_output, tt_output, 0.9999)

    run_test(mesh_device, run_op, check_op)
