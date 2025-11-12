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
    "in_shape, dim, index_shape, src_shape",
    [
        ([1, 1, 32, 8], 3, [1, 1, 32, 4], [1, 1, 32, 4]),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("index_layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("index_mem_config", [ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_scatter(
    mesh_device, in_shape, dim, index_shape, src_shape, dtype, layout, mem_config, index_layout, index_mem_config
):
    profiler.clear()

    # Index tensor generation is somewhat hardcoded
    index_dtype = ttnn.uint16
    index_torch_dtype = torch.int64  # torch.scatter expects int64 for index tensor

    torch_index = torch.zeros(index_shape, dtype=index_torch_dtype)
    for n in range(index_shape[0]):
        for c in range(index_shape[1]):
            for h in range(index_shape[2]):
                # Generate random unique indices
                min_index = 0
                max_index = in_shape[dim]
                indices = torch.randperm(max_index - min_index, dtype=index_torch_dtype) + min_index
                indices = indices[: index_shape[3]]

                torch_index[n, c, h, :] = indices

    torch_input = random_torch_tensor(dtype, in_shape)
    torch_src = random_torch_tensor(dtype, src_shape)
    torch_output = torch_input.scatter(dim, torch_index, torch_src)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_index = ttnn.from_torch(
        torch_index,
        device=mesh_device,
        dtype=index_dtype,
        layout=index_layout,
        memory_config=index_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_src = ttnn.from_torch(
        torch_src,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.scatter(tt_input, dim, tt_index, tt_src)

    def check_op(tt_output):
        assert_with_pcc(torch_output, tt_output, 0.9999)

    run_test(mesh_device, run_op, check_op)
