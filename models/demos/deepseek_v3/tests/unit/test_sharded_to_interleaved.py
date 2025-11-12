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
    "shape, shard_type, cores, out_mem_config",
    [
        ([1, 1, 32, 896], "W", (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        ([1, 32, 16, 64], "H", (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        ([1, 32, 1, 64], "H", (1, 1), ttnn.DRAM_MEMORY_CONFIG),
        ([1, 4, 128, 512], "H", (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 32, 896],   "W",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 32, 7168], "W", (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 32, 896],   "W",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 32, 16, 64],   "H",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 32, 1, 64],    "H",    (1, 1), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 4, 128, 512],  "H",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 32, 896],   "W",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 32, 7168],  "W",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 32, 896],   "W",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 32, 16, 64],   "H",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 32, 1, 64],    "H",    (1, 1), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 4, 128, 512],  "H",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 32, 896],   "W",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 32, 7168],  "W",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 32, 896],   "W",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 32, 16, 64],   "H",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 32, 1, 64],    "H",    (1, 1), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 4, 128, 512],  "H",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 32, 896],   "W",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 32, 7168],  "W",    (2, 2), ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 32, 7168], "W", (2, 2), ttnn.L1_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_sharded_to_interleaved(mesh_device, shape, shard_type, cores, out_mem_config, dtype, layout):
    profiler.clear()

    torch_input = random_torch_tensor(dtype, shape)
    torch_output = torch_input

    # Shard spec
    num_cores = cores[0] * cores[1]
    core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores[0] - 1, cores[1] - 1)),
        }
    )
    padded_shape = shape
    if layout == ttnn.TILE_LAYOUT:
        padded_shape[-1] = ttnn.core.roundup(padded_shape[-1], ttnn.TILE_SIZE)
        padded_shape[-2] = ttnn.core.roundup(padded_shape[-2], ttnn.TILE_SIZE)
    if shard_type == "H":
        mem_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        shard_shape = ((padded_shape[0] * padded_shape[1] * padded_shape[2]) // num_cores, padded_shape[3])
    else:
        mem_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        shard_shape = (padded_shape[0] * padded_shape[1] * padded_shape[2], padded_shape[3] // num_cores)
    mem_config = ttnn.MemoryConfig(
        mem_layout, ttnn.BufferType.L1, ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.sharded_to_interleaved(tt_input, out_mem_config)

    def check_op(tt_output):
        assert_with_pcc(torch_output, tt_output, 0.9999)

    run_test(mesh_device, run_op, check_op)
