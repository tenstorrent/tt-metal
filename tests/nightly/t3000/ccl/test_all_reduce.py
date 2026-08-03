# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
import math
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.nightly.tg.ccl.test_all_reduce_async import (
    run_all_reduce_test,
    run_all_reduce_with_mesh_tensor_along_row,
)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 1),
        # (8, 1), # skipped as 8 devices result in hang in all gather
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, layout, input_dtype, mem_config",
    [
        ([1, 1, 32, 4096], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
        ([1, 1, 32, 8192], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
        ([1, 1, 32, 1024], ttnn.TILE_LAYOUT, ttnn.bfloat8_b, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
        ([1, 1, 32, 2048], ttnn.TILE_LAYOUT, ttnn.bfloat8_b, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
        ([1, 1, 4096, 32], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
        # ([1, 1, 8192, 32]), # skipped as it hangs in reduce scatter part.
        ([1, 1, 1024, 32], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
        ([1, 1, 2048, 32], ttnn.TILE_LAYOUT, ttnn.bfloat8_b, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
        ([4, 1, 32, 4096], ttnn.TILE_LAYOUT, ttnn.bfloat8_b, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
        ([8, 1, 32, 1024], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
        ([1, 4, 1024, 32], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
        ([2, 4, 2048, 32], ttnn.TILE_LAYOUT, ttnn.bfloat8_b, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
    ],
    ids=[
        "1x1x32x4096-bfloat16-DRAM",
        "1x1x32x8192-bfloat16-L1",
        "1x1x32x1024-bfloat8_b-DRAM",
        "1x1x32x2048-bfloat8_b-L1",
        "1x1x4096x32-bfloat16-DRAM",
        "1x1x1024x32-bfloat16-L1",
        "1x1x2048x32-bfloat8_b-DRAM",
        "4x1x32x4096-bfloat8_b-L1",
        "8x1x32x1024-bfloat16-DRAM",
        "1x4x1024x32-bfloat16-L1",
        "2x4x2048x32-bfloat8_b-DRAM",
    ],
)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_ring_all_reduce_post_commit(
    mesh_device,
    num_devices,
    num_links,
    per_chip_output_shape,
    layout,
    input_dtype,
    mem_config,
    math_op,
    function_level_defaults,
    num_iters=2,
):
    run_all_reduce_test(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        function_level_defaults,
        num_iters=num_iters,
    )


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (2, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape",
    [
        ([2, 2, 64, 64]),
        ([1, 1, 64, 64]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
    ],
)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_ring_all_reduce_post_commit_2chip(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    function_level_defaults,
    num_iters=2,
):
    run_all_reduce_with_mesh_tensor_along_row(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        num_iters=num_iters,
        cluster_axis=None,
        use_semaphore_free_all_reduce_impl=True,
    )


def _get_tensors(input_shape, cluster_axis, mesh_shape, dtype, layout, memory_config, device):
    """
    Generates a replicated input tensor for the mesh and computes the golden reference tensor.
    """
    num_devices = math.prod(mesh_shape)

    torch_inputs = [torch.rand(input_shape).bfloat16() for _ in range(num_devices)]
    torch_input = torch.concat(torch_inputs, dim=0)

    torch_reference = torch.reshape(torch_input, tuple(list(mesh_shape) + input_shape))
    torch_reference = torch.sum(torch_reference, dim=cluster_axis)

    torch_reference_copies = []
    for x in range(mesh_shape[0]):
        for y in range(mesh_shape[1]):
            i, j = (x, y) if cluster_axis == 1 else (y, x)
            torch_reference_copies.append(torch_reference[i])

    torch_reference = torch.concat(torch_reference_copies, dim=0)

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
        memory_config=memory_config,
        device=device,
        dtype=dtype,
    )

    return tt_input, torch_reference


MESH_SHAPE = (2, 4)
LAYOUT = ttnn.TILE_LAYOUT
NUM_ITERS = 2


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        [128, 128],
        [544, 2880],
        [8, 8, 128, 128],
        [8, 128, 128],
        [8, 8, 8, 8, 128, 128],
        [8, 8, 8, 16, 16],
        [256],
        [96],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_nd(mesh_device, input_shape, cluster_axis, dtype, memory_config, topology):
    tt_input, torch_reference = _get_tensors(
        input_shape, cluster_axis, tuple(mesh_device.shape), dtype, LAYOUT, memory_config, mesh_device
    )

    for _ in range(NUM_ITERS):
        tt_out_tensor = ttnn.all_reduce(
            tt_input,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
            topology=topology,
        )

        tt_output_tensor = torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(tt_out_tensor)])
        eq, mess = comp_pcc(torch_reference, tt_output_tensor)
        assert eq, mess


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8), (2, 4)], indirect=True, ids=["1x8", "2x4"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("N", [32, 96, 256])
def test_rank1(mesh_device, layout, N):
    torch.manual_seed(0)
    num_devices = tuple(mesh_device.shape)[1]  # cluster_axis 1
    torch_input = torch.rand([N]).bfloat16()

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )
    logger.info(f"N={N} layout={layout} logical={tt_input.shape} padded={tt_input.padded_shape}")

    tt_output = ttnn.all_reduce(
        tt_input,
        cluster_axis=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
    )

    assert list(tt_output.shape) == [N], f"expected a rank-1 output of shape [{N}], got {tt_output.shape}"

    torch_reference = torch_input.float() * num_devices
    for i, device_tensor in enumerate(ttnn.get_device_tensors(tt_output)):
        eq, mess = comp_pcc(torch_reference, ttnn.to_torch(device_tensor).float())
        assert eq, f"device {i}: {mess}"


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True, ids=["1x2", "1x4"])
@pytest.mark.parametrize("N", [96, 256])
def test_rank1_explicit_semaphores(mesh_device, N):
    torch.manual_seed(0)
    num_devices = mesh_device.get_num_devices()

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    rs_global_semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(3)]
    ag_global_semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]
    barrier_semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]

    torch_input = torch.rand([N]).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )
    logger.info(f"N={N} devices={num_devices} logical={tt_input.shape} padded={tt_input.padded_shape}")

    try:
        tt_output = ttnn.experimental.all_reduce_async(
            tt_input,
            num_devices=num_devices,
            barrier_semaphores=barrier_semaphores,
            rs_global_semaphores=rs_global_semaphores,
            ag_global_semaphores=ag_global_semaphores,
            math_op=ttnn.ReduceType.Sum,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            subdevice_id=worker_sub_device_id,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()

    assert list(tt_output.shape) == [N], f"expected a rank-1 output of shape [{N}], got {tt_output.shape}"

    torch_reference = torch_input.float() * num_devices
    for i, device_tensor in enumerate(ttnn.get_device_tensors(tt_output)):
        eq, mess = comp_pcc(torch_reference, ttnn.to_torch(device_tensor).float())
        assert eq, f"device {i}: {mess}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}, {"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
    ids=["fabric_linear", "fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("input_shape", [[2, 2, 32, 32]])
def test_all_reduce_2x4_non_flat_mesh(mesh_device, input_shape):
    torch.manual_seed(520)
    devices = mesh_device.get_num_devices()
    input_shape[-1] *= devices

    torch_inputs_per_device = [torch.rand(input_shape, dtype=torch.bfloat16) for _ in range(devices)]

    torch_reference = torch.zeros_like(torch_inputs_per_device[0])
    for i in range(devices):
        torch_reference += torch_inputs_per_device[i]

    tt_input = ttnn.from_torch(
        torch.cat(torch_inputs_per_device, dim=0),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        device=mesh_device,
    )  # [2, 2, 32, 32*devices] per device

    tt_output = ttnn.all_reduce(tt_input)  # [2, 2, 32, 32] per device
    torch_output = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )  # [2, 2, 32, 32*devices]

    # chunk and make sure each is equal to the reference
    torch_output_slices = torch.chunk(torch_output, devices, dim=0)
    for i, torch_output_slice in enumerate(torch_output_slices):
        assert torch.allclose(
            torch_reference, torch_output_slice, atol=1e-1, rtol=1e-2
        ), f"Output slice {i} mismatch between torch and ttnn reduce-scatter"
