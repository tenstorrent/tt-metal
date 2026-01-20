#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def _valid_cluster_div(input_shape, dim, cluster_axis, mesh_shape, **kwargs):
    return input_shape[dim] % (math.prod(mesh_shape) if cluster_axis is None else mesh_shape[cluster_axis]) == 0


def _get_tensors(
    input_shape,
    mesh_shape,
    dim,
    cluster_axis,
    dtype,
    memory_config,
    layout,
    device,
    math_op=ttnn.ReduceType.Sum,
):
    assert _valid_cluster_div(input_shape, dim, cluster_axis, mesh_shape)

    num_devices = math.prod(mesh_shape)
    axis_devices = num_devices if cluster_axis is None else mesh_shape[cluster_axis]

    elems = math.prod(input_shape)
    torch_inputs = [
        torch.linspace(i * elems, (i + 1) * elems, elems).reshape(input_shape).bfloat16() for i in range(axis_devices)
    ]
    torch_input = torch.concat(torch_inputs, dim=0)

    if cluster_axis == 1:
        torch_reference = torch_input.reshape([1, axis_devices] + input_shape)
        torch_reference = torch_reference.repeat([num_devices // axis_devices, 1] + [1, 1, 1, 1])
    else:
        torch_reference = torch_input.reshape([axis_devices, 1] + input_shape)
        torch_reference = torch_reference.repeat([1, num_devices // axis_devices] + [1, 1, 1, 1])
    torch_reference = torch.sum(torch_reference, dim=cluster_axis)

    dim_per_device = input_shape[dim] // mesh_shape[cluster_axis]

    torch_reference_slices = []
    for x in range(mesh_shape[0]):
        for y in range(mesh_shape[1]):
            i, j = (x, y) if cluster_axis == 1 else (y, x)

            torch_reference_slice = torch_reference[i].split(dim_per_device, dim=dim)[j]
            torch_reference_slices.append(torch_reference_slice)

    shard_dims = (None, 0) if cluster_axis == 1 else (0, None)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=mesh_shape),
        device=device,
    )

    return tt_input, torch_reference_slices


@pytest.mark.requires_device(
    [
        "TG",
    ]
)  # TODO: Add N300, T3K, DUAL, QUAD
@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize("enable_trace", [True])
@pytest.mark.parametrize(
    "input_shape, dim, layout, dtype, memory_config, num_iters",
    [
        ([1, 1, 32, 1536], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, 10),
        ([1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, 10),
        # ([1, 1, 32, 1536], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, 10),  # duplicate
        # ([1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, 10),  # duplicate
        # ([1, 1, 32, 1536], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, 10),  # duplicate
        # ([1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, 10),  # duplicate
        # ([1, 1, 32, 1536], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, 10),  # duplicate
        ([1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, 10),
        # ([1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, 10),  # duplicate
    ],
)
@pytest.mark.parametrize(
    "device_params, topology, cluster_axis",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1171456}, ttnn.Topology.Linear, 1),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_nd(
    mesh_device, enable_trace, input_shape, dim, layout, dtype, memory_config, num_iters, topology, cluster_axis
):
    if dim >= len(input_shape):
        pytest.skip("Invalid scatter dim")

    tt_input, torch_reference_slices = _get_tensors(
        input_shape,
        tuple(mesh_device.shape),
        dim,
        cluster_axis,
        dtype,
        memory_config,
        layout,
        mesh_device,
    )

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(3)]

    for i in range(num_iters):
        tt_out_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            tt_input,
            dim=dim,
            multi_device_global_semaphore=semaphores,
            cluster_axis=cluster_axis,
            topology=topology,
        )

        coords = list(tt_out_tensor.tensor_topology().mesh_coords())
        view = mesh_device.get_view() if ttnn.using_distributed_env() else None
        torch_outputs = []
        torch_references = []
        for coord, tt_out, torch_ref in zip(coords, ttnn.get_device_tensors(tt_out_tensor), torch_reference_slices):
            if view is not None and not view.is_local(coord):
                continue
            torch_outputs.append(ttnn.to_torch(tt_out))
            torch_references.append(torch_ref)
        tt_output_tensor = torch.cat(torch_outputs)
        torch_reference = torch.cat(torch_references)
        eq, mess = comp_pcc(torch_reference, tt_output_tensor)
        assert eq, mess
