# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""All-gather test for 32x4 mesh (4-Galaxy QUAD exabox), cluster_axis=0."""

import math

import pytest
import torch

import ttnn
from tests.sweep_framework.sweep_utils.ccl_common import get_mem_configs
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.utils_for_testing import maybe_trace


def _get_tensors(input_shape, mesh_shape, dim, cluster_axis, buffer_type, dtype, layout, device):
    num_devices = math.prod(mesh_shape)
    replicate = mesh_shape[cluster_axis]
    torch.manual_seed(0)
    torch_input = torch.cat([torch.rand(input_shape).bfloat16() for _ in range(replicate)], dim=dim)

    input_memory_config, output_memory_config = get_mem_configs(buffer_type, None, layout, torch_input.shape)

    shard_dims = (dim, None) if cluster_axis == 0 else (None, dim)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=dtype,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=mesh_shape),
        device=device,
    )

    torch_reference = torch_input.repeat([num_devices] + [1] * (len(input_shape) - 1))
    return tt_input, torch_reference, output_memory_config


@pytest.mark.requires_device(["DUAL_BH"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((16, 4), id="16x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("enable_trace", [True, False])
@pytest.mark.parametrize(
    "input_shape, dtype, buffer_type",
    [
        ([1, 1, 32, 224], ttnn.bfloat16, ttnn.BufferType.DRAM),
        ([1, 1, 32, 896], ttnn.bfloat16, ttnn.BufferType.DRAM),
    ],
    ids=["small_dram", "large_dram"],
)
def test_all_gather_16x4(
    mesh_device,
    cluster_axis,
    dim,
    topology,
    enable_trace,
    input_shape,
    dtype,
    buffer_type,
):
    tt_input, torch_reference, output_mem_config = _get_tensors(
        input_shape, tuple(mesh_device.shape), dim, cluster_axis, buffer_type, dtype, ttnn.TILE_LAYOUT, mesh_device
    )

    def run_op():
        return ttnn.all_gather(
            tt_input, dim, cluster_axis=cluster_axis, topology=topology, memory_config=output_mem_config
        )

    tt_output_tensor = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)

    coords = list(tt_output_tensor.tensor_topology().mesh_coords())
    coord_to_index = {coord: idx for idx, coord in enumerate(coords)}
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    device_tensors = ttnn.get_device_tensors(tt_output_tensor)
    coord_iter = coords
    if view is not None and len(device_tensors) != len(coords):
        coord_iter = [coord for coord in coords if view.is_local(coord)]

    per_device_batch = torch_reference.shape[0] // math.prod(mesh_device.shape)
    torch_reference_slices = torch_reference.split(per_device_batch, dim=0)
    for coord, tt_out in zip(coord_iter, device_tensors):
        if view is not None and not view.is_local(coord):
            continue
        device_idx = coord_to_index[coord]
        eq, mess = comp_equal(torch_reference_slices[device_idx], ttnn.to_torch(tt_out))
        assert eq, mess


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
            ttnn.Topology.Linear,
            id="fabric_1d-linear",
        ),
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
            ttnn.Topology.Linear,
            id="fabric_1d_ring-linear",
        ),
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
            ttnn.Topology.Ring,
            id="fabric_1d_ring-ring",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((32, 4), id="32x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [pytest.param(0, id="axis0"), pytest.param(1, id="axis1")])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("enable_trace", [True, False])
@pytest.mark.parametrize(
    "input_shape, dtype, buffer_type",
    [
        ([1, 1, 32, 224], ttnn.bfloat16, ttnn.BufferType.DRAM),
        ([1, 1, 32, 896], ttnn.bfloat16, ttnn.BufferType.DRAM),
    ],
    ids=["small_dram", "large_dram"],
)
def test_all_gather_32x4(
    mesh_device,
    cluster_axis,
    dim,
    topology,
    enable_trace,
    input_shape,
    dtype,
    buffer_type,
):
    tt_input, torch_reference, output_mem_config = _get_tensors(
        input_shape, tuple(mesh_device.shape), dim, cluster_axis, buffer_type, dtype, ttnn.TILE_LAYOUT, mesh_device
    )

    def run_op():
        return ttnn.all_gather(
            tt_input, dim, cluster_axis=cluster_axis, topology=topology, memory_config=output_mem_config
        )

    tt_output_tensor = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)

    coords = list(tt_output_tensor.tensor_topology().mesh_coords())
    coord_to_index = {coord: idx for idx, coord in enumerate(coords)}
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    device_tensors = ttnn.get_device_tensors(tt_output_tensor)
    coord_iter = coords
    if view is not None and len(device_tensors) != len(coords):
        coord_iter = [coord for coord in coords if view.is_local(coord)]

    per_device_batch = torch_reference.shape[0] // math.prod(mesh_device.shape)
    torch_reference_slices = torch_reference.split(per_device_batch, dim=0)
    for coord, tt_out in zip(coord_iter, device_tensors):
        if view is not None and not view.is_local(coord):
            continue
        device_idx = coord_to_index[coord]
        eq, mess = comp_equal(torch_reference_slices[device_idx], ttnn.to_torch(tt_out))
        assert eq, mess
