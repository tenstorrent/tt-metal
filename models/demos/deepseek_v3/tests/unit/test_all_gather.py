#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0


import math

import pytest
import torch

import ttnn
from tests.sweep_framework.sweep_utils.ccl_common import get_mem_configs, get_serializable_shard_specs
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.utils_for_testing import maybe_trace


def _get_tensors(input_shape, mesh_shape, dim, cluster_axis, buffer_type, dtype, layout, shard_specs, device):
    num_devices = math.prod(mesh_shape)
    replicate = mesh_shape[cluster_axis] if cluster_axis is not None else num_devices
    torch.manual_seed(0)
    torch_input = torch.cat([torch.rand(input_shape).bfloat16() for _ in range(replicate)], dim=dim)

    input_memory_config, output_memory_config = get_mem_configs(buffer_type, shard_specs, layout, torch_input.shape)

    shard_dims = (None, dim) if cluster_axis == 1 else (dim, None)
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


SHARD_SPEC_0 = get_serializable_shard_specs(
    input_shape=(32, 32),
    input_cores=(1, 1),
    input_strategy="w",
    output_shape=None,  # (32, 128) in production on Galaxy
    output_cores=(2, 4),
    output_strategy="w",
    valid_tensor_shapes=[[1, 1, 32, 32]],
)

SHARD_SPEC_1 = get_serializable_shard_specs(
    input_shape=(32, 32),
    input_cores=(4, 7),
    input_strategy="w",
    output_shape=None,  # (32, 128) in production on Galaxy
    output_cores=(4, 7),
    output_strategy="w",
    valid_tensor_shapes=[[1, 1, 32, 896]],
)

LAYOUT = ttnn.TILE_LAYOUT

SHAPE_DTYPE_BUFFER_TYPE_SHARD_SPEC = [
    ([1, 1, 32, 224], ttnn.float32, ttnn.BufferType.DRAM, None),
    ([1, 1, 32, 32], ttnn.bfloat16, ttnn.BufferType.L1, SHARD_SPEC_0),
    ([1, 1, 32, 192], ttnn.bfloat16, ttnn.BufferType.DRAM, None),
    ([1, 1, 32, 576], ttnn.bfloat16, ttnn.BufferType.DRAM, None),
    ([1, 1, 32, 896], ttnn.bfloat16, ttnn.BufferType.DRAM, None),
    ([1, 1, 32, 896], ttnn.bfloat16, ttnn.BufferType.DRAM, None),
    ([1, 1, 32, 896], ttnn.bfloat16, ttnn.BufferType.L1, SHARD_SPEC_1),
]


@pytest.mark.requires_device(["N300", "T3K", "TG", "DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}], indirect=True
)
@pytest.mark.parametrize("shape_dtype_buffer_type_shard_spec", SHAPE_DTYPE_BUFFER_TYPE_SHARD_SPEC)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("enable_trace", [True, False])
def test_deepseek(mesh_device, shape_dtype_buffer_type_shard_spec, layout, dim, cluster_axis, topology, enable_trace):
    shape, dtype, buffer_type, shard_spec = shape_dtype_buffer_type_shard_spec

    tt_input, torch_reference, output_mem_config = _get_tensors(
        shape, tuple(mesh_device.shape), dim, cluster_axis, buffer_type, dtype, layout, shard_spec, mesh_device
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
