# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
##
from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal


def _linear_coord(coord, mesh_shape):
    return coord[0] * mesh_shape[1] + coord[1]


def run_test_col_step(mesh_shape, col_dim_idx, data_tensor):
    for row_dim_idx in range(mesh_shape[1]):
        source_coord = (col_dim_idx, row_dim_idx)
        dest_coord = (col_dim_idx + 1, row_dim_idx)
        assert dest_coord[0] < mesh_shape[0]

        ttnn.point_to_point(
            data_tensor,
            ttnn.MeshCoordinate(dest_coord),
            ttnn.MeshCoordinate(source_coord),
            ttnn.Topology.Linear,
            optional_output_tensor=data_tensor,
        )


def compare_mesh_row(col_dim_idx, ref_tensor_torch, test_tensor_tt, mesh_device, mesh_shape):
    test_tensor_torch = ttnn.to_torch(test_tensor_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    lrow_start_idx = _linear_coord((col_dim_idx, 0), mesh_shape)
    lrow_end_idx = _linear_coord((col_dim_idx, mesh_shape[1]), mesh_shape)
    assert_equal(ref_tensor_torch, test_tensor_torch[lrow_start_idx:lrow_end_idx, :, :, :])


MESH_SHAPE = (4, 2) if ttnn.get_num_devices() == 8 else (4, 8)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("batches_per_col_device", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_moe_p2p(mesh_device, batches_per_col_device, hidden_size, dtype):
    mesh_shape = tuple(mesh_device.shape)
    devices = prod(mesh_shape)
    batch = batches_per_col_device * mesh_shape[0]

    data_shape = (1, 1, batch, hidden_size)
    sharded_data_shape = tuple(s * (devices if i == 0 else 1) for i, s in enumerate(data_shape))

    data_tensor_torch = torch.zeros(sharded_data_shape, dtype=dtype)

    # fill the first row
    for row_dim_idx in range(mesh_shape[1]):
        lc = _linear_coord((0, row_dim_idx), mesh_shape)
        data_tensor_torch[lc, :, :, :] = (
            torch.linspace(1, prod(data_shape), prod(data_shape)).reshape(data_shape).to(dtype=dtype)
        )
    data_tensor = ttnn.from_torch(
        data_tensor_torch, device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
    )

    # test transferring down each row of the columns
    for col_dim_idx in range(mesh_shape[0] - 1):
        run_test_col_step(mesh_shape, col_dim_idx, data_tensor)
        compare_mesh_row(col_dim_idx, data_tensor_torch[: mesh_shape[1], :, :, :], data_tensor, mesh_device, mesh_shape)


def _broadcast_through_column(source_col_dim_idx, row_dim_index, data_tensor, mesh_shape):
    source_coord = ttnn.MeshCoordinate(source_col_dim_idx, row_dim_index)
    for col_dim_idx in range(mesh_shape[0]):
        if col_dim_idx == source_col_dim_idx:
            continue
        dest_coord = ttnn.MeshCoordinate(col_dim_idx, row_dim_index)

        ttnn.point_to_point(
            data_tensor,
            dest_coord,
            source_coord,
            ttnn.Topology.Linear,
            optional_output_tensor=data_tensor,
        )


def _check_col(source_col_dim_idx, row_dim_index, data_tensor_tt, mesh_device, mesh_shape):
    data_tensor_torch = ttnn.to_torch(data_tensor_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    lcoord_source = _linear_coord((source_col_dim_idx, row_dim_index), mesh_shape)
    ref_data = data_tensor_torch[lcoord_source, :, :, :]

    for col_dim_idx in range(mesh_shape[0]):
        if col_dim_idx == source_col_dim_idx:
            continue
        lcoord_test = _linear_coord((col_dim_idx, row_dim_index), mesh_shape)
        test_data = data_tensor_torch[lcoord_test, :, :, :]
        assert_equal(ref_data, test_data)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("source_col_dim_idx", [0, 1, 2])
@pytest.mark.parametrize("batches_per_col_device", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_moe_p2p_broadcast(mesh_device, source_col_dim_idx, batches_per_col_device, hidden_size, dtype):
    mesh_shape = tuple(mesh_device.shape)
    devices = prod(mesh_shape)
    batch = batches_per_col_device * mesh_shape[0]

    data_shape = (1, 1, batch, hidden_size)
    sharded_data_shape = tuple(s * (devices if i == 0 else 1) for i, s in enumerate(data_shape))

    data_tensor_torch = torch.zeros(sharded_data_shape, dtype=dtype)

    # fill source row of shards with data
    for row_dim_idx in range(mesh_shape[1]):
        lc = _linear_coord((source_col_dim_idx, row_dim_idx), mesh_shape)
        data_tensor_torch[lc, :, :, :] = (
            torch.linspace(1, prod(data_shape), prod(data_shape)).reshape(data_shape).to(dtype=dtype)
        )

    data_tensor = ttnn.from_torch(
        data_tensor_torch, device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
    )

    # loop over columns, broadcast through each and check results
    for row_dim_idx in range(mesh_shape[1]):
        _broadcast_through_column(source_col_dim_idx, row_dim_idx, data_tensor, mesh_shape)
        _check_col(source_col_dim_idx, row_dim_idx, data_tensor, mesh_device, mesh_shape)
