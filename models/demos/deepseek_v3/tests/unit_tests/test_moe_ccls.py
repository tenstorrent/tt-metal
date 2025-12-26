import pytest
import torch

import ttnn
from models.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_equal


def _check_row_broadcast(source_row_dim_idx, data_tensor_tt, mesh_device, mesh_shape):
    """Check that broadcast was successful by comparing data across all rows."""

    data_tensor_torch = ttnn.to_torch(
        data_tensor_tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_shape)
    )

    for row_dim_idx in range(mesh_shape[0]):
        if row_dim_idx == source_row_dim_idx:
            continue
        assert_equal(data_tensor_torch[row_dim_idx, :, :, :], data_tensor_torch[source_row_dim_idx, :, :, :])


def _check_row_to_row_transfer(source_row_dim_idx, dest_row_dim_idx, data_tensor_tt, mesh_device, mesh_shape):
    """Check that row-to-row transfer was successful by comparing data between specific rows."""

    data_tensor_torch = ttnn.to_torch(
        data_tensor_tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_shape)
    )

    assert_equal(data_tensor_torch[dest_row_dim_idx, :, :, :], data_tensor_torch[source_row_dim_idx, :, :, :])


def broadcast_row_to_mesh(tt_input, mesh_device, src_row):
    """Broadcast data from a source row to all other rows in the mesh."""
    mesh_shape = tuple(mesh_device.shape)

    # Create semaphore for synchronization
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    semaphore = ttnn.create_global_semaphore(mesh_device, cores, 0)

    # Broadcast from src_row to mesh
    # Loop over rows, skip src_row
    for row_dim_idx in range(mesh_shape[0]):
        if row_dim_idx == src_row:
            continue

        # Do p2p transfer from nodes in src_row to other nodes in that column
        for col_dim_idx in range(mesh_shape[1]):
            source_coord = ttnn.MeshCoordinate(src_row, col_dim_idx)
            dest_coord = ttnn.MeshCoordinate(row_dim_idx, col_dim_idx)

            ttnn.point_to_point(
                tt_input,
                dest_coord,
                source_coord,
                ttnn.Topology.Linear,
                semaphore,
                optional_output_tensor=tt_input,
            )

    return tt_input


def transfer_row(tt_input, mesh_device, src_row, dst_row):
    """Transfer data from source row to destination row in the mesh."""
    mesh_shape = tuple(mesh_device.shape)

    # Create semaphore for synchronization
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    semaphore = ttnn.create_global_semaphore(mesh_device, cores, 0)

    # Transfer from src_row to dest_row
    # Do p2p transfer from nodes in src_row to corresponding nodes in dest_row
    for col_dim_idx in range(mesh_shape[1]):
        source_coord = ttnn.MeshCoordinate(src_row, col_dim_idx)
        dest_coord = ttnn.MeshCoordinate(dst_row, col_dim_idx)

        ttnn.point_to_point(
            tt_input,
            dest_coord,
            source_coord,
            ttnn.Topology.Linear,
            semaphore,
            optional_output_tensor=tt_input,
        )

    return tt_input


def partition_batch_on_mesh(tt_input, dim, memory_config, cluster_axis=0):
    """Partition tensor along specified dimension across mesh cluster axis."""
    tt_out_tensor = ttnn.mesh_partition(
        tt_input,
        dim,
        cluster_axis=cluster_axis,
        memory_config=memory_config,
    )
    return tt_out_tensor


def gather_batch_on_mesh(tt_input, mesh_device, dim, cluster_axis, memory_config):
    mesh_shape = tuple(mesh_device.shape)
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    semaphore = ttnn.create_global_semaphore(mesh_device, cores, 0)

    """Gather tensor along specified dimension across mesh cluster axis."""
    tt_out_tensor = ttnn.experimental.all_gather_async(
        tt_input,
        mesh_device=mesh_device,
        cluster_axis=cluster_axis,
        dim=dim,
        multi_device_global_semaphore=semaphore,
        num_links=3,
        memory_config=memory_config,
        topology=ttnn.Topology.Linear,
    )
    return tt_out_tensor


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 128),
        ("prefill", 2048),
    ],
)
@pytest.mark.parametrize(
    "src_row",
    [
        0,
    ],
)
def test_row_to_mesh_broadcast(
    mesh_device,
    hf_config,
    mode,
    seq_len,
    src_row,
):
    torch.manual_seed(0)

    batch_size = 1
    mesh_shape = tuple(mesh_device.shape)

    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size)

    # Convert input to TTNN
    if mode == "decode":
        memory_config = ttnn.DRAM_MEMORY_CONFIG  # L1_MEMORY_CONFIG has PCC issue
    else:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_shape),
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # Perform broadcast operation
    tt_input = broadcast_row_to_mesh(tt_input, mesh_device, src_row)

    # Check that broadcast was successful
    _check_row_broadcast(src_row, tt_input, mesh_device, mesh_shape)

    # Compare outputs - at this point all devices should have the same data as the source row
    output_torch = ttnn.to_torch(
        tt_input,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_shape),
    )

    # Cleanup
    ttnn.deallocate(tt_input)

    print(f"Row broadcast test completed successfully for mode={mode}, seq_len={seq_len}, src_row={src_row}")


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 128),
        ("prefill", 2048),
    ],
)
@pytest.mark.parametrize(
    "src_row, dst_row",
    [
        (0, 1),
        (1, 2),
    ],
)
def test_row_to_row_transfer(
    mesh_device,
    hf_config,
    mode,
    seq_len,
    src_row,
    dst_row,
):
    torch.manual_seed(0)

    batch_size = 1
    mesh_shape = tuple(mesh_device.shape)

    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size)

    # Convert input to TTNN
    if mode == "decode":
        memory_config = ttnn.DRAM_MEMORY_CONFIG  # L1_MEMORY_CONFIG has PCC issue
    else:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_shape),
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # Perform row-to-row transfer operation
    tt_input = transfer_row(tt_input, mesh_device, src_row, dst_row)

    # Check that row-to-row transfer was successful
    _check_row_to_row_transfer(src_row, dst_row, tt_input, mesh_device, mesh_shape)

    # Compare outputs - at this point all devices should have the same data as the source row
    output_torch = ttnn.to_torch(
        tt_input,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_shape),
    )

    # Cleanup
    ttnn.deallocate(tt_input)

    print(
        f"Row-to-row transfer test completed successfully for mode={mode}, seq_len={seq_len}, src_row={src_row}, dest_row={dst_row}"
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 128),
        ("prefill", 2048),
    ],
)
def test_batch_partitioning_on_mesh(
    mesh_device,
    hf_config,
    mode,
    seq_len,
):
    torch.manual_seed(0)

    batch_size = 1
    mesh_shape = tuple(mesh_device.shape)

    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size)

    # Convert input to TTNN
    if mode == "decode":
        memory_config = ttnn.DRAM_MEMORY_CONFIG  # L1_MEMORY_CONFIG has PCC issue
    else:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=mesh_shape),
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # Partition batch on mesh
    tt_output = partition_batch_on_mesh(tt_input, -2, memory_config, cluster_axis=0)

    # Compare outputs - at this point all devices should have the same data as the source row
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_shape),
    )
    reference_output = torch_input
    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, 0.9999)
    assert passing, f"Batch partitioning test failed: {pcc_message} for mode={mode}, seq_len={seq_len}"


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 128),
        ("prefill", 2048),
    ],
)
def test_batch_gather_on_mesh(
    mesh_device,
    hf_config,
    mode,
    seq_len,
):
    torch.manual_seed(0)

    batch_size = 1
    mesh_shape = tuple(mesh_device.shape)

    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size)

    # Convert input to TTNN
    if mode == "decode":
        memory_config = ttnn.DRAM_MEMORY_CONFIG  # L1_MEMORY_CONFIG has PCC issue
    else:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_shape),
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = gather_batch_on_mesh(tt_input, mesh_device, -2, 0, memory_config)

    # Compare outputs - at this point all devices should have the same data as the source row
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_shape),
    )[0]
    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    passing, pcc_message = comp_pcc(torch_input, tt_output_torch, 0.9999)
    assert passing, f"Batch gather test failed: {pcc_message} for mode={mode}, seq_len={seq_len}"
