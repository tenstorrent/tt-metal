# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.ccl.test_all_to_all_combine_t3000 import (
    check_results,
    get_output_combined_contribs,
    get_input_sparse_contribs,
)
from tests.ttnn.unit_tests.operations.ccl.test_all_to_all_dispatch_t3000 import (
    gen_tokens,
    gen_expert_mapping,
    get_metadata_tensor,
    get_expert_indices,
    get_output_tensor as get_sparse_tokens,
)


def gen_tensors_integration(
    batch, experts, selected_experts_k, hidden_size, seq, mesh_shape, replication_axis, devices, scheme="random"
):
    torch.manual_seed(42)
    # create input tokens
    assert batch % devices == 0
    assert experts % devices == 0
    assert selected_experts_k < experts

    input_tokens = gen_tokens(batch, hidden_size, seq, mesh_shape, devices, scheme)
    expert_mapping = gen_expert_mapping(experts, devices, scheme)
    expert_indices = get_expert_indices(batch, experts, selected_experts_k, seq, mesh_shape, scheme)

    sparse_dispatched_tokens = get_sparse_tokens(input_tokens, expert_indices, expert_mapping, seq, mesh_shape)
    input_sparse_contribs_tensor = get_input_sparse_contribs(
        sparse_dispatched_tokens, expert_indices, expert_mapping, mesh_shape, replication_axis, apply_fake_expert=False
    )

    output_tensor, data_map = get_output_combined_contribs(
        input_sparse_contribs_tensor, expert_indices, expert_mapping, mesh_shape, replication_axis
    )

    metadata_tensor = get_metadata_tensor(expert_indices, expert_mapping, mesh_shape)

    # create expert indices
    return (
        input_tokens,
        expert_indices,
        sparse_dispatched_tokens,
        input_sparse_contribs_tensor,
        expert_mapping,
        metadata_tensor,
        output_tensor,
        data_map,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
def test_integration(mesh_device, mesh_shape):
    mesh_device.enable_program_cache()
    devices = mesh_shape[0] * mesh_shape[1]
    batch = 1 * devices
    seq_len = 1
    experts = 2 * devices
    select_experts_k = 8
    hidden_size = 16
    num_iters = 1
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    axis = 1
    dtype = ttnn.bfloat16

    devices = mesh_shape[0] * mesh_shape[1]
    # input, output, interm core range set
    compute_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    subdevice_shard_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(compute_grid[0] - 1, compute_grid[1] - 1),
            ),
        }
    )

    (
        input_tokens,
        expert_indices,
        sparse_dispatched_tokens,
        input_sparse_contribs_tensor,
        expert_mapping,
        metadata_tensor,
        output_tensor,
        data_map,
    ) = gen_tensors_integration(
        batch, experts, select_experts_k, hidden_size, seq_len, mesh_shape, axis, devices, scheme="random"
    )

    preallocated_output_tensor = torch.zeros((devices, batch, seq_len, hidden_size), dtype=torch.bfloat16)
    preallocated_metadata_tensor = torch.zeros((devices, batch, seq_len, select_experts_k), dtype=torch.int32)

    if axis is None:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    elif axis == 1:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=mesh_shape)
    elif axis == 0:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_shape)

    ccl_sub_device_crs = subdevice_shard_cores_grid
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    global_semaphore1 = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
    global_semaphore2 = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

    ## INPUTS TO DISPATCH ##
    # [batch(/devices),1,1,hidden]
    tt_input = ttnn.from_torch(
        input_tokens,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=input_memory_config,
        mesh_mapper=mesh_mapper,
    )

    # [batch,1,seq,k]
    tt_expert_indices = ttnn.from_torch(
        expert_indices,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=mesh_mapper,
    )
    # [devices(/devices),1,experts, devices]
    tt_expert_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    num_links = 1
    topology = ttnn.Topology.Linear
    ## OUTPUTS FROM DISPATCH ##
    # [devices (/devices), batch, seq, hidden], [devices(/devices),1,experts, devices]
    tt_output_tensor, tt_metadata_tensor = ttnn.all_to_all_dispatch(
        tt_input,
        tt_expert_indices,
        tt_expert_mapping,
        num_links=num_links,
        topology=topology,
        cluster_axis=axis,
        memory_config=output_memory_config,
        global_semaphore=global_semaphore1,
        subdevice_id=worker_sub_device_id,
    )
    ttnn.synchronize_device(mesh_device)

    ### Transform data such that it resembles output from experts MLP layers ###
    #
    # tt_output_tensor is [devices (/devices), batch * replicate_dim, seq, hidden_size]
    # Simulate contributions from experts on each device.
    # [experts (/devices), batch*replicate_dim, seq, hidden]
    repeated_sparse_output_tensor = ttnn.repeat(tt_output_tensor, [experts // devices, 1, 1, 1])

    ### COMBINE OUTPUT ###
    # [k, batch (/devices), seq hidden]
    tt_out_tensor = ttnn.all_to_all_combine(
        repeated_sparse_output_tensor,
        tt_expert_mapping,
        tt_metadata_tensor,
        num_links=num_links,
        topology=topology,
        memory_config=output_memory_config,
        global_semaphore=global_semaphore2,
        axis=axis,
        subdevice_id=worker_sub_device_id,
    )

    torch_final_output = ttnn.to_torch(tt_out_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
    check_results(torch_final_output, output_tensor, data_map)
