# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from tests.ttnn.unit_tests.operations.ccl.test_all_to_all_dispatch_t3000 import (
    gen_tokens,
    gen_expert_mapping,
    get_metadata_tensor,
    get_expert_indices,
    get_output_tensor as get_sparse_tokens,
)


def _get_experts_on_device(num_experts, expert_mapping, device):
    return [e for e in range(num_experts) if expert_mapping[0, 0, e, device] == 1]


def get_input_sparse_contribs(sparse_tokens, expert_indices, expert_mapping):
    # sparse tokens is [devices, batch, 1, hidden_size]
    # desired expert contributions tensor is [devices, experts/devices, batch, hidden_size]
    # we'll multiply the tokens by the index of their assigned expert to mock expert application.

    batch = expert_indices.shape[0]
    devices = expert_mapping.shape[-1]
    experts = expert_mapping.shape[-2]
    hidden_size = sparse_tokens.shape[-1]
    selected_experts_k = expert_indices.shape[-1]

    assert experts % devices == 0
    experts_per_device = experts // devices

    input_contribs_tensor = sparse_tokens.permute([0, 2, 1, 3]).repeat([1, experts_per_device, 1, 1])

    token_expert_count = 0
    for d in range(devices):
        experts_on_device = _get_experts_on_device(experts, expert_mapping, d)
        assert len(experts_on_device) == experts_per_device
        for b in range(batch):
            for k in range(selected_experts_k):
                expert_idx = expert_indices[b, 0, 0, k].item()

                if expert_idx not in experts_on_device:
                    continue

                local_expert_idx = experts_on_device.index(expert_idx)

                # multiply by expert index to mock application of expert
                input_contribs_tensor[d, local_expert_idx, b, :] = sparse_tokens[d, b, 0, :] * (
                    -1 if expert_idx == 0 else expert_idx
                )
                token_expert_count += 1

    assert token_expert_count == batch * selected_experts_k

    return input_contribs_tensor


def get_output_combined_contribs(sparse_contribs, expert_indices, expert_mapping, mesh_shape, replication_axis):
    # output recalled contribs is [K, batch * mesh_shape[0], 1, hidden]
    batch = expert_indices.shape[0]
    experts = expert_mapping.shape[-2]
    selected_experts_k = expert_indices.shape[-1]
    hidden = sparse_contribs.shape[-1]

    if replication_axis == 1:
        replication_dim = mesh_shape[0]
        replication_group = mesh_shape[1]
    elif replication_axis == 0:
        replication_dim = mesh_shape[1]
        replication_group = mesh_shape[0]
    else:
        assert replication_axis == -1
        replication_dim = 1
        replication_group = devices

    devices = mesh_shape[0] * mesh_shape[1]

    batch_per_device = batch // replication_group

    assert experts % devices == 0
    experts_per_device = experts // devices

    output_combined_contribs_tensor = torch.ones(selected_experts_k, batch * replication_dim, 1, hidden) * 7
    real_data_map = torch.zeros(output_combined_contribs_tensor.shape[:-2])

    def _rep_idx(m0, m1, b):
        if replication_axis == 0:
            return m1 * batch + b
        elif replication_axis == 1:
            return m0 * batch + b
        else:
            return b

    token_expert_count = 0
    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            d = m0 * mesh_shape[1] + m1
            experts_on_device = _get_experts_on_device(experts, expert_mapping, d)
            for b in range(batch):
                for k in range(selected_experts_k):
                    expert_idx = expert_indices[b, 0, 0, k].item()

                    if expert_idx not in experts_on_device:
                        continue

                    local_expert_idx = experts_on_device.index(expert_idx)
                    dense_batch_idx = _rep_idx(m0, m1, b)

                    output_combined_contribs_tensor[k, dense_batch_idx, 0, :] = sparse_contribs[
                        d, local_expert_idx, b, :
                    ]

                    real_data_map[k, dense_batch_idx] = 1
                    token_expert_count += 1

    assert token_expert_count == batch * selected_experts_k
    return output_combined_contribs_tensor, real_data_map


def gen_tensors(
    batch, experts, selected_experts_k, hidden_size, mesh_shape, replication_axis, devices, scheme="random"
):
    torch.manual_seed(42)
    # create input tokens
    assert batch % devices == 0
    assert experts % devices == 0
    assert selected_experts_k < experts

    seq = 1

    input_tokens = gen_tokens(batch, hidden_size, seq, mesh_shape, devices, scheme)
    expert_mapping = gen_expert_mapping(experts, devices, scheme)
    expert_indices = get_expert_indices(batch, experts, selected_experts_k, seq, mesh_shape, scheme)

    sparse_dispatched_tokens = get_sparse_tokens(input_tokens, expert_indices, expert_mapping, seq, mesh_shape)
    input_sparse_contribs_tensor = get_input_sparse_contribs(sparse_dispatched_tokens, expert_indices, expert_mapping)

    output_tensor, data_map = get_output_combined_contribs(
        input_sparse_contribs_tensor, expert_indices, expert_mapping, mesh_shape, replication_axis
    )

    metadata_tensor = get_metadata_tensor(expert_indices, expert_mapping, mesh_shape)

    # create expert indices
    return (
        sparse_dispatched_tokens,
        input_sparse_contribs_tensor,
        expert_mapping,
        metadata_tensor,
        output_tensor,
        data_map,
    )


def run_all_to_all_combine_test(
    mesh_device,
    mesh_shape,
    axis,
    batch,
    experts,
    select_experts_k,
    hidden_size,
    num_iters,
    num_links,
    scheme="random",
    use_regular_grid=False,
    input_grid=None,
    output_grid=None,
    dtype=ttnn.bfloat16,
    topology=ttnn.Topology.Linear,
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
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

    expert_mapping_tensors = []
    input_tensors = []
    metadata_tensors = []

    output_tensor_goldens_list = []

    for iter in range(num_iters):
        _, input_contrib, expert_mapping, metadata_tensor, output_contrib_tensor, data_map = gen_tensors(
            batch, experts, select_experts_k, hidden_size, mesh_shape, axis, devices, scheme=scheme
        )

        output_tensor_goldens_list.append((output_contrib_tensor, data_map))

        tt_input_contribs = ttnn.from_torch(
            input_contrib,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        tt_expert_mapping = ttnn.from_torch(
            expert_mapping,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
        )

        tt_metadata = ttnn.from_torch(
            metadata_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        input_tensors.append(tt_input_contribs)
        expert_mapping_tensors.append(tt_expert_mapping)
        metadata_tensors.append(tt_metadata)

    ccl_sub_device_crs = subdevice_shard_cores_grid
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )

    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]

    tt_out_tensor_list = []

    def run_op(n_iters, store_all_results=True):
        tt_output_list = []

        for i in range(n_iters):
            tt_out_tensor = ttnn.all_to_all_combine(
                input_tensors[i],
                expert_mapping_tensors[i],
                metadata_tensors[i],
                num_links=num_links,
                topology=topology,
                memory_config=output_memory_config,
                global_semaphore=ccl_semaphore_handles[i],
                axis=axis,
            )

            ttnn.synchronize_device(mesh_device)
            if store_all_results:
                tt_output_list.append(tt_out_tensor)
        if store_all_results:
            return tt_output_list
        else:
            return [tt_out_tensor]

    tt_out_tensor_list = run_op(num_iters, store_all_results=True)

    for tt_out, (ref, data_map) in zip(tt_out_tensor_list, output_tensor_goldens_list):
        if axis == 0:
            # need to roll my own mesh composer here for the transposed ordering
            device_shards = [ttnn.to_torch(ittout, mesh_composer=None) for ittout in ttnn.get_device_tensors(tt_out)]
            ordered_shards = []
            for ir in range(mesh_shape[1]):
                for ic in range(mesh_shape[0]):
                    ordered_shards.append(device_shards[ic * mesh_shape[1] + ir])
            tt_out_agg = torch.cat(ordered_shards, dim=1)

        else:
            tt_out_agg = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
        check_results(tt_out_agg, ref, data_map)


def check_results(test_tensor, ref_tensor, data_map):
    for k in range(ref_tensor.shape[0]):
        for b in range(ref_tensor.shape[1]):
            if data_map[k, b].item() == 1:
                assert_with_pcc(test_tensor[k, b, 0, :], ref_tensor[k, b, 0, :])


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7, 16, 32, 33, 50, 7000])
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_combine_no_trace(
    mesh_device,
    mesh_shape,
    axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    num_iters,
    input_memory_config,
    output_memory_config,
    num_links,
    topology,
    dtype,
):
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    run_all_to_all_combine_test(
        mesh_device,
        mesh_shape,
        axis,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        num_iters,
        num_links=num_links,
        scheme="random",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
    )


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
def test_simple_tensor_gen(mesh_device, mesh_shape):
    torch.set_printoptions(threshold=10000)
    devices = mesh_shape[0] * mesh_shape[1]
    batch = 8 * devices
    experts = 8 * devices
    select_experts_k = 8
    hidden_size = 7000
    axis = 0
    mesh_dim = mesh_shape[0 if axis == 1 else 1]
    (
        sparse_dispatched_tokens,
        input_sparse_contribs_tensor,
        expert_mapping,
        metadata_tensor,
        output_tensor,
        _,
    ) = gen_tensors(batch, experts, select_experts_k, hidden_size, mesh_shape, axis, devices, scheme="random")

    assert sparse_dispatched_tokens.shape == (devices, batch, 1, hidden_size)
    assert input_sparse_contribs_tensor.shape == (devices, experts // devices, batch, hidden_size)
    assert output_tensor.shape == (select_experts_k, batch * mesh_dim, 1, hidden_size)
    assert expert_mapping.shape == (1, 1, experts, devices)
    assert metadata_tensor.shape == (devices, batch, 1, select_experts_k)
