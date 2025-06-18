# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from tests.nightly.tg.ccl.test_all_to_all_dispatch import (
    PACKET_WORKER_CRS,
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

    input_contribs_tensor = sparse_tokens.reshape([devices, 1, batch, hidden_size]).repeat(
        [1, experts_per_device, 1, 1]
    )

    for d in range(devices):
        experts_on_device = _get_experts_on_device(experts, expert_mapping, d)
        for b in range(batch):
            for k in range(selected_experts_k):
                expert_idx = expert_indices[b, 0, 0, k].item()

                if expert_idx not in experts_on_device:
                    continue

                local_expert_idx = expert_idx % experts_per_device

                # multiply by expert index to mock application of expert
                input_contribs_tensor[d, local_expert_idx, b, :] *= expert_idx

    return input_contribs_tensor


def get_output_combined_contribs(sparse_contribs, expert_indices, expert_mapping, mesh_shape):
    # output recalled contribs is [K, batch * mesh_shape[0], 1, hidden]

    batch = expert_indices.shape[0]
    experts = expert_mapping.shape[-2]
    selected_experts_k = expert_indices.shape[-1]
    hidden = sparse_contribs.shape[-1]
    mesh_dim = mesh_shape[0]
    devices = mesh_shape[0] * mesh_shape[1]

    assert experts % devices == 0
    experts_per_device = experts // devices

    output_combined_contribs_tensor = torch.zeros(selected_experts_k, batch * mesh_dim, 1, hidden)

    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            d = m0 * mesh_shape[0] + m1
            experts_on_device = _get_experts_on_device(experts, expert_mapping, d)
            for b in range(batch):
                for k in range(selected_experts_k):
                    expert_idx = expert_indices[b, 0, 0, k].item()

                    if expert_idx not in experts_on_device:
                        continue
                    local_expert_idx = expert_idx % experts_per_device

                    output_combined_contribs_tensor[k, m0 * batch + b, 0, :] = sparse_contribs[
                        d, local_expert_idx, b, :
                    ]
    return output_combined_contribs_tensor


def gen_tensors(batch, experts, selected_experts_k, hidden_size, mesh_shape, devices, scheme="random"):
    torch.manual_seed(42)
    # create input tokens
    assert batch % devices == 0
    assert experts % devices == 0
    assert selected_experts_k < experts

    input_tokens = gen_tokens(batch, hidden_size, mesh_shape, devices, scheme)
    expert_mapping = gen_expert_mapping(experts, devices, scheme)
    expert_indices = get_expert_indices(batch, experts, selected_experts_k, scheme)

    sparse_dispatched_tokens = get_sparse_tokens(input_tokens, expert_indices, expert_mapping, mesh_shape)
    input_sparse_contribs_tensor = get_input_sparse_contribs(sparse_dispatched_tokens, expert_indices, expert_mapping)

    output_tensor = get_output_combined_contribs(
        input_sparse_contribs_tensor, expert_indices, expert_mapping, mesh_shape
    )

    metadata_tensor = get_metadata_tensor(expert_indices, expert_mapping, mesh_shape)

    # create expert indices
    return sparse_dispatched_tokens, input_sparse_contribs_tensor, expert_mapping, metadata_tensor, output_tensor


def run_all_to_all_combine_test(
    mesh_device,
    mesh_shape,
    batch,
    experts,
    select_experts_k,
    hidden_size,
    num_iters,
    num_links=1,  # currently not passed through
    scheme="random",
    use_regular_grid=False,
    input_grid=None,
    output_grid=None,
    dtype=ttnn.bfloat16,
    topology=ttnn.Topology.Linear,
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    mesh_device.enable_program_cache()
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
        _, input_contrib, expert_mapping, metadata_tensor, output_contrib_tensor = gen_tensors(
            batch, experts, select_experts_k, hidden_size, mesh_shape, devices, scheme=scheme
        )

        output_tensor_goldens_list.append(output_contrib_tensor)

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

    # worker_sub_device_id = ttnn.SubDeviceId(0)
    #     sub_device_stall_group = [worker_sub_device_id]
    #     sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    #     mesh_device.load_sub_device_manager(sub_device_manager)
    #     mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]

    tt_out_tensor_list = []

    def run_op(n_iters, store_all_results=True):
        tt_output_list = []
        tt_metadata_list = []

        for i in range(n_iters):
            tt_out_tensor = ttnn.all_to_all_combine(
                input_tensors[i],
                expert_mapping_tensors[i],
                metadata_tensors[i],
                num_links=num_links,
                topology=topology,
                memory_config=output_memory_config,
                global_semaphore=ccl_semaphore_handles[i],
                axis=0
                # subdevice_id=worker_sub_device_id,
            )

            ttnn.synchronize_device(mesh_device)
            if store_all_results:
                tt_output_list.append(tt_out_tensor)
                tt_metadata_list.append(tt_metadata)
        if store_all_results:
            return tt_output_list, tt_metadata_list
        else:
            return [tt_out_tensor], [tt_metadata]

    tt_out_tensor_list = run_op(num_iters, store_all_results=True)

    #     mesh_device.reset_sub_device_stall_group()

    # passed = True
    #     metadata_passed = True
    #     first_failed_tensor_index = None
    #     first_failed_metadata_index = None
    #     failed_indices = []
    #     failed_metadata_indices = []
    #     expected_pcc = 0.9999 if dtype == ttnn.bfloat8_b else 0.999990

    for tt_out, ref in zip(tt_out_tensor_list, output_tensor_goldens_list):
        assert_with_pcc(ttnn.to_torch(tt_out), ref)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
def test_all_to_all_combine_no_trace(mesh_device, mesh_shape):
    devices = mesh_shape[0] * mesh_shape[1]
    batch = 1 * devices
    experts = 2 * devices
    select_experts_k = 4
    hidden_size = 32
    num_iters = 1
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    num_links = 1
    topology = ttnn.Topology.Linear

    run_all_to_all_combine_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        num_iters,
        num_links=num_links,
        scheme="sequential",
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
    batch = 1 * devices
    experts = 2 * devices
    select_experts_k = 2
    hidden_size = 32
    mesh_dim = mesh_shape[0]
    (
        sparse_dispatched_tokens,
        input_sparse_contribs_tensor,
        expert_mapping,
        metadata_tensor,
        output_tensor,
    ) = gen_tensors(batch, experts, select_experts_k, hidden_size, mesh_shape, devices, scheme="sequential")

    print(f"{expert_mapping=}")
    print(f"{metadata_tensor=}")

    assert sparse_dispatched_tokens.shape == (devices, batch, 1, hidden_size)
    assert input_sparse_contribs_tensor.shape == (devices, experts // devices, batch, hidden_size)
    assert output_tensor.shape == (select_experts_k, batch * mesh_dim, 1, hidden_size)
    assert expert_mapping.shape == (1, 1, experts, devices)
    assert metadata_tensor.shape == (devices, batch, 1, select_experts_k)

    assert_with_pcc(output_tensor, output_tensor)
