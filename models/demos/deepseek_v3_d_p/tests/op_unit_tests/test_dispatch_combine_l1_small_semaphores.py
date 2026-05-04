# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test that deepseek_prefill dispatch and combine global semaphores can be placed
in L1_SMALL to prevent L1 fragmentation.

Pattern (same as test_ccl_l1_small_semaphores.py):
1. Allocate tensor_a in L1 (~80% full)
2. Run dispatch+combine (creates global semaphores in L1 or L1_SMALL)
3. Free all tensors (global semaphores persist in program cache)
4. Try to allocate tensor_c in L1 (~100%)
   - L1 semaphores  -> OOM (semaphores fragment L1)
   - L1_SMALL sems  -> success (L1 is clean)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    extract_mesh_config,
    get_dispatch_input_mesh_mapper,
    get_gate_outputs,
    get_max_payload_size,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.utils.test_utils import print_l1_buffers, print_l1_small_buffers


def run_dispatch_op(mesh_device, use_l1_small):
    """Run dispatch op with small tensors in DRAM, creating 2 global semaphores."""
    torch.manual_seed(42)

    seq_len_per_chip = 64
    emb_dim = 256
    num_routed_experts = 16
    num_experts_per_tok = 2
    # ceil(N/2) of the most conservative integer N such that dgs*seq*N >= theoretical
    # worst-case dispatch buffer. Real traffic never approaches the worst case.
    dispatch_buffer_capacity_factor = 2

    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices,
        dispatch_group_size,
        dispatch_buffer_capacity_factor,
    )

    x, weights, indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
    )

    mesh_mapper = get_dispatch_input_mesh_mapper(mesh_device, sp_axis)

    tt_x = ttnn.from_torch(
        x, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_weights = ttnn.from_torch(
        weights, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
    )

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    ep_mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(1, 0))
    tt_expert_offsets = ttnn.from_torch(
        expert_offsets,
        mesh_mapper=ep_mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    tt_expert_token_counts = ttnn.from_torch(
        expert_token_counts,
        mesh_mapper=ep_mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    tt_expert_region_offsets = ttnn.from_torch(
        expert_region_offsets,
        mesh_mapper=ep_mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)

    dispatched_buffer, dispatch_metadata = ttnn.experimental.deepseek_prefill.dispatch(
        input_tensor=tt_x,
        weights_tensor=tt_weights,
        indices_tensor=tt_indices,
        expert_offsets_tensor=tt_expert_offsets,
        expert_dispatch_table_tensor=tt_expert_dispatch_table,
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        cluster_axis=sp_axis,
        num_links=1,
        topology=ttnn.Topology.Linear,
        use_l1_small_for_semaphores=use_l1_small,
    )
    ttnn.synchronize_device(mesh_device)

    # Return tensors for combine, plus intermediates to deallocate
    return (
        dispatched_buffer,
        dispatch_metadata,
        tt_expert_token_counts,
        tt_expert_region_offsets,
        dispatch_group_size,
        experts_per_chip,
        num_experts_per_tok,
        seq_len_per_chip,
        sp_axis,
        [tt_x, tt_weights, tt_indices, tt_expert_offsets, tt_expert_dispatch_table],
    )


def run_combine_op(
    mesh_device,
    dispatched_buffer,
    dispatch_metadata,
    tt_expert_token_counts,
    tt_expert_region_offsets,
    dispatch_group_size,
    experts_per_chip,
    num_experts_per_tok,
    seq_len_per_chip,
    sp_axis,
    use_l1_small,
):
    """Run combine op using dispatch outputs, creating 1 global semaphore."""
    output = ttnn.experimental.deepseek_prefill.combine(
        dispatched_buffer,
        dispatch_metadata,
        tt_expert_token_counts,
        tt_expert_region_offsets,
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=1,
        topology=ttnn.Topology.Linear,
        init_zeros=True,
        use_l1_small_for_semaphores=use_l1_small,
    )
    ttnn.synchronize_device(mesh_device)
    return output


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
                "l1_small_size": 512,
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="linear-4x1",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("ccl_op", ["dispatch", "combine"])
@pytest.mark.parametrize("use_l1_small", [False, True], ids=["l1_default", "l1_small"])
def test_deepseek_prefill_l1_small_semaphores(mesh_device, device_params, ccl_op, use_l1_small):
    """Test that dispatch/combine semaphores can be placed in L1_SMALL to prevent L1 fragmentation."""

    compute_grid = mesh_device.compute_with_storage_grid_size()
    num_worker_cores = compute_grid.x * compute_grid.y
    logger.info(f"Compute grid: {compute_grid.x}x{compute_grid.y} = {num_worker_cores} cores")

    # Tensor A: 0.8MB * num_worker_cores in L1
    tensor_a_bytes = int(0.8 * 1024 * 1024) * num_worker_cores
    tensor_a_elements = tensor_a_bytes // 2  # bfloat16
    tensor_a_cols = tensor_a_elements // 32
    logger.info(f"Tensor A: shape [1, 1, 32, {tensor_a_cols}], {tensor_a_bytes} bytes total")

    tensor_a = ttnn.from_torch(
        torch.randn(1, 1, 32, tensor_a_cols),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    print_l1_buffers(mesh_device, f"before_{ccl_op}")
    print_l1_small_buffers(mesh_device, f"before_{ccl_op}")

    # Run the op(s) — global semaphores are created in L1 or L1_SMALL
    logger.info(f"Running {ccl_op} with use_l1_small_for_semaphores={use_l1_small}")

    tensors_to_free = []

    # Run dispatch to produce tensors (needed by both dispatch and combine tests).
    # Always use the same l1_small flag so dispatch semaphores don't pollute L1.
    (
        dispatched_buffer,
        dispatch_metadata,
        tt_expert_token_counts,
        tt_expert_region_offsets,
        dispatch_group_size,
        experts_per_chip,
        num_experts_per_tok,
        seq_len_per_chip,
        sp_axis,
        dispatch_intermediates,
    ) = run_dispatch_op(mesh_device, use_l1_small)
    tensors_to_free.extend(dispatch_intermediates)

    if ccl_op == "combine":
        combine_output = run_combine_op(
            mesh_device,
            dispatched_buffer,
            dispatch_metadata,
            tt_expert_token_counts,
            tt_expert_region_offsets,
            dispatch_group_size,
            experts_per_chip,
            num_experts_per_tok,
            seq_len_per_chip,
            sp_axis,
            use_l1_small,
        )
        tensors_to_free.append(combine_output)

    tensors_to_free.extend([dispatched_buffer, dispatch_metadata, tt_expert_token_counts, tt_expert_region_offsets])

    print_l1_buffers(mesh_device, f"after_{ccl_op}")
    print_l1_small_buffers(mesh_device, f"after_{ccl_op}")

    # Free tensor A and all op tensors
    tensor_a.deallocate(True)
    for t in tensors_to_free:
        t.deallocate(True)

    print_l1_buffers(mesh_device, "before_tensor_c")
    print_l1_small_buffers(mesh_device, "before_tensor_c")

    # Tensor C: 1.0MB * num_worker_cores in L1
    # With L1 semaphores: should fail (OOM — semaphores fragment L1)
    # With L1_SMALL semaphores: should succeed (semaphores in L1_SMALL, L1 is clean)
    tensor_c_bytes = int(1.0 * 1024 * 1024) * num_worker_cores
    tensor_c_elements = tensor_c_bytes // 2
    tensor_c_cols = tensor_c_elements // 32
    logger.info(f"Tensor C: shape [1, 1, 32, {tensor_c_cols}], {tensor_c_bytes} bytes total")

    if not use_l1_small:
        with pytest.raises(RuntimeError):
            ttnn.from_torch(
                torch.randn(1, 1, 32, tensor_c_cols),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
    else:
        ttnn.from_torch(
            torch.randn(1, 1, 32, tensor_c_cols),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        print_l1_buffers(mesh_device, "after_tensor_c")
        print_l1_small_buffers(mesh_device, "after_tensor_c")
