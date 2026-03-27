# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TG (Single Galaxy) MoE Combine Test for Quad Galaxy Validation

Tests ttnn.experimental.selective_reduce_combine operation in isolation on a 4x8 mesh (32 devices).
Based on tests/nightly/tg/ccl/moe/test_selective_combine_6U.py adapted for quad validation.

Key Design Principles:
1. **Same Work Per Device**: Each device handles 2 experts (same as quad: 256/128 = 2)
2. **Isolated Testing**: Tests only the combine operation with mocked compute outputs
3. **All-to-All Gather**: Validates expert outputs are correctly gathered back to source devices

Configuration:
- Mesh: 4x8 (32 devices)
- Experts: 64 (2 per device)
- Batch: 64 (reduced from quad's 512 due to L1 constraints)
- cluster_axis: 0 (dispatch along axis-0)
"""

import os
import random

import pytest
import torch
from loguru import logger

import ttnn

# Import helper functions from the reference test
from tests.nightly.tg.ccl.moe.test_selective_combine_6U import (
    _get_tt_dense_metadata,
    _get_tt_dense_token_maps,
    _get_tt_sharded_dense_input,
    gen_tensors,
)


def run_combine_test(
    mesh_device,
    mesh_shape,
    batch,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq,
    cluster_axis,
    worker_core_range,
    token_parallel_core_dim,
    data_parallel_core_dim,
    num_links,
    mux_core_range,
    num_test_iters,
):
    """
    Run MoE combine test on TG 4x8 mesh.

    This test:
    1. Generates mock compute outputs (dense contributions from experts)
    2. Generates activation metadata (which tokens activated which experts)
    3. Runs selective_reduce_combine to gather expert outputs
    4. Verifies outputs against golden references
    """
    torch.manual_seed(42)
    random.seed(42)

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis]
    num_replicated_devices = num_devices // num_dispatch_devices
    experts = experts_per_device * num_devices

    # NOTE: gen_tensors expects batch to be per-replicated-group, not total batch
    # For cluster_axis=0 with 4x8 mesh, we have 8 replicated groups
    # If we want 64 total tokens, we pass 64/8 = 8 tokens per group
    batch_per_group = batch // num_replicated_devices

    logger.info("=" * 80)
    logger.info(f"TG MoE Combine Test Configuration:")
    logger.info(f"  Mesh shape: {mesh_shape} ({num_devices} devices)")
    logger.info(f"  Cluster axis: {cluster_axis}")
    logger.info(f"  Batch: {batch} total ({batch_per_group} per replicated group)")
    logger.info(f"  Seq: {seq}")
    logger.info(f"  Experts: {experts} ({experts_per_device} per device)")
    logger.info(f"  Selected experts K: {select_experts_k}")
    logger.info(f"  Hidden size: {hidden_size}")
    logger.info(f"  Num replicated devices: {num_replicated_devices}")
    logger.info("=" * 80)

    # Create core ranges
    worker_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in worker_core_range])])
    mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in mux_core_range])])

    assert worker_cores.num_cores() == token_parallel_core_dim * data_parallel_core_dim, (
        f"Worker cores {worker_cores.num_cores()} != "
        f"token_parallel_dim * data_parallel_dim {token_parallel_core_dim * data_parallel_core_dim}"
    )

    # Generate test tensors and golden references
    # Pass batch_per_group since gen_tensors expects per-replicated-group batch
    (
        dense_metadata_tensor,
        dense_token_maps,
        dense_token_counts_tensor,
        dense_contribs_tensor,
        output_ref,
        output_data_map,
    ) = gen_tensors(
        batch_per_group,
        experts,
        select_experts_k,
        hidden_size,
        seq,
        mesh_shape,
        cluster_axis,
        num_devices,
        token_parallel_core_dim,
        scheme="random",
    )

    # Create TT tensors
    tt_dense_contribs = _get_tt_sharded_dense_input(
        dense_contribs_tensor,
        worker_cores,
        token_parallel_core_dim,
        data_parallel_core_dim,
        mesh_device,
        cluster_axis,
    )

    tt_dense_metadata = _get_tt_dense_metadata(dense_metadata_tensor, mesh_device)
    tt_dense_token_maps = _get_tt_dense_token_maps(dense_token_maps, mesh_device)

    tt_token_counts = ttnn.from_torch(
        dense_token_counts_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Create output tensor
    output_tensor = torch.zeros([select_experts_k, batch * seq, hidden_size], dtype=torch.bfloat16)
    tt_output_tensor = ttnn.from_torch(
        output_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )

    # Create barrier semaphore
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)

    # Convert CoreRangeSet to list of CoreCoords (API expects Sequence[CoreCoord])
    core_list = list(ttnn.corerange_to_cores(worker_cores))

    logger.info("Running combine operation...")

    # Run operation multiple times
    for i in range(num_test_iters):
        tt_out = ttnn.experimental.selective_reduce_combine(
            tt_dense_contribs,
            tt_dense_metadata,
            tt_dense_token_maps,
            tt_token_counts,
            hidden_size,
            batch,
            seq,
            select_experts_k,
            experts,
            cluster_axis,
            topology=ttnn.Topology.Ring,
            num_links=num_links,
            token_parallel_core_dim=token_parallel_core_dim,
            data_parallel_core_dim=data_parallel_core_dim,
            worker_cores=core_list,
            mux_core_range_set=mux_cores,
            output_tensor=tt_output_tensor,
            optional_cross_device_semaphore=barrier_semaphore,
        )

        logger.info(f"Iteration {i + 1}/{num_test_iters} completed")

    ttnn.synchronize_device(mesh_device)
    logger.info("All iterations completed")

    # Validate output
    logger.info("Validating output...")

    if cluster_axis == 0:
        # Custom mesh composer for cluster_axis=0 (transposed ordering)
        device_shards = [ttnn.to_torch(shard, mesh_composer=None) for shard in ttnn.get_device_tensors(tt_out)]
        ordered_shards = []
        for ir in range(mesh_shape[1]):  # For each column
            for ic in range(mesh_shape[0]):  # For each row
                ordered_shards.append(device_shards[ic * mesh_shape[1] + ir])
        tt_out_agg = torch.cat(ordered_shards, dim=1)
    else:
        tt_out_agg = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))

    # Check shapes match
    assert tt_out_agg.shape == output_ref.shape, f"Shape mismatch: tt_out {tt_out_agg.shape} != ref {output_ref.shape}"

    # Verify each element where data is valid
    all_passed = True
    mismatches = []
    for k in range(tt_out_agg.shape[0]):
        for t in range(tt_out_agg.shape[1]):
            if output_data_map[k, t].item() == 1:
                if not torch.equal(tt_out_agg[k, t, :], output_ref[k, t, :]):
                    all_passed = False
                    mismatches.append((k, t))
                    if len(mismatches) <= 10:  # Log first 10 mismatches
                        logger.warning(
                            f"Mismatch at k={k}, t={t}: " f"expected {output_ref[k, t, :5]}, got {tt_out_agg[k, t, :5]}"
                        )

    if mismatches:
        logger.warning(f"Total mismatches: {len(mismatches)}")

    logger.info(f"TG MoE Combine Test: {'PASSED' if all_passed else 'FAILED'}")
    assert all_passed, f"TG MoE Combine test failed with {len(mismatches)} mismatches!"
    logger.info("✓ TG MoE Combine test passed!")


@pytest.mark.requires_device("TG")
@pytest.mark.skip(
    reason="Metadata tensor format incompatibility with cluster_axis=0. "
    "The gen_tensors helper from 1x8/1x16 tests generates metadata in a format "
    "incompatible with 4x8 mesh (cluster_axis=0). Combine is still tested in E2E test."
)
@pytest.mark.skipif(
    (os.getenv("USE_TORUS_MODE") is None),
    reason="Requires ring fabric",
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((4, 8), (4, 8), id="4x8_tg"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("batch", [64])  # Reduced from 512 due to L1 constraints
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("experts_per_device", [2])
@pytest.mark.parametrize("worker_core_range", [((0, 0), (3, 3))])
@pytest.mark.parametrize("token_parallel_core_dim", [4])
@pytest.mark.parametrize("data_parallel_core_dim", [4])
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("mux_core_range", [((4, 0), (5, 7))])
@pytest.mark.parametrize("num_test_iters", [3])
def test_combine_correctness(
    mesh_device,
    mesh_shape,
    batch,
    select_experts_k,
    hidden_size,
    seq,
    cluster_axis,
    experts_per_device,
    worker_core_range,
    token_parallel_core_dim,
    data_parallel_core_dim,
    num_links,
    mux_core_range,
    num_test_iters,
):
    """Correctness test for TG selective_reduce_combine operation."""
    run_combine_test(
        mesh_device,
        mesh_shape,
        batch,
        experts_per_device,
        select_experts_k,
        hidden_size,
        seq,
        cluster_axis,
        worker_core_range,
        token_parallel_core_dim,
        data_parallel_core_dim,
        num_links,
        mux_core_range,
        num_test_iters,
    )
