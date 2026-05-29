# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for ttnn.masked_bincount operation in isolation.

Verifies that the TTNN masked_bincount (per-device expert histogram) matches
a simple PyTorch reference: torch.bincount masked by expert presence.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, extract_mesh_config, get_ep_mesh_composer
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import compare_exact, validate_composed
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_validation_results


def torch_masked_bincount(
    indices: torch.Tensor, expert_dispatch_table: torch.Tensor, n_routed_experts: int
) -> torch.Tensor:
    """
    Reference implementation: count expert occurrences masked by expert_dispatch_table.

    Args:
        indices: [sp_dim, topk] int tensor of expert indices.
        expert_dispatch_table: [n_routed_experts] int32 tensor (>= 0 = present, -1 = absent).
        n_routed_experts: number of experts (output size).

    Returns:
        [n_routed_experts] int tensor of masked histogram counts.
    """
    counts = torch.bincount(indices.flatten().to(torch.int64), minlength=n_routed_experts).to(torch.int32)
    mask = (expert_dispatch_table >= 0).to(torch.int32)
    return counts[:n_routed_experts] * mask


@pytest.mark.parametrize(
    "sp_dim, topk, n_routed_experts",
    [
        (4096, 8, 256),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 1),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 1), topology="linear"),
            id="single",
        ),
        pytest.param(
            (1, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 2), topology="linear"),
            id="linear-1x2",
        ),
        pytest.param(
            (1, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="mesh-1x4"),
            id="linear-1x4",
        ),
        pytest.param(
            (2, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_masked_bincount(
    mesh_device,
    sp_dim,
    topk,
    n_routed_experts,
):
    """Test ttnn.masked_bincount against PyTorch reference on each device."""
    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    assert sp_axis == 0

    logger.info(
        f"Testing masked_bincount: {mesh_device.shape=}, {sp_dim=}, {topk=}, "
        f"{n_routed_experts=}, {num_dispatch_groups=}"
    )
    ttnn.visualize_mesh_device(mesh_device)

    torch.manual_seed(42)

    # Generate expert indices as 2D [total_sp, topk] — sharded across dispatch axis, each device sees [sp_dim, topk]
    total_sp = dispatch_group_size * sp_dim
    indices = torch.randint(0, n_routed_experts, (total_sp, topk), dtype=torch.int32)

    # Expert dispatch table: maps expert_id -> chip_id (>= 0) or -1 (absent)
    dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=n_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    # Compute torch reference per (dispatch_group, chip) pair
    # torch_histograms[group][chip] = histogram for that group/chip combination
    torch_histograms = {}
    for group in range(num_dispatch_groups):
        for chip in range(dispatch_group_size):
            chip_indices = indices[chip * sp_dim : (chip + 1) * sp_dim]
            hist = torch_masked_bincount(chip_indices, dispatch_table[group], n_routed_experts)
            torch_histograms[(group, chip)] = hist

    # Height-shard config matching the gate module pattern: 8x8 core grid
    num_cores = 64
    assert sp_dim % num_cores == 0, f"sp_dim={sp_dim} must be divisible by {num_cores}"
    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(sp_dim // num_cores, topk),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    # Shard dim 0 across the dispatch axis, replicate across TP axis
    mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None) if sp_axis == 0 else (None, 0),
    )

    # UINT16 as required by the kernel
    tt_indices = ttnn.from_torch(
        indices.to(torch.int16),
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint16,
    )
    tt_indices = ttnn.to_memory_config(tt_indices, sharded_mem_config)

    tt_dispatch_table = ttnn.from_torch(
        dispatch_table,
        device=mesh_device,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(None, 0),
        ),
    )

    # Run ttnn op
    tt_histograms = ttnn.experimental.deepseek_prefill.masked_bincount(
        tt_indices, tt_dispatch_table, n_routed_experts, topk
    )

    # Build reference as (num_dispatch_groups, dispatch_group_size, n_routed_experts)
    reference = torch.zeros(num_dispatch_groups, dispatch_group_size, n_routed_experts, dtype=torch.int32)
    for group in range(num_dispatch_groups):
        for chip in range(dispatch_group_size):
            reference[group, chip] = torch_histograms[(group, chip)]

    # Compose TTNN output using EP mesh composer
    tt_histograms_4d = ttnn.unsqueeze_to_4D(tt_histograms)
    composer = get_ep_mesh_composer(mesh_device)
    composed = ttnn.to_torch(tt_histograms_4d, mesh_composer=composer).squeeze(2).to(torch.int32)

    # Trim to n_routed_experts in case of padding
    composed = composed[..., :n_routed_experts]

    result = validate_composed(
        composed,
        reference,
        num_dispatch_groups,
        dispatch_group_size,
        compare_exact,
        name="masked_bincount",
    )
    log_validation_results(
        results=[result],
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        title="Masked Bincount Validation",
    )
    result.assert_passed("masked_bincount output does not match torch reference")
    logger.info("masked_bincount matches torch reference!")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 1),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 1), topology="linear"),
            id="single",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_masked_bincount_tree_reduction_race(mesh_device):
    """
    Stress test for tree-reduction race condition in masked_bincount.

    Crafts workload skew to maximize out-of-order child completion in the
    binary-tree gather phase: one core's shard targets present experts (slow,
    many atomic increments) while all other cores target absent experts (fast,
    zero increments). This creates a timing gap where deeper subtrees signal
    the root's gather_sem before the slow leaf finishes, exposing data races
    where a parent reads a child's incomplete histogram.
    """
    sp_dim = 4096
    topk = 8
    n_routed_experts = 256
    num_cores = 64
    rows_per_core = sp_dim // num_cores  # 64
    num_iterations = 50

    # Custom dispatch table: only experts 0-7 are "present" (chip_id=0),
    # rest are absent (-1). Shape: (1, 256) for single dispatch group.
    dispatch_table = torch.full((1, n_routed_experts), -1, dtype=torch.int32)
    num_present = 8
    for e in range(num_present):
        dispatch_table[0, e] = 0

    # Height-shard config matching the gate module pattern: 8x8 core grid
    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(rows_per_core, topk),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    # Place dispatch table on device (static across iterations)
    tt_dispatch_table = ttnn.from_torch(
        dispatch_table,
        device=mesh_device,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(None, 0),
        ),
    )

    composer = get_ep_mesh_composer(mesh_device)
    num_mismatches = 0

    for iteration in range(num_iterations):
        # Craft skewed indices:
        # - Core 1 (rows 64-127): present experts 0..7 → many atomic increments (slow)
        # - All other cores: absent experts 128..255 → zero increments (fast)
        indices = torch.randint(128, n_routed_experts, (sp_dim, topk), dtype=torch.int32)
        core1_start = 1 * rows_per_core
        core1_end = 2 * rows_per_core
        indices[core1_start:core1_end] = torch.randint(0, num_present, (rows_per_core, topk), dtype=torch.int32)

        # Torch reference
        reference = torch_masked_bincount(indices, dispatch_table[0], n_routed_experts)

        # Place indices on device
        tt_indices = ttnn.from_torch(
            indices.to(torch.int16),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, None)),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint16,
        )
        tt_indices = ttnn.to_memory_config(tt_indices, sharded_mem_config)

        # Run TTNN op
        tt_hist = ttnn.experimental.deepseek_prefill.masked_bincount(
            tt_indices, tt_dispatch_table, n_routed_experts, topk
        )

        # Read back and compare
        tt_hist_4d = ttnn.unsqueeze_to_4D(tt_hist)
        composed = ttnn.to_torch(tt_hist_4d, mesh_composer=composer).squeeze(2).to(torch.int32)
        composed = composed[..., :n_routed_experts]
        actual = composed.squeeze(0).squeeze(0)  # (n_routed_experts,)

        if not torch.equal(actual, reference):
            num_mismatches += 1
            diff_mask = actual != reference
            diff_indices = diff_mask.nonzero(as_tuple=True)[0]
            logger.error(
                f"Iteration {iteration}: MISMATCH at {len(diff_indices)} experts. "
                f"First 5: {diff_indices[:5].tolist()}, "
                f"actual={actual[diff_indices[:5]].tolist()}, "
                f"expected={reference[diff_indices[:5]].tolist()}"
            )

    assert num_mismatches == 0, (
        f"Tree-reduction race detected: {num_mismatches}/{num_iterations} iterations " f"produced incorrect histograms"
    )
    logger.info(f"All {num_iterations} iterations match torch reference (race test passed)")
