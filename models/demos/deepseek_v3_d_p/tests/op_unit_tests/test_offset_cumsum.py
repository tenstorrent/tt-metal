# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for ttnn.offset_cumsum operation in isolation.

Verifies that the TTNN offset_cumsum (all_gather + shifted prefix sum of
per-device expert histograms) matches a PyTorch reference.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_xy_device_params,
    torus_y_device_params,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import extract_mesh_config


def torch_offset_cumsum(
    histograms: torch.Tensor, experts_per_chip: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference implementation: global dispatch offsets from per-device expert histograms.

    Given per-device histograms of shape [dispatch_group_size, n_routed_experts],
    returns:
      - global_dispatch_offsets: [dispatch_group_size, n_routed_experts] combining local offsets
        (shifted prefix sum across devices) with expert region offsets (exclusive prefix
        sum of totals within each chip's expert group).
      - total_counts_per_expert: [1, n_routed_experts] sum of all rows.
      - expert_region_offsets: [1, n_routed_experts] — only the expert region component,
        shared across all source devices.

    Args:
        histograms: [dispatch_group_size, n_routed_experts] int tensor.
        experts_per_chip: Number of experts per chip.

    Returns:
        Tuple of (global_dispatch_offsets, total_counts_per_expert, expert_region_offsets).
    """
    dispatch_group_size, n_routed_experts = histograms.shape

    # Local offsets: shifted prefix sum across devices
    cum = torch.cumsum(histograms, dim=0)
    zeros = torch.zeros(1, n_routed_experts, dtype=histograms.dtype)
    full = torch.cat([zeros, cum], dim=0)
    local_offsets = full[:-1, :]
    totals = full[-1:, :]

    # Expert region offsets: exclusive prefix sum of tile-aligned totals within each chip group
    # Pad each expert's count to TILE_SIZE so each expert starts at a tile boundary
    total_num_chips = n_routed_experts // experts_per_chip
    totals_grouped = totals.reshape(total_num_chips, experts_per_chip)
    aligned_totals = ((totals_grouped + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    partial = torch.cumsum(aligned_totals, dim=-1)
    partial = torch.cat([torch.zeros(total_num_chips, 1, dtype=histograms.dtype), partial[:, :-1]], dim=-1)
    expert_region_offsets_1row = partial.reshape(1, n_routed_experts)
    expert_region_offsets = expert_region_offsets_1row.expand(dispatch_group_size, -1)

    return local_offsets + expert_region_offsets, totals, expert_region_offsets_1row


def _to_2d(tt_tensor) -> torch.Tensor:
    """Per-device shard as a rank-2 int32 tensor. Leading dims must all be 1; anything else is a
    shape regression."""
    out = ttnn.to_torch(tt_tensor).to(torch.int32)
    if out.dim() > 2:
        assert all(d == 1 for d in out.shape[:-2]), f"unexpected leading dims in {tuple(out.shape)}"
        out = out.reshape(out.shape[-2], out.shape[-1])
    return out


def _check(name: str, dev_idx: int, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Compare one device's shard against a reference, logging what differs."""
    logger.info(f"Device {dev_idx} {name}: tt_shape={actual.shape}, ref_shape={expected.shape}")
    if actual.shape != expected.shape:
        logger.error(f"Device {dev_idx}: {name} shape mismatch tt={actual.shape} ref={expected.shape}")
        return False
    if not torch.equal(actual, expected):
        num_diff = (actual != expected).sum().item()
        logger.error(f"Device {dev_idx}: {name} {num_diff}/{expected.numel()} elements differ")
        logger.error(f"  Max abs diff: {(actual - expected).abs().max().item()}")
        return False
    logger.info(f"Device {dev_idx} {name}: PASS")
    return True


@pytest.mark.parametrize(
    "n_routed_experts",
    # 264 * 4 B is not a whole 64 B DRAM page, so the kernel's row strides differ from the width.
    [256, 32, 264],
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (4, 1),
            torus_y_device_params(),
            1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="ring"),
            id="torus-y-4x1",
        ),
        pytest.param(
            (4, 2),
            fabric2d_device_params(),
            1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="fabric2d-mesh-4x2",
        ),
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        ),
        pytest.param(
            (8, 4),
            fabric2d_device_params(),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-8x4",
        ),
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_offset_cumsum(
    mesh_device,
    device_params,
    n_routed_experts,
    num_links,
    memory_config,
):
    """Test ttnn.offset_cumsum against PyTorch reference."""
    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups
    num_devices = num_dispatch_groups * dispatch_group_size
    if n_routed_experts % num_devices != 0:
        pytest.skip(f"{n_routed_experts} experts do not divide across {mesh_device.shape} devices")
    experts_per_chip = n_routed_experts // num_devices

    logger.info(
        f"Testing offset_cumsum: {mesh_device.shape=}, {sp_axis=}, "
        f"{dispatch_group_size=}, {n_routed_experts=}, {experts_per_chip=}"
    )
    ttnn.visualize_mesh_device(mesh_device)

    torch.manual_seed(42)

    mesh_rows, mesh_cols = mesh_device.shape

    def get_row_idx(dev_idx):
        coord_row = dev_idx // mesh_cols
        coord_col = dev_idx % mesh_cols
        return coord_row if sp_axis == 0 else coord_col

    def get_group_idx(dev_idx):
        coord_row = dev_idx // mesh_cols
        coord_col = dev_idx % mesh_cols
        return coord_col if sp_axis == 0 else coord_row

    def shard_per_group(t):
        """Place a [num_dispatch_groups, dispatch_group_size, W] histogram on the mesh.

        The dispatch-group dimension goes on the axis that is NOT cluster_axis, so each group gets
        different data. Replicating one histogram across the groups would make every group compute
        the same table, and a device reading a neighbouring group's row would still pass.
        """
        dims = (1, 0) if sp_axis == 0 else (0, 1)
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=dims),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    all_passed = True

    # Two invocations with different data. The second must be a program-cache hit (asserted below):
    # that is the path production takes, and the one where a buffer address held in the wrong place
    # goes stale.
    mesh_device.enable_program_cache()
    for run_idx, seed in enumerate((42, 43)):
        torch.manual_seed(seed)
        # Simulate per-device expert histograms (output of masked_bincount), which in production
        # differ across dispatch groups because masked_bincount applies a per-group expert mask.
        # Shape: [num_dispatch_groups, dispatch_group_size, n_routed_experts]
        histograms = torch.randint(
            0, 32, (num_dispatch_groups, dispatch_group_size, n_routed_experts), dtype=torch.int32
        )

        # One reference per dispatch group; groups are independent.
        refs = [torch_offset_cumsum(histograms[g], experts_per_chip) for g in range(num_dispatch_groups)]

        tt_offsets, tt_totals, tt_expert_region, tt_all_offsets = ttnn.experimental.deepseek_prefill.offset_cumsum(
            shard_per_group(histograms),
            cluster_axis=sp_axis,
            num_links=num_links,
            experts_per_chip=experts_per_chip,
            memory_config=memory_config,
        )
        if run_idx == 0:
            cache_entries = mesh_device.num_program_cache_entries()
        else:
            assert mesh_device.num_program_cache_entries() == cache_entries, "second run missed the program cache"

        tag = f"run{run_idx}"
        own_offsets = ttnn.get_device_tensors(tt_offsets)
        all_tables = ttnn.get_device_tensors(tt_all_offsets)
        dev_totals = ttnn.get_device_tensors(tt_totals)
        dev_regions = ttnn.get_device_tensors(tt_expert_region)

        for dev_idx in range(len(own_offsets)):
            row_idx = get_row_idx(dev_idx)
            ref_offsets, ref_totals, ref_region = refs[get_group_idx(dev_idx)]
            all_passed &= _check(
                f"{tag} offsets", dev_idx, _to_2d(own_offsets[dev_idx]), ref_offsets[row_idx : row_idx + 1, :]
            )
            all_passed &= _check(f"{tag} totals", dev_idx, _to_2d(dev_totals[dev_idx]), ref_totals)
            all_passed &= _check(f"{tag} expert_region", dev_idx, _to_2d(dev_regions[dev_idx]), ref_region)
            # all_offsets is replicated along the dispatch axis, so every device holds every row of
            # its OWN group's table -- and none of a neighbouring group's.
            all_passed &= _check(f"{tag} all_offsets", dev_idx, _to_2d(all_tables[dev_idx]), ref_offsets)

        # Consumer contract: a consumer derives device k's run length from rows k and k+1, and the last
        # device's from the totals. Assert it from the op's own outputs; the last row is the easy
        # one to get wrong.
        for g in range(num_dispatch_groups):
            dev_idx = next(i for i in range(len(all_tables)) if get_group_idx(i) == g)
            table = _to_2d(all_tables[dev_idx])
            region = _to_2d(dev_regions[dev_idx])
            totals = _to_2d(dev_totals[dev_idx])
            for k in range(dispatch_group_size):
                derived = (
                    table[k + 1 : k + 2, :] - table[k : k + 1, :]
                    if k + 1 < dispatch_group_size
                    else totals + region - table[k : k + 1, :]
                )
                all_passed &= _check(
                    f"{tag} group {g} derived count row {k}",
                    dev_idx,
                    derived,
                    histograms[g, k : k + 1, :].to(torch.int32),
                )

    assert all_passed, "offset_cumsum output does not match torch reference on one or more devices"
    logger.info("offset_cumsum matches torch reference!")


# 1x8 along axis 1 is H = 8, the production dispatch-group size.
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((2, 4), fabric2d_device_params(fabric_payload_size=6144, l1_small_size=1216), id="2x4"),
        pytest.param((1, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 1216}, id="1x8"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
# 260 % 8 != 0: covers the kernel's scalar tail.
@pytest.mark.parametrize("width, experts_per_chip", [(256, 8), (260, 4)], ids=["w256", "w260"])
def test_offset_cumsum_distinct_groups_cache(mesh_device, cluster_axis, memory_config, width, experts_per_chip):
    rows, cols = tuple(mesh_device.shape)
    if mesh_device.shape[cluster_axis] == 1:
        pytest.skip("all_gather needs more than one device along cluster_axis")
    retained = []
    mesh_device.enable_program_cache()

    def make_input(seed):
        generator = torch.Generator().manual_seed(seed)
        histograms = torch.randint(0, 64, (rows, cols, width), dtype=torch.int32, generator=generator)
        tensor = ttnn.from_torch(
            histograms,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, 1)),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        retained.append(tensor)
        return histograms, tensor

    def run(tensor):
        return ttnn.experimental.deepseek_prefill.offset_cumsum(
            tensor,
            cluster_axis=cluster_axis,
            num_links=2,
            experts_per_chip=experts_per_chip,
            memory_config=memory_config,
            use_l1_small_for_semaphores=True,
        )

    def check(histograms, outputs):
        per_device = [ttnn.get_device_tensors(t) for t in outputs]
        assert all(len(tensors) == rows * cols for tensors in per_device)
        for chip in range(rows * cols):
            row, col = divmod(chip, cols)
            data = histograms[:, col, :] if cluster_axis == 0 else histograms[row, :, :]
            position = row if cluster_axis == 0 else col
            total = data.sum(0)
            aligned = ((total + 31) // 32 * 32).reshape(-1, experts_per_chip)
            region = (aligned.cumsum(-1) - aligned).reshape(width)
            offset = data[:position].sum(0) + region
            all_offsets = (data.cumsum(0) - data + region).reshape(-1)
            references = (offset, total, region, all_offsets)
            assert len(per_device) == len(references)
            for reference, tensors in zip(references, per_device):
                actual = ttnn.to_torch(tensors[chip]).reshape(-1).to(torch.int64)
                assert torch.equal(actual, reference), (cluster_axis, row, col)

    for iteration in range(3):
        host, tensor = make_input(1234 + iteration)
        outputs = run(tensor)
        check(host, outputs)
        if iteration == 0:
            entries = mesh_device.num_program_cache_entries()
        else:
            assert mesh_device.num_program_cache_entries() == entries
        retained.extend(outputs)
