# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit test for ttnn.experimental.deepseek_prefill.subgroup_gather_histograms.

Runs on a Blackhole LoudBox with num_dispatch_subgroups=2. Each chip is fed a
distinct per-chip histogram (so any cross-subgroup or cross-chip contamination
is detectable), and we assert the gathered output on every chip matches only
its own subgroup's histograms.

Covers both 1D (8x1 linear) and 2D (4x2 mesh) mesh topologies. The 2D variant
exercises chip/mesh-id-based fabric routing needed for diagonal peers in a
(2, 2) subgroup.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size


def _make_tt_histograms(mesh_device, histograms, dtype):
    """
    Shard a per-chip histogram tensor of shape (mesh_rows, mesh_cols, W) across
    the mesh so each chip (r, c) gets its own (1, W) row; returns the TTNN tensor.
    """
    mesh_rows, mesh_cols = mesh_device.shape

    if mesh_cols == 1:
        # 1D mesh: shard along axis 0, input shape (mesh_rows, W).
        flat = histograms.reshape(mesh_rows, -1)
        return ttnn.from_torch(
            flat,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, None)),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # 2D mesh: shard both axes with input shape (mesh_rows, mesh_cols, W). The op
    # accepts >=1-D inputs with leading 1s, so per-chip (1, 1, W) works directly.
    return ttnn.from_torch(
        histograms,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, 1)),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("n_routed_experts", [64])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, num_dispatch_subgroups, dispatch_group_size",
    [
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            1,
            2,
            4,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="subgroups-2x4-linear-1link",
        ),
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            1,
            2,
            4,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="subgroups-2x2x2-mesh-4x2-1link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_subgroup_gather_histograms(
    mesh_device,
    n_routed_experts,
    num_links,
    num_dispatch_subgroups,
    dispatch_group_size,
):
    torch.manual_seed(42)

    mesh_rows, mesh_cols = mesh_device.shape
    num_devices = mesh_device.get_num_devices()
    assert num_devices == mesh_rows * mesh_cols
    assert mesh_rows % num_dispatch_subgroups == 0, "mesh_rows must split evenly into subgroups"
    subgroup_rows = mesh_rows // num_dispatch_subgroups
    assert dispatch_group_size == subgroup_rows * mesh_cols

    # Build distinct per-chip histograms so contamination is detectable.
    # chip (r, c) gets row [(r*mesh_cols + c)*1000 + 0, ..., (r*mesh_cols + c)*1000 + W-1].
    histograms = torch.zeros(mesh_rows, mesh_cols, n_routed_experts, dtype=torch.int64)
    for r in range(mesh_rows):
        for c in range(mesh_cols):
            linear = r * mesh_cols + c
            histograms[r, c, :] = torch.arange(n_routed_experts, dtype=torch.int64) + linear * 1000
    histograms = histograms.to(torch.int32)

    tt_histograms = _make_tt_histograms(mesh_device, histograms, dtype=ttnn.uint32)

    tt_gathered = ttnn.experimental.deepseek_prefill.subgroup_gather_histograms(
        tt_histograms,
        cluster_axis=0,
        num_dispatch_subgroups=num_dispatch_subgroups,
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Per-chip expected output: within its subgroup, rows indexed by local linearized coord
    # (local_row * mesh_cols + local_col). Each subgroup spans rows [sg*subgroup_rows, (sg+1)*subgroup_rows).
    all_passed = True
    for dev_idx, dt in enumerate(ttnn.get_device_tensors(tt_gathered)):
        tt_out = ttnn.to_torch(dt).to(torch.int64)
        r = dev_idx // mesh_cols
        c = dev_idx % mesh_cols
        sg_idx = r // subgroup_rows
        base_row = sg_idx * subgroup_rows
        expected = (
            histograms[base_row : base_row + subgroup_rows, :, :]
            .reshape(subgroup_rows * mesh_cols, n_routed_experts)
            .to(torch.int64)
        )

        if tt_out.shape != expected.shape:
            logger.error(
                f"device {dev_idx} (r={r}, c={c}, sg={sg_idx}): shape mismatch "
                f"tt={tt_out.shape} expected={expected.shape}"
            )
            all_passed = False
            continue
        if not torch.equal(tt_out, expected):
            diff_mask = tt_out != expected
            first_bad = torch.nonzero(diff_mask)[0].tolist()
            logger.error(
                f"device {dev_idx} (r={r}, c={c}, sg={sg_idx}): mismatch at {first_bad}; "
                f"tt={tt_out[tuple(first_bad)]} vs expected={expected[tuple(first_bad)]}"
            )
            all_passed = False
        else:
            logger.info(f"OK device {dev_idx} (r={r}, c={c}, sg={sg_idx}) matches")

    assert all_passed, "subgroup_gather_histograms did not match reference on one or more devices"
