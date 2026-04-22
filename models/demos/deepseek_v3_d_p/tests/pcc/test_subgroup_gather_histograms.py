# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit test for ttnn.experimental.deepseek_prefill.subgroup_gather_histograms.

Runs on an 8x1 Blackhole LoudBox with num_dispatch_subgroups=2. Each chip is fed a
distinct per-subgroup histogram (so we can detect any cross-subgroup contamination),
and we assert the gathered output on chips 0..3 matches only the 4 histograms from
chips 0..3, and chips 4..7 match only theirs.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size


@pytest.mark.parametrize("n_routed_experts", [64])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology, num_dispatch_subgroups, dispatch_group_size",
    [
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            1,
            ttnn.Topology.Linear,
            2,
            4,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="subgroups-2x4-linear-1link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_subgroup_gather_histograms(
    mesh_device,
    n_routed_experts,
    num_links,
    topology,
    num_dispatch_subgroups,
    dispatch_group_size,
):
    torch.manual_seed(42)

    num_devices = mesh_device.get_num_devices()
    assert num_devices == dispatch_group_size * num_dispatch_subgroups

    # Build distinct per-chip histograms so contamination is detectable.
    # chip i gets row [i * 1000 + 0, i * 1000 + 1, ..., i * 1000 + W-1] (mod 2^16 to stay in uint32).
    histograms = torch.zeros(num_devices, n_routed_experts, dtype=torch.int64)
    for i in range(num_devices):
        histograms[i, :] = torch.arange(n_routed_experts, dtype=torch.int64) + i * 1000
    histograms = histograms.to(torch.int32)

    tt_histograms = ttnn.from_torch(
        histograms,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, None)),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_gathered = ttnn.experimental.deepseek_prefill.subgroup_gather_histograms(
        tt_histograms,
        cluster_axis=0,
        num_dispatch_subgroups=num_dispatch_subgroups,
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    all_passed = True
    for dev_idx, dt in enumerate(ttnn.get_device_tensors(tt_gathered)):
        tt_out = ttnn.to_torch(dt).to(torch.int64)
        sg_idx = dev_idx // dispatch_group_size
        expected = histograms[sg_idx * dispatch_group_size : (sg_idx + 1) * dispatch_group_size, :].to(torch.int64)

        if tt_out.shape != expected.shape:
            logger.error(f"device {dev_idx}: shape mismatch tt={tt_out.shape} expected={expected.shape}")
            all_passed = False
            continue
        if not torch.equal(tt_out, expected):
            diff_mask = tt_out != expected
            first_bad = torch.nonzero(diff_mask)[0].tolist()
            logger.error(
                f"device {dev_idx} (subgroup {sg_idx}): mismatch at {first_bad}; "
                f"tt={tt_out[tuple(first_bad)]} vs expected={expected[tuple(first_bad)]}"
            )
            all_passed = False
        else:
            logger.info(f"✅ device {dev_idx} (subgroup {sg_idx}) matches")

    assert all_passed, "subgroup_gather_histograms did not match reference on one or more devices"
