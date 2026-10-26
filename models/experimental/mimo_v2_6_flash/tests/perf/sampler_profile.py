# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Runs DeviceSampler's stages eagerly for the device profiler (TT_METAL_DEVICE_PROFILER=1); the per-kernel durations
land in $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv."""

import pytest
import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS, dp_submeshes
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan
from models.experimental.mimo_v2_6_flash.tt.device_sampler import DeviceSampler

VOCAB = 152576
B = 32


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_sampler_profile(mesh_device):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    s = DeviceSampler(plan, VOCAB // plan.tp, 4, B)
    logits = torch.randn(1, 1, B, VOCAB) * 4
    logits_tt = ttnn.from_torch(
        logits,
        device=plan.mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=plan.shard(3),
    )
    host, _ = s.host_params(
        torch.tensor([1.0, 0.7, 1.0, 0.0]), torch.tensor([-1, 20, -1, -1]), torch.tensor([0.95] * 4)
    )
    s.set_params(host)
    for _ in range(4):
        s(logits_tt)
    ttnn.synchronize_device(plan.mesh_device)
    ttnn.ReadDeviceProfiler(plan.mesh_device)
