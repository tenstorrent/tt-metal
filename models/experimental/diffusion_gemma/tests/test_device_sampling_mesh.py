# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh smoke tests for regenerated DiffusionGemma sampling noise (#47472)."""

import os

import pytest

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.experimental.diffusion_gemma.tt import sampling as TS

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
    ),
    pytest.mark.use_module_device,
]


@parametrize_mesh_with_fabric([(1, 4)])
def test_sample_gumbel_noise_runs_on_qb2_mesh(mesh_device):
    noise = TS.sample_gumbel_noise((1, 1, 32, 32), device=mesh_device, seed=47472)

    host_noise = ttnn.to_torch(ttnn.get_device_tensors(noise)[0])
    assert host_noise.shape == (1, 1, 32, 32)
    noise.deallocate(True)
