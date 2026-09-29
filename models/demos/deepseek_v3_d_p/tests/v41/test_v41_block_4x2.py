# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""``test_block_v41`` on the LoudBox 4x2 mesh (SP4 x TP2; dev-spec §8: block tests on 2x4 and 4x2).

Same cases, bars and checks as ``test_v41_blocks_on_device_state`` (called as is); only the mesh differs, which
exercises the TP=2 splits (heads, groups, hidden) and SP=4 cache geometry that 2x4 does not.

The device MoE weight cache holds per-mesh-coordinate shards (a 2x4 cache fails to load on 4x2: coordinate [0, 2]
is not in a 4x2 mesh), so this test keeps its own cache root under ``test_block_v41``'s.
"""

import pytest

from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41 import test_block_v41 as blocks


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kv_format", list(blocks.KV_FORMATS))
@pytest.mark.parametrize("prompt", ["full", "padded", "tiny"])
@pytest.mark.parametrize("schedule", ["stack", "swa"])
@pytest.mark.parametrize("chunks", [1, 2], ids=["single_chunk", "two_chunks"])
@pytest.mark.parametrize("weights", ["small", "synthetic", "real"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (4, 2),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="fabric2d-mesh-4x2",
        )
    ],
    indirect=True,
)
def test_v41_blocks_on_device_state_4x2(mesh_device, device_params, weights, chunks, schedule, prompt, kv_format):
    blocks.test_v41_blocks_on_device_state(mesh_device, device_params, weights, chunks, schedule, prompt, kv_format)
