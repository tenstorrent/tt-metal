# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.deepseek_v3.utils.config_helpers import get_fabric_config

# Reuse the shared test logic from the migrated nightly test (same pattern as
# test_combine_tg.py reusing test_selective_combine_6U). This single-device unit
# test runs the op on a (1, 1) submesh of the TG; the nightly file additionally
# exercises a (1, 8) Blackhole mesh.
from tests.nightly.tg.ccl.moe.test_deepseek_moe_post_combine_tilize import run_post_combine_tilize_test


@pytest.mark.requires_device(["TG"])
@pytest.mark.parametrize("iterations", [10])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": get_fabric_config(),
            "trace_region_size": 0,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_moe_post_combine_tilize(mesh_device, iterations):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    run_post_combine_tilize_test(submesh_device, iterations)
