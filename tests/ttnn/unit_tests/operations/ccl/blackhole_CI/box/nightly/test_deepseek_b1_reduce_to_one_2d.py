# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.common.utility_functions import skip_for_wormhole_b0

# Import helper functions from 1D test file
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_deepseek_b1_reduce_to_one import (
    run_reduce_to_one,
    run_reduce_to_one_with_trace,
)


# === Basic Tests ===
@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
def test_reduce_to_one_2d(bh_2d_mesh_device):
    """Test reduce_to_one with 2D fabric."""
    run_reduce_to_one(bh_2d_mesh_device, ttnn.Topology.Linear)


# === Trace Tests ===
@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 425984})],
    indirect=["device_params"],
    ids=["fabric_2d_trace"],
)
def test_reduce_to_one_with_trace_2d(bh_2d_mesh_device):
    """Test reduce_to_one with trace capture/replay on 2D fabric."""
    run_reduce_to_one_with_trace(bh_2d_mesh_device, ttnn.Topology.Linear)
