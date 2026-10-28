# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.matmul_reduce_scatter_prefill on a Blackhole Galaxy (8x4 mesh, wrap links on both axes).

Groups along axis 0 are 8-device columns, along axis 1 4-device rows; every group of the mesh runs at once.
"""

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_matmul_reduce_scatter_prefill_bh import (
    router_config,
    run_case,
)
from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0

GALAXY_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "fabric_router_config": router_config(),
    "require_exact_physical_num_devices": True,
}

# (a_shape, w_shape, scatter_dim, weight_dtype)
CASES = {
    "kimi_k27_o_proj": ((1, 1, 640, 2048), (2048, 7168), -1, ttnn.bfloat8_b),
    "mimo_v2_o_proj": ((1, 1, 2048, 2048), (2048, 4096), -2, ttnn.bfloat8_b),
    "chunked_n": ((1, 1, 2048, 1024), (1024, 16384), -1, ttnn.bfloat16),
}


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis0_8dev", "axis1_4dev"])
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring, ttnn.Topology.Linear], ids=["ring", "line"])
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_galaxy(mesh_device, cluster_axis, topology, case):
    a_shape, w_shape, scatter_dim, weight_dtype = case
    run_case(
        mesh_device,
        a_shape,
        w_shape,
        cluster_axis=cluster_axis,
        scatter_dim=scatter_dim,
        topology=topology,
        weight_dtype=weight_dtype,
    )
