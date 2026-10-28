# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-time perf gates for ttnn.experimental.matmul_reduce_scatter_prefill on a Blackhole Galaxy (8x4 mesh).

The box perf cases (see the box test for the measurement and the gate) run as a ring along both cluster axes: axis 0
groups are 8-device columns, axis 1 groups 4-device rows (the TP layout of the DeepSeek prefill mesh); every group of
the mesh runs at once.
"""

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_matmul_reduce_scatter_prefill_bh import (
    router_config,
)
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_matmul_reduce_scatter_prefill_perf_bh import (
    PERF_CASES,
    perf_device_params,
    run_perf_case,
)
from tt_py_test_utils_common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher

GALAXY_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "fabric_router_config": router_config(),
    "require_exact_physical_num_devices": True,
}

# Expected device time [ns] per (cluster_axis, case) on a Blackhole Galaxy (8x4 ring, 2 links); not measured yet.
EXPECTED_NS = {(axis, case): None for axis in (0, 1) for case in PERF_CASES}


@run_for_blackhole("matmul_reduce_scatter_prefill perf gates are Blackhole numbers")
@pytest.mark.requires_host_iommu
@skip_with_llk_assert("LLK asserts perturb kernel timing; perf checks are not meaningful with them.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("device_params", [perf_device_params(GALAXY_DEVICE_PARAMS)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis0_8dev", "axis1_4dev"])
@pytest.mark.parametrize("case_name", PERF_CASES.keys())
def test_galaxy_ring_perf(mesh_device, cluster_axis, case_name):
    run_perf_case(
        mesh_device,
        case_name,
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Ring,
        expected_ns=EXPECTED_NS[(cluster_axis, case_name)],
        label=f"galaxy axis{cluster_axis} ring {case_name}",
    )
