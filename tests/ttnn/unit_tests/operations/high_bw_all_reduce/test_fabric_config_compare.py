# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does FABRIC_1D_NEIGHBOR_EXCHANGE speed up the one-hop CCLs? Runs high_bw_all_gather's own perf
measurement on the QuietBox 4x1 line under several fabric configs, all with the 14 KiB router payload.
Run with `-s` to see the HIGH_BW_ALL_GATHER lines."""

import pytest
import ttnn

import tests.ttnn.unit_tests.operations.experimental.test_high_bw_all_gather as ag

_CONFIGS = [
    pytest.param(ag._device_params(ttnn.FabricConfig.FABRIC_1D), id="fabric_1d"),
    pytest.param(ag._device_params(ttnn.FabricConfig.FABRIC_1D_NEIGHBOR_EXCHANGE), id="fabric_1d_neighbor_exchange"),
    pytest.param(ag._device_params(ttnn.FabricConfig.FABRIC_2D), id="fabric_2d"),
]


@pytest.mark.parametrize("device_params", _CONFIGS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 1)], indirect=True)
@pytest.mark.parametrize(
    "case_name,dtype,width,layout,expected_page_size",
    [c for c in ag._TEST_CASES if c[0] in ("bf16_1152b_rows", "bf16_tiles")],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_all_gather_fabric_config(mesh_device, case_name, dtype, width, layout, expected_page_size):
    print(f"FABRIC_CONFIG_COMPARE op=high_bw_all_gather fabric={ttnn.get_fabric_config()} case={case_name}")
    ag._run_high_bw_all_gather_perf(
        mesh_device, dtype, width, layout, expected_page_size, min_bandwidth_gbps=0.0, cluster_axis=0
    )


import tests.ttnn.unit_tests.operations.high_bw_all_reduce.test_high_bw_all_reduce_bw_compare as ar_bw


@pytest.mark.parametrize("device_params", _CONFIGS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 1), (2, 2)], ids=["mesh4x1", "mesh2x2"], indirect=True)
@pytest.mark.parametrize("num_links", [1, 2], ids=["links1", "links2"])
def test_all_reduce_fabric_config(mesh_device, num_links):
    """Axis-0 line all-reduce (G = 4 on the 4x1 mesh, G = 2 on the 2x2 mesh), correctness + median time."""
    print(
        f"FABRIC_CONFIG_COMPARE op=high_bw_all_reduce fabric={ttnn.get_fabric_config()} mesh={tuple(mesh_device.shape)}"
    )
    ar_bw.test_bw(mesh_device, (1, 1, 8192, 4096), 0, ttnn.Topology.Linear, num_links)
