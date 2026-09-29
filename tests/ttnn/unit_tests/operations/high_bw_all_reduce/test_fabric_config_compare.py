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


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(ag._device_params(ttnn.FabricConfig.FABRIC_1D, payload), id=f"fabric_1d_payload{payload}")
        for payload in (8704, 14336, 15232)  # 15232 = Blackhole router maximum
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 1)], indirect=True)
@pytest.mark.parametrize(
    "case_name,dtype,width,layout,expected_page_size",
    [c for c in ag._TEST_CASES if c[0] in ("bf16_1152b_rows", "bf16_tiles")],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_all_gather_payload(mesh_device, case_name, dtype, width, layout, expected_page_size, monkeypatch):
    payload = ttnn.get_tt_fabric_max_payload_size_bytes()
    print(f"FABRIC_PAYLOAD_COMPARE op=high_bw_all_gather payload={payload} case={case_name}")
    # The gather's perf helper pins its own config with `assert payload == 14 KiB`; the op itself reads the
    # real router payload, so only that test-side guard is bypassed here.
    monkeypatch.setattr(ttnn, "get_tt_fabric_max_payload_size_bytes", lambda: 14 * 1024)
    ag._run_high_bw_all_gather_perf(
        mesh_device, dtype, width, layout, expected_page_size, min_bandwidth_gbps=0.0, cluster_axis=0
    )


@pytest.mark.parametrize("device_params", _CONFIGS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 1)], indirect=True)
@pytest.mark.parametrize("rows_per_device", [8192, 65536, 262144], ids=lambda r: f"rows{r}")
def test_all_gather_tensor_size(mesh_device, rows_per_device):
    """bf16 tiles, width 576: 65536 rows = 75.5 MB per device (the default perf size)."""
    case_name, dtype, width, layout, expected_page_size = next(c for c in ag._TEST_CASES if c[0] == "bf16_tiles")
    mb = rows_per_device * width * 2 / 1e6
    print(f"FABRIC_SIZE_COMPARE fabric={ttnn.get_fabric_config()} rows={rows_per_device} per_device={mb:.1f}MB")
    ag._run_high_bw_all_gather_perf(
        mesh_device,
        dtype,
        width,
        layout,
        expected_page_size,
        min_bandwidth_gbps=0.0,
        cluster_axis=0,
        rows_per_device=rows_per_device,
    )


# Router payload x fabric sweep at the fabric microbenchmark's best payloads (7616 / 8704 B) and the ops' 14336 B.
_PAYLOAD_SWEEP = [
    pytest.param(ag._device_params(cfg, payload), id=f"{name}_payload{payload}")
    for name, cfg in (("fabric_1d", ttnn.FabricConfig.FABRIC_1D), ("fabric_2d", ttnn.FabricConfig.FABRIC_2D))
    for payload in (7616, 8704, 14336)
]


@pytest.mark.parametrize("device_params", _PAYLOAD_SWEEP, indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 1)], indirect=True)
def test_all_gather_payload_sweep(mesh_device, monkeypatch):
    case_name, dtype, width, layout, expected_page_size = next(c for c in ag._TEST_CASES if c[0] == "bf16_tiles")
    payload = ttnn.get_tt_fabric_max_payload_size_bytes()
    print(f"FABRIC_PAYLOAD_SWEEP op=high_bw_all_gather fabric={ttnn.get_fabric_config()} payload={payload}")
    monkeypatch.setattr(ttnn, "get_tt_fabric_max_payload_size_bytes", lambda: 14 * 1024)  # test-side guard only
    ag._run_high_bw_all_gather_perf(
        mesh_device, dtype, width, layout, expected_page_size, min_bandwidth_gbps=0.0, cluster_axis=0
    )


@pytest.mark.parametrize("device_params", _PAYLOAD_SWEEP, indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 1), (2, 2)], ids=["mesh4x1", "mesh2x2"], indirect=True)
def test_all_reduce_payload_sweep(mesh_device):
    print(
        f"FABRIC_PAYLOAD_SWEEP op=high_bw_all_reduce fabric={ttnn.get_fabric_config()} "
        f"payload={ttnn.get_tt_fabric_max_payload_size_bytes()} mesh={tuple(mesh_device.shape)}"
    )
    ar_bw.test_bw(mesh_device, (1, 1, 8192, 4096), 0, ttnn.Topology.Linear, 2)
