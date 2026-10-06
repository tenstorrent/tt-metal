# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model placement agrees with coordinate-aware MeshDevice and one-device views."""

import json
import os

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn import bf4, decode_matmul

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held four-die mesh")


def coordinates(cores):
    return tuple((int(core.x), int(core.y)) for core in cores)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_existing_mesh_dram_assignment_model_consumers(mesh_device, record_property):
    old_query = getattr(ttnn.device, "get_optimal_dram_bank_to_logical_worker_assignment_at_mesh_coordinate", None)
    rows = []
    model_signatures = decode_matmul.mesh_dram_bank_worker_signatures(mesh_device)
    for column in range(4):
        coordinate = ttnn.MeshCoordinate(0, column)
        unit = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=coordinate)
        for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
            bank_map = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(noc, coordinate)
            assignment = [bank_map[bank] for bank in sorted(bank_map)]
            expected = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(unit, noc)
            assert coordinates(assignment) == coordinates(expected)
            if old_query is not None:
                assert coordinates(old_query(mesh_device, noc, coordinate)) == coordinates(assignment)
            if noc == ttnn.NOC.RISCV_0_default:
                assert model_signatures[(0, column)] == coordinates(assignment)
            rows.append(
                {
                    "coordinate": [0, column],
                    "noc": str(noc),
                    "banks": sorted(bank_map),
                    "workers": coordinates(assignment),
                    "old_wrapper_compared": old_query is not None,
                }
            )
    ring_order = bf4.qualify_live_bf4_ring(mesh_device)
    assert len(ring_order) == len(rows[0]["banks"])
    record_property("dram_assignments", json.dumps(rows, sort_keys=True))
    record_property("bf4_ring_order", json.dumps(ring_order))
    print("DRAM_ASSIGNMENTS=" + json.dumps({"queries": rows, "bf4_ring_order": ring_order}, sort_keys=True))
