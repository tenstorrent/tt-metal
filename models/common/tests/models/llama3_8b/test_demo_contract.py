# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from models.demos.utils.trace_region_sizes import resolve_trace_region_size

_DEMO_SOURCE = Path("models/common/tests/demos/llama3_8b/demo.py").read_text(encoding="utf-8")


def test_demo_exposes_p300_as_ring_two_chip_mesh():
    assert '"P300": (1, 2)' in _DEMO_SOURCE
    assert 'mesh_device_name == "P300"' in _DEMO_SOURCE
    assert "ttnn.FabricConfig.FABRIC_1D_RING" in _DEMO_SOURCE


def test_demo_keeps_p300_dp2_case_in_manifest():
    assert '"ci-b1-DP-2": DemoCase(' in _DEMO_SOURCE


def test_p150_batch32_uses_dynamic_trace_allocation():
    assert 'resolve_trace_region_size("llama3.1-8b", mesh_device_name)' in _DEMO_SOURCE
    assert resolve_trace_region_size("llama3.1-8b", "P150") == 0
