# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Implementer checks beyond the acceptance test. DO NOT DELETE.

The kernel is scheme-agnostic (height/width, page bytes and destination come from the host plan),
so the TARGET cells outside Phase 0 SUPPORTED run here with SUPPORTED widened to TARGET, at every
`num_writer_riscs` value. Also the profiling entry point for the perf lamps.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved_program_descriptor as pd
from ttnn.operations.g6_sharded_to_interleaved import SUPPORTED, describe_plan, g6_sharded_to_interleaved

# The acceptance test's scenarios and checks, loaded by path (the test dir is not a package).
_spec = importlib.util.spec_from_file_location(
    "g6_s2i_acceptance", Path(__file__).with_name("test_g6_sharded_to_interleaved.py")
)
_acceptance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acceptance)
P1_OTHER_SPEC = _acceptance.P1_OTHER_SPEC
P1_PANOPTIC = _acceptance.P1_PANOPTIC
P2_FLUX = _acceptance.P2_FLUX
P2_OTHER_SPEC = _acceptance.P2_OTHER_SPEC
P2_TO_L1 = _acceptance.P2_TO_L1
_Case = _acceptance._Case
_scenario_id = _acceptance._scenario_id

TARGET = {
    "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
    "in_scheme": ["width_sharded", "height_sharded"],
    "buffer": ["dram", "l1"],
}

SCENARIOS = [P2_FLUX, P1_PANOPTIC, P2_OTHER_SPEC, P1_OTHER_SPEC, P2_TO_L1]


@pytest.fixture
def target_supported(monkeypatch):
    for axis, values in TARGET.items():
        monkeypatch.setitem(SUPPORTED, axis, values)


@pytest.mark.parametrize("rotate", [False, True], ids=["inorder", "rotated"])
@pytest.mark.parametrize("num_writer_riscs", [1, 2])
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[_scenario_id(s) for s in SCENARIOS])
def test_target_cells(device, target_supported, monkeypatch, scenario, num_writer_riscs, rotate):
    monkeypatch.setattr(pd, "NUM_WRITER_RISCS", num_writer_riscs)
    monkeypatch.setattr(pd, "ROTATE_WRITE_ORDER", rotate)
    case = _Case(device, scenario, salt=20)
    plan = describe_plan(case.tt_input, case.output_mc)
    assert plan["num_writer_riscs"] == num_writer_riscs
    assert plan["rotate_write_order"] == rotate
    written = sum(b["block"][2] * b["block"][3] for c in plan["cores"] for b in c["blocks"])
    assert written == plan["tensor_tiles"][0] * plan["tensor_tiles"][1], "plan does not cover every page once"
    output = g6_sharded_to_interleaved(case.tt_input, case.output_mc)
    case.check_output(output)
    case.check_input_intact()
    ttnn.deallocate(output)
    ttnn.deallocate(case.tt_input)


@pytest.mark.parametrize("rotate", [False, True, None], ids=["inorder", "rotated", "auto"])
@pytest.mark.parametrize("num_writer_riscs", [1, 2])
@pytest.mark.parametrize("scenario", [P2_FLUX, P1_PANOPTIC], ids=["p2", "p1"])
def test_profile(device, target_supported, monkeypatch, scenario, num_writer_riscs, rotate):
    """One warm call after a compile call; read DEVICE KERNEL DURATION from the profiler CSV."""
    monkeypatch.setattr(pd, "NUM_WRITER_RISCS", num_writer_riscs)
    monkeypatch.setattr(pd, "ROTATE_WRITE_ORDER", rotate)
    case = _Case(device, scenario, salt=21)
    for _ in range(2):
        output = g6_sharded_to_interleaved(case.tt_input, case.output_mc)
        ttnn.deallocate(output)
    ttnn.deallocate(case.tt_input)


@pytest.mark.parametrize("scenario,rotated", [(P2_FLUX, False), (P1_PANOPTIC, True)], ids=["p2", "p1"])
def test_default_plan(device, target_supported, scenario, rotated):
    """Default knobs: both DM RISCs write; rotation only where in-order steps share banks."""
    case = _Case(device, scenario, salt=22)
    plan = describe_plan(case.tt_input, case.output_mc)
    assert plan["regime"] == "owner_writes_whole_shard"
    assert plan["num_writer_riscs"] == 2
    assert plan["rotate_write_order"] is rotated
    output = g6_sharded_to_interleaved(case.tt_input, case.output_mc)
    case.check_output(output)
    ttnn.deallocate(output)
    ttnn.deallocate(case.tt_input)
