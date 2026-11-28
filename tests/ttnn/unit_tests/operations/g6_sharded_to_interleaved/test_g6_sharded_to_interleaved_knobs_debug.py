# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Debug coverage of the op's generic path and its num_write_lanes knob.

Bypasses the Phase 0 support gate (calls the internal dispatch after the structural checks), so
the cells a refinement will open (height sharded, bfloat8_b, L1 output) and the two-lane split are
checked bit for bit with the acceptance test's own inputs and grading.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import ttnn

from ttnn.operations.g6_sharded_to_interleaved import describe_plan
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved import (
    _dispatch,
    _scenario,
    _structural_checks,
)

_spec = importlib.util.spec_from_file_location(
    "g6_s2i_acceptance_for_knobs", Path(__file__).with_name("test_g6_sharded_to_interleaved.py")
)
acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acc)

_SCENARIOS = [acc.P2_FLUX, acc.P1_PANOPTIC, acc.P2_OTHER_SPEC, acc.P1_OTHER_SPEC, acc.P2_FLUX_TO_L1]


def _ungated_call(tt_input, scenario, num_write_lanes):
    memory_config = acc._OUTPUT_MEMORY_CONFIG[scenario["buffer"]]
    _structural_checks(tt_input, memory_config)
    return _dispatch(tt_input, memory_config, _scenario(tt_input, memory_config), num_write_lanes)


@pytest.mark.parametrize("num_write_lanes", [1, 2])
@pytest.mark.parametrize("scenario", _SCENARIOS, ids=acc._scenario_id)
def test_ungated_exact(device, scenario, num_write_lanes):
    torch.manual_seed(42)
    created = []
    try:
        tt_input, expected = acc._make_input(device, scenario, created)
        input_address = tt_input.buffer_address()
        output = _ungated_call(tt_input, scenario, num_write_lanes)
        created.append(output)
        acc._assert_output_contract(output, scenario)
        acc._assert_bit_equal(ttnn.to_torch(output), expected, "output")
        assert tt_input.buffer_address() == input_address
        acc._assert_bit_equal(ttnn.to_torch(tt_input), expected, "input")
    finally:
        acc._free(created)


def test_describe_plan_p2(device):
    torch.manual_seed(42)
    created = []
    try:
        tt_input, _ = acc._make_input(device, acc.P2_FLUX, created)
        plan = describe_plan(tt_input, ttnn.DRAM_MEMORY_CONFIG)
        assert plan["plan"] == "owner_writes"
        writers = [c for c in plan["cores"] if c["role"] == "writer"]
        assert len(writers) == 32
        assert all(c["block"][2] * c["block"][3] == 5 for c in writers)
    finally:
        acc._free(created)
