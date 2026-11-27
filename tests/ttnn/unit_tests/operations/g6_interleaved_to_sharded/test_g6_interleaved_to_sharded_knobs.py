# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every value of the LAST_ROW_FORM, READER_WEIGHTS, READER_PROCESSOR and READER_NOC knobs
(perf lamps L1-L3 of op_design.md) is bit exact on the Phase 0 placements.

flux58: every tile row is a last row (trimmed). h33_width: row 0 interior (whole pages),
row 1 trimmed. NaN in the input's tile padding must not reach the logical values.
"""

from __future__ import annotations

import copy

import pytest
import torch
import ttnn

from eval.golden_tests.g6_interleaved_to_sharded.feature_spec import FLUX_58
from eval.golden_tests.g6_interleaved_to_sharded.helpers import (
    build_memory_config,
    check_output,
    create_ttnn_input_tensor,
    make_torch_input,
)
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded_program_descriptor as descriptor

H33_WIDTH = copy.deepcopy(FLUX_58)
H33_WIDTH["shape"] = [33, 5120]
H33_WIDTH["out_memory"]["shard_shape"] = [64, 160]

SCENARIOS = pytest.mark.parametrize("scenario", [FLUX_58, H33_WIDTH], ids=["flux58", "h33_width"])


def _run_exact(scenario, device):
    torch.manual_seed(0)
    torch_input = make_torch_input(scenario["shape"], scenario["dtype"])
    ttnn_input = create_ttnn_input_tensor(
        torch_input,
        device,
        dtype=scenario["dtype"],
        layout=scenario["layout"],
        memory_config=build_memory_config(scenario["in_memory"], device),
    )
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, float("nan"), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    expected = ttnn.to_torch(ttnn_input).clone()
    output_config = build_memory_config(scenario["out_memory"], device)
    output = g6_interleaved_to_sharded(ttnn_input, output_config)
    check_output(
        output,
        expected,
        shape=scenario["shape"],
        dtype=scenario["dtype"],
        layout=scenario["layout"],
        memory_config=output_config,
        gate="exact",
    )


@SCENARIOS
@pytest.mark.parametrize("form", descriptor.LAST_ROW_FORMS)
def test_last_row_form(scenario, form, device, monkeypatch):
    monkeypatch.setattr(descriptor, "LAST_ROW_FORM", form)
    _run_exact(scenario, device)


@SCENARIOS
@pytest.mark.parametrize("form", ["two_spans", "one_span"])
@pytest.mark.parametrize("weights", [(1,), (1, 1), (3, 1)], ids=["1reader", "split1:1", "split3:1"])
def test_reader_weights(scenario, form, weights, device, monkeypatch):
    monkeypatch.setattr(descriptor, "READER_WEIGHTS", weights)
    monkeypatch.setattr(descriptor, "LAST_ROW_FORM", form)
    _run_exact(scenario, device)


@pytest.mark.parametrize(
    "processor,noc",
    [
        (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
    ],
    ids=["ncrisc_noc1", "brisc_noc0", "brisc_noc1"],
)
def test_reader_placement(processor, noc, device, monkeypatch):
    monkeypatch.setattr(descriptor, "READER_PROCESSOR", processor)
    monkeypatch.setattr(descriptor, "READER_NOC", noc)
    _run_exact(H33_WIDTH, device)
