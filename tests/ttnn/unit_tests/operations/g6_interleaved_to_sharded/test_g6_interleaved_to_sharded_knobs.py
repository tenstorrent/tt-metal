# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every value of the partial_tile_transfer, reads_in_flight, reader_split and reader_processor knobs
is bit exact on the Phase 0 placements.

One op call per case, so a --profile run of this file gives one CSV row per knob value, in
parametrize order, for A/B measurement.
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


@pytest.mark.parametrize("scenario", [FLUX_58, H33_WIDTH], ids=["flux58", "h33_width"])
@pytest.mark.parametrize("transfer", ["valid_rows", "valid_rows_span", "whole_page"])
def test_partial_tile_transfer(scenario, transfer, device, monkeypatch):
    monkeypatch.setattr(descriptor, "PARTIAL_TILE_TRANSFER", transfer)
    _run_exact(scenario, device)


# 1: a barrier after every read. 3: a barrier inside a page's segment pair on flux58 (2 reads per page).
@pytest.mark.parametrize("scenario", [FLUX_58, H33_WIDTH], ids=["flux58", "h33_width"])
@pytest.mark.parametrize("reads_in_flight", [1, 3])
def test_reads_in_flight(scenario, reads_in_flight, device, monkeypatch):
    monkeypatch.setattr(descriptor, "READS_IN_FLIGHT", reads_in_flight)
    _run_exact(scenario, device)


@pytest.mark.parametrize("scenario", [FLUX_58, H33_WIDTH], ids=["flux58", "h33_width"])
@pytest.mark.parametrize("transfer", ["valid_rows", "valid_rows_span"])
def test_reader_split(scenario, transfer, device, monkeypatch):
    monkeypatch.setattr(descriptor, "READER_SPLIT", 2)
    monkeypatch.setattr(descriptor, "PARTIAL_TILE_TRANSFER", transfer)
    _run_exact(scenario, device)


@pytest.mark.parametrize("reader_processor", ["brisc_noc0", "brisc_noc1"])
def test_reader_processor(reader_processor, device, monkeypatch):
    monkeypatch.setattr(descriptor, "READER_PROCESSOR", reader_processor)
    _run_exact(H33_WIDTH, device)
