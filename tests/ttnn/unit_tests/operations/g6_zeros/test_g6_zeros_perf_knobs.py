# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-lamp sweep for g6_zeros: turn one host knob, run p2 then p3, check zeros.

Run under `scripts/run_safe_pytest.sh --profile --no-py-zones`; the profiler CSV lists the
two programs of each parametrization in order (p2, p3). Correctness is asserted, perf is read
from the CSV.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.g6_zeros import g6_zeros

# The package re-exports the function g6_zeros, which shadows the submodule of the same name.
op_module = importlib.import_module("ttnn.operations.g6_zeros.g6_zeros")
descriptor_module = importlib.import_module("ttnn.operations.g6_zeros.g6_zeros_program_descriptor")

SHAPES = [(16, 12, 128, 128), (8, 12, 128, 128)]

KNOBS = [
    pytest.param({}, id="default"),
    pytest.param({"WRITER_CORES_PER_BANK": None}, id="L1-whole-grid"),
    pytest.param({"WRITER_CORES_PER_BANK": 4}, id="L1-4-per-bank"),
    pytest.param({"BANKS_PER_CORE": 8}, id="L4-banks_per_core8"),
]


@pytest.mark.parametrize("knobs", KNOBS)
def test_g6_zeros_perf_knobs(knobs, device, monkeypatch):
    for name, value in knobs.items():
        monkeypatch.setattr(descriptor_module, name, value)
    # The host cache key holds no knob: a turned knob needs a fresh descriptor.
    monkeypatch.setattr(op_module, "_DESCRIPTOR_CACHE", {})
    for shape in SHAPES:
        output = g6_zeros(
            list(shape),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        bits = ttnn.to_torch(output).contiguous().view(torch.int32)
        assert not bool(bits.any()), f"{knobs} {shape}: not all +0.0"
        ttnn.deallocate(output)
