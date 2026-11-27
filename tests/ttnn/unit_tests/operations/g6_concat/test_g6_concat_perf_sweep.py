# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blocking-knob sweep on the perf-focus case p1 (run under scripts/run_safe_pytest.sh --profile).

One dispatch per knob setting, in parametrize order, so the n-th generic_op row of the profiler CSV
is the n-th setting. The knobs are in the program-cache key, so the settings never share a program.
Each output is checked bit-exact, so a fast-but-wrong setting cannot read as a win, and the sweep
doubles as a correctness pin for every knob value it visits.

G6C_SWEEP=<id>,<id>,... selects a subset of SWEEP; the default runs all of it.
"""

import os

import pytest
import torch
import ttnn

from ttnn.operations.g6_concat import g6_concat
from ttnn.operations.g6_concat import g6_concat_program_descriptor as program_descriptor

KIB = 1024

P1_SHAPES = [(108, 4096, 128), (84, 4096, 128)]

# Knob overrides per setting; knobs not named keep their module default.
SWEEP = {
    "default": {},
    "whole_pages": {"TRANSFER_BYTES_MAX": 4096},
    "xfer1024": {"TRANSFER_BYTES_MAX": 1024},
    "cap64": {"BLOCK_BYTES_CAP": 64 * KIB},
    "cap256": {"BLOCK_BYTES_CAP": 256 * KIB},
    "d3": {"OUTPUT_PAGES_DEPTH": 3},
    "d4": {"OUTPUT_PAGES_DEPTH": 4},
    "stagger": {"BANK_STAGGER": True},
    "whole_pages_stagger": {"TRANSFER_BYTES_MAX": 4096, "BANK_STAGGER": True},
    "trid2_d2": {"READ_TRIDS": 2},
    "trid2_d3": {"READ_TRIDS": 2, "OUTPUT_PAGES_DEPTH": 3},
    "trid3_d4": {"READ_TRIDS": 3, "OUTPUT_PAGES_DEPTH": 4},
    "trid4_d4": {"READ_TRIDS": 4, "OUTPUT_PAGES_DEPTH": 4},
    "cap32_d4": {"OUTPUT_PAGES_DEPTH": 4, "BLOCK_BYTES_CAP": 32 * KIB},
    "trid3_d4_cap32": {"READ_TRIDS": 3, "OUTPUT_PAGES_DEPTH": 4, "BLOCK_BYTES_CAP": 32 * KIB},
    "whole_pages_trid3_d4": {"TRANSFER_BYTES_MAX": 4096, "READ_TRIDS": 3, "OUTPUT_PAGES_DEPTH": 4},
}


def _selected():
    wanted = os.environ.get("G6C_SWEEP")
    ids = wanted.split(",") if wanted else list(SWEEP)
    return [pytest.param(SWEEP[i], id=i) for i in ids]


_P1_CACHE = {}


@pytest.fixture
def p1(device):
    # Built once per device: the module-scoped device keeps the inputs alive across settings. The
    # device is part of the key because precompile's collect pass runs the bodies on its own
    # capture device first, and its tensors must not leak into the real pass.
    cached = _P1_CACHE.get("p1")
    if cached is not None and cached[0] is device:
        return cached[1]
    if "torch" not in _P1_CACHE:
        torch.manual_seed(0)
        torch_inputs = [torch.randn(s, dtype=torch.float32) for s in P1_SHAPES]
        _P1_CACHE["torch"] = (torch_inputs, torch.cat(torch_inputs, dim=0))
    torch_inputs, expected = _P1_CACHE["torch"]
    tt_inputs = [
        ttnn.from_torch(
            t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for t in torch_inputs
    ]
    _P1_CACHE["p1"] = (device, (tt_inputs, expected))
    return _P1_CACHE["p1"][1]


@pytest.mark.parametrize("knobs", _selected())
def test_g6_concat_p1_knob_sweep(knobs, p1, device):
    tt_inputs, expected = p1
    saved = {name: getattr(program_descriptor, name) for name in knobs}
    for name, value in knobs.items():
        setattr(program_descriptor, name, value)
    try:
        output = g6_concat(tt_inputs, 0)
        ttnn.synchronize_device(device)
    finally:
        for name, value in saved.items():
            setattr(program_descriptor, name, value)
    got = ttnn.to_torch(output)
    output.deallocate()
    assert torch.equal(got.view(torch.int32), expected.view(torch.int32))


# Small multi-block shapes for the knobs p1 cannot reach: trimmed pages (v = 16), runs that cross
# inputs, ragged last blocks (MAX_CORES = 7) and ROW_MAJOR sticks of 5120 B (three pieces).
PIN_CASES = [
    pytest.param([(1, 4, 48, 1024), (1, 4, 48, 2048)], -1, ttnn.float32, ttnn.TILE_LAYOUT, id="fp32_w_trimmed"),
    pytest.param([(2, 3, 64, 128), (5, 3, 64, 128)], 0, ttnn.bfloat16, ttnn.TILE_LAYOUT, id="bf16_outer"),
    pytest.param([(3, 40, 1280), (2, 40, 1280)], 0, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, id="fp32_rm_outer"),
]
PIN_KNOBS = {
    "stagger": {"BANK_STAGGER": True},
    "trid2": {"READ_TRIDS": 2},
    "trid3_d4_stagger": {"READ_TRIDS": 3, "OUTPUT_PAGES_DEPTH": 4, "BANK_STAGGER": True},
    "xfer1024_stagger": {"TRANSFER_BYTES_MAX": 1024, "BANK_STAGGER": True},
}
_TORCH_BITS = {ttnn.bfloat16: (torch.bfloat16, torch.int16), ttnn.float32: (torch.float32, torch.int32)}


@pytest.mark.parametrize("knobs", [pytest.param(k, id=i) for i, k in PIN_KNOBS.items()])
@pytest.mark.parametrize("shapes, dim, dtype, layout", PIN_CASES)
def test_g6_concat_knob_pins(shapes, dim, dtype, layout, knobs, device):
    torch_dtype, bits = _TORCH_BITS[dtype]
    torch.manual_seed(0)
    torch_inputs = [torch.randn(s, dtype=torch.float32).to(torch_dtype) for s in shapes]
    tt_inputs = [
        ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in torch_inputs
    ]
    knobs = {"MAX_CORES": 7, **knobs}
    saved = {name: getattr(program_descriptor, name) for name in knobs}
    for name, value in knobs.items():
        setattr(program_descriptor, name, value)
    try:
        output = g6_concat(tt_inputs, dim)
    finally:
        for name, value in saved.items():
            setattr(program_descriptor, name, value)
    got = ttnn.to_torch(output).to(torch_dtype)
    expected = torch.cat(torch_inputs, dim=dim)
    assert torch.equal(got.contiguous().view(bits), expected.contiguous().view(bits))
