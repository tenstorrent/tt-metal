# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profiling harness for g6_pad: one call per Phase 0 production shape named in op_design.md's
examples, for `run_safe_pytest.sh --profile`. Prints the plan each call selected (BB-08).
Correctness is gated by test_g6_pad.py, not here."""

import os

import pytest
import torch
import ttnn

import ttnn.operations.g6_pad as g6
from ttnn.operations.g6_pad import g6_pad_program_descriptor as g6_descriptors

# Knob sweeps: G6PAD_KNOBS="E_SECOND_RISC_SHARE=(1,3);MIN_CORE_WORK_TILES=2"
for assignment in filter(None, os.environ.get("G6PAD_KNOBS", "").split(";")):
    knob, value = assignment.split("=", 1)
    assert hasattr(g6_descriptors, knob), knob
    setattr(g6_descriptors, knob, eval(value))

CASES = {
    "p2_258": ((1, 1, 2, 62080), ((0, 0), (0, 0), (0, 30), (0, 0)), 0.0),
    "p3_253": ((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 8), (0, 0)), 0),
    "p4_291": ((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 24), (0, 0)), 0.0),
    "c247": ((1, 3844, 1536), ((0, 0), (0, 28), (0, 0)), 0.0),
    "c254": ((16, 1, 1536), ((0, 0), (0, 31), (0, 0)), 0.0),
    "c248": ((1, 3844, 12), ((0, 0), (0, 28), (0, 0)), 0.0),
    "c64": ((1, 196, 128), ((0, 60), (0, 0)), 0),
    "c65": ((1, 1, 196, 196), ((0, 60), (0, 60)), -1.7014118346046923e38),
    "c72": ((1, 1, 207, 207), ((0, 49), (0, 49)), -1.7014118346046923e38),
}


SELECTED = [n for n in os.environ.get("G6PAD_CASES", "").split(",") if n] or list(CASES)


@pytest.mark.parametrize("name", SELECTED)
def test_g6_pad_profile(device, name):
    shape, padding, value = CASES[name]
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = g6.g6_pad(tt_input, padding, value=value)
    ttnn.synchronize_device(device)
    plan = g6.last_plan()
    plan.pop("roles", None)
    print(f"PLAN {name}: {plan}")
    assert list(out.shape) == [
        d + h for d, h in zip(shape, [0] * (len(shape) - len(padding)) + [p[1] for p in padding])
    ]
