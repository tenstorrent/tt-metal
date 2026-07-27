# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""onorm knob comparison with TRIAL-LOOP discipline — the only trustworthy sweep.

DO NOT DELETE.

Why this file exists (learned the hard way on this op): single-shot
`--profile` numbers for onorm are NOT reproducible — the same configuration
(DM_BLOCK_TILES=4, B=1/T=640) measured 248 us in one process and 102 us in
another. The op reads two large interleaved DRAM tensors per block, so its
achieved bandwidth depends on where those tensors land in DRAM, which depends on
allocation history within the process. A number taken from one shot of one
process is therefore meaningless for comparing knobs.

The discipline this file enforces:

  * every configuration is measured N_TRIALS times, and
  * the configurations are INTERLEAVED (trial-major ordering), so any drift over
    the life of the process hits every configuration equally, and
  * fresh input tensors are allocated per trial.

Read the emitted CSV and take the MEDIAN per configuration (and look at the
spread before believing any difference):

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/onorm/test_onorm_trials.py

Each trial is also a correctness check, so the sweep cannot win by being wrong.
"""

import pytest
import torch

import ttnn
import ttnn.operations.onorm.onorm_program_descriptor as pd
from ttnn.operations.onorm import default_compute_kernel_config, onorm

from tests.ttnn.utils_for_testing import assert_with_pcc

HV, V = 32, 128
FLAT = HV * V
PCC = 0.995

N_TRIALS = 7

# (label, {knob: value}) — the candidates under comparison. The first entry is
# the current default and acts as the control.
CANDIDATES = [
    ("blk4_dep2_o2_OLD_DEFAULT", {"DM_BLOCK_TILES": 4, "DM_DEPTH": 2, "O_DEPTH": 2}),
    ("blk8_dep2_o2", {"DM_BLOCK_TILES": 8, "DM_DEPTH": 2, "O_DEPTH": 2}),
    ("blk8_dep4_o2_DEFAULT", {"DM_BLOCK_TILES": 8, "DM_DEPTH": 4, "O_DEPTH": 2}),
    ("blk4_dep4_o2", {"DM_BLOCK_TILES": 4, "DM_DEPTH": 4, "O_DEPTH": 2}),
]

# trial-major interleave: (t0,c0), (t0,c1), ..., (t1,c0), ...
CASES = [(t, label, knobs) for t in range(N_TRIALS) for (label, knobs) in CANDIDATES]


@pytest.fixture
def restore_knobs():
    keys = ("TOKENS_PER_BLOCK", "NORM_CHUNK_TOKENS", "GATE_CHUNK_TILES", "DM_BLOCK_TILES", "DM_DEPTH", "O_DEPTH")
    saved = {k: getattr(pd, k) for k in keys}
    yield
    for k, v in saved.items():
        setattr(pd, k, v)


def _run(device, batch, tokens):
    torch.manual_seed(42)
    t_o = torch.randn(batch, tokens, HV, V, dtype=torch.bfloat16)
    t_g = torch.randn(batch, tokens, FLAT, dtype=torch.bfloat16)
    t_w = torch.randn(1, 1, 1, V, dtype=torch.bfloat16)
    o = ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(t_g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = onorm(o, g, w, compute_kernel_config=default_compute_kernel_config())
    f = t_o.to(torch.float32)
    ref = f * torch.rsqrt(f.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    ref = ref * t_w.to(torch.float32).reshape(1, 1, 1, V)
    ref = ref.reshape(batch, tokens, FLAT) * torch.sigmoid(t_g.to(torch.float32))
    assert_with_pcc(ref, ttnn.to_torch(out).to(torch.float32), PCC)


@pytest.mark.parametrize("trial, label, knobs", CASES, ids=lambda v: str(v))
def test_trial(device, restore_knobs, trial, label, knobs):
    for k, v in knobs.items():
        setattr(pd, k, v)
    _run(device, 1, 640)


def test_repeated_dispatch_same_tensors(device):
    """Same tensors, 6 back-to-back dispatches, at the DEFAULT knobs.

    Isolates cold-first-dispatch from steady state. If row 0 of the profiler CSV
    is far slower than rows 1..5, the "248 us vs 102 us" gap between processes is
    a first-dispatch / placement effect and NOT a knob effect — which is what the
    module docstring warns about.
    """
    torch.manual_seed(42)
    batch, tokens = 1, 640
    t_o = torch.randn(batch, tokens, HV, V, dtype=torch.bfloat16)
    t_g = torch.randn(batch, tokens, FLAT, dtype=torch.bfloat16)
    t_w = torch.randn(1, 1, 1, V, dtype=torch.bfloat16)
    o = ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(t_g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    f = t_o.to(torch.float32)
    ref = f * torch.rsqrt(f.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    ref = ref * t_w.to(torch.float32).reshape(1, 1, 1, V)
    ref = ref.reshape(batch, tokens, FLAT) * torch.sigmoid(t_g.to(torch.float32))

    for _ in range(6):
        out = onorm(o, g, w, compute_kernel_config=default_compute_kernel_config())
        assert_with_pcc(ref, ttnn.to_torch(out).to(torch.float32), PCC)


# ---------------------------------------------------------------------------
# Cross-shape controlled comparison: default vs the best DM candidate.
# Trial-major interleave again, so drift over the process hits both equally.
# ---------------------------------------------------------------------------

SHAPE_CASES = [
    (t, b, tok, label, knobs)
    for t in range(5)
    for (b, tok) in [(1, 64), (1, 128), (1, 640), (8, 640)]
    for (label, knobs) in [
        ("OLD_blk4_dep2", {"DM_BLOCK_TILES": 4, "DM_DEPTH": 2}),
        ("NEW_DEFAULT_blk8_dep4", {"DM_BLOCK_TILES": 8, "DM_DEPTH": 4}),
    ]
]


@pytest.mark.parametrize("trial, batch, tokens, label, knobs", SHAPE_CASES, ids=lambda v: str(v))
def test_shape_trial(device, restore_knobs, trial, batch, tokens, label, knobs):
    for k, v in knobs.items():
        setattr(pd, k, v)
    _run(device, batch, tokens)
