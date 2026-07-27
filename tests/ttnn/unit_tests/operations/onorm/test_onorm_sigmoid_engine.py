# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — SIGMOID_ENGINE measurement + correctness harness.

DO NOT DELETE.

Two jobs, one file:

1. **Correctness** (`test_engine_correct`) — every SHIPPING engine ("math",
   "pack") must produce the same answer.  Runs without the profiler.

2. **Measurement** (`test_engine_trial`) — trial-major interleaved comparison of
   the three engines, including the "ablate" engine that removes the sigmoid
   payload while keeping every CB wait/push, DEST window and NoC transfer.  Run
   under the profiler and read the MEDIAN `DEVICE KERNEL DURATION [ns]` per
   configuration:

       scripts/run_safe_pytest.sh --profile --run-all \
           tests/ttnn/unit_tests/operations/onorm/test_onorm_sigmoid_engine.py

   `median(math) - median(ablate)` is the sigmoid payload's TRUE contribution to
   the critical path.  That number is what the per-phase `MaybeDeviceZoneScope`
   around P7b CANNOT give you: the zone wraps the helper's own
   `cb_wait_front(cb_gate_tiles)`, so a phase starved by the reader reads as an
   expensive phase.  op_requirements.md's own measurement-discipline note says
   exactly this — "never attribute cost to a phase on zone time alone".

The trial-major interleave and the N_TRIALS median are the discipline
`test_onorm_trials.py` documents: single-shot onorm numbers are not reproducible
across processes (a 248 us vs 102 us swing on identical config is on record).
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

N_TRIALS = 5

# The two occupancy regimes Refinement 1's "Done when" names, plus a small-T
# shape so a regression at low core count cannot hide.
SHAPES = [(1, 128), (1, 640), (8, 640)]

# Engines under comparison.  "math" is the Phase-0 default and the control.
ENGINES = ["math", "pack", "ablate"]

# Engines that must be numerically correct.  "ablate" is a measurement stub.
SHIPPING_ENGINES = ["math", "pack"]


@pytest.fixture
def restore_engine():
    saved = (pd.SIGMOID_ENGINE, pd.ALLOW_SIGMOID_ABLATION)
    yield
    pd.SIGMOID_ENGINE, pd.ALLOW_SIGMOID_ABLATION = saved


def _inputs(batch, tokens):
    torch.manual_seed(42)
    t_o = torch.randn(batch, tokens, HV, V, dtype=torch.bfloat16)
    t_g = torch.randn(batch, tokens, FLAT, dtype=torch.bfloat16)
    t_w = torch.randn(1, 1, 1, V, dtype=torch.bfloat16)
    return t_o, t_g, t_w


def _reference(t_o, t_g, t_w, batch, tokens):
    f = t_o.to(torch.float32)
    ref = f * torch.rsqrt(f.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    ref = ref * t_w.to(torch.float32).reshape(1, 1, 1, V)
    return ref.reshape(batch, tokens, FLAT) * torch.sigmoid(t_g.to(torch.float32))


def _run(device, batch, tokens, check):
    t_o, t_g, t_w = _inputs(batch, tokens)
    o = ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(t_g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = onorm(o, g, w, compute_kernel_config=default_compute_kernel_config())
    got = ttnn.to_torch(out).to(torch.float32)
    if check:
        assert_with_pcc(_reference(t_o, t_g, t_w, batch, tokens), got, PCC)
    return got


# ---------------------------------------------------------------------------
# 1. Correctness — both shipping engines agree with torch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", SHIPPING_ENGINES)
@pytest.mark.parametrize("batch, tokens", [(1, 32), (1, 128), (2, 64)])
def test_engine_correct(device, restore_engine, engine, batch, tokens):
    pd.SIGMOID_ENGINE = engine
    _run(device, batch, tokens, check=True)


def test_ablation_is_guarded(device, restore_engine, expect_error):
    """`ablate` must not be reachable without the explicit opt-in flag."""
    pd.SIGMOID_ENGINE = "ablate"
    pd.ALLOW_SIGMOID_ABLATION = False
    t_o, t_g, t_w = _inputs(1, 32)
    o = ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(t_g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(AssertionError, "ALLOW_SIGMOID_ABLATION"):
        onorm(o, g, w)


# ---------------------------------------------------------------------------
# 2. Measurement — trial-major interleaved across engine x shape.
# ---------------------------------------------------------------------------

TRIAL_CASES = [(t, b, tok, engine) for t in range(N_TRIALS) for (b, tok) in SHAPES for engine in ENGINES]


@pytest.mark.parametrize("trial, batch, tokens, engine", TRIAL_CASES, ids=lambda v: str(v))
def test_engine_trial(device, restore_engine, trial, batch, tokens, engine):
    pd.SIGMOID_ENGINE = engine
    pd.ALLOW_SIGMOID_ABLATION = engine == "ablate"
    # The ablation engine is numerically wrong on purpose — it is a payload stub,
    # not a candidate.  The shipping engines are still correctness-gated here so
    # the sweep cannot win by being wrong.
    _run(device, batch, tokens, check=engine != "ablate")
