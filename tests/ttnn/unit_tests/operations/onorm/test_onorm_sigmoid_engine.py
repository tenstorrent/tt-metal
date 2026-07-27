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


# Tiles per DEST window in the two gate phases.  1 is the Phase-0 (byte-identical)
# setting; 4 is DEST_AUTO_LIMIT under a 32-bit DEST, 8 under the 16-bit DEST that
# Refinement 1b made the default.  The descriptor CLAMPS the request to whatever
# the active compute config can stage, so all four values are legal everywhere.
DEST_TILES = [1, 2, 4, 8]


@pytest.fixture
def restore_engine():
    saved = (pd.SIGMOID_ENGINE, pd.ALLOW_SIGMOID_ABLATION, pd.GATE_DEST_TILES)
    yield
    pd.SIGMOID_ENGINE, pd.ALLOW_SIGMOID_ABLATION, pd.GATE_DEST_TILES = saved


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


def _run(device, batch, tokens, check, cfg=None):
    t_o, t_g, t_w = _inputs(batch, tokens)
    o = ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(t_g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = onorm(o, g, w, compute_kernel_config=cfg or default_compute_kernel_config())
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


@pytest.mark.parametrize("engine", SHIPPING_ENGINES)
@pytest.mark.parametrize("dest_tiles", DEST_TILES)
@pytest.mark.parametrize("batch, tokens", [(1, 32), (1, 128)])
def test_gate_dest_tiles_correct(device, restore_engine, engine, dest_tiles, batch, tokens):
    """Every legal (SIGMOID_ENGINE, GATE_DEST_TILES) cell gives the same answer.

    The two knobs are orthogonal by construction — the engine picks WHICH TRISC
    issues the SFPU, the block factor picks HOW MANY tiles share a DEST window —
    so the cross product is what proves neither silently constrains the other.
    """
    pd.SIGMOID_ENGINE = engine
    pd.GATE_DEST_TILES = dest_tiles
    _run(device, batch, tokens, check=True)


def test_gate_dest_tiles_over_limit_rejected(device, restore_engine, expect_error):
    """A GATE_DEST_TILES above the widest DEST budget is refused, naming the limit.

    16 exceeds `DEST_AUTO_LIMIT` even at the 16-bit DEST (8), so no compute
    config can stage it — that is the case the host assert must still catch.
    """
    pd.GATE_DEST_TILES = 16
    t_o, t_g, t_w = _inputs(1, 32)
    o = ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(t_g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(AssertionError, "GATE_DEST_TILES"):
        onorm(o, g, w)


@pytest.mark.parametrize("dest_tiles", DEST_TILES)
def test_gate_dest_tiles_clamped_for_fp32_dest(device, restore_engine, dest_tiles):
    """The op's own 8-tile default must not break a caller who asks for fp32 DEST.

    Refinement 1b made `DEST_AUTO_LIMIT` a function of the caller's DEST width
    (8 at 16-bit, 4 at 32-bit), so `GATE_DEST_TILES` became a REQUEST that the
    descriptor clamps.  This is the guard on that clamp: the shipping default of
    8 — which a 32-bit DEST cannot stage — must still produce a correct answer
    under `fp32_dest_acc_en=True`, from the same unmodified module-level knob.
    """
    pd.GATE_DEST_TILES = dest_tiles
    _run(device, 1, 32, check=True, cfg=_FP32_ON)


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

# (engine, gate_dest_tiles) candidates, all at the SHIPPING compute config.
# "math"/1 is the Phase-0 control.  R1b added "math"/8: the 16-bit DEST doubled
# DEST_AUTO_LIMIT, so 8 extends the block-factor curve past where R1 could go.
CANDIDATES = [
    ("math", 1),
    ("math", 2),
    ("math", 4),
    ("math", 8),
    ("pack", 1),
    ("ablate", 1),
    ("ablate", 8),
]

TRIAL_CASES = [
    (t, b, tok, engine, dest) for t in range(N_TRIALS) for (b, tok) in SHAPES for (engine, dest) in CANDIDATES
]


@pytest.mark.parametrize("trial, batch, tokens, engine, dest_tiles", TRIAL_CASES, ids=lambda v: str(v))
def test_engine_trial(device, restore_engine, trial, batch, tokens, engine, dest_tiles):
    pd.SIGMOID_ENGINE = engine
    pd.GATE_DEST_TILES = dest_tiles
    pd.ALLOW_SIGMOID_ABLATION = engine == "ablate"
    # The ablation engine is numerically wrong on purpose — it is a payload stub,
    # not a candidate.  The shipping engines are still correctness-gated here so
    # the sweep cannot win by being wrong.
    _run(device, batch, tokens, check=engine != "ablate")


# ---------------------------------------------------------------------------
# 3. DEST width: how much of the SFPU cost is the 32-bit DEST?  (R1 priced it,
#    Refinement 1b adopted it and re-swept GATE_DEST_TILES against it.)
# ---------------------------------------------------------------------------
#
# R1 measured this as information only (its constraint (a) forbade *reaching
# for* the flag).  R1b established that route 1 — preserving P1's fp32
# sum-of-squares accumulation by another mechanism — is unavailable on this
# hardware (the packer L1 accumulator, the only fp32 datapath that bypasses
# DEST, is fp32-DEST-only), took route 2 (documented deviation), and made the
# 16-bit DEST the default.  So `_FP32_OFF` is now the SHIPPING configuration and
# `_FP32_ON` is the caller-override path that must keep working.
#
# The 16-bit DEST doubles DEST_AUTO_LIMIT 4 -> 8, which is the named free rider:
# the sweep below re-measures GATE_DEST_TILES at 4 vs 8 on both occupancy
# regimes, since 8 was not reachable before.

_FP32_OFF = default_compute_kernel_config()  # the shipping default since R1b
assert _FP32_OFF.fp32_dest_acc_en is False, "R1b: default_compute_kernel_config() must ship the 16-bit DEST"

# The caller-override path (the Phase-0 / R1 default), spelled out rather than
# taken from the factory so this stays a fixed reference point if the default
# moves again.
_FP32_ON = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
    dst_full_sync_en=False,
)

# label -> (compute config, GATE_DEST_TILES request, sigmoid engine).
# `fp32_dest_on_d4` is the R1 shipped default and therefore the control.
DEST_ACC_VARIANTS = {
    "fp32_dest_on_d4": (_FP32_ON, 4, "math"),
    "fp32_dest_off_d4": (_FP32_OFF, 4, "math"),
    "fp32_dest_off_d8": (_FP32_OFF, 8, "math"),
    "fp32_dest_off_ablate": (_FP32_OFF, 8, "ablate"),
    "fp32_dest_on_ablate": (_FP32_ON, 4, "ablate"),
}

# Both occupancy regimes the R1b "Done when" names.
DEST_ACC_SHAPES = [(1, 640), (8, 640)]

DEST_ACC_CASES = [
    (t, b, tok, label) for t in range(N_TRIALS) for (b, tok) in DEST_ACC_SHAPES for label in DEST_ACC_VARIANTS
]


@pytest.mark.parametrize("trial, batch, tokens, label", DEST_ACC_CASES, ids=lambda v: str(v))
def test_dest_acc_trial(device, restore_engine, trial, batch, tokens, label):
    """Prices the sigmoid's SFPU payload under a 16-bit vs 32-bit DEST.

    The two `*_ablate` cells price the sigmoid payload separately at each DEST
    width: `median(math) - median(ablate)` at the SAME width is the payload's own
    contribution, so the pair shows the 16-bit win is the SFPU's and not the
    surrounding scaffolding's.
    """
    cfg, dest_tiles, engine = DEST_ACC_VARIANTS[label]
    pd.SIGMOID_ENGINE = engine
    pd.GATE_DEST_TILES = dest_tiles
    pd.ALLOW_SIGMOID_ABLATION = engine == "ablate"
    _run(device, batch, tokens, check=engine != "ablate", cfg=cfg)


# ---------------------------------------------------------------------------
# 4. Non-regression guard set for the adopted GATE_DEST_TILES default.
# ---------------------------------------------------------------------------
#
# op_requirements.md's "config-spanning guard set": the shape/occupancy span
# (1 / 4 / 20 / 110 cores) plus the config span (math_approx_mode, LoFi).  Each
# cell is measured trial-major interleaved at BOTH the OLD shipped configuration
# (32-bit DEST, GATE_DEST_TILES=4 — R1's default) and the NEW one (16-bit DEST,
# GATE_DEST_TILES=8), so "no regression" is a paired comparison inside one
# process rather than a comparison against a number from another process.
#
# The config span tracks the DEFAULT, so `approx` / `lofi` carry the shipping
# DEST width; `fp32on` is the extra cell that keeps the public
# `fp32_dest_acc_en=True` override on the guard set in its own right.

_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
    dst_full_sync_en=False,
)
_APPROX = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
    dst_full_sync_en=False,
)

GUARD_CELLS = [
    ((1, 32), "default"),
    ((1, 128), "default"),
    ((1, 640), "default"),
    ((8, 640), "default"),
    ((1, 640), "approx"),
    ((1, 640), "lofi"),
    ((1, 640), "fp32on"),
]
_GUARD_CFGS = {"default": _FP32_OFF, "approx": _APPROX, "lofi": _LOFI, "fp32on": _FP32_ON}

# (old shipped setting, new shipped setting).  `_GUARD_CFGS[cfg_name]` supplies
# the DEST width for the "new" arm; the "old" arm pins the R1 pair explicitly.
# The new arm reads GATE_DEST_TILES from the module rather than restating it, so
# this guard always measures WHAT ACTUALLY SHIPS — including after Refinement 3
# re-sweeps the knob.  Captured at import, before any test mutates it.
_SHIPPING_GATE_DEST_TILES = pd.GATE_DEST_TILES
GUARD_ARMS = {"r1": (_FP32_ON, 4), "r1b": (None, _SHIPPING_GATE_DEST_TILES)}

GUARD_CASES = [
    (t, shape, cfg_name, arm) for t in range(N_TRIALS) for (shape, cfg_name) in GUARD_CELLS for arm in GUARD_ARMS
]


@pytest.mark.parametrize("trial, shape, cfg_name, arm", GUARD_CASES, ids=lambda v: str(v))
def test_guard_set_trial(device, restore_engine, trial, shape, cfg_name, arm):
    arm_cfg, dest_tiles = GUARD_ARMS[arm]
    pd.GATE_DEST_TILES = dest_tiles
    _run(device, shape[0], shape[1], check=True, cfg=arm_cfg or _GUARD_CFGS[cfg_name])
