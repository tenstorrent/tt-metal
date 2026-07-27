# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""onorm compute-block-surface co-tune sweep (Refinement 3, lever 2).

DO NOT DELETE.  This is the measurement vehicle for "co-tune the block factors
against the FINAL structure" — i.e. after Refinement 1b's 16-bit DEST and
Refinement 2's cross-core re-tile, which together changed which knobs are even
live and what their ceilings are.

Read this before adding a candidate — the cross-core split makes the compute-side
knobs **shape-dependent**, because the descriptor clamps each block factor to the
slice the core actually owns:

    tokens_per_core = TOKENS_PER_BLOCK / G      norm_chunk = min(NORM_CHUNK_TOKENS, tokens_per_core)
    cols_per_core   = flat_tiles / G            gate_chunk = min(GATE_CHUNK_TILES, tile_rows*cols_per_core)
                                                gate_dest  = min(GATE_DEST_TILES, dest_limit, gate_chunk)

At the `auto` policy's pick per shape (Refinement 3 recalibrated that policy, which
is what put B=1/T=640 on G=4 — see `test_onorm_retile_group.py`) that gives:

  | shape     | G  | tokens/core | cols/core | norm_chunk | gate_chunk | live knobs        |
  |-----------|----|-------------|-----------|------------|------------|-------------------|
  | B=1,T=32  | 32 | 1           | 4         | 1 (floor)  | 4 (floor)  | depths only       |
  | B=1,T=64  | 32 | 1           | 4         | 1 (floor)  | 4 (floor)  | depths only       |
  | B=1,T=128 | 16 | 2           | 8         | 2          | 8          | norm only         |
  | B=1,T=640 | 4  | 8           | 32        | 2          | 8          | **all of them**   |
  | B=8,T=640 | 2  | 16          | 64        | 2          | 8          | **all of them**   |

So the sweep needs BOTH T=640 cells: the knobs are live on both and their optima
DIFFER (B=8/T=640 peaks at norm 4, B=1/T=640 at norm 2 — the knob is global, so
shipping norm 2 is a priced trade: -1.65 % there, +12.9 % here).  The two smallest
shapes sit at the block-size FLOOR whatever these knobs say, because Refinement 2
spent their coarseness on parallelism (8-16x, far more than coarsening ever paid),
so they cannot regress and are covered by `test_onorm_r3_guard.py` instead.

Direction of the result, because it is counter-intuitive: FINER wins here, the
opposite of the catalog's `compute_block_size` advice.  These knobs set the
PIPELINE FILL before the next stage can start (the writer waits for a whole gate
chunk to be sigmoided; the exchange waits for a whole normalize chunk to be
untilized), not a fixed per-invocation cost — every phase of this kernel sits
between two NoC streams.  See the knob comments in `onorm_program_descriptor.py`.

Discipline (see `test_onorm_trials.py`): trial-major interleaved inside ONE
process, medians over >= 5 trials, every trial also a correctness check.

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/onorm/test_onorm_block_sweep.py
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

# --- lever 2 on the only cell where the compute-side factors are live ---------
# The first entry is the shipping default and acts as the control.  Every
# candidate names the direction it tests, per the catalog's "name the direction,
# sweep, take the measured optimum".
BLOCK_CANDIDATES = [
    # The shipping default is the control.  Both directions of both knobs are kept
    # so the surface's single peak is re-provable on any box, not just asserted.
    ("SHIPPING_n2_g8", {"NORM_CHUNK_TOKENS": 2, "GATE_CHUNK_TILES": 8}),
    # --- NORM_CHUNK_TOKENS, coarser then finer ---
    ("n16_g8", {"NORM_CHUNK_TOKENS": 16, "GATE_CHUNK_TILES": 8}),
    ("n8_g8", {"NORM_CHUNK_TOKENS": 8, "GATE_CHUNK_TILES": 8}),
    ("n4_g8", {"NORM_CHUNK_TOKENS": 4, "GATE_CHUNK_TILES": 8}),
    ("n1_g8", {"NORM_CHUNK_TOKENS": 1, "GATE_CHUNK_TILES": 8}),
    # --- GATE_CHUNK_TILES, coarser then finer ---
    ("n2_g64", {"NORM_CHUNK_TOKENS": 2, "GATE_CHUNK_TILES": 64}),
    ("n2_g32", {"NORM_CHUNK_TOKENS": 2, "GATE_CHUNK_TILES": 32}),
    ("n2_g16", {"NORM_CHUNK_TOKENS": 2, "GATE_CHUNK_TILES": 16}),
    ("n2_g4", {"NORM_CHUNK_TOKENS": 2, "GATE_CHUNK_TILES": 4}),
    # --- the pre-R3 (Refinement 2) corner, for the headline number ---
    ("R2_n8_g64", {"NORM_CHUNK_TOKENS": 8, "GATE_CHUNK_TILES": 64}),
    # --- GATE_DEST_TILES: R1b's knob, re-swept against this surface ---
    ("n2_g8_dest8", {"NORM_CHUNK_TOKENS": 2, "GATE_CHUNK_TILES": 8, "GATE_DEST_TILES": 8}),
    ("n2_g8_dest2", {"NORM_CHUNK_TOKENS": 2, "GATE_CHUNK_TILES": 8, "GATE_DEST_TILES": 2}),
]

# Both cells where the compute-side knobs are live (see the table above).
BLOCK_SHAPES = [(1, 640), (8, 640)]
BLOCK_CASES = [
    (t, b, tok, label, knobs)
    for t in range(N_TRIALS)
    for (b, tok) in BLOCK_SHAPES
    for (label, knobs) in BLOCK_CANDIDATES
]

# --- buffer depths: live at every shape, so swept on an under-filled cell too --
DEPTH_CANDIDATES = [
    ("base_dm8x4_o2_rm2", {"DM_BLOCK_TILES": 8, "DM_DEPTH": 4, "O_DEPTH": 2, "RM_LOCAL_DEPTH": 2}),
    ("o3", {"DM_BLOCK_TILES": 8, "DM_DEPTH": 4, "O_DEPTH": 3, "RM_LOCAL_DEPTH": 2}),
    ("rm3", {"DM_BLOCK_TILES": 8, "DM_DEPTH": 4, "O_DEPTH": 2, "RM_LOCAL_DEPTH": 3}),
    ("dmdepth8", {"DM_BLOCK_TILES": 8, "DM_DEPTH": 8, "O_DEPTH": 2, "RM_LOCAL_DEPTH": 2}),
    ("o3_rm3_dmdepth8", {"DM_BLOCK_TILES": 8, "DM_DEPTH": 8, "O_DEPTH": 3, "RM_LOCAL_DEPTH": 3}),
]

DEPTH_SHAPES = [(1, 640), (8, 640)]
DEPTH_CASES = [
    (t, b, tok, label, knobs)
    for t in range(N_TRIALS)
    for (b, tok) in DEPTH_SHAPES
    for (label, knobs) in DEPTH_CANDIDATES
]

_KNOB_KEYS = (
    "TOKENS_PER_BLOCK",
    "NORM_CHUNK_TOKENS",
    "GATE_CHUNK_TILES",
    "GATE_DEST_TILES",
    "DM_BLOCK_TILES",
    "DM_DEPTH",
    "O_DEPTH",
    "RM_LOCAL_DEPTH",
    "RECONFIG_MODE",
    "RETILE_GROUP_CORES",
)


@pytest.fixture
def restore_knobs():
    saved = {k: getattr(pd, k) for k in _KNOB_KEYS}
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


@pytest.mark.parametrize("trial, batch, tokens, label, knobs", BLOCK_CASES, ids=lambda v: str(v))
def test_block_sweep(device, restore_knobs, trial, batch, tokens, label, knobs):
    """Compute-side block factors, on both cells where they are unclamped."""
    for k, v in knobs.items():
        setattr(pd, k, v)
    _run(device, batch, tokens)


@pytest.mark.parametrize("trial, batch, tokens, label, knobs", DEPTH_CASES, ids=lambda v: str(v))
def test_depth_sweep(device, restore_knobs, trial, batch, tokens, label, knobs):
    """Buffer depths, which stay live at every group size."""
    for k, v in knobs.items():
        setattr(pd, k, v)
    _run(device, batch, tokens)
