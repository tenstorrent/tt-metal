# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Configuration sweep (offline autotuner) for ttnn.experimental.small_m_matmul.

One command. Tuning is OFFLINE BY DESIGN: this measures on hardware and emits entries for `kTable` in
small_m_matmul_config.cpp, which the shipped picker already consults. The operator gains no file I/O, no
global state, and no first-call measurement.

RUN IT

    # tune shapes, print the winning tuple per shape
    SMALL_M_TUNE_SHAPES=512x6144x768,32x6144x6144 \
      scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/matmul/test_small_m_matmul_autotune.py -q -s

    # wider shortlist, more confirmation relaunches, and WRITE the kTable patch (prints the diff)
    SMALL_M_TUNE_SHAPES=512x6144x768 SMALL_M_TUNE_TOPK=16 SMALL_M_TUNE_RELAUNCHES=3 \
      SMALL_M_TUNE_APPLY=1 scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/matmul/test_small_m_matmul_autotune.py -q -s

    # direct invocation -- this test is a thin wrapper; prefer this in scripts
    python3 tools/mm_sweep/picker_gen/autotune.py 512x6144x768 --topk 8 --relaunches 2 [--apply]

    env var                      default   meaning
    SMALL_M_TUNE_SHAPES         (none)    MxKxN[,MxKxN...]; unset => this test skips
    SMALL_M_TUNE_TOPK           8         shortlist depth PER RANKER (3 rankers, unioned)
    SMALL_M_TUNE_RELAUNCHES     2         fresh processes used to confirm a win
    SMALL_M_TUNE_MIN_GAIN       1.5       percent; below this the shipped pick is kept
    SMALL_M_TUNE_APPLY          unset     write kTable and show the diff

Options are environment variables rather than pytest flags on purpose: custom flags would have to be
registered in the shared conftest.py for this directory, which every other matmul test also loads.

WHAT IT GUARANTEES

 1. FEASIBLE CONFIGURATIONS ONLY. Candidates come from autotune_feas.enumerate_full, an exact mirror of the
    C++ pick_plan / compute_cb_sizes rules, so nothing that would TT_FATAL at program build is ever launched.
    Keep that mirror in step with the C++: a stale mirror once rejected configs the picker accepts, and a
    heuristic validated on the resulting restricted set then regressed 4-34%.
 2. CORRECTNESS BEFORE TIMING. Each candidate's FIRST call is untimed and gated on PCC >= 0.999 against a
    torch reference AND an explicit zero-non-finite check. A candidate failing either is discarded before its
    timing is looked at, so a wrong config can never win on speed. Both gates are needed -- a handful of
    NaN/Inf among millions of elements barely moves PCC -- a reduce-scatter CB wrap bug was exactly that.
 3. WARM, REPEATED, AND RELAUNCHED. Per candidate: 2 blocks x [2 warmup + 12 timed] iterations on resident
    inputs, device time from the profiler rather than host wall. The winner is then re-confirmed against the
    shipped pick across SMALL_M_TUNE_RELAUNCHES *fresh processes*, and every relaunch must agree. One reading
    is not enough on this hardware: that gate rejected 6 of 32 apparent wins in the original campaign, and see
    the worked example below.
 4. THE SHIPPED CONFIGURATION IS ALWAYS A CANDIDATE. config=None (the production picker) is measured alongside
    the shortlist, and a winner is reported only if it beats it by more than MIN_GAIN percent. The tool
    therefore cannot propose something slower than what already ships.
 5. WINNING TUPLE, AND OPTIONALLY A PATCH. Prints the tuple per shape; --apply writes kTable via
    apply_table.py (which verifies brace shape, updates in place or appends) and shows the diff to review.

WHY MEASURE AT ALL

The analytic picker is a good RANKER and a poor CHOOSER. Held out over ~17,000 timed configs on 27 shapes:
picking 1 config (what ships) is ~7.8% mean regret vs optimal; measuring its top 4 is ~3.1%, top 8 ~1.6%, top
16 ~0.6%. Five attempts to improve the chooser formula all failed to generalise, so the leverage is in
measuring a handful, not in a better formula.

It also fixes table staleness: 14 of the original 44 kTable rows were measured winners when added and were
invalidated by later kernel work. Re-running this after a kernel change re-measures instead of letting rows rot.

WORKED EXAMPLE (2026-08-11, bh-glx-120-c02u02, topk=4 relaunches=2, 13m12s)

    [32x2048x2048] 8 shortlisted; shipped pick 21.78 us already best (shortlist best 4,2,1,2,4 @ 22.13 us)
    [512x6144x768] shortlist 11 measured, best 6,1,2,2,3 @ 51.78 us -- keep shipped pick (-0.3%/+0.1%)
    # nothing to apply: the shipped pick was within 1.5% on every shape

Both outcomes are successes. The second is the relaunch gate earning its keep: one reading looked like a win,
two fresh relaunches disagreed in sign (-0.3%, +0.1%), so it was rejected as noise. 6,1,2,2,3 is in fact
already the kTable entry for that shape -- the tool independently re-derived the shipped pick.

RUNTIME

Minutes per shape. Budget roughly `shapes x (2*topk + 2*relaunches + 1)` process launches; each pays a device
open plus a JIT compile for a config never built before.

The repo-wide 300s pytest-timeout is disabled for this test (see the marker below) because of that. Hang
protection is not lost, it moves to where it belongs: run_safe_pytest.sh sets
TT_METAL_OPERATION_TIMEOUT_SECONDS, which fires per dispatch (~ms for these matmuls) rather than on total wall
time, and resets the device if it trips.

On Galaxy note that run_safe_pytest.sh resets with `tt-smi -r`, which on bh-glx-120-c02u02 left ethernet links
down and downgraded the mesh to 8x2 (16 of 32 chips); recovery needed `tt-smi -glx_reset`.

Needs real hardware and TT_METAL_HOME. Opt-in via SMALL_M_TUNE_SHAPES, so a plain pytest run skips.
"""
import os
import subprocess
import sys

import pytest

TOOL = "tools/mm_sweep/picker_gen/autotune.py"


# A tuning sweep is unbounded by design (shapes x shortlist x relaunches, each a fresh process + JIT), so the
# repo-wide 300s pytest-timeout would kill it mid-run -- it did, twice, at 301s. 0 disables it here only; see
# RUNTIME above for where hang detection lives instead.
@pytest.mark.timeout(0)
def test_small_m_config_sweep():
    """Tune every requested shape. Fails only if the sweep itself breaks -- the shipped pick winning is a
    normal, successful outcome (it means the table needs no entry for that shape)."""
    shapes = os.environ.get("SMALL_M_TUNE_SHAPES", "").strip()
    if not shapes:
        pytest.skip("set SMALL_M_TUNE_SHAPES=MxKxN[,MxKxN...] to run the configuration sweep")
    root = os.environ.get("TT_METAL_HOME")
    assert root, "TT_METAL_HOME must be set (the profiler worker needs it)"

    cmd = [sys.executable, TOOL] + [s for s in shapes.split(",") if s.strip()]
    cmd += ["--topk", os.environ.get("SMALL_M_TUNE_TOPK", "8")]
    cmd += ["--relaunches", os.environ.get("SMALL_M_TUNE_RELAUNCHES", "2")]
    cmd += ["--min-gain", os.environ.get("SMALL_M_TUNE_MIN_GAIN", "1.5")]
    if os.environ.get("SMALL_M_TUNE_APPLY"):
        cmd += ["--apply"]

    # Streamed, not captured: this runs for minutes per shape and the per-shape verdicts ARE the deliverable,
    # so they should appear as they happen under `-s` rather than only in a failure dump.
    assert subprocess.run(cmd, cwd=root).returncode == 0, "autotune failed"
