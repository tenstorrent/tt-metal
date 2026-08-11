# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""User-facing configuration sweep for ttnn.experimental.regime_a_matmul.

ONE COMMAND, offline. Tuning is deliberately NOT in the runtime operator: this measures on hardware and emits
entries for `kTable` in regime_a_matmul_config.cpp, which the shipped picker already consults. The operator
gains no file I/O, no global state, and no first-call measurement.

    # tune one or more shapes (MxKxN) and print the winning tuple per shape
    REGIME_A_TUNE_SHAPES=512x6144x768,32x6144x6144 \
      pytest tests/ttnn/unit_tests/operations/matmul/test_regime_a_autotune.py -s

    # widen the shortlist, confirm with more relaunches, and WRITE the kTable patch (shows the diff)
    REGIME_A_TUNE_SHAPES=512x6144x768 REGIME_A_TUNE_TOPK=16 REGIME_A_TUNE_RELAUNCHES=3 \
      REGIME_A_TUNE_APPLY=1 pytest tests/ttnn/unit_tests/operations/matmul/test_regime_a_autotune.py -s

    # equivalent direct invocation -- the test is a thin wrapper; prefer this in scripts
    python3 tools/mm_sweep/picker_gen/autotune.py 512x6144x768 --topk 8 --relaunches 2 [--apply]

Options are environment variables rather than pytest flags on purpose: custom flags would have to be
registered in the shared conftest.py for this directory, which every other matmul test also loads.

WHAT IT GUARANTEES, in order:
  1. FEASIBLE ONLY -- candidates come from autotune_feas.enumerate_full, an exact mirror of the C++
     pick_plan / compute_cb_sizes rules, so nothing that would TT_FATAL at program build is ever launched.
  2. CORRECTNESS BEFORE TIMING -- each candidate's first call is untimed and checked: PCC >= 0.999 against a
     torch reference, AND an explicit zero-non-finite check. A candidate that fails either is discarded
     before its timing is looked at, so a wrong config can never win on speed. (The two gates catch different
     things: a handful of NaN/Inf among millions of elements barely moves PCC -- see BUG_rscatter_nonfinite.md.)
  3. WARM AND REPEATED -- per candidate, 2 blocks x [2 warmup + 12 timed] iterations on resident inputs, with
     device time from the profiler rather than host wall. The winner is then re-confirmed against the shipped
     pick across REGIME_A_TUNE_RELAUNCHES fresh processes, because a single reading is not enough on this
     hardware: that gate rejected 6 of 32 apparent wins during the original campaign.
  4. THE SHIPPED CONFIG IS ALWAYS A CANDIDATE -- config=None (the production picker) is measured alongside the
     shortlist, and a winner is reported only if it beats it by more than REGIME_A_TUNE_MIN_GAIN percent. The
     tool therefore cannot propose something slower than what already ships.

RUNTIME: minutes per shape (each candidate is a fresh process: device open + JIT + 24 timed iterations).
Budget roughly `shapes x (2*topk + 2*relaunches + 1)` process launches. The repo-wide 300s pytest timeout is
disabled for this test because of that; per-dispatch hang detection still applies under run_safe_pytest.sh.

Needs real hardware and TT_METAL_HOME. Opt-in via REGIME_A_TUNE_SHAPES, so a plain pytest run skips.
"""
import os
import subprocess
import sys

import pytest

TOOL = "tools/mm_sweep/picker_gen/autotune.py"


# pytest-timeout is 300s repo-wide, and a tuning sweep is unbounded BY DESIGN: its runtime scales with
# shapes x shortlist x relaunches, and each candidate pays a fresh device open plus a JIT compile for a config
# that has never been built. 0 disables the timeout for this test only. Hang protection is not lost -- it moves
# to where it belongs: run_safe_pytest.sh sets TT_METAL_OPERATION_TIMEOUT_SECONDS, which fires per dispatch
# (~ms for these matmuls) rather than on total wall time, and resets the device if it trips.
@pytest.mark.timeout(0)
def test_regime_a_config_sweep():
    """Tune every requested shape. Fails only if the sweep itself breaks -- the shipped pick winning is a
    normal, successful outcome (it means the table needs no entry for that shape)."""
    shapes = os.environ.get("REGIME_A_TUNE_SHAPES", "").strip()
    if not shapes:
        pytest.skip("set REGIME_A_TUNE_SHAPES=MxKxN[,MxKxN...] to run the configuration sweep")
    root = os.environ.get("TT_METAL_HOME")
    assert root, "TT_METAL_HOME must be set (the profiler worker needs it)"

    cmd = [sys.executable, TOOL] + [s for s in shapes.split(",") if s.strip()]
    cmd += ["--topk", os.environ.get("REGIME_A_TUNE_TOPK", "8")]
    cmd += ["--relaunches", os.environ.get("REGIME_A_TUNE_RELAUNCHES", "2")]
    cmd += ["--min-gain", os.environ.get("REGIME_A_TUNE_MIN_GAIN", "1.5")]
    if os.environ.get("REGIME_A_TUNE_APPLY"):
        cmd += ["--apply"]

    # Streamed, not captured: this runs for minutes per shape and the per-shape verdicts ARE the deliverable,
    # so they should appear as they happen under `-s` rather than only in a failure dump.
    assert subprocess.run(cmd, cwd=root).returncode == 0, "autotune failed"
