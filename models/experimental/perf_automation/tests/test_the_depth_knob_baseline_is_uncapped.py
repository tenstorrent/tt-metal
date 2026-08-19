# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The depth-knob bridge compared a capped run against a capped run.

Run 10, 2026-08-19:

    depth-knob bridge: {'TT_PERF_LAYERS': '2', 'TT_PERF_STACK0_LAYERS': '2',
                        'TT_PERF_STACK1_LAYERS': '2'} did not reduce work
                        (op-count 3572->3572); ignoring

on a pipeline that implements the cap correctly for BOTH towers -- n_layers for the text stack,
n_encode_layers for the audio stack, applied through _cap_stack_from_end.

THE FIX FOR THIS EXISTS AND WAS APPLIED TO ONE CALLER. 2026-07-19 (8e07303854) added `full_hint` so
the bridge would reuse the coverage probe's full-model signal -- "no fragile 2nd detection probe" --
and wired before_loop.py to pass it. run.py's two call sites came from 9b90eed89d, one day EARLIER,
and were never brought along. So they call the bridge with no baseline at all: `full_op =
int(full_hint)` is 0, the "uncapped" probe runs through _run_op_sigs(..., _cov_int), and
_run_op_sigs calls _set_depth(env, k) -- writing the CAP. Both halves ran at depth 2, identical by
construction, and the knob could never be shown to work.

Both halves of that are fixed here: run.py's call sites pass the signal they already hold in
_cov_facts, and the baseline probe -- reached only when no hint exists at all -- asks for ALL layers
rather than for the cap.

WHY IT SURVIVED: the OTHER caller works, so the bridge visibly succeeds in the same logs. Both
outcomes appear in one run, same model, same variables:

    enforcing {'TT_PERF_LAYERS': '2', 'TT_PERF_DECODE_LAYERS': '2', ...} (op-count 25034->3612)
    ... {'TT_PERF_LAYERS': '2', 'TT_PERF_STACK0_LAYERS': '2', ...} did not reduce work (3612->3612)

3612 is the CAPPED count from the first line arriving as the second's "full" baseline. The
discriminator is which caller asked, not which knob the model reads -- an earlier reading of this
blamed the model, and the first line disproves it.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def test_the_baseline_probe_asks_for_all_layers():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _bridge_depth_env(")
    body = src[i : src.index("\ndef ", i + 1)]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    j = code.index("if full_op <= 0:")
    baseline = code[j : j + 300]
    assert "_run_op_sigs(repo_root, mcp_env, devices, node, case, 0)" in baseline, (
        "the uncapped baseline is measured at the cap again: %s" % baseline[:160]
    )
    assert "_cov_int)" not in baseline, "the cap is still being used for the baseline"


def test_zero_means_all_layers_not_zero_layers():
    """The convention the fix relies on: a non-positive depth CLEARS the cap and arms the guard,
    rather than writing a literal 0 that a builder reads as 'build zero layers'."""
    from agent.layer_depth import set_depth, ENV, FORCE_ALL

    capped = set_depth({}, 2)
    assert capped[ENV] == "2" and FORCE_ALL not in capped

    uncapped = set_depth({}, 0)
    assert ENV not in uncapped, "a literal depth survived into the all-layers env"
    assert uncapped[FORCE_ALL] == "1"


def test_the_two_halves_of_the_comparison_use_different_depths():
    """The property that was violated: whatever the baseline runs at, it must not be the cap."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _bridge_depth_env(")
    body = src[i : src.index("\ndef ", i + 1)]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))

    baseline_call = code[code.index("if full_op <= 0:") :].split("\n")[1]
    probe_call = next(ln for ln in code.splitlines() if "seq2 = _run_op_sigs" in ln)
    assert baseline_call.strip() != probe_call.strip(), "baseline and capped probe are the same call"
    assert baseline_call.rstrip().endswith("case, 0)"), baseline_call


def test_the_callers_pass_the_baseline_they_already_measured():
    """The bridge without full_hint has no baseline and must probe for one. 8e07303854 added the
    parameter and wired before_loop; run.py's two call sites predate it by a day and never were."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    calls = [i for i in range(len(src)) if src.startswith("_bridge_depth_env(", i)]
    calls = [i for i in calls if src[max(0, i - 4) : i] != "def "]  # the definition is not a call
    assert len(calls) == 2, "call sites changed; check they still pass a baseline"
    for i in calls:
        assert "full_hint=" in src[i : i + 420], "a call site probes blind again"

    bl = (_PA / "agent" / "before_loop.py").read_text()
    j = bl.index("_bridge_depth_env(")
    assert "full_hint=" in bl[j : j + 420], "before_loop stopped passing its measured baseline"
