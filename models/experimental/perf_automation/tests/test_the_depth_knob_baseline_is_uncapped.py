# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The depth-knob bridge compared a capped run against a capped run.

Run 10, 2026-08-19:

    depth-knob bridge: {'TT_PERF_LAYERS': '2', 'TT_PERF_STACK0_LAYERS': '2',
                        'TT_PERF_STACK1_LAYERS': '2'} did not reduce work
                        (op-count 3572->3572); ignoring

on a pipeline that implements the cap correctly for BOTH towers -- n_layers for the text stack,
n_encode_layers for the audio stack, applied through _cap_stack_from_end.

The baseline was the broken half. Neither caller passes full_hint, so `full_op = int(full_hint)` is
0 and the "uncapped" probe runs through _run_op_sigs(..., _cov_int) -- and _run_op_sigs calls
_set_depth(env, k), writing the CAP. Both halves therefore ran at depth 2 and were identical by
construction. The knob could never be shown to work, so every profile ran at full depth.

WHY IT SURVIVED: it only bites a model whose depth knob IS this tool's own env convention. A model
exposing a custom knob ignores TT_PERF_LAYERS, so writing it leaves that baseline genuinely uncapped
and the comparison works -- which is every model the bridge was developed against. An
emit-e2e-shaped pipeline reads exactly the variable under test.
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
