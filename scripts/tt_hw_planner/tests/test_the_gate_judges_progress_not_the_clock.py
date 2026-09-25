# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pin: the e2e gate decides on FORWARD PROGRESS, never on a typed deadline.

The gate ran pytest under subprocess.run(timeout=N) with N a constant, and anything outliving N was
reported as "likely device/fabric hang". But the gate's runtime is not knowable in advance: it
includes building the reference the PCC is scored against, and that cost belongs to the MODEL. A
text model's reference is a generate() of a few dozen tokens (nemotron: seconds); Qwen-Image-Edit's
is 32 samples x 50 diffusion steps x 2 CFG in fp32 on CPU.

Measured 2026-09-25 on a HEALTHY T3K -- the reference had run 3 h 55 m on ~11 cores when the 4 h
budget killed it, and the verdict was:

    G2/G3: tests/e2e exceeded 14400s with no verdict (likely device/fabric hang)
           -- device recovery: board answering after recovery (verified)

It accused the fabric and confirmed the fabric was fine in one sentence, reset a healthy board, and
-- because the reference is saved only once complete -- discarded ~4 h and started again every
round: a loop that cannot converge however long it is given.

probes._execute already decides this correctly, and its docstring names the same failure ("a fixed
wall-clock kill cannot tell 'hung' from 'slow'"). What is pinned here: the gate uses it, no wall is
typed in this module, and a stall is reported as a stall rather than as a hardware accusation.
"""

from __future__ import annotations

import inspect

from scripts.tt_hw_planner.commands import emit_e2e as E

GATE = inspect.getsource(E._run_deterministic_gates)


def _code_of(fn) -> str:
    """Source with the docstring and comments stripped -- prose explaining the change must not
    satisfy (or break) an assertion about the CODE."""
    out = []
    for line in inspect.getsource(fn).splitlines():
        st = line.strip()
        if st.startswith("#"):
            continue
        out.append(line)
    src = "\n".join(out)
    doc = inspect.getdoc(fn)
    if doc:
        for line in doc.splitlines():
            src = src.replace(line, "")
    return src


def test_the_gate_runs_through_the_progress_watchdog():
    assert "_pr._execute(" in _code_of(E._run_deterministic_gates)


def test_the_gate_no_longer_imposes_a_stopwatch():
    """subprocess.run(timeout=...) on the GATE'S PYTEST is what killed a healthy run.

    Scoped to that call. The G6 stack survey further down legitimately keeps a short stopwatch: it
    is a cheap, batch-independent probe whose cost does not depend on the model's reference."""
    code = _code_of(E._run_deterministic_gates)
    assert "hang_timeout" not in code, "the gate's own wall is gone, including from G6's derivation"
    i = code.index("_pr._execute(")
    assert "timeout=" not in code[i : i + 400], "the gate's pytest must not carry a stopwatch"


def test_the_cheap_g6_probe_keeps_its_own_short_wall():
    """Not everything needs the watchdog -- only work whose cost the model decides. G6 must still be
    bounded, and by the caller's budget now that the gate's wall no longer exists."""
    code = _code_of(E._run_deterministic_gates)
    assert "g6_hang = min(int(timeout_s)" in code


def test_no_gate_wall_is_typed_in_this_module():
    """The caller's budget is the only number, and it is a fuse rather than the judge."""
    assert not hasattr(E, "_gate_wall_s"), "the typed wall helper is gone"
    assert not hasattr(E, "_GATE_WALL_PER_SAMPLE_S")
    assert "2700" not in _code_of(E._run_deterministic_gates)


def test_the_callers_budget_is_passed_through_untouched():
    """timeout_s reaches _execute as-is: it reports the budget and keeps a hard ceiling behind it,
    instead of this module clamping it to something it invented."""
    code = _code_of(E._run_deterministic_gates)
    i = code.index("_pr._execute(")
    call = code[i : i + 400]
    assert "int(timeout_s)," in call, "the budget must reach the watchdog"
    assert "min(" not in call, "the gate must not clamp the caller's budget any more"


def test_a_stall_is_reported_as_a_stall_not_as_a_hardware_fault():
    """The old text accused the fabric in the same sentence that verified it was healthy."""
    code = _code_of(E._run_deterministic_gates)
    assert "made no forward progress" in code
    assert "likely device/fabric hang" not in code


def test_a_stall_still_reaches_device_recovery_with_its_partial_output():
    """Recovery must keep the evidence: a real stall may well be a wedge, and the log names the chip."""
    code = _code_of(E._run_deterministic_gates)
    i = code.index("TracyHangError")
    assert "_reset_device(" in code[i : i + 500]
    assert "pytest_out" in code[i : i + 500]


def test_the_watchdog_is_the_shared_one_not_a_copy():
    """No second implementation of hung-vs-slow."""
    from models.experimental.perf_automation.agent import probes as _PR

    assert callable(_PR._execute)
    code = _code_of(E._run_deterministic_gates)
    for invented in ("last_progress", "no log growth", "_kill_tree"):
        assert invented not in code, f"{invented!r} suggests a re-implementation of the watchdog"
