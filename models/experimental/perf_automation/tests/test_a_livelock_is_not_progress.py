# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU is not evidence of progress, and treating it as such cost ten hours.

RUN 12, 2026-08-20. A perf-test validation hung inside one ttnn.from_torch call, uploading the audio
encoder's first attention weight. Ten hours later it was still there:

    CPU              1 core, pinned
    its own log      last written 00:56, static
    syscr / syscw    41580 -> 41580 across a 20s window
    read/write bytes unchanged
    stack            byte-identical across samples 12s apart
    ctxt switches    6,224 voluntary vs 147,973 involuntary -- it never yields, only gets preempted

Every watchdog in the tool let it run. The stall detector's rule was

    if size > last_size or cpu > last_cpu + 10:
        last_progress = now

so a process burning a core reset its own progress clock on every poll, forever. Its message said
"stalled/hung: no log growth AND ~no CPU" -- it was written for a DEADLOCK, a process asleep waiting
on something that never comes. This was a LIVELOCK: moving constantly, arriving nowhere. And
run.py's copy was weaker still, counting `_llm_child_alive(pgid)` -- the mere existence of a child
-- as liveness, which no hung run can fail.

The signals that separate the three states:

                CPU     log/syscalls/bytes    stack
    working     high    moving                moving
    deadlock    none    still                 still
    livelock    high    still                 still

so progress_signature reads the second and third columns and ignores the first.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def test_the_signature_excludes_cpu():
    """The whole point. A signature that moves with CPU cannot see a livelock."""
    import inspect

    from agent.probes import progress_signature

    src = inspect.getsource(progress_signature)
    body = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith(("#", '"')))
    assert "cpu" not in body.lower() or "jiffies" not in body.lower(), body


def test_it_reads_syscalls_and_bytes(monkeypatch):
    import agent.probes as P

    monkeypatch.setattr(P, "_pgroup_io_counters", lambda pgid: (100, 4096))
    monkeypatch.setattr(P, "_stack_fingerprint", lambda pid: "frame-a")
    assert P.progress_signature(1234, None, 99) == (0, 100, 4096, "frame-a")


def test_a_working_process_moves_the_signature():
    """Real work crosses into the kernel constantly -- this test process included."""
    import os
    import time

    from agent.probes import progress_signature

    g = os.getpgrp()
    a = progress_signature(g)
    for _ in range(2000):
        open("/proc/self/stat").close()
    time.sleep(0.2)
    b = progress_signature(g)
    assert a[:3] != b[:3], "a process doing real IO looks idle"


def test_a_livelock_does_not(monkeypatch):
    """The run-12 shape: counters frozen, stack frozen, CPU irrelevant."""
    import agent.probes as P

    monkeypatch.setattr(P, "_pgroup_io_counters", lambda pgid: (41580, 21520384))
    monkeypatch.setattr(P, "_stack_fingerprint", lambda pid: "from_torch (ttnn/operations/core.py:352)")
    a = P.progress_signature(1234, None, 99)
    b = P.progress_signature(1234, None, 99)
    assert a == b, "a livelock still looks like it is getting somewhere"


def test_both_supervisors_use_it_and_neither_uses_cpu():
    """One fixed loop and one left blind is the duplication that produced this in the first place."""
    probes = (_PA / "agent" / "probes.py").read_text()
    run = (_PA / "cc_optimize" / "run.py").read_text()

    for name, src, call in (("probes", probes, "progress_signature("), ("run.py", run, "_progress_signature(")):
        code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
        assert call in code, "%s does not use the shared signature" % name
        assert "cpu > last_cpu" not in code, "%s still treats CPU as progress" % name

    code = "\n".join(ln for ln in run.splitlines() if not ln.lstrip().startswith("#"))
    assert "_llm_child_alive(pgid) or" not in code, "a live child still counts as progress"


def test_cooling_is_still_exempt():
    """A cooling child is idle ON PURPOSE -- it sleeps against a thermometer. That exemption must
    survive, or the tool kills its own thermal wait."""
    run = (_PA / "cc_optimize" / "run.py").read_text()
    i = run.index("moved = _sig_moved")
    assert "_cooling_now()" in run[i : i + 200]


def test_there_is_a_ceiling_behind_the_detector():
    """The detector cannot catch a step that genuinely re-executes work forever -- fresh syscalls, a
    moving stack, no end. Nothing in this tool's own loops does that; they are bounded by counters
    (rounds, restarts, regens, kv attempts, engine max_steps). Model code can, and this supervises
    model code."""
    probes = (_PA / "agent" / "probes.py").read_text()
    assert "_HARD_CEILING_MULT" in probes
    # the GUARD, not the warning line that also names the constant
    i = probes.index("if timeout_s and now - start >= timeout_s * _HARD_CEILING_MULT:")
    assert "_kill_and_raise" in probes[i : i + 400], "the ceiling does not actually stop it"


def test_the_ceiling_fails_the_attempt_rather_than_the_run():
    """Why a ceiling is safe to have at all now. The 3-hour timer it replaces killed the RUN, so
    firing wrongly was catastrophic and it had to be set uselessly high. This raises the same error
    every other detection raises, which callers already treat as a failed attempt -- the perf-test
    loop regenerates, the supervisor restarts."""
    import inspect

    from agent.probes import TracyHangError, _execute

    src = inspect.getsource(_execute)
    i = src.index("if timeout_s and now - start >= timeout_s * _HARD_CEILING_MULT:")
    assert "_kill_and_raise" in src[i : i + 400]
    assert issubclass(TracyHangError, Exception)


def test_both_supervised_loops_have_the_ceiling_not_just_one():
    """RUN 12'S ACTUAL FAILURE. The ceiling went into probes._execute and NOT into
    run.py._run_device_proc -- and _run_device_proc is the loop that printed "over its 10800s budget
    ... not killing it" and then held the board silently for nine hours. A detector that covers one
    of two supervised loops is a detector with a hole in it."""
    run = Path(__file__).resolve().parents[1].joinpath("cc_optimize", "run.py").read_text()
    probes = Path(__file__).resolve().parents[1].joinpath("agent", "probes.py").read_text()

    assert "_HARD_CEILING_MULT" in probes, "probes lost its ceiling"
    assert "_hard_ceiling_mult" in run, "run.py's device loop has no hard ceiling"

    # and it must KILL, not merely narrate
    i = run.index("if _ceiling_mult and timeout_s and _worked >= timeout_s * _ceiling_mult:")
    assert "raise subprocess.TimeoutExpired" in run[i : i + 900], "run.py's ceiling does not stop anything"


def test_the_budget_message_does_not_still_claim_cpu_is_the_signal():
    """The line that let run 12 hang said "STILL WORKING (tree CPU is moving)". CPU is no longer
    consulted, so a message that still says so would send the next reader after the wrong signal."""
    run = Path(__file__).resolve().parents[1].joinpath("cc_optimize", "run.py").read_text()
    assert "STILL WORKING (tree CPU is moving)" not in run


def test_the_ceiling_is_measured_on_working_time_not_wall_clock():
    """Cooling is legitimate non-progress and is already excluded from the budget; the ceiling must
    exclude it too, or a long cooldown would spend the run's ceiling for it."""
    run = Path(__file__).resolve().parents[1].joinpath("cc_optimize", "run.py").read_text()
    i = run.index("_worked = now - start - _cool_total()")
    assert "_worked >= timeout_s * _ceiling_mult" in run[i : i + 1800], "ceiling ignores cooling credit"


def _code_only(path):
    """Source with comments and docstrings removed.

    These assertions are about what the code DOES. A docstring that quotes the old bad line -- and
    the fix's docstrings do quote it, deliberately, so the next reader knows what was wrong -- must
    not read as the bad line still being present."""
    import ast
    import io
    import tokenize

    src = Path(path).read_text()
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        out.append(tok)
    stripped = tokenize.untokenize(out)
    tree = ast.parse(stripped)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                docs.add(d)
    for d in docs:
        stripped = stripped.replace(d, "")
    return stripped


# ------------------------------------------------------------------ the third loop, and the LLM


def test_the_third_supervised_loop_is_not_still_on_cpu():
    """perf_mcp has its own Popen poll loop. It was missed on the first pass: it still read
    `cpu > last_cpu + 10` as progress. Its absolute backstop meant it could not hang outright, but
    the stall check was blind for the whole hour leading up to that backstop."""
    root = Path(__file__).resolve().parents[1]
    code = _code_only(root / "cc_optimize" / "perf_mcp.py")
    assert "cpu > last_cpu" not in code, "perf_mcp's loop still treats CPU as progress"
    assert "_pg_progress_signature" in code, "perf_mcp's loop has no progress signature"


def test_no_supervised_loop_anywhere_still_uses_cpu_as_liveness():
    """The whole point. Three loops, one rule -- and nothing left over that says otherwise."""
    root = Path(__file__).resolve().parents[1]
    for rel in (("agent", "probes.py"), ("cc_optimize", "run.py"), ("cc_optimize", "perf_mcp.py")):
        code = _code_only(root.joinpath(*rel))
        assert "cpu > last_cpu" not in code, "%s still reads CPU as progress" % (rel,)
    run = root.joinpath("cc_optimize", "run.py").read_text()
    assert "def _llm_child_alive" not in run, "the dead child-alive probe is still here"


def test_an_llm_verdict_cannot_re_arm_a_round_forever():
    """`wait` resets the round's progress clock, so an agent that keeps answering `wait` kept a
    stuck round alive with no bound -- the round-level twin of the budget that was demoted to a
    warning. A judgement is worth having; an unlimited number of them is not a bound."""
    run = Path(__file__).resolve().parents[1].joinpath("cc_optimize", "run.py").read_text()
    assert "_MAX_WATCHDOG_REPRIEVES" in run, "watchdog reprieves are unbounded"
    i = run.index('if _verdict == "wait":')
    stanza = run[i : i + 1200]
    assert "_reprieves[0] < _MAX_WATCHDOG_REPRIEVES" in stanza, "the reprieve is not counted"
    assert "break" in stanza, "running out of reprieves does not end the round"


def test_the_watchdogs_own_ceiling_is_not_reset_by_a_reprieve():
    """watchdog_decide already refuses to wait past an operator ceiling -- but it judged
    `since_commit`, which was computed from a clock the reprieve rewound. The bound was being
    cleared by the thing it existed to bound."""
    run = Path(__file__).resolve().parents[1].joinpath("cc_optimize", "run.py").read_text()
    assert '"since_commit": _now - last_real' not in run, "the stuck clock is still rewound by reprieves"
    assert '"since_commit": _now - (_stuck_since[0] or _now)' in run
    # and only REAL progress clears it
    j = run.index("_stuck_since[0] = None")
    assert "real progress" in run[max(0, j - 200) : j + 120]


def test_the_operator_ceiling_still_kills_a_waiting_agent():
    """Behavioural, not textual: an agent that says wait forever must still be overruled."""
    from cc_optimize.run import watchdog_decide

    ev = {
        "op": "round",
        "op_elapsed": 10**6,
        "since_commit": 10**6,
        "cpu_hist": [1],
        "txt_hist": [1],
        "actions": 5,
        "distinct_actions": 1,
        "action_seq": [],
        "log_tail": "",
        "observed": {},
        "ceiling": 100,
    }
    assert watchdog_decide(ev, agent=lambda _e: "wait") == "kill", "a confused agent can still wait forever"
