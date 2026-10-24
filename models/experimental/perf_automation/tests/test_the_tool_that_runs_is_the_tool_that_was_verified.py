# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The copy that RUNS is not the copy that was edited, and nothing checked the difference.

The tool is developed in one checkout and synced into the repo the run executes from. Three
distinct failures came out of that gap, each costing a run:

  * an edit that applied to the WRONG PLACE -- DEFAULT_ISL_TOKENS landed inside a template STRING
    instead of at module scope. The module imported cleanly and the symbol did not exist;
  * a module reachable by package name but not by PATH -- the report loader uses
    spec_from_file_location, which gives no package context and no sys.path entry for the module's
    own directory, so both relative and absolute imports raise. The report rendered with three
    blank sections and every failure was silent;
  * a `git stash` during a debugging detour that never popped, so two committed fixes were absent
    from the tree that then ran.

Each was found hours in, from an unrelated symptom, and each would have been caught by running the
tool's own suite against the tree about to be used -- ~90 seconds against a run measured in hours.

The rule this encodes is the one that kept being broken by hand: a preflight that could not RUN has
not cleared anything. A timeout, a missing suite, a crashed pytest -- none of those are a pass.

  r1  a red suite stops the run
  r2  a preflight that could not run is UNKNOWN, not OK
  r3  the skip is explicit and says so
  r4  it runs against the RUN'S repo, not the developer's
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))
sys.path.insert(0, str(_PA.parent.parent.parent))


def _run():
    spec = importlib.util.spec_from_file_location("cc_run_preflight", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["cc_run_preflight"] = m
    spec.loader.exec_module(m)
    return m


def _repo(tmp_path):
    (tmp_path / "models/experimental/perf_automation/tests").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _result(rc, stdout=""):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


# --------------------------------------------------------------------------- r1
def test_r1_a_red_suite_stops_the_run(monkeypatch, tmp_path, capsys):
    m = _run()
    monkeypatch.delenv("PERF_MCP_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _result(1, "FAILED tests/test_x.py::test_y\n1 failed\n"))
    assert m._preflight_tool(_repo(tmp_path)) is False
    out = capsys.readouterr().out
    assert "preflight FAILED" in out and "test_x.py::test_y" in out


def test_r1_a_green_suite_proceeds(monkeypatch, tmp_path, capsys):
    m = _run()
    monkeypatch.delenv("PERF_MCP_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _result(0, "2504 passed, 5 skipped in 87s\n"))
    assert m._preflight_tool(_repo(tmp_path)) is True
    assert "preflight OK" in capsys.readouterr().out


# --------------------------------------------------------------------------- r2
def test_r2_a_preflight_that_could_not_run_is_not_a_pass(monkeypatch, tmp_path, capsys):
    """A guard initialised to the passing value and wrapped in `except` is defect shape 2 in
    agent/integrity.py. It proceeds by default -- a broken preflight must not brick every run --
    but it SAYS it is unknown, and PERF_MCP_REQUIRE_PREFLIGHT=1 makes it a stop."""
    m = _run()
    monkeypatch.delenv("PERF_MCP_SKIP_PREFLIGHT", raising=False)

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=900)

    monkeypatch.setattr(m.subprocess, "run", _boom)
    assert m._preflight_tool(_repo(tmp_path)) is True
    assert "treating as UNKNOWN, not as passed" in capsys.readouterr().out

    monkeypatch.setenv("PERF_MCP_REQUIRE_PREFLIGHT", "1")
    assert m._preflight_tool(_repo(tmp_path)) is False


def test_r2_a_missing_suite_is_unknown_too(monkeypatch, tmp_path, capsys):
    m = _run()
    monkeypatch.delenv("PERF_MCP_SKIP_PREFLIGHT", raising=False)
    monkeypatch.delenv("PERF_MCP_REQUIRE_PREFLIGHT", raising=False)
    assert m._preflight_tool(tmp_path) is True  # no tests dir at all
    assert "cannot verify the tool" in capsys.readouterr().out


# --------------------------------------------------------------------------- r3
def test_r3_the_skip_is_explicit_and_announced(monkeypatch, tmp_path, capsys):
    """A silent skip is how the check stops existing."""
    m = _run()
    monkeypatch.setenv("PERF_MCP_SKIP_PREFLIGHT", "1")
    called = []
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: called.append(1) or _result(0))
    assert m._preflight_tool(_repo(tmp_path)) is True
    assert not called, "the suite ran despite the skip"
    assert "preflight SKIPPED" in capsys.readouterr().out


# --------------------------------------------------------------------------- r4
def test_r4_it_tests_the_repo_the_run_will_use(monkeypatch, tmp_path):
    """Against the DEVELOPER's checkout it proves nothing: the sync is the step that fails."""
    m = _run()
    monkeypatch.delenv("PERF_MCP_SKIP_PREFLIGHT", raising=False)
    seen = {}

    def _cap(cmd, **k):
        seen["cmd"], seen["cwd"] = cmd, k.get("cwd")
        return _result(0, "ok")

    monkeypatch.setattr(m.subprocess, "run", _cap)
    repo = _repo(tmp_path)
    m._preflight_tool(repo)
    assert seen["cwd"] == str(repo)
    assert str(repo / "models/experimental/perf_automation/tests") in seen["cmd"]


# --------------------------------------------------------------------------- r6 CODE, NOT STATE
def test_r6_the_suite_does_not_inherit_this_runs_configuration(monkeypatch, tmp_path):
    """A check meant to PROTECT the run must not be able to damage it.

    _KERNEL_LOG_PATH is resolved AT IMPORT from PERF_MCP_KERNEL_LOG. With the run's value inherited,
    a suite that calls record_kernel_attempt writes into the LIVE ladder -- the state the run
    resumes from. It happened not to fire only because the tests that write were the ones failing.

    It is also the difference between a test of the CODE and a test of this run's state:
    test_all_boards_is_the_last_resort read the ambient PERF_MCP_DEVICES and never reached the
    fallback it exists to check."""
    m = _run()
    monkeypatch.delenv("PERF_MCP_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", "/live/ladder.json")
    monkeypatch.setenv("PERF_MCP_STATE_DIR", "/live/state")
    monkeypatch.setenv("TT_PERF_LAYERS", "2")
    monkeypatch.setenv("TT_METAL_HOME", "/repo")
    seen = {}

    def _cap(cmd, **k):
        seen["env"] = k.get("env") or {}
        return _result(0, "ok")

    monkeypatch.setattr(m.subprocess, "run", _cap)
    m._preflight_tool(_repo(tmp_path))
    for gone in ("PERF_MCP_KERNEL_LOG", "PERF_MCP_STATE_DIR", "TT_PERF_LAYERS"):
        assert gone not in seen["env"], "%s reached the suite" % gone
    assert seen["env"].get("TT_METAL_HOME") == "/repo", "stripped a variable that locates the CODE"


def test_r6_stripping_is_by_prefix_not_by_list():
    """A list's failure mode is the next variable someone adds not being on it."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _preflight_tool")
    body = src[i : src.index("\ndef ", i + 1)]
    assert 'startswith(("PERF_MCP_", "TT_PERF_"))' in body


# --------------------------------------------------------------------------- r5 A REFUSAL IS NOT A CRASH
def test_r5_a_refusal_exits_with_its_own_code():
    """The auto-restart supervisor exists for a native tt-metal SIGSEGV, and read ANY non-zero exit
    as that case. The first real preflight refusal was therefore reported as "likely native crash /
    device wedge", the board was reset, and the same decision was re-derived from the same evidence
    -- three times, ten minutes, for a verdict available at once."""
    m = _run()
    assert m.EXIT_REFUSED != 0 and m.EXIT_REFUSED != 1, m.EXIT_REFUSED


def test_r5_a_refusal_is_retried_but_bounded():
    """REVERSED 2026-08-18, and the reasoning that justified the old rule no longer applies.

    It used to return immediately on EXIT_REFUSED: "relaunching re-derives the same decision from
    the same evidence". True for a refusal grounded in something fixed -- a red preflight, a dirty
    tree -- and false for the one that actually fires. The refusal that occurs in practice is the
    lead review rejecting a discovery plan an AGENT wrote, and the next attempt writes a different
    plan. The verdict is not re-derived; it is re-earned.

    The harm that made this non-retryable was never the retry. It was the restart leaving the
    previous attempt's process tree alive, so two runs loaded the model onto one board and wedged it
    past what tt-smi -r could restart. That is fixed at its source: the supervisor reaps the tree and
    refuses to start again if anything survives SIGKILL.

    Still bounded by the same restart limit, so a refusal grounded in something fixed costs a few
    attempts and stops rather than looping.
    """
    sup = (_PA.parent.parent.parent / "scripts/tt_hw_planner/commands/optimize.py").read_text()
    i = sup.index("if _rc == _EXIT_REFUSED:")
    body = sup[i : i + 2000]
    assert "retrying" in body, "a refusal is not retried"
    assert "if _n >= _max:" in body, "a refusal retries without a bound"
    assert "return _rc" in body, "an exhausted refusal never terminates"
    # the tree reaping is what makes a retry safe; it must still be there
    assert "survived SIGKILL" in sup, "the retry is safe only while a leaked tree stops the next attempt"


def test_r5_the_exit_code_has_one_definition():
    """A second literal in the supervisor that drifted from run.py's would turn every refusal back
    into three device resets, silently."""
    m = _run()
    sup = (_PA.parent.parent.parent / "scripts/tt_hw_planner/commands/optimize.py").read_text()
    assert "from models.experimental.perf_automation.cc_optimize.run import EXIT_REFUSED" in sup
    assert "_EXIT_REFUSED = %d" % m.EXIT_REFUSED in sup, "the supervisor fallback disagrees with run.py"


def test_r5_both_deliberate_refusals_use_it():
    """The dirty-tree refusal is the same kind of decision and was the same rc=1."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    assert src.count("raise SystemExit(EXIT_REFUSED)") >= 2, "a deliberate refusal still exits 1"


def test_r4_the_run_calls_it_before_touching_the_device():
    """Ordered ahead of discovery, which spends an agent call, and ahead of any device work."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    body = src[src.index("def run_cc_optimize") :]
    _end = body.find("\ndef ", 1)
    body = body if _end < 0 else body[:_end]
    assert body.index("_preflight_tool") < body.index("discover("), "preflight runs after discovery"


def test_r5_a_rejected_discovery_is_a_refusal_not_a_crash():
    """THE COST OF GETTING THIS WRONG, observed on a real Voxtral run.

    The lead agent rejected the discovery plan (the correctness gate covered a strict subset of the
    perf surface). before_loop returned 1 for it, like any exception. The supervisor read 1 as a
    likely native crash and restarted the child -- so a second optimize came up carrying the very
    gate that had just been rejected, and raced the corrected run for the same board. Both were
    loading the model onto the device at once; the profile came back with no ops_perf_results CSV
    and the board wedged hard enough that `tt-smi -r` could not restart the ARC core.

    A verdict must terminate. Both halves are asserted: the code before_loop returns, and run.py
    refusing to launder that code through its complete-manifest fallback.
    """
    bl = (_PA / "agent" / "before_loop.py").read_text()
    # ANCHORED ON THE BRANCH, not a 900-character window from the log line. The window stopped
    # covering the return as soon as the handler grew, failing a test whose subject had not changed
    # -- the fourth character-window assertion in this suite to break that way.
    i = bl.index("discovery failed (")
    tail = bl[i : bl.index("\n    p = result[", i)]
    assert "DiscoveryRejected" in tail, "a rejected discovery still exits like a crash"
    assert "EXIT_REFUSED" in tail

    run_src = (_PA / "cc_optimize" / "run.py").read_text()
    j = run_src.index("rc == EXIT_REFUSED")
    assert "raise SystemExit(EXIT_REFUSED)" in run_src[j : j + 700], "a refused discovery is not propagated"
    # and it must be decided BEFORE the fallback that continues on a complete manifest
    assert j < run_src.index("but the manifest is complete"), "the refusal is checked after the override"


def test_the_refusal_path_can_actually_return_the_refusal_code():
    """RUN 9, 2026-08-17: it could not, and the failure was invisible until a discovery was rejected.

        File ".../agent/before_loop.py", line 1239, in main
            from ..cc_optimize.run import EXIT_REFUSED
        ImportError: attempted relative import beyond top-level package

    before_loop runs as `python -m ...agent.before_loop`, so its package is `agent` and `..` walks
    off the top. The import sat INSIDE the handler whose whole job is to return EXIT_REFUSED, so a
    refused discovery raised on its way to reporting itself, exited rc=1, and the supervisor read
    that as a crash and restarted it -- "racing the corrected run for the same board until both
    wedged it", which is what the comment two lines above the import warns against. The line meant
    to prevent that outcome produced it.

    Asserted on the source rather than by importing, because the failure is a property of HOW the
    module is loaded: imported normally by a test, the relative form resolves and the bug hides.
    """
    src = (_PA / "agent" / "before_loop.py").read_text()
    i = src.index("isinstance(exc, DiscoveryRejected)")
    body = src[i : i + 1500]
    assert "from ..cc_optimize" not in body, "the refusal path imports beyond its top-level package again"
    assert "EXIT_REFUSED" in body, "the refusal no longer reports a refusal"


def test_the_refusal_code_agrees_across_the_two_places_that_report_it():
    """before_loop RETURNS it and the supervisor READS it. Two literals that drifted would turn
    every refusal back into three device resets, silently."""
    m = _run()
    bl = (_PA / "agent" / "before_loop.py").read_text()
    i = bl.index("isinstance(exc, DiscoveryRejected)")
    assert "EXIT_REFUSED = %d" % m.EXIT_REFUSED in bl[i : i + 1500], "before_loop's fallback disagrees with run.py"


# --------------------------------------------------------------------------- the review verdict


def test_the_verdict_survives_a_newline_in_its_reasoning():
    """RUN 9, 2026-08-17: "Invalid control character at: line 1 column 1029 (char 1028)".

    The prompt asked for {"decision": ..., "reasoning": <2-3 sentences>} -- free prose inside a JSON
    string, where a literal newline is illegal. The reviewer is a language model writing that prose,
    so a long enough answer eventually wraps a line. The verdict was sound and was thrown away, the
    run was refused for it, and the retry passed only because the next answer happened to fit on one
    line. The prompt has asked for that shape since 2026-06-27; it needed a big discovery to break.
    """
    from agent.probes import parse_review_verdict

    exact = '{"decision": "continue", "reasoning": "the gate covers prefill\nand decode"}'
    assert parse_review_verdict(exact)[0] == "continue"


def test_the_token_form_cannot_break_on_a_line_break():
    """The fix is the FORMAT, not a more forgiving parser: with the decision on its own line the
    reasoning is no longer inside a quoted string, so its shape cannot invalidate the answer."""
    from agent.probes import parse_review_verdict

    d, why = parse_review_verdict("DECISION: stop\nREASON: the gate covers only\nprefill, and decode\nis unmeasured")
    assert d == "stop" and why.startswith("the gate covers only")


def test_both_shapes_are_accepted():
    """A format instruction is a request, not a guarantee -- and an older harness may face a newer
    prompt, or the reverse. Neither shape is required; both are read."""
    from agent.probes import parse_review_verdict

    for text, want in (
        ("DECISION: continue\nREASON: fine", "continue"),
        ('{"decision": "stop", "reasoning": "no"}', "stop"),
        ("Let me review.\n\nDECISION: continue\nREASON: fine", "continue"),
        ("```\nDECISION: stop\nREASON: bad gate\n```", "stop"),
        ("decision: continue\nreason: ok", "continue"),
        ('Here is my verdict:\n{"decision": "stop", "reasoning": "no"}\nThanks!', "stop"),
    ):
        assert parse_review_verdict(text)[0] == want, text[:40]


def test_a_fence_is_not_reasoning():
    from agent.probes import parse_review_verdict

    assert parse_review_verdict("```\nDECISION: stop\nREASON: bad gate\n```")[1] == "bad gate"


def test_the_word_stop_in_prose_is_not_a_decision():
    """Substring matching on free text would let the reasoning veto the decision."""
    from agent.probes import parse_review_verdict

    assert parse_review_verdict("DECISION: continue\nREASON: I nearly said stop, but no")[0] == "continue"


def test_no_decision_at_all_is_none_and_the_gate_asks_again():
    """The parser reports None; the GATE turns that into a refusal, which regenerates discovery and
    asks again rather than proceeding.

    Continuing would run a plan nobody approved -- the gate bypassable by any reply that failed to
    state a decision. Refusing used to kill the run, which is what made proceeding look safer; a
    refusal now costs one more attempt instead. A reviewer that is systematically unreadable
    exhausts the retries and stops the run, which is correct: never getting a verdict is not
    approval."""
    from agent.probes import parse_review_verdict

    assert parse_review_verdict("I could not evaluate this.")[0] is None
    assert parse_review_verdict("")[0] is None

    body = _gate_source()
    assert "raise DiscoveryRejected" in body, "an unreadable verdict proceeds again"
    assert '"decision": "continue"' not in body, "the gate still manufactures approval"


def _gate_source() -> str:
    """The whole cli_lead_review_gate body -- anchored to the function, not to a character window,
    because every window this file has used has broken on an unrelated edit above it."""
    src = (_PA / "agent" / "probes.py").read_text()
    i = src.index("def cli_lead_review_gate(")
    j = src.index("\ndef ", i + 1)
    return src[i:j]


class _FakeCompleted:
    def __init__(self, stdout: str) -> None:
        self.returncode, self.stdout, self.stderr = 0, stdout, ""


def _gate_over(answers, monkeypatch):
    """Run the real gate against a scripted reviewer; return (verdict-or-exception, prompts sent)."""
    from agent import probes

    asked: list[str] = []

    def _fake_run(cmd, **_kw):
        asked.append(cmd[cmd.index("-p") + 1])
        return _FakeCompleted(answers[min(len(asked) - 1, len(answers) - 1)])

    monkeypatch.setattr(probes.subprocess, "run", _fake_run)
    monkeypatch.setattr("agent.agent_bin.resolve_claude_bin", lambda: "claude")
    pathmap = {k: "x" for k in ("perf_test", "pcc", "components", "summary", "warnings")}
    try:
        return probes.cli_lead_review_gate(pathmap), asked
    except probes.DiscoveryRejected as exc:
        return exc, asked


def test_an_unreadable_answer_is_asked_again_in_place_not_after_a_rebuild(monkeypatch):
    """The retry used to live ONLY at the run level: an unusable reply threw away discovery and cost
    fifteen to twenty minutes of regeneration before anyone asked a second time. Run 9 paid exactly
    that for a line break. The reviewer is asked again here, in seconds, with nothing regenerated."""
    got, asked = _gate_over(["mumble", "no idea", "DECISION: continue\nREASON: sound"], monkeypatch)

    assert getattr(got, "get", lambda _k: None)("decision") == "continue", got
    assert len(asked) == 3, "the gate did not re-ask in place"


def test_the_retry_tells_the_reviewer_what_was_wrong_with_its_answer(monkeypatch):
    """Repeating an identical prompt to a model that just misread it mostly buys an identical
    answer. The second ask names the defect and quotes the reply back, so it is a correction."""
    _got, asked = _gate_over(["DECISION: continue|stop", "DECISION: stop\nREASON: no gate"], monkeypatch)

    assert len(asked) == 2
    retry = asked[1]
    assert "quoted the FORMAT back" in retry, "the retry did not say what was wrong"
    assert "DECISION: continue|stop" in retry.split("PREVIOUS ANSWER")[1], "it did not quote the reply back"
    assert asked[0] in retry, "the retry dropped the findings"


def test_a_reviewer_that_never_states_a_decision_runs_out_and_refuses(monkeypatch):
    """The bound is real. A systematically unreadable reviewer exhausts these attempts, then the
    supervisor's, and stops the run -- never getting a verdict is not approval."""
    got, asked = _gate_over(["I cannot tell."] * 9, monkeypatch)

    from agent.probes import DiscoveryRejected

    assert isinstance(got, DiscoveryRejected), got
    assert len(asked) == 5, "the in-place retry is unbounded or missing"
    assert "no `DECISION:` line" in str(got), "the refusal does not say what was wrong"


def test_the_complaint_can_only_name_defects_the_parser_agrees_are_defects(monkeypatch):
    """A retry that scolds the reviewer for something the parser would have ACCEPTED is worse than no
    retry: it teaches the model to change an answer that was already fine. The complaint and the
    parse read the same regexes, so anything that parses never reaches the complaint."""
    from agent.probes import parse_review_verdict, review_verdict_complaint

    for reply, expected in [
        ("", "EMPTY"),
        ("DECISION: continue|stop\nREASON: x", "quoted the FORMAT back"),
        ("DECISION: continue\nOn reflection:\nDECISION: stop", "BOTH decisions"),
        ("I would continue here.", "never put it on a `DECISION:` line"),
        ("Hello.", "no `DECISION:` line"),
    ]:
        assert parse_review_verdict(reply)[0] is None, f"the parser accepts {reply!r}; do not complain"
        assert expected in review_verdict_complaint(reply), reply


def test_the_prompt_asks_for_the_shape_that_cannot_break():
    from agent.probes import REVIEW_PROMPT

    assert "DECISION:" in REVIEW_PROMPT and "REASON:" in REVIEW_PROMPT
    assert "ONLY a JSON object" not in REVIEW_PROMPT, "the prompt asks for prose inside JSON again"


def test_the_format_spec_quoted_back_is_not_a_verdict():
    """`DECISION: continue|stop` is the INSTRUCTION, and a naive match reads its first alternative as
    an answer -- handing an automatic continue to any model that restates the format before
    answering. Found by adversarial testing, not by a run; it would have been a silent pass."""
    from agent.probes import parse_review_verdict

    assert parse_review_verdict("DECISION: continue|stop\nDECISION: stop\nREASON: bad gate")[0] == "stop"
    assert parse_review_verdict("I should answer DECISION: continue|stop\n\nDECISION: stop\nREASON: x")[0] == "stop"


def test_decoration_does_not_hide_a_decision():
    """Models bold and bullet things. `**DECISION:** stop` returning nothing loses a real refusal."""
    from agent.probes import parse_review_verdict

    for text in ("**DECISION:** stop\nREASON: bad", "- **DECISION**: stop\n- REASON: bad", "DECISION stop\nREASON bad"):
        assert parse_review_verdict(text)[0] == "stop", text


def test_two_different_answers_are_no_answer():
    """A reply that says continue and later stop has not decided. Picking the first -- or the last --
    invents a verdict out of ordering, on the one question the gate exists to answer."""
    from agent.probes import parse_review_verdict

    assert parse_review_verdict("DECISION: continue\nREASON: ok\nActually\nDECISION: stop")[0] is None
    assert parse_review_verdict("DECISION: stop\nREASON: bad\nDECISION: stop")[0] == "stop"
