# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A run killed by the clock is not a run killed by a wedge, and must not be treated as one.

THE CASE. Qwen-Image-Edit, 2026-10-01: `optimize` spent all three perf-test regenerations and aborted
discovery without ever reaching the optimisation loop. Each attempt was reported as

    · perf-test regen N/3: device wedged on a non-capturable step — reset + regenerating

and nothing had wedged. The three attempts lived 3645s, <=3681s and 3626s against a ceiling of
3600s (`PERF_MCP_VALIDATE_TIMEOUT` 900 x probes._HARD_CEILING_MULT 4), and the last line each one
wrote was `TRACE_STAGE_WAITING[replay x16] 750s on device` -- still emitting device output when it
died, which is the definition of the ceiling branch rather than the stall branch. All three measured
the same three stages to the decimal (68109.7 / 12728.7 / 700.1 ms) and died in the fourth, `denoise`,
because that is where the cumulative cost crosses the ceiling:

    stage           per step   (5 + 16) x cost   cumulative
    vision_encode     68.1s            1430s        1680s
    text_encode       12.7s             267s        1947s
    vae_encode         0.7s              15s        1962s
    denoise          119.9s            2518s        4479s   <- killed here
    vae_decode        ~50.0s           1050s        5529s

5529s of work, 3600s allowed. THREE SEPARATE DEFECTS kept that invisible and unfixable:

  1. The child was never told the budget. trace_replay already sizes its replay count to one
     (_measurement_budget_s -> _affordable_iters, fair-shared across the stages), and its own comment
     says "the caller ... puts its budget in the environment and the capture child inherits it" --
     but the validation path put neither name of _REPLAY_BUDGET_ENVS in the child env, so the budget
     read as 0, sizing was skipped, and every stage replayed the full 16 regardless of cost.
  2. The sizer was handed a share as if the unavoidable work were free. Each stage runs its step in
     full five times before a replay is timed (1 for _count_op_dispatches, _WARMUP_ITERS for _warm,
     1 inside the capture) and _affordable_iters only sizes replays -- 599s a stage on denoise.
  3. A clock kill was reported as a hang. `rc == 124` returned "device hung capturing the module's
     forward" whatever caused it, and the regen loop printed a FIXED summary that dropped the reason,
     so the only copy of it lived in the attempt's perf-node log, which is deleted when the node
     rotates. Three attempts' evidence was thrown away, and the generator was asked three times to
     rewrite a test that had measured three stages correctly.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import sys
import textwrap
import types

import pytest

G = importlib.import_module("models.experimental.perf_automation.agent.perf_test_gen")
P = importlib.import_module("models.experimental.perf_automation.agent.probes")


@pytest.fixture
def TR(monkeypatch):
    """trace_replay imports ttnn at module scope, so it is loaded against a stub.

    THE STUB MUST NOT OUTLIVE THE TEST. Installing it with sys.modules.setdefault shadowed the real
    ttnn for every test that ran afterwards in the same session -- a nemotron pipeline test then died
    on "No module named 'ttnn.device'; 'ttnn' is not a package", and a second test passed or failed
    depending on collection order. monkeypatch.setitem removes it again, which is the idiom
    test_a_measurement_fits_the_budget_it_was_given.py already uses for this same import.
    """
    monkeypatch.setitem(sys.modules, "ttnn", types.SimpleNamespace())
    monkeypatch.delitem(sys.modules, "models.experimental.perf_automation.agent.trace_replay", raising=False)
    return importlib.import_module("models.experimental.perf_automation.agent.trace_replay")


@pytest.fixture(autouse=True)
def _clean_env():
    keep = {k: os.environ.get(k) for k in (G._VALIDATE_TIMEOUT_ENV, "TT_PERF_REPLAY_BUDGET_S")}
    for k in keep:
        os.environ.pop(k, None)
    yield
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---------------------------------------------------------------------------------------------
# 3. the reason survives: which watchdog fired is answerable
# ---------------------------------------------------------------------------------------------


def _stall_text():
    return "label %s for 600s -- no log growth, no syscalls, no bytes and an unchanged stack" % P.KILL_STALL


def _ceiling_text():
    return "label %s: 3626s of a 3600s ceiling (4x its 900s budget). It was still moving" % P.KILL_CEILING


def test_the_two_kills_are_told_apart():
    assert P.kill_kind(_stall_text()) == P.KILL_STALL
    assert P.kill_kind(_ceiling_text()) == P.KILL_CEILING
    assert P.kill_kind("some unrelated failure") == ""
    assert P.kill_kind(None) == ""
    assert P.kill_kind("") == ""


def test_the_raise_sites_use_the_same_constants_the_classifier_reads():
    """The wording and the classification cannot drift: _execute writes these constants, kill_kind
    reads them. A second copy of the phrase is how a reworded reason silently stops matching."""
    src = inspect.getsource(P._execute)
    assert "KILL_STALL" in src, "the stall kill no longer writes the constant kill_kind matches on"
    assert "KILL_CEILING" in src, "the ceiling kill no longer writes the constant kill_kind matches on"


def test_a_ceiling_kill_is_not_reported_as_a_hang():
    detail = G._wedge_detail(_ceiling_text())
    assert "BUDGET" in detail
    assert "NOT a hang" in detail
    assert "hung" not in detail.replace("NOT a hang", ""), detail


def test_a_real_hang_is_still_reported_as_one():
    detail = G._wedge_detail(_stall_text())
    assert "WEDGE" in detail and "no forward progress" in detail
    assert "BUDGET" not in detail


def test_an_unclassifiable_kill_says_so_instead_of_inventing_a_cause():
    detail = G._wedge_detail("killed, nothing recorded")
    assert "WEDGE" in detail
    assert "hung capturing" not in detail, "still asserting a cause the output does not support"


# ---------------------------------------------------------------------------------------------
# 1 + 2. the budget reaches the child, and the sizer reserves what it cannot cut
# ---------------------------------------------------------------------------------------------


def test_the_child_is_told_the_budget_it_will_be_killed_for_exceeding():
    """Sizing is dormant unless the env carries a budget; the parent builds that env."""
    src = inspect.getsource(G._run_perf_node)
    assert "TT_PERF_REPLAY_BUDGET_S" in src, "the child still receives no budget, so sizing stays off"
    assert "_HARD_CEILING_MULT" in src, "sized against the soft budget, not the ceiling that kills it"
    assert "setdefault" in src, "an operator's own budget must win"


def test_the_budget_name_the_parent_sets_is_one_the_child_reads(TR):
    assert "TT_PERF_REPLAY_BUDGET_S" in TR._REPLAY_BUDGET_ENVS
    os.environ["TT_PERF_REPLAY_BUDGET_S"] = "3600"
    assert TR._measurement_budget_s(1) == 3600.0
    assert TR._measurement_budget_s(5) == 720.0  # fair-shared across the stages about to be measured


def test_the_fixed_pre_replay_calls_are_derived_not_typed(TR):
    """1 for _count_op_dispatches + _WARMUP_ITERS for _warm + 1 inside the capture."""
    assert TR._FIXED_STEP_CALLS == 1 + TR._WARMUP_ITERS + 1


def test_a_share_reserves_the_calls_the_sizer_cannot_cut(TR):
    assert TR._replay_share(720.0, 119.9) == pytest.approx(720.0 - 5 * 119.9)
    assert TR._replay_share(300.0, 119.9) == 0.0  # fixed cost alone exceeds the share
    assert TR._replay_share(720.0, 0.7) == pytest.approx(720.0 - 5 * 0.7)


def test_an_unstated_budget_stays_unstated(TR):
    """0 means "nothing stated" to _affordable_iters, which then honours the request in full. A share
    that turned 0 into a small positive number would silently cut every model's sample count."""
    assert TR._replay_share(0, 119.9) == 0.0
    assert TR._replay_share(None, 119.9) == 0.0
    assert TR._affordable_iters(16, 119.9, TR._replay_share(0, 119.9)) == 16
    assert TR._replay_share(720.0, 0) == 720.0  # no cost observed -> nothing to reserve


def test_this_models_measured_run_now_fits_its_ceiling(TR):
    """The arithmetic the case history records, end to end: 5529s before, under 3600s after."""
    per_iter = {"vision_encode": 68.1, "text_encode": 12.7, "vae_encode": 0.7, "denoise": 119.9, "vae_decode": 50.0}
    ceiling, build = 3600.0, 250.0
    before = build + sum((TR._FIXED_STEP_CALLS + 16) * c for c in per_iter.values())
    assert before > ceiling, "the case no longer reproduces; this test is asserting nothing"

    os.environ["TT_PERF_REPLAY_BUDGET_S"] = str(int(ceiling))
    share = TR._measurement_budget_s(len(per_iter))
    after = build
    for cost in per_iter.values():
        n = TR._affordable_iters(16, cost, TR._replay_share(share, cost))
        assert n >= TR._MIN_REPLAY_ITERS, "an average needs two samples"
        assert n <= 16, "never more than was asked for"
        after += (TR._FIXED_STEP_CALLS + n) * cost
    assert after < ceiling, "still overruns: %.0fs of %.0fs" % (after, ceiling)


def test_a_cheap_stage_is_measured_exactly_as_before(TR):
    """The sizing must only bite where a stage is expensive; a millisecond step keeps all 16."""
    os.environ["TT_PERF_REPLAY_BUDGET_S"] = "3600"
    share = TR._measurement_budget_s(5)
    assert TR._affordable_iters(16, 0.003, TR._replay_share(share, 0.003)) == 16


# ---------------------------------------------------------------------------------------------
# the escape hatch: when even the floor does not fit, the budget grows -- but bounded
# ---------------------------------------------------------------------------------------------


def test_the_budget_has_one_reader_so_a_raise_is_visible():
    assert G._validate_timeout_s() == G._VALIDATE_TIMEOUT_DEFAULT
    os.environ[G._VALIDATE_TIMEOUT_ENV] = "1234"
    assert G._validate_timeout_s() == 1234
    os.environ[G._VALIDATE_TIMEOUT_ENV] = "not-a-number"
    assert G._validate_timeout_s() == G._VALIDATE_TIMEOUT_DEFAULT


def test_the_validation_timeout_is_read_through_that_one_reader():
    src = inspect.getsource(G.validate_generated_perf_test)
    assert "_validate_timeout_s()" in src
    assert 'os.environ.get("PERF_MCP_VALIDATE_TIMEOUT"' not in src, "a second reader would not see a raise"


def test_a_raise_grows_the_budget_and_then_stops_at_the_cap():
    was, now = G._raise_validate_budget()
    assert now == was * G._BUDGET_RAISE_MULT, (was, now)
    assert G._validate_timeout_s() == now, "the raise must be visible to the next attempt"
    for _ in range(12):
        G._raise_validate_budget()
    assert G._validate_timeout_s() <= G._BUDGET_RAISE_CAP_S, "an unbounded raise deletes the backstop"


def test_the_raise_reuses_the_shared_measurement_to_budget_arithmetic():
    """probes.sized_budget owns "a measurement becomes a budget"; this must not re-derive it."""
    src = inspect.getsource(G._raise_validate_budget)
    assert "sized_budget" in src
    assert "*" not in src.split("sized_budget")[0].split("\n")[-1], "arithmetic inlined instead of reused"


def test_a_clock_kill_retries_the_same_draft_without_spending_a_regeneration():
    """The draft that ran out of time is not wrong, so regenerating it cannot help -- and spending a
    regeneration on it is how three attempts were lost. The raise must come BEFORE `stall += 1`."""
    src = textwrap.dedent(inspect.getsource(G.generate_perf_test))
    body = ast.unparse(ast.parse(src).body[0])
    i_raise = body.find("_raise_validate_budget")
    i_stall = body.find("stall += 1", body.find("_kill_kind"))
    assert i_raise != -1, "no budget raise in the loop"
    assert i_raise < i_stall, "the raise happens after the regeneration is already counted"
    assert "_BUDGET_RAISE_LIMIT" in body, "the raise is unbounded"


# ---------------------------------------------------------------------------------------------
# the constraints this change is held to
# ---------------------------------------------------------------------------------------------


def test_no_second_copy_of_the_kill_phrases():
    """One owner for each phrase: probes writes it, probes classifies it, everyone else asks."""
    for mod in (G, TR):
        src = inspect.getsource(mod)
        assert "made no forward progress" not in src, "%s retypes the stall phrase" % mod.__name__
        assert "exceeded its budget" not in src, "%s retypes the ceiling phrase" % mod.__name__


def test_it_names_no_model_component_or_stage(TR):
    """Every number here is measured or derived; nothing branches on what the model is called."""
    for fn in (G._wedge_detail, G._raise_validate_budget, G._validate_timeout_s, TR._replay_share):
        code = "\n".join(ln for ln in inspect.getsource(fn).splitlines() if not ln.strip().startswith("#"))
        code = "".join(code.split('"""')[::2]).lower()  # drop docstrings; case history names it on purpose
        for name in ("qwen", "image_edit", "denoise", "vae", "text_encode", "vision", "llama", "voxtral"):
            assert name not in code, "%r is hardcoded in %s" % (name, fn.__name__)
