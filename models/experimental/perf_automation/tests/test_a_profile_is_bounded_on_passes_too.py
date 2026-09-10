"""A profile has to be bounded on how many PASSES it captures, not only how deep each one builds.

Depth bounds how much of the model is constructed. It does not bound how many times the built graph
is DISPATCHED -- and tracy's ceiling is on dispatches: it names every program `Program_<N>` and
refuses past 32K "static or dynamic source locations". A recurring stage replays its whole graph
once per token, so on a model with one the token window, not the layer window, decides whether a
capture survives.

Measured on voxtral_mini_3b_2507 (2026-09-09), layers already capped to 2 in BOTH runs:

    128 passes -> 26,268 distinct program names -> "Instrumentation failure", 26.0M rows
      2 passes ->  2,514,086 zones             -> clean, 2.48M rows

Capping depth alone removed 6% of the capture. Capping the window removed 90%. Every capture from
2026-09-03 to 2026-09-09 was truncated for want of this bound -- 163 of 254 -- and everything
derived from the whole capture went with them.

The window rides the same channel as the depth cap and is named beside it, which matters twice
over: the profiling server needs it, and the FULL-depth gate needs to take it back off. A window
left on measures a two-pass decode and reports it as the whole request.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

_PERF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PERF))

from agent import layer_depth  # noqa: E402


@pytest.fixture()
def run(tmp_path, monkeypatch):
    for var in ("PERF_MCP_STATE_DIR", "PERF_MCP_LEDGER_DIR", "PERF_MCP_RUN_ID"):
        monkeypatch.delenv(var, raising=False)
    spec = importlib.util.spec_from_file_location("_run_passes", str(_PERF / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_run_passes"] = m
    spec.loader.exec_module(m)
    return m


_NODE = "models/somewhere/a_demo/tests/e2e/test_it.py"


def _seed(run, root, cap):
    (root / run.CC_DIR).mkdir(parents=True, exist_ok=True)
    (root / _NODE).parent.mkdir(parents=True, exist_ok=True)
    (root / _NODE).write_text("def test_it():\n    pass\n")
    run._depth_cache_put(root, _NODE, cap)


def _env_of(run, root):
    pipe = {"perf_test": _NODE + "::test_it", "pcc_test": "p::c", "case": "test_it"}
    return run._mcp_config(root, "/m.json", pipe, "1", "/k.json")["mcpServers"]["perf-mcp"]["env"]


# ---------------------------------------------------------------- the window itself


def test_a_recurring_stage_gets_more_than_one_pass():
    """One pass is the pass that fills what the others read -- never the representative one."""
    assert layer_depth.token_window(1) >= 2
    assert layer_depth.token_window(0) >= 2
    assert layer_depth.token_window(None) >= 2


def test_the_window_follows_the_coverage_answer_when_that_asks_for_more():
    """Not a second opinion on 'how much is representative' -- the same one, reused."""
    assert layer_depth.token_window(8) == 8


def test_the_window_is_never_unbounded():
    """Whatever arrives, a number comes back -- an unbounded capture is the failure being fixed."""
    for bad in (None, "", "all", -5, object()):
        assert isinstance(layer_depth.token_window(bad), int)
        assert layer_depth.token_window(bad) >= 2


# ---------------------------------------------------------------- it has to reach, and be removable


def test_the_window_reaches_the_profiling_server(run, tmp_path):
    cap = {"A_DEPTH_VAR": "2", layer_depth.TOKENS_ENV: "2"}
    _seed(run, tmp_path, cap)

    env = _env_of(run, tmp_path)

    assert json.loads(env["PERF_MCP_PROFILE_ENV"])[layer_depth.TOKENS_ENV] == "2"


def test_the_window_is_named_so_the_full_depth_gate_can_remove_it(run, tmp_path):
    """THE dangerous half. check_full_pipeline_latency times the recurring stage at full length and
    strips the profiling bounds by NAME. A window left on times two passes and reports them as the
    whole request -- the same class of error the layer caps caused on 2026-08-21, 2.47 ms/token
    against a true 17.96."""
    cap = {"A_DEPTH_VAR": "2", layer_depth.TOKENS_ENV: "2"}
    _seed(run, tmp_path, cap)

    env = _env_of(run, tmp_path)

    named = run._split_depth_vars(env.get("PERF_MCP_DEPTH_VARS"))
    assert layer_depth.TOKENS_ENV in named, "the window would survive into the full-length measurement"
    assert set(cap) <= named


def test_the_bound_is_expressed_through_the_module_that_owns_the_names(run):
    """One owner for every variable a profiling bound is expressed through -- the depth cap already
    resolves its name that way, and a second spelling here is how the two drift apart."""
    assert run._tokens_env() == layer_depth.TOKENS_ENV
    assert run._token_window(2) == layer_depth.token_window(2)
    src = (_PERF / "cc_optimize" / "run.py").read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    assert '"%s"' % layer_depth.TOKENS_ENV not in code, "the name is typed here instead of resolved"


def test_the_bridge_bounds_the_passes_it_verifies(run, tmp_path, monkeypatch):
    """The load-bearing half: the cap the bridge PRODUCES has to carry the window.

    Everything else here checks the window survives the trip once it exists. This checks it exists.
    The bridge's probes are stubbed -- what is under test is the env it builds and hands back, not
    the device work it does to verify it.
    """
    node = _NODE
    (tmp_path / node).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / node).write_text("def test_it():\n    pass\n")

    monkeypatch.setattr(run, "_model_root_from_node", lambda *_a, **_k: tmp_path)
    monkeypatch.setattr(run, "_depth_cache_get", lambda *_a, **_k: None)
    monkeypatch.setattr(run, "_depth_cache_put", lambda *_a, **_k: None)
    # the capped probe must look like it reduced work, or the bridge discards the cap by design
    monkeypatch.setattr(run, "_run_op_sigs", lambda *_a, **_k: (None, None, []))
    monkeypatch.setattr(run, "_work_signal", lambda _seq: 100)
    monkeypatch.setattr(run, "_blocks_ran", lambda _seq: 0)

    env = run._bridge_depth_env(
        tmp_path, {}, "1", node, "test_it", 2, full_hint=1000, full_blocks=0, knob={"A_DEPTH_VAR": "2"}
    )

    assert env, "the bridge produced no cap at all"
    assert layer_depth.TOKENS_ENV in env, "the cap bounds depth but not the number of passes"
    assert int(env[layer_depth.TOKENS_ENV]) >= 2


def test_the_full_length_gate_keeps_its_own_unit_after_the_strip(monkeypatch, tmp_path):
    """Stripping the profiling window must not also strip the gate's deliberate choice of unit.

    The window is now in the strip list, and the gate sets that same variable a few lines earlier to
    the unit it means to report. Dropping it lands on the perf test's own default, which agrees --
    so nothing moves, until the override is set for a cheap steering measurement and is silently
    ignored. The gate re-asserts after the drop; these are two different statements.
    """
    import importlib

    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", str(tmp_path / "k.json"))
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)

    src = Path(m.__file__).read_text()
    i = src.index("dropped the profiling depth cap")
    after = src[i : i + 1400]
    assert "PERF_MCP_FULLPIPE_TOKENS" in after, "the gate never puts its own unit back after the strip"
    assert m._tokens_env() == layer_depth.TOKENS_ENV
