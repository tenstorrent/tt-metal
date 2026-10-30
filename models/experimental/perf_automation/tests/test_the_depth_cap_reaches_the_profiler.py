"""The depth cap has to survive the process that computed it.

before_loop works out which spelling of the layer-cap variable actually reaches the builder, and at
what depth, then records it with ``os.environ[...] = ...``. But discover() runs before_loop as a
SUBPROCESS, so that assignment sets the child's environment and dies with the child. The parent
never sees it; cc_env copies a parent os.environ that never had it; and the profiling server -- which
by design inherits nothing -- is handed an env dict without it.

So the cap was computed correctly on every run and applied to nothing except the child's own baseline
profile. Every profile the ranking actually reads ran the model UNCAPPED.

That was harmless while the uncapped forward fit inside tracy's 32K source-location budget, and
silent because nothing errors -- the capture just stops early. On voxtral_mini_3b_2507 the model
crossed that budget on 2026-09-03: hand-written kernels are new source locations, so the optimizer's
own wins pushed it over. 163 of the next 254 captures came back "Instrumentation failure", and the
per-stage split, the bucket shares and every ranking built on them went with them.

The value was durable the whole time -- the bridge writes it to the coverage cache. It just was
never read back where the server's environment is assembled.

No variable name is written here: the cap's spelling is whatever the bridge discovered for this
model, and these tests invent their own to prove nothing recognises a particular one.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

_PERF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PERF))


@pytest.fixture()
def run(tmp_path, monkeypatch):
    for var in ("PERF_MCP_STATE_DIR", "PERF_MCP_LEDGER_DIR", "PERF_MCP_RUN_ID"):
        monkeypatch.delenv(var, raising=False)
    spec = importlib.util.spec_from_file_location("_run_under_test", str(_PERF / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_run_under_test"] = m
    spec.loader.exec_module(m)
    return m


# A cap this model happens to spell these ways. The code must never recognise the words.
_NODE = "models/somewhere/a_demo/tests/e2e/test_it.py"
_CASE = "test_it"
_CAP = {"SOME_DEPTH_VAR": "2", "ANOTHER_DEPTH_VAR": "2"}


def _pipe(node=_NODE, case=_CASE):
    return {"perf_test": "%s::%s" % (node, case), "pcc_test": "p::c", "case": case}


def _seed_cache(run, repo_root, node, cap):
    """Put a cap in the cache the way the bridge does, so the fingerprint matches."""
    (repo_root / run.CC_DIR).mkdir(parents=True, exist_ok=True)
    (repo_root / node).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / node).write_text("def test_it():\n    pass\n")
    run._depth_cache_put(repo_root, node, cap)


def _env_of(cfg):
    return cfg["mcpServers"]["perf-mcp"]["env"]


# ---------------------------------------------------------------- the bug


def test_a_computed_cap_reaches_the_profiling_server(run, tmp_path):
    """The whole defect: computed in one process, needed in another, carried by neither."""
    _seed_cache(run, tmp_path, _NODE, _CAP)

    env = _env_of(run._mcp_config(tmp_path, "/m.json", _pipe(), "1", "/k.json"))

    assert "PERF_MCP_PROFILE_ENV" in env, "the cap did not cross into the server's environment"
    assert json.loads(env["PERF_MCP_PROFILE_ENV"]) == _CAP


def test_the_case_is_dropped_before_the_lookup(run, tmp_path):
    """The cache keys on the test FILE; a caller holding `path::case` matches nothing without this."""
    _seed_cache(run, tmp_path, _NODE, _CAP)

    assert run._node_path("%s::%s" % (_NODE, _CASE)) == _NODE
    assert run._depth_cache_get(tmp_path, run._node_path(_pipe()["perf_test"])) == _CAP


def test_with_no_cap_recorded_the_environment_is_unchanged(run, tmp_path):
    """Silent where there is nothing to carry -- a model with no depth knob profiles as before."""
    env = _env_of(run._mcp_config(tmp_path, "/m.json", _pipe(), "1", "/k.json"))

    assert "PERF_MCP_PROFILE_ENV" not in env


def test_a_cap_recorded_for_a_different_test_is_not_borrowed(run, tmp_path):
    """The cap is a property of the node that was probed, not of the run."""
    _seed_cache(run, tmp_path, "models/somewhere/a_demo/tests/e2e/test_other.py", _CAP)

    env = _env_of(run._mcp_config(tmp_path, "/m.json", _pipe(), "1", "/k.json"))

    assert "PERF_MCP_PROFILE_ENV" not in env


def test_no_cap_variable_is_named_in_the_source(run):
    """The spelling is discovered per model; recognising one here would be the hardcoding the
    bridge exists to avoid."""
    src = (_PERF / "cc_optimize" / "run.py").read_text()
    body = src[src.index("def _mcp_config(") : src.index("def _mcp_config(") + 4000]
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    for typed in ("TT_PERF_LAYERS", "TT_PERF_DECODE_LAYERS", "TT_PERF_ENCODE_LAYERS"):
        assert typed not in code, typed


def test_every_capped_variable_is_named_so_the_full_depth_gate_can_strip_it(run, tmp_path):
    """Carrying the cap is only half of it -- one gate has to take it back off.

    check_full_pipeline_latency times the model at FULL depth with tracy off, so it strips the cap
    first. It can only strip names it can derive: PERF_MCP_DEPTH_VARS, plus the stage spellings it
    reads from the model. The bridge caps per stage as well as globally, and a stage name the
    derivation misses is a cap left on -- which times a two-layer model and reports it as the whole
    one (measured on this model 2026-08-21: 2.47 ms/token against a true 17.96). Naming the keys
    makes the strip independent of that derivation.
    """
    cap = dict(_CAP)
    cap["A_PER_STAGE_VAR"] = "2"
    _seed_cache(run, tmp_path, _NODE, cap)

    env = _env_of(run._mcp_config(tmp_path, "/m.json", _pipe(), "1", "/k.json"))

    named = run._split_depth_vars(env.get("PERF_MCP_DEPTH_VARS"))
    assert set(cap) <= named, "capped but unnamed: %s" % sorted(set(cap) - named)


def test_naming_the_cap_does_not_drop_names_already_listed(run, tmp_path):
    """The caller knows the global knob; this knows the per-stage keys. Neither may clobber the
    other -- a dropped name is a cap the full-depth gate cannot take off."""
    _seed_cache(run, tmp_path, _NODE, _CAP)
    env = _env_of(run._mcp_config(tmp_path, "/m.json", _pipe(), "1", "/k.json"))
    env["PERF_MCP_DEPTH_VARS"] = ",".join(
        sorted(run._split_depth_vars(env["PERF_MCP_DEPTH_VARS"]) | {"AN_EARLIER_VAR"})
    )

    merged = run._split_depth_vars(env["PERF_MCP_DEPTH_VARS"])

    assert "AN_EARLIER_VAR" in merged
    assert set(_CAP) <= merged
