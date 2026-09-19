# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HARD STRESS: the run root handed to perf_mcp.

This value is how a SEPARATE PROCESS locates the run. Get it wrong and nothing errors -- the
lookups just quietly return None, which is exactly how 14 PCC-gated matmul shapes were measured,
written, and then ignored for a whole 6-round run.

The failure mode to defend against is not "wrong string" but "plausible string": ".", the user's
real tree instead of the worktree, or a path that simply does not exist. All three look fine in a
log and silently break every lookup.

  s1  DERIVATION: 400 node ids -> the model dir, against an independent oracle
  s2  ABSOLUTE + INSIDE THE WORKTREE: never ".", never the user's real tree
  s3  ROUND TRIP: the sweep writes, perf_mcp reads -- the lookup that returned None all run
  s4  the pipe shape is the PRODUCTION one (no invented model_rel key)
  s5  hostile pipes never raise and never emit a bogus path
  s6  determinism + no collateral changes to the other env keys
"""

import importlib.util
import json
import random
import string
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _run_mod():
    spec = importlib.util.spec_from_file_location("cc_run_root_stress", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_M = _run_mod()
_DERIVE = _M._model_rel_from_perf_test

GEMMA = "models/demos/multimodal/gemma3"
PIPE = {
    "task": "main",
    "perf_test": f"{GEMMA}/tests/e2e/test_main_perf.py::test_main_perf",
    "pcc_test": f"{GEMMA}/tests/e2e/test_pcc_hf.py::test_e2e_pcc_hf",
}


def _env(tmp_path, pipe=None):
    cfg = _M._mcp_config(tmp_path, str(tmp_path / "manifest.json"), pipe or PIPE, "0", "/tmp/k.json")
    return cfg["mcpServers"]["perf-mcp"]["env"]


# --------------------------------------------------------------------------- s1
def _oracle(node):
    """Independent restatement: the model dir is everything above the first `tests/` segment."""
    s = str(node or "").split("::")[0].strip()
    if not s:
        return ""
    parts = Path(s).parts
    if "tests" in parts:
        return str(Path(*parts[: parts.index("tests")]))
    return str(Path(s).parent)


def test_s1_400_node_ids_match_the_oracle():
    rng = random.Random(20260731)
    depths = [
        ["models", "demos", "gemma3"],
        ["models", "demos", "multimodal", "gemma3"],
        ["models", "experimental", "a", "b", "c"],
        ["models", "demos", "llama3_1_8b_p150"],
    ]
    tails = [
        ["tests", "e2e", "test_main_perf.py"],
        ["tests", "pcc", "test_mod.py"],
        ["tests", "test_x.py"],
        ["perf.py"],
        ["demo", "text_demo.py"],
    ]
    for i in range(400):
        node = "/".join(rng.choice(depths) + rng.choice(tails))
        if rng.random() < 0.5:
            node += "::test_" + "".join(rng.choice(string.ascii_lowercase) for _ in range(4))
        assert _DERIVE(node) == _oracle(node), f"case {i}: {node}"


def test_s1_first_tests_segment_wins():
    """A model whose own path contains 'tests' deeper down must not be truncated at the wrong one."""
    assert _DERIVE("models/demos/x/tests/e2e/tests/test_y.py::t") == "models/demos/x"


def test_s1_case_suffix_is_stripped():
    assert _DERIVE(f"{GEMMA}/tests/e2e/t.py::test_a[param-1]") == GEMMA


# --------------------------------------------------------------------------- s2
def test_s2_never_dot_and_always_absolute(tmp_path):
    root = _env(tmp_path)["PERF_MCP_MODEL_ROOT"]
    assert root not in (".", "", None)
    assert Path(root).is_absolute()
    assert not root.startswith("./")


def test_s2_points_inside_the_given_repo_root(tmp_path):
    """The worktree, not the user's real tree: the sweep writes into the worktree copy."""
    root = Path(_env(tmp_path)["PERF_MCP_MODEL_ROOT"])
    assert str(root).startswith(str(tmp_path)), f"{root} escaped {tmp_path}"


@pytest.mark.parametrize("repo", ["/tmp/tt_hw_planner_gemma3_123", "/home/ttuser/tt-metal-gemma3", "/a/b/c"])
def test_s2_tracks_whichever_root_it_is_given(repo):
    root = _M._mcp_config(Path(repo), "m.json", PIPE, "0", "/tmp/k.json")["mcpServers"]["perf-mcp"]["env"][
        "PERF_MCP_MODEL_ROOT"
    ]
    assert root == str((Path(repo) / GEMMA).resolve())


# --------------------------------------------------------------------------- s3
def test_s3_sweep_write_then_perf_mcp_read(tmp_path):
    """The full round trip that failed: sweep writes to demo_dir, perf_mcp resolves via the env."""
    demo = tmp_path / GEMMA
    demo.mkdir(parents=True)
    (demo / "matmul_sweep.json").write_text(
        json.dumps(
            {
                "ok": True,
                "shapes": 2,
                "seeds": [
                    {"shape": {"m": 32, "k": 3840, "n": 15360}, "fidelity": "LoFi", "dtype": "bfloat8_b"},
                    {"shape": {"m": 128, "k": 15360, "n": 3840}, "fidelity": "HiFi2", "dtype": "bfloat16"},
                ],
            }
        )
    )
    root = Path(_env(tmp_path)["PERF_MCP_MODEL_ROOT"])
    spec = importlib.util.spec_from_file_location("pmcp_rt", str(_PA / "cc_optimize" / "perf_mcp.py"))
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    assert pm._warm_start_for(root, "MatmulDeviceOperation 32 x 3840 x 15360") == {
        "fidelity": "LoFi",
        "dtype": "bfloat8_b",
    }
    assert pm._warm_start_for(root, "MatmulDeviceOperation 128 x 15360 x 3840") == {
        "fidelity": "HiFi2",
        "dtype": "bfloat16",
    }
    # a shape the sweep never measured must still be a clean miss, not a wrong answer
    assert pm._warm_start_for(root, "MatmulDeviceOperation 7 x 7 x 7") is None


def test_s3_the_old_behaviour_would_have_missed(tmp_path):
    """Control: resolving from "." finds nothing, which is what the run actually did."""
    demo = tmp_path / GEMMA
    demo.mkdir(parents=True)
    (demo / "matmul_sweep.json").write_text(json.dumps({"seeds": []}))
    spec = importlib.util.spec_from_file_location("pmcp_ctl", str(_PA / "cc_optimize" / "perf_mcp.py"))
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    assert pm._warm_start_for(Path("."), "MatmulDeviceOperation 32 x 3840 x 15360") is None


# --------------------------------------------------------------------------- s4
def test_s4_production_pipe_shape():
    """pipelines_from_manifest emits exactly these keys. If a future change adds model_rel, good --
    but the derivation must not DEPEND on it, which is the bug this replaced."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def pipelines_from_manifest")
    body = src[i : i + 2000]
    assert '"perf_test":' in body
    assert '"model_rel":' not in body, "pipe now carries model_rel; simplify the derivation"


# --------------------------------------------------------------------------- s5
@pytest.mark.parametrize(
    "pipe",
    [
        {"task": "main", "perf_test": "", "pcc_test": ""},
        {"task": "main", "perf_test": None, "pcc_test": ""},
        {"task": "main", "perf_test": "::justacase", "pcc_test": ""},
        {"task": "main", "perf_test": "no_slashes.py", "pcc_test": ""},
    ],
)
def test_s5_hostile_pipes_never_raise(tmp_path, pipe):
    # NOTE: a pipe MISSING perf_test entirely is out of scope -- _mcp_config has always indexed
    # pipe["perf_test"] directly (run.py:262), and pipelines_from_manifest always sets it. Changing
    # that is a separate decision; these cases cover perf_test PRESENT but unusable.
    env = _env(tmp_path, pipe)
    assert isinstance(env, dict)
    root = env.get("PERF_MCP_MODEL_ROOT")
    # either absent (honest: we could not tell) or a real absolute path -- never "." or a fragment
    assert root is None or (Path(root).is_absolute() and not root.endswith("/."))


def test_s5_empty_perf_test_omits_the_key_rather_than_guessing(tmp_path):
    """Absent is honest; "." is a wrong answer that silently breaks every lookup."""
    env = _env(tmp_path, {"task": "main", "perf_test": "", "pcc_test": ""})
    assert env.get("PERF_MCP_MODEL_ROOT") in (None, "")


# --------------------------------------------------------------------------- s6
def test_s6_deterministic(tmp_path):
    a = _env(tmp_path)["PERF_MCP_MODEL_ROOT"]
    for _ in range(20):
        assert _env(tmp_path)["PERF_MCP_MODEL_ROOT"] == a


def test_s6_other_env_keys_unchanged(tmp_path):
    env = _env(tmp_path)
    for k in (
        "PERF_MCP_MANIFEST",
        "PERF_MCP_PERF_TEST",
        "PERF_MCP_PCC_TEST",
        "PERF_MCP_KERNEL_LOG",
        "TT_METAL_HOME",
        "PYTHONPATH",
        "PATH",
    ):
        assert k in env, f"dropped {k}"


def test_s6_case_key_still_forwarded(tmp_path):
    env = _env(tmp_path, {**PIPE, "case": "perf-1"})
    assert env["PERF_MCP_PERF_CASE"] == "perf-1"
    assert env["PERF_MCP_MODEL_ROOT"].endswith(GEMMA)
