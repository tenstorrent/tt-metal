# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""perf_mcp must be TOLD the run root; it cannot guess it.

perf_mcp.py:68 resolves the model directory as

    _MODEL_ROOT = Path(os.environ.get("PERF_MCP_MODEL_ROOT") or _MANIFEST["config"]["model_root"] or ".")

and PERF_MCP_MODEL_ROOT is never set by anything -- so it lands on the manifest value or, failing
that, on "." (the server's own cwd). perf_mcp runs as a SEPARATE PROCESS from the engine, so "."
has no relationship to the run.

Observed on gemma-3-12b-it, 2026-07-31. The matmul sweep wrote 14 PCC-gated shapes to

    /tmp/tt_hw_planner_gemma3_1785461949/models/demos/multimodal/gemma3/matmul_sweep.json

and _warm_start_for(_MODEL_ROOT, ...) looked for ./matmul_sweep.json. Every lookup returned None,
so next_target["warm_start"] was never populated and the whole pre-pass was invisible to the
deterministic path. (The agent still found it by globbing, per run.py's prompt -- which is why the
bf8_b wins landed anyway and the failure went unnoticed.)

The file was never in the wrong place. The process looking for it was never told where to look.
_mcp_config already hands the server PERF_MCP_MANIFEST / PERF_MCP_PERF_TEST / PERF_MCP_KERNEL_LOG;
this adds the one that was missing.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _run_mod():
    spec = importlib.util.spec_from_file_location("cc_run_mcproot", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PIPE = {
    "task": "main",
    "perf_test": "models/demos/multimodal/gemma3/tests/e2e/test_main_perf.py::test_main_perf",
    "pcc_test": "models/demos/multimodal/gemma3/tests/e2e/test_pcc_hf.py::test_e2e_pcc_hf",
}


def _env(tmp_path, pipe=None):
    m = _run_mod()
    cfg = m._mcp_config(tmp_path, str(tmp_path / "manifest.json"), pipe or PIPE, "0", "/tmp/k.json")
    return cfg["mcpServers"]["perf-mcp"]["env"]


def test_model_root_is_handed_to_the_server(tmp_path):
    env = _env(tmp_path)
    got = env.get("PERF_MCP_MODEL_ROOT")
    if not got:
        pytest.fail(
            "PERF_MCP_MODEL_ROOT is not in the MCP server env, so perf_mcp's _MODEL_ROOT still "
            "falls back to '.' and every warm-start lookup misses the sweep table."
        )
    assert got.endswith("models/demos/multimodal/gemma3"), got


def test_it_is_an_absolute_path(tmp_path):
    """The server has a different cwd -- a relative path would reintroduce the same bug."""
    env = _env(tmp_path)
    assert Path(env["PERF_MCP_MODEL_ROOT"]).is_absolute()


def test_it_is_under_the_run_root(tmp_path):
    """It must point INSIDE the isolated worktree, not at the user's real tree: the sweep writes
    into the worktree copy, and that is the file the lookup has to find."""
    env = _env(tmp_path)
    assert str(tmp_path) in env["PERF_MCP_MODEL_ROOT"]


def test_the_sweep_table_resolves_from_it(tmp_path):
    """End to end: write the table where the sweep writes it, then resolve it the way perf_mcp
    does. This is the lookup that returned None for all 14 shapes."""
    import json

    demo = tmp_path / "models/demos/multimodal/gemma3"
    demo.mkdir(parents=True)
    (demo / "matmul_sweep.json").write_text(
        json.dumps(
            {
                "ok": True,
                "shapes": 1,
                "seeds": [{"shape": {"m": 32, "k": 3840, "n": 15360}, "fidelity": "LoFi", "dtype": "bfloat8_b"}],
            }
        )
    )
    root = Path(_env(tmp_path)["PERF_MCP_MODEL_ROOT"])
    assert (root / "matmul_sweep.json").is_file(), f"sweep table not reachable from {root}"

    spec = importlib.util.spec_from_file_location("pmcp_root", str(_PA / "cc_optimize" / "perf_mcp.py"))
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    assert pm._warm_start_for(root, "MatmulDeviceOperation 32 x 3840 x 15360") == {
        "fidelity": "LoFi",
        "dtype": "bfloat8_b",
    }


def test_derived_from_perf_test_not_a_model_rel_key(tmp_path):
    """PRODUCTION pipes have no model_rel key -- pipelines_from_manifest takes it as a PARAMETER.
    A first cut of this fix read pipe["model_rel"], which is always None in a real run, so it
    silently no-opped while the test (whose fixture invented the key) passed."""
    assert "model_rel" not in PIPE, "the fixture must match the real pipe shape"
    env = _env(tmp_path)
    assert env["PERF_MCP_MODEL_ROOT"].endswith("models/demos/multimodal/gemma3")


@pytest.mark.parametrize(
    "node,want",
    [
        ("models/demos/multimodal/gemma3/tests/e2e/test_main_perf.py::t", "models/demos/multimodal/gemma3"),
        ("models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py::t", "models/demos/llama3_1_8b_p150"),
        ("models/demos/x/tests/pcc/test_mod.py", "models/demos/x"),
        ("models/demos/y/perf.py::t", "models/demos/y"),
        ("", ""),
        (None, ""),
    ],
)
def test_model_rel_derivation(node, want):
    assert _run_mod()._model_rel_from_perf_test(node) == want


def test_the_other_keys_are_untouched(tmp_path):
    env = _env(tmp_path)
    for k in ("PERF_MCP_MANIFEST", "PERF_MCP_PERF_TEST", "PERF_MCP_PCC_TEST", "PERF_MCP_KERNEL_LOG"):
        assert k in env, f"dropped pre-existing key {k}"
