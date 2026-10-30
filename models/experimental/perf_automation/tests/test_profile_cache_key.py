# SPDX-License-Identifier: Apache-2.0
"""Profile-cache key must be scoped per module, not per model source.

Module-level optimize runs every module against the SAME source tree but a different
per-module perf test. A source-only cache key made all modules collide on one entry,
so a module could be handed another module's cached profile. The key must fold in the
module identity (PERF_MCP_TASK) and the resolved perf-test node.
"""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_fp",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)


def _seed_stubs(tmp_path):
    stubs = tmp_path / "_stubs"
    stubs.mkdir()
    (stubs / "a.py").write_text("x = 1\n")
    (stubs / "b.py").write_text("y = 2\n")
    return tmp_path


def _fp(monkeypatch, root, task, test_path, case=None):
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT", root)
    monkeypatch.setattr(perf_mcp, "_MANIFEST", {"perf_test_resolved": {"path": test_path, "case": case}})
    monkeypatch.setenv("PERF_MCP_TASK", task)
    return perf_mcp._model_source_fingerprint()


def test_same_source_different_module_gives_different_key(tmp_path, monkeypatch):
    root = _seed_stubs(tmp_path)
    fp_a = _fp(monkeypatch, root, "ace_step_attention", "tests/pcc/test_ace_step_attention.py")
    fp_b = _fp(monkeypatch, root, "ace_step_di_t_model", "tests/pcc/test_ace_step_di_t_model.py")
    assert fp_a and fp_b and fp_a != fp_b


def test_same_module_same_source_is_stable(tmp_path, monkeypatch):
    root = _seed_stubs(tmp_path)
    fp1 = _fp(monkeypatch, root, "mlp", "tests/pcc/test_mlp.py")
    fp2 = _fp(monkeypatch, root, "mlp", "tests/pcc/test_mlp.py")
    assert fp1 == fp2


def test_editing_source_changes_key(tmp_path, monkeypatch):
    root = _seed_stubs(tmp_path)
    fp1 = _fp(monkeypatch, root, "mlp", "tests/pcc/test_mlp.py")
    (root / "_stubs" / "a.py").write_text("x = 999\n")
    fp2 = _fp(monkeypatch, root, "mlp", "tests/pcc/test_mlp.py")
    assert fp1 != fp2


def test_no_stubs_disables_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT", tmp_path)
    monkeypatch.setattr(perf_mcp, "_MANIFEST", {})
    monkeypatch.setenv("PERF_MCP_TASK", "x")
    assert perf_mcp._model_source_fingerprint() == ""
