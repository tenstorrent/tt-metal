"""Discovery sub-agent tests (PLAN section 7.3, expanded neutral schema)."""

import json

import pytest

from agent.model_files import ModelFilesError, build_prompt, read_model_files


def _mock_tree(root):
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_e2e.py").write_text("# end-to-end PCC + perf test\n")
    (root / "tests" / "test_attention.py").write_text("# attention PCC test\n")
    (root / "model.py").write_text("# model source\n")
    (root / "attention.py").write_text("# attention source\n")


GOOD = {
    "perf_test": {"path": "tests/test_e2e.py", "case": "S128"},
    "pcc": {"end_to_end": {"path": "tests/test_e2e.py", "threshold": 0.99}, "attention": "tests/test_attention.py"},
    "components": {"attention": "tests/test_attention.py"},
    "model_files": ["model.py", "attention.py"],
}


def test_read_model_files_returns_expanded_pathmap(tmp_path):
    _mock_tree(tmp_path)
    captured = {}

    def runner(prompt):
        captured["prompt"] = prompt
        return json.dumps(GOOD)

    result = read_model_files(tmp_path, runner=runner)
    assert result["perf_test"]["path"] == "tests/test_e2e.py"
    assert result["perf_test"]["case"] == "S128"
    assert result["pcc"]["end_to_end"]["path"] == "tests/test_e2e.py"
    assert result["pcc"]["end_to_end"]["threshold"] == 0.99
    assert result["components"]["attention"]["path"] == "tests/test_attention.py"
    assert "model.py" in result["model_files"]
    assert str(tmp_path) in captured["prompt"]
    # prompt forbids invented case ids (the S512 lesson)
    assert "never invent an id" in build_prompt(tmp_path)


def test_components_open_map_is_architecture_neutral(tmp_path):
    """A conv-net style pathmap flows through with zero schema changes."""
    _mock_tree(tmp_path)
    conv = dict(GOOD)
    conv["components"] = {"backbone": "tests/test_attention.py", "dcn_head": "tests/test_e2e.py"}
    result = read_model_files(tmp_path, runner=lambda p: json.dumps(conv))
    assert set(result["components"]) == {"backbone", "dcn_head"}


def test_missing_perf_test_falls_back_with_warning(tmp_path):
    """Absent/null perf_test is a WARNING, not fatal: profile the e2e PCC test."""
    _mock_tree(tmp_path)
    bad = {k: v for k, v in GOOD.items() if k != "perf_test"}
    result = read_model_files(tmp_path, runner=lambda p: json.dumps(bad))
    assert result["perf_test"]["path"] == "tests/test_e2e.py"
    assert result["perf_test"]["case"] is None
    assert any(w["code"] == "no_perf_test" for w in result["warnings"])


def test_rejects_empty_case(tmp_path):
    _mock_tree(tmp_path)
    bad = dict(GOOD)
    bad["perf_test"] = {"path": "tests/test_e2e.py", "case": ""}
    with pytest.raises(ModelFilesError, match="case"):
        read_model_files(tmp_path, runner=lambda p: json.dumps(bad))


def test_rejects_non_json(tmp_path):
    _mock_tree(tmp_path)
    with pytest.raises(ModelFilesError):
        read_model_files(tmp_path, runner=lambda p: "not json")


def test_rejects_missing_end_to_end(tmp_path):
    _mock_tree(tmp_path)
    bad = dict(GOOD)
    bad["pcc"] = {"attention": "tests/test_attention.py"}
    with pytest.raises(ModelFilesError):
        read_model_files(tmp_path, runner=lambda p: json.dumps(bad))


def test_rejects_path_off_tree(tmp_path):
    _mock_tree(tmp_path)
    bad = dict(GOOD)
    bad["pcc"] = {"end_to_end": "tests/does_not_exist.py"}
    with pytest.raises(ModelFilesError):
        read_model_files(tmp_path, runner=lambda p: json.dumps(bad))


def test_runner_required(tmp_path):
    with pytest.raises(ValueError):
        read_model_files(tmp_path)


def test_accepts_pytest_node_ids(tmp_path):
    """Sub-agents return path::test_fn (more precise than a bare path) — the
    real BGE-M3 discovery did exactly this; validate the file part only."""
    _mock_tree(tmp_path)
    nid = dict(GOOD)
    nid["perf_test"] = {"path": "tests/test_e2e.py::test_full", "case": "S128"}
    nid["pcc"] = {"end_to_end": {"path": "tests/test_e2e.py::test_full", "threshold": 0.99}}
    result = read_model_files(tmp_path, runner=lambda p: json.dumps(nid))
    assert result["perf_test"]["path"].endswith("::test_full")


def test_notes_are_preserved(tmp_path):
    """Evidence notes ride through normalization for the lead review."""
    _mock_tree(tmp_path)
    rich = dict(GOOD)
    rich["pcc"] = {
        "end_to_end": {"path": "tests/test_e2e.py", "threshold": 0.99, "note": "full model vs HF ref, PCC>0.99"}
    }
    rich["summary"] = "standard encoder layout"
    result = read_model_files(tmp_path, runner=lambda p: json.dumps(rich))
    assert "PCC>0.99" in result["pcc"]["end_to_end"]["note"]
    assert result["summary"] == "standard encoder layout"


def test_e2e_without_threshold_is_fatal(tmp_path):
    """User decision 2026-06-10: e2e file found but no extractable threshold -> stop."""
    _mock_tree(tmp_path)
    bad = dict(GOOD)
    bad["pcc"] = {"end_to_end": "tests/test_e2e.py"}
    with pytest.raises(ModelFilesError, match="no_pcc_threshold"):
        read_model_files(tmp_path, runner=lambda p: json.dumps(bad))


def test_component_without_threshold_is_warning(tmp_path):
    _mock_tree(tmp_path)
    result = read_model_files(tmp_path, runner=lambda p: json.dumps(GOOD))
    assert result["pcc"]["attention"]["threshold"] is None
    assert any(w["code"] == "no_component_pcc_threshold" for w in result["warnings"])


def test_threshold_out_of_range_rejected(tmp_path):
    _mock_tree(tmp_path)
    bad = dict(GOOD)
    bad["pcc"] = {"end_to_end": {"path": "tests/test_e2e.py", "threshold": 1.5}}
    with pytest.raises(ModelFilesError, match="threshold"):
        read_model_files(tmp_path, runner=lambda p: json.dumps(bad))


def test_regen_reconciles_baseline_to_pipeline_perf_test(tmp_path, monkeypatch):
    """Under PERF_REGEN_PERF_TEST=1 the baseline target MUST be the deterministic pipeline perf
    test, not the discovery sub-agent's pick. Regression guard for the nemotron case where the
    sub-agent chose the prefill-only test_perf.py while the real pipeline test was test_*_perf.py
    -> a prefill baseline instead of the decode-inclusive whole pipeline."""
    _mock_tree(tmp_path)
    # the deterministic pipeline test (found by _enumerate_pipelines' 'main' fallback)
    e2e = tmp_path / "tests" / "e2e"
    e2e.mkdir(parents=True)
    (e2e / "test_main_perf.py").write_text("def test_main_perf(device):\n    pass\n")

    monkeypatch.setenv("PERF_REGEN_PERF_TEST", "1")
    # sub-agent picks a DIFFERENT (would-be partial) test
    result = read_model_files(tmp_path, runner=lambda p: json.dumps(GOOD))
    assert result["perf_test"]["path"] == "tests/e2e/test_main_perf.py", "baseline must be the pipeline test"
    assert result["perf_test"]["case"] == "test_main_perf"


def test_no_regen_keeps_subagent_perf_test(tmp_path, monkeypatch):
    """Flag off (unit/dev): no override -> the sub-agent's perf_test stands (non-breaking)."""
    _mock_tree(tmp_path)
    e2e = tmp_path / "tests" / "e2e"
    e2e.mkdir(parents=True)
    (e2e / "test_main_perf.py").write_text("def test_main_perf(device):\n    pass\n")
    monkeypatch.delenv("PERF_REGEN_PERF_TEST", raising=False)
    result = read_model_files(tmp_path, runner=lambda p: json.dumps(GOOD))
    assert result["perf_test"]["path"] == "tests/test_e2e.py"
