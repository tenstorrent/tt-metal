"""Integration tests for the Stage-1 driver (ladder rung A, all mocks)."""

import json
from pathlib import Path

import pytest

from agent.before_loop import (
    before_loop,
    make_mock_model_runner,
    mock_collect_cases,
    mock_env_probe,
    mock_preflight,
    mock_review,
    mock_run_profiled,
)
from agent.model_files import ModelFilesError

PKG = Path(__file__).parent.parent
FACTORY = lambda perf, case: mock_run_profiled


@pytest.fixture
def model_root(tmp_path, monkeypatch):
    """A model dir with BOTH a perf test and a usable correctness gate.

    The gate stub is required, not decoration: before_loop now refuses to start when neither a
    --pcc-test nor a cached HF reference is available, because without ground truth optimize is free
    to commit a lever that silently degrades the model. These tests exercise manifest / telemetry /
    metric-default behaviour, none of which is about the gate, so they supply the cheapest gate that
    satisfies the policy -- a file declaring a numeric PCC threshold.
    """
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "test_e2e.py").write_text("# perf test stub\n")
    (tmp_path / "model" / "attention.py").write_text("# attn stub\n")
    (tmp_path / "model" / "test_pcc.py").write_text(
        "def test_pcc():\n"
        "    pcc = 1.0\n"
        "    print(f'PCC: {pcc}')\n"
        "    assert_with_pcc(expected, actual, 0.99)\n"
    )
    (tmp_path / "model" / "test_e2e.py").write_text("def test_perf():\n    pass\n")

    from agent import perf_test_gen as _ptg

    monkeypatch.setattr(
        _ptg,
        "generate_perf_test",
        # `stacks` is not optional here on purpose: the generator has always accepted it and no
        # production caller passed it, so every generated perf test assumed one stack. A double
        # that quietly swallowed **kwargs would let that regress unnoticed.
        lambda root, task, case=None, force=False, source_abs=None, source_kind="", stacks=None: str(
            Path(root) / "test_e2e.py"
        ),
    )
    return tmp_path / "model"


def _pcc_gate(model_root):
    return "%s::test_pcc" % (Path(model_root) / "test_pcc.py")


def _run(tmp_path, model_root, config_extra=None, runner=None):
    config = {
        "model_root": str(model_root),
        "metric": "wall_ms",
        "direction": "min",
        "target": 12.0,
        "runs": 3,
        "pcc_test": _pcc_gate(model_root),
    }
    config.update(config_extra or {})
    return before_loop(
        config,
        env_probe=mock_env_probe,
        model_runner=runner or make_mock_model_runner(model_root),
        run_profiled_factory=FACTORY,
        preflight=mock_preflight,
        review=mock_review,
        collect=mock_collect_cases,
        runs_root=tmp_path / "runs",
        playbook_dir=PKG / "GUIDELINES",
        cache_path=tmp_path / "cache" / "playbook_index.json",
        tt_metal_root=tmp_path,
    )


def test_before_loop_all_mocks_produces_manifest_and_baseline(tmp_path, model_root):
    result = _run(tmp_path, model_root)
    run_dir = Path(result["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["env"]["arch"] in ("wormhole", "blackhole")
    assert manifest["pathmap"]["perf_test"]["path"]
    assert manifest["playbook_sections"] >= 50
    state = json.loads((run_dir / "state.json").read_text())
    assert state["state"] == "BEFORE_LOOP_DONE"
    assert state["metric"]["baseline"] == state["metric"]["current"]
    assert result["profile"]["buckets"]
    assert (run_dir / "profiles" / "iter_baseline_report.csv").is_file()
    assert (run_dir / "events.jsonl").is_file()
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [e["stage"] for e in events if e["event"] == "start"] == [
        "environment_check",
        "startup_reset",
        "cache_playbook",
        # "agent_sdk_health" was here: a subprocess probe that pre-empted a broken in-process
        # claude_agent_sdk import before discovery touched it. There is no in-process SDK any more --
        # every model call is a `claude` CLI subprocess -- so the stage went with it.
        "ensure_tt_lang",
        # Read-only, sub-second, no device -- and BEFORE discovery, because a model that cannot be
        # measured the way this tool measures should be told so before the perf test is generated and
        # the weights load, not forty minutes later as a crash whose cause is not in the message.
        "model_contract",
        "discover",
        "lead_review",
        "preflight",
        # Injecting the stage marks is a STEP, not a side effect: it rewrites the file the run is
        # about to profile, and it must precede resolve_signposts so the start/stop pair it emits is
        # found by that scan rather than defaulted.
        "stage_marks",
        "resolve_signposts",
        "tracy_baseline",
    ]
    manifest2 = json.loads((run_dir / "manifest.json").read_text())
    assert manifest2["discovery_review"]["decision"] == "continue"


def test_before_loop_fatal_flag_stops_run(tmp_path, model_root):
    def fatal_runner(prompt):
        return json.dumps(
            {
                "perf_test": None,
                "pcc": {},
                "components": {},
                "model_files": ["attention.py"],
                "flags": [{"level": "fatal", "code": "no_end_to_end_pcc", "detail": "no e2e PCC test found"}],
            }
        )

    with pytest.raises(ModelFilesError, match="CANNOT CONTINUE"):  # allow-pytest.raises: no expect_error fixture
        _run(tmp_path, model_root, runner=fatal_runner)


def test_before_loop_perf_warning_falls_back_to_pcc(tmp_path, model_root):
    def warn_runner(prompt):
        return json.dumps(
            {
                "perf_test": None,
                "pcc": {"end_to_end": {"path": "test_e2e.py", "threshold": 0.99}},
                "components": {},
                "model_files": ["attention.py"],
                "flags": [{"level": "warning", "code": "no_perf_test", "detail": "no perf.py; using e2e pcc"}],
            }
        )

    result = _run(tmp_path, model_root, runner=warn_runner)
    manifest = json.loads((Path(result["run_dir"]) / "manifest.json").read_text())
    # fallback: tracy profiles the e2e PCC test
    assert manifest["perf_test_resolved"]["path"].endswith("test_e2e.py")
    assert manifest["pathmap"]["warnings"][0]["code"] == "no_perf_test"


def test_before_loop_metric_block_max_direction(tmp_path, model_root):
    result = _run(tmp_path, model_root, config_extra={"metric": "fps", "direction": "max", "runs": 1})
    state = json.loads((Path(result["run_dir"]) / "state.json").read_text())
    assert state["metric"]["name"] == "fps"
    assert state["metric"]["direction"] == "max"


def test_default_case_is_first_collected(tmp_path, model_root):
    """No case from user or sub-agent -> FIRST collected case, loudly logged."""

    def null_case_runner(prompt):
        return json.dumps(
            {
                "perf_test": None,
                "pcc": {"end_to_end": {"path": "test_e2e.py", "threshold": 0.99}},
                "components": {},
                "model_files": ["attention.py"],
                "flags": [{"level": "warning", "code": "no_perf_test", "detail": "x"}],
            }
        )

    result = _run(tmp_path, model_root, runner=null_case_runner)
    manifest = json.loads((Path(result["run_dir"]) / "manifest.json").read_text())
    assert manifest["perf_test_resolved"]["case"] == "mock"  # from mock_collect_cases
    events = (Path(result["run_dir"]) / "events.jsonl").read_text()
    assert "DEFAULTING to FIRST collected case" in events


def test_manifest_written_even_when_tracy_fails(tmp_path, model_root):
    def boom(perf, case):
        def rp(*a, **k):
            raise RuntimeError("device exploded")

        return rp

    config = {"model_root": str(model_root), "metric": "wall_ms", "runs": 1, "pcc_test": _pcc_gate(model_root)}
    with pytest.raises(RuntimeError, match="device exploded"):  # allow-pytest.raises: no expect_error fixture
        before_loop(
            config,
            mock_env_probe,
            make_mock_model_runner(model_root),
            boom,
            mock_preflight,
            mock_review,
            mock_collect_cases,
            runs_root=tmp_path / "runs",
            cache_path=tmp_path / "c" / "i.json",
            tt_metal_root=tmp_path,
        )
    run_dirs = [d for d in (tmp_path / "runs").iterdir() if d.is_dir()]
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    assert manifest["discovery_review"]["decision"] == "continue"


def test_devices_single_recorded_in_manifest(tmp_path, model_root):
    result = _run(tmp_path, model_root, config_extra={"devices": "single"})
    manifest = json.loads((Path(result["run_dir"]) / "manifest.json").read_text())
    # devices spec is recorded, but visibility is NEVER restricted (a chip-subset pin crashes fabric
    # auto-discovery on multi-chip boards): 'single' leaves TT_VISIBLE_DEVICES unset -> visible_devices None.
    assert manifest["config"]["devices"] == "single"
    assert manifest["config"]["visible_devices"] is None


def test_default_metric_is_device_time(tmp_path, model_root):
    # call before_loop directly with NO metric key, so the default is exercised
    config = {"model_root": str(model_root), "runs": 1, "pcc_test": _pcc_gate(model_root)}
    result = before_loop(
        config,
        mock_env_probe,
        make_mock_model_runner(model_root),
        FACTORY,
        mock_preflight,
        mock_review,
        mock_collect_cases,
        runs_root=tmp_path / "runs",
        playbook_dir=PKG / "GUIDELINES",
        cache_path=tmp_path / "c" / "i.json",
        tt_metal_root=tmp_path,
    )
    state = json.loads((Path(result["run_dir"]) / "state.json").read_text())
    assert state["metric"]["name"] == "device_ms"
    total = sum(b["device_ms"] for b in result["profile"]["buckets"])
    assert abs(state["metric"]["baseline"] - total) < 1e-3
    assert state["metric"]["baseline"] < result["profile"]["wall_ms"]


def test_check_dependencies_reports_missing(monkeypatch):
    import shutil

    from agent.before_loop import check_dependencies

    assert check_dependencies() == []  # this env has both
    monkeypatch.setattr(shutil, "which", lambda name: None)
    missing = check_dependencies()
    assert any("tt-perf-report" in m for m in missing)


def test_baseline_profile_json_persisted(tmp_path, model_root):
    """ROUTE reads the tagged buckets from profiles/baseline_profile.json, not the CSVs."""
    result = _run(tmp_path, model_root)
    prof = json.loads((Path(result["run_dir"]) / "profiles" / "baseline_profile.json").read_text())
    assert prof["buckets"] and "device_ms" in prof


def test_agent_call_telemetry_persisted(tmp_path, model_root):
    """One row per query() in agent_calls.jsonl; cumulative tokens+cost in state."""
    result = _run(tmp_path, model_root)
    rows = [json.loads(l) for l in (Path(result["run_dir"]) / "agent_calls.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["discover", "lead_review"]
    assert all(r["tokens_in"] > 0 and r["cost_usd"] > 0 for r in rows)
    state = json.loads((Path(result["run_dir"]) / "state.json").read_text())
    assert state["tokens_in"] == sum(r["tokens_in"] for r in rows)
    assert state["tokens_out"] == sum(r["tokens_out"] for r in rows)
    assert state["cost_usd"] == round(sum(r["cost_usd"] for r in rows), 6)
