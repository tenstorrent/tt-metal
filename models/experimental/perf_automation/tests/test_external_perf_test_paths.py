import json
import subprocess
from types import SimpleNamespace


def test_pipeline_preserves_explicit_external_absolute_perf_path(tmp_path):
    from cc_optimize.run import pipelines_from_manifest

    external = tmp_path / "outside" / "test_perf.py"
    manifest = {
        "perf_test_resolved": {"path": f"{external}::test_perf", "case": "test_perf"},
        "pathmap": {
            "pipelines": [
                {
                    "task": "main",
                    "perf_test": "../../../outside/test_perf.py::test_perf",
                    "pcc_test": "tests/test_pcc.py::test_pcc",
                }
            ]
        },
    }

    pipes = pipelines_from_manifest(manifest, "models/demo")

    assert pipes[0]["perf_test"] == str(external)
    assert pipes[0]["case"] == "test_perf"


def test_collected_external_node_keeps_absolute_file(tmp_path):
    from agent.probes import resolve_node_id

    external = tmp_path / "outside" / "test_perf.py"
    external.parent.mkdir()
    external.write_text("def test_perf(): pass\n")

    def fake_runner(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="outside/test_perf.py::test_perf\n",
            stderr="",
        )

    node = resolve_node_id(tmp_path, str(external), case="test_perf", runner=fake_runner)

    assert node == f"{external}::test_perf"


def test_explicit_perf_gate_requires_optimizer_markers_and_depth_control(tmp_path, monkeypatch):
    import cc_optimize.run as run

    gate = tmp_path / "test_perf.py"
    gate.write_text(
        "def test_perf():\n"
        "    print('TRACE_PER_TOKEN_MS=1.0')\n"
        "    depth = 'TT_PERF_LAYERS'\n"
    )
    monkeypatch.setattr(
        run.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=f"{gate}::test_perf\n", stderr=""
        ),
    )

    ok, reason = run._preflight_explicit_gate(tmp_path, f"{gate}::test_perf", performance=True)

    assert ok, reason


def test_explicit_perf_gate_rejects_standalone_perf_output(tmp_path):
    import cc_optimize.run as run

    gate = tmp_path / "test_perf.py"
    gate.write_text("def test_perf():\n    print('PERF wall_ms=1.0')\n")

    ok, reason = run._preflight_explicit_gate(tmp_path, str(gate), performance=True)

    assert not ok
    assert "TRACE_PER_TOKEN_MS" in reason


def test_failed_discovery_without_baseline_never_reaches_agent_round(tmp_path, monkeypatch):
    import cc_optimize.run as run

    repo = tmp_path / "repo"
    perf_dir = repo / run.PERF_DIR
    run_dir = perf_dir / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    manifest = {
        "pathmap": {
            "pipelines": [{"task": "main", "perf_test": "test_perf.py"}],
            "pcc": {"end_to_end": {"path": "test_pcc.py"}},
        }
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(run, "_run_device_step", lambda *args, **kwargs: (1, "failed"))

    result = run.discover(tmp_path / "model", repo, "all", "wall_ms")

    assert result is None


def test_failed_discovery_cannot_launder_a_degenerate_profile(tmp_path, monkeypatch):
    import cc_optimize.run as run

    repo = tmp_path / "repo"
    run_dir = repo / run.PERF_DIR / "runs" / "run-1"
    (run_dir / "profiles").mkdir(parents=True)
    manifest = {
        "pathmap": {
            "pipelines": [{"task": "main", "perf_test": "test_perf.py"}],
            "pcc": {"end_to_end": {"path": "test_pcc.py"}},
        }
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "profiles" / "baseline_profile.json").write_text(
        json.dumps({"device_ms": 0.085, "buckets": [{"id": "datamove", "count": 22}]})
    )
    monkeypatch.setattr(run, "_run_device_step", lambda *args, **kwargs: (1, "failed"))
    monkeypatch.setattr(
        run,
        "_perf_mcp",
        lambda: SimpleNamespace(_is_credible_profile=lambda profile: False),
    )

    result = run.discover(tmp_path / "model", repo, "all", "wall_ms")

    assert result is None


def test_validated_discovery_can_be_reused_without_another_agent_call(tmp_path, monkeypatch):
    import cc_optimize.run as run

    repo = tmp_path / "repo"
    model = repo / "models" / "demo"
    model.mkdir(parents=True)
    run_dir = repo / run.PERF_DIR / "runs" / "run-1"
    (run_dir / "profiles").mkdir(parents=True)
    perf = "/outside/test_perf.py::test_perf"
    pcc = "/outside/test_pcc.py::test_pcc"
    manifest = {
        "config": {"model_root": str(model), "perf_test": perf, "pcc_test": pcc},
        "pathmap": {
            "pipelines": [{"task": "main", "perf_test": "test_perf.py"}],
            "pcc": {"end_to_end": {"path": "test_pcc.py"}},
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (run_dir / "profiles" / "baseline_profile.json").write_text(
        json.dumps({"device_ms": 8.1, "buckets": [{"id": "matmul", "count": 72}]})
    )
    monkeypatch.setenv("PERF_MCP_REUSE_DISCOVERY_MANIFEST", str(manifest_path))
    monkeypatch.setattr(
        run,
        "_perf_mcp",
        lambda: SimpleNamespace(_is_credible_profile=lambda profile: True),
    )
    monkeypatch.setattr(
        run,
        "_run_device_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("discovery must not run")),
    )

    result = run.discover(model, repo, "all", "wall_ms", perf_test=perf, pcc_test=pcc)

    assert result == manifest
