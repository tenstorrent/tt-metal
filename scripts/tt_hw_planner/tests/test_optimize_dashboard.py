# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import json
import urllib.request

from scripts.tt_hw_planner.optimize_dashboard import (
    _attempt_status,
    collect_state,
    find_run_dir,
    make_server,
    repo_root_for_run,
    run_slug,
    state_dir_candidates,
)


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)


def _make_run(tmp_path, slug="some_model_x"):
    """A minimal run dir + state dir, with DELIBERATELY non-standard stage names: the dashboard must
    render what the model reported, never a stage set baked into the code."""
    repo = tmp_path / "repo"
    run_dir = repo / "models/experimental/perf_automation/runs" / ("2026-08-30T10-00-00-" + slug)
    _write(run_dir / "manifest.json", {"config": {"model_root": "/x/" + slug, "metric": "device_ms", "devices": "0"}})
    _write(
        run_dir / "state.json",
        {
            "state": "LOOP",
            "iteration": 3,
            "metric": {
                "name": "device_ms",
                "unit": "ms",
                "direction": "min",
                "baseline": 100.0,
                "current": 80.0,
                "target": 50.0,
            },
        },
    )
    _write(
        run_dir / "events.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "2026-08-30T10:00:01Z",
                        "phase": "loop",
                        "stage": "measure",
                        "event": "start",
                        "detail": "d1",
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-08-30T10:00:02Z",
                        "phase": "loop",
                        "stage": "measure",
                        "event": "done",
                        "detail": "d2",
                        "seconds": 1.2,
                    }
                ),
            ]
        ),
    )
    _write(
        run_dir / "profiles" / "baseline_profile.json",
        {
            "buckets": [
                {
                    "id": "weirdop",
                    "device_ms": 60.0,
                    "pct": 75.0,
                    "count": 10,
                    "tags": {"bound": "compute"},
                    "top_ops": [{"op_code": "WeirdOpDeviceOperation 4 x 4"}],
                }
            ]
        },
    )
    state = tmp_path / "state"
    _write(
        state / ("perf_mcp_stage_ms_%s_main.json" % slug),
        {
            "stages": {"audio_encode": 12.0, "lm_head": 30.0},
            "paths": {"audio_encode": "trace+1cq"},
            "bytes": {"audio_encode": 1024},
        },
    )
    _write(
        state / ("perf_mcp_full_pipeline_baseline_1cq_%s_main.json" % slug),
        {"full_pipeline_ms": 30.0, "unit": "token", "stages": {"audio_encode": 14.0, "lm_head": 30.0}},
    )
    _write(
        state / ("perf_measurements_%s_main.jsonl" % slug),
        "\n".join(
            [
                json.dumps({"kind": "fullpipe_e2e", "phase": "before", "value_ms": 30.0}),
                json.dumps({"kind": "fullpipe_e2e", "phase": "after", "value_ms": 25.0}),
            ]
        ),
    )
    _write(
        state / ("cc_kernlog_%s_main.json.cumulative" % slug),
        [
            {
                "op_signature": "WeirdOpDeviceOperation 4 x 4",
                "kernel_kind": "dtype",
                "measured_ms": 55.0,
                "beat_baseline": True,
                "claimed_beat_baseline": True,
                "note": "kept: 60->55",
                "fullpipe_ms": 80.0,
                "fullpipe_best_ms": 100.0,
                "commit": "a3f9c21deadbeef",
            },
            {
                "op_signature": "OtherOp",
                "kernel_kind": "grid",
                "measured_ms": 10.0,
                "beat_baseline": False,
                "claimed_beat_baseline": True,
            },
            {"op_signature": "ThirdOp", "kernel_kind": "trace", "measured_ms": 9.0, "wedged": True},
        ],
    )
    return repo, run_dir, state, slug


def test_collect_discovers_stages_and_metrics_from_data(tmp_path):
    repo, run_dir, state, slug = _make_run(tmp_path)
    s = collect_state(run_dir, [state], slug)
    names = [st["name"] for st in s["stages"]]
    assert names == ["lm_head", "audio_encode"]  # hottest first, names straight from the file
    assert s["metric"]["current"] == 80.0
    assert s["fullpipe_ms"] == 30.0
    # unit "token" -> tok/s from the ledger's committed after-row (25 ms), not the baseline file
    assert s["throughput"]["unit"] == "tok/s"
    assert abs(s["throughput"]["current"] - 40.0) < 1e-6
    assert abs(s["throughput"]["baseline"] - (1000.0 / 30.0)) < 1e-6


def test_attempts_status_and_opportunity_matching(tmp_path):
    repo, run_dir, state, slug = _make_run(tmp_path)
    s = collect_state(run_dir, [state], slug)
    by_op = {a["op"]: a for a in s["attempts"]}
    assert by_op["WeirdOpDeviceOperation 4 x 4"]["status"] == "kept"
    assert by_op["OtherOp"]["status"] == "reverted"
    assert by_op["ThirdOp"]["status"] == "wedged"
    opp = {o["id"]: o for o in s["opportunities"]}["weirdop"]
    assert opp["tried_rungs"] == ["dtype"]
    assert opp["status"] == "cleared"


def test_history_row_carries_before_after_delta_and_commit(tmp_path):
    """The history table's Before/After/Δ%/Commit all come straight from the attempt record — the
    same kernel-log row RUN_REPORT.md renders, plus the sha git_commit now stamps on the win."""
    repo, run_dir, state, slug = _make_run(tmp_path)
    s = collect_state(run_dir, [state], slug)
    kept = next(a for a in s["attempts"] if a["status"] == "kept")
    assert kept["before_ms"] == 100.0 and kept["after_ms"] == 80.0
    assert abs(kept["delta_pct"] - (-20.0)) < 1e-9
    assert kept["commit"] == "a3f9c21deadbeef"
    # an attempt with no end-to-end of its own must not invent a delta
    plain = next(a for a in s["attempts"] if a["op"] == "OtherOp")
    assert plain["delta_pct"] is None and plain["commit"] is None


def test_attempt_status_words():
    assert _attempt_status({"beat_baseline": True}) == "kept"
    assert _attempt_status({"claimed_beat_baseline": True}) == "reverted"
    assert _attempt_status({}) == "no-gain"
    assert _attempt_status({"wedged": True}) == "wedged"
    assert _attempt_status({"measurement_failed": True}) == "wedged"


def test_hitl_proposal_is_peeked_not_consumed(tmp_path):
    repo, run_dir, state, slug = _make_run(tmp_path)
    _write(run_dir / "hitl_proposal.json", {"tried": {"lever": "dtype", "op": "WeirdOp"}, "step": 2})
    s = collect_state(run_dir, [state], slug)
    assert s["hitl_proposal"]["tried"]["lever"] == "dtype"
    assert (run_dir / "hitl_proposal.json").is_file(), "the orchestrator's read consumes — the dashboard's must not"


def test_find_run_dir_latest_and_slug_filter(tmp_path):
    repo = tmp_path / "repo"
    runs = repo / "models/experimental/perf_automation/runs"
    _write(runs / "2026-08-30T10-00-00-model_a" / "state.json", {})
    _write(runs / "2026-08-30T11-00-00-model_b" / "state.json", {})
    assert find_run_dir(repo).name.endswith("model_b")
    assert find_run_dir(repo, slug="model_a").name.endswith("model_a")
    assert find_run_dir(repo, run_ref="2026-08-30T10-00-00-model_a").name.endswith("model_a")
    assert find_run_dir(repo, slug="nope") is None
    assert run_slug(runs / "2026-08-30T10-00-00-model_a") == "model_a"


def test_missing_files_yield_empty_sections(tmp_path):
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    s = collect_state(run_dir, [tmp_path / "no_state"], None)
    assert s["stages"] == [] and s["attempts"] == [] and s["opportunities"] == []
    assert s["metric"] is None and s["run"]["live"] is False


def test_state_dir_candidates_prefers_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path / "envd"))
    cands = state_dir_candidates(tmp_path / "repo", "slug")
    assert cands[0] == tmp_path / "envd"
    assert (tmp_path / "repo" / "models/experimental/perf_automation/.state/slug") in cands


def test_repo_root_for_run_derives_the_owning_checkout(tmp_path):
    other = tmp_path / "other_checkout"
    run_dir = other / "models/experimental/perf_automation/runs/2026-08-30T10-00-00-x"
    run_dir.mkdir(parents=True)
    assert repo_root_for_run(run_dir, tmp_path / "cwd_repo") == other
    # a run dir NOT under the standard layout keeps the caller's default
    odd = tmp_path / "elsewhere"
    odd.mkdir()
    assert repo_root_for_run(odd, tmp_path / "cwd_repo") == tmp_path / "cwd_repo"


def test_server_serves_html_and_state(tmp_path):
    repo, run_dir, state, slug = _make_run(tmp_path)
    srv = make_server("127.0.0.1", 0, lambda: collect_state(run_dir, [state], slug))
    import threading

    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        port = srv.server_address[1]
        html = urllib.request.urlopen("http://127.0.0.1:%d/" % port, timeout=5).read().decode()
        assert "Optimization Opportunities" in html
        payload = json.loads(urllib.request.urlopen("http://127.0.0.1:%d/api/state" % port, timeout=5).read())
        assert payload["model"]["slug"] == slug
        assert [st["name"] for st in payload["stages"]] == ["lm_head", "audio_encode"]
    finally:
        srv.shutdown()
        srv.server_close()
