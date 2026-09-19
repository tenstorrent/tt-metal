# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import json
import urllib.request

from scripts.tt_hw_planner.optimize_dashboard import (
    _attempt_status,
    _serving_metrics,
    collect_state,
    find_run_dir,
    make_server,
    post_hitl_decision,
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
    _write(
        run_dir / "manifest.json",
        {
            "config": {"model_root": "/x/" + slug, "metric": "device_ms", "devices": "0"},
            "env": {"arch": "testarch", "dram_bw_gbps": 512.0, "worker_cores": 130},
        },
    )
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
                    "top_ops": [
                        {"op_code": "WeirdOpDeviceOperation 4 x 4"},
                        {
                            "op_code": "ContractionOp 2 x 3 x 4",
                            "shape": "2x3 @ 3x4",
                            "device_ms": 2.0,
                            "bytes": 48.0,
                            "count": 5,
                        },
                    ],
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
                json.dumps({"kind": "peak_flops", "phase": "before", "value_ms": 175.5e12}),
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
                "stages": [
                    {"name": "lm_head", "ms": 80.0, "dominant": True},
                    {"name": "audio_encode", "ms": 20.0},
                ],
            },
            {
                "op_signature": "OtherOp",
                "kernel_kind": "grid",
                "measured_ms": 10.0,
                "beat_baseline": False,
                "claimed_beat_baseline": True,
                "stages": [
                    {"name": "lm_head", "ms": 82.0, "dominant": True},
                    {"name": "audio_encode", "ms": 14.0},
                ],
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


def test_state_dir_candidates_include_the_shared_root_above_the_model_dir(tmp_path, monkeypatch):
    """A --persist run points PERF_MCP_STATE_DIR at .state/<slug>, but the board-level profiles (thermal
    /power) are shared across models and sit in .state itself. Searching only the model dir left the
    Power Analysis tab empty while its data was on disk."""
    root = tmp_path / "perf" / ".state"
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(root / "some_model"))
    cands = state_dir_candidates(tmp_path / "repo", "some_model")
    assert cands[0] == root / "some_model"
    assert root in cands and cands.index(root) < len(cands) - 1

    _write(root / "perf_mcp_thermal_profile.json", {"clean_at": [61.0, 64.0], "clamped_at": [72.0]})
    repo, run_dir, state, slug = _make_run(tmp_path)
    s = collect_state(run_dir, [root / "some_model", root], slug)
    assert s["thermal"]["clamped_at"] == [72.0]


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
        assert "Optimization History" in html and "Recommendations" in html and "Performance Metrics" in html
        payload = json.loads(urllib.request.urlopen("http://127.0.0.1:%d/api/state" % port, timeout=5).read())
        assert payload["model"]["slug"] == slug
        assert [st["name"] for st in payload["stages"]] == ["lm_head", "audio_encode"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_serving_metrics_are_derived_from_values_not_names(tmp_path):
    """TTFT/TPOT/E2EL must come out of the numbers: the per-token stage is the one matching the
    banked per-token pipeline time, the first-token stage is the dominant one-shot — no stage NAME
    is ever consulted (the fixture's names are deliberately not prefill/decode)."""
    repo, run_dir, state, slug = _make_run(tmp_path)
    s = collect_state(run_dir, [state], slug)
    sv = s["serving"]
    assert sv["per_token"]["stage"] == "lm_head" and sv["per_token"]["ms"] == 30.0
    assert sv["first_token"]["stage"] == "audio_encode" and sv["first_token"]["ms"] == 12.0
    assert sv["e2e_latency"]["ms"] == 42.0
    assert abs(sv["throughput"]["per_s"] - (1000.0 / 30.0)) < 1e-6
    # headroom: ledger modeled_floor (none in this fixture) -> absent, never fabricated
    assert s["headroom"] is None


def test_roofline_points_and_roofs_from_run_anchors(tmp_path):
    """Chart points come from contraction shapes in the profile (2*M*K*N is definitional); the roofs
    come from the run's own anchors (manifest env bandwidth, ledger peak) — nothing restated."""
    repo, run_dir, state, slug = _make_run(tmp_path)
    s = collect_state(run_dir, [state], slug)
    rf = s["roofline"]
    assert rf["bw_gbps"] == 512.0
    assert abs(rf["peak_tflops"] - 175.5) < 1e-9
    assert len(rf["points"]) == 1  # the op without a contraction shape is not plotted
    p = rf["points"][0]
    flops = 2 * 2 * 3 * 4 * 5  # M*K*N per call, x count
    assert abs(p["intensity"] - flops / 48.0) < 1e-9
    assert abs(p["tflops"] - flops / 0.002 / 1e12) < 1e-6
    assert s["env"]["dram_bw_gbps"] == 512.0


def test_overlapping_archive_and_live_log_list_each_attempt_once(tmp_path):
    """The engine's _fold_cumulative COPIES live rows into the archive, so the two logs overlap by
    design. Listing both would double every attempt — 24 recorded rungs read as 48, the history chart
    would restart its best-so-far halfway through, and the status counts would be twice the truth.
    Collapse on the SAME identity key the engine's _load_attempts_all uses."""
    repo, run_dir, state, slug = _make_run(tmp_path)
    archive = json.loads((state / ("cc_kernlog_%s_main.json.cumulative" % slug)).read_text())
    # the live log holds the same rows the archive already folded in, plus one newer attempt
    fresh = {"op_signature": "FourthOp", "kernel_kind": "grid", "measured_ms": 7.0, "beat_baseline": True}
    _write(state / ("cc_kernlog_%s_main.json" % slug), archive + [fresh])

    s = collect_state(run_dir, [state], slug)
    ops = [a["op"] for a in s["attempts"]]
    assert len(ops) == len(set(ops)) == 4, ops
    assert "FourthOp" in ops

    # two DIFFERENT variants of one rung are not the same attempt and must both survive, each
    # spending its own retry (differing measurement, differing rationale).
    v1 = {"op_signature": "SameOp", "kernel_kind": "dtype", "measured_ms": 9.0, "note": "bf8_b"}
    v2 = {"op_signature": "SameOp", "kernel_kind": "dtype", "measured_ms": 8.0, "note": "bf4_b"}
    _write(state / ("cc_kernlog_%s_main.json" % slug), [v1, v2, dict(v1)])
    again = collect_state(run_dir, [state], slug)
    same = [a for a in again["attempts"] if a["op"] == "SameOp"]
    assert len(same) == 2, same


def test_attempts_carry_the_per_stage_timings_they_recorded(tmp_path):
    """Each stack gets its own history curve, and the only source for that is the per-stage timing list
    the agent passed to record_kernel_attempt. It must reach the payload per attempt and unaltered:
    the non-per-token stacks have no banked verdict of their own, so these readings are all there is."""
    repo, run_dir, state, slug = _make_run(tmp_path)
    s = collect_state(run_dir, [state], slug)
    by_op = {a["op"]: a for a in s["attempts"]}

    first = by_op["WeirdOpDeviceOperation 4 x 4"]["stages"]
    assert [(st["name"], st["ms"]) for st in first] == [("lm_head", 80.0), ("audio_encode", 20.0)]
    assert first[0]["dominant"] is True
    # a second reading of the SAME stack, so a per-stage trend has two points to draw between
    second = by_op["OtherOp"]["stages"]
    assert [(st["name"], st["ms"]) for st in second] == [("lm_head", 82.0), ("audio_encode", 14.0)]
    # an attempt that recorded none keeps an empty list rather than borrowing another attempt's
    assert by_op["ThirdOp"]["stages"] == []


def test_liveness_follows_the_agent_log_heartbeat(tmp_path):
    """A run is live while the AGENT is working. The run dir cannot answer that: state.json moves once
    per iteration and the attempt log only when a rung RESOLVES (minutes apart), and under --persist
    the run dir lives in a sandbox checkout the agent never touches. The agent log is the heartbeat."""
    import os
    import time

    repo, run_dir, state, slug = _make_run(tmp_path)
    stale = time.time() - 3600
    for p in (run_dir / "state.json", run_dir / "events.jsonl"):
        os.utime(p, (stale, stale))
    for p in state.glob("cc_kernlog_*"):
        os.utime(p, (stale, stale))
    assert collect_state(run_dir, [state], slug)["run"]["live"] is False

    _write(state / ("cc_kernlog_%s_main.json.agent.log" % slug), "working")
    s = collect_state(run_dir, [state], slug)
    assert s["run"]["live"] is True and s["run"]["age_s"] < 45.0


def test_hitl_decision_requires_a_pending_proposal(tmp_path):
    repo, run_dir, state, slug = _make_run(tmp_path)
    ok, msg = post_hitl_decision(run_dir, "commit")
    assert not ok and "no proposal" in msg
    assert not (run_dir / "hitl_decision.json").is_file()
    _write(run_dir / "hitl_proposal.json", {"tried": {"lever": "dtype", "op": "WeirdOp"}})
    ok, _ = post_hitl_decision(run_dir, "revert")
    assert ok
    dec = json.loads((run_dir / "hitl_decision.json").read_text())
    assert dec["action"] == "revert"
    assert (run_dir / "hitl_proposal.json").is_file(), "answering must not consume the proposal"
    ok, _ = post_hitl_decision(run_dir, "bogus")
    assert not ok
