# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Cost attribution tests use explicit identities, without accessing real sessions."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "session_cost", Path(__file__).with_name("session_cost.py")
)
cost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cost)


def turn(mid, output=100, model="claude-opus-5", **usage):
    return {
        "type": "assistant",
        "timestamp": "2026-09-21T10:00:00Z",
        "message": {
            "id": mid,
            "model": model,
            "usage": {"output_tokens": output, **usage},
        },
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def tree(tmp_path, sid, rows, kind="main"):
    path = write_jsonl(tmp_path / "project" / f"{sid}.jsonl", rows)
    return sid, path, path.parent / sid / "subagents", kind


def registry(tmp_path, sessions, run_id="run-1"):
    (tmp_path / "run.json").write_text(json.dumps({"run_id": "run-1"}))
    (tmp_path / "session_registry.json").write_text(
        json.dumps(
            {
                "schema": "issue-solver.session-registry",
                "version": 1,
                "run_id": run_id,
                "sessions": sessions,
            }
        )
    )


def link(sid="child", parent="root"):
    return {
        "session_id": sid,
        "parent_session_id": parent,
        "project_cwd": "/project/child",
        "kind": "autodebug",
    }


def test_message_ids_deduplicate_streaming_and_subagents(tmp_path):
    root = tree(tmp_path, "root", [turn("msg-1", 100), turn("msg-1", 250)])
    write_jsonl(root[2] / "agent-a.jsonl", [turn("msg-1", 200), turn("msg-2", 10)])
    totals = cost._aggregate_roots([root], None, None, None, None)
    assert totals["output"] == 260
    assert totals["cost_usd"] == pytest.approx(260 * 25 / 1e6)


def test_actual_sonnet5_and_one_hour_cache_override_fallback_model(tmp_path):
    root = tree(
        tmp_path,
        "root",
        [
            turn(
                "m",
                100,
                "claude-sonnet-5",
                input_tokens=10,
                cache_read_input_tokens=1000,
                cache_creation={
                    "ephemeral_5m_input_tokens": 100,
                    "ephemeral_1h_input_tokens": 200,
                },
            )
        ],
    )
    totals = cost._aggregate_roots([root], None, "opus", None, None)
    assert totals["cost_usd"] == pytest.approx((20 + 1000 + 200 + 250 + 800) / 1e6)
    assert totals["cache_creation"] == 300
    assert cost._tier("claude-sonnet-4-6") == "sonnet"


def test_native_root_plus_transcript_child_is_added_once(tmp_path):
    root = tree(tmp_path, "root", [turn("root-msg", 100)])
    child = tree(tmp_path, "child", [turn("child-msg", 200)], "autodebug")
    sink = write_jsonl(tmp_path / "otel.jsonl", [{"session_id": "root", "cost_usd": 2}])
    totals = cost._aggregate_roots([root, child, child], None, None, str(sink), None)
    assert totals["cost_usd"] == 2.005
    assert totals["output"] == 300
    assert [s["source"] for s in totals["cost_accounting"]["sessions"]] == [
        "otel",
        "transcript",
    ]


def test_native_each_root_and_since_filter(tmp_path):
    root = tree(tmp_path, "root", [turn("m", 100)])
    child = tree(tmp_path, "child", [turn("n", 200)], "autodebug")
    since = cost._parse_ts("2026-09-21T09:00:00Z")
    nano = int(since.timestamp()) * 10**9
    sink = write_jsonl(
        tmp_path / "otel.jsonl",
        [
            {"session_id": "root", "cost_usd": 50, "ts": str(nano - 1)},
            {"session_id": "root", "cost_usd": 2, "ts": str(nano)},
            {"session_id": "child", "cost_usd": 3, "ts": str(nano)},
            {"session_id": "unrelated", "cost_usd": 100, "ts": str(nano)},
        ],
    )
    totals = cost._aggregate_roots([root, child], since, None, str(sink), None)
    assert totals["cost_usd"] == 5


def test_cli_root_does_not_erase_linked_cost(tmp_path):
    root = tree(tmp_path, "root", [turn("m", 100)])
    child = tree(tmp_path, "child", [turn("n", 200)], "autodebug")
    (tmp_path / "cli_output.json").write_text('{"total_cost_usd": 2}')
    totals = cost._aggregate_roots([root, child], None, None, None, tmp_path)
    assert totals["cost_usd"] == 2.005


def test_cross_tree_duplicate_prefers_native_covered_copy(tmp_path):
    root = tree(tmp_path, "root", [turn("shared", 100)])
    child = tree(tmp_path, "child", [turn("shared", 200), turn("new", 20)], "autodebug")
    sink = write_jsonl(
        tmp_path / "otel.jsonl", [{"session_id": "child", "cost_usd": 3}]
    )
    totals = cost._aggregate_roots([root, child], None, None, str(sink), None)
    assert totals["cost_usd"] == 3
    assert totals["output"] == 220


def test_registry_deduplicates_root_and_explicit_links(tmp_path):
    registry(
        tmp_path, [{"session_id": "root"}, link(), link(), link("grandchild", "child")]
    )
    assert [e["session_id"] for e in cost._linked_sessions(tmp_path, "root")] == [
        "child",
        "grandchild",
    ]


@pytest.mark.parametrize(
    "sessions,run_id",
    [
        ([link()], "other-run"),
        ([link(parent="unrelated")], "run-1"),
        ([link(parent="child")], "run-1"),
        ([link("../escape")], "run-1"),
        ([link(), {**link(), "project_cwd": "/different"}], "run-1"),
    ],
)
def test_registry_rejects_misattribution(tmp_path, sessions, run_id):
    registry(tmp_path, sessions, run_id)
    with pytest.raises(ValueError):
        cost._linked_sessions(tmp_path, "root")


def test_missing_registered_child_is_explicitly_incomplete(tmp_path):
    root = tree(tmp_path, "root", [turn("m", 100)])
    missing = ("absent", tmp_path / "absent.jsonl", tmp_path / "subs", "autodebug")
    totals = cost._aggregate_roots([root, missing], None, None, None, None)
    assert not totals["cost_accounting"]["complete"]
    assert totals["cost_accounting"]["sessions"][1]["source"] == "missing"
    assert totals["cost_usd"] == 0.0025


def test_explicit_missing_session_never_discovers_another(monkeypatch, capsys):
    monkeypatch.setattr(cost, "_find_by_session_id", lambda sid: None)
    monkeypatch.setattr(
        cost, "_build_paths", lambda sid, cwd: (Path("/missing/a"), Path("/missing/b"))
    )
    monkeypatch.setattr(
        cost, "_discover_session", lambda pid: pytest.fail("must not discover")
    )
    assert cost.main(["--session-id", "absent", "--otel-sink", "/missing/sink"]) == 0
    totals = json.loads(capsys.readouterr().out)
    assert not totals["cost_accounting"]["complete"]


def test_cli_loads_registry_and_patches_accounting(monkeypatch, tmp_path, capsys):
    main = tree(tmp_path, "root", [turn("m", 100)])
    child = tree(tmp_path, "child", [turn("n", 200)], "autodebug")
    registry(tmp_path, [link()])
    monkeypatch.setattr(cost, "_find_by_session_id", lambda sid: main[1:3])
    monkeypatch.setattr(cost, "_build_paths", lambda sid, cwd: child[1:3])
    assert (
        cost.main(
            [
                "--session-id",
                "root",
                "--log-dir",
                str(tmp_path),
                "--otel-sink",
                "/missing/sink",
            ]
        )
        == 0
    )
    totals = json.loads(capsys.readouterr().out)
    patched = json.loads((tmp_path / "run.json").read_text())
    assert patched["cost_usd"] == totals["cost_usd"] == 0.0075
    assert len(patched["cost_accounting"]["sessions"]) == 2
