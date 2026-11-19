# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tt/benchmark_phases.py"
spec = importlib.util.spec_from_file_location("benchmark_phases_cpu", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fifo_completion_and_host_boundary(tmp_path):
    complete = set()
    recorder = module.PhaseRecorder(
        None, tmp_path / "events.json", query=lambda event: event in complete, record=lambda: 0, start_thread=False
    )
    metadata = {"slots": [0]}
    prefill = recorder.begin("prefill", metadata)
    metadata["slots"].append(1)
    recorder.end(prefill)
    decode = recorder.begin("decode", {"slots": [0], "positions": [4096]})
    recorder.end(decode, event=1)
    recorder._poll_once()
    assert recorder.snapshot()["pending_tickets"] == [0, 1]
    complete.add(0)
    recorder._poll_once()
    recorder.host_complete(prefill)
    assert recorder.snapshot()["pending_tickets"] == [1]
    complete.add(1)
    recorder._poll_once()
    recorder.host_complete(decode)
    snapshot = recorder.flush()
    assert snapshot["pending_tickets"] == []
    assert snapshot["events"][0]["metadata"] == {"slots": [0]}
    for row in snapshot["events"]:
        assert row["begin_ns"] <= row["completion_lower_ns"] <= row["completion_observed_ns"]
        assert row["completion_observed_ns"] <= row["host_complete_ns"]
    assert json.loads((tmp_path / "events.json").read_text()) == snapshot


def test_query_failure_is_preserved(tmp_path):
    def query(_):
        raise RuntimeError("failed completion reader")

    recorder = module.PhaseRecorder(None, tmp_path / "events.json", query=query, record=lambda: 0, start_thread=False)
    recorder.end(recorder.begin("prefill", {}))
    recorder._observe()
    # Failures must be retained without a serving shutdown or explicit close.
    retained = json.loads(recorder.log_path.read_text())
    assert "failed completion reader" in retained["errors"][0]["error"]
    snapshot = recorder.close()
    assert snapshot["pending_tickets"] == [0]
    assert "failed completion reader" in snapshot["errors"][0]["error"]


def test_export_control_is_post_measurement(tmp_path):
    recorder = module.PhaseRecorder(
        None, tmp_path / "events.json", query=lambda _: True, record=lambda: 0, start_thread=False
    )
    recorder.end(recorder.begin("decode", {}), event=4)
    recorder._poll_once()
    assert not (tmp_path / "events.json").exists()
    recorder.close()
    assert (tmp_path / "events.json").exists()


def test_annotate_freezes_completed_chunks(tmp_path):
    recorder = module.PhaseRecorder(
        None, tmp_path / "events.json", query=lambda _: True, record=lambda: 0, start_thread=False
    )
    ticket = recorder.begin("prefill", {"chunks": []})
    chunks = [{"start": 0, "end": 8}]
    recorder.annotate(ticket, chunks=chunks)
    chunks[0]["end"] = 16
    assert recorder.snapshot()["events"][0]["metadata"]["chunks"] == [{"start": 0, "end": 8}]


def test_background_flush_marker_exports_without_device_wait(tmp_path):
    path = tmp_path / "events.json"
    recorder = module.PhaseRecorder(None, path, query=lambda _: True, record=lambda: 0)
    recorder.end(recorder.begin("prefill", {}))
    recorder.flush_path.touch()
    deadline = time.monotonic() + 2
    while recorder.flush_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    recorder.close()
    assert not recorder.flush_path.exists()
    snapshot = json.loads(path.read_text())
    assert not snapshot["pending_tickets"] and not snapshot["errors"]
