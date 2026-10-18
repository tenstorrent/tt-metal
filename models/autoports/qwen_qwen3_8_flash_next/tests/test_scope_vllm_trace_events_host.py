# SPDX-License-Identifier: Apache-2.0
"""CPU-only negative/ownership checks; synthetic fixtures are not serving evidence."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name("scope_vllm_trace_events.py")
SPEC = importlib.util.spec_from_file_location("scope_vllm_trace_events", SOURCE)
SCOPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCOPER)


def line(kind, when, **kwargs):
    # Deliberately noncanonical spacing: the scoped output must retain these bytes.
    return (json.dumps(dict(event=kind, time=when, pid=9, **kwargs), separators=(", ", ": ")) + "\n").encode()


def fixture():
    rows = [
        line("preparation_memory", 0, memory={"allocated": 1024}),
        line("capture", 1, signature="decode", trace_id=12),
        line("prepared", 2, counters={"captures": 1, "decode_replays": 7}),
        line("decode_submit", 3, sampling="host"),
        line("prefill_schedule", 11),
        line("prefill_chunk", 12),
        line("prefill_complete", 13),
        line("decode_submit", 14, sampling="sample", counters={"decode_replays": 19}),
        line("replay", 15, signature="sample"),
        line("decode_submit", 30, sampling="host"),
        line("shutdown_release", 31, signature="decode", trace_id=12),
    ]
    client = dict(status="failed", completed_requests=1, requests=[dict(start=10, end=20)])
    return rows, client


class ScopeTests(unittest.TestCase):
    def test_exact_bytes_baseline_and_shutdown_survive(self):
        rows, client = fixture()
        source = b"".join(rows)
        view, manifest = SCOPER.scope(source, client)
        self.assertEqual(view, b"".join(rows[:3] + rows[4:9] + rows[10:]))
        self.assertEqual(manifest["omitted_counts"], {"decode_submit": 2})
        self.assertEqual(manifest["client_status"], "failed")
        self.assertIn(rows[2], view)  # baseline counters untouched
        self.assertIn(rows[7], view)  # workload counters not rebased
        rebuilt = b"".join(source[s["start_byte"] : s["end_byte"]] for s in manifest["retained_line_byte_spans"])
        self.assertEqual(rebuilt, view)
        self.assertEqual(manifest["source_sha256"], SCOPER.sha(source))

    def test_outside_lifecycle_and_unknown_events_never_hidden(self):
        rows, client = fixture()
        extra = [
            line(k, 5) for k in ("capture", "retire", "invalidation", "evict", "recapture", "release", "new_warning")
        ]
        view, manifest = SCOPER.scope(b"".join(rows[:4] + extra + rows[4:]), client)
        for raw in extra:
            self.assertIn(raw, view)
        self.assertEqual(manifest["retained_counts"]["capture"], 2)

    def test_byte_markers_cannot_hide_in_window_events(self):
        rows, client = fixture()
        source = b"".join(rows)
        start = sum(map(len, rows[:4]))
        end = sum(map(len, rows[:9]))
        view, _ = SCOPER.scope(source, client, start_byte=start, end_byte=end)
        self.assertIn(rows[1], view)
        self.assertIn(rows[-1], view)
        with self.assertRaisesRegex(ValueError, "excludes an event"):
            SCOPER.scope(source, client, start_byte=sum(map(len, rows[:5])))
        with self.assertRaisesRegex(ValueError, "record boundary"):
            SCOPER.scope(source, client, end_byte=end - 1)

    def test_partial_snapshot_invalid_clock_and_wrong_pid_fail(self):
        rows, client = fixture()
        source = b"".join(rows)
        with self.assertRaisesRegex(ValueError, "complete newline"):
            SCOPER.scope(source[:-1], client)
        with self.assertRaisesRegex(ValueError, "entire time interval"):
            SCOPER.scope(source, client, start_time=12)
        with self.assertRaisesRegex(ValueError, "multiple server"):
            SCOPER.scope(source + b'{"event":"release","time":32,"pid":10}\n', client)
        changed = copy.deepcopy(client)
        changed["requests"][0]["end"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            SCOPER.scope(source, changed)

    def test_wider_marked_window_keeps_request_events_before_first_http(self):
        rows, client = fixture()
        view, manifest = SCOPER.scope(b"".join(rows), client, start_time=3, end_time=30)
        self.assertEqual(view, b"".join(rows))
        self.assertEqual(manifest["omitted_records"], 0)


if __name__ == "__main__":
    unittest.main()
