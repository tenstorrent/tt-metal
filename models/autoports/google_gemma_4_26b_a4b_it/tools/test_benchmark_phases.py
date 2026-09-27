# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run directly with Python to avoid hardware repository conftest loading."""

import unittest

from benchmark_phases import reduce_phases


def pair(sid, phase, start, end, ids=("measured",)):
    return [
        dict(submission_id=sid, phase=phase, timestamp_ns=timestamp, event=event, request_ids=list(ids))
        for event, timestamp in (("dispatch", start), ("completion", end))
    ]


class PhaseReducerTests(unittest.TestCase):
    def test_async_overlap_counted_once_and_gap_included(self):
        events = pair("a", "decode", 10, 30) + pair("b", "decode", 20, 40) + pair("c", "decode", 60, 70)
        result = reduce_phases(events, ["measured"])
        self.assertEqual(result["elapsed_ns"], 60)
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["duration_ns"], 60)

    def test_transition_gap_goes_to_following_phase(self):
        events = pair("a", "prefill", 10, 20) + pair("b", "decode", 30, 50)
        result = reduce_phases(events, ["measured"])
        self.assertEqual([s["duration_ns"] for s in result["segments"]], [10, 30])
        self.assertEqual(result["segments"][1]["first_dispatch_ns"], 30)
        self.assertEqual(sum(s["duration_ns"] for s in result["segments"]), result["elapsed_ns"])

    def test_warmup_excluded_and_unordered_events_supported(self):
        events = pair("warmup", "prefill", 0, 5, ("probe",)) + pair("a", "prefill", 10, 20)
        result = reduce_phases(list(reversed(events)), ["measured"])
        self.assertEqual(result["excluded_submission_ids"], ["warmup"])
        self.assertEqual(result["elapsed_ns"], 10)

    def test_invalid_evidence_rejected(self):
        valid = pair("a", "prefill", 10, 20)
        cases = {
            "missing completion": valid[:1],
            "duplicate completion": valid + valid[1:],
            "wrong request coverage": pair("a", "prefill", 10, 20, ("other",)),
            "mixed batch": pair("a", "prefill", 10, 20, ("measured", "other")),
            "unrelated in gap": valid + pair("b", "decode", 40, 50) + pair("x", "decode", 25, 30, ("other",)),
            "cross phase overlap": valid + pair("b", "decode", 15, 30),
            "nested cross phase overlap": pair("a", "prefill", 10, 50)
            + pair("b", "prefill", 20, 30)
            + pair("c", "decode", 40, 60),
            "invalid interval": pair("a", "prefill", 20, 20),
            "metadata changed": [valid[0], dict(valid[1], request_ids=["other"])],
        }
        for name, events in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                reduce_phases(events, ["measured"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
