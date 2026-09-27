# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Source-derived binding state regression; run directly without device imports."""

import unittest

from benchmark_roofline import decode_execution_counts


def event(sid, phase, ids, device=True):
    return dict(
        submission_id=str(sid),
        timestamp_ns=sid * 100,
        event="dispatch",
        phase=phase,
        request_ids=ids,
        device_sampling=device,
        batch_slots=len(ids),
        positions=[4096] * len(ids),
    )


class WarmAccountingTests(unittest.TestCase):
    def test_prefill_anchor_rebind_and_unchanged_feedback(self):
        events = [
            event(1, "prefill", ["a", "b"]),
            event(2, "decode", ["a", "b"]),
            event(3, "decode", ["a", "b"]),
            event(4, "decode", ["b"]),
            event(5, "decode", ["b"]),
        ]
        self.assertEqual(decode_execution_counts(events, ["a", "b"]), {"2": 2, "3": 1, "4": 2, "5": 1})

    def test_reprefill_invalidates_same_size_trace(self):
        events = [
            event(1, "prefill", ["warm"]),
            event(2, "decode", ["warm"]),
            event(3, "prefill", ["actual"]),
            event(4, "decode", ["actual"]),
        ]
        self.assertEqual(decode_execution_counts(events, ["actual"]), {"4": 2})

    def test_unanchored_stream_fails(self):
        with self.assertRaises(ValueError):
            decode_execution_counts([event(1, "decode", ["a"])], ["a"])

    def test_capture_is_not_third_execution(self):
        prefill, decode = event(1, "prefill", ["a"]), event(2, "decode", ["a"])
        prefill["submission_id"], decode["submission_id"] = "opaque:z", "opaque:a"
        result = decode_execution_counts([decode, prefill], ["a"])
        self.assertEqual(result["opaque:a"], 2)


if __name__ == "__main__":
    unittest.main()
