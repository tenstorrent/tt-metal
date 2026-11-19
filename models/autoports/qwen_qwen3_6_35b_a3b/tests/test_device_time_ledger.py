# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regression controls for direct-job ownership versus quoted transcript time."""

import tempfile
import unittest
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import summarize_fused_device_time as ledger


def marker(stamp, action):
    return f"2026-11-13 {stamp} | info | Device | {action} (tt_cluster.cpp:252)\n"


class DeviceTimeLedgerTest(unittest.TestCase):
    def parse(self, text, name="direct_job.log", after="2026-11-13T00:00:00+00:00"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_text(text)
            return ledger.parse_log(path, name, ledger.utc(after))

    def test_cross_job_codex_excerpts_never_form_session(self):
        text = "OpenAI Codex v0.163.0\nsession id: example\n"
        text += marker("03:14:26.399", ledger.OPEN)
        text += "...separate tool read, different job...\n"
        text += marker("04:45:47.248", ledger.CLOSE)
        sessions, issues, count, _ = self.parse(text, "investigation.log")
        self.assertEqual(sessions, [])
        self.assertEqual(count, 2)
        self.assertEqual(issues[0]["classification"], "quoted_transcript")

    def test_launcher_and_queue_names_require_direct_evidence(self):
        text = marker("03:00:00", ledger.OPEN) + marker("03:00:10", ledger.CLOSE)
        for name in ("launcher.log", "autodebug_launcher_retry.log", "validation_queue.log"):
            with self.subTest(name=name):
                sessions, issues, _, _ = self.parse(text, name)
                self.assertEqual(sessions, [])
                self.assertEqual(issues[0]["classification"], "quoted_transcript")

    def test_renamed_tail_excerpt_is_still_excluded(self):
        text = "--- last 8 lines of first_job.log ---\n" + marker("03:00:00", ledger.OPEN)
        text += "--- last 8 lines of second_job.log ---\n" + marker("04:00:00", ledger.CLOSE)
        sessions, issues, _, _ = self.parse(text, "summary.log")
        self.assertFalse(sessions)
        self.assertEqual(issues[0]["classification"], "quoted_transcript")

    def test_direct_jobs_are_clipped_and_union_deduplicated(self):
        a, _, _, _ = self.parse(
            marker("03:00:00", ledger.OPEN) + marker("03:00:20", ledger.CLOSE),
            after="2026-11-13T03:00:05+00:00",
        )
        b, _, _, _ = self.parse(marker("03:00:10", ledger.OPEN) + marker("03:00:30", ledger.CLOSE))
        self.assertTrue(a[0]["clipped_to_first_candidate"])
        self.assertEqual(a[0]["seconds"], 15)
        self.assertEqual(sum(x["seconds"] for x in ledger.union_intervals(a + b + a)), 25)

    def test_live_abort_and_ambiguous_job_remain_unaccounted(self):
        cases = (
            marker("03:00:00", ledger.OPEN),
            marker("03:00:00", ledger.OPEN) + "Aborted (core dumped)\n",
            marker("03:00:00", ledger.OPEN) + marker("03:00:01", ledger.OPEN) + marker("03:00:10", ledger.CLOSE),
        )
        for text in cases:
            with self.subTest(text=text):
                sessions, issues, _, _ = self.parse(text)
                self.assertFalse(sessions)
                self.assertTrue(issues)

    def test_complete_direct_jobs_not_replaced_by_excerpt_duration(self):
        first, _, _, _ = self.parse(marker("03:14:26.399", ledger.OPEN) + marker("03:14:57.527", ledger.CLOSE))
        last, _, _, _ = self.parse(marker("04:45:21.291", ledger.OPEN) + marker("04:45:47.248", ledger.CLOSE))
        quoted, _, _, _ = self.parse(
            "OpenAI Codex v0.163.0\n" + marker("03:14:26.399", ledger.OPEN) + marker("04:45:47.248", ledger.CLOSE)
        )
        self.assertAlmostEqual(sum(x["seconds"] for x in ledger.union_intervals(first + last + quoted)), 57.085)


if __name__ == "__main__":
    unittest.main()
