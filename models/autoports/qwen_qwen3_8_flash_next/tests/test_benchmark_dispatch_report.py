# SPDX-License-Identifier: Apache-2.0
"""Host-only report provenance checks for completed and interrupted dispatch."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "doc/benchmark/finalize_report.py"
SPEC = importlib.util.spec_from_file_location("dispatch_report", SOURCE)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class DispatchReportTests(unittest.TestCase):
    def test_actual_algorithm_and_completion_control_report(self):
        for algorithm, phrase in (
            ("stable_longest_estimated_completion_first_v1", "descending"),
            ("stable_round_robin_by_task_first_appearance_v1", "round-robin"),
        ):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                row = dict(event="dispatch", algorithm=algorithm, requests=384, runtime_estimates_sha256="test-hash")
                journal = root / "request_dispatch.jsonl"
                journal.write_text(json.dumps(row) + "\n")
                text = REPORT.dispatch_protocol(root)
                self.assertIn(phrase, text)
                self.assertIn("No complete inverse-mapping event", text)
                with journal.open("a") as stream:
                    stream.write(
                        json.dumps(
                            dict(event="restored", algorithm=algorithm, responses=384, original_order_restored=True)
                        )
                        + "\n"
                    )
                self.assertIn("completion journal confirms", REPORT.dispatch_protocol(root))


if __name__ == "__main__":
    unittest.main(verbosity=2)
