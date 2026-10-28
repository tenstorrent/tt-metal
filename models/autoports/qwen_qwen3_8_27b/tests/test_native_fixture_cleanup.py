# SPDX-License-Identifier: Apache-2.0
"""CPU regression coverage for release/close failures and pass-artifact ordering."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("cleanup", Path(__file__).with_name("native_fixture_cleanup.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CleanupTest(unittest.TestCase):
    def test_success_is_saved_only_after_close_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "result.json"
            calls = []

            def close():
                self.assertFalse(target.exists())
                calls.append("close")

            result = dict(status="pass")
            module.finalize_run(result, target, lambda: calls.append("release"), close)
            self.assertEqual(calls, ["release", "close"])
            self.assertEqual(json.loads(target.read_text())["status"], "pass")
            self.assertTrue(result["close_mesh_device_returned"])

    def test_release_failure_still_closes_and_never_saves_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "result.json"
            calls = []

            def release():
                raise RuntimeError("release failed")

            with self.assertRaisesRegex(RuntimeError, "release failed"):
                module.finalize_run(dict(status="pass"), target, release, lambda: calls.append("closed"))
            self.assertEqual(calls, ["closed"])
            self.assertEqual(json.loads(target.read_text())["status"], "fail")

    def test_close_failure_is_a_failed_artifact_and_process_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "result.json"

            def close():
                raise RuntimeError("close failed")

            with self.assertRaisesRegex(RuntimeError, "close failed"):
                module.finalize_run(dict(status="pass"), target, lambda: None, close)
            result = json.loads(target.read_text())
            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["close_mesh_device_returned"])

    def test_cleanup_does_not_mask_original_body_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "result.json"

            def release():
                raise RuntimeError("cleanup failed")

            with self.assertRaisesRegex(ValueError, "body failed"):
                try:
                    raise ValueError("body failed")
                finally:
                    module.finalize_run(dict(status="fail"), target, release, lambda: None)
            self.assertEqual(json.loads(target.read_text())["status"], "fail")


if __name__ == "__main__":
    unittest.main()
