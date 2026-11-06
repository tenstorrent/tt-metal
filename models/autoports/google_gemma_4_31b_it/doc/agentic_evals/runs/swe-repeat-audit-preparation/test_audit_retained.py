import importlib.util
import json
import pathlib
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("helper", pathlib.Path(__file__).with_name("audit_retained.py"))
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class AuditTests(unittest.TestCase):
    def test_network_candidate_not_plain_url(self):
        self.assertEqual(helper.network_tools("echo https://example.test/license"), [])
        self.assertEqual(helper.network_tools('sed -n "https://example.test" source.py'), [])
        self.assertEqual(helper.network_tools("curl -I https://example.test"), ["curl"])

    def test_wilson_zero_not_zero_width(self):
        lo, hi = helper.wilson(0, 2)
        self.assertAlmostEqual(lo, 0)
        self.assertAlmostEqual(hi, 0.6576197724933469)

    def test_partial_run_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "result.json").write_text("{}")
            with self.assertRaises(ValueError):
                helper.audit(root)

    def test_retention_metadata_and_no_raw_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "agent").mkdir()
            trial = {"task_name": "synthetic", "verifier_result": {"rewards": {"reward": 0}}}
            response = {
                "choices": [{"finish_reason": "length", "provider_specific_fields": {"token_ids": [1, 2]}}],
                "usage": {"completion_tokens": 2, "prompt_tokens": 10},
            }
            trajectory = {
                "info": {"model_stats": {"api_calls": 2}},
                "messages": [
                    {"role": "assistant", "content": "PRIVATE_SENTINEL", "extra": {"response": response}},
                    {"role": "user", "extra": {"interrupt_type": "FormatError"}},
                ],
            }
            (root / "result.json").write_text(json.dumps(trial))
            (root / "agent/mini-swe-agent.trajectory.json").write_text(json.dumps(trajectory))
            result = helper.audit_trial(root / "result.json", 2)
            self.assertEqual(result["missing_response_count"], 1)
            self.assertEqual(result["token_id_covered_responses"], 1)
            self.assertEqual(result["responses"][0]["token_ids_match_usage"], True)
            self.assertEqual(result["cap_candidates"], [0])
            self.assertFalse(result["format_errors"][0]["failed_response_retained"])
            self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
            self.assertNotIn('"token_ids":', json.dumps(result))


if __name__ == "__main__":
    unittest.main()
