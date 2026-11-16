import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

DRIVER = Path(__file__).with_name("run_candidate.py")
spec = importlib.util.spec_from_file_location("requests5414_cap1024_executor", DRIVER)
executor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor)


class CandidateExecutorTests(unittest.TestCase):
    def mock_case(self):
        return {"prompt_token_ids": list(range(2048)), "continuations": {"chosen": list(range(64))}}

    def mock_output(self, directory, generated_ids, case, case_path):
        result = {
            "status": "complete",
            "case": str(case_path),
            "case_sha256": executor.sha(case_path),
            "prompt_source": "Saved prompt_token_ids, no chat rerender",
            "prompt_tokens": 2048,
            "revision": executor.REVISION,
            "device": "cpu",
            "weights_dtype": "bfloat16",
            "prefill_chunk": 512,
            "case_metadata": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}},
            "results": [
                {
                    "seed": 44,
                    "requested_tokens": 1024,
                    "generated_tokens": len(generated_ids),
                    "generated_token_ids": generated_ids,
                }
            ],
            "teacher_forced_chosen": {
                "positions": [
                    {"position": i, "forced_token": token} for i, token in enumerate(case["continuations"]["chosen"])
                ]
            },
        }
        (Path(directory) / (executor.CASE_NAME + ".hf.private.json")).write_text(json.dumps(result))

    def test_static_candidate_pins_pass_without_admission_or_runner(self):
        spec = executor.candidate_static()
        self.assertEqual(spec["commands"]["commands"][0]["cap"], 1024)
        self.assertEqual(spec["prior_token_ids"], spec["prior_result"]["results"][0]["generated_token_ids"])
        self.assertFalse(executor.ADMISSION.exists())
        self.assertFalse(executor.OUTPUT.exists())

    def test_output_requires_exact_first512_generated_ids(self):
        case = self.mock_case()
        with tempfile.TemporaryDirectory() as td:
            case_path = Path(td) / "case.json"
            case_path.write_text(json.dumps(case))
            output = Path(td) / "out"
            output.mkdir()
            prior = list(range(512))
            self.mock_output(output, prior + list(range(512, 1024)), case, case_path)
            valid = executor.validate_output(output, case, case_path, prior)
            self.assertTrue(valid["prior_512_prefix_match"])
            self.assertEqual(valid["extension_tokens"], 512)
            result_path = output / (executor.CASE_NAME + ".hf.private.json")
            result = json.loads(result_path.read_text())
            result["results"][0]["generated_token_ids"][511] = -1
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(RuntimeError, "prefix differs"):
                executor.validate_output(output, case, case_path, prior)

    def test_output_rejects_wrong_case_sampling_seed_or_cap(self):
        case = self.mock_case()
        with tempfile.TemporaryDirectory() as td:
            case_path = Path(td) / "case.json"
            case_path.write_text(json.dumps(case))
            output = Path(td) / "out"
            output.mkdir()
            self.mock_output(output, list(range(1024)), case, case_path)
            result_path = output / (executor.CASE_NAME + ".hf.private.json")
            result = json.loads(result_path.read_text())
            result["case_metadata"]["sampling"]["top_p"] = 0.9
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(RuntimeError, "sampling settings mismatch"):
                executor.validate_output(output, case, case_path, list(range(512)))

    def test_resource_gates_and_deadline(self):
        self.assertTrue(executor.resource_preflight([], [], executor.MIN_MEM_BYTES))
        with self.assertRaisesRegex(RuntimeError, "active HF"):
            executor.resource_preflight([{"pid": 1}], [], executor.MIN_MEM_BYTES)
        with self.assertRaisesRegex(RuntimeError, "TT device owner"):
            executor.resource_preflight([], [{"pid": 2}], executor.MIN_MEM_BYTES)
        with self.assertRaisesRegex(RuntimeError, "below 100 GiB"):
            executor.resource_preflight([], [], executor.MIN_MEM_BYTES - 1)
        self.assertTrue(executor.check_deadline(5000, 3660, 1000))
        with self.assertRaisesRegex(RuntimeError, "ceiling >= frozen command timeout"):
            executor.check_deadline(5000, 3659, 1000)
        with self.assertRaisesRegex(RuntimeError, "exceeds authorized stage deadline"):
            executor.check_deadline(4600, 3660, 1000)

    def test_validate_cli_is_inert(self):
        self.assertEqual(executor.main(["--validate-only"]), 0)
        self.assertFalse(executor.OUTPUT.exists())
        self.assertFalse(executor.LAUNCH.exists())

    def test_execution_preflight_error_leaves_terminal_record_without_child(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            child = Mock(side_effect=AssertionError("child must not run"))
            with patch.object(executor, "LAUNCH", base / "launch.json"), patch.object(
                executor, "COMPLETION", base / "completion.json"
            ), patch.object(executor, "STOP", base / "STOP_BEFORE_LAUNCH"), patch.object(
                executor, "ADMISSION", base / "admission.json"
            ), patch.object(
                executor, "OUTPUT", base / "runs"
            ), patch.object(
                executor, "candidate_static", return_value={}
            ), patch.object(
                executor, "verify_admission", side_effect=RuntimeError("no root admission")
            ):
                code, record = executor._execute(child=child)
            self.assertEqual(code, 2)
            self.assertEqual(record["status"], "preflight_failed")
            self.assertIn("no root admission", record["error"])
            self.assertTrue((base / "completion.json").is_file())
            self.assertFalse((base / "launch.json").exists())
            child.assert_not_called()

    def test_child_success_without_valid_output_records_terminal_validation_failure(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            case_file = base / "case.json"
            case = self.mock_case()
            case_file.write_text(json.dumps(case))
            (base / "admission.json").write_text("{}\n")
            fake_spec = {
                "commands": {"commands": [{"argv": ["mock-runner"]}]},
                "forecast": {"sensitivity_range_seconds": [10, 100], "point_seconds": 50},
                "fixture": case,
                "manifest": [{"case": str(case_file)}],
                "prior_token_ids": list(range(512)),
            }
            child = Mock(return_value=SimpleNamespace(returncode=0))
            with patch.object(executor, "LAUNCH", base / "launch.json"), patch.object(
                executor, "COMPLETION", base / "completion.json"
            ), patch.object(executor, "STOP", base / "STOP_BEFORE_LAUNCH"), patch.object(
                executor, "HERE", base
            ), patch.object(
                executor, "ADMISSION", base / "admission.json"
            ), patch.object(
                executor, "OUTPUT", base / "runs"
            ), patch.object(
                executor, "candidate_static", return_value=fake_spec
            ), patch.object(
                executor, "verify_admission", return_value=({}, {"accepted": True})
            ), patch.object(
                executor, "process_snapshot", return_value=([], [])
            ), patch.object(
                executor, "available_memory", return_value=executor.MIN_MEM_BYTES
            ):
                code, record = executor._execute(child=child)
            self.assertEqual(code, 2)
            self.assertEqual(record["status"], "output_validation_failed")
            self.assertEqual(record["child_exit_code"], 0)
            self.assertTrue((base / "completion.json").is_file())
            child.assert_called_once()


if __name__ == "__main__":
    unittest.main()
