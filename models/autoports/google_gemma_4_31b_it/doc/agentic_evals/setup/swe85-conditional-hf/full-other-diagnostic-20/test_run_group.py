import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

DRIVER = Path(__file__).with_name("run_group.py")
spec = importlib.util.spec_from_file_location("swe85_full_other_driver", DRIVER)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DriverGateTests(unittest.TestCase):
    def test_cli_is_inert_without_explicit_mode(self):
        with self.assertRaises(SystemExit) as err:
            driver.main([])
        self.assertEqual(err.exception.code, 2)
        with self.assertRaises(SystemExit) as err:
            driver.main(["--execute"])
        self.assertEqual(err.exception.code, 2)

    def test_static_candidate_pins_validate_without_runner(self):
        bundle = driver.verify_static_bundle()
        self.assertEqual(len(bundle[1]["groups"]), 8)
        self.assertFalse((DRIVER.parent / "admission.json").exists())

    def test_resource_gates(self):
        ok = driver.check_resource_snapshot([], [], driver.MIN_MEM)
        self.assertEqual(ok["mem_available_bytes"], driver.MIN_MEM)
        with self.assertRaisesRegex(RuntimeError, "active HF"):
            driver.check_resource_snapshot([{"pid": 7}], [], driver.MIN_MEM)
        with self.assertRaisesRegex(RuntimeError, "TT device owners"):
            driver.check_resource_snapshot([], [{"pid": 8}], driver.MIN_MEM)
        with self.assertRaisesRegex(RuntimeError, "below 100GiB"):
            driver.check_resource_snapshot([], [], driver.MIN_MEM - 1)

    def test_admission_requires_exact_group_and_forecast_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name in (
                "proposal.json",
                "commands.json",
                "forecast.json",
                "source-integrity.json",
                "driver-contract.json",
                "test_run_group.py",
                "driver-integrity.json",
                "predecessor-gate.json",
            ):
                (base / name).write_text("{}\n")
            group = {"order": 1}
            forecast = {"groups": [{"group": 1, "sensitivity_range_seconds": [10, 100]}]}
            bundle = ({}, {}, forecast, {}, {}, {})
            auth = {
                "status": "authorized",
                "reviewer": "root",
                "forecast_sha256": digest(base / "forecast.json"),
                "forecast_range_seconds": [10, 100],
                "accepted_runtime_ceiling_seconds": 18060,
                "deadline_unix": 4_000_000_000,
            }
            admission = {
                "status": "admitted",
                "reviewer": "root",
                "driver_integrity_sha256": digest(base / "driver-integrity.json"),
                "driver_sha256": digest(DRIVER),
                "proposal_sha256": digest(base / "proposal.json"),
                "commands_sha256": digest(base / "commands.json"),
                "forecast_sha256": digest(base / "forecast.json"),
                "source_integrity_sha256": digest(base / "source-integrity.json"),
                "driver_contract_sha256": digest(base / "driver-contract.json"),
                "tests_sha256": digest(base / "test_run_group.py"),
                "authorized_groups": [1],
                "group_authorizations": {"1": auth},
                "predecessor_gate_sha256": digest(base / "predecessor-gate.json"),
            }
            (base / "admission.json").write_text(json.dumps(admission))
            result = driver.verify_admission(base, group, bundle, now=1_000_000)
            self.assertEqual(result[2], auth)
            # A short root ceiling cannot silently truncate the frozen 18,000 + 60 s command.
            auth["accepted_runtime_ceiling_seconds"] = 100
            (base / "admission.json").write_text(json.dumps(admission))
            with self.assertRaisesRegex(RuntimeError, "below the fixed subprocess timeout"):
                driver.verify_admission(base, {"order": 1, "timeout_seconds": 18000}, bundle, now=1_000_000)
            auth["accepted_runtime_ceiling_seconds"] = 100
            auth["deadline_unix"] = 1_000_105
            (base / "admission.json").write_text(json.dumps(admission))
            with self.assertRaisesRegex(RuntimeError, "below the fixed subprocess timeout and grace"):
                driver.verify_admission(base, {"order": 1, "timeout_seconds": 50}, bundle, now=1_000_000)
            auth["accepted_runtime_ceiling_seconds"] = 18060
            auth["deadline_unix"] = 4_000_000_000
            admission["authorized_groups"] = [1, 2]
            (base / "admission.json").write_text(json.dumps(admission))
            with self.assertRaisesRegex(RuntimeError, "exactly this single group"):
                driver.verify_admission(base, group, bundle, now=1_000_000)

    def test_predecessor_gate_requires_all_three_root_terminal_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ref = (
                driver.ROOT
                / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup/swe85-other-hf-coverage-c.json"
            )
            gate = {
                "status": "root_accepted_predecessors_terminal",
                "reviewer": "root",
                "predecessors": {
                    name: {
                        "status": "completed",
                        "exit_code": 0,
                        "completion_path": str(ref),
                        "completion_sha256": digest(ref),
                    }
                    for name in ("hf9", "skhf", "p3")
                },
            }
            (base / "predecessor-gate.json").write_text(json.dumps(gate))
            self.assertEqual(len(driver.verify_predecessors(base)["predecessors"]), 3)
            gate["predecessors"]["p3"]["exit_code"] = 1
            (base / "predecessor-gate.json").write_text(json.dumps(gate))
            with self.assertRaisesRegex(RuntimeError, "p3"):
                driver.verify_predecessors(base)

    def test_prior_group_requires_completion_and_root_forecast_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            launch_path = base / "launch-group-01.json"
            completion_path = base / "completion-group-01.json"
            review_path = base / "root-review-group-01.json"
            admission_sha = "a" * 64
            launch = {"status": "launched", "group": 1, "admission_sha256": admission_sha}
            launch_path.write_text(json.dumps(launch))
            rel = "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup/swe85-other-hf-coverage-c.json"
            result_path = driver.ROOT / rel
            result_hashes = {rel: digest(result_path)}
            completion = {
                "status": "completed",
                "group": 1,
                "driver_exit_code": 0,
                "output_validation_status": "passed",
                "admission_sha256": admission_sha,
                "launch_sha256": digest(launch_path),
                "forecast_range_seconds": [10, 20],
                "elapsed_seconds": 15,
                "result_sha256s": result_hashes,
            }
            completion_path.write_text(json.dumps(completion))
            review = {
                "status": "root_reviewed_forecast_and_outputs_next_group_authorized",
                "reviewer": "root",
                "candidate_admission_sha256": admission_sha,
                "launch_sha256": digest(launch_path),
                "completion_sha256": digest(completion_path),
                "forecast_gate": "hit",
                "result_sha256s": result_hashes,
            }
            review_path.write_text(json.dumps(review))
            forecast = {"groups": [{"group": 1, "sensitivity_range_seconds": [10, 20]}]}
            self.assertEqual(driver.verify_prior_group(base, {"order": 1}, forecast)["forecast_gate"], "hit")
            completion["elapsed_seconds"] = 25
            completion_path.write_text(json.dumps(completion))
            review["completion_sha256"] = digest(completion_path)
            review["forecast_gate"] = "root_accepted_miss"
            review_path.write_text(json.dumps(review))
            with self.assertRaises(FileNotFoundError):
                driver.verify_prior_group(base, {"order": 1}, forecast)

    def test_output_validation_binds_case_seed_cap_revision_and_hash(self):
        proposal = json.loads((DRIVER.parent / "proposal.json").read_text())
        commands = json.loads((DRIVER.parent / "commands.json").read_text())
        group = commands["groups"][0]
        jobs = json.loads((DRIVER.parent / group["manifest"]).read_text())
        by_name = {x["name"]: x for x in proposal["workload"]["observations"]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "results"
            output.mkdir()
            test_group = dict(group, output_dir=str(output))
            first_path = None
            for job in jobs:
                rec = by_name[job["name"]]
                case = json.loads(Path(job["case"]).read_text())
                valid = {
                    "status": "complete",
                    "case": job["case"],
                    "case_sha256": digest(job["case"]),
                    "revision": driver.REVISION,
                    "device": "cpu",
                    "weights_dtype": "bfloat16",
                    "prompt_source": "Saved prompt_token_ids, no chat rerender",
                    "prefill_chunk": 512,
                    "prompt_tokens": len(case["prompt_token_ids"]),
                    "teacher_forced_chosen": {"positions": [{}] * rec["chosen_teacher_forcing_positions"]},
                    "results": [
                        {
                            "seed": group["seed"],
                            "requested_tokens": group["cap"],
                            "generated_tokens": 1,
                            "generated_token_ids": [1],
                        }
                    ],
                    "tt_comparisons": {},
                }
                result_path = output / (job["name"] + ".hf.private.json")
                result_path.write_text(json.dumps(valid))
                if first_path is None:
                    first_path = result_path
            result_hashes = driver.verify_outputs(DRIVER.parent, root, test_group)
            self.assertEqual(result_hashes[first_path.relative_to(root).as_posix()], digest(first_path))
            first = json.loads(first_path.read_text())
            first["results"][0]["seed"] += 1
            first_path.write_text(json.dumps(first))
            with self.assertRaisesRegex(RuntimeError, "trial seed mismatch"):
                driver.verify_outputs(DRIVER.parent, root, test_group)

    def test_terminal_record_written_on_preflight_failure_without_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            group = {"order": 1, "command": ["never-run"], "output_dir": str(base / "out"), "seed": 42, "cap": 512}
            commands = {"groups": [group]}
            bundle = ({}, commands, {"groups": []}, {}, {}, {})
            child = Mock(side_effect=AssertionError("child must not run"))
            outcome = driver._run_group(
                1,
                base,
                base,
                {
                    "verify_bundle": lambda *_: bundle,
                    "verify_admission": lambda *_: (_ for _ in ()).throw(RuntimeError("no root admission")),
                    "subprocess_run": child,
                },
            )
            self.assertEqual(outcome["status"], "preflight_failed")
            self.assertIn("no root admission", outcome["error"])
            self.assertTrue((base / "completion-group-01.json").is_file())
            self.assertFalse((base / "launch-group-01.json").exists())
            child.assert_not_called()

    def test_mocked_success_is_one_group_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "forecast.json").write_text("{}\n")
            group = {
                "order": 1,
                "command": ["mock-child"],
                "output_dir": str(base / "out"),
                "seed": 42,
                "cap": 512,
                "timeout_seconds": 5,
            }
            commands = {"groups": [group]}
            forecast = {"groups": [{"group": 1, "sensitivity_range_seconds": [1, 20]}]}
            bundle = ({}, commands, forecast, {}, {}, {})
            admission = ({}, "b" * 64, {})
            predecessor = {"gate_sha256": "c" * 64}
            fake_run = Mock(return_value=SimpleNamespace(returncode=0))
            result_file = driver.ROOT / "mock-result.json"
            result_file.write_text("mock\n")
            outcome = driver._run_group(
                1,
                base,
                base,
                {
                    "verify_bundle": lambda *_: bundle,
                    "verify_admission": lambda *_: admission,
                    "verify_predecessors": lambda *_: predecessor,
                    "preflight_resources": lambda: {"mem_available_bytes": driver.MIN_MEM},
                    "subprocess_run": fake_run,
                    "verify_outputs": lambda *_: {"mock-result.json": digest(result_file)},
                },
            )
            self.assertEqual(outcome["status"], "completed")
            self.assertEqual(outcome["child_exit_code"], 0)
            self.assertTrue((base / "completion-group-01.json").is_file())
            self.assertFalse((base / "launch-group-02.json").exists())
            fake_run.assert_called_once()

    def test_mocked_success_completion_passes_next_group_root_review_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "forecast.json").write_text("{}\n")
            group = {
                "order": 1,
                "command": ["mock-child"],
                "output_dir": str(base / "out"),
                "seed": 42,
                "cap": 512,
                "timeout_seconds": 5,
            }
            commands = {"groups": [group]}
            forecast = {
                "groups": [{"group": 1, "sensitivity_range_seconds": [1, 20], "point_cap_length_scenario_seconds": 10}]
            }
            bundle = ({}, commands, forecast, {}, {}, {})
            admission = ({}, "b" * 64, {"accepted_runtime_ceiling_seconds": 65, "deadline_unix": 4_000_000_000})
            predecessor = {"gate_sha256": "c" * 64}
            fake_run = Mock(return_value=SimpleNamespace(returncode=0))
            result_file = driver.ROOT / "mock-result.json"
            result_file.write_text("mock\n")
            outcome = driver._run_group(
                1,
                base,
                base,
                {
                    "verify_bundle": lambda *_: bundle,
                    "verify_admission": lambda *_: admission,
                    "verify_predecessors": lambda *_: predecessor,
                    "preflight_resources": lambda: {"mem_available_bytes": driver.MIN_MEM},
                    "subprocess_run": fake_run,
                    "verify_outputs": lambda *_: {"mock-result.json": digest(result_file)},
                },
            )
            self.assertEqual(outcome["status"], "completed")
            completion_path = base / "completion-group-01.json"
            launch_path = base / "launch-group-01.json"
            self.assertEqual(outcome["forecast_range_seconds"], [1, 20])
            self.assertEqual(outcome["forecast_point_seconds"], 10)
            outcome["elapsed_seconds"] = 10
            completion_path.write_text(json.dumps(outcome))
            launch = json.loads(launch_path.read_text())
            completion = json.loads(completion_path.read_text())
            review = {
                "status": "root_reviewed_forecast_and_outputs_next_group_authorized",
                "reviewer": "root",
                "candidate_admission_sha256": "b" * 64,
                "launch_sha256": digest(launch_path),
                "completion_sha256": digest(completion_path),
                "forecast_gate": "hit",
                "result_sha256s": completion["result_sha256s"],
            }
            (base / "root-review-group-01.json").write_text(json.dumps(review))
            second_group = {"order": 2}
            result = driver.verify_prior_group(base, group, forecast)
            self.assertEqual(result["forecast_gate"], "hit")

    def test_mocked_child_validation_failure_records_child_zero_and_driver_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "forecast.json").write_text("{}\n")
            group = {
                "order": 1,
                "command": ["mock-child"],
                "output_dir": str(base / "out"),
                "seed": 42,
                "cap": 512,
                "timeout_seconds": 5,
            }
            commands = {"groups": [group]}
            forecast = {"groups": [{"group": 1, "sensitivity_range_seconds": [1, 20]}]}
            bundle = ({}, commands, forecast, {}, {}, {})
            admission = ({}, "b" * 64, {})
            predecessor = {"gate_sha256": "c" * 64}
            fake_run = Mock(return_value=SimpleNamespace(returncode=0))
            outcome = driver._run_group(
                1,
                base,
                base,
                {
                    "verify_bundle": lambda *_: bundle,
                    "verify_admission": lambda *_: admission,
                    "verify_predecessors": lambda *_: predecessor,
                    "preflight_resources": lambda: {"mem_available_bytes": driver.MIN_MEM},
                    "subprocess_run": fake_run,
                    "verify_outputs": lambda *_: (_ for _ in ()).throw(RuntimeError("bad output")),
                },
            )
            self.assertEqual(outcome["status"], "output_validation_failed")
            self.assertEqual(outcome["child_exit_code"], 0)
            self.assertEqual(outcome["driver_exit_code"], 2)
            self.assertTrue((base / "completion-group-01.json").is_file())
            fake_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
