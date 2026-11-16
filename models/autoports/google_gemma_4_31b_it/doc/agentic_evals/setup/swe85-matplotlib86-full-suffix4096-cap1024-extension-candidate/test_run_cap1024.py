"""Host-only tests for the inert cap1024 driver; fake child writes JSON, never imports TTNN."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cap1024_driver", HERE / "run_cap1024.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class DriverTests(unittest.TestCase):
    def make_candidate(self, root: Path):
        cand = root / "candidate"
        (cand / "fixtures").mkdir(parents=True)
        (cand / "prior").mkdir()
        (root / "h9").mkdir()
        (cand / "setup").mkdir()
        runtime = {
            "config_id": "stage12_bf16_hifi4_diagnostic",
            "weight_groups": {"attention": "bf16", "mlp": "bf16", "down": "bf16", "lm_head": "bf16"},
            "layer_exceptions": {},
            "compute_fidelities": {"attention": "HiFi4", "mlp": "HiFi4", "down": "HiFi4", "lm_head": "HiFi4"},
            "activation_dtype": "bfloat16",
            "residual_dtype": "bfloat16",
            "ccl_dtype": {"prefill": "bfloat16", "decode": "bfloat16"},
            "kv_cache_dtype": "bfloat16",
            "logits_dtype": "bfloat16",
            "sampling_dtype": "bfloat16",
            "embedding_dtype": "bfloat16",
            "norm_dtype": "bfloat16",
            "max_seq_len": 262144,
            "sdpa_compute_fidelity": "HiFi4",
        }
        build = {"batch_size": 1, "precision_config": "setup/full_precision_config.json"}
        cfg = {"example": "fullbf16"}
        fixture = {
            "prompt_token_ids": ([7, 8, 9] * 1365 + [7]),
            "continuations": {"chosen": [1] * 64},
            "metadata": {
                "task": "fixture",
                "call": 1,
                "seed": 43,
                "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
            },
        }
        prior_ids = list(range(512))
        prior = {
            "case": "candidate/fixtures/input.json",
            "runtime_precision": runtime,
            "free_running": [{"sampling": {"seed": 43}, "token_ids": prior_ids}],
        }
        for p, obj in [
            (root / "setup/full_precision_build_kwargs.json", build),
            (root / "setup/full_precision_config.json", cfg),
            (cand / "fixtures/input.json", fixture),
            (cand / "prior/baseline.json", prior),
            (cand / "prior/load.json", {"build_kwargs": build, "wall_s": 1.0}),
        ]:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(obj))
        fake_source = root / "source.py"
        fake_source.write_text("host-only pinned source\n")
        runtime_doc = {
            "build_kwargs_path": "setup/full_precision_build_kwargs.json",
            "build_kwargs_sha256": driver.digest(root / "setup/full_precision_build_kwargs.json"),
            "precision_config_path": "setup/full_precision_config.json",
            "precision_config_sha256": driver.digest(root / "setup/full_precision_config.json"),
            "expected_runtime_precision": runtime,
            "expected_runtime_precision_sha256": hashlib.sha256(
                json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "expected_config_id": "stage12_bf16_hifi4_diagnostic",
        }
        (cand / "runtime-pins.json").write_text(json.dumps(runtime_doc))
        source_doc = {
            "files": {
                "fixture": {
                    "path": "candidate/fixtures/input.json",
                    "sha256": driver.digest(cand / "fixtures/input.json"),
                },
                "prior_full_512": {
                    "path": "candidate/prior/baseline.json",
                    "sha256": driver.digest(cand / "prior/baseline.json"),
                },
                "prior_full_load": {
                    "path": "candidate/prior/load.json",
                    "sha256": driver.digest(cand / "prior/load.json"),
                },
                "trace_prep_proxy_log": {"path": "source.py", "sha256": driver.digest(fake_source)},
                "batch_runner": {"path": "source.py", "sha256": driver.digest(fake_source)},
                "probe_runner": {"path": "source.py", "sha256": driver.digest(fake_source)},
            }
        }
        (cand / "source-pins.json").write_text(json.dumps(source_doc))
        job = {
            "name": "matplotlib-test-cap1024-seed43",
            "case": "candidate/fixtures/input.json",
            "branches": ["chosen"],
            "free_run_tokens": 1024,
            "free_run_trials": 1,
        }
        (cand / "cases.json").write_text(json.dumps([job]))
        forecast = {"calculation": {"forecast_range_wall_s": [0, 99999]}}
        (cand / "forecast.json").write_text(json.dumps(forecast))
        cmd = [
            "env",
            "HF_HUB_OFFLINE=1",
            "HF_DATASETS_OFFLINE=1",
            "READINESS_TRACE_REGION_SIZE=134217728",
            "python",
            "-m",
            "models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch",
            "--cases",
            str((cand / "cases.json").resolve()),
            "--build-kwargs-file",
            str((root / "setup/full_precision_build_kwargs.json").resolve()),
            "--output-dir",
            str((cand / "runs/full").resolve()),
        ]
        command_doc = {"argv": cmd, "expected_output": str((cand / "runs/full" / (job["name"] + ".json")).resolve())}
        (cand / "command.json").write_text(json.dumps(command_doc))
        proposal = {"status": "candidate_only_not_admitted_not_run"}
        (cand / "proposal.json").write_text(json.dumps(proposal))
        (cand / "admission-template.json").write_text("{}")
        import shutil

        shutil.copyfile(HERE / "run_cap1024.py", cand / "run_cap1024.py")
        shutil.copyfile(HERE / "test_run_cap1024.py", cand / "test_run_cap1024.py")
        # Create an eight-group terminal H9 evidence bundle with pinned files.
        groups = []
        group_evidence = []
        miss_acceptances = []
        for i in range(1, 9):
            comp = {"serial_order": i, "exit_code": 0}
            cp = root / f"h9/completion-group-{i:02}.json"
            cp.write_text(json.dumps(comp))
            csha = driver.digest(cp)
            review = {"reviewer": "root", "accepted": True, "completion_sha256": csha}
            rp = root / f"h9/root-group-{i:02}-review.json"
            rp.write_text(json.dumps(review))
            rsha = driver.digest(rp)
            row = {"serial_order": i, "status": "completed", "completion_sha256": csha}
            if i == 2:
                row["forecast_miss"] = {"measured_seconds": 20, "forecast_range_seconds": [10, 15]}
                accept = {
                    "status": "root_accepted_forecast_miss",
                    "reviewer": "root",
                    "accepted": True,
                    "completion_sha256": csha,
                }
                ap = root / "h9/forecast-miss-accepted-group-02.json"
                ap.write_text(json.dumps(accept))
                miss_acceptances.append(
                    {"serial_order": i, "path": "h9/forecast-miss-accepted-group-02.json", "sha256": driver.digest(ap)}
                )
            groups.append(row)
            group_evidence.append(
                {
                    "serial_order": i,
                    "completion": {"path": f"h9/completion-group-{i:02}.json", "sha256": csha},
                    "root_review": {"path": f"h9/root-group-{i:02}-review.json", "sha256": rsha},
                }
            )
        progress = {"status": "all_groups_completed", "groups": groups}
        prog = root / "h9/progress.json"
        prog.write_text(json.dumps(progress))
        forecast_review = {
            "reviewer": "root",
            "accepted": True,
            "forecast_sha256": driver.digest(cand / "forecast.json"),
            "forecast_range_wall_s": [0, 99999],
        }
        fr = root / "h9/candidate-forecast-review.json"
        fr.write_text(json.dumps(forecast_review))
        candidate_pins = {name: driver.digest(cand / name) for name in driver.PINNED_FILES}
        admission = {
            "status": "admitted",
            "reviewer": "root",
            "accepted": True,
            "deadline_utc": "2026-11-16T01:15:25Z",
            "timeout_seconds": 1200,
            "candidate_pins": candidate_pins,
            "candidate_forecast_review": {"path": "h9/candidate-forecast-review.json", "sha256": driver.digest(fr)},
            "h9_predecessor": {
                "progress": {"path": "h9/progress.json", "sha256": driver.digest(prog)},
                "terminal_status": "all_groups_completed",
                "groups": group_evidence,
                "forecast_miss_acceptances": miss_acceptances,
            },
        }
        (cand / "admission.json").write_text(json.dumps(admission))
        return cand, root, fixture, prior, runtime

    @staticmethod
    def fake_child_for(candidate, root, fixture, prior, runtime, token_ids=None, returncode=0):
        def child(argv, cwd, stdout, stderr, timeout, check=False):
            outdir = Path(argv[argv.index("--output-dir") + 1])
            outdir.mkdir(parents=True, exist_ok=True)
            build = json.loads((root / "setup/full_precision_build_kwargs.json").read_text())
            (outdir / "load.json").write_text(json.dumps({"build_kwargs": build, "wall_s": 1.0}))
            ids = list(prior["free_running"][0]["token_ids"]) + [777] * 512 if token_ids is None else token_ids
            job = json.loads((candidate / "cases.json").read_text())[0]
            result = {
                "case": job["case"],
                "runtime_precision": runtime,
                "continuations": {
                    "chosen": {
                        "tokens": 64,
                        "positions": [{}] * 64,
                        "conditioning_token_ids": fixture["prompt_token_ids"],
                    }
                },
                "free_running": [
                    {
                        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 43},
                        "token_ids": ids,
                        "flag": False,
                    }
                ],
            }
            (outdir / (job["name"] + ".json")).write_text(json.dumps(result))
            return SimpleNamespace(returncode=returncode)

        return child

    def test_success_requires_entire_512_prefix_and_records_completion(self):
        with tempfile.TemporaryDirectory() as td:
            cand, root, fixture, prior, runtime = self.make_candidate(Path(td))
            comp = driver.execute_candidate(
                cand,
                root,
                self.fake_child_for(cand, root, fixture, prior, runtime),
                lambda: [],
                lambda: [],
                lambda: driver.DEADLINE - 2000,
            )
            self.assertEqual(comp["status"], "completed_forecast_hit")
            self.assertEqual(comp["outputs"]["prefix_identity_tokens"], 512)
            self.assertEqual(comp["outputs"]["result_tokens"], 1024)
            self.assertTrue((cand / "completion.json").is_file())

    def test_short_result_cannot_pass_min_prefix_check_and_completion_is_written(self):
        with tempfile.TemporaryDirectory() as td:
            cand, root, fixture, prior, runtime = self.make_candidate(Path(td))
            # Deliberately generate 511 tokens that all match prior[0:511]; min(512,n) would wrongly pass.
            child = self.fake_child_for(
                cand, root, fixture, prior, runtime, token_ids=prior["free_running"][0]["token_ids"][:511]
            )
            comp = driver.execute_candidate(cand, root, child, lambda: [], lambda: [], lambda: driver.DEADLINE - 2000)
            self.assertEqual(comp["status"], "invalid_outputs")
            self.assertIn("len>=512", comp["output_validation_error"])
            self.assertEqual(comp["process_exit_code"], 0)

    def test_mismatched_full_512_prefix_fails_and_completion_is_written(self):
        with tempfile.TemporaryDirectory() as td:
            cand, root, fixture, prior, runtime = self.make_candidate(Path(td))
            ids = list(prior["free_running"][0]["token_ids"]) + [777] * 512
            ids[511] = -1
            comp = driver.execute_candidate(
                cand,
                root,
                self.fake_child_for(cand, root, fixture, prior, runtime, token_ids=ids),
                lambda: [],
                lambda: [],
                lambda: driver.DEADLINE - 2000,
            )
            self.assertEqual(comp["status"], "invalid_outputs")
            self.assertIn("entire prior512-token prefix", comp["output_validation_error"])
            self.assertTrue((cand / "completion.json").is_file())

    def test_nonzero_fake_child_always_leaves_terminal_completion(self):
        with tempfile.TemporaryDirectory() as td:
            cand, root, fixture, prior, runtime = self.make_candidate(Path(td))

            def child(*args, **kwargs):
                return SimpleNamespace(returncode=9)

            comp = driver.execute_candidate(cand, root, child, lambda: [], lambda: [], lambda: driver.DEADLINE - 2000)
            self.assertEqual(comp["status"], "failed")
            self.assertEqual(comp["process_exit_code"], 9)
            self.assertIsNotNone(comp["output_validation_error"])
            self.assertTrue((cand / "completion.json").is_file())

    def test_timeout_exception_leaves_completion_record(self):
        with tempfile.TemporaryDirectory() as td:
            cand, root, fixture, prior, runtime = self.make_candidate(Path(td))

            def child(*args, **kwargs):
                raise driver.subprocess.TimeoutExpired(cmd="fake", timeout=1200)

            comp = driver.execute_candidate(cand, root, child, lambda: [], lambda: [], lambda: driver.DEADLINE - 2000)
            self.assertEqual(comp["status"], "failed")
            self.assertIsNone(comp["process_exit_code"])
            self.assertIn("TimeoutExpired", comp["child_error"])
            self.assertTrue((cand / "completion.json").is_file())

    def test_expected_output_accepts_repo_relative_command_path(self):
        with tempfile.TemporaryDirectory() as td:
            cand, root, fixture, prior, runtime = self.make_candidate(Path(td))
            command_path = cand / "command.json"
            command = json.loads(command_path.read_text())
            command["expected_output"] = str(Path(command["expected_output"]).relative_to(root))
            command_path.write_text(json.dumps(command))
            admission_path = cand / "admission.json"
            admission = json.loads(admission_path.read_text())
            admission["candidate_pins"]["command.json"] = driver.digest(command_path)
            admission_path.write_text(json.dumps(admission))
            _, _, ctx = driver.verify_candidate(cand, root)
            self.assertEqual(ctx["job"]["free_run_tokens"], 1024)

    def test_nonterminal_predecessor_refuses_before_fake_child(self):
        with tempfile.TemporaryDirectory() as td:
            cand, root, fixture, prior, runtime = self.make_candidate(Path(td))
            p = root / "h9/progress.json"
            data = json.loads(p.read_text())
            data["status"] = "group_running"
            p.write_text(json.dumps(data))
            admission = json.loads((cand / "admission.json").read_text())
            admission["h9_predecessor"]["progress"]["sha256"] = driver.digest(p)
            (cand / "admission.json").write_text(json.dumps(admission))
            called = []

            def child(*a, **k):
                called.append(True)
                return SimpleNamespace(returncode=0)

            with self.assertRaisesRegex(RuntimeError, "progress status"):
                driver.execute_candidate(cand, root, child, lambda: [], lambda: [], lambda: driver.DEADLINE - 2000)
            self.assertFalse(called)

    def test_existing_output_directory_refuses_before_fake_child(self):
        with tempfile.TemporaryDirectory() as td:
            cand, root, fixture, prior, runtime = self.make_candidate(Path(td))
            (cand / "runs/full").mkdir(parents=True)
            called = []

            def child(*a, **k):
                called.append(True)
                return SimpleNamespace(returncode=0)

            with self.assertRaisesRegex(RuntimeError, "output directory exists"):
                driver.execute_candidate(cand, root, child, lambda: [], lambda: [], lambda: driver.DEADLINE - 2000)
            self.assertFalse(called)

    def test_forecast_miss_is_recorded_for_root_review(self):
        with tempfile.TemporaryDirectory() as td:
            cand, root, fixture, prior, runtime = self.make_candidate(Path(td))
            fp = cand / "forecast.json"
            forecast = json.loads(fp.read_text())
            forecast["calculation"]["forecast_range_wall_s"] = [9999, 10000]
            fp.write_text(json.dumps(forecast))
            rp = root / "h9/candidate-forecast-review.json"
            review = json.loads(rp.read_text())
            review["forecast_sha256"] = driver.digest(fp)
            review["forecast_range_wall_s"] = [9999, 10000]
            rp.write_text(json.dumps(review))
            ap = cand / "admission.json"
            admission = json.loads(ap.read_text())
            admission["candidate_pins"]["forecast.json"] = driver.digest(fp)
            admission["candidate_forecast_review"]["sha256"] = driver.digest(rp)
            ap.write_text(json.dumps(admission))
            child = self.fake_child_for(cand, root, fixture, prior, runtime)
            comp = driver.execute_candidate(cand, root, child, lambda: [], lambda: [], lambda: driver.DEADLINE - 2000)
            self.assertEqual(comp["status"], "completed_forecast_miss_requires_root_review")
            self.assertEqual(comp["forecast_gate"]["status"], "miss_requires_root_review")

    def test_candidate_pin_tampering_refuses_before_fake_child(self):
        with tempfile.TemporaryDirectory() as td:
            cand, root, fixture, prior, runtime = self.make_candidate(Path(td))
            (cand / "command.json").write_text("{}")
            called = []

            def child(*a, **k):
                called.append(True)
                return SimpleNamespace(returncode=0)

            with self.assertRaisesRegex(RuntimeError, "Candidate pin mismatch: command.json"):
                driver.execute_candidate(cand, root, child, lambda: [], lambda: [], lambda: driver.DEADLINE - 2000)
            self.assertFalse(called)

    def test_hf_process_matcher_covers_both_reference_runners(self):
        self.assertTrue(driver.is_hf_reference_process("python tools/precision_repro_hf.py --mode full"))
        self.assertTrue(driver.is_hf_reference_process("python tools/actual_branch_logits.py --mode hf"))
        self.assertFalse(driver.is_hf_reference_process("python tools/precision_repro_batch.py --mode full"))

    def test_owner_or_deadline_failure_refuses_before_fake_child(self):
        with tempfile.TemporaryDirectory() as td:
            cand, root, fixture, prior, runtime = self.make_candidate(Path(td))
            called = []

            def child(*a, **k):
                called.append(True)
                return SimpleNamespace(returncode=0)

            with self.assertRaisesRegex(RuntimeError, "TT device owner"):
                driver.execute_candidate(
                    cand, root, child, lambda: [], lambda: [{"owner": "busy"}], lambda: driver.DEADLINE - 2000
                )
            self.assertFalse(called)
            with self.assertRaisesRegex(RuntimeError, "timeout\+60"):
                driver.execute_candidate(cand, root, child, lambda: [], lambda: [], lambda: driver.DEADLINE - 1200)
            self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
