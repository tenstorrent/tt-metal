# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Exercise the CI shell orchestration without MPI, model weights, or devices.

Run directly with ``python3 -m unittest discover -s
models/demos/common/prefill/runners/ci -p test_run_multirank_pcc.py``.
The actual shell harness runs with stub launchers; its inline verdict gate remains real.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS = Path(__file__).with_name("run_multirank_pcc.sh")
# tt-smi's ENABLED_GDDR on an emulated Bin6 Galaxy: chip i is missing DRAM bank i % 8.
BIN6_GDDR = "0xfe,0xfd,0xfb,0xf7,0xef,0xdf,0xbf,0x7f"
STUB = r"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

args = sys.argv[1:]
name = Path(sys.argv[0]).name
root = Path(os.environ["TEST_ROOT"])

def record(kind):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=root / "records", delete=False) as f:
        selected_env = {key: value for key, value in os.environ.items()
                        if key.startswith(("PREFILL_", "TEST_", "TT_MESH_", "TT_METAL_OPERATION_"))}
        json.dump({"kind": kind, "args": args, "env": selected_env}, f)

def shell_command():
    index = args.index("bash")
    return subprocess.call(["bash", "-c", args[index + 2]], env=os.environ)

if name == "sleep":
    time.sleep(0.02)
elif name == "tt-smi":
    # A 32-chip snapshot whose ENABLED_GDDR masks cycle through TEST_ENABLED_GDDR; unset, tt-smi fails.
    if "TEST_ENABLED_GDDR" not in os.environ:
        sys.exit(1)
    masks = os.environ["TEST_ENABLED_GDDR"].split(",")
    print(json.dumps({"device_info": [{"smbus_telem": {"ENABLED_GDDR": masks[i % len(masks)]}} for i in range(32)]}))
elif name.startswith("mpirun"):
    record("mpi")
    if "--output-filename" in args:
        log = Path(args[args.index("--output-filename") + 1]) / "1" / "rank.0" / "stdout"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("stub rank log\n")
    os.environ["TEST_MPI_HOST"] = args[args.index("--host") + 1] if "--host" in args else ""
    os.environ["OMPI_COMM_WORLD_RANK"] = "0"
    if "bash" in args:
        sys.exit(shell_command())
    index = next(i for i, arg in enumerate(args) if arg.endswith("python3"))
    sys.exit(subprocess.call(args[index:], env=os.environ))
elif args and args[0].endswith("ttrun.py"):
    record("runner_launch")
    # Explicit bindings do not produce discovery artifacts in real tt-run.
    if "--rank-binding" not in args:
        rankfile = Path.cwd() / "generated/ttrun/test/rankfile"
        rankfile.parent.mkdir(parents=True, exist_ok=True)
        hosts = ["worker%d" % i for i in range(4)]
        rankfile.write_text("".join("rank %d=%s slot=0\n" % (i, host) for i, host in enumerate(hosts)))
    sys.exit(shell_command())
elif args and args[0].endswith("gen_pipeline_binding.py"):
    record("probe")
    out = Path(args[args.index("--out") + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("rank_bindings: []\n")
elif args[:1] == ["-m"] and args[1].endswith("prefill_runner"):
    record("runner")
    Path(os.environ["PREFILL_MIGRATION_TABLE_PATH"]).touch()
    deadline = time.monotonic() + 15
    while not (root / "producer_done").exists():
        if time.monotonic() > deadline:
            sys.exit(99)
        time.sleep(0.01)
    sys.exit(int(os.environ.get("TEST_RUNNER_RC", "0")))
elif args[:1] == ["-m"] and args[1].endswith("prefill_producer"):
    record("producer")
    verdict_dir = Path(os.environ["PREFILL_PCC_SUMMARY_DIR"])
    verdict_dir.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("TEST_VERDICT", "pass")
    for rank in range(4):
        if rank == 3 and mode == "missing":
            continue
        # The real producer echoes the threshold it was given; the gate checks it is the model's.
        threshold = float(os.environ["PREFILL_STANDALONE_CHUNKED_PCC"])
        verdict = {"rank": rank, "ok": True, "min_pcc": 0.99, "threshold": threshold,
                   "slots_checked": 1, "per_cache": {"kvpe": 0.99}}
        if rank == 3:
            if mode == "false":
                verdict["ok"] = False
            elif mode == "truthy":
                verdict["ok"] = "yes"
            elif mode == "empty":
                verdict["slots_checked"] = 0
                verdict["per_cache"] = {}
            elif mode == "nan":
                verdict["min_pcc"] = float("nan")
                verdict["per_cache"]["kvpe"] = float("nan")
        filename_rank = 4 if rank == 3 and mode == "wrong_rank" else rank
        (verdict_dir / ("rank%d.json" % filename_rank)).write_text(json.dumps(verdict))
    (root / "producer_done").touch()
elif args and args[0].endswith("summarize_ci_run.py"):
    if os.environ.get("TEST_TPS"):
        summary_name = args[args.index("--summary-name") + 1]
        out = Path(os.environ["PREFILL_SUMMARIES"]) / "perf_json"
        out.mkdir(parents=True, exist_ok=True)
        (out / ("%s.json" % summary_name)).write_text(json.dumps({
            "chunk_index": {"@5k": 0, "@55k": 10},
            "throughput_tok_s": {"@5k": 37234.8, "@55k": float(os.environ["TEST_TPS"])},
        }))
elif args and args[0].endswith("plot_pipeline_trace.py"):
    pass
elif args == ["-c", "import matplotlib"]:
    pass
else:
    os.execv(os.environ["TEST_REAL_PYTHON"], [os.environ["TEST_REAL_PYTHON"], *args])
"""


class MultirankPccHarnessTests(unittest.TestCase):
    def run_harness(
        self,
        model,
        *,
        verdict="pass",
        runner_rc=0,
        missing_cache=False,
        hosts=None,
        config=None,
        extra_env=None,
        options=(),
        perf_baselines=None,
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "records").mkdir()
        bin_dir = root / "bin"
        bin_dir.mkdir()
        for name in ("python3", "mpirun", "mpirun-ulfm", "sleep", "tt-smi"):
            executable = bin_dir / name
            executable.write_text(f"#!{sys.executable}\n" + STUB)
            executable.chmod(0o755)
        metal = root / "metal"
        metal.mkdir()
        mgd_dir = metal / "models/demos/common/prefill/runners/topology_configuration/ci"
        mgd_dir.mkdir(parents=True)
        for system in ("sc1", "sc4"):
            (mgd_dir / f"{system}_mgd.textproto").touch()
        # manifest_env() reads PREFILL_MAX_SEQ_LEN / PREFILL_NUM_USERS straight off these, so the
        # fake tree needs them or the harness dies before it launches anything. Values mirror the
        # real manifests; only their presence and the sc1 override path are under test here.
        manifest_dir = metal / "models/demos/deepseek_v3_d_p/tt/runners/manifests"
        manifest_dir.mkdir(parents=True)
        for model_key, max_seq_len, num_users in (
            ("kimi27", 256000, 86),
            ("glm53", 1064960, 54),
            ("mistral4", 61440, 1),
        ):
            (manifest_dir / f"{model_key}.json").write_text(
                json.dumps({"env": {"PREFILL_MAX_SEQ_LEN": max_seq_len, "PREFILL_NUM_USERS": num_users}})
            )
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        cache = root / "cache"
        if not missing_cache:
            (cache / "mistral_small_4_bh_8dev/8x1").mkdir(parents=True)
        golden = root / "golden"
        (golden / "kv_cache").mkdir(parents=True)
        for layer in range(36):
            (golden / "kv_cache" / f"layer_{layer}.safetensors").touch()
        harness = metal / "run_multirank_pcc.sh"
        shutil.copyfile(HARNESS, harness)
        perf_gate = metal / "models/demos/common/prefill/runners/ci/perf_gate.py"
        perf_gate.parent.mkdir(parents=True)
        shutil.copyfile(HARNESS.with_name("perf_gate.py"), perf_gate)
        if perf_baselines is not None:
            (perf_gate.parent / "perf_baselines.json").write_text(
                json.dumps({"margin": {"throughput_tok_s": 0.015}, "baselines": perf_baselines})
            )
        telemetry = metal / "models/demos/deepseek_v3_d_p/utils/smbus_telemetry.py"
        telemetry.parent.mkdir(parents=True)
        shutil.copyfile(HARNESS.parents[4] / "deepseek_v3_d_p/utils/smbus_telemetry.py", telemetry)
        ttrun_dir = root / "ttop"
        ttrun_dir.mkdir()
        if hosts is None:
            hosts = ["worker"] if model == "mistral4" else [f"worker{i}" for i in range(4)]
        (ttrun_dir / "hostfile").write_text("\n".join(hosts) + "\n")
        env = {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "TT_METAL_HOME": str(metal),
            "PREFILL_SUMMARIES": str(root / "prefill_summaries"),
            "TTRUN_DIR": str(ttrun_dir),
            "PREFILL_HF_MODEL": str(checkpoint),
            "PREFILL_TTNN_CACHE": str(cache),
            "PREFILL_TRACE_DIR": str(golden),
            "TEST_ROOT": str(root),
            "TEST_REAL_PYTHON": sys.executable,
            "TEST_VERDICT": verdict,
            "TEST_RUNNER_RC": str(runner_rc),
        }
        env.update(extra_env or {})
        command = ["bash", str(harness), model]
        if config is not None:
            command.append(config)
        command += options
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)
        records = [json.loads(path.read_text()) for path in (root / "records").glob("*.json")]
        return result, records

    def one_record(self, records, kind):
        matches = [record for record in records if record["kind"] == kind]
        self.assertEqual(len(matches), 1, (kind, records))
        return matches[0]

    def test_mistral_uses_single_host_rank_binding_and_remote_probe(self):
        result, records = self.run_harness("mistral4")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        launch = self.one_record(records, "runner_launch")
        self.assertIn("--rank-binding", launch["args"])
        self.assertNotIn("--mesh-graph-descriptor", launch["args"])
        self.assertNotIn("--hosts", launch["args"])
        probe = self.one_record(records, "probe")
        self.assertEqual(probe["env"]["TEST_MPI_HOST"].split(":")[0], "worker")
        self.assertTrue(
            probe["env"]["TT_MESH_GRAPH_DESC_PATH"].endswith("single_bh_galaxy_torus_xy_graph_descriptor.textproto")
        )
        runner = self.one_record(records, "runner")["env"]
        self.assertEqual(runner["PREFILL_FABRIC_MODE"], "2d_torus_y")
        self.assertEqual(runner["PREFILL_USE_TRACE"], "1")
        producer = self.one_record(records, "producer")["env"]
        for key, value in {
            "PREFILL_SP": "8",
            "PREFILL_TP": "1",
            "PREFILL_MAX_SEQ_LEN": "61440",
            "PREFILL_PRODUCER_CHUNKS": "11",
            "PREFILL_PRODUCER_CHECK_PCC": "1",
        }.items():
            self.assertEqual(producer[key], value, key)
        self.assertEqual(producer["TEST_MPI_HOST"], "worker:4")
        producer_mpi = [
            record for record in records if record["kind"] == "mpi" and "prefill_producer" in record["args"][-1]
        ]
        self.assertEqual(len(producer_mpi), 1)
        mpi_args = producer_mpi[0]["args"]
        self.assertEqual(mpi_args[mpi_args.index("-np") + 1], "4")
        # Both get the base path; each process scopes it to its rank (rank_scoped_device_map_path).
        self.assertTrue(producer["PREFILL_MIGRATION_DEVICE_MAP_PATH"].endswith("device_map.json"))
        self.assertEqual(runner["PREFILL_MIGRATION_DEVICE_MAP_PATH"], producer["PREFILL_MIGRATION_DEVICE_MAP_PATH"])
        self.assertIn("PCC GATE PASS", result.stdout)

    def test_existing_models_use_discovery_and_four_hosts(self):
        for model, max_seq_len, request_len in (("kimi27", 256000, 256000), ("glm53", 1064960, 1044480)):
            with self.subTest(model=model):
                result, records = self.run_harness(model)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                launch = self.one_record(records, "runner_launch")["args"]
                self.assertIn("--mesh-graph-descriptor", launch)
                self.assertIn("--hosts", launch)
                self.assertNotIn("--rank-binding", launch)
                self.assertFalse([record for record in records if record["kind"] == "probe"])
                producer = self.one_record(records, "producer")["env"]
                self.assertEqual(producer["TEST_MPI_HOST"], "worker0:1,worker1:1,worker2:1,worker3:1")
                self.assertEqual(producer["PREFILL_MAX_SEQ_LEN"], str(max_seq_len))
                self.assertEqual(producer["PREFILL_PRODUCER_CHUNKS"], str(request_len // 5120))

    def test_sc1_windows_preserve_model_defaults(self):
        for model, expected_length in (("mistral4", 61440), ("kimi27", 256000), ("glm53", 1064960)):
            with self.subTest(model=model):
                result, records = self.run_harness(model, config="sc1")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.one_record(records, "runner")["env"]["PREFILL_NUM_USERS"], "1")
                self.assertEqual(
                    self.one_record(records, "producer")["env"]["PREFILL_MAX_SEQ_LEN"], str(expected_length)
                )

    def test_bin6_runs_glm53_sc1_at_a_7_bank_aligned_length(self):
        # A glm53 baseline the run would miss by far, so gating Bin6 against it would fail the run.
        result, records = self.run_harness(
            "glm53",
            config="sc1",
            options=["--bin6"],
            extra_env={"TEST_ENABLED_GDDR": BIN6_GDDR, "TEST_TPS": "6000"},
            perf_baselines={"glm53": {"sc1": {"probe": "@55k", "8kw": {"throughput_tok_s": 99999}}}},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("every chip on worker0 has 7 DRAM banks", result.stdout)
        runner = self.one_record(records, "runner")["env"]
        producer = self.one_record(records, "producer")["env"]
        self.assertEqual(runner["PREFILL_MAX_SEQ_LEN"], "1075200")
        self.assertEqual(runner["PREFILL_NUM_USERS"], "1")
        # The 8-bank leg's prompt, so the same golden tail gates it.
        self.assertEqual(producer["PREFILL_MAX_SEQ_LEN"], "1075200")
        self.assertEqual(producer["PREFILL_PRODUCER_CHUNKS"], str(1044480 // 5120))
        self.assertIn("TPS: 6,000 tok/s @55k (no PREFILL_EXPECTED_TPS set -- reporting, not gating)", result.stdout)
        self.assertIn("PCC GATE PASS", result.stdout)

    def test_bin6_skips_a_machine_without_7_banks_on_every_chip(self):
        for gddr, banks in (("0xff", "8"), ("0xff,0xfe", "7,8")):
            with self.subTest(banks=banks):
                result, records = self.run_harness(
                    "glm53", config="sc1", options=["--bin6"], extra_env={"TEST_ENABLED_GDDR": gddr}
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(
                    f"skipped: --bin6 needs 7 DRAM banks on every chip, and worker0 has {banks}", result.stdout
                )
                self.assertFalse([record for record in records if record["kind"] in ("runner_launch", "producer")])

    def test_bin6_fails_when_the_bank_count_is_unreadable(self):
        result, records = self.run_harness("glm53", config="sc1", options=["--bin6"])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("could not read the DRAM bank count on worker0", result.stderr)
        self.assertFalse([record for record in records if record["kind"] == "runner_launch"])

    def test_bin6_rejects_runs_it_does_not_cover(self):
        for model, config, options, message in (
            ("glm53", "sc4", ["--bin6"], "--bin6 runs sc1 only"),
            ("kimi27", "sc1", ["--bin6"], "kimi27 has no Bin6 configuration"),
            ("glm53", "sc1", ["--bin6", "--variant", "mtp_kv_share"], "--bin6 does not combine with --variant"),
        ):
            with self.subTest(model=model, config=config, options=options):
                result, records = self.run_harness(
                    model, config=config, options=options, extra_env={"TEST_ENABLED_GDDR": BIN6_GDDR}
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(message, result.stderr)
                self.assertFalse(records)

    def test_perf_gate_fails_when_no_throughput_was_measured(self):
        result, _ = self.run_harness("mistral4", extra_env={"PREFILL_EXPECTED_TPS": "17000"})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("TPS GATE FAIL", result.stdout + result.stderr)

    def test_perf_gate_requires_a_measurement_before_a_baseline_exists(self):
        result, _ = self.run_harness("mistral4", extra_env={"PREFILL_REQUIRE_TPS": "1"})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("TPS GATE FAIL", result.stdout + result.stderr)

    def test_perf_gate_reads_the_deepest_probe(self):
        # Shallow probes run faster; gating the wrong one would hide a full-context regression.
        result, _ = self.run_harness("mistral4", extra_env={"TEST_TPS": "23689", "PREFILL_EXPECTED_TPS": "23689"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("TPS GATE @55k", result.stdout)

    def test_perf_gate_still_verifies_pcc(self):
        # Multi-rank producers refuse PREFILL_PRODUCER_CHECK_PCC=0: ranks 1..N are validators.
        _, records = self.run_harness("mistral4", extra_env={"PREFILL_REQUIRE_TPS": "1"})
        self.assertEqual(self.one_record(records, "producer")["env"]["PREFILL_PRODUCER_CHECK_PCC"], "1")

    def test_perf_gate_passes_within_margin(self):
        result, _ = self.run_harness("mistral4", extra_env={"TEST_TPS": "17000", "PREFILL_EXPECTED_TPS": "17500"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("TPS GATE PASS", result.stdout)
        self.assertIn("PCC GATE PASS", result.stdout)

    def test_perf_gate_fails_outside_margin(self):
        result, _ = self.run_harness(
            "mistral4", extra_env={"TEST_TPS": "10000", "PREFILL_EXPECTED_TPS": "17000", "GITHUB_ACTIONS": "true"}
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("::error title=Prefill perf gate::TPS GATE FAIL", result.stdout)
        # Log summaries read the tail: the failed gate must come after the rank-log dump.
        tail = result.stdout.rsplit("==================== gate verdicts ====================", 1)
        self.assertEqual(len(tail), 2, result.stdout)
        self.assertIn("ranklog tails", tail[0])
        self.assertIn("TPS GATE FAIL", tail[1])
        self.assertNotIn("ranklog tails", tail[1])

    def test_perf_gate_reports_when_no_baseline_is_set(self):
        # The state the CI leg ships in: measure and print, do not compare.
        result, _ = self.run_harness("mistral4", extra_env={"TEST_TPS": "17000", "PREFILL_REQUIRE_TPS": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("TPS: 17,000 tok/s @55k", result.stdout)
        self.assertIn("reporting, not gating", result.stdout)

    def test_perf_knobs_absent_leaves_pcc_gating_untouched(self):
        result, _ = self.run_harness("mistral4")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("TPS GATE", result.stdout + result.stderr)

    def test_mistral_rejects_sc4(self):
        result, records = self.run_harness("mistral4", config="sc4")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mistral4 does not support config 'sc4'", result.stderr)
        self.assertIn("supported: sc1", result.stderr)
        self.assertFalse(records)

    def test_incomplete_or_invalid_verdicts_fail(self):
        # The expected reason is asserted, not just the failure: every one of these would also fail
        # for the wrong reason if a clause were dropped, and the gate names the clause that tripped.
        reasons = {
            "missing": "expected exactly ranks 0..3",
            "false": "ok is not literally true",
            "truthy": "ok is not literally true",
            "empty": "slots_checked is not the 1 slots fed",
            "nan": "non-finite",
            "wrong_rank": "expected exactly ranks 0..3",
        }
        for verdict, reason in reasons.items():
            with self.subTest(verdict=verdict):
                result, records = self.run_harness("mistral4", verdict=verdict)
                self.one_record(records, "producer")
                output = result.stdout + result.stderr
                self.assertNotEqual(result.returncode, 0, output)
                self.assertIn("PCC GATE FAIL", output)
                self.assertIn(reason, output)

    def test_missing_pp4_cache_fails_before_probe(self):
        result, records = self.run_harness("mistral4", missing_cache=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Stage the PP4", result.stdout + result.stderr)
        self.assertFalse([record for record in records if record["kind"] in ("probe", "runner_launch")])
        mpi_args = self.one_record(records, "mpi")["args"]
        self.assertEqual(mpi_args[mpi_args.index("--host") + 1], "worker:1")

    def test_mistral_rejects_multiple_worker_hosts(self):
        result, records = self.run_harness("mistral4", hosts=["worker0", "worker1"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires exactly one Galaxy host", result.stdout + result.stderr)
        self.assertFalse([record for record in records if record["kind"] in ("mpi", "probe", "runner_launch")])

    def test_runner_failure_after_successful_producer_fails(self):
        result, records = self.run_harness("mistral4", runner_rc=17)
        self.one_record(records, "producer")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
