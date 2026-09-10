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
elif name.startswith("mpirun"):
    record("mpi")
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
        verdict = {"rank": rank, "ok": True, "min_pcc": 0.99, "threshold": 0.85,
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
elif args and (args[0].endswith("summarize_ci_run.py") or args[0].endswith("plot_pipeline_trace.py")):
    pass
elif args == ["-c", "import matplotlib"]:
    pass
else:
    os.execv(os.environ["TEST_REAL_PYTHON"], [os.environ["TEST_REAL_PYTHON"], *args])
"""


class MultirankPccHarnessTests(unittest.TestCase):
    def run_harness(
        self, model, *, verdict="pass", runner_rc=0, missing_cache=False, hosts=None, config=None, extra_env=None
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "records").mkdir()
        bin_dir = root / "bin"
        bin_dir.mkdir()
        for name in ("python3", "mpirun", "mpirun-ulfm", "sleep"):
            executable = bin_dir / name
            executable.write_text(f"#!{sys.executable}\n" + STUB)
            executable.chmod(0o755)
        metal = root / "metal"
        metal.mkdir()
        mgd_dir = metal / "models/demos/common/prefill/runners/topology_configuration/ci"
        mgd_dir.mkdir(parents=True)
        for model_key in ("kimi27", "glm52"):
            for system in ("sc1", "sc4"):
                (mgd_dir / f"{model_key}_{system}_mgd.textproto").touch()
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
            "PREFILL_PCC_GOLDEN_LEN": "56320",
            "PREFILL_MAX_SEQ_LEN": "56320",
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
        self.assertTrue(producer["PREFILL_MIGRATION_DEVICE_MAP_PATH"].endswith("device_map_r0.json"))
        self.assertEqual(
            runner["PREFILL_MIGRATION_DEVICE_MAP_PATH"].replace(".json", "_r0.json"),
            producer["PREFILL_MIGRATION_DEVICE_MAP_PATH"],
        )
        self.assertIn("PCC GATE PASS", result.stdout)

    def test_existing_models_use_discovery_and_four_hosts(self):
        for model in ("kimi27", "glm52"):
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
                expected_length = 256000 if model == "kimi27" else 1049600
                self.assertEqual(producer["PREFILL_MAX_SEQ_LEN"], str(expected_length))
                self.assertEqual(producer["PREFILL_PRODUCER_CHUNKS"], str(expected_length // 5120))

    def test_sc1_windows_preserve_model_defaults(self):
        for model, expected_length in (("mistral4", 56320), ("kimi27", 256000), ("glm52", 256000)):
            with self.subTest(model=model):
                result, records = self.run_harness(model, config="sc1")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.one_record(records, "runner")["env"]["PREFILL_NUM_USERS"], "1")
                self.assertEqual(
                    self.one_record(records, "producer")["env"]["PREFILL_MAX_SEQ_LEN"], str(expected_length)
                )

    def test_perf_gate_fails_when_no_throughput_was_measured(self):
        result, _ = self.run_harness("mistral4", extra_env={"PREFILL_SKIP_PCC": "1", "PREFILL_EXPECTED_TPS": "17000"})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("TPS GATE FAIL", result.stdout + result.stderr)
        self.assertIn("PCC GATE SKIPPED", result.stdout + result.stderr)

    def test_perf_knobs_absent_leaves_pcc_gating_untouched(self):
        result, _ = self.run_harness("mistral4")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("TPS GATE", result.stdout + result.stderr)
        self.assertNotIn("PCC GATE SKIPPED", result.stdout + result.stderr)

    def test_mistral_rejects_sc4(self):
        result, records = self.run_harness("mistral4", config="sc4")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mistral4 PP4 requires config sc1", result.stderr)
        self.assertFalse(records)

    def test_incomplete_or_invalid_verdicts_fail(self):
        for verdict in ("missing", "false", "truthy", "empty", "nan", "wrong_rank"):
            with self.subTest(verdict=verdict):
                result, records = self.run_harness("mistral4", verdict=verdict)
                self.one_record(records, "producer")
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("PCC GATE FAIL", result.stdout + result.stderr)

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
