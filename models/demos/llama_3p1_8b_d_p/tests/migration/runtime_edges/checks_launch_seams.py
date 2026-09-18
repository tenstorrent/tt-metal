import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from edge_guard import check_plan
from runtime_launch import bind_owner_plan, owner_state, validate
from supervision import supervise

HERE = Path(__file__).parent
TMP = None


def base():
    p = json.loads((HERE / "plan.example.json").read_bytes())
    p.update(
        reviewed=True,
        dispatch_enabled=True,
        run_nonce="a" * 32,
        node="bh-glx-120-c06u14",
        job_id="109100",
        lock_path="/tmp/prefill-device-120-c06u14.lock",
    )
    return p


def direct_log():
    return (
        "Opening user mode device driver\nOpening local chip ids/PCIe ids: {"
        + ", ".join(map(str, range(32)))
        + "}\nFabric initialized on 32 devices\nClosing devices in cluster completed.\n"
    )


class LaunchSeamsTests(unittest.TestCase):
    # Only the assigned endpoint changes; retired c10, wrong locks and missing root authorization must fail before native work.
    def test_endpoint_and_closed_binding(self):
        p = base()
        validate(p)
        env = dict(
            SLURM_JOB_ID="109100",
            SLURM_CPUS_PER_TASK="1",
            PREFILL_FABRIC_MODE="1d_ring",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        check_plan(p, env, p["node"])
        for delta in (
            {"node": "bh-glx-110-c10u14"},
            {"job_id": "invalid"},
            {"lock_path": "/wrong"},
            {"reviewed": False},
        ):
            with self.assertRaises((ValueError, RuntimeError)):
                validate(dict(p, **delta))
            with self.assertRaises(ValueError):
                check_plan(dict(p, **delta), env, p["node"])
        with self.assertRaises(RuntimeError):
            validate(dict(p, payload_budget_seconds=1799))

    # A clean runtime receipt requires the exact direct-owner32-chip close; neither report success nor PID exit alone releases the lock.
    def test_cleanup_and_lifecycle_both_required(self):
        p = base()
        report = dict(
            run_nonce=p["run_nonce"],
            scope=p["scope"],
            owner_cleanup_complete=True,
            cleanup_errors=[],
            recovery_required=False,
        )
        with tempfile.TemporaryDirectory(dir=TMP) as tmp:
            f = Path(tmp) / "report.json"
            log = Path(tmp) / "owner.log"
            f.write_text(json.dumps(report))
            log.write_text(direct_log())
            self.assertIn("result", owner_state(f, log, p, "hash"))
            for text in ("", direct_log().replace("32 devices", "31 devices"), direct_log() + direct_log()):
                log.write_text(text)
                self.assertNotIn("result", owner_state(f, log, p, "hash"))
            log.write_text(direct_log())
            f.write_text(json.dumps(dict(report, cleanup_errors=["cache.v failed"], recovery_required=True)))
            self.assertIn("recovery", owner_state(f, log, p, "hash"))

    # Stale nonce, malformed report and a snapshot from another owner plan cannot establish safe completion.
    def test_foreign_or_malformed_report(self):
        p = base()
        with tempfile.TemporaryDirectory(dir=TMP) as tmp:
            f = Path(tmp) / "r"
            log = Path(tmp) / "l"
            log.write_text(direct_log())
            for row in (
                [],
                dict(run_nonce="b" * 32, scope=p["scope"]),
                dict(run_nonce=p["run_nonce"], scope=p["scope"], baseline={"identity": {"plan_sha256": "wrong"}}),
            ):
                f.write_text(json.dumps(row))
                with self.assertRaises(RuntimeError):
                    owner_state(f, log, p, "expected")

    # Node-owned evidence adds only its receipt path/hash; every owner workload/source field stays bound to the root plan.
    def test_owner_receipt_derivation(self):
        p = base()
        receipt = {k: p[k] for k in ("node", "job_id", "run_nonce")}
        with tempfile.TemporaryDirectory(dir=TMP) as tmp:
            path = Path(tmp) / "launch.json"
            path.write_text(json.dumps(receipt))
            bound = bind_owner_plan(p, receipt, path)
            self.assertEqual(
                {k: v for k, v in bound.items() if k not in ("launch_receipt", "launch_receipt_sha256")},
                {k: v for k, v in p.items() if k not in ("launch_receipt", "launch_receipt_sha256")},
            )
            self.assertEqual(bound["launch_receipt_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(RuntimeError):
                bind_owner_plan(p, dict(receipt, job_id="109097"), path)

    # The accepted deadline path records a stop request and retains owner/lock after grace without sending any process signal.
    def test_timeout_retains_live_owner(self):
        process = Mock()
        process.poll.return_value = None
        now = [0.0]
        stops = []

        def sleep(seconds):
            now[0] += seconds

        result = supervise(
            process,
            lambda: {},
            stops.append,
            normal_seconds=1,
            cancel_seconds=1,
            lease_seconds=3,
            recovery_reserve=1,
            clock=lambda: now[0],
            sleep=sleep,
        )
        self.assertEqual(result["phase"], "recovery_hold")
        self.assertFalse(result["release_lock"])
        self.assertTrue(result["buffers_retained"])
        self.assertEqual(len(stops), 1)
        process.kill.assert_not_called()
        process.terminate.assert_not_called()
        process.send_signal.assert_not_called()

    # Early exit without close holds; a failed but fully cleaned owner may release safely while retaining its nonzero result.
    def test_exit_and_clean_failure_are_distinct(self):
        process = Mock()
        process.poll.return_value = 1
        hold = supervise(
            process, lambda: {}, lambda _: None, normal_seconds=1, cancel_seconds=1, lease_seconds=3, recovery_reserve=1
        )
        self.assertFalse(hold["release_lock"])
        done = supervise(
            process,
            lambda: {
                "result": {
                    "owner_cleanup_complete": True,
                    "cleanup_errors": [],
                    "verified_clean_close": "close",
                    "errors": ["primary failure"],
                }
            },
            lambda _: None,
            normal_seconds=1,
            cancel_seconds=1,
            lease_seconds=3,
            recovery_reserve=1,
        )
        self.assertTrue(done["release_lock"])
        self.assertEqual(done["owner_exit"], 1)
        self.assertEqual(done["result"]["errors"], ["primary failure"])

    # Published helper hashes bind the formatted source; inherited flock and one-CPU exec precede owner/native work.
    def test_exact_reuse_and_inherited_lock_route(self):
        origins = json.loads((HERE / "publication-helper-hashes.json").read_text())
        for name, digest in origins.items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest)
        supervisor = (HERE / "node-supervisor.py").read_text()
        tree = ast.parse(supervisor)
        expected_fds = ast.dump(ast.parse("(lock.fileno(),)", mode="eval").body)
        self.assertTrue(
            any(
                isinstance(node, ast.keyword) and node.arg == "pass_fds" and ast.dump(node.value) == expected_fds
                for node in ast.walk(tree)
            )
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Assign)
                and any(ast.unparse(target) == "env['EDGE_NODE_LOCK_FD']" for target in node.targets)
                and ast.unparse(node.value) == "str(lock.fileno())"
                for node in ast.walk(tree)
            )
        )
        self.assertLess(supervisor.index("fcntl.flock"), supervisor.index("subprocess.Popen"))
        controller = (HERE / "controller.py").read_text()
        self.assertIn("single_cpu_exec.py", controller)
        self.assertNotIn('"timeout"', controller)
        for action in ("owner.kill(", "owner.terminate(", "owner.send_signal("):
            self.assertNotIn(action, supervisor)


if __name__ == "__main__":
    unittest.main()
