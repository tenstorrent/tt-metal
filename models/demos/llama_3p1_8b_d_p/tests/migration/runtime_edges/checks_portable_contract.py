"""Configured assignment and actual controller imports, without native dependencies."""
import importlib.abc
import json
import sys
import unittest
from pathlib import Path


class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "ttnn", "tt_lib", "tt_d_gen", "transformers"}:
            raise AssertionError("native import forbidden: " + fullname)
        return None


sys.meta_path.insert(0, BlockNative())
import controller
from edge_guard import check_plan
from runtime_launch import validate
from slurm_scope import target_step_lines, validate_allocation


class PortableRuntimeTests(unittest.TestCase):
    # A bound plan is portable across assigned nodes but actual host/job and physical lock must agree.
    def test_configured_assignment_rejects_wrong_host_job_lock(self):
        p = json.loads(Path("plan.example.json").read_text())
        p.update(
            reviewed=True,
            dispatch_enabled=True,
            node="bh-glx-120-fixture",
            job_id="123",
            lock_path="/tmp/prefill-device-120-fixture.lock",
            run_nonce="a" * 32,
        )
        env = dict(
            SLURM_JOB_ID="123",
            SLURM_CPUS_PER_TASK="1",
            PREFILL_FABRIC_MODE="1d_ring",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        validate(p)
        check_plan(p, env, p["node"])
        for wrong in (dict(job_id="124"), dict(node="bh-glx-120-other"), dict(lock_path="/tmp/wrong.lock")):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                check_plan(dict(p, **wrong), env, p["node"])

    # Read-only Slurm parsers retain exact configured ownership and do not reject an unrelated node's step.
    def test_scheduler_owner_and_target_step_parser(self):
        runner = lambda argv: ["bh-glx-120-fixture" if argv[-1] == "fixture" else "bh-glx-120-other"][0]
        validate_allocation("fixture-user|RUNNING|fixture", "bh-glx-120-fixture", runner, expected_owner="fixture-user")
        with self.assertRaises(ValueError):
            validate_allocation("other|RUNNING|fixture", "bh-glx-120-fixture", runner, expected_owner="fixture-user")
        self.assertEqual(target_step_lines("123.1|other-task|other", "bh-glx-120-fixture", runner), [])

    # The actual controller imports this directory's validation functions, not a historical private packet.
    def test_controller_resolves_local_contract(self):
        self.assertIs(controller.validate, validate)
        self.assertEqual(Path(controller.__file__).resolve().parent, Path(__file__).resolve().parent)


if __name__ == "__main__":
    unittest.main()
