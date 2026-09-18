"""Closed launch and CPU boundary tests; no scheduler/process/device command executes."""

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import controller

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("capacity_node_entry", HERE / "node-entry.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class LaunchTests(unittest.TestCase):
    # Copying the template or opening only one flag can never dispatch through the contract.
    def test_template_and_half_open_contract_rejected(self):
        plan = json.loads((HERE / "plan.example.json").read_bytes())
        self.assertFalse(plan["reviewed"])
        self.assertFalse(plan["launch_authorized"])
        with self.assertRaises(RuntimeError):
            controller.validate_plan(plan)
        plan["reviewed"] = True
        with self.assertRaises(RuntimeError):
            controller.validate_plan(plan)

    # Shared-job membership is exact, not inferred from an arbitrary node name.
    def test_assignment_identity_is_exact(self):
        plan = dict(
            source=dict(host="node-a"),
            allocation_nodes=["node-a", "node-b"],
            allocation_nodelist="node-a,node-b",
            allocation_owner="fixture-user",
        )
        controller.validate_shared_assignment(plan, "source", "fixture-user|RUNNING|node-a,node-b")
        for bad in (
            "other|RUNNING|node-a,node-b",
            "fixture-user|PENDING|node-a,node-b",
            "fixture-user|RUNNING|node-a,node-c",
        ):
            with self.assertRaises(RuntimeError):
                controller.validate_shared_assignment(plan, "source", bad)

    # Affinity is narrowed before either preflight or supervisor execution, and inherited by both.
    def test_cpu_narrowing_precedes_process_entry(self):
        events = []
        with patch.object(entry.os, "sched_getaffinity", side_effect=[{3, 7}, {3}]), patch.object(
            entry.os, "sched_setaffinity", side_effect=lambda *a: events.append(("affinity", a))
        ), patch.object(
            entry.subprocess, "run", side_effect=lambda *a, **k: events.append(("preflight", a))
        ), patch.object(
            entry.os, "execv", side_effect=lambda *a: events.append(("owner", a))
        ), patch.object(
            sys, "argv", ["entry", "/plan", "hash", "source"]
        ):
            entry.main()
        self.assertEqual([x[0] for x in events], ["affinity", "preflight", "owner"])
        self.assertEqual(events[0][1], (0, {3}))

    # A failed fresh resource/process preflight prevents entering the native supervisor entirely.
    def test_preflight_failure_prevents_owner(self):
        with patch.object(entry.os, "sched_getaffinity", side_effect=[{3}, {3}]), patch.object(
            entry.os, "sched_setaffinity"
        ), patch.object(
            entry.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["preflight"])
        ), patch.object(
            entry.os, "execv"
        ) as execute, patch.object(
            sys, "argv", ["entry", "/plan", "hash", "source"]
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                entry.main()
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
