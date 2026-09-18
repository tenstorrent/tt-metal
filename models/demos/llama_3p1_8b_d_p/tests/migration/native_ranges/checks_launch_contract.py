import ast
import json
import unittest
from pathlib import Path

import controller


class SharedAllocationTests(unittest.TestCase):
    def plan(self):
        return dict(
            allocation_owner="fixture-user",
            allocation_nodelist="bh-glx-120-c02u14,bh-glx-120-c03u14",
            allocation_nodes=["bh-glx-120-c02u14", "bh-glx-120-c03u14"],
            source=dict(host="bh-glx-120-c03u14"),
        )

    # The exact two-node shared allocation is accepted for the selected endpoint.
    def test_exact_shared_allocation_is_accepted(self):
        controller.validate_shared_assignment(
            self.plan(), "source", "fixture-user|RUNNING|bh-glx-120-c02u14,bh-glx-120-c03u14"
        )

    # A truncated or different allocation is rejected before endpoint dispatch.
    def test_wrong_shared_allocation_is_rejected(self):
        with self.assertRaises(RuntimeError):
            controller.validate_shared_assignment(self.plan(), "source", "fixture-user|RUNNING|bh-glx-120-c03u14")

    # Source inspection guards singleton srun flags and separate locks.
    def test_dispatch_is_singleton_and_node_scoped(self):
        ast.parse(Path(controller.__file__).read_text())
        text = Path(controller.__file__).read_text()
        for token in ("--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--nodelist="):
            self.assertIn(token, text)
        plan = json.loads(Path("plan.example.json").read_text())
        self.assertNotEqual(plan["source"]["node_lock"], plan["passive"]["node_lock"])
        self.assertFalse(plan["reviewed"])
        self.assertFalse(plan["launch_authorized"])
