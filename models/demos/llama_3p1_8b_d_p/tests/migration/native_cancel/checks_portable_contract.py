"""Configured host-only plans exercise actual CLI import routes without native work."""

import copy
import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(root):
    plan = json.loads((HERE / "plan.example.json").read_bytes())
    plan.update(
        reviewed=True,
        launch_authorized=True,
        run_nonce="a" * 32,
        run_dir=str(root / "run"),
        allocation_owner="fixture-user",
        allocation_nodes=["bh-glx-120-source", "bh-glx-120-passive"],
        allocation_nodelist="bh-glx-120-source,bh-glx-120-passive",
        etcd_endpoint="http://bh-glx-120-source:29401",
    )
    pins = {str(p): digest(p) for p in HERE.glob("*.py") if not p.name.startswith(("checks_", "test_"))}
    plan["pins"] = pins

    def file(name, data=b"host-only fixture"):
        p = root / name
        p.write_bytes(data)
        pins[str(p)] = digest(p)
        return str(p)

    for key in (
        "accepted_gate_spec",
        "accepted_source_pins",
        "input_ids",
        "restart_input_ids",
        "runtime_edge_evidence",
        "cancel_spec",
        "seed_helper",
        "environment_script",
        "manager_binary",
        "dmk_elf",
        "source_client",
        "passive_client",
        "etcd_binary",
        "native_probe",
        "startup_evidence",
        "bridge_manifest",
    ):
        plan[key] = file(key)
    plan["bridge_manifest_sha256"] = pins[plan["bridge_manifest"]]
    for index, role in enumerate(("source", "passive")):
        plan[role].update(
            host="bh-glx-120-" + role,
            job_id="123",
            node_lock="/tmp/prefill-device-120-" + role + ".lock",
            endpoint_id=index + 1,
            health_receipt=file(role + "-health"),
        )
    plan["manager_libraries"] = {}
    for name in ("libtt_metal.so", "libtt-umd.so", "libtt_stl.so", "libtracy.so"):
        path = file(name)
        plan["manager_libraries"][name] = dict(path=path, sha256=pins[path])
    receipt = dict(
        reviewed=True,
        actual_exit=0,
        verified_exit=0,
        cpp_cases_passed=48,
        no_metal_umd_linkage=True,
        bridge_manifest_sha256=plan["bridge_manifest_sha256"],
        binaries={plan[k]: pins[plan[k]] for k in ("source_client", "passive_client")},
    )
    plan["bridge_host_validation"] = file("host-validation.json", json.dumps(receipt).encode())
    return plan


class PortableContractTests(unittest.TestCase):
    def reject_at_entry(self, name, root, plan):
        module = importlib.import_module(name)
        path = root / "plan.json"
        path.write_text(json.dumps(plan))
        argv = [module.__file__, "--plan", str(path), "--plan-sha256", digest(path), "--role", "source"]
        with patch.object(sys, "argv", argv), patch(
            "socket.gethostname", side_effect=AssertionError("endpoint guard reached before contract rejection")
        ), patch("subprocess.Popen", side_effect=AssertionError("child forbidden")):
            with self.assertRaises(RuntimeError):
                module.main()
        self.assertFalse(Path(plan["run_dir"]).exists())

    # Configurable manifest identities remain hash-bound through both actual entrypoints.
    def test_wrong_manifest_rejected_by_owner_and_supervisor(self):
        for name in ("owner_runner", "supervise_owner"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan = make_plan(root)
                plan["bridge_manifest_sha256"] = "0" * 64
                self.reject_at_entry(name, root, plan)

    # Rehashing a false case-count, linkage or binary receipt cannot bypass the native gate.
    def test_rebound_bad_host_receipt_rejected_by_actual_import_routes(self):
        for name in ("owner_runner", "supervise_owner"):
            for mutation in ("cases", "linkage", "binary"):
                with self.subTest(name=name, mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    plan = make_plan(root)
                    path = Path(plan["bridge_host_validation"])
                    receipt = json.loads(path.read_text())
                    if mutation == "cases":
                        receipt["cpp_cases_passed"] = 34
                    elif mutation == "linkage":
                        receipt["no_metal_umd_linkage"] = False
                    else:
                        receipt["binaries"][plan["source_client"]] = "0" * 64
                    path.write_text(json.dumps(receipt))
                    plan["pins"][str(path)] = digest(path)
                    self.reject_at_entry(name, root, plan)

    # The environment and actual imported helper must both be pinned before any native entry.
    def test_missing_environment_and_changed_imported_helper_rejected(self):
        for mutation in ("environment", "helper"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan = make_plan(root)
                if mutation == "environment":
                    del plan["pins"][plan["environment_script"]]
                else:
                    plan["pins"][str(HERE / "cancel_restart_contract.py")] = "0" * 64
                self.reject_at_entry("owner_runner", root, plan)

    # Configured allocation ownership must match exactly; role-specific physical locks stay strict.
    def test_configured_assignment_and_wrong_role_lock(self):
        import controller
        from transfer_contract import validate_plan

        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp))
            validate_plan(plan)
            controller.validate_shared_assignment(plan, "source", "fixture-user|RUNNING|" + plan["allocation_nodelist"])
            with self.assertRaises(RuntimeError):
                controller.validate_shared_assignment(plan, "source", "other|RUNNING|" + plan["allocation_nodelist"])
            bad = copy.deepcopy(plan)
            bad["source"]["node_lock"] = bad["passive"]["node_lock"]
            with self.assertRaises(RuntimeError):
                validate_plan(bad)
