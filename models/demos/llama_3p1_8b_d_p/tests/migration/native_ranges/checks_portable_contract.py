"""Portable configuration tests; only stdlib fixtures and actual Python contracts."""
import copy
import hashlib
import importlib.abc
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "ttnn", "tt_lib", "tt_d_gen", "transformers"}:
            raise AssertionError("native import forbidden in publication checks: " + fullname)
        return None


sys.meta_path.insert(0, BlockNative())
import controller
import owner_runner
import supervise_owner
from transfer_contract import validate_plan

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(root):
    pins = {str(p): digest(p) for p in HERE.glob("*.py") if not p.name.startswith(("test_", "checks_"))}

    def file(name, data=b"host-only fixture"):
        p = root / name
        p.write_bytes(data)
        pins[str(p)] = digest(p)
        return str(p)

    plan = dict(
        reviewed=True,
        launch_authorized=True,
        scope="2k_paired_selected_ranges",
        capacity=2048,
        run_nonce="a" * 32,
        run_dir=str(root / "never-created"),
        allocation_owner="fixture-user",
        allocation_nodes=["bh-glx-120-source", "bh-glx-120-passive"],
        allocation_nodelist="bh-glx-120-[source,passive]",
        etcd_endpoint="http://bh-glx-120-source:29401",
        etcd_peer_endpoint="http://127.0.0.1:29402",
        pins=pins,
    )
    for key in (
        "accepted_gate_spec",
        "accepted_source_pins",
        "scenario",
        "fixtures",
        "decoder_source",
        "seed_helper",
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
        plan[role] = dict(
            host="bh-glx-120-" + role,
            job_id="123",
            node_lock="/tmp/prefill-device-120-" + role + ".lock",
            endpoint_id=index + 1,
            health_receipt=file(role + "-health"),
            control_port=29403,
            health_port=29404,
            manager_port=29405,
            manager_control_port=29406,
        )
    plan["manager_libraries"] = {
        name: dict(path=(p := file(name)), sha256=pins[p])
        for name in ("libtt_metal.so", "libtt-umd.so", "libtt_stl.so", "libtracy.so")
    }
    receipt = dict(
        reviewed=True,
        actual_exit=0,
        verified_exit=0,
        bridge_manifest_sha256=plan["bridge_manifest_sha256"],
        cpp_cases_passed=48,
        no_metal_umd_linkage=True,
        binaries={plan[k]: pins[plan[k]] for k in ("source_client", "passive_client")},
    )
    plan["bridge_host_validation"] = file("host-validation.json", json.dumps(receipt).encode())
    return plan


class PortableContractTests(unittest.TestCase):
    # Configured manifest paths work only with the same hash-bound48-case/binary receipt.
    def test_configured_manifest_and_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp))
            validate_plan(plan)
            for key, value in [("bridge_manifest_sha256", "0" * 64), ("capacity", 4096)]:
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    validate_plan(dict(plan, **{key: value}))

    # Rehashing a bad receipt cannot hide a wrong test inventory or binary identity.
    def test_wrong_receipt_case_inventory_linkage_and_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp))
            path = Path(plan["bridge_host_validation"])
            original = json.loads(path.read_text())
            changes = [
                dict(cpp_cases_passed=34),
                dict(no_metal_umd_linkage=False),
                dict(bridge_manifest_sha256="0" * 64),
                dict(
                    binaries={
                        plan["source_client"]: "0" * 64,
                        plan["passive_client"]: plan["pins"][plan["passive_client"]],
                    }
                ),
            ]
            for change in changes:
                path.write_text(json.dumps(dict(original, **change)))
                plan["pins"][str(path)] = digest(path)
                with self.subTest(change=change), self.assertRaises(RuntimeError):
                    validate_plan(plan)

    # Both exact role CLIs validate the same contract before host/job checks or child creation.
    def test_actual_owner_and_supervisor_stop_before_native_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = make_plan(root)
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            for module in (owner_runner, supervise_owner):
                argv = [module.__file__, "--plan", str(path), "--plan-sha256", digest(path), "--role", "source"]
                with self.subTest(module=module.__name__), patch.object(sys, "argv", argv), patch(
                    "socket.gethostname", return_value="wrong-node"
                ), patch("subprocess.Popen", side_effect=AssertionError("child forbidden")):
                    with self.assertRaisesRegex(RuntimeError, "Wrong assigned (node|host)"):
                        module.main()
            self.assertFalse(Path(plan["run_dir"]).exists())

    # Foreign allocation owner and role-specific lock paths are rejected by the real guards.
    def test_assignment_owner_and_physical_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp))
            controller.validate_shared_assignment(plan, "source", "fixture-user|RUNNING|" + plan["allocation_nodelist"])
            with self.assertRaises(RuntimeError):
                controller.validate_shared_assignment(
                    plan, "source", "other-user|RUNNING|" + plan["allocation_nodelist"]
                )
            for change in (
                dict(host="wrong"),
                dict(job_id="not-a-job"),
                dict(node_lock="/tmp/prefill-device-120-passive.lock"),
            ):
                bad = copy.deepcopy(plan)
                bad["source"].update(change)
                with self.subTest(change=change), self.assertRaises(RuntimeError):
                    validate_plan(bad)

    # A caller's working directory must not select a different token fixture.
    def test_scenario_fixture_is_sibling_relative(self):
        from range_contract import scenario

        old = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                value = scenario(HERE / "scenario.json")
                self.assertEqual(value["expected"]["compute_calls"], 7)
                self.assertEqual(value["expected"]["layer_acks"], 224)
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
