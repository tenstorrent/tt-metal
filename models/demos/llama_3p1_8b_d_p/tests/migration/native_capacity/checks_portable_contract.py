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
        scope="real_prefill_capacity_selected_ranges",
        capacity=4096,
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
        "book_manifest",
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
    plan.update(
        transfer_phase_timeout_seconds=590,
        warmup_policy="both_prompt_geometries_before_native_clients",
        source_setup_timeout_seconds=300,
        warmup_timeout_seconds=600,
        supervisor=dict(normal_seconds=1800, cancel_seconds=420, recovery_reserve_seconds=1800),
        manager_memory_reviewed=True,
        minimum_host_available_bytes=128 * 1024**3,
        minimum_shared_disk_free_bytes=32 * 1024**3,
        manager_rss_limit_bytes=dict(source=16 * 1024**3, passive=16 * 1024**3),
        owner_mode="real_h2d_prefill",
        full_cache_snapshots=False,
    )
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
        cpp_cases_passed=54,
        no_metal_umd_linkage=True,
        binaries={plan[k]: pins[plan[k]] for k in ("source_client", "passive_client")},
    )
    plan["bridge_host_validation"] = file("host-validation.json", json.dumps(receipt).encode())
    return plan


class PortableContractTests(unittest.TestCase):
    # Configured manifest paths work only with the same hash-bound54-case/binary receipt.
    def test_configured_manifest_and_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp))
            validate_plan(plan)
            for key, value in [("bridge_manifest_sha256", "0" * 64), ("capacity", 131072)]:
                with self.subTest(key=key), self.assertRaises((RuntimeError, ValueError)):
                    validate_plan(dict(plan, **{key: value}))

    # Rehashing a bad receipt cannot hide a wrong test inventory or binary identity.
    def test_wrong_receipt_case_inventory_linkage_and_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp))
            path = Path(plan["bridge_host_validation"])
            original = json.loads(path.read_text())
            changes = [
                dict(cpp_cases_passed=34),
                dict(cpp_cases_passed=48),
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
                for host, job in (("wrong-node", "123"), (plan["source"]["host"], "999")):
                    with self.subTest(module=module.__name__, host=host, job=job), patch.object(
                        sys, "argv", argv
                    ), patch("socket.gethostname", return_value=host), patch.dict(os.environ, SLURM_JOB_ID=job), patch(
                        "subprocess.Popen", side_effect=AssertionError("child forbidden")
                    ):
                        with self.assertRaisesRegex(RuntimeError, "Wrong assigned (node|host|job)"):
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

    # The actual entry must retain one-CPU preflight ordering and use its sibling supervisor.
    def test_node_entry_resolves_local_helper_closure(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("portable_capacity_entry", HERE / "node-entry.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
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
        self.assertIn(str(HERE / "node-preflight.py"), events[1][1][0])
        self.assertIn(str(HERE / "supervise_owner.py"), events[2][1][1])

    # Both import routes must reach this capacity contract, not a retired range/34-case adapter.
    def test_actual_transitive_validator_identity(self):
        import runner_support
        import verify_capacity
        import verify_final

        self.assertIs(verify_final.verify, verify_capacity.verify)
        self.assertIs(controller.validate_plan, runner_support.validate_plan)
        self.assertIs(supervise_owner.validate_plan, runner_support.validate_plan)
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp))
            for module in (controller, supervise_owner, owner_runner):
                module.validate_plan(plan)
                with self.assertRaises((RuntimeError, ValueError)):
                    module.validate_plan(dict(plan, capacity=131072))

    # A tampered environment file must fail before its shell body or any node entry runs.
    def test_environment_digest_checked_before_source(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "must-not-exist"
            environment = root / "environment.sh"
            environment.write_text("touch " + str(marker) + "\n")
            plan = dict(environment_script=str(environment), pins={str(environment): "0" * 64})
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            result = subprocess.run(
                ["bash", str(HERE / "node-run.sh"), str(path), digest(path), "source"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())

    # Temporary host tokens exercise the real configured manifest seam and reject changed token bytes.
    def test_configured_token_manifest_hashes_and_distinctness(self):
        from capacity_execution import make_cases
        from checks_fixtures import token_manifest

        with token_manifest() as (path, expected):
            doc, fixtures = make_cases(4096, path, expected)
            self.assertEqual([x["valid_prompt_tokens"] for x in doc["phases"]], [4096, 4064])
            self.assertNotEqual(fixtures["slot0"][:4064], fixtures["slot1"])
            token_file = path.parent / json.loads(path.read_text())["fixtures"][0]["token_ids_file"]
            token_file.write_text("[]")
            with self.assertRaisesRegex(ValueError, "Token bytes changed"):
                make_cases(4096, path, expected)
            with self.assertRaisesRegex(ValueError, "Book manifest changed"):
                make_cases(4096, path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
