"""Timeout contracts exercised through portable config and actual validator routes."""

import copy
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import controller
import owner_runner
import supervise_owner
from bridge_network import prepare_bridge_config, validate_bridge_network
from checks_portable_contract import HERE, digest, make_plan


def bridge_plan():
    return {
        "capacity": 65536,
        "run_nonce": "a" * 32,
        "etcd_endpoint": "http://source-node:29401",
        "etcd_peer_endpoint": "http://127.0.0.1:29402",
        "source": {
            "host": "source-node",
            "job_id": "123",
            "endpoint_id": 1,
            "control_port": 29403,
            "health_port": 29404,
            "manager_port": 29405,
            "manager_control_port": 29406,
        },
        "passive": {
            "host": "passive-node",
            "job_id": "123",
            "endpoint_id": 2,
            "control_port": 29503,
            "health_port": 29504,
            "manager_port": 29505,
            "manager_control_port": 29506,
        },
    }


def resolved(plan, role):
    peer = plan["passive" if role == "source" else "source"]
    address = "10.0.0.2" if role == "source" else "10.0.0.1"
    rows = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, peer["control_port"]))]
    with patch("bridge_network.socket.gethostname", return_value=plan[role]["host"]), patch(
        "bridge_network.socket.getaddrinfo", return_value=rows
    ):
        return prepare_bridge_config(plan, role)


class TimeoutConfigTests(unittest.TestCase):
    # Omitting overrides must serialize the compatible native defaults for either role.
    def test_omitted_timeouts_reach_both_child_configs(self):
        for role in ("source", "passive"):
            with self.subTest(role=role):
                config, receipt = resolved(bridge_plan(), role)
                self.assertEqual(config.get("inbound_timeout_ms"), 600000)
                self.assertEqual(config.get("bridge_timeout_ms"), 1500000)
                validate_bridge_network(bridge_plan(), role, config, receipt)

    # Reviewed longer waits must reach both actual native inputs without changing their endpoint bindings.
    def test_explicit_timeouts_reach_both_child_configs(self):
        plan = dict(bridge_plan(), inbound_timeout_ms=1800000, bridge_timeout_ms=5400000)
        for role in ("source", "passive"):
            with self.subTest(role=role):
                config, receipt = resolved(plan, role)
                self.assertEqual(config.get("inbound_timeout_ms"), 1800000)
                self.assertEqual(config.get("bridge_timeout_ms"), 5400000)
                validate_bridge_network(plan, role, config, receipt)

    # Reject ambiguous JSON storage and uint32 overflow before a native child can receive the config.
    def test_invalid_native_timeout_storage_and_ranges(self):
        for key in ("inbound_timeout_ms", "bridge_timeout_ms"):
            for value in (True, 1.0, "600000", None, 0, -1, 2**32):
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    resolved(dict(bridge_plan(), **{key: value}), "source")

    # Dropping or changing either serialized wait invalidates both child-config and resolver-receipt proof.
    def test_missing_or_changed_saved_timeouts_rejected(self):
        plan = dict(bridge_plan(), inbound_timeout_ms=1800000, bridge_timeout_ms=5400000)
        config, receipt = resolved(plan, "source")
        for key in ("inbound_timeout_ms", "bridge_timeout_ms"):
            for target in ("config", "receipt"):
                for remove in (False, True):
                    candidate, proof = copy.deepcopy(config), copy.deepcopy(receipt)
                    changed = candidate if target == "config" else proof["config"]
                    if remove:
                        changed.pop(key)
                    else:
                        changed[key] += 1
                    with self.subTest(key=key, target=target, remove=remove), self.assertRaises(RuntimeError):
                        validate_bridge_network(plan, "source", candidate, proof)

    # Controller, supervisor and owner must all enforce the same native deadline and ten-second margin.
    def test_actual_import_routes_enforce_phase_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp))
            plan.update(inbound_timeout_ms=1800000, bridge_timeout_ms=5400000, transfer_phase_timeout_seconds=1790)
            for module in (controller, supervise_owner, owner_runner):
                with self.subTest(module=module.__name__):
                    module.validate_plan(plan)
                    for value in (True, 1790.0, 0, 1791):
                        with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "ten seconds"):
                            module.validate_plan(dict(plan, transfer_phase_timeout_seconds=value))
                    with self.assertRaisesRegex(RuntimeError, "positive uint32"):
                        module.validate_plan(dict(plan, inbound_timeout_ms=True))

    # The actual validator must bind the new helper; absent or stale helper bytes cannot be approved.
    def test_timeout_helper_missing_or_changed_pin_rejected(self):
        helper = str(HERE / "timeout_config.py")
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp))
            controller.validate_plan(plan)
            self.assertEqual(plan["pins"][helper], digest(helper))
            for remove in (False, True):
                bad = copy.deepcopy(plan)
                if remove:
                    bad["pins"].pop(helper)
                else:
                    bad["pins"][helper] = "0" * 64
                with self.subTest(remove=remove), self.assertRaisesRegex(
                    RuntimeError, "Unpinned helper timeout_config"
                ):
                    controller.validate_plan(bad)

    # The closed portable example provides explicit defaults that its real config builder can consume.
    def test_closed_example_timeout_defaults_are_consumable(self):
        example = json.loads((HERE / "plan.example.json").read_text())
        self.assertFalse(example["reviewed"])
        self.assertFalse(example["launch_authorized"])
        self.assertFalse(example["manager_memory_reviewed"])
        plan = dict(
            bridge_plan(),
            inbound_timeout_ms=example["inbound_timeout_ms"],
            bridge_timeout_ms=example["bridge_timeout_ms"],
        )
        config, receipt = resolved(plan, "source")
        self.assertEqual(config["inbound_timeout_ms"], 600000)
        self.assertEqual(config["bridge_timeout_ms"], 1500000)
        validate_bridge_network(plan, "source", config, receipt)


if __name__ == "__main__":
    unittest.main()
