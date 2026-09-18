"""Cheap host guards for the newly wired epoch boundaries, not native validation."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from epoch_contract import check_fresh_identities, check_role_events, epoch_nonce, epoch_plan
from page_io import PAGE
from runner_support import sha256, write_json
from transfer_lifetime import owner_started, peer_stopped, wait_peer_stopped
from verify_final import page_file

HERE = Path(__file__).resolve().parent


def example_plan():
    plan = json.loads((HERE / "plan.example.json").read_bytes())
    plan["run_nonce"] = "a" * 32
    plan["input_ids"] = str(HERE / "input-ids.json")
    plan["restart_input_ids"] = str(HERE / "restart-fixtures.json")
    return plan


CYCLE = "Cluster constructor started.\nCluster constructor completed.\nCluster destructor started.\nCluster destructor completed.\n"
EVENTS = [
    "epoch-a-started",
    "epoch-a-drained",
    "epoch-a-stopped",
    "epoch-a-pair-stopped",
    "sentinel-ready",
    "epoch-b-started",
    "epoch-b-drained",
    "epoch-b-stopped",
    "epoch-b-pair-stopped",
    "cache-release",
]


class EpochGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.run.mkdir()
        self.plan = dict(
            run_dir=str(self.run),
            run_nonce="a" * 32,
            source=dict(host="source", job_id="1"),
            passive=dict(host="passive", job_id="1"),
        )
        self.owner = dict(pid=11, start_ticks=12)
        for role in ("source", "passive"):
            (self.run / (role + "-supervisor")).mkdir()
            write_json(
                self.run / (role + "-supervisor") / "started.json",
                dict(run_nonce="a" * 32, role=role, owner=self.owner),
            )

    def stopped(self, epoch, role="passive"):
        ep = epoch_plan(self.plan, epoch)
        out = Path(ep["run_dir"]) / role
        out.mkdir(parents=True, exist_ok=True)
        manager = dict(pid=20 if epoch == "a" else 30, start_ticks=21 if epoch == "a" else 31)
        write_json(
            out / "manager-started.json",
            dict(run_nonce=ep["run_nonce"], role=role, owner=self.owner, endpoint=ep[role], manager=manager),
        )
        (out / "manager.log").write_text(CYCLE)
        row = dict(
            run_nonce=ep["run_nonce"],
            role=role,
            owner=self.owner,
            endpoint=ep[role],
            ok=True,
            manager=dict(manager, exit_code=0),
            manager_log=dict(path=str(out / "manager.log"), sha256=sha256(out / "manager.log")),
            manager_never_started=False,
            native_io_stopped=True,
            device_reset_performed=False,
        )
        write_json(out / "native-stopped.json", row)
        return ep, out, row

    # The original supervisor record stays untouched while each epoch gets a distinct namespace.
    def test_epoch_nonce_and_actual_supervisor_binding(self):
        a, b = epoch_plan(self.plan, "a"), epoch_plan(self.plan, "b")
        self.assertNotEqual(a["run_nonce"], b["run_nonce"])
        self.assertNotEqual(a["run_nonce"], self.plan["run_nonce"])
        self.assertEqual(owner_started(a, "source")["run_nonce"], self.plan["run_nonce"])
        self.assertNotIn("owner_supervision", self.plan)

    # A clean cancelled epoch must not satisfy the future epoch's release barrier.
    def test_prior_stop_cannot_release_epoch_b(self):
        a, _, _ = self.stopped("a")
        self.assertIsNotNone(peer_stopped(a, "source"))
        self.assertIsNone(peer_stopped(epoch_plan(self.plan, "b"), "source"))

    # A delayed peer stop times out instead of manufacturing cleanup permission.
    def test_delayed_peer_keeps_stop_unproven(self):
        ep = epoch_plan(self.plan, "b")
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        with self.assertRaises(TimeoutError):
            wait_peer_stopped(ep, "source", timeout=0.2, clock=lambda: clock[0], sleep=sleep)
        self.assertGreaterEqual(clock[0], 0.2)

    # Both the manager generation and actual outer owner must match the stop receipt.
    def test_valid_stop_then_wrong_start_ticks_rejected(self):
        ep, out, row = self.stopped("b")
        self.assertIsNotNone(peer_stopped(ep, "source"))
        row["manager"]["start_ticks"] += 1
        (out / "native-stopped.json").write_text(json.dumps(row))
        with self.assertRaises(RuntimeError):
            peer_stopped(ep, "source")

    # A stale nonce with a clean log cannot validate a restarted manager.
    def test_stale_epoch_nonce_rejected(self):
        ep, out, row = self.stopped("b")
        row["run_nonce"] = epoch_nonce(self.plan["run_nonce"], "a")
        (out / "native-stopped.json").write_text(json.dumps(row))
        with self.assertRaises(RuntimeError):
            peer_stopped(ep, "source")

    # Claimed outer ownership cannot redirect the reader to unrelated supervisor receipts.
    def test_epoch_cannot_escape_outer_run(self):
        ep = epoch_plan(self.plan, "b")
        ep["owner_supervision"]["run_dir"] = str(self.run / "other")
        with self.assertRaises(RuntimeError):
            owner_started(ep, "source")

    # Every role drains before its stop, and neither cache may release before both epochs finish.
    def test_role_order_accepts_only_complete_sequence(self):
        check_role_events(EVENTS)
        for a, b in [(1, 2), (3, 4), (5, 7), (8, 9)]:
            changed = list(EVENTS)
            changed[a], changed[b] = changed[b], changed[a]
            with self.assertRaises(RuntimeError):
                check_role_events(changed)

    # Fresh manager and bridge identities are required for both roles, not only the source.
    def test_both_roles_need_fresh_manager_and_bridge(self):
        old = {
            k: {r: dict(pid=10 + i + j * 10, start_ticks=100 + i + j * 10) for i, r in enumerate(("source", "passive"))}
            for j, k in enumerate(("manager", "bridge"))
        }
        new = {
            k: {r: dict(pid=v["pid"] + 100, start_ticks=v["start_ticks"] + 100) for r, v in rows.items()}
            for k, rows in old.items()
        }
        check_fresh_identities(old, new)
        for kind in old:
            for role in old[kind]:
                bad = copy.deepcopy(new)
                bad[kind][role] = old[kind][role]
                with self.assertRaises(RuntimeError):
                    check_fresh_identities(old, bad)


class FileAndScopeGuards(unittest.TestCase):
    # The verifier reconstructs each of512 group hashes from actual raw bytes, rejecting a fabricated digest map.
    def test_saved_page_inventory_and_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "source"
            output.mkdir()
            path = output / "selected-before.bin"
            raw = b"".join(bytes([i % 251 + 1]) * PAGE for i in range(512))
            path.write_bytes(raw)
            groups = {
                f"{c}:{l}": hashlib.sha256(raw[(c * 32 + l) * PAGE : (c * 32 + l + 1) * PAGE]).hexdigest()
                for c in range(16)
                for l in range(32)
            }
            row = dict(
                path=str(path), pages=512, bytes=len(raw), slot=0, begin=0, end=32, sha256=sha256(path), groups=groups
            )
            write_json(output / "selected-before.json", row)
            self.assertEqual(page_file(output, "selected-before"), (raw, groups))
            row["groups"]["15:31"] = "0" * 64
            (output / "selected-before.json").write_text(json.dumps(row))
            with self.assertRaises(RuntimeError):
                page_file(output, "selected-before")

    # A closed plan fails before reading health or opening any process/device.
    def test_closed_template_rejected(self):
        from transfer_contract import validate_plan

        plan = example_plan()
        with self.assertRaisesRegex(RuntimeError, "Closed"):
            validate_plan(plan)
        self.assertIsNone(plan["source"]["health_receipt"])
        self.assertIsNone(plan["passive"]["health_receipt"])

    # The frozen real input must distinguish the restart's entire first valid page;
    # using a repeated epoch-A prompt would invalidate the write-progress oracle.
    def test_real_fixture_distinguishes_restart(self):
        from runtime_cancel import prompt_pair

        plan = example_plan()
        a, b = prompt_pair(
            json.loads(Path(plan["input_ids"]).read_bytes()), json.loads(Path(plan["restart_input_ids"]).read_bytes())
        )
        self.assertEqual((len(a), len(b)), (2048, 32))
        self.assertNotEqual(a[:32], b)

    # A closed invocation must reject before it can create a controller child or write launch state.
    def test_actual_wrapper_rejects_closed_plan_before_spawn(self):
        import runpy
        import sys

        path = HERE / "plan.example.json"
        with patch.object(
            sys, "argv", [str(HERE / "launch-wrapper.py"), "--plan", str(path), "--plan-sha256", sha256(path)]
        ), patch("subprocess.Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "unarmed"):
                runpy.run_path(str(HERE / "launch-wrapper.py"), run_name="__main__")
            spawn.assert_not_called()

    # Exercise both real entrypoint import chains. Armed integrated48 validation must
    # succeed before the wrong-host guard rejects; no process, lock or native import is reached.
    def test_supervisor_and_owner_reach_host_guard_with_integrated48(self):
        import importlib
        import sys

        from checks_portable_contract import make_plan

        with tempfile.TemporaryDirectory() as directory:
            plan = make_plan(Path(directory))
            self.assertEqual(json.loads(Path(plan["bridge_host_validation"]).read_bytes())["cpp_cases_passed"], 48)
            path = Path(directory) / "reviewed.json"
            path.write_text(json.dumps(plan))
            for name in ("supervise_owner", "owner_runner"):
                module = importlib.import_module(name)
                with patch.object(
                    sys, "argv", [name, "--plan", str(path), "--plan-sha256", sha256(path), "--role", "source"]
                ), patch.object(module.socket, "gethostname", return_value="wrong-host"), patch(
                    "subprocess.Popen"
                ) as spawn:
                    with self.assertRaisesRegex(RuntimeError, "Wrong assigned"):
                        module.main()
                    spawn.assert_not_called()
        import transfer_contract

        self.assertEqual(Path(transfer_contract.__file__).resolve(), HERE / "transfer_contract.py")

    # Both epochs must have distinct discovery identities but identical assigned ports
    # and the correct epoch-local source/passive tables, regardless of stale parent variables.
    def test_actual_manager_environment_for_both_roles_and_epochs(self):
        import os

        from manager_environment import manager_environment

        plan = example_plan()
        names = set()
        for epoch in ("a", "b"):
            ep = epoch_plan(plan, epoch)
            for role in ("source", "passive"):
                peer = "passive" if role == "source" else "source"
                local = ep[role]
                with patch("socket.gethostname", return_value=local["host"]), patch.dict(
                    os.environ, {"KV_MANAGER_STALE": "bad", "KVM_ID": "old"}
                ):
                    env = manager_environment(
                        ep, role, Path(ep["run_dir"]), Path(ep["run_dir"]) / role, list(range(32))
                    )
                self.assertNotIn("KV_MANAGER_STALE", env)
                self.assertEqual(env["KVM_ID"], "llama-" + ep["run_nonce"] + "-" + role)
                names.add(env["KVM_ID"])
                self.assertEqual(env["PEERS"], "llama-" + ep["run_nonce"] + "-" + peer)
                self.assertEqual(env["KV_MANAGER_DEVICE_IDS"], ",".join(map(str, range(32))))
                self.assertEqual(
                    env["KV_MANAGER_PREFILL_KV_CHUNK_TABLE_PATH"], str(Path(ep["run_dir"]) / "source/table.pb")
                )
                self.assertEqual(
                    env["KV_MANAGER_DECODE_KV_CHUNK_TABLE_PATH"], str(Path(ep["run_dir"]) / "passive/table.pb")
                )
                self.assertEqual(env["KV_MANAGER_TRANSPORT_ENDPOINT"], f"tcp://127.0.0.1:{local['manager_port']}")
                self.assertEqual(
                    env["KV_MANAGER_CONTROL_MSG_ENDPOINT"], f"tcp://0.0.0.0:{local['manager_control_port']}"
                )
                self.assertEqual(env["KV_MANAGER_ETCD_ENDPOINT"], plan["etcd_endpoint"])
        self.assertEqual(len(names), 4)

    # The exact bridge configuration resolves the peer, not the self hostname, and
    # preserves the epoch nonce and production manager/client listener ports.
    def test_actual_bridge_network_for_both_roles_and_epochs(self):
        import socket

        from bridge_network import prepare_bridge_config, validate_bridge_network

        plan = example_plan()
        for epoch in ("a", "b"):
            ep = epoch_plan(plan, epoch)
            for role in ("source", "passive"):
                peer = "passive" if role == "source" else "source"
                answers = [
                    (
                        socket.AF_INET,
                        socket.SOCK_STREAM,
                        socket.IPPROTO_TCP,
                        "",
                        ("172.20.1.2", ep[peer]["control_port"]),
                    )
                ]
                with patch("socket.gethostname", return_value=ep[role]["host"]), patch(
                    "socket.getaddrinfo", return_value=answers
                ) as resolve:
                    cfg, receipt = prepare_bridge_config(ep, role)
                self.assertEqual(resolve.call_args.args[:2], (ep[peer]["host"], ep[peer]["control_port"]))
                self.assertEqual(receipt["run_nonce"], ep["run_nonce"])
                validate_bridge_network(ep, role, dict(cfg, journal="/future/journal"), receipt)
                bad = dict(cfg, peer_port=cfg["peer_port"] + 1)
                with self.assertRaises(RuntimeError):
                    validate_bridge_network(ep, role, bad, receipt)

    # Affinity is set in the exec parent so the retained owner and all its children inherit one CPU.
    def test_affinity_exec_precedes_supervisor(self):
        shell = (HERE / "node-run.sh").read_text()
        self.assertIn("single_cpu_exec.py", shell)
        source = (HERE / "single_cpu_exec.py").read_text()
        self.assertLess(source.index("sched_setaffinity"), source.index("os.execv"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
