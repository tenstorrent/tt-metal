#!/usr/bin/env python3
"""One normal Slurm assignment, declared producer and serial arms. Root review required."""
import argparse
import importlib.util
import configparser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

import owned_seed_copy as owned
import slurm_pair_checks as checks
import producer_profile as profiles
import seed_storage
import host_fd_probe
import hf_local_stage
import inner_host_quiet

BASELINE = "21754c005c015161fe5de2d1cdb10bf64a6fd5a1"
CANDIDATE = "ee072dd216c2d19c6b228be26f695757d8ec1500"
PLUGIN = "c1c85eb6bfe14a3e47afc450e25f7a8ddc81d974"
VLLM = "568afb3a13806beb53bb2e6bd518269357b237c0"
IMAGE = "ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64@sha256:3643bc059bf70bd14d3031f5f53f8853abc564cd2a5d9a3dcff1b54c7003e25b"
MEMBERS = {
    "native": {
        "path": "ttm_any.tar.zst",
        "bytes": 158000635,
        "sha256": "18e4ac99c2f4b19ef8576d5d989580f6a1560b50c60515fe547e0cef4969a33b",
    },
    "wheel": {
        "path": "ttnn-0.75.0rc10.dev2408+gee072dd216-cp310-cp310-manylinux_2_34_x86_64.whl",
        "bytes": 99960835,
        "sha256": "3e18daf10bf0d52e95ec46745d8905cff63e114786504af4a1ac864c0a4bbe28",
    },
}
cancelled = False


def interrupt(signum, frame):
    global cancelled
    cancelled = True


def git(path, *args, seconds=30):
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.check_output(["git", "-C", str(path), *args], text=True, env=env, timeout=seconds).strip()


def public_git_config(path):
    cfg = configparser.RawConfigParser(strict=False)
    cfg.read(path)
    for section in cfg.sections():
        assert not section.startswith(("credential", "http")), "Private Git configuration"
        for key, value in cfg.items(section):
            assert key.lower() not in {"fsmonitor", "hookspath", "sshcommand"}, "Executable Git configuration"
            if key.lower() == "url":
                assert value.startswith("https://github.com/") and "@" not in value, "Only public HTTPS source remotes"


def validate_source(capsule, deadline=None):
    # Full status, including ignored/untracked paths, is retained on shared storage.
    # Only its source-check timeout may grow; the original preparation end caps it.
    def source_git(root, *args):
        seconds = 30
        if deadline is not None:
            assert not cancelled, "Source verification cancelled"
            remaining = deadline - time.monotonic()
            assert remaining > 0, "Original preparation deadline exhausted"
            seconds = min(120 if args[0] == "status" else 30, remaining)
        result = git(root, *args, seconds=seconds)
        if deadline is not None:
            assert not cancelled and time.monotonic() < deadline, "Original preparation deadline/cancellation"
        return result

    heads = {"baseline": BASELINE, "candidate": CANDIDATE, "plugin": PLUGIN, "vllm-source": VLLM}
    for name, revision in heads.items():
        root = capsule / name
        common = Path(source_git(root, "rev-parse", "--git-common-dir"))
        common = common if common.is_absolute() else root / common
        assert common.resolve().is_relative_to(capsule.resolve()), "External Git object directory is not sealed"
        assert not (common / "objects/info/alternates").exists()
        public_git_config(common / "config")
        assert Path(source_git(root, "rev-parse", "--show-toplevel")).resolve() == root.resolve(), "External worktree"
        assert source_git(root, "rev-parse", "HEAD") == revision
        # Submodule .git files may point outside an otherwise clean checkout.
        for directory, names, files in os.walk(root):
            if ".git" in names or ".git" in files:
                subroot = Path(directory)
                subcommon = Path(source_git(subroot, "rev-parse", "--git-common-dir"))
                subcommon = subcommon if subcommon.is_absolute() else subroot / subcommon
                assert subcommon.resolve().is_relative_to(capsule.resolve()), "External submodule Git storage"
                assert not (subcommon / "objects/info/alternates").exists()
                public_git_config(subcommon / "config")
            if ".git" in names:
                names.remove(".git")
        assert not source_git(
            root, "status", "--porcelain", "--ignored", "--untracked-files=all"
        ), "Source must be a pristine dedicated checkout"
        for line in source_git(root, "submodule", "status", "--recursive").splitlines():
            assert not line.startswith(("-", "+", "U")), "Unadmitted submodule"
    manifest = json.loads((capsule / "control/.github/scripts/utils/formatter_pair_manifest.json").read_text())
    assert manifest["native_members"] == MEMBERS
    assert (
        source_git(capsule / "candidate", "diff", "--name-only", BASELINE, CANDIDATE).splitlines()
        == manifest["source_diff"]
    )
    native_trees = {}
    for directory in ("tt_metal", "ttnn"):
        trees = [source_git(capsule / "candidate", "rev-parse", rev + ":" + directory) for rev in (BASELINE, CANDIDATE)]
        assert trees[0] == trees[1]
        native_trees[directory] = trees[0]
    return {
        "heads": heads,
        "native_trees": native_trees,
        "diff": manifest["source_diff"],
        "same_ee_native_for_both": True,
        "ordered_device_state_limitation": True,
        "timing_qualified": False,
    }


def validate_payload(capsule, manifest, digest, checkpoint=lambda: None):
    """Seal adapter/reference code and artifacts BEFORE executing any helper."""
    expected = dict(manifest["files"])
    expected["capsule.json"] = {"bytes": (capsule / "capsule.json").stat().st_size, "sha256": digest}
    inventory = owned.inventory(capsule, payload=True)
    assert set(inventory) == set(expected), "Unsealed capsule membership"
    roots = {"adapter", "control", "baseline", "candidate", "plugin", "vllm-source", "artifacts", "capsule.json"}
    for name, stamp in inventory.items():
        assert Path(name).parts[0] in roots, "Private or unrelated capsule member"
        checkpoint()
        if isinstance(stamp, dict):
            assert expected[name] == {"link": stamp["link"]}
        elif name.startswith(("adapter/", "control/", "artifacts/")) or name == "capsule.json":
            assert (capsule / name).stat().st_size == expected[name]["bytes"]
            assert checks.sha(capsule / name, checkpoint) == expected[name]["sha256"]
    for category, member in MEMBERS.items():
        assert expected["artifacts/" + category + "/" + member["path"]] == {k: member[k] for k in ("bytes", "sha256")}
    for required in (
        "slurm_pair_launcher.py",
        "slurm_pair_checks.py",
        "host_fd_probe.py",
        "owned_seed_copy.py",
        "owned_exec.py",
        "slurm_leg.py",
        "inner_host_quiet.py",
        "producer_profile.py",
        "seed_storage.py",
        "owned_stock.py",
        "runtime_guard/sitecustomize.py",
    ):
        assert "adapter/" + required in expected
    return expected, inventory


def owned_operations_closed(evidence):
    for receipt in evidence.glob("*-ownership.json"):
        row = json.loads(receipt.read_text())
        if not row.get("closed") or not row.get("close", {}).get("reaped") or row["close"]["escalated"]:
            return False
    return True


def checked_json(path, digest, checkpoint=lambda: None):
    assert re.fullmatch(r"[0-9a-f]{64}", digest) and checks.sha(path, checkpoint) == digest
    return json.loads(path.read_text())


def command(command, seconds=30):
    return subprocess.check_output(command, text=True, timeout=seconds).strip()


def mount_identity(path):
    rows = json.loads(command(["findmnt", "--json", "--target", str(path), "--output", "SOURCE,FSTYPE,TARGET,UUID"]))[
        "filesystems"
    ]
    assert len(rows) == 1 and all(
        rows[0].get(k) for k in ("source", "fstype", "target")
    ), "No reliable mount/source identity"
    return {
        "canonical_path": str(path),
        "st_dev": path.stat().st_dev,
        "mount": {k: rows[0].get(k) for k in ("source", "fstype", "target", "uuid")},
    }


def gated_attach_command(identifier):
    return [
        sys.executable,
        "-B",
        str(Path(__file__).with_name("owned_exec.py")),
        "docker",
        "start",
        "--attach",
        identifier,
    ]


def live_assignment():
    from formatter_ci_binding import live_assignment as ci_assignment

    return ci_assignment()


class Controller:
    def __init__(self, workspace, assignment, drivers, nonce):
        self.workspace, self.evidence = workspace, workspace / "evidence"
        self.assignment, self.drivers, self.nonce = assignment, drivers, nonce
        self.deadline = time.monotonic() + assignment["remaining_controller_seconds"]
        # CI checkouts/downloads belong to the same45min preparation envelope.
        self.prep_deadline = min(self.deadline - 175 * 60, time.monotonic() + 45 * 60)
        self.created = []
        self.probe_attempted = False

    def checkpoint(self):
        assert not cancelled and time.monotonic() < self.prep_deadline, "Preparation cancelled or bounded deadline"

    def registry(self):
        owned.save(self.evidence / "created-containers.json", self.created)

    def inspect(self, item):
        reference = item["id"] or item["name"]
        argv = ["docker", "container", "inspect", reference]

        def failure_receipt(returncode, stdout, stderr, missing=False, timed_out=False):
            stdout = stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else stdout or ""
            stderr = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr or ""
            owned.save(
                self.evidence / ("container-inspect-failure-" + uuid.uuid4().hex + ".json"),
                {
                    "argv": argv,
                    "returncode": returncode,
                    "stdout": stdout[:4096],
                    "stderr": stderr[:4096],
                    "stdout_truncated": len(stdout) > 4096,
                    "stderr_truncated": len(stderr) > 4096,
                    "timeout_seconds": 15,
                    "timed_out": timed_out,
                    "exact_object_missing": missing,
                },
            )

        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired as error:
            failure_receipt(None, error.stdout, error.stderr, timed_out=True)
            raise
        if result.returncode:
            prefix, separator, missing_reference = result.stderr.strip().rpartition(": ")
            missing = (
                result.returncode == 1
                and len(result.stdout) <= 4096
                and len(result.stderr) <= 4096
                and result.stdout.strip() in {"", "[]"}
                and bool(separator)
                and missing_reference == reference
                and prefix.casefold()
                in {
                    "error: no such object",
                    "error: no such container",
                    "error response from daemon: no such container",
                }
            )
            failure_receipt(result.returncode, result.stdout, result.stderr, missing=missing)
            assert missing, "Cannot establish exact container identity"
            return None
        rows = json.loads(result.stdout)
        assert len(rows) == 1
        row = rows[0]
        labels = row["Config"]["Labels"]
        assert (item["id"] is None or row["Id"] == item["id"]) and row["Name"].lstrip("/") == item["name"]
        assert labels.get("codex.formatter-pair.job") == str(self.assignment["job_id"])
        assert (
            labels.get("codex.formatter-pair.nonce") == self.nonce
            and labels.get("codex.formatter-pair.role") == item["role"]
        )
        assert row["Config"]["Image"] == IMAGE
        return row

    def close_container(self, item):
        row = self.inspect(item)
        if row is None:
            assert not item.get("cleanup", {}).get("escalated"), "Previous escalated closure remains failed"
            return item.get("cleanup", {"missing": True, "creation_unconfirmed": item["id"] is None})
        if row["State"]["Running"]:
            subprocess.run(
                ["docker", "stop", "--time", "60", row["Id"]], check=True, timeout=75, stdout=subprocess.DEVNULL
            )
            row = self.inspect(item)
        assert row and not row["State"]["Running"], "Owned container did not close"
        state = row["State"]
        escalated = state["ExitCode"] == 137 or state["OOMKilled"]
        subprocess.run(["docker", "rm", row["Id"]], check=True, timeout=30, stdout=subprocess.DEVNULL)
        assert self.inspect(item) is None
        result = {"id": row["Id"], "exited": True, "exit_code": state["ExitCode"], "escalated": escalated}
        item["cleanup"] = result
        self.registry()
        assert not escalated, "Escalated container cleanup; no candidate"
        return result

    def docker(self, role, arguments, mounts, seconds, devices=False, ignore_cancel=False):
        name = f"formatter58542-{self.nonce}-{role}-for-reservation-{self.assignment['job_id']}"
        item = {"name": name, "id": None, "role": role}
        self.created.append(item)
        self.registry()
        argv = [
            "docker",
            "create",
            "--name",
            name,
            "--entrypoint",
            "python3",
            "--init",
            "--user",
            "0",
            "--label",
            "codex.formatter-pair.job=" + str(self.assignment["job_id"]),
            "--label",
            "codex.formatter-pair.nonce=" + self.nonce,
            "--label",
            "codex.formatter-pair.role=" + role,
            "--pid=host",
            "--cap-add=SYS_PTRACE",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
        ]
        if devices:
            for row in self.drivers:
                argv += ["--device", row["path"] + ":" + row["path"]]
            argv += ["-v", "/dev/hugepages-1G:/dev/hugepages-1G"]
        for host, target, mode in mounts:
            assert host.exists() and ":" not in str(host), "Missing or invalid exact bind source"
            argv += ["-v", str(host) + ":" + target + ":" + mode]
        if role in ("producer", "baseline", "candidate"):
            argv += [
                "-w",
                "/work",
                "--memory",
                str(profiles.budget()["memory_limit_bytes"]),
                "--memory-swap",
                str(profiles.budget()["memory_limit_bytes"]),
            ]
        argv += [IMAGE, "-B", *arguments]
        child, saved, released = None, None, False
        try:
            identifier = command(argv, 60)
            assert re.fullmatch(r"[0-9a-f]{64}", identifier)
            item["id"] = identifier
            self.registry()
            admitted_container = self.inspect(item)
            assert admitted_container
            if role in ("producer", "baseline", "candidate"):
                assert admitted_container["HostConfig"]["Memory"] == profiles.budget()["memory_limit_bytes"]
                assert admitted_container["HostConfig"]["MemorySwap"] == profiles.budget()["memory_limit_bytes"]
            with (self.evidence / ("docker-" + role + ".log")).open("x") as log:
                child = subprocess.Popen(
                    gated_attach_command(identifier),
                    stdin=subprocess.PIPE,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                saved = owned.identity(child.pid)
                assert saved
                item["attach_client"] = saved
                self.registry()
                if cancelled and not ignore_cancel:
                    raise owned.Cancelled("Cancelled before attach release")
                child.stdin.write("GO\n")
                child.stdin.flush()
                child.stdin.close()
                released = True
                item["attach_released"] = True
                self.registry()
                end = time.monotonic() + seconds
                last_budget = 0
                inner_handled = set()
                while child.poll() is None:
                    if role in ("producer", "baseline", "candidate"):
                        scope = next(
                            host for host, target, mode in mounts if target == "/pair/evidence" and mode == "rw"
                        )
                        inner_host_quiet.serve(self, scope, role, item, inner_handled, min(end, self.deadline))
                    if role in ("producer", "baseline", "candidate") and time.monotonic() - last_budget >= 2:
                        self.live_budget(role)
                        last_budget = time.monotonic()
                    if (cancelled and not ignore_cancel) or time.monotonic() >= end:
                        raise owned.Cancelled("Bounded owned container stopped; preserve and no candidate")
                    time.sleep(0.2)
                row = self.inspect(item)
                state = row["State"]
                assert state["Status"] == "exited" and not state["OOMKilled"]
                assert child.returncode == state["ExitCode"]
                if role in ("producer", "baseline", "candidate") and state["ExitCode"] in (0, 1):
                    assert inner_handled == set(inner_host_quiet.POINTS), "Fresh inner host checkpoints incomplete"
                return state["ExitCode"]
        finally:
            try:
                self.close_container(item)
            finally:
                if child is not None:
                    if child.stdin and not child.stdin.closed:
                        child.stdin.close()
                    if saved:
                        closure = owned.close_owned(child, saved, 10)
                    else:
                        # Before GO the direct child can only wait on stdin.
                        # EOF exits the reviewed gate; waitpid reaps this exact
                        # child without signalling an unverified PID. A timeout
                        # raises, so closure can never be admitted implicitly.
                        assert not released, "No attach work without birth capture"
                        code = child.wait(timeout=10)
                        closure = {
                            "reaped": True,
                            "exit_code": code,
                            "escalated": False,
                            "identity_capture_failed": True,
                            "gate_released": False,
                        }
                    item["attach_client_closure"] = closure
                    self.registry()
                    assert not closure["escalated"], "Docker client did not close normally"

    def live_budget(self, role):
        limits = profiles.budget()
        assert profiles.available_memory() >= limits["live_memory_floor_bytes"], "Live host memory reserve"
        stat = os.statvfs(self.workspace)
        assert stat.f_bavail * stat.f_frsize >= limits["disk_reserve_bytes"], "Live task disk reserve"
        if hasattr(self, "docker_root"):
            docker_stat = os.statvfs(self.docker_root)
            assert docker_stat.f_bavail * docker_stat.f_frsize >= 8 * profiles.GIB, "Live Docker disk reserve"
        if role == "producer":
            assert (
                profiles.regular_bytes(self.workspace / "task-cache/producer/meta-llama--Llama-3.1-8B-Instruct/T3K")
                <= limits["cache_cap_bytes"]
            )

    def probe(self, capsule, role, ignore_cancel=False):
        self.probe_attempted = True
        code, receipt = host_fd_probe.observe(
            Path(__file__).resolve().parent,
            self.evidence,
            role,
            cancel=lambda: cancelled and not ignore_cancel,
        )
        assert code == 0 and receipt["quiet"], "Incomplete visibility or existing driver holder"
        return receipt


def completed_scope(payload, scope, phase, code):
    stock_name = "baseline" if phase == "producer" else phase
    output = scope / stock_name
    assert code in (0, 1), "Incomplete diagnostic leg; see leg-incomplete.json source frames"
    assert (output / "leg-status.json").is_file(), "Completed leg evidence missing; candidate prohibited"
    status = json.loads((output / "leg-status.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "sealed_original_control", payload / "control/.github/scripts/utils/formatter_serial_control.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    manifest = json.loads((payload / "control/.github/scripts/utils/formatter_pair_manifest.json").read_text())
    assert status["control_reference_sha256"] == checks.sha(
        payload / "control/.github/scripts/utils/formatter_serial_control.py"
    )
    assert json.loads((output / "phase-0.json").read_text())["exit_code"] == 0
    assert json.loads((output / "phase-1.json").read_text())["exit_code"] == code
    original.completed_assertions(code, output, manifest["assertion_payload_proof"])
    for operation in ("install", "topology", "phase-0", "phase-1", "server"):
        ownership = json.loads((output / (operation + "-ownership.json")).read_text())
        closed = json.loads((output / (operation + "-cleanup.json")).read_text())
        assert ownership["released"] is True and ownership["owned"]["pid"] > 0
        assert closed["direct_child_reaped"] is True and closed["escalated"] is False
        assert closed["owned_birth_captured_before_start"] == ownership["owned"]
    assert not json.loads((output / "server-cleanup.json").read_text())["escalated"]
    assert json.loads((output / "post-cleanup.json").read_text())["quiet"]
    assert not (output / "cache-generation-prohibited").exists() and not (output / "native-hang-sentinel").exists()
    assert not original.HANG.search((output / "server.log").read_text(errors="replace"))
    assert (output / ("cache-cap.jsonl" if phase == "producer" else "cache-guard.jsonl")).stat().st_size > 0
    captured = json.loads((output / "model-profile.json").read_text())
    assert (
        captured["phase"] == phase and captured["source"] == status["source"] and captured["return_passthrough"] is True
    )
    assert captured["profile"] == status["profile"]
    return status


def admit_leg_status(payload, evidence, name, code):
    status = completed_scope(payload, evidence, name, code)
    manifest = json.loads((payload / "control/.github/scripts/utils/formatter_pair_manifest.json").read_text())
    assert status["exit_code"] == code
    seed_storage.completed(status, name, MEMBERS, manifest["assertion_payload_proof"]["payloads_sha256"])
    return status


def unchanged_tracked_source(payload, name):
    """Runtime may add owned build metadata, but never change tracked Python."""
    for directory in (name, "plugin", "vllm-source"):
        assert (
            git(payload / directory, "rev-parse", "HEAD")
            == {"baseline": BASELINE, "candidate": CANDIDATE, "plugin": PLUGIN, "vllm-source": VLLM}[directory]
        )
        assert not git(
            payload / directory,
            "diff",
            "HEAD",
            "--name-only",
            "--",
            "models",
            "src",
            "benchmarks",
            "tests",
            "docs",
            "examples",
        ), "Tracked scientific source changed"


def launch(args):
    capsule = args.capsule.resolve(strict=True)
    assignment = live_assignment()
    drivers = checks.driver_nodes()
    parent = args.work_parent.resolve(strict=True)
    assert parent.is_dir() and parent.stat().st_uid == os.getuid()
    assert re.fullmatch(
        r"[A-Za-z0-9_/.-]+", str(parent)
    ), "Operation paths must also be safe in the original timeout sentinel"
    nonce = uuid.uuid4().hex[:12]
    workspace = parent / (str(assignment["job_id"]) + "-" + nonce)
    workspace.mkdir(exist_ok=False)
    evidence = workspace / "evidence"
    evidence.mkdir()
    controller_identity = owned.identity(os.getpid())
    assert controller_identity, "Controller birth must be captured before any read/copy"
    owned.save(
        evidence / "controller-identity.json",
        {
            "controller": controller_identity,
            "operation_directory": str(workspace),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "manual_or_interarm_reset": False,
            "normal_admin_start_hook_before_producer_allowed": True,
            "no_governor_change": True,
            "task_invokes_stock_hooks": False,
        },
    )
    assignment.update({"drivers": drivers, "nonce": nonce})
    owned.save(evidence / "slurm-assignment.json", assignment)
    owned.save(evidence / "drivers.json", drivers)
    controller = Controller(workspace, assignment, drivers, nonce)
    payload, result, failure = workspace / "payload", None, None
    try:
        controller.checkpoint()
        capsule_manifest = checked_json(capsule / "capsule.json", args.capsule_sha256, controller.checkpoint)
        storage = checked_json(args.storage_admission, args.storage_sha256, controller.checkpoint)
        assert capsule_manifest["schema"] == 1
        manifest = json.loads((capsule / "control/.github/scripts/utils/formatter_pair_manifest.json").read_text())
        seed_storage.input_gate(storage, manifest["protocols"]["chunked"], args.snapshot)
        expected_payload, payload_inventory = validate_payload(
            capsule, capsule_manifest, args.capsule_sha256, controller.checkpoint
        )
        owned.save(
            evidence / "source-boundary.json",
            validate_source(capsule, min(controller.prep_deadline, controller.deadline - 175 * 60)),
        )
        assert capsule_manifest["reference_control_sha256"] == checks.sha(
            capsule / "control/.github/scripts/utils/formatter_serial_control.py"
        )
        snapshot = checks.seal_snapshot(args.snapshot, storage["snapshot_files"], controller.checkpoint)
        assert snapshot["logical_bytes"] <= profiles.budget()["checkpoint_logical_byte_bound"]
        owned.save(evidence / "snapshot-seal.json", snapshot)
        owned.save(
            evidence / "storage-mount-identities.json",
            {"snapshot": mount_identity(Path(snapshot["canonical"])), "owned_task": mount_identity(workspace)},
        )
        owned.owned_command(
            [sys.executable, "-B", str(capsule / "adapter/owned_exec.py"), "docker", "pull", IMAGE],
            evidence / "image-pull-ownership.json",
            workspace,
            min(600, controller.prep_deadline - time.monotonic()),
            lambda: cancelled,
        )
        image = json.loads(command(["docker", "image", "inspect", IMAGE]))[0]
        assert IMAGE in image["RepoDigests"]
        owned.save(
            evidence / "image.json",
            {"id": image["Id"], "repo_digest": IMAGE, "entrypoint_overridden": "python3", "size": image["Size"]},
        )
        controller.probe(capsule, "initial-quiet")
        assert (
            controller.docker(
                "native-admission",
                [
                    "/pair/adapter/slurm_pair_checks.py",
                    "native",
                    "/pair/artifacts/native/" + MEMBERS["native"]["path"],
                    "/evidence/native-admission.json",
                ],
                [(capsule, "/pair", "ro"), (evidence, "/evidence", "rw")],
                180,
            )
            == 0
        )
        native = json.loads((evidence / "native-admission.json").read_text())
        payload_bytes = sum(row[2] for row in payload_inventory.values() if not isinstance(row, dict))
        baseline_bytes = sum(
            row[2]
            for name, row in payload_inventory.items()
            if name.startswith("baseline/") and not isinstance(row, dict)
        )
        limits = profiles.budget()
        required = (
            payload_bytes
            + baseline_bytes
            + snapshot["logical_bytes"]
            + 3 * limits["cache_cap_bytes"]
            + 3 * native["regular_file_bytes"]
            + limits["disk_reserve_bytes"]
        )
        free = os.statvfs(workspace).f_bavail * os.statvfs(workspace).f_frsize
        docker_root = Path(command(["docker", "info", "--format", "{{.DockerRootDir}}"]))
        controller.docker_root = docker_root
        ds = os.statvfs(docker_root)
        assert (
            free >= required
            and ds.f_bavail * ds.f_frsize >= 32 * profiles.GIB
            and profiles.available_memory() >= limits["memory_limit_bytes"]
        )
        owned.save(
            evidence / "storage-budget.json",
            {
                "free_bytes": free,
                "required_bytes": required,
                "payload_bytes": payload_bytes,
                "producer_source_bytes": baseline_bytes,
                "local_HF_asset_bytes": snapshot["logical_bytes"],
                "three_cache_cap_bytes": 3 * limits["cache_cap_bytes"],
                "three_native_expansion_bytes": 3 * native["regular_file_bytes"],
                "limits": limits,
                "host_memory_available_bytes": profiles.available_memory(),
                "docker_free_bytes": ds.f_bavail * ds.f_frsize,
            },
        )
        controller.checkpoint()
        hf_hub = workspace / "hf-stage"
        hf_source_seal = evidence / "snapshot-seal.json"
        hf_copy_deadline = min(controller.prep_deadline, controller.deadline - 175 * 60)
        assert hf_copy_deadline > time.monotonic(), "No original preparation/three-boot budget remains"
        owned.owned_command(
            [
                sys.executable,
                "-B",
                str(capsule / "adapter/hf_local_stage.py"),
                "--worker",
                "--seal",
                str(hf_source_seal),
                "--hub",
                str(hf_hub),
                "--proof",
                str(evidence / "hf-stage.json"),
                "--seconds",
                str(hf_copy_deadline - time.monotonic()),
            ],
            evidence / "hf-stage-ownership.json",
            hf_hub,
            hf_copy_deadline - time.monotonic(),
            lambda: cancelled,
        )
        staged_hf = json.loads((evidence / "hf-stage.json").read_text())
        owned.save(
            evidence / "hf-stage-independent-verification.json",
            hf_local_stage.verify(snapshot, staged_hf, controller.checkpoint),
        )
        helper = capsule / "adapter/owned_seed_copy.py"

        def copy(source, destinations, expected, role, deadline, payload_mode=False):
            owned.save(evidence / (role + "-expected.json"), expected)
            argv = [
                sys.executable,
                "-B",
                str(helper),
                "--worker",
                "--source",
                str(source),
                "--expected",
                str(evidence / (role + "-expected.json")),
                "--proof",
                str(evidence / (role + "-copy.json")),
                "--seconds",
                str(max(1, deadline - time.monotonic())),
            ]
            if payload_mode:
                argv += ["--payload"]
            for destination in destinations:
                argv += ["--destination", str(destination)]
            owned.owned_command(
                argv,
                evidence / (role + "-ownership.json"),
                destinations[0],
                deadline - time.monotonic(),
                lambda: cancelled,
            )

        copy(capsule, [payload], expected_payload, "payload-copy", controller.prep_deadline, True)
        producer_source = workspace / "producer-source"
        producer_expected = {
            name.removeprefix("baseline/"): row
            for name, row in expected_payload.items()
            if name.startswith("baseline/")
        }
        copy(
            capsule / "baseline",
            [producer_source],
            producer_expected,
            "producer-source-copy",
            controller.prep_deadline,
            True,
        )
        controller.checkpoint()
        validate_source(payload, min(controller.prep_deadline, controller.deadline - 175 * 60))
        assert controller.deadline - time.monotonic() >= 175 * 60, "Insufficient producer/seal/two-arm budget"
        (payload / "evidence").mkdir()
        (evidence / "fixtures").mkdir()
        producer_evidence = evidence / "producer"
        producer_evidence.mkdir()
        (producer_evidence / "fixtures").mkdir()
        # Same frozen fixtures are mounted at the same paths in all three boots.
        for name in ("slurm-assignment.json", "native-admission.json"):
            (producer_evidence / name).write_bytes((evidence / name).read_bytes())

        def hf_checkpoint():
            assert not cancelled and time.monotonic() < controller.deadline, "HF verification cancelled/deadline"

        hf_mounts = hf_local_stage.mounts(snapshot, staged_hf, workspace / "hf-hub", controller.checkpoint)
        seed = workspace / "task-cache/producer/meta-llama--Llama-3.1-8B-Instruct/T3K"
        seed.mkdir(parents=True)

        def mounts(source, scope, cache):
            return [
                (payload, "/pair", "ro"),
                (source, "/work", "rw"),
                (payload / "plugin", "/pair/plugin", "rw"),
                (scope, "/pair/evidence", "rw"),
                (evidence / "fixtures", "/pair/evidence/fixtures", "rw"),
                (cache, "/task-cache", "rw"),
            ] + hf_mounts

        current = live_assignment()
        assert current["job_id"] == assignment["job_id"] and current["driver_indices"] == assignment["driver_indices"]
        assert current["runner_id"] == assignment["runner_id"] and current["driver_namespace"] == drivers
        hf_local_stage.verify(snapshot, staged_hf, controller.checkpoint)
        controller.probe(payload, "pre-producer-quiet")
        assert profiles.available_memory() >= limits["memory_limit_bytes"]
        code = controller.docker(
            "producer",
            ["/pair/adapter/slurm_leg.py", "producer"],
            mounts(producer_source, producer_evidence, workspace / "task-cache/producer"),
            min(47 * 60, controller.deadline - time.monotonic()),
            devices=True,
        )
        # The producer score is recorded separately and never enters pair scoring.
        producer_status = completed_scope(payload, producer_evidence, "producer", code)
        assert producer_status["exit_code"] == code
        seed_storage.completed(
            producer_status, "producer", MEMBERS, manifest["assertion_payload_proof"]["payloads_sha256"]
        )
        assert git(producer_source, "rev-parse", "HEAD") == BASELINE and not git(
            producer_source, "diff", "HEAD", "--name-only", "--", "models", "ttnn", "tt_metal"
        )
        unchanged_tracked_source(payload, "baseline")
        controller.probe(payload, "closed-producer-quiet")
        checks.unchanged_snapshot(snapshot)
        # Seal after positive owned closure; independent rehash precedes copy.
        seal_deadline = min(controller.deadline, time.monotonic() + 30 * 60)

        def seal_checkpoint():
            assert not cancelled and time.monotonic() < seal_deadline, "Seed seal/copy bounded deadline"

        seal = seed_storage.seal(seed, producer_status, seal_checkpoint)
        owned.save(evidence / "producer-seed-seal.json", seal)
        owned.save(
            evidence / "producer-seed-independent-verification.json",
            seed_storage.verify(seed, seal, producer_status["profile"], seal_checkpoint),
        )
        destinations = [
            workspace / "task-cache" / name / "meta-llama--Llama-3.1-8B-Instruct/T3K"
            for name in ("baseline", "candidate")
        ]
        copy(seed, destinations, seal["files"], "seed-copy", seal_deadline)
        for name, destination in zip(("baseline", "candidate"), destinations):
            owned.save(
                evidence / (name + "-seed-verification.json"),
                seed_storage.verify(destination, seal, producer_status["profile"], seal_checkpoint),
            )
        assert controller.deadline - time.monotonic() >= 97 * 60, "Insufficient two-arm/ordinary-close budget"
        results = {}
        for name in ("baseline", "candidate"):
            assert not cancelled and controller.deadline - time.monotonic() >= 47 * 60
            current = live_assignment()
            assert (
                current["job_id"] == assignment["job_id"] and current["driver_indices"] == assignment["driver_indices"]
            )
            assert current["runner_id"] == assignment["runner_id"] and current["driver_namespace"] == drivers
            hf_local_stage.verify(snapshot, staged_hf, hf_checkpoint)
            controller.probe(payload, "pre-" + name + "-quiet")

            def arm_seed_checkpoint():
                assert not cancelled and time.monotonic() < controller.deadline, "Arm seed verification bound"

            arm_seed = workspace / "task-cache" / name / "meta-llama--Llama-3.1-8B-Instruct/T3K"
            owned.save(
                evidence / (name + "-seed-before-arm.json"),
                seed_storage.verify(arm_seed, seal, producer_status["profile"], arm_seed_checkpoint),
            )
            unchanged_tracked_source(payload, name)
            assert profiles.available_memory() >= limits["memory_limit_bytes"]
            code = controller.docker(
                name,
                ["/pair/adapter/slurm_leg.py", name],
                mounts(payload / name, evidence, workspace / "task-cache" / name),
                min(47 * 60, controller.deadline - time.monotonic()),
                devices=True,
            )
            status = admit_leg_status(payload, evidence, name, code)
            assert status["profile"] == producer_status["profile"], "Observed arm profile differs from producer"
            assert (evidence / name / "packages.txt").read_bytes() == (
                producer_evidence / "baseline/packages.txt"
            ).read_bytes()
            assert (evidence / name / "binding.txt").read_bytes() == (
                producer_evidence / "baseline/binding.txt"
            ).read_bytes()
            results[name] = status
            unchanged_tracked_source(payload, name)
            controller.probe(payload, "closed-" + name + "-quiet")
            checks.unchanged_snapshot(snapshot)
            hf_local_stage.verify(snapshot, staged_hf, hf_checkpoint)

            def final_seed_checkpoint():
                assert not cancelled and time.monotonic() < controller.deadline, "Final seed verification bound"

            seed_storage.verify(seed, seal, producer_status["profile"], final_seed_checkpoint)
            owned.save(
                evidence / (name + "-seed-after-arm.json"),
                seed_storage.verify(arm_seed, seal, producer_status["profile"], final_seed_checkpoint),
            )
            owned.save(
                evidence / "pair-results.json",
                {"results": results, "producer_scored": False, "diagnostic_only": True, "timing_qualified": False},
            )
        owned.save(
            evidence / "snapshot-final-byte-verification.json",
            checks.seal_snapshot(args.snapshot, storage["snapshot_files"], final_seed_checkpoint),
        )
        hf_local_stage.verify(snapshot, staged_hf, hf_checkpoint)
        result = max(row["exit_code"] for row in results.values())
    except BaseException as error:
        failure = error
        owned.save(
            evidence / "incomplete.json",
            {
                "error_type": type(error).__name__,
                "source_frames": inner_host_quiet.source_frames(error),
                "cancelled": cancelled,
                "diagnostic_complete": False,
                "candidate_prohibited": True,
            },
        )
    finally:
        cleanup = []
        try:
            for item in list(controller.created):
                cleanup.append(controller.close_container(item))
            if controller.probe_attempted:
                controller.probe(payload if payload.is_dir() else capsule, "final-quiet", ignore_cancel=True)
            workers_closed = owned_operations_closed(evidence)
            assert workers_closed, "Owned preparation CLI/copy closure is incomplete"
            final = evidence / "final-quiet.json"
            final_quiet = final.is_file() and json.loads(final.read_text())["quiet"] is True
            assert final_quiet or not controller.created, "No final device-quiet proof"
            owned.save(
                evidence / "controller-cleanup.json",
                {
                    "owned_copy_workers_closed_before_cleanup": workers_closed,
                    "containers": cleanup,
                    "directory_deletion_performed": False,
                    "partial_and_completed_owned_directories_retained": True,
                    "driver_quiet": final_quiet,
                    "closure_complete": True,
                },
            )
        except BaseException as error:
            owned.save(
                evidence / "controller-cleanup.json",
                {"closure_complete": False, "error_type": type(error).__name__, "directory_deletion_performed": False},
            )
            failure = failure or error
    print("Formatter diagnostic evidence: " + str(evidence), flush=True)
    if failure:
        raise failure
    return result


if __name__ == "__main__":
    # Direct Slurm exec, with cancellation installed before any long read/copy.
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    p = argparse.ArgumentParser()
    p.add_argument("--capsule", type=Path, required=True)
    p.add_argument("--capsule-sha256", required=True)
    p.add_argument("--storage-admission", type=Path, required=True)
    p.add_argument("--storage-sha256", required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--work-parent", type=Path, required=True)
    try:
        raise SystemExit(launch(p.parse_args()))
    except Exception as error:
        print("Stopped diagnostic controller: " + type(error).__name__, flush=True)
        raise SystemExit(2)
