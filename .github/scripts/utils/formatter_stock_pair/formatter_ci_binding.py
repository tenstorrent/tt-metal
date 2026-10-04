#!/usr/bin/env python3
"""CI ownership and source packaging for the unchanged three stock boots."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import select
import signal
import socket
import subprocess
import time
from types import SimpleNamespace

import owned_seed_copy as owned
import producer_profile as profiles
import slurm_pair_checks as checks
import slurm_pair_launcher as pair

JOB_NAME = "Formatter serial T3K diagnostic"
LABELS = {"arch-wormhole_b0", "config-t3000", "pipeline-functional", "in-service"}
SNAPSHOT = checks.HF_HUB / checks.MODEL_REPO / "snapshots" / checks.SNAPSHOT
HEADS = {"baseline": pair.BASELINE, "candidate": pair.CANDIDATE, "plugin": pair.PLUGIN, "vllm-source": pair.VLLM}
CONTROL = {
    "formatter_serial_control.py": "d645053130e0656b370c09d028d324b1a1579ed61aa41e8e903503ab5daaf11c",
    "formatter_pair_manifest.json": "39161fe8b6c6de0a0244dac34a8f85197a3cc6e8eb2446680a04735351680c1d",
    "formatter_frozen_requests.py": "390f53c0f9e35397bad7ff005eddd794dd2a4e9a9b3a5c2acf10a93a94476413",
}


def api(endpoint):
    # Never preserve response headers, auth environment or signed download URLs.
    return json.loads(subprocess.check_output(["gh", "api", endpoint], text=True, timeout=30))


def assignment_record(run, jobs, env, now):
    assert env["GITHUB_REPOSITORY"] == "tenstorrent/tt-metal"
    assert env["GITHUB_RUN_ATTEMPT"] == "1", "No diagnostic job rerun"
    assert run["id"] == int(env["GITHUB_RUN_ID"]) and run["head_sha"] == env["GITHUB_SHA"]
    assert run["event"] == "workflow_dispatch" and run["status"] == "in_progress"
    assert run["path"] == ".github/workflows/test-dispatch.yaml"
    matches = [
        j
        for j in jobs
        if j["name"] == JOB_NAME and j["status"] == "in_progress" and j["runner_name"] == env["RUNNER_NAME"]
    ]
    assert len(matches) == 1, "No unique live CI assignment"
    job = matches[0]
    assert job["runner_id"] > 0 and LABELS.issubset(job["labels"])
    start = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00")).timestamp()
    elapsed = now - start
    assert 0 <= elapsed < 14400, "Outside original four-hour CI assignment"
    return {
        "kind": "normal-exclusive-t3k-ci-assignment",
        "job_id": job["id"],
        "run_id": run["id"],
        "diagnostic_head": run["head_sha"],
        "started_at": job["started_at"],
        "runner_name": job["runner_name"],
        "runner_id": job["runner_id"],
        "runner_labels": sorted(job["labels"]),
        "node": socket.gethostname(),
        "uid": os.getuid(),
        "driver_indices": [0, 1, 2, 3],
        "expected_asics": 8,
        "scheduler_elapsed_seconds": elapsed,
        "remaining_controller_seconds": 13200 - elapsed,
        "normal_admin_start_hook_before_producer_allowed": True,
        "manual_or_interarm_reset": False,
        "timing_qualified": False,
    }


def observe_assignment():
    prefix = "repos/tenstorrent/tt-metal/actions/runs/" + os.environ["GITHUB_RUN_ID"]
    run = api(prefix)
    result = api(prefix + "/jobs?per_page=100")
    assert result["total_count"] == 1 and len(result["jobs"]) == 1, "Only one assigned diagnostic job"
    return assignment_record(run, result["jobs"], os.environ, time.time())


def live_assignment():
    row = observe_assignment()
    assert row["remaining_controller_seconds"] > 0, "Original 220-minute work deadline"
    row["driver_namespace"] = checks.driver_nodes()
    return row


def assemble(workspace, source, checkpoint):
    """Seal normal CI checkouts; Git administrative bytes may differ from f886."""
    checkpoint()
    capsule = workspace / "capsule"
    capsule.mkdir(exist_ok=False)
    for name in (*HEADS, "artifacts"):
        root = workspace / name
        assert root.is_dir() and not root.is_symlink() and root.stat().st_uid == os.getuid()
        root.rename(capsule / name)  # One owned filesystem; no long uncaptured copy.
    control = capsule / "control/.github/scripts/utils"
    control.mkdir(parents=True)
    for name, digest in CONTROL.items():
        checkpoint()
        original = source.parent / name
        assert checks.sha(original, checkpoint) == digest
        shutil.copyfile(original, control / name)
    seals = json.loads((source / "adapter-seals.json").read_text())
    adapter = capsule / "adapter"
    adapter.mkdir()
    for name, row in seals["files"].items():
        checkpoint()
        original = source / name
        assert original.is_file() and not original.is_symlink()
        assert original.stat().st_size == row["bytes"] and checks.sha(original, checkpoint) == row["sha256"]
        target = adapter / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    proof = pair.validate_source(capsule, time.monotonic() + max(1, checkpoint()))
    assert proof == seals["source_boundary"], "Frozen f886 scientific source boundary differs"
    inventory = owned.inventory(capsule, payload=True)
    rows = {}
    for name, stamp in inventory.items():
        checkpoint()
        file = capsule / name
        if isinstance(stamp, dict):
            rows[name] = {"link": stamp["link"]}
        else:
            rows[name] = {"bytes": stamp[2], "sha256": checks.sha(file, checkpoint)}
            assert owned.stamp(file.stat()) == stamp, "Source changed during CI assembly"
    assert owned.inventory(capsule, payload=True) == inventory
    for category, member in pair.MEMBERS.items():
        assert rows["artifacts/" + category + "/" + member["path"]] == {k: member[k] for k in ("bytes", "sha256")}
    manifest = {
        "schema": 1,
        "source_boundary": proof,
        "files": rows,
        "reference_control_sha256": CONTROL["formatter_serial_control.py"],
    }
    owned.save(capsule / "capsule.json", manifest)
    digest = checks.sha(capsule / "capsule.json", checkpoint)
    pair.validate_payload(capsule, manifest, digest, checkpoint)
    return capsule, digest


def prepare_storage(snapshot, seals, protocol):
    canonical, repo, rows = checks.snapshot_files(snapshot)
    files = {}
    for name, row in rows.items():
        files[name] = {
            "bytes": checks.SNAPSHOT_ASSETS[name]["bytes"],
            "sha256": seals["hf_sha256"][name],
            "stamp": row["stamp"],
            "binding": row["binding"],
        }
    return {
        "schema": 2,
        "snapshot_revision": checks.SNAPSHOT,
        "snapshot_canonical_path": str(canonical),
        "snapshot_files": files,
        "producer": {
            "source": pair.BASELINE,
            "protocol": protocol,
            "limits": profiles.budget(),
            "scope": "declared-stock-baseline-only",
        },
    }


def host(workspace):
    protocol_name = os.environ["PAIR_PROTOCOL"]
    assert protocol_name in ("chunked", "structured"), "Only the two frozen protocol rows"
    assignment = live_assignment()
    evidence = workspace / "evidence"
    evidence.mkdir(exist_ok=True)
    captured = owned.identity(os.getpid())
    assert captured, "Capture controller before any source assembly"
    owned.save(
        evidence / "ci-controller-identity.json",
        {
            "controller": captured,
            "assignment": assignment,
            "operation_directory": str(workspace),
            "scientific_capsule_origin": "f886afda",
        },
    )
    prep_end = time.monotonic() + 2700 - assignment["scheduler_elapsed_seconds"]

    def checkpoint():
        assert not pair.cancelled and time.monotonic() < prep_end, "Original 45-minute preparation deadline"
        return prep_end - time.monotonic()

    source = Path(__file__).resolve().parent
    capsule, digest = assemble(workspace, source, checkpoint)
    seals = json.loads((source / "adapter-seals.json").read_text())
    protocol = json.loads((capsule / "control/.github/scripts/utils/formatter_pair_manifest.json").read_text())[
        "protocols"
    ][protocol_name]
    owned.save(evidence / "ci-hf-snapshot-layout.json", checks.snapshot_layout(SNAPSHOT))
    storage = prepare_storage(SNAPSHOT, seals, protocol)
    admission = workspace / "storage-admission.json"
    owned.save(admission, storage)
    owned.save(
        evidence / "ci-capsule.json",
        {
            "manifest_sha256": digest,
            "scientific_source_boundary": seals["source_boundary"],
            "git_administrative_bytes_reassembled": True,
            "frozen_control": CONTROL,
        },
    )
    args = SimpleNamespace(
        protocol=protocol_name,
        capsule=capsule,
        capsule_sha256=digest,
        storage_admission=admission,
        storage_sha256=checks.sha(admission),
        snapshot=SNAPSHOT,
        work_parent=workspace,
    )
    checkpoint()
    return pair.launch(args)


def cleanup(workspace):
    """Close only this job's captured children/containers; retain every directory."""
    initial_path = workspace / "evidence/ci-controller-identity.json"
    if not initial_path.exists():
        assert not (workspace / "capsule").exists() and not (workspace / "storage-admission.json").exists()
        assert not list(workspace.glob("[0-9]*-*/evidence")), "Missing controller identity for existing work"
        owned.save(
            workspace / "evidence/ci-final-cleanup.json",
            {
                "controlled_preparation_started": False,
                "driver_quiet_observed": False,
                "directory_deletion_performed": False,
            },
        )
        return
    current = observe_assignment()
    initial = json.loads(initial_path.read_text())
    assert all(
        current[k] == initial["assignment"][k]
        for k in ("job_id", "run_id", "diagnostic_head", "runner_id", "runner_name", "uid", "node")
    )
    assert not owned.same_process(initial["controller"]), "Original controller still active"
    scopes = sorted(workspace.glob(str(current["job_id"]) + "-*/evidence"))
    assert len(scopes) <= 1, "Duplicate controller workspace"
    for evidence in scopes:
        assignment = json.loads((evidence / "slurm-assignment.json").read_text())
        assert assignment["kind"] == current["kind"] and assignment["job_id"] == current["job_id"]
        controller = pair.Controller(evidence.parent, assignment, checks.driver_nodes(), assignment["nonce"])
        controller.created = (
            json.loads((evidence / "created-containers.json").read_text())
            if (evidence / "created-containers.json").is_file()
            else []
        )
        closures = [controller.close_container(item) for item in controller.created]
        stop_captured_preparation(evidence)
        for item in controller.created:
            if "attach_client" in item:
                end = time.monotonic() + 10
                while owned.same_process(item["attach_client"]) and time.monotonic() < end:
                    time.sleep(0.1)
                assert not owned.same_process(item["attach_client"]), "Owned attach client remains"
        # Parent cancellation normally records and reaps its own workers. If it
        # was forcibly lost, preserve incomplete closure; never claim waitpid
        # from this different parent or delete an active operation directory.
        workers_closed = pair.owned_operations_closed(evidence)
        if (evidence / "final-quiet.json").is_file():
            role = "ci-cleanup-quiet"
        else:
            role = "final-quiet"
        receipt = controller.probe(workspace / "capsule", role, ignore_cancel=True)
        owned.save(
            evidence / "ci-final-cleanup.json",
            {
                "containers": closures,
                "owned_workers_closed": workers_closed,
                "driver_quiet": receipt["quiet"],
                "directory_deletion_performed": False,
            },
        )
        assert workers_closed, "Owned worker closure incomplete; preserve failed cancellation"


def stop_captured_preparation(evidence):
    """A lost parent cannot reap a child; prove absence and preserve that limit."""
    result = []
    for role in ("image-pull", "hf-stage", "payload-copy", "producer-source-copy", "seed-copy"):
        path = evidence / (role + "-ownership.json")
        if not path.is_file():
            continue
        row = json.loads(path.read_text())
        child = row["worker"]
        if owned.same_process(child):
            operation = Path(row["operation_dir"])
            assert operation.is_relative_to(evidence.parent)
            assert operation != evidence.parent or role == "image-pull"
            stop_child_pidfd(child, os.getuid(), 60)
        absent = not owned.same_process(child)
        result.append(
            {
                "role": role,
                "pid": child["pid"],
                "birth_ticks": child["birth_ticks"],
                "absent": absent,
                "direct_child_reaped_by_this_parent": False,
            }
        )
        owned.save(evidence / "ci-preparation-cancellation.json", result)
        assert absent, "Owned preparation did not close normally; no escalation or directory cleanup"


def pidfd_pid(descriptor):
    values = dict(
        line.split(":", 1)
        for line in Path("/proc/self/fdinfo/" + str(descriptor)).read_text().splitlines()
        if ":" in line
    )
    return int(values["Pid"].strip())


def stop_child_pidfd(child, uid, seconds):
    """Bind before UID/birth validation; TERM only through the kernel handle."""
    import host_fd_probe

    assert 0 < seconds <= 60
    end = time.monotonic() + seconds
    try:
        descriptor = os.pidfd_open(child["pid"], 0)
    except ProcessLookupError:
        assert not owned.same_process(child), "Lost handle for a still-live captured child"
        return
    try:
        if select.select([descriptor], [], [], 0)[0]:
            while owned.same_process(child):
                assert time.monotonic() < end, "Exited handle does not prove captured birth absence"
                time.sleep(0.05)
            return
        assert pidfd_pid(descriptor) == child["pid"], "Wrong pidfd binding"
        assert owned.same_process(child), "Captured birth changed after handle binding"
        assert host_fd_probe.process_uid(child["pid"]) == uid, "Unowned preparation child"
        assert pidfd_pid(descriptor) == child["pid"] and owned.same_process(child), "Handle/birth changed"
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        except ProcessLookupError:
            assert select.select([descriptor], [], [], 0)[0], "Signal failed without captured-child exit"
        assert select.select([descriptor], [], [], max(0, end - time.monotonic()))[
            0
        ], "Owned pidfd close deadline; no escalation"
    finally:
        os.close(descriptor)
    while owned.same_process(child):
        assert time.monotonic() < end, "Captured birth remains after pidfd exit"
        time.sleep(0.05)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, pair.interrupt)
    signal.signal(signal.SIGINT, pair.interrupt)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("host", "cleanup"))
    parser.add_argument("workspace", type=Path)
    options = parser.parse_args()
    root = options.workspace.resolve(strict=True)
    try:
        raise SystemExit(host(root) if options.mode == "host" else cleanup(root))
    except Exception as error:
        print("Stopped CI adapter: " + type(error).__name__, flush=True)
        if (root / "evidence").is_dir():
            import inner_host_quiet

            owned.save(
                root / ("evidence/ci-" + options.mode + "-incomplete.json"),
                {
                    "error_type": type(error).__name__,
                    "source_frames": inner_host_quiet.source_frames(error),
                    "diagnostic_complete": False,
                },
            )
        raise SystemExit(2)
