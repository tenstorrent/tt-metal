#!/usr/bin/env python3
"""Offline disposable command validation; default is CPU-only and never invokes Docker."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import time

SETUP = pathlib.Path(__file__).resolve().parent
ROOT = SETUP.parents[5]
CANDIDATES = SETUP.parent / "anomalies/swe85-below-band/fetch-controls"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")
    path.chmod(0o600)


def safe_file(directory, name):
    rel = pathlib.PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or ".git" in rel.parts:
        raise ValueError("Unsafe submitted path: " + name)
    path = directory / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Missing or nonlocal regular submitted file: " + name)
    return path


def prepare(task):
    if pathlib.Path(task).name != task:
        raise ValueError("Task must be one directory name")
    candidate = CANDIDATES / task
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "swe85-fetch-next-action-v1" or manifest["task"] != task:
        raise ValueError("Wrong candidate schema/task")
    action = ROOT / manifest["action_script"]
    if action.resolve() != (candidate / "candidate-action.private.sh").resolve() or action.is_symlink():
        raise ValueError("Action must be the reviewed local candidate script")
    if sha(action) != manifest["action_sha256"]:
        raise ValueError("Candidate action changed")
    check = subprocess.run(["bash", "-n", str(action)], capture_output=True, text=True)
    if check.returncode:
        raise ValueError("Candidate bash syntax failed: " + check.stderr)
    pre = candidate / "precall"
    if (ROOT / manifest["precall_dir"]).resolve() != pre.resolve():
        raise ValueError("Unexpected precall location")
    files = {}
    for row in manifest["precall_files"]:
        p = safe_file(pre, row["file"])
        if sha(p) != row["sha256"]:
            raise ValueError("Precall source hash changed: " + row["file"])
        files[row["file"]] = {"path": str(p), "sha256": row["sha256"]}
    for row in manifest.get("audited_prior_tracked_edits", []):
        if row["file"] not in files:
            raise ValueError("Audited prior edit absent from copied state")
    if set(files) != {str(p.relative_to(pre)) for p in pre.rglob("*") if p.is_file()}:
        raise ValueError("Unlisted precall file")
    expected = manifest["validation_expectations"]
    if not expected["required_output_substrings"] or not expected["allowed_returncodes"]:
        raise ValueError("Missing semantic output/exit expectations")
    if expected["target_path_prefix"] != "/testbed/":
        raise ValueError("Only local task-source target allowed")
    base_hashes = {}
    # These are copied from the pinned stopped image, not later target releases.
    for p in (candidate / "base").rglob("*"):
        if p.is_file() and ".git" not in p.relative_to(candidate / "base").parts:
            if p.is_symlink():
                raise ValueError("Base extraction symlink unsupported")
            base_hashes["/" + str(p.relative_to(candidate / "base"))] = sha(p)
    if not base_hashes:
        raise ValueError("Missing stopped-image evidence")
    case = ROOT / manifest["action_case"]
    data = json.loads(case.read_text())
    if not data["continuations"].get("chosen") or not data["continuations"].get("correct"):
        raise ValueError("Missing matched continuation branches")
    return manifest, {
        "task": task,
        "call": manifest["call"],
        "status": "cpu_validated_not_executed",
        "valid_next_action": None,
        "full_solution_claim": False,
        "diagnostic_only": True,
        "original_trial_modified": False,
        "candidate_manifest_sha256": sha(manifest_path),
        "action_sha256": sha(action),
        "action_case_sha256": sha(case),
        "image_id": manifest["extraction"]["image_id"],
        "image_tag": manifest["extraction"]["image_tag"],
        "image_head": manifest["image_head"],
        "copied_prestate_files": files,
        "base_image_file_hashes": base_hashes,
        "expectations": expected,
        "scope": manifest["scope"],
        "state_scope": "Pinned image plus every materialized pre-call tracked edit/helper. Exact submitted bytes are checked. Prior base-interpreter installs/builds are not replayed; action switches to image testbed environment. This is not native whole-filesystem equivalence.",
        "state_limits": manifest["prior_dependency_state"],
        "limits": {"cpus": 1, "memory_gib": 4, "job_seconds": 300, "cleanup_seconds": 60},
        "network": "none",
        "mounts": [],
        "slot_admission": "Coordinator admits worker; shared lock serializes semantic validators.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    os.umask(0o077)
    manifest, summary = prepare(args.task)
    if not args.execute:
        print(json.dumps(summary, indent=2))
        return
    if args.output is None:
        parser.error("--execute requires fresh --output")
    lock = (SETUP / ".semantic-verifier-worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + summary["limits"]["job_seconds"]
    name = "v5-fetch-next-action-" + args.task + "-" + str(time.time_ns())
    created = False
    calls = []
    summary["started_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    action = ROOT / manifest["action_script"]

    def run(argv, check=True):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Diagnostic budget expired")
        index = len(calls)
        record = {"argv": argv}
        calls.append(record)
        stdout = stderr = ""
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=remaining)
            record["returncode"] = result.returncode
            stdout, stderr = result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            record["timeout"] = True
            stdout, stderr = exc.stdout or b"", exc.stderr or b""
            stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
            stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
            raise
        finally:
            (out / f"{index:02d}.stdout.private.txt").write_text(stdout)
            (out / f"{index:02d}.stderr.private.txt").write_text(stderr)
        if check and result.returncode:
            raise RuntimeError("Command failed: " + str(index))
        return result

    def remote_hash(path):
        return run(["docker", "exec", name, "sha256sum", path]).stdout.split()[0]

    try:
        image = run(["docker", "image", "inspect", "--format", "{{.Id}}", summary["image_tag"]]).stdout.strip()
        if image != summary["image_id"]:
            raise ValueError("Pinned image tag changed")
        run(
            [
                "docker",
                "create",
                "--name",
                name,
                "--network",
                "none",
                "--cpus",
                "1",
                "--memory",
                "4g",
                "--memory-swap",
                "4g",
                "--pids-limit",
                "128",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "DAC_OVERRIDE",
                "--security-opt",
                "no-new-privileges",
                "--user",
                "0:0",
                "-e",
                "PIP_NO_INDEX=1",
                "-e",
                "UV_OFFLINE=1",
                "-e",
                "UV_PYTHON_DOWNLOADS=never",
                "--entrypoint",
                "/bin/bash",
                image,
                "-c",
                "sleep 360",
            ]
        )
        created = True
        run(["docker", "start", name])
        head = run(["docker", "exec", name, "git", "-C", "/testbed", "rev-parse", "HEAD"]).stdout.strip()
        if head != summary["image_head"]:
            raise ValueError("Pinned image HEAD changed")
        for path, expected in summary["base_image_file_hashes"].items():
            if remote_hash(path) != expected:
                raise ValueError("Image extraction hash differs: " + path)
        for filename, row in summary["copied_prestate_files"].items():
            destination = "/testbed/" + filename
            run(["docker", "exec", name, "mkdir", "-p", str(pathlib.PurePosixPath(destination).parent)])
            run(["docker", "cp", row["path"], name + ":" + destination])
            if remote_hash(destination) != row["sha256"]:
                raise ValueError("Copied prestate differs: " + filename)
        run(["docker", "cp", str(action), name + ":/tmp/fetch-next-action.sh"])
        if remote_hash("/tmp/fetch-next-action.sh") != summary["action_sha256"]:
            raise ValueError("Copied command changed")
        run(["docker", "inspect", name])  # Records actual isolation limits/network/mounts.
        result = run(["docker", "exec", "-w", "/testbed", name, "/bin/bash", "/tmp/fetch-next-action.sh"], check=False)
        (out / "action.stdout.private.txt").write_text(result.stdout)
        (out / "action.stderr.private.txt").write_text(result.stderr)
        expected = summary["expectations"]
        combined = result.stdout + "\n" + result.stderr
        matches = {text: text in combined for text in expected["required_output_substrings"]}
        target_lines = [line.split() for line in result.stdout.splitlines() if line.startswith("LOCAL_TARGET ")]
        target_ok = (
            len(target_lines) == 1
            and len(target_lines[0]) >= 4
            and target_lines[0][1].startswith("/testbed/")
            and target_lines[0][-1] == "/opt/miniconda3/envs/testbed/bin/python"
        )
        valid = result.returncode in expected["allowed_returncodes"] and all(matches.values()) and target_ok
        summary.update(
            status="command_completed",
            valid_next_action=valid,
            action_returncode=result.returncode,
            output_checks=matches,
            target_provenance_records=target_lines,
            target_provenance_pass=target_ok,
            interpretation="Only expected local diagnostic action validated; does not establish full solution or native transcript equivalence.",
        )
        # Save changed tracked files and all submitted prestate files before cleanup.
        changed = run(["docker", "exec", name, "git", "-C", "/testbed", "diff", "--name-only"]).stdout.splitlines()
        snapshots = set(summary["copied_prestate_files"]) | set(changed)
        saved = {}
        for filename in sorted(snapshots):
            rel = pathlib.PurePosixPath(filename)
            if rel.is_absolute() or ".." in rel.parts or ".git" in rel.parts:
                raise ValueError("Unsafe post-state path")
            dest = out / "poststate" / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            result_copy = run(["docker", "cp", name + ":/testbed/" + filename, str(dest)], check=False)
            saved[filename] = {
                "copy_returncode": result_copy.returncode,
                "sha256": sha(dest) if dest.is_file() else None,
            }
        summary["poststate_files"] = saved
        run(["docker", "exec", name, "git", "-C", "/testbed", "diff", "--no-ext-diff"])
        if (
            sha(action) != summary["action_sha256"]
            or sha(CANDIDATES / args.task / "manifest.json") != summary["candidate_manifest_sha256"]
        ):
            raise ValueError("Host candidate changed during validation")
    except Exception as exc:
        summary.update(status="validation_error", valid_next_action=None, error=repr(exc))
    finally:
        # Command output is already saved on the host. Persist provenance before removal.
        summary.update(commands=calls, elapsed_wall_seconds=time.monotonic() - started)
        write(out / "semantic-validation.json", summary)
        if created:
            cleanup = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=60)
            summary["cleanup_returncode"] = cleanup.returncode
            summary["cleanup_stdout"] = cleanup.stdout
            summary["cleanup_stderr"] = cleanup.stderr
        summary["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        summary["artifacts"] = {
            str(p.relative_to(out)): sha(p)
            for p in out.rglob("*")
            if p.is_file() and p.name != "semantic-validation.json"
        }
        write(out / "semantic-validation.json", summary)
    if summary.get("valid_next_action") is not True or summary.get("cleanup_returncode") != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
