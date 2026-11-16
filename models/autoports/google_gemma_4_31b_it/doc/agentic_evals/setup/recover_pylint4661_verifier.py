#!/usr/bin/env python3
"""Run-specific recovery; default validates files only, --execute starts Docker."""
import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import time

SETUP = pathlib.Path(__file__).resolve().parent
ROOT = SETUP.parents[5]
PLAN = SETUP / "pylint4661-verifier-recovery-plan.json"
TASK = "pylint-dev__pylint-4661"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--output", type=pathlib.Path)
    args = ap.parse_args()
    plan = json.loads(PLAN.read_text())
    wheels = SETUP / "verifier-recovery-wheels"
    wheel_manifest = json.loads((wheels / "manifest.json").read_text())
    appdirs = next(row for row in wheel_manifest["wheels"] if row["name"] == "appdirs" and row["version"] == "1.4.4")
    assert sha((wheels / appdirs["file"]).read_bytes()) == appdirs["sha256"]
    reconstruction = plan["submitted_patch_reconstruction"]
    source = reconstruction["files"]["pylint/config/__init__.py"].encode()
    assert len(source) == 4497 == reconstruction["final_file_bytes"]
    assert sha(source) == reconstruction["final_file_sha256"]
    inputs = {
        key: ROOT / value
        for key, value in plan["verifier_inputs_to_preserve"].items()
        if key in ("test_script", "test_config", "task_definition", "dockerfile")
    }
    script = inputs["test_script"].read_text()
    assert "pytest -rA tests/lint/unittest_lint.py || true" in script
    assert "python -m pip install -e ." in script
    assert "cat > parser.py <<EOF\n" in script
    assert 'json.dump(report_map, open("/logs/verifier/report.json", "w")' in script
    assert json.loads(inputs["test_config"].read_text())["instance_id"] == TASK
    hashes = {key: sha(path.read_bytes()) for key, path in inputs.items()}
    env = plan["environment_and_future_docker_commands"]
    summary = {
        "status": "validated_not_executed",
        "task": TASK,
        "source_sha256": sha(source),
        "source_bytes": len(source),
        "input_hashes": hashes,
        "image": env["docker_image"],
        "expected_image_id": env["image_digest_from_existing_run_manifest"],
        "limits": {"cpus": 1, "memory": "4g", "memory_swap": "4g", "pids": 256, "total_job_seconds": 1800},
        "network_on_execution": "No execution network: hash-pinned local appdirs wheel copied via docker cp; original verifier and parser offline.",
        "unchanged_verifier": True,
        "original_trial_modified": False,
    }
    if not args.execute:
        print(json.dumps(summary, indent=2))
        return
    if args.output is None:
        ap.error("--execute requires a new --output directory")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "verifier").mkdir()
    (out / "submitted_config.py").write_bytes(source)
    name = "v5-pylint4661-recovery-" + str(time.time_ns())
    started = time.monotonic()
    summary["started_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary[
        "slot_enforcement"
    ] = "Coordinator checks free worker slot before launch; driver enforces per-container resources only."
    deadline = started + 1800
    calls = []

    def run(argv, *, check=True):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Recovery total job budget expired")
        result = subprocess.run(argv, capture_output=True, text=True, timeout=min(remaining, 1800))
        index = len(calls)
        (out / f"{index:02d}.stdout.txt").write_text(result.stdout)
        (out / f"{index:02d}.stderr.txt").write_text(result.stderr)
        calls.append({"argv": argv, "returncode": result.returncode})
        if check and result.returncode:
            raise RuntimeError(f"Command {index} failed: {result.returncode}")
        return result

    created = False
    try:
        image_id = run(["docker", "image", "inspect", "--format", "{{.Id}}", env["docker_image"]]).stdout.strip()
        assert image_id == env["image_digest_from_existing_run_manifest"], image_id
        run(
            [
                "docker",
                "create",
                "--name",
                name,
                "--cpus",
                "1",
                "--memory",
                "4g",
                "--memory-swap",
                "4g",
                "--pids-limit",
                "256",
                "--user",
                "0:0",
                "--network",
                "none",
                "-e",
                "UV_OFFLINE=1",
                "-e",
                "UV_PYTHON_DOWNLOADS=never",
                "-e",
                "PIP_NO_INDEX=1",
                "--entrypoint",
                "/bin/bash",
                image_id,
                "-lc",
                "sleep 1800",
            ]
        )
        created = True
        run(["docker", "start", name])
        run(["docker", "exec", name, "mkdir", "-p", "/tests", "/logs/verifier"])
        run(["docker", "cp", str(inputs["test_script"].parent) + "/.", name + ":/tests"])
        for key in ("test_script", "test_config"):
            copied = run(["docker", "exec", name, "sha256sum", "/tests/" + inputs[key].name]).stdout.split()[0]
            assert copied == hashes[key], "Copied verifier input mismatch"
        run(["docker", "exec", name, "chmod", "-R", "a-w", "/tests"])
        run(
            [
                "docker",
                "exec",
                name,
                "/root/.cache/uv/environments-v2/parser-466021493977ef71/bin/python",
                "-c",
                'import importlib.metadata as m; expected={"swebench":"4.0.3","datasets":"2.16.1","fastcore":"1.10.5"}; actual={p:m.version(p) for p in expected}; print(actual); assert actual == expected',
            ]
        )
        run(["docker", "cp", str(out / "submitted_config.py"), name + ":/testbed/pylint/config/__init__.py"])
        before = run(["docker", "exec", name, "sha256sum", "/testbed/pylint/config/__init__.py"]).stdout.split()[0]
        assert before == sha(source)
        run(["docker", "exec", name, "mkdir", "-p", "/wheels"])
        run(["docker", "cp", str(wheels / appdirs["file"]), name + ":/wheels/" + appdirs["file"]])
        assert (
            run(["docker", "exec", name, "sha256sum", "/wheels/" + appdirs["file"]]).stdout.split()[0]
            == appdirs["sha256"]
        )
        run(
            [
                "docker",
                "exec",
                name,
                "/bin/bash",
                "-lc",
                "source /opt/miniconda3/bin/activate && conda activate testbed && "
                "python -m pip install --no-index --find-links /wheels --no-deps appdirs==1.4.4 && "
                "python -m pip show appdirs",
            ]
        )
        # Source the unchanged script so its EXIT trap can retain its own mktemp LOG_FILE.
        # The script still runs its original setup, complete test invocation and parser.
        wrapper = """mkdir -p /logs/verifier
trap 'status=$?; if [ -n "${LOG_FILE:-}" ] && [ -f "$LOG_FILE" ]; then cp "$LOG_FILE" /logs/verifier/test-output.txt; fi; if [ -f /parser.py ]; then cp /parser.py /logs/verifier/parser.py; fi; exit "$status"' EXIT
cd /testbed
source /tests/test.sh
"""
        result = run(["docker", "exec", name, "/bin/bash", "-lc", wrapper], check=False)
        (out / "verifier/test-stdout.txt").write_text(result.stdout)
        (out / "verifier/test-stderr.txt").write_text(result.stderr)
        after = run(["docker", "exec", name, "sha256sum", "/testbed/pylint/config/__init__.py"]).stdout.split()[0]
        assert after == sha(source), "Verifier unexpectedly changed submitted source"
        assert all(sha(path.read_bytes()) == hashes[key] for key, path in inputs.items())
        run(["docker", "cp", name + ":/logs/verifier/.", str(out / "verifier")])
        for key in ("test_script", "test_config"):
            copied = run(["docker", "exec", name, "sha256sum", "/tests/" + inputs[key].name]).stdout.split()[0]
            assert copied == hashes[key], "Verifier changed copied input"
        summary.update(
            status="verifier_finished",
            verifier_exit_code=result.returncode,
            submitted_source_hash_before=before,
            submitted_source_hash_after=after,
        )
        for filename in ("report.json", "reward.txt", "test-output.txt", "parser.py"):
            path = out / "verifier" / filename
            summary.setdefault("artifacts", {})[filename] = sha(path.read_bytes()) if path.exists() else None
        if not (out / "verifier/report.json").exists():
            summary["status"] = "verifier_incomplete_no_report"
    except BaseException as exc:
        summary.update(status="recovery_error", error=repr(exc))
        raise
    finally:
        if created:
            # Retain partial artifacts even when the bounded verifier failed or timed out.
            copied = subprocess.run(
                ["docker", "cp", name + ":/logs/verifier/.", str(out / "verifier")],
                capture_output=True,
                text=True,
                timeout=30,
            )
            summary["final_artifact_copy_returncode"] = copied.returncode
            cleanup = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=30)
            summary["cleanup_returncode"] = cleanup.returncode
        summary["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        summary["elapsed_wall_seconds"] = time.monotonic() - started
        summary["cleanup_allowance_seconds"] = 60
        summary["commands"] = calls
        (out / "recovery-provenance.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
