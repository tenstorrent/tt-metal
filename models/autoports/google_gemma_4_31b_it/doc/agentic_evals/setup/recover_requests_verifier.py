#!/usr/bin/env python3
"""Faithful run-specific Requests recovery. Default is CPU-only validation."""
import argparse
import datetime
import hashlib
import json
import pathlib
import shlex
import subprocess
import time

SETUP = pathlib.Path(__file__).resolve().parent
ROOT = SETUP.parents[5]
MANIFEST = SETUP / "requests-recovery-source/manifest.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["1724", "1766", "2317"], required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--legacy-pytest",
        action="store_true",
        help="1766/2317 only: install real cached pytest4.6.11/pluggy0.13.1 offline; abort if unavailable",
    )
    parser.add_argument(
        "--certifi",
        action="store_true",
        help="2317 only: install authentic certifi2020.6.20 offline via existing certs.py dependency hook",
    )
    args = parser.parse_args()
    if args.certifi and args.task != "2317":
        parser.error("--certifi is only supported for2317")
    certifi_wheel = SETUP / "requests-recovery-source/ca-diagnosis/certifi-2020.6.20-py2.py3-none-any.whl"
    certifi_sha = "8fc0819f1f30ba15bdb34cceffb9ef04d99f420f68eb75d901e9560b8749fc41"
    if args.certifi:
        assert sha(certifi_wheel) == certifi_sha
    if args.legacy_pytest and args.task not in ("1766", "2317"):
        parser.error("--legacy-pytest is only supported for1766/2317")
    wheels = SETUP / "verifier-recovery-wheels"
    wheel_manifest = json.loads((wheels / "manifest.json").read_text())
    if args.legacy_pytest:
        for row in wheel_manifest["wheels"]:
            assert sha(wheels / row["file"]) == row["sha256"]
        assert sha(wheels / "legacy-pytest-requirements.txt") == wheel_manifest["requirements_sha256"]
    spec = json.loads(MANIFEST.read_text())[args.task]
    provenance = spec["image_git_provenance"]
    assert sha(ROOT / provenance["evidence"]) == provenance["evidence_sha256"]
    for file in spec["files"].values():
        if file["base_file"] is not None:
            assert sha(ROOT / file["base_file"]) == file["base_sha256"]
        assert sha(ROOT / file["submitted_file"]) == file["submitted_sha256"]
    assert sha(ROOT / spec["trajectory"]) == spec["trajectory_sha256"]
    tests = ROOT / spec["tests"]
    for filename, expected in spec["test_hashes"].items():
        assert sha(tests / filename) == expected
    summary = {
        "task": args.task,
        "status": "cpu_validated_not_executed",
        "image_id": spec["image_id"],
        "image_git_provenance": provenance,
        "source_files": spec["files"],
        "test_hashes": spec["test_hashes"],
        "original_trial_modified": False,
        "limits": {"cpus": 1, "memory_gib": 4, "job_seconds": 1800, "cleanup_seconds": 60},
        "ca_override": None,
        "certifi_dependency": {"version": "2020.6.20", "wheel_sha256": certifi_sha} if args.certifi else None,
        "network": "Original public-network test endpoints unchanged; parser strictly offline.",
        "slot_admission": "Coordinator checks worker availability; driver enforces per-container bounds only.",
    }
    if not args.execute:
        print(json.dumps(summary, indent=2))
        return
    if args.output is None:
        parser.error("--execute requires a fresh --output directory")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "verifier").mkdir()
    started = time.monotonic()
    deadline = started + 1800
    summary["started_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    calls = []
    name = "v5-requests-recovery-" + args.task + "-" + str(time.time_ns())
    created = False

    def run(argv, check=True):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Recovery job budget expired")
        r = subprocess.run(argv, capture_output=True, text=True, timeout=remaining)
        n = len(calls)
        (out / f"{n:02d}.stdout.txt").write_text(r.stdout)
        (out / f"{n:02d}.stderr.txt").write_text(r.stderr)
        calls.append({"argv": argv, "returncode": r.returncode})
        if check and r.returncode:
            raise RuntimeError(f"Command{n} failed with{r.returncode}")
        return r

    def container_hash(path):
        return run(["docker", "exec", name, "sha256sum", path]).stdout.split()[0]

    try:
        image_id = run(["docker", "image", "inspect", "--format", "{{.Id}}", spec["image"]]).stdout.strip()
        assert image_id == spec["image_id"]
        workers = run(["docker", "ps", "--format", "{{.Names}}"]).stdout.splitlines()
        assert len(workers) < 12, workers
        summary["workers_before_start"] = workers
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
        head = run(["docker", "exec", name, "git", "-C", "/testbed", "rev-parse", "HEAD"]).stdout.strip()
        assert head == provenance["head"], head
        for revision, expected in [
            ("HEAD^", spec["base_commit"]),
            ("HEAD^{tree}", provenance["head_tree"]),
            (spec["base_commit"] + "^{tree}", provenance["base_tree"]),
        ]:
            observed = run(["docker", "exec", name, "git", "-C", "/testbed", "rev-parse", revision]).stdout.strip()
            assert observed == expected, (revision, observed, expected)
        assert not run(["docker", "exec", name, "git", "-C", "/testbed", "status", "--porcelain"]).stdout.strip()
        summary["image_git_provenance_verified"] = True
        for filename, file in spec["files"].items():
            if file["base_sha256"] is not None:
                assert container_hash("/testbed/" + filename) == file["base_sha256"]
            else:
                run(["docker", "exec", name, "test", "!", "-e", "/testbed/" + filename])
            run(["docker", "cp", str(ROOT / file["submitted_file"]), name + ":/testbed/" + filename])
            assert container_hash("/testbed/" + filename) == file["submitted_sha256"]
        run(["docker", "exec", name, "mkdir", "-p", "/tests", "/logs/verifier"])
        run(["docker", "cp", str(tests) + "/.", name + ":/tests"])
        for filename, expected in spec["test_hashes"].items():
            assert container_hash("/tests/" + filename) == expected
        run(["docker", "exec", name, "chmod", "-R", "a-w", "/tests"])
        run(
            [
                "docker",
                "exec",
                name,
                "/root/.cache/uv/environments-v2/parser-466021493977ef71/bin/python",
                "-c",
                'import importlib.metadata as m; e={"swebench":"4.0.3","datasets":"2.16.1","fastcore":"1.10.5"}; a={p:m.version(p) for p in e}; print(a); assert a==e',
            ]
        )
        if args.legacy_pytest:
            run(["docker", "cp", str(wheels), name + ":/wheels"])
            for row in wheel_manifest["wheels"]:
                assert container_hash("/wheels/" + row["file"]) == row["sha256"]
            run(
                [
                    "docker",
                    "exec",
                    name,
                    "/bin/bash",
                    "-lc",
                    "source /opt/miniconda3/bin/activate && conda activate testbed && "
                    "python -m pip install --no-index --find-links /wheels -r /wheels/legacy-pytest-requirements.txt && "
                    "python -m pytest --version && "
                    "python -c "
                    + shlex.quote(
                        "import sys, pytest, pluggy, packaging; from packaging.utils import canonicalize_version; assert packaging.__version__ == '24.2'; assert canonicalize_version('1.0', strip_trailing_zero=False) == '1.0'; assert sys.version_info[:2] == (3,9); assert pytest.__version__ == '4.6.11'; assert pluggy.__version__ == '0.13.1'; pytest.raises(ValueError, 'int(chr(120))'); print('LEGACY_PYTEST_PREFLIGHT_OK')"
                    ),
                ]
            )
        if args.legacy_pytest:
            run(
                [
                    "docker",
                    "exec",
                    name,
                    "/bin/bash",
                    "-lc",
                    "source /opt/miniconda3/bin/activate && conda activate testbed && "
                    "cd /testbed && mkdir -p /tmp/requests-egg-preflight && "
                    "python setup.py egg_info --egg-base /tmp/requests-egg-preflight && "
                    "python -m pip install . && "
                    'python -c \'import importlib.metadata as m; print({p: m.version(p) for p in ["setuptools", "pip", "packaging", "pytest"]})\' && '
                    "echo LOCAL_SOURCE_INSTALL_PREFLIGHT_OK",
                ]
            )
        run(
            [
                "docker",
                "exec",
                name,
                "/bin/bash",
                "-lc",
                "source /opt/miniconda3/bin/activate && conda activate testbed && python --version && python -m pip freeze",
            ]
        )
        if args.certifi:
            run(["docker", "cp", str(certifi_wheel), name + ":/tmp/" + certifi_wheel.name])
            assert container_hash("/tmp/" + certifi_wheel.name) == certifi_sha
            run(
                [
                    "docker",
                    "exec",
                    name,
                    "/bin/bash",
                    "-lc",
                    "source /opt/miniconda3/bin/activate && conda activate testbed && "
                    "python -m pip install --no-index --no-deps /tmp/" + certifi_wheel.name + " && "
                    "cd /testbed && python -c "
                    + shlex.quote(
                        "import certifi, requests; assert certifi.__version__ == '2020.06.20'; assert requests.certs.where() == certifi.where(); assert requests.utils.DEFAULT_CA_BUNDLE_PATH == certifi.where(); print('CERTIFI_DEPENDENCY_OK', certifi.where())"
                    ),
                ]
            )
        wrapper = """mkdir -p /logs/verifier
trap 'status=$?; if [ -n "${LOG_FILE:-}" ] && [ -f "$LOG_FILE" ]; then cp "$LOG_FILE" /logs/verifier/test-output.txt; fi; if [ -f /parser.py ]; then cp /parser.py /logs/verifier/parser.py; fi; exit "$status"' EXIT
cd /testbed
source /tests/test.sh
"""
        result = run(["docker", "exec", name, "/bin/bash", "-lc", wrapper], check=False)
        (out / "verifier/test-stdout.txt").write_text(result.stdout)
        (out / "verifier/test-stderr.txt").write_text(result.stderr)
        for filename, file in spec["files"].items():
            assert container_hash("/testbed/" + filename) == file["submitted_sha256"]
        for filename, expected in spec["test_hashes"].items():
            assert container_hash("/tests/" + filename) == expected
            assert sha(tests / filename) == expected
        run(["docker", "cp", name + ":/logs/verifier/.", str(out / "verifier")])
        summary.update(
            status="verifier_finished" if (out / "verifier/report.json").exists() else "incomplete_no_report",
            verifier_exit_code=result.returncode,
            source_and_test_hashes_preserved=True,
            artifacts={p.name: sha(p) for p in (out / "verifier").iterdir() if p.is_file()},
        )
    except BaseException as exc:
        summary.update(status="recovery_error", error=repr(exc))
        raise
    finally:
        if created:
            copied = subprocess.run(
                ["docker", "cp", name + ":/logs/verifier/.", str(out / "verifier")],
                capture_output=True,
                text=True,
                timeout=30,
            )
            summary["final_copy_returncode"] = copied.returncode
            removed = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=30)
            summary["cleanup_returncode"] = removed.returncode
        summary.update(
            ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            elapsed_wall_seconds=time.monotonic() - started,
            commands=calls,
        )
        (out / "recovery-provenance.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
