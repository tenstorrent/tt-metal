import shlex

#!/usr/bin/env python3
"""Diagnostic-only before/after semantic validation. Default is CPU-only; never scores agents."""
import argparse
import datetime
import fcntl
import hashlib
import json
import pathlib
import subprocess
import sys
import time

SETUP = pathlib.Path(__file__).resolve().parent
ROOT = SETUP.parents[5]
CANDIDATES = SETUP.parent / "anomalies/swe85-below-band/correct-actions"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--state", choices=["before", "after", "both"], default="both")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--network",
        choices=["none", "bridge"],
        default="none",
        help="Default offline; bridge is an explicit endpoint-dependent diagnostic and must not be described as offline.",
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--wheelhouse",
        type=pathlib.Path,
        help="Hash-verified real PyPI build dependencies for unchanged offline installer",
    )
    parser.add_argument(
        "--collections-callable-compat",
        action="store_true",
        help="Diagnostic Sympy12489 interpreter binding; changes reference environment",
    )
    args = parser.parse_args()
    if args.collections_callable_compat and args.task != "sympy__sympy-12489":
        parser.error("Callable shim is reviewed only for Sympy12489")
    if pathlib.Path(args.task).name != args.task:
        parser.error("Task must be a single directory name")
    candidate = CANDIDATES / args.task
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("action_case") or not manifest.get("precall_files"):
        parser.error("Candidate lacks actionable state; explicitly blocked, not executable")
    if args.state == "both":
        # Sequential subprocesses preserve isolation, including Python/tool state.
        if args.execute:
            if args.output is None:
                parser.error("--execute requires fresh --output")
            args.output.mkdir(parents=True, exist_ok=False)
        summaries = []
        for state in ("before", "after"):
            command = [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--task",
                args.task,
                "--state",
                state,
                "--network",
                args.network,
            ]
            if args.collections_callable_compat:
                command += ["--collections-callable-compat"]
            if args.wheelhouse:
                command += ["--wheelhouse", str(args.wheelhouse)]
            if args.execute:
                command += ["--execute", "--output", str(args.output / state)]
            result = subprocess.run(command, capture_output=True, text=True)
            summaries.append(
                {"state": state, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            )
            if result.returncode:
                break
        if args.execute:
            (args.output / "paired-driver-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        else:
            print(json.dumps(summaries, indent=2))
        if any(row["returncode"] for row in summaries):
            raise SystemExit(1)
        return
    wheel_files = {}
    if args.wheelhouse:
        wheel_manifest = json.loads((args.wheelhouse / "manifest.json").read_text())
        for row in wheel_manifest["files"]:
            path = args.wheelhouse / row["filename"]
            assert path.name == row["filename"] and sha(path) == row["sha256"]
            wheel_files[path.name] = {"path": str(path.resolve()), "sha256": row["sha256"], "url": row["url"]}
    selected = "precall" if args.state == "before" else "candidate"
    spec = {
        "image": manifest["extraction"]["image_tag"],
        "image_id": manifest["extraction"]["image_id"],
        "base_commit": manifest["image_head"],
        "files": {},
        "tests": str(SETUP / "swe85-candidate/tasks85-offline-verifier" / manifest["task"] / "tests"),
    }
    for row in manifest["precall_files"]:
        filename = row["file"]
        if (
            pathlib.Path(filename).is_absolute()
            or ".." in pathlib.Path(filename).parts
            or ".git" in pathlib.Path(filename).parts
        ):
            parser.error("Unsafe source path")
        if not (candidate / "base" / filename).is_file():
            missing = [
                r
                for r in manifest["extraction"]["files"]
                if r.get("path") == filename
                and r.get("returncode")
                and "Could not find the file" in r.get("stderr", "")
            ]
            if not missing:
                parser.error("Missing base source lacks explicit stopped-image absence evidence")
        for folder, key in [("precall", "before_sha256"), ("candidate", "after_sha256")]:
            if row[key] is None:
                if (candidate / folder / filename).exists():
                    parser.error("Expected absent source exists")
                continue
            if sha(candidate / folder / filename) != row[key]:
                parser.error("Candidate source hash changed")
        spec["files"][filename] = {
            "base_file": str(candidate / "base" / filename) if (candidate / "base" / filename).is_file() else None,
            "base_sha256": sha(candidate / "base" / filename) if (candidate / "base" / filename).is_file() else None,
            "submitted_file": str(candidate / selected / filename),
            "submitted_sha256": row["before_sha256" if args.state == "before" else "after_sha256"],
        }
    known = {row["file"] for row in manifest["precall_files"]}
    if any(
        ("file" not in row and "files" not in row) or not set(row.get("files", [row.get("file")])).issubset(known)
        for row in manifest.get("audited_prior_tracked_edits", [])
    ):
        parser.error("Audited prior tracked edit absent from copied state")
    spec["test_hashes"] = {
        filename: sha(pathlib.Path(spec["tests"]) / filename) for filename in ("test.sh", "config.json")
    }
    for file in spec["files"].values():
        if file["base_file"] is not None:
            assert sha(ROOT / file["base_file"]) == file["base_sha256"]
        if file["submitted_sha256"] is not None:
            assert sha(ROOT / file["submitted_file"]) == file["submitted_sha256"]
    tests = ROOT / spec["tests"]
    for filename, expected in spec["test_hashes"].items():
        assert sha(tests / filename) == expected
    summary = {
        "task": args.task,
        "state": args.state,
        "diagnostic_only": True,
        "candidate_manifest_sha256": sha(manifest_path),
        "source_scope": manifest.get("source_scope"),
        "state_limits": "Relevant production files only; not native whole-tree/dependency replay. Known prior tracked edits copied in both states.",
        "action_execution": "Copies exact CPU-materialized after-state; does not execute candidate shell action under verifier Python. Editing action may be valid in agent Python3.11 independently of verifier environment.",
        "status": "cpu_validated_not_executed",
        "image_id": spec["image_id"],
        "source_files": spec["files"],
        "test_hashes": spec["test_hashes"],
        "original_trial_modified": False,
        "recovery_wheels": wheel_files,
        "limits": {"cpus": 1, "memory_gib": 4, "job_seconds": 1800, "cleanup_seconds": 60},
        "ca_override": None,
        "network": {
            "mode": args.network,
            "offline": args.network == "none",
            "attribution": "Original test endpoints unchanged. Endpoint-dependent failures under none are infrastructure-limited, not evidence against candidate semantics. Bridge explicitly permits original endpoint access; parser remains offline.",
        },
        "slot_admission": "Coordinator checks worker availability; driver enforces per-container bounds only.",
    }
    summary["environment_compatibility"] = {
        "collections_callable_prebound": args.collections_callable_compat,
        "unchanged_reference_environment": not args.collections_callable_compat,
    }
    if not args.execute:
        print(json.dumps(summary, indent=2))
        return
    if args.output is None:
        parser.error("--execute requires a fresh --output directory")
    lock = (SETUP / ".semantic-verifier-worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o700)
    (out / "verifier").mkdir()
    started = time.monotonic()
    deadline = started + 1800
    summary["started_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    calls = []
    name = "v5-semantic-validation-" + args.task + "-" + str(time.time_ns())
    created = False

    def run(argv, check=True):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Recovery job budget expired")
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            n = len(calls)
            for stream in ("stdout", "stderr"):
                text = getattr(exc, stream) or b""
                if isinstance(text, bytes):
                    text = text.decode(errors="replace")
                (out / f"{n:02d}.{stream}.partial.txt").write_text(text)
            calls.append({"argv": argv, "timeout": True})
            raise
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
        run(
            [
                "docker",
                "create",
                "--name",
                name,
                "--network",
                args.network,
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
                "PIP_FIND_LINKS=/tmp/semantic-build-wheels",
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
        assert head == spec["base_commit"], head  # Pinned image HEAD, not dataset base SHA.
        for filename, file in spec["files"].items():
            if file["base_sha256"] is not None:
                assert container_hash("/testbed/" + filename) == file["base_sha256"]
            else:
                run(["docker", "exec", name, "test", "!", "-e", "/testbed/" + filename])
            if file["submitted_sha256"] is not None:
                run(["docker", "exec", name, "mkdir", "-p", str(pathlib.PurePosixPath("/testbed/" + filename).parent)])
                run(["docker", "cp", str(ROOT / file["submitted_file"]), name + ":/testbed/" + filename])
                assert container_hash("/testbed/" + filename) == file["submitted_sha256"]
            elif file["base_sha256"] is not None:
                parser.error("Deletion of an existing image file is not supported; explicit review needed")
        if wheel_files:
            run(["docker", "exec", name, "mkdir", "-p", "/tmp/semantic-build-wheels"])
            for filename, row in wheel_files.items():
                run(["docker", "cp", row["path"], name + ":/tmp/semantic-build-wheels/" + filename])
                assert container_hash("/tmp/semantic-build-wheels/" + filename) == row["sha256"]
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
        wrapper = """mkdir -p /logs/verifier
trap 'status=$?; if [ -n "${LOG_FILE:-}" ] && [ -f "$LOG_FILE" ]; then cp "$LOG_FILE" /logs/verifier/test-output.txt; fi; if [ -f /parser.py ]; then cp /parser.py /logs/verifier/parser.py; fi; exit "$status"' EXIT
cd /testbed
source /tests/test.sh
"""
        if args.collections_callable_compat:
            shim = out / "sitecustomize.py"
            shim.write_text("import collections, collections.abc\ncollections.Callable = collections.abc.Callable\n")
            run(["docker", "exec", name, "mkdir", "-p", "/tmp/semantic-compat"])
            run(["docker", "cp", str(shim), name + ":/tmp/semantic-compat/sitecustomize.py"])
            assert container_hash("/tmp/semantic-compat/sitecustomize.py") == sha(shim)
            summary["environment_compatibility"].update(
                shim_sha256=sha(shim),
                shim=shim.read_text(),
                scope="Only prebinds deprecated collections.Callable; no warning filters or assertions disabled",
            )
            wrapper = "export PYTHONPATH=/tmp/semantic-compat${PYTHONPATH:+:$PYTHONPATH}\n" + wrapper
        result = run(["docker", "exec", name, "/bin/bash", "-lc", wrapper], check=False)
        (out / "verifier/test-stdout.txt").write_text(result.stdout)
        (out / "verifier/test-stderr.txt").write_text(result.stderr)
        if args.collections_callable_compat:
            summary["fresh_process_isolated_checks"] = []
            for test_name in ("test_from_sequence", "test_printing_non_cyclic"):
                code = (
                    "from sympy.combinatorics.tests.test_permutations import "
                    + test_name
                    + "; "
                    + test_name
                    + '(); print("ISOLATED_PASS '
                    + test_name
                    + '")'
                )
                isolated = run(
                    [
                        "docker",
                        "exec",
                        name,
                        "/bin/bash",
                        "-lc",
                        "cd /testbed && source /opt/miniconda3/bin/activate && conda activate testbed && python -c "
                        + shlex.quote(code),
                    ],
                    check=False,
                )
                summary["fresh_process_isolated_checks"].append(
                    {
                        "test": test_name,
                        "returncode": isolated.returncode,
                        "stdout": isolated.stdout,
                        "stderr": isolated.stderr,
                        "shim_applied": False,
                    }
                )
        summary["verifier_generated_build_mutations"] = []
        for filename, file in spec["files"].items():
            if file["submitted_sha256"] is not None:
                actual = container_hash("/testbed/" + filename)
                if args.task == "psf__requests-1142" and filename == "build/lib/requests/models.py":
                    production = spec["files"]["requests/models.py"]["submitted_sha256"]
                    assert actual == production, "Generated build copy must equal submitted production source"
                    summary["verifier_generated_build_mutations"].append(
                        {
                            "file": filename,
                            "submitted_sha256": file["submitted_sha256"],
                            "post_test_sha256": actual,
                            "equals_production_file": "requests/models.py",
                            "attribution": "Unchanged test.sh pip install . regenerates build copy; production source checked independently",
                        }
                    )
                else:
                    assert actual == file["submitted_sha256"], filename
            else:
                run(["docker", "exec", name, "test", "!", "-e", "/testbed/" + filename])
        for filename, expected in spec["test_hashes"].items():
            assert container_hash("/tests/" + filename) == expected
            assert sha(tests / filename) == expected
        run(["docker", "cp", name + ":/logs/verifier/.", str(out / "verifier")])
        assert sha(manifest_path) == summary["candidate_manifest_sha256"], "Candidate manifest changed during run"
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
        (out / "semantic-validation-provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
        for artifact in out.rglob("*"):
            if artifact.is_file():
                artifact.chmod(0o600)


if __name__ == "__main__":
    main()
