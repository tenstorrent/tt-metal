#!/usr/bin/env python3
"""Explicitly inert, single-shot executor for the conditional caret HF probe."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[7]
assert (ROOT / "bringup/run_contract.json").is_file(), "repository root resolution failed"
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
H9 = D / "setup/swe85-conditional-hf/revision-adaptive-expanded-call91-8192-reviewed"
INTEGRITY = HERE / "driver-integrity.json"
ADMISSION = HERE / "admission.json"
STOP = HERE / "STOP_BEFORE_LAUNCH"
LAUNCH = HERE / "launch.json"
COMPLETION = HERE / "completion.json"
OUTPUT = HERE / "runs"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
PYTHON_TIMEOUT_GRACE_S = 60


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def integrity_doc():
    doc = read(INTEGRITY)
    require(doc.get("status") == "candidate_driver_integrity", "driver integrity status mismatch")
    for rel, digest in doc.get("files", {}).items():
        path = HERE / rel
        require(path.is_file() and sha(path) == digest, f"candidate file pin mismatch: {rel}")
    require(doc.get("driver_sha256") == sha(__file__), "driver self hash mismatch")
    return doc


def candidate_static():
    pins = integrity_doc()
    review = read(HERE / "root-static-review.json")
    require(
        review.get("status") == "root_static_review_passed_unadmitted" and review.get("reviewer") == "root",
        "root static review is missing or not accepted",
    )
    for key, filename in (
        ("proposal_sha256", "proposal.json"),
        ("command_sha256", "command.json"),
        ("fixture_sha256", "pytest7432-post-first-fulltt-token.private.json"),
        ("forecast_sha256", "forecast.json"),
    ):
        require(review.get(key) == sha(HERE / filename), f"root static review pin mismatch: {key}")
    inventory = read(HERE / "source-hashes.json")
    for rel, digest in inventory.get("files", {}).items():
        path = ROOT / rel
        require(path.is_file() and sha(path) == digest, f"candidate source pin mismatch: {rel}")
    case_file = HERE / "pytest7432-post-first-fulltt-token.private.json"
    case = read(case_file)
    manifest = read(HERE / "candidate-cases.json")
    command = read(HERE / "command.json")
    require(
        len(manifest) == 1 and (ROOT / manifest[0].get("case", "")).resolve() == case_file.resolve(),
        "manifest case mismatch",
    )
    require(
        manifest[0].get("name")
        == "anomaly.pytest-dev__pytest-7432.call19.onset.optional-suffix512-plus-fulltt-first-token",
        "candidate name changed",
    )
    prompt = case.get("prompt_token_ids")
    chosen = case.get("continuations", {}).get("chosen")
    require(isinstance(prompt, list) and len(prompt) == 513, "candidate must contain exactly 513 saved prompt IDs")
    require(isinstance(chosen, list) and len(chosen) == 63, "candidate must contain exactly 63 chosen positions")
    argv = command.get("argv")
    require(
        command.get("status") == "frozen_candidate_command_not_admitted_not_run"
        and command.get("timeout_seconds") == 2400
        and command.get("seed") == [44]
        and command.get("requested_free_tokens") == 512
        and isinstance(argv, list),
        "frozen command contract mismatch",
    )
    require(
        argv[0:2] == ["timeout", "2400"]
        and argv[argv.index("--tokens") + 1] == "512"
        and argv[argv.index("--seeds") + 1] == "44",
        "frozen argv mismatch",
    )
    require(
        argv[argv.index("--cases") + 1] == str(HERE / "candidate-cases.json")
        and argv[argv.index("--output") + 1] == str(OUTPUT),
        "frozen command paths mismatch",
    )
    return {"pins": pins, "case": case, "case_path": case_file, "command": command}


def verify_h9(admission):
    """Require root admission to name and hash-bind terminal primary-HF9 evidence."""
    progress_path = ROOT / admission.get("primary_hf9_progress_path", "")
    review_path = ROOT / admission.get("primary_hf9_root_review_path", "")
    require(progress_path.is_file() and review_path.is_file(), "primary HF9 terminal evidence paths missing")
    require(progress_path.is_relative_to(H9), "HF9 progress pin points outside primary HF9")
    require(sha(progress_path) == admission.get("primary_hf9_progress_sha256"), "HF9 progress pin mismatch")
    require(sha(review_path) == admission.get("primary_hf9_root_review_sha256"), "HF9 root review pin mismatch")
    progress, review = read(progress_path), read(review_path)
    require(
        progress.get("status") in ("completed", "all_groups_completed", "completed_root_reviewed")
        and progress.get("active_group") is None,
        "primary HF9 is not terminal",
    )
    groups = progress.get("groups")
    require(
        isinstance(groups, list) and len(groups) == 8 and all(row.get("status") == "completed" for row in groups),
        "primary HF9 must have all eight groups completed",
    )
    require(
        review.get("reviewer") == "root" and review.get("accepted") is True,
        "primary HF9 terminal root review is not accepted",
    )
    require(review.get("progress_sha256") == sha(progress_path), "HF9 root review does not bind terminal progress")
    review_admission = review.get("admission_sha256", review.get("candidate_admission_sha256"))
    require(review_admission == progress.get("candidate_admission_sha256"), "HF9 review/progress admission mismatch")
    for field in ("launch_sha256", "completion_sha256", "output_sha256"):
        value = review.get(field)
        require(isinstance(value, dict) and len(value) >= 8, f"HF9 root review lacks hash-bound {field}")
    return {
        "progress_path": str(progress_path),
        "progress_sha256": sha(progress_path),
        "root_review_path": str(review_path),
        "root_review_sha256": sha(review_path),
        "admission_sha256": progress.get("candidate_admission_sha256"),
    }


def verify_admission():
    require(ADMISSION.is_file(), "real root admission.json is required")
    admission = read(ADMISSION)
    require(
        admission.get("status") in ("admitted", "root_admitted") and admission.get("reviewer") == "root",
        "candidate lacks explicit root admission",
    )
    candidate_static()
    expected = {
        "driver_sha256": sha(__file__),
        "driver_integrity_sha256": sha(INTEGRITY),
        "proposal_sha256": sha(HERE / "proposal.json"),
        "command_sha256": sha(HERE / "command.json"),
        "fixture_sha256": sha(HERE / "pytest7432-post-first-fulltt-token.private.json"),
        "manifest_sha256": sha(HERE / "candidate-cases.json"),
        "forecast_sha256": sha(HERE / "forecast.json"),
        "root_static_review_sha256": sha(HERE / "root-static-review.json"),
    }
    for key, digest in expected.items():
        require(admission.get(key) == digest, f"admission pin mismatch: {key}")
    h9 = verify_h9(admission)
    return admission, h9


def active_processes():
    hf, owners = [], []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            args = ""
        if "precision_repro_hf.py" in args:
            hf.append({"pid": int(proc.name), "cmdline": args[:300]})
        for fd in (proc / "fd").glob("*"):
            try:
                if os.readlink(fd).startswith("/dev/tenstorrent/"):
                    owners.append({"pid": int(proc.name), "cmdline": args[:300]})
                    break
            except OSError:
                pass
    return hf, owners


def mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            return int(fields[1]) * 1024
    raise RuntimeError("MemAvailable unavailable")


def execute_preflight():
    admission, h9 = verify_admission()
    require(not STOP.exists(), "STOP_BEFORE_LAUNCH is present")
    require(not LAUNCH.exists() and not COMPLETION.exists(), "candidate already has launch/completion")
    require(not OUTPUT.exists(), "candidate output path already exists")
    hf, owners = active_processes()
    require(not hf, f"active HF process(es): {hf}")
    require(not owners, f"active TT device owner(s): {owners}")
    memory = mem_available()
    require(memory >= 100 * 1024**3, f"MemAvailable below 100 GiB: {memory}")
    deadline = admission.get("stage_deadline_unix")
    require(isinstance(deadline, (int, float)), "root admission must pin stage_deadline_unix")
    required = 2400 + PYTHON_TIMEOUT_GRACE_S
    remaining = deadline - time.time()
    require(remaining >= required, f"remaining stage deadline {remaining:.1f}s < {required}s")
    return {
        "h9": h9,
        "memory_bytes": memory,
        "deadline_unix": deadline,
        "remaining_seconds": remaining,
        "active_hf": hf,
        "device_owners": owners,
    }


def result_path(case, output_dir=OUTPUT):
    return Path(output_dir) / (case["name"] + ".hf.private.json")


def validate_output(case_path, case, name, output_dir=OUTPUT):
    path = result_path({"name": name}, output_dir)
    require(path.is_file(), f"missing result output: {path}")
    result = read(path)
    require(result.get("status") == "complete", "HF output is not complete")
    require(result.get("case_sha256") == sha(case_path), "HF output case SHA mismatch")
    require(
        result.get("prompt_tokens") == 513 and len(case["prompt_token_ids"]) == 513,
        "HF output prompt token count mismatch",
    )
    require(result.get("revision") == REVISION, "HF output revision mismatch")
    require(
        result.get("device") == "cpu" and result.get("weights_dtype") == "bfloat16", "HF output must report CPU BF16"
    )
    trials = result.get("results")
    require(
        isinstance(trials, list) and len(trials) == 1 and trials[0].get("seed") == 44,
        "HF output must contain exactly seed44",
    )
    trial = trials[0]
    require(
        trial.get("requested_tokens") == 512
        and isinstance(trial.get("generated_tokens"), int)
        and 0 <= trial["generated_tokens"] <= 512,
        "HF output token cap mismatch",
    )
    require(
        isinstance(trial.get("generated_token_ids"), list)
        and len(trial["generated_token_ids"]) == trial["generated_tokens"],
        "HF generated token IDs/count mismatch",
    )
    positions = result.get("teacher_forced_chosen", {}).get("positions", [])
    require(
        len(positions) == 63 and [p.get("position") for p in positions] == list(range(63)),
        "HF output must contain exactly 63 chosen positions",
    )
    require(
        [p.get("forced_token") for p in positions] == case["continuations"]["chosen"],
        "HF chosen-position tokens mismatch",
    )
    return sha(path)


def execute(child=subprocess.run):
    preflight = execute_preflight()
    spec = candidate_static()
    argv = spec["command"]["argv"]
    # The frozen shell timeout remains authoritative; Python timeout only adds cleanup grace.
    started = time.time()
    launch = {
        "status": "launched",
        "candidate_admission_sha256": sha(ADMISSION),
        "driver_integrity_sha256": sha(INTEGRITY),
        "command": argv,
        "preflight": preflight,
        "started_unix": started,
    }
    write_exclusive(LAUNCH, launch)
    completion = {
        "status": "running",
        "candidate_admission_sha256": sha(ADMISSION),
        "launch_sha256": sha(LAUNCH),
        "command": argv,
        "started_unix": started,
    }
    child_error = None
    try:
        OUTPUT.mkdir(parents=False, exist_ok=False)
        proc = child(argv, cwd=ROOT, capture_output=True, text=True, timeout=2400 + PYTHON_TIMEOUT_GRACE_S, check=False)
        completion.update(
            {"child_returncode": proc.returncode, "stdout": proc.stdout[-8000:], "stderr": proc.stderr[-8000:]}
        )
        completion["exit_code"] = proc.returncode
    except subprocess.TimeoutExpired as exc:
        completion.update(
            {
                "child_returncode": 124,
                "exit_code": 124,
                "timed_out": True,
                "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            }
        )
        child_error = "subprocess.TimeoutExpired"
    except Exception as exc:  # Always leave a terminal record if launching the child fails.
        completion.update({"child_returncode": 127, "exit_code": 127, "child_error": f"{type(exc).__name__}: {exc}"})
        child_error = completion["child_error"]
    completion["elapsed_seconds"] = time.time() - started
    completion["finished_unix"] = time.time()
    if completion["exit_code"] != 0:
        completion.update(
            {
                "status": "failed",
                "output_validation_status": "not_run_child_failed",
                "driver_exit_code": completion["exit_code"] or 1,
            }
        )
    else:
        try:
            case_file = spec["case_path"]
            case = spec["case"]
            name = read(HERE / "candidate-cases.json")[0]["name"]
            result_hash = validate_output(case_file, case, name, OUTPUT)
            completion.update(
                {
                    "status": "completed",
                    "output_validation_status": "passed",
                    "result_sha256": result_hash,
                    "driver_exit_code": 0,
                }
            )
        except Exception as exc:
            completion.update(
                {
                    "status": "output_validation_failed",
                    "output_validation_status": "failed",
                    "output_validation_error": f"{type(exc).__name__}: {exc}",
                    "driver_exit_code": 2,
                }
            )
    write_exclusive(COMPLETION, completion)
    return completion["driver_exit_code"]


def mock_tests():
    """Exercise execute() terminal paths using isolated files and a stub child."""
    import tempfile

    global ADMISSION, INTEGRITY, LAUNCH, COMPLETION, OUTPUT, execute_preflight, candidate_static
    original = {name: globals()[name] for name in ("ADMISSION", "INTEGRITY", "LAUNCH", "COMPLETION", "OUTPUT")}
    original_preflight, original_static = execute_preflight, candidate_static
    scenarios = {}

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        try:
            for label in ("child-error", "timeout", "output-mismatch", "success"):
                scenario = td / label
                scenario.mkdir()
                ADMISSION = scenario / "admission.json"
                INTEGRITY = scenario / "driver-integrity.json"
                LAUNCH = scenario / "launch.json"
                COMPLETION = scenario / "completion.json"
                OUTPUT = scenario / "runs"
                ADMISSION.write_text("{}\n")
                INTEGRITY.write_text("{}\n")
                case_path = scenario / "case.json"
                case = {"prompt_token_ids": [1] * 513, "continuations": {"chosen": list(range(100, 163))}}
                case_path.write_text(json.dumps(case))
                (scenario / "candidate-cases.json").write_text(json.dumps([{"name": "case"}]))
                fake_command = {"argv": ["mock-hf-runner"]}
                execute_preflight = lambda: {"mock_preflight": True}
                candidate_static = lambda: {"case": case, "case_path": case_path, "command": fake_command}

                def child_error(*_args, **_kwargs):
                    raise OSError("mock child launch error")

                def child_timeout(*_args, **_kwargs):
                    raise subprocess.TimeoutExpired(["mock-hf-runner"], 1)

                def child_output(payload):
                    def child(*_args, **_kwargs):
                        out = (
                            OUTPUT
                            / "anomaly.pytest-dev__pytest-7432.call19.onset.optional-suffix512-plus-fulltt-first-token.hf.private.json"
                        )
                        value = {
                            "status": "complete",
                            "case_sha256": sha(case_path) if payload else "wrong",
                            "prompt_tokens": 513,
                            "revision": REVISION,
                            "device": "cpu",
                            "weights_dtype": "bfloat16",
                            "results": [
                                {"seed": 44, "requested_tokens": 512, "generated_tokens": 1, "generated_token_ids": [7]}
                            ],
                            "teacher_forced_chosen": {
                                "positions": [
                                    {"position": i, "forced_token": token}
                                    for i, token in enumerate(case["continuations"]["chosen"])
                                ]
                            },
                        }
                        out.write_text(json.dumps(value))
                        return subprocess.CompletedProcess(["mock-hf-runner"], 0, "mock ok", "")

                    return child

                child = {
                    "child-error": child_error,
                    "timeout": child_timeout,
                    "output-mismatch": child_output(False),
                    "success": child_output(True),
                }[label]
                exit_code = execute(child=child)
                completion = read(COMPLETION)
                scenarios[label] = {
                    "exit_code": exit_code,
                    "status": completion.get("status"),
                    "driver_exit_code": completion.get("driver_exit_code"),
                    "output_validation_error": completion.get("output_validation_error"),
                    "completion_written": COMPLETION.is_file(),
                }
            assert (
                scenarios["child-error"]["status"] == "failed" and scenarios["child-error"]["driver_exit_code"] == 127
            ), scenarios
            assert (
                scenarios["timeout"]["status"] == "failed" and scenarios["timeout"]["driver_exit_code"] == 124
            ), scenarios
            assert (
                scenarios["output-mismatch"]["status"] == "output_validation_failed"
                and scenarios["output-mismatch"]["driver_exit_code"] == 2
            ), scenarios
            assert (
                scenarios["success"]["status"] == "completed" and scenarios["success"]["driver_exit_code"] == 0
            ), scenarios
        finally:
            for name, value in original.items():
                globals()[name] = value
            execute_preflight, candidate_static = original_preflight, original_static
    return {"scenarios": scenarios, "runner_invoked": False}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--mock-tests", action="store_true", help="run isolated host mocks; valid only with --validate-only"
    )
    args = parser.parse_args()
    try:
        if args.validate_only:
            candidate_static()
            tests = mock_tests() if args.mock_tests else None
            print(json.dumps({"status": "validated_inert_no_inference", "mock_tests": tests}, sort_keys=True))
            return 0
        require(not args.mock_tests, "--mock-tests is only available with --validate-only")
        return execute()
    except Exception as exc:
        print(f"candidate driver: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
