#!/usr/bin/env python3
"""Inert, root-admitted executor for one Requests5414 cap-extension diagnostic."""
from __future__ import annotations

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
PROPOSAL = HERE / "proposal.json"
COMMANDS = HERE / "commands.json"
MANIFEST = HERE / "manifest.json"
FORECAST = HERE / "forecast.json"
SOURCE_INTEGRITY = HERE / "source-integrity.json"
SEMANTIC_CONTEXT = HERE / "semantic-context.json"
DRIVER_INTEGRITY = HERE / "driver-integrity.json"
ADMISSION = HERE / "admission.json"
STOP = HERE / "STOP_BEFORE_LAUNCH"
LAUNCH = HERE / "launch.json"
COMPLETION = HERE / "completion.json"
OUTPUT = HERE / "runs"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
SEED = 44
CAP = 1024
PRIOR_CAP = 512
TIMEOUT_S = 3600
TIMEOUT_GRACE_S = 60
MIN_MEM_BYTES = 100 * 1024**3
CASE_NAME = "anomaly.psf__requests-5414.call11.onset"
PRIOR_RESULT = (
    D
    / "setup/swe85-conditional-hf/revision-adaptive-expanded-call91-8192-reviewed/results/seed44-suffix2048/anomaly.psf__requests-5414.call11.onset.hf.private.json"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
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


def candidate_static():
    proposal, commands, manifest = read(PROPOSAL), read(COMMANDS), read(MANIFEST)
    forecast, sources, integrity = read(FORECAST), read(SOURCE_INTEGRITY), read(DRIVER_INTEGRITY)
    semantic = read(SEMANTIC_CONTEXT)
    require(proposal.get("status") == "candidate_only_not_admitted_not_run", "candidate is not inert/unadmitted")
    require(commands.get("status") == "candidate_command_not_admitted_not_run", "command is not inert/unadmitted")
    require(forecast.get("status") == "candidate_forecast_not_execution", "forecast status mismatch")
    require(len(commands.get("commands", [])) == 1 and len(manifest) == 1, "expected one command and one case")
    for field, path in (
        ("commands", COMMANDS),
        ("manifest", MANIFEST),
        ("forecast", FORECAST),
        ("source_integrity", SOURCE_INTEGRITY),
        ("semantic_context", SEMANTIC_CONTEXT),
    ):
        require(proposal.get(field) == {"path": path.name, "sha256": sha(path)}, f"proposal {field} pin mismatch")
    cmd = commands["commands"][0]
    job = manifest[0]
    require(
        cmd.get("order") == 1
        and cmd.get("seed") == SEED
        and cmd.get("cap") == CAP
        and cmd.get("timeout_seconds") == TIMEOUT_S,
        "frozen command identity mismatch",
    )
    require(
        cmd.get("manifest") == "manifest.json" and cmd.get("manifest_sha256") == sha(MANIFEST),
        "command manifest pin mismatch",
    )
    argv = cmd.get("argv")
    require(argv == proposal.get("command"), "proposal/commands frozen argv mismatch")
    require(isinstance(argv, list) and argv[:3] == ["timeout", str(TIMEOUT_S), "env"], "frozen argv timeout mismatch")
    require(argv[argv.index("--cases") + 1] == str(MANIFEST.resolve()), "argv case manifest path mismatch")
    require(argv[argv.index("--output") + 1] == str(OUTPUT.resolve()), "argv output path mismatch")
    require(
        argv[argv.index("--tokens") + 1] == str(CAP)
        and argv[argv.index("--seeds") + 1] == str(SEED)
        and argv[argv.index("--prefill-chunk") + 1] == "512",
        "argv cap/seed/chunk mismatch",
    )
    require("--validate-only" not in argv, "saved generation command must not contain validation mode")
    for key in (
        "HF_HUB_OFFLINE=1",
        "HF_DATASETS_OFFLINE=1",
        "OMP_NUM_THREADS=16",
        "MKL_NUM_THREADS=16",
        "OPENBLAS_NUM_THREADS=16",
    ):
        require(key in argv, f"missing frozen runtime environment {key}")
    require(
        job.get("name") == CASE_NAME
        and job.get("case") == str((ROOT / proposal["identity"]["fixture_path"]).resolve()),
        "exact case fixture/name mismatch",
    )
    require(
        job.get("branches") == ["chosen"] and job.get("free_run_trials") == 1 and job.get("free_run_tokens") == CAP,
        "manifest branch/trial/cap mismatch",
    )
    fixture = read(job["case"])
    identity = proposal["identity"]
    require(sha(job["case"]) == identity["fixture_sha256"], "fixture hash mismatch")
    require(
        len(fixture.get("prompt_token_ids", [])) == 2048 == identity["prompt_tokens"], "exact prompt length mismatch"
    )
    prompt_hash = hashlib.sha256(json.dumps(fixture["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest()
    require(prompt_hash == identity["prompt_token_ids_sha256"], "exact prompt ID hash mismatch")
    chosen = fixture.get("continuations", {}).get("chosen", [])
    chosen_hash = hashlib.sha256(json.dumps(chosen, separators=(",", ":")).encode()).hexdigest()
    require(
        len(chosen) == 64 == identity["chosen_tokens"] and chosen_hash == identity["chosen_sha256"],
        "chosen source mismatch",
    )
    require(
        identity.get("seed") == SEED
        and identity.get("cap_prior") == PRIOR_CAP
        and identity.get("cap_candidate") == CAP
        and identity.get("revision") == REVISION,
        "candidate source/seed/cap/revision lineage mismatch",
    )
    require(
        semantic.get("status") == "semantic_context_candidate_only_not_adjudication", "semantic context status mismatch"
    )
    require(integrity.get("status") == "candidate_executor_integrity_unadmitted", "driver integrity status mismatch")
    for rel, digest in integrity.get("files", {}).items():
        require((HERE / rel).is_file() and sha(HERE / rel) == digest, f"candidate artifact pin mismatch: {rel}")
    require(integrity.get("driver_sha256") == sha(__file__), "executor self hash mismatch")
    require(integrity.get("tests_sha256") == sha(HERE / "test_run_candidate.py"), "host test hash mismatch")
    require(integrity.get("source_integrity_sha256") == sha(SOURCE_INTEGRITY), "source integrity hash mismatch")
    for rel, digest in sources.get("sources", {}).items():
        path = ROOT / rel
        require(path.is_file() and sha(path) == digest, f"frozen source changed: {rel}")
    require(sources.get("semantics_sha256") == sha(SEMANTIC_CONTEXT), "semantic context pin mismatch")
    require(
        integrity.get("runner_sha256")
        == sources.get("sources", {}).get("models/autoports/google_gemma_4_31b_it/tools/precision_repro_hf.py"),
        "pinned runner hash mismatch",
    )
    result = read(PRIOR_RESULT)
    prior_trial = result["results"][0]
    require(sha(PRIOR_RESULT) == proposal["lineage"]["source_result_sha256"], "prior result source hash mismatch")
    require(
        result.get("case_sha256") == identity["fixture_sha256"] and result.get("revision") == REVISION,
        "prior result fixture/revision lineage mismatch",
    )
    require(
        prior_trial.get("seed") == SEED
        and prior_trial.get("requested_tokens") == PRIOR_CAP
        and prior_trial.get("generated_tokens") == PRIOR_CAP,
        "prior trial is not exact cap512 seed44 source",
    )
    require(len(prior_trial.get("generated_token_ids", [])) == PRIOR_CAP, "prior generated-token ID count mismatch")
    return {
        "proposal": proposal,
        "commands": commands,
        "manifest": manifest,
        "forecast": forecast,
        "source_integrity": sources,
        "fixture": fixture,
        "prior_result": result,
        "prior_token_ids": prior_trial["generated_token_ids"],
        "integrity": integrity,
    }


def verify_hf9_terminal(admission):
    progress_path = Path(admission.get("primary_hf9_progress_path", ""))
    review_path = Path(admission.get("primary_hf9_root_review_path", ""))
    if not progress_path.is_absolute():
        progress_path = ROOT / progress_path
    if not review_path.is_absolute():
        review_path = ROOT / review_path
    require(progress_path.is_file() and review_path.is_file(), "root-admitted primary HF9 terminal paths are missing")
    require(
        progress_path.is_relative_to(H9) and review_path.is_relative_to(H9),
        "primary HF9 evidence points outside primary package",
    )
    require(sha(progress_path) == admission.get("primary_hf9_progress_sha256"), "primary HF9 progress hash mismatch")
    require(
        sha(review_path) == admission.get("primary_hf9_root_review_sha256"), "primary HF9 root-review hash mismatch"
    )
    progress, review = read(progress_path), read(review_path)
    require(
        progress.get("status") in ("completed", "all_groups_completed", "completed_root_reviewed")
        and progress.get("active_group") is None,
        "primary HF9 is not terminal",
    )
    require(
        review.get("reviewer") == "root" and review.get("accepted") is True,
        "primary HF9 terminal review is not root-accepted",
    )
    require(review.get("progress_sha256") == sha(progress_path), "HF9 root review does not bind progress")
    require(
        review.get("candidate_admission_sha256") == progress.get("candidate_admission_sha256"),
        "HF9 admission lineage mismatch",
    )
    return {
        "progress_path": str(progress_path),
        "progress_sha256": sha(progress_path),
        "root_review_path": str(review_path),
        "root_review_sha256": sha(review_path),
    }


def verify_admission(now=None):
    require(ADMISSION.is_file(), "root admission.json is required")
    admission = read(ADMISSION)
    require(
        admission.get("status") in ("admitted", "root_admitted") and admission.get("reviewer") == "root",
        "root admission status/reviewer mismatch",
    )
    candidate_static()
    expected = {
        "driver_sha256": sha(__file__),
        "driver_integrity_sha256": sha(DRIVER_INTEGRITY),
        "proposal_sha256": sha(PROPOSAL),
        "commands_sha256": sha(COMMANDS),
        "manifest_sha256": sha(MANIFEST),
        "forecast_sha256": sha(FORECAST),
        "source_integrity_sha256": sha(SOURCE_INTEGRITY),
        "semantic_context_sha256": sha(SEMANTIC_CONTEXT),
        "tests_sha256": sha(HERE / "test_run_candidate.py"),
    }
    for key, digest in expected.items():
        require(admission.get(key) == digest, f"root admission pin mismatch: {key}")
    h9 = verify_hf9_terminal(admission)
    deadline = admission.get("stage_deadline_unix")
    ceiling = admission.get("accepted_runtime_ceiling_seconds")
    now = time.time() if now is None else now
    check_deadline(deadline, ceiling, now)
    return admission, h9


def check_deadline(deadline, ceiling, now):
    require(
        isinstance(deadline, (int, float))
        and isinstance(ceiling, (int, float))
        and ceiling >= TIMEOUT_S + TIMEOUT_GRACE_S,
        "root admission must provide stage deadline and ceiling >= frozen command timeout plus grace",
    )
    require(now + ceiling <= deadline, "accepted ceiling exceeds authorized stage deadline")
    require(
        now + TIMEOUT_S + TIMEOUT_GRACE_S <= deadline, "frozen command timeout plus grace exceeds authorized deadline"
    )
    return True


def process_snapshot():
    hf, owners = [], []
    own = os.getpid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == own:
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


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def resource_preflight(hf, owners, memory):
    require(not hf, f"active HF process blocks run: {hf}")
    require(not owners, f"active TT device owner blocks run: {owners}")
    require(memory >= MIN_MEM_BYTES, f"MemAvailable below 100 GiB: {memory}")
    return {
        "active_hf_processes": hf,
        "device_owners": owners,
        "mem_available_bytes": memory,
        "required_mem_available_bytes": MIN_MEM_BYTES,
    }


def validate_output(output_dir, case, fixture_path, prior_ids):
    path = Path(output_dir) / (CASE_NAME + ".hf.private.json")
    require(path.is_file(), f"missing HF result: {path}")
    result = read(path)
    require(result.get("status") == "complete", "HF result is not complete")
    require(Path(result.get("case", "")).resolve() == fixture_path.resolve(), "HF output exact case path mismatch")
    require(result.get("case_sha256") == sha(fixture_path), "HF output fixture hash mismatch")
    require(
        result.get("prompt_source") == "Saved prompt_token_ids, no chat rerender", "HF output prompt source mismatch"
    )
    require(result.get("prompt_tokens") == len(case["prompt_token_ids"]) == 2048, "HF output prompt length mismatch")
    require(
        result.get("revision") == REVISION
        and result.get("device") == "cpu"
        and result.get("weights_dtype") == "bfloat16"
        and result.get("prefill_chunk") == 512,
        "HF output revision/device/dtype/chunk mismatch",
    )
    require(
        result.get("case_metadata", {}).get("sampling") == {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "HF output sampling settings mismatch",
    )
    trials = result.get("results", [])
    require(len(trials) == 1 and trials[0].get("seed") == SEED, "HF result must contain exactly seed44")
    trial = trials[0]
    require(
        trial.get("requested_tokens") == CAP
        and isinstance(trial.get("generated_tokens"), int)
        and PRIOR_CAP <= trial["generated_tokens"] <= CAP,
        "HF output requested/generated cap mismatch",
    )
    token_ids = trial.get("generated_token_ids", [])
    require(len(token_ids) == trial["generated_tokens"], "HF generated token ID count mismatch")
    require(token_ids[:PRIOR_CAP] == prior_ids, "generated token prefix differs from prior exact 512-token output")
    positions = result.get("teacher_forced_chosen", {}).get("positions", [])
    chosen = case["continuations"]["chosen"]
    require(
        len(positions) == len(chosen) == 64 and [p.get("position") for p in positions] == list(range(64)),
        "HF chosen-position count/indices mismatch",
    )
    require([p.get("forced_token") for p in positions] == chosen, "HF chosen-position tokens mismatch")
    return {
        "path": str(path),
        "sha256": sha(path),
        "generated_tokens": trial["generated_tokens"],
        "prior_512_prefix_match": True,
        "extension_tokens": trial["generated_tokens"] - PRIOR_CAP,
        "seed": SEED,
        "requested_tokens": CAP,
    }


def _execute(child=subprocess.run, now=None, resources=None):
    # Retain terminal evidence for any admitted execution attempt, including failed preflight.
    require(not LAUNCH.exists() and not COMPLETION.exists(), "refusing overwrite of launch/completion evidence")
    started = time.time()
    terminal = {
        "status": "preflight_failed",
        "started_unix": started,
        "driver_exit_code": 2,
        "child_exit_code": None,
        "candidate_admission_sha256": None,
    }
    try:
        spec = candidate_static()
        admission, h9 = verify_admission(now=now)
        terminal.update(
            {
                "candidate_admission_sha256": sha(ADMISSION),
                "primary_hf9_terminal_review": h9,
                "forecast_sha256": sha(FORECAST),
                "forecast_range_seconds": spec["forecast"]["sensitivity_range_seconds"],
                "forecast_point_seconds": spec["forecast"]["point_seconds"],
            }
        )
        require(not STOP.exists(), "STOP_BEFORE_LAUNCH is present")
        require(
            not OUTPUT.exists() and not (HERE / "runner.log").exists(),
            "candidate output directory or runner log already exists",
        )
        if resources is None:
            hf, owners = process_snapshot()
            memory = available_memory()
        else:
            hf, owners, memory = resources
        preflight = resource_preflight(hf, owners, memory)
        terminal["preflight"] = preflight
        cmd = spec["commands"]["commands"][0]
        write_exclusive(
            LAUNCH,
            {
                "status": "launched",
                "candidate_admission_sha256": sha(ADMISSION),
                "driver_sha256": sha(__file__),
                "command": cmd["argv"],
                "started_unix": started,
                "preflight": preflight,
                "primary_hf9_terminal_review": h9,
            },
        )
        terminal.update({"status": "running", "launch_sha256": sha(LAUNCH), "command": cmd["argv"]})
        OUTPUT.mkdir(parents=False, exist_ok=False)
        try:
            with open(HERE / "runner.log", "xb") as log:
                proc = child(
                    cmd["argv"],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=TIMEOUT_S + TIMEOUT_GRACE_S,
                    check=False,
                )
            child_code = proc.returncode
            terminal["child_exit_code"] = child_code
        except subprocess.TimeoutExpired as exc:
            terminal.update({"child_exit_code": 124, "timed_out": True, "child_error": "subprocess.TimeoutExpired"})
            child_code = 124
        except Exception as exc:
            terminal.update({"child_exit_code": 127, "child_error": f"{type(exc).__name__}: {exc}"})
            child_code = 127
        terminal["elapsed_seconds"] = time.time() - started
        log = HERE / "runner.log"
        if log.is_file():
            terminal["runner_log_sha256"] = sha(log)
        if child_code != 0:
            terminal.update(
                {
                    "status": "child_failed",
                    "output_validation_status": "not_run_child_failed",
                    "driver_exit_code": child_code or 1,
                }
            )
        else:
            try:
                validated = validate_output(
                    OUTPUT, spec["fixture"], Path(spec["manifest"][0]["case"]), spec["prior_token_ids"]
                )
                terminal.update(
                    {
                        "status": "completed",
                        "output_validation_status": "passed",
                        "result": validated,
                        "driver_exit_code": 0,
                    }
                )
            except Exception as exc:
                terminal.update(
                    {
                        "status": "output_validation_failed",
                        "output_validation_status": "failed",
                        "output_validation_error": f"{type(exc).__name__}: {exc}",
                        "driver_exit_code": 2,
                    }
                )
    except Exception as exc:
        terminal["error"] = f"{type(exc).__name__}: {exc}"
    terminal["finished_unix"] = time.time()
    if not COMPLETION.exists():
        write_exclusive(COMPLETION, terminal)
    return int(terminal.get("driver_exit_code", 2)), terminal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            spec = candidate_static()
            print(
                json.dumps(
                    {
                        "status": "static_validation_passed_no_runner_no_model_no_device",
                        "case": CASE_NAME,
                        "seed": SEED,
                        "cap": CAP,
                        "forecast_range_seconds": spec["forecast"]["sensitivity_range_seconds"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        code, record = _execute()
        print(
            json.dumps(
                {
                    k: record.get(k)
                    for k in ("status", "driver_exit_code", "child_exit_code", "error", "output_validation_error")
                },
                indent=2,
            )
        )
        return code
    except Exception as exc:
        print(f"candidate executor: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
