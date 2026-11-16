#!/usr/bin/env python3
"""Inert-until-admitted serial gate for the isolated down BFP8 LoFi panel."""
import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd().resolve()
assert (ROOT / "bringup/run_contract.json").is_file(), "run from repository root"
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
P1 = D / "setup/swe85-precision-bisect/head-bfp8-hifi2-first-panel-after-w"
P2 = D / "setup/swe85-precision-bisect/head-bf16-hifi4-fixed108-proposal"
SK512 = D / "setup/swe85-conditional-hf/sklearn13135-missingtool-suffix512"
SK512_BFP8 = D / "setup/swe85-conditional-hf/sklearn13135-suffix512-head-bfp8-correction"
SK1024 = D / "setup/swe85-conditional-hf/sklearn13135-missingtool-suffix1024"
H9 = D / "setup/swe85-conditional-hf/revision-adaptive-expanded-call91-8192-reviewed"
P2_ADMISSION_SHA = "74bef162c97e23e882612eb85dae365f32c9da1f13ba4b9b8e1e1f8f582ad1be"
STOP = HERE / "STOP_BEFORE_LAUNCH"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_review(review, expected_range, elapsed, *, contention=False):
    require(review.get("reviewer") == "root" and review.get("accepted") is True, "root review missing or not accepted")
    require(review.get("forecast_range_seconds") == list(expected_range), "review forecast range mismatch")
    require(review.get("reviewed_elapsed_seconds") == elapsed, "review elapsed mismatch")
    in_range = expected_range[0] <= elapsed <= expected_range[1]
    require(
        review.get("forecast_status") == ("in_range" if in_range else "accepted_miss"),
        "review forecast status does not match measured elapsed",
    )
    if contention:
        require(
            review.get("contention_disclosed") is True and review.get("speed_ranking_used") is False,
            "P2 contention must be disclosed and excluded from policy speed ranking",
        )
    return in_range


def source_inventory(path):
    inv = read(path)
    rows = inv.get("files", {})
    require(isinstance(rows, dict) and rows, f"empty/malformed source inventory: {path}")
    for label, item in rows.items():
        if isinstance(item, str):
            source_path, digest = label, item
        else:
            require(
                isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str),
                f"bad inventory row {label}",
            )
            source_path, digest = item["path"], item["sha256"]
        source = ROOT / source_path
        require(source.is_file() and sha(source) == digest, f"source pin mismatch: {source_path}")
    return inv


def candidate_gate(admission_required=False):
    inventory = source_inventory(HERE / "source-hashes.json")
    admission = None
    if admission_required:
        ap = HERE / "admission.json"
        require(ap.is_file(), "explicit root admission required")
        admission = read(ap)
        require(admission.get("status") == "admitted", "candidate admission is not admitted")
        for field, path in (
            ("proposal_sha256", HERE / "proposal.json"),
            ("driver_sha256", Path(__file__)),
            ("source_inventory_sha256", HERE / "source-hashes.json"),
            ("command_sha256", HERE / "command.json"),
            ("forecast_sha256", HERE / "forecast.json"),
            ("precision_sha256", HERE / "down_bfp8_lofi_isolated_precision.json"),
            ("build_kwargs_sha256", HERE / "down_bfp8_lofi_isolated_build_kwargs.json"),
            ("manifest_sha256", HERE / "candidate-cases.json"),
        ):
            require(sha(path) == admission.get(field), f"candidate admission pin mismatch: {field}")
    return admission, inventory


def validate_static():
    inventory = source_inventory(HERE / "source-hashes.json")
    manifest_path = HERE / "candidate-cases.json"
    jobs = read(manifest_path)
    require(len(jobs) == 108, "fixed108 manifest must contain 108 jobs")
    seeds = {42, 43, 44}
    pairs = [(j["source_onset"], j["seed"]) for j in jobs]
    require(
        len(set(pairs)) == 108 and len({j["source_onset"] for j in jobs}) == 36,
        "manifest must cover 36 unique cases x three unique seeds",
    )
    require({j["seed"] for j in jobs} == seeds, "manifest seed set mismatch")
    require(
        all(j.get("branches") == [] and j.get("free_run_trials") == 1 for j in jobs),
        "panel must be free-run-only with one trial per job",
    )
    require(
        all(isinstance(j.get("free_run_tokens"), int) and j["free_run_tokens"] > 0 for j in jobs),
        "manifest has invalid free-run caps",
    )
    # This panel intentionally preserves per-case caps, including longer extension caps.
    baseline = read(D / "setup/head_bfp8_precision_config.json")
    new = read(HERE / "down_bfp8_lofi_isolated_precision.json")
    for key in baseline:
        if key == "config_id":
            continue
        if key in ("weight_groups", "compute_fidelities"):
            for group in baseline[key]:
                if group == "down":
                    expected = "bfp8_lofi" if key == "weight_groups" else "LoFi"
                    require(new[key][group] == expected, f"down-group {key} delta mismatch")
                else:
                    require(new[key][group] == baseline[key][group], f"unintended {group} {key} change")
        else:
            require(new[key] == baseline[key], f"unintended policy field changed: {key}")
    require(
        baseline["weight_groups"]["down"] == "bfp4"
        and baseline["weight_groups"]["lm_head"] == new["weight_groups"]["lm_head"] == "bfp8_lofi"
        and baseline["compute_fidelities"]["lm_head"] == new["compute_fidelities"]["lm_head"] == "LoFi",
        "candidate must isolate down from the served BFP8 LoFi lm_head baseline",
    )
    require(new["max_seq_len"] == 262144, "context contract changed")
    precision_module = runpy.run_path(str(ROOT / "models/autoports/google_gemma_4_31b_it/tt/precision.py"))
    require(precision_module["load_precision"](new) == new, "candidate precision policy did not validate unchanged")
    expected_precision, build_kwargs = candidate_runtime_policy(precision_module)
    require(expected_precision == new, "result precision expectation differs from candidate precision file")
    require(
        build_kwargs
        == {
            "batch_size": 1,
            "precision_config": "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup/swe85-precision-bisect/down-bfp8-lofi-isolated-fixed108-proposal/down_bfp8_lofi_isolated_precision.json",
        },
        "build kwargs must select only the candidate precision file at batch size 1",
    )
    require("ttnn" not in sys.modules, "static policy validation imported TTNN")
    spec = read(HERE / "command.json")
    argv = spec["argv"]
    require(spec["case_count"] == 108 and spec["mode"] == "full", "frozen command scope mismatch")
    require(argv[:3] == ["timeout", "9000", "env"], "frozen timeout changed")
    require(any(x.startswith("PYTHONPATH=") and str(ROOT) in x for x in argv), "explicit PYTHONPATH missing")
    require("READINESS_TRACE_REGION_SIZE=134217728" in argv, "trace region setting missing")
    require(spec["source_panel_sha256"] == sha(manifest_path), "command manifest pin mismatch")
    return {
        "status": "host_static_validation_passed_no_ttnn_import",
        "jobs": len(jobs),
        "unique_cases": len({j["source_onset"] for j in jobs}),
        "seeds": sorted(seeds),
        "caps": {
            str(cap): sum(j["free_run_tokens"] == cap for j in jobs)
            for cap in sorted({j["free_run_tokens"] for j in jobs})
        },
        "source_count": len(inventory["files"]),
        "stop_present": STOP.is_file(),
        "admission_present": (HERE / "admission.json").is_file(),
    }


def candidate_runtime_policy(precision_module=None):
    """Resolve the precise runtime policy selected by this candidate's build kwargs."""
    spec = read(HERE / "command.json")
    argv = spec["argv"]
    require("--build-kwargs-file" in argv, "candidate command lacks build kwargs file")
    kwargs_arg = Path(argv[argv.index("--build-kwargs-file") + 1]).resolve()
    kwargs_path = (HERE / "down_bfp8_lofi_isolated_build_kwargs.json").resolve()
    require(kwargs_arg == kwargs_path, "candidate command does not use its pinned build kwargs")
    kwargs = read(kwargs_path)
    policy_ref = kwargs.get("precision_config")
    require(isinstance(policy_ref, str) and policy_ref, "candidate build kwargs lack precision_config")
    policy_path = (ROOT / policy_ref).resolve()
    require(
        policy_path == (HERE / "down_bfp8_lofi_isolated_precision.json").resolve(),
        "build kwargs do not resolve to the down-only candidate policy",
    )
    policy = read(policy_path)
    if precision_module is None:
        precision_module = runpy.run_path(str(ROOT / "models/autoports/google_gemma_4_31b_it/tt/precision.py"))
    validated = precision_module["load_precision"](policy)
    require(validated == policy, "candidate precision validation changed policy fields")
    return validated, kwargs


def validate_panel_outputs(outdir, jobs, expected_precision):
    outputs, errors = [], []
    for job in jobs:
        path = outdir / (job["name"] + ".json")
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing/empty output: {path}")
            continue
        try:
            row = read(path)
            free = row.get("free_running", [])
            if row.get("case") != job["case"]:
                errors.append(f"fixture mismatch: {job['name']}")
            if row.get("metadata", {}).get("seed") != job["seed"]:
                errors.append(f"metadata seed mismatch: {job['name']}")
            if row.get("runtime_precision") != expected_precision:
                errors.append(f"runtime precision mismatch: {job['name']}")
            if (
                len(free) != 1
                or not isinstance(free[0].get("token_ids"), list)
                or not (0 < len(free[0]["token_ids"]) <= job["free_run_tokens"])
            ):
                errors.append(f"free-run trial/cap mismatch: {job['name']}")
            elif free[0].get("sampling", {}).get("seed") != job["seed"]:
                errors.append(f"sampling seed mismatch: {job['name']}")
            outputs.append(
                {
                    "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                    "sha256": sha(path),
                    "tokens": len(free[0].get("token_ids", [])) if len(free) == 1 else None,
                    "runtime_precision_match": row.get("runtime_precision") == expected_precision,
                }
            )
        except Exception as exc:
            errors.append(f"invalid output {job['name']}: {type(exc).__name__}: {exc}")
    return outputs, errors


def run_panel_with_completion(
    argv,
    outdir,
    jobs,
    expected_precision,
    launch,
    launch_path,
    completion_path,
    log_path,
    *,
    timeout_seconds=9010,
    runner=None,
):
    """Run once and persist a terminal record for success, failure, timeout, or bad outputs."""
    launch_path.write_text(json.dumps(launch, indent=2) + "\n")
    started = time.monotonic()
    process_exit = None
    process_error = None
    timed_out = False
    try:
        if runner is None:
            runner = subprocess.run
        with log_path.open("w") as log:
            proc = runner(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_seconds, check=False)
        process_exit = proc.returncode
    except subprocess.TimeoutExpired as exc:
        process_exit = 124
        timed_out = True
        process_error = f"TimeoutExpired after {timeout_seconds}s"
        if exc.stdout:
            process_error += "; subprocess returned partial stdout"
    except Exception as exc:
        process_exit = 127
        process_error = f"{type(exc).__name__}: {exc}"

    try:
        outputs, validation_errors = validate_panel_outputs(outdir, jobs, expected_precision)
    except Exception as exc:
        outputs = []
        validation_errors = [f"output validation raised {type(exc).__name__}: {exc}"]
    if process_exit != 0:
        status = "failed"
    elif validation_errors:
        status = "invalid_outputs"
    else:
        status = "completed"
    completion = {
        **launch,
        "status": status,
        "end_unix": time.time(),
        "elapsed_seconds": time.monotonic() - started,
        "exit_code": process_exit,
        "timed_out": timed_out,
        "process_error": process_error,
        "validation_errors": validation_errors,
        "outputs": outputs,
    }
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    return completion


def check_p2_terminal():
    admission_path = P2 / "admission.json"
    require(admission_path.is_file() and sha(admission_path) == P2_ADMISSION_SHA, "P2 actual admission bytes mismatch")
    admission = read(admission_path)
    require(admission.get("status") == "admitted", "P2 not admitted")
    for field, path in (
        ("proposal_sha256", P2 / "proposal.json"),
        ("driver_sha256", P2 / "run_after_p1_x.py"),
        ("source_inventory_sha256", P2 / "source-hashes.json"),
    ):
        require(path.is_file() and sha(path) == admission.get(field), f"P2 admission pin mismatch: {field}")
    source_inventory(P2 / "source-hashes.json")
    frozen_command = read(P2 / "command.json").get("argv")
    require(isinstance(frozen_command, list) and frozen_command, "P2 frozen command malformed")
    queue_path, launch_path, completion_path = P2 / "queue.json", P2 / "launch.json", P2 / "completion.json"
    require(
        queue_path.is_file() and launch_path.is_file() and completion_path.is_file(),
        "P2 queue/launch/completion not terminal",
    )
    queue, launch, completion = read(queue_path), read(launch_path), read(completion_path)
    require(
        queue.get("status") == "completed" and queue.get("exit_code") == 0, "P2 queue did not complete successfully"
    )
    require(queue.get("candidate_admission_sha256") == sha(admission_path), "P2 queue admission mismatch")
    require(
        launch.get("status") == "running" and launch.get("candidate_admission_sha256") == sha(admission_path),
        "P2 launch admission/status mismatch",
    )
    require(launch.get("command") == frozen_command, "P2 launch command differs from frozen argv")
    require(
        completion.get("status") == "completed" and completion.get("exit_code") == 0, "P2 completion did not succeed"
    )
    require(
        completion.get("candidate_admission_sha256") == sha(admission_path)
        and completion.get("command") == frozen_command,
        "P2 completion admission/command mismatch",
    )
    elapsed = completion.get("elapsed_seconds")
    bounds_obj = launch.get("forecast", {}).get("forecast_seconds", {})
    bounds = (bounds_obj.get("low"), bounds_obj.get("high"))
    require(
        elapsed is not None and all(isinstance(x, (int, float)) for x in bounds) and bounds[0] <= bounds[1],
        "P2 elapsed or forecast bounds missing",
    )
    review_path = HERE / "predecessors/p2-forecast-review.json"
    require(review_path.is_file(), "root P2 forecast review missing")
    review = read(review_path)
    require(
        review.get("completion_sha256") == sha(completion_path)
        and review.get("admission_sha256") == sha(admission_path),
        "P2 root review evidence hash mismatch",
    )
    in_range = validate_review(review, bounds, elapsed, contention=True)
    if not in_range:
        miss_path = P2 / "forecast-miss-accepted.json"
        require(miss_path.is_file(), "P2 forecast miss has no root acceptance")
        miss = read(miss_path)
        require(
            miss.get("reviewer") == "root"
            and miss.get("accepted") is True
            and miss.get("status") == "root_accepted_forecast_miss"
            and miss.get("completion_sha256") == sha(completion_path)
            and miss.get("admission_sha256") == sha(admission_path)
            and miss.get("measured_seconds") == elapsed
            and miss.get("forecast_range_seconds") == list(bounds)
            and miss.get("contention_disclosed") is True
            and miss.get("speed_ranking_used") is False,
            "P2 forecast-miss acceptance is not bound to actual completion/range/contention",
        )
    return {
        "admission_sha256": sha(admission_path),
        "queue_sha256": sha(queue_path),
        "launch_sha256": sha(launch_path),
        "completion_sha256": sha(completion_path),
        "elapsed_seconds": elapsed,
        "forecast_range_seconds": list(bounds),
        "forecast_review_sha256": sha(review_path),
    }


def _result_rows(root_review):
    rows = root_review.get("rows", [])
    out = {}
    for row in rows:
        path = row.get("result") or row.get("source")
        digest = row.get("sha256") or row.get("source_sha256")
        require(isinstance(path, str) and isinstance(digest, str), "root result review row lacks path/hash")
        out[Path(path).name] = digest
    return out


def _verify_result_set(run_dir, expected_seeds, expected_precision, suffix_tokens):
    expected_names = {
        f"anomaly.scikit-learn__scikit-learn-13135.call29.onset.suffix{suffix_tokens}.seed{s}.json"
        for s in expected_seeds
    }
    files = {p.name: p for p in run_dir.glob("*.seed*.json")}
    require(files.keys() == expected_names, f"SK result set mismatch in {run_dir}: {sorted(files)}")
    hashes = {}
    for name, path in files.items():
        row = read(path)
        precision = row.get("runtime_precision", {})
        require(precision == expected_precision, f"SK runtime precision mismatch: {path}")
        free = row.get("free_running", [])
        require(
            len(free) == 1 and free[0].get("sampling", {}).get("seed") == row.get("metadata", {}).get("seed"),
            f"SK output seed/trial mismatch: {path}",
        )
        hashes[name] = sha(path)
    return hashes


def _verify_review_rows(review_path, expected_hashes, *, triage=False):
    review = read(review_path)
    if triage:
        rows = review.get("rows", []) or (review.get("served_rows", []) + review.get("full_rows", []))
    else:
        rows = review.get("rows", [])
    actual = {}
    for row in rows:
        path = row.get("result") or row.get("result_file") or row.get("source")
        digest = row.get("result_sha256") if triage else (row.get("sha256") or row.get("source_sha256"))
        if not digest:
            digest = row.get("sha256")
        require(
            isinstance(path, str) and isinstance(digest, str), f"review row missing result path/hash: {review_path}"
        )
        actual[Path(path).name] = digest
    require(actual == expected_hashes, f"review rows do not bind exact outputs: {review_path}")
    if not triage:
        require(
            review.get("reviewer", "root") == "root"
            and review.get("status", "completed_root_reviewed") == "completed_root_reviewed",
            f"root output review is not terminal: {review_path}",
        )
    return review


def _verify_mode_completion(
    base, mode, admission_path, expected_precision, review_path, triage_path, *, old_schema=False, suffix_tokens=512
):
    old_full_arm = old_schema and mode == "full"
    completion_path = (
        base / ("completion-full.json" if old_full_arm else "completion.json")
        if old_schema
        else base / f"completion-{mode}.json"
    )
    launch_path = (
        base / ("launch-full.json" if old_full_arm else "launch.json") if old_schema else base / f"launch-{mode}.json"
    )
    completion, launch = read(completion_path), read(launch_path)
    require(
        completion.get("exit_code") == 0 and completion.get("status", "completed") == "completed",
        f"SK {mode} completion is not successful",
    )
    admission_hash = sha(admission_path)
    bound_admission = completion.get("candidate_admission_sha256") or completion.get("admission_sha256")
    require(bound_admission == admission_hash, f"SK {mode} completion admission mismatch")
    require(completion.get("command") == launch.get("command"), f"SK {mode} launch/completion command mismatch")
    require(
        launch.get("candidate_admission_sha256", launch.get("admission_sha256")) == admission_hash,
        f"SK {mode} launch admission mismatch",
    )
    if old_schema and mode == "full":
        require(
            completion.get("manifest_sha256") == sha(base / "full-cases.json"),
            "original full BF16 completion manifest mismatch",
        )
        require(
            "--cases" in completion.get("command", [])
            and completion["command"][completion["command"].index("--cases") + 1] == str(base / "full-cases.json"),
            "original full BF16 command does not use the pinned full manifest",
        )
    elif not old_schema:
        cmds = read(base / "commands.json")
        require(
            completion.get("command") == cmds[mode].get("execution_command"),
            f"SK {mode} command differs from frozen command",
        )
        expected_manifest = sha(base / f"{mode}-cases.json")
        require(completion.get("manifest_sha256") == expected_manifest, f"SK {mode} manifest pin mismatch")
    run_dir = base / ("runs/full" if old_full_arm else "runs") if old_schema else base / f"runs/{mode}"
    hashes = _verify_result_set(run_dir, (42, 43, 44), expected_precision, suffix_tokens)
    _verify_review_rows(review_path, hashes)
    triage = _verify_review_rows(triage_path, hashes, triage=True)
    return {
        "completion_sha256": sha(completion_path),
        "launch_sha256": sha(launch_path),
        "root_review_sha256": sha(review_path),
        "triage_sha256": sha(triage_path),
        "result_sha256": hashes,
        "review": read(review_path),
        "triage": triage,
    }


def _hf_evidence_package(package_path, expected_sk_evidence):
    package = read(package_path)
    require(
        package.get("reviewer") == "root"
        and package.get("accepted") is True
        and package.get("status") == "completed_root_reviewed",
        "HF evidence package lacks root completion review",
    )
    require(
        package.get("sk_evidence_sha256") == expected_sk_evidence,
        "HF evidence package is not bound to the exact SK evidence set",
    )
    admission_ref = package.get("admission", {})
    admission_path = ROOT / admission_ref.get("path", "")
    require(
        admission_path.is_file() and sha(admission_path) == admission_ref.get("sha256"),
        "HF evidence admission path/hash mismatch",
    )
    admission = read(admission_path)
    require(admission.get("status") in ("admitted", "root_admitted"), "HF candidate admission is not admitted")
    admission_sha = sha(admission_path)
    completion_hashes = {}
    for item in package.get("completions", []):
        path = ROOT / item.get("path", "")
        require(path.is_file() and sha(path) == item.get("sha256"), "HF completion path/hash mismatch")
        completion = read(path)
        require(
            completion.get("exit_code") == 0 and completion.get("status") in ("completed", "complete"),
            "HF completion is not successful",
        )
        completion_admission = completion.get("candidate_admission_sha256") or completion.get("admission_sha256")
        require(completion_admission == admission_sha, "HF completion admission mismatch")
        completion_hashes[item["path"]] = item["sha256"]
    require(completion_hashes, "HF evidence package has no terminal completion files")
    outputs = {}
    for item in package.get("outputs", []):
        path = ROOT / item.get("path", "")
        require(path.is_file() and sha(path) == item.get("sha256"), "HF output path/hash mismatch")
        row = read(path)
        require(row.get("status") in ("complete", "completed"), "HF result is not complete")
        outputs[item["path"]] = item["sha256"]
    require(outputs, "HF evidence package has no terminal result outputs")
    review_ref = package.get("root_output_review", {})
    review_path = ROOT / review_ref.get("path", "")
    require(
        review_path.is_file() and sha(review_path) == review_ref.get("sha256"),
        "HF root output review path/hash mismatch",
    )
    review = read(review_path)
    require(
        review.get("reviewer") == "root"
        and review.get("accepted") is True
        and review.get("status") == "completed_root_reviewed"
        and review.get("output_sha256") == outputs,
        "HF root output review does not bind the exact result set",
    )
    require(
        package.get("completion_sha256") == completion_hashes and package.get("output_sha256") == outputs,
        "HF evidence package completion/output inventory mismatch",
    )
    return {
        "package_sha256": sha(package_path),
        "admission_sha256": admission_sha,
        "completion_sha256": completion_hashes,
        "output_sha256": outputs,
        "root_output_review_sha256": sha(review_path),
    }


def _verify_h9_terminal_review(path, expected_sk_evidence):
    h9_admission_path = H9 / "admission.json"
    progress_path = H9 / "progress.json"
    require(h9_admission_path.is_file() and progress_path.is_file(), "H9 is not admitted and running")
    admission = read(h9_admission_path)
    require(admission.get("status") == "admitted", "H9 admission is not active")
    for field, relpath in (
        ("proposal_sha256", "proposal.json"),
        ("driver_sha256", "run_expanded_after_predecessors.py"),
        ("commands_sha256", "commands.json"),
        ("source_integrity_sha256", "source-integrity.json"),
        ("forecast_sha256", "forecast.json"),
    ):
        require(sha(H9 / relpath) == admission.get(field), f"H9 {field} pin mismatch")
    progress = read(progress_path)
    require(progress.get("status") == "all_groups_completed", "H9 HF serialization batch is not terminal")
    require(progress.get("candidate_admission_sha256") == sha(h9_admission_path), "H9 progress admission mismatch")
    groups = progress.get("groups", [])
    require(
        len(groups) == 8 and all(g.get("status") == "completed" for g in groups),
        "H9 groups are not all terminal-success",
    )
    frozen_groups = read(H9 / "commands.json").get("groups", [])
    require(len(frozen_groups) == 8, "H9 frozen command set is not eight groups")
    launches, completions, outputs = {}, {}, {}
    for group in groups:
        index = int(group["serial_order"])
        lp, cp = H9 / f"launch-group-{index:02d}.json", H9 / f"completion-group-{index:02d}.json"
        require(
            lp.is_file()
            and cp.is_file()
            and sha(lp) == group.get("launch_sha256")
            and sha(cp) == group.get("completion_sha256"),
            f"H9 group{index} launch/completion pin mismatch",
        )
        launch, completion = read(lp), read(cp)
        frozen = next((g for g in frozen_groups if g.get("serial_order") == index), None)
        require(
            frozen is not None and launch.get("group") == frozen and launch.get("command") == frozen.get("command"),
            f"H9 group{index} launch differs from frozen command",
        )
        require(
            completion.get("exit_code") == 0
            and completion.get("candidate_admission_sha256") == sha(h9_admission_path)
            and completion.get("command") == frozen.get("command"),
            f"H9 group{index} did not complete its admitted command successfully",
        )
        launches[str(index)] = sha(lp)
        completions[str(index)] = sha(cp)
        results = group.get("results", {})
        require(results, f"H9 group{index} has no output inventory")
        for output, digest in results.items():
            require(Path(output).is_file() and sha(output) == digest, f"H9 output hash mismatch: {output}")
            outputs[output] = digest
    root_review = read(path)
    expected = {
        "reviewer": "root",
        "accepted": True,
        "status": "completed_root_reviewed",
        "admission_sha256": sha(h9_admission_path),
        "progress_sha256": sha(progress_path),
        "launch_sha256": launches,
        "completion_sha256": completions,
        "output_sha256": outputs,
        "sk_evidence_sha256": expected_sk_evidence,
    }
    for key, value in expected.items():
        require(root_review.get(key) == value, f"H9 root terminal review binding mismatch: {key}")
    return {
        "review_sha256": sha(path),
        "admission_sha256": sha(h9_admission_path),
        "progress_sha256": sha(progress_path),
        "completion_sha256": completions,
        "output_sha256": outputs,
    }


def check_sk_serial():
    """Verify current SK evidence, root-bound HF disposition, and H9 serialization closure."""
    corr_admission_path = SK512_BFP8 / "admission.json"
    require(corr_admission_path.is_file(), "corrected selected-BFP8 SK512 admission missing")
    corr_admission = read(corr_admission_path)
    require(corr_admission.get("status") == "root_admitted", "corrected SK512 arm is not admitted")
    for path, digest in corr_admission.get("source_pins", {}).items():
        require((ROOT / path).is_file() and sha(ROOT / path) == digest, f"corrected SK512 source pin mismatch: {path}")
    bfp8_expected = {
        "config_id": "stage12_head_bfp8_lofi",
        "weight_groups": {"attention": "bfp4", "mlp": "bfp4", "down": "bfp4", "lm_head": "bfp8_lofi"},
        "layer_exceptions": {},
        "compute_fidelities": {"attention": "LoFi", "mlp": "LoFi", "down": "LoFi", "lm_head": "LoFi"},
        "activation_dtype": "bfloat8_b",
        "residual_dtype": "bfloat16",
        "ccl_dtype": {"prefill": "bfloat8_b", "decode": "bfloat16"},
        "kv_cache_dtype": "bfloat8_b",
        "logits_dtype": "bfloat16",
        "sampling_dtype": "bfloat16",
        "embedding_dtype": "bfloat16",
        "norm_dtype": "bfloat16",
        "max_seq_len": 262144,
    }
    bfp16_expected = {
        "config_id": "stage12_bf16_hifi4_diagnostic",
        "weight_groups": {"attention": "bf16", "mlp": "bf16", "down": "bf16", "lm_head": "bf16"},
        "layer_exceptions": {},
        "compute_fidelities": {"attention": "HiFi4", "mlp": "HiFi4", "down": "HiFi4", "lm_head": "HiFi4"},
        "activation_dtype": "bfloat16",
        "residual_dtype": "bfloat16",
        "ccl_dtype": {"prefill": "bfloat16", "decode": "bfloat16"},
        "kv_cache_dtype": "bfloat16",
        "logits_dtype": "bfloat16",
        "sampling_dtype": "bfloat16",
        "embedding_dtype": "bfloat16",
        "norm_dtype": "bfloat16",
        "max_seq_len": 262144,
        "sdpa_compute_fidelity": "HiFi4",
    }
    correction = _verify_mode_completion(
        SK512_BFP8,
        "served",
        corr_admission_path,
        bfp8_expected,
        SK512_BFP8 / "root-review.json",
        SK512_BFP8 / "triage-d.json",
        old_schema=True,
        suffix_tokens=512,
    )
    old_full_admission_path = SK512 / "admission.json"
    require(old_full_admission_path.is_file(), "original full BF16 SK512 admission missing")
    old_full_admission = read(old_full_admission_path)
    require(old_full_admission.get("status") == "admitted", "original full BF16 SK512 admission is not admitted")
    for field, path in (
        ("proposal_sha256", SK512 / "proposal.json"),
        ("driver_sha256", SK512 / "run_serial_candidate.py"),
        ("source_inventory_sha256", SK512 / "source-hashes.json"),
        ("commands_sha256", SK512 / "commands.json"),
        ("forecast_sha256", SK512 / "forecast.json"),
    ):
        require(sha(path) == old_full_admission.get(field), f"original full BF16 SK512 admission pin mismatch: {field}")
    source_inventory(SK512 / "source-hashes.json")
    old_full = _verify_mode_completion(
        SK512,
        "full",
        old_full_admission_path,
        bfp16_expected,
        SK512 / "root-full-review.json",
        SK512 / "triage-d.json",
        old_schema=True,
        suffix_tokens=512,
    )

    sk_admission_path = SK1024 / "admission.json"
    require(
        sk_admission_path.is_file() and read(sk_admission_path).get("status") == "admitted",
        "matched SK1024 admission missing/not admitted",
    )
    sk_admission = read(sk_admission_path)
    for field, relpath in (
        ("proposal_sha256", "proposal.json"),
        ("driver_sha256", "run_serial_candidate.py"),
        ("source_inventory_sha256", "source-hashes.json"),
        ("commands_sha256", "commands.json"),
        ("forecast_sha256", "forecast.json"),
    ):
        require(sha(SK1024 / relpath) == sk_admission.get(field), f"SK1024 admission pin mismatch: {field}")
    source_inventory(SK1024 / "source-hashes.json")
    served = _verify_mode_completion(
        SK1024,
        "served",
        sk_admission_path,
        bfp8_expected,
        SK1024 / "root-served-review.json",
        SK1024 / "triage-d-served.json",
        suffix_tokens=1024,
    )
    full = _verify_mode_completion(
        SK1024,
        "full",
        sk_admission_path,
        bfp16_expected,
        SK1024 / "root-full-review.json",
        SK1024 / "triage-d-full.json",
        suffix_tokens=1024,
    )
    summary_path = SK1024 / "completion-summary.json"
    summary = read(summary_path)
    require(
        summary.get("status") == "completed_root_and_independent_triage_reconciled",
        "SK1024 root/independent triage reconciliation is not terminal",
    )
    require(
        summary.get("triage_hashes", {}).get("served") == sha(SK1024 / "triage-d-served.json")
        and summary.get("triage_hashes", {}).get("full") == sha(SK1024 / "triage-d-full.json"),
        "SK1024 completion summary does not bind independent triage",
    )
    require(
        served["completion_sha256"] == served["review"].get("forecast", {}).get("completion_sha256")
        and full["completion_sha256"] == full["review"].get("forecast", {}).get("completion_sha256"),
        "SK1024 root forecast review does not bind exact completion",
    )
    sk_evidence = {
        "corrected_bfp8_admission_sha256": sha(corr_admission_path),
        "corrected_bfp8_completion_sha256": correction["completion_sha256"],
        "corrected_bfp8_root_review_sha256": correction["root_review_sha256"],
        "corrected_bfp8_triage_sha256": correction["triage_sha256"],
        "old_full_bf16_admission_sha256": sha(old_full_admission_path),
        "old_full_bf16_completion_sha256": old_full["completion_sha256"],
        "old_full_bf16_root_review_sha256": old_full["root_review_sha256"],
        "old_full_bf16_triage_sha256": old_full["triage_sha256"],
        "suffix1024_admission_sha256": sha(sk_admission_path),
        "suffix1024_served_completion_sha256": served["completion_sha256"],
        "suffix1024_full_completion_sha256": full["completion_sha256"],
        "suffix1024_served_root_review_sha256": served["root_review_sha256"],
        "suffix1024_full_root_review_sha256": full["root_review_sha256"],
        "suffix1024_served_triage_sha256": served["triage_sha256"],
        "suffix1024_full_triage_sha256": full["triage_sha256"],
        "suffix1024_summary_sha256": sha(summary_path),
    }

    review_path = HERE / "predecessors/sklearn-serial-review.json"
    require(review_path.is_file(), "root SK serial review is missing")
    review = read(review_path)
    require(
        review.get("reviewer") == "root"
        and review.get("accepted") is True
        and review.get("status") == "sk_evidence_root_reviewed",
        "root SK serial review is not accepted/terminal",
    )
    require(
        review.get("sk_evidence_sha256") == sk_evidence,
        "root SK review does not bind current corrected/SK1024 evidence",
    )
    hf_decision_path = HERE / "predecessors/hf-followup-decision.json"
    require(
        review.get("hf_decision_path") == hf_decision_path.relative_to(ROOT).as_posix()
        and review.get("hf_decision_sha256") == sha(hf_decision_path),
        "root SK review does not bind exact HF decision file",
    )
    hf_decision = read(hf_decision_path)
    require(
        hf_decision.get("reviewer") == "root" and hf_decision.get("accepted") is True,
        "root HF follow-up decision missing/not accepted",
    )
    require(
        hf_decision.get("sk_evidence_sha256") == sk_evidence, "HF follow-up decision is not bound to exact SK evidence"
    )
    h9_review_path = HERE / "predecessors/h9-terminal-review.json"
    require(
        review.get("h9_terminal_review_path") == h9_review_path.relative_to(ROOT).as_posix()
        and review.get("h9_terminal_review_sha256") == sha(h9_review_path),
        "root SK review does not bind exact H9 terminal serialization record",
    )
    h9 = _verify_h9_terminal_review(h9_review_path, sk_evidence)
    hf_state = hf_decision.get("decision")
    if hf_state == "completed":
        hf_ref = hf_decision.get("hf_evidence_package", {})
        package_path = ROOT / hf_ref.get("path", "")
        require(
            package_path.is_file() and sha(package_path) == hf_ref.get("sha256"),
            "HF completed decision lacks a hash-bound evidence package",
        )
        hf_result = _hf_evidence_package(package_path, sk_evidence)
    elif hf_state == "not_required":
        require(
            isinstance(hf_decision.get("reason"), str) and hf_decision["reason"].strip(),
            "HF not_required decision needs an explicit reason",
        )
        require(
            hf_decision.get("not_required_basis_sha256") == sk_evidence,
            "HF not_required decision is not bound to exact reviewed SK evidence",
        )
        hf_result = {
            "decision": "not_required",
            "decision_sha256": sha(hf_decision_path),
            "reason": hf_decision["reason"],
        }
    else:
        raise RuntimeError("HF follow-up decision must be completed or not_required")
    return {
        "sk_evidence_sha256": sk_evidence,
        "serial_review_sha256": sha(review_path),
        "hf_decision_sha256": sha(hf_decision_path),
        "hf_followup": hf_result,
        "h9_terminal": h9,
        "h9_terminal_review_sha256": sha(h9_review_path),
    }


def active_hf_processes():
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except (OSError, PermissionError):
            continue
        if "precision_repro_hf.py" in args:
            found.append({"pid": int(proc.name), "cmdline": args[:300]})
    return found


def device_owners():
    sys.path.insert(0, str(ROOT / "models/autoports/google_gemma_4_31b_it/tools"))
    import benchmark_server

    return benchmark_server.device_owners()


def self_test():
    # Pure in-memory mock checks: no subprocess, device query, or TTNN import.
    assert validate_review(
        {
            "reviewer": "root",
            "accepted": True,
            "forecast_range_seconds": [10, 20],
            "reviewed_elapsed_seconds": 15,
            "forecast_status": "in_range",
        },
        (10, 20),
        15,
    )
    assert not validate_review(
        {
            "reviewer": "root",
            "accepted": True,
            "forecast_range_seconds": [10, 20],
            "reviewed_elapsed_seconds": 25,
            "forecast_status": "accepted_miss",
            "contention_disclosed": True,
            "speed_ranking_used": False,
        },
        (10, 20),
        25,
        contention=True,
    )
    for bad in (
        {
            "reviewer": "root",
            "accepted": True,
            "forecast_range_seconds": [11, 20],
            "reviewed_elapsed_seconds": 15,
            "forecast_status": "in_range",
        },
        {
            "reviewer": "not-root",
            "accepted": True,
            "forecast_range_seconds": [10, 20],
            "reviewed_elapsed_seconds": 15,
            "forecast_status": "in_range",
        },
    ):
        try:
            validate_review(bad, (10, 20), 15)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid root forecast review accepted")
    expected = {"config_id": "test-policy", "weight_groups": {"down": "bfp8_lofi"}}
    job = {"name": "mock-case", "case": "case.json", "seed": 42, "free_run_tokens": 4}
    with tempfile.TemporaryDirectory(prefix="p3-gate-mock-") as tmp:
        tmp = Path(tmp)
        outdir = tmp / "out"
        outdir.mkdir()
        launch_path, completion_path, log_path = tmp / "launch.json", tmp / "completion.json", tmp / "panel.log"

        def success_runner(argv, **kwargs):
            (outdir / "mock-case.json").write_text(
                json.dumps(
                    {
                        "case": "case.json",
                        "metadata": {"seed": 42},
                        "runtime_precision": expected,
                        "free_running": [{"token_ids": [1, 2], "sampling": {"seed": 42}}],
                    }
                )
            )
            return type("Completed", (), {"returncode": 0})()

        success = run_panel_with_completion(
            ["mock"],
            outdir,
            [job],
            expected,
            {"status": "running"},
            launch_path,
            completion_path,
            log_path,
            runner=success_runner,
        )
        assert success["status"] == "completed" and completion_path.is_file()

        def invalid_runner(argv, **kwargs):
            (outdir / "mock-case.json").write_text(
                json.dumps(
                    {
                        "case": "case.json",
                        "metadata": {"seed": 42},
                        "runtime_precision": {"config_id": "wrong"},
                        "free_running": [{"token_ids": [1], "sampling": {"seed": 42}}],
                    }
                )
            )
            return type("Completed", (), {"returncode": 0})()

        invalid = run_panel_with_completion(
            ["mock"],
            outdir,
            [job],
            expected,
            {"status": "running"},
            launch_path,
            completion_path,
            log_path,
            runner=invalid_runner,
        )
        assert invalid["status"] == "invalid_outputs" and any(
            "runtime precision mismatch" in e for e in invalid["validation_errors"]
        )

        def timeout_runner(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

        timeout = run_panel_with_completion(
            ["mock"],
            outdir,
            [job],
            expected,
            {"status": "running"},
            launch_path,
            completion_path,
            log_path,
            runner=timeout_runner,
        )
        assert timeout["status"] == "failed" and timeout["exit_code"] == 124 and timeout["timed_out"] is True
        assert completion_path.is_file() and json.loads(completion_path.read_text())["process_error"]
    print("down candidate host-gate self-test passed; no TTNN import or device query")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    static = validate_static()
    if args.validate_only:
        print(json.dumps(static, sort_keys=True))
        return
    require(args.execute, "explicit --execute required")
    admission, _ = candidate_gate(admission_required=True)
    require(not STOP.exists(), "STOP_BEFORE_LAUNCH is present")
    p2 = check_p2_terminal()
    sklearn = check_sk_serial()
    review_path = HERE / "predecessors/forecast-review.json"
    require(review_path.is_file(), "root candidate forecast review missing")
    review = read(review_path)
    forecast = read(HERE / "forecast.json")
    expected = (forecast["forecast_seconds"]["low"], forecast["forecast_seconds"]["high"])
    require(
        review.get("forecast_sha256") == sha(HERE / "forecast.json")
        and review.get("forecast_range_seconds") == list(expected)
        and review.get("reviewer") == "root"
        and review.get("accepted") is True
        and review.get("uncertainty_acknowledged") is True
        and review.get("p2_contention_excluded_from_speed_calibration") is True,
        "root candidate forecast review missing, stale, or incomplete",
    )
    hf = active_hf_processes()
    require(not hf, f"active HF process(es): {hf}")
    owners = device_owners()
    require(not owners, f"device owner(s): {owners}")
    launch_path = HERE / "launch.json"
    completion_path = HERE / "completion.json"
    require(not launch_path.exists() and not completion_path.exists(), "duplicate launch refused")
    outdir = ROOT / read(HERE / "command.json")["argv"][read(HERE / "command.json")["argv"].index("--output-dir") + 1]
    require(not outdir.exists() or not any(outdir.iterdir()), "nonempty output directory refused")
    argv = read(HERE / "command.json")["argv"]
    expected_precision, build_kwargs = candidate_runtime_policy()
    outdir.mkdir(parents=True, exist_ok=True)
    launch = {
        "status": "running",
        "command": argv,
        "start_unix": time.time(),
        "timeout_seconds": 9000,
        "candidate_admission_sha256": sha(HERE / "admission.json"),
        "candidate_proposal_sha256": sha(HERE / "proposal.json"),
        "candidate_source_inventory_sha256": sha(HERE / "source-hashes.json"),
        "forecast": forecast,
        "expected_runtime_precision": expected_precision,
        "build_kwargs": build_kwargs,
        "p2_terminal": p2,
        "sklearn_serial": sklearn,
        "forecast_review_sha256": sha(review_path),
        "active_hf_processes": hf,
        "device_owner_preflight": owners,
    }
    completion = run_panel_with_completion(
        argv,
        outdir,
        read(HERE / "candidate-cases.json"),
        expected_precision,
        launch,
        launch_path,
        completion_path,
        HERE / "panel.log",
        timeout_seconds=9010,
    )
    if completion["status"] != "completed":
        print(
            json.dumps(
                {
                    "status": completion["status"],
                    "exit_code": completion["exit_code"],
                    "timed_out": completion["timed_out"],
                    "validation_errors": completion["validation_errors"],
                }
            )
        )
        raise RuntimeError("panel ended without a valid complete result set; no retry")
    print(
        json.dumps(
            {
                "status": completion["status"],
                "elapsed_seconds": completion["elapsed_seconds"],
                "outputs": len(completion["outputs"]),
            }
        )
    )


if __name__ == "__main__":
    main()
