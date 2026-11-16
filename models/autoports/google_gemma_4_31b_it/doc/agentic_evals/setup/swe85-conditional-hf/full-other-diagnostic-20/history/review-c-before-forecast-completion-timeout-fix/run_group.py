#!/usr/bin/env python3
"""Run one root-admitted full-other HF group. Inert without explicit mode and group."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[8]
assert (ROOT / "bringup/run_contract.json").is_file(), "repository root resolution failed"
OUT = Path(__file__).resolve().parent
PROPOSAL = OUT / "proposal.json"
COMMANDS = OUT / "commands.json"
FORECAST = OUT / "forecast.json"
SOURCE_INTEGRITY = OUT / "source-integrity.json"
DRIVER_INTEGRITY = OUT / "driver-integrity.json"
DRIVER_CONTRACT = OUT / "driver-contract.json"
ADMISSION = OUT / "admission.json"
PREDECESSOR_GATE = OUT / "predecessor-gate.json"
STOP = OUT / "STOP_BEFORE_LAUNCH"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
MIN_MEM = 100 * 1024**3


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path | str):
    return json.loads(Path(path).read_text())


def write_exclusive(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def verify_hash(path: Path, expected: str, label: str):
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash changed: expected {expected}, got {actual}")
    return actual


def verify_static_bundle(base: Path = OUT, root: Path = ROOT):
    """Verify the frozen proposal and every saved input without importing model code."""
    proposal = read(base / "proposal.json")
    commands = read(base / "commands.json")
    forecast = read(base / "forecast.json")
    source_integrity = read(base / "source-integrity.json")
    driver_integrity = read(base / "driver-integrity.json")
    contract = read(base / "driver-contract.json")

    if proposal.get("status") != "candidate_only_not_admitted_not_run":
        raise RuntimeError("proposal status is not inert candidate-only")
    if commands.get("status") != "candidate_only_unadmitted_not_run":
        raise RuntimeError("commands status is not inert candidate-only")
    if forecast.get("status") != "planning_forecast_candidate_only_not_execution":
        raise RuntimeError("forecast status changed")
    if len(commands.get("groups", [])) != 8 or len(forecast.get("groups", [])) != 8:
        raise RuntimeError("expected exactly eight frozen groups")
    if proposal.get("commands") != {"path": "commands.json", "sha256": sha(base / "commands.json")}:
        raise RuntimeError("proposal does not pin commands")
    if proposal.get("forecast") != {"path": "forecast.json", "sha256": sha(base / "forecast.json")}:
        raise RuntimeError("proposal does not pin forecast")
    if proposal.get("source_integrity") != {
        "path": "source-integrity.json",
        "sha256": sha(base / "source-integrity.json"),
    }:
        raise RuntimeError("proposal does not pin source integrity")
    if commands.get("source_integrity_sha256") != sha(base / "source-integrity.json"):
        raise RuntimeError("commands source-integrity pin mismatch")
    if commands.get("forecast_sha256") != sha(base / "forecast.json"):
        raise RuntimeError("commands forecast pin mismatch")
    if driver_integrity.get("candidate_pins") != {
        "proposal_sha256": sha(base / "proposal.json"),
        "commands_sha256": sha(base / "commands.json"),
        "forecast_sha256": sha(base / "forecast.json"),
        "source_integrity_sha256": sha(base / "source-integrity.json"),
    }:
        raise RuntimeError("driver integrity candidate pins mismatch")
    driver_path = Path(__file__).resolve()
    if driver_integrity.get("driver_sha256") != sha(driver_path):
        raise RuntimeError("execution driver hash mismatch")
    if driver_integrity.get("contract_sha256") != sha(base / "driver-contract.json"):
        raise RuntimeError("driver contract hash mismatch")
    if driver_integrity.get("tests_sha256") != sha(base / "test_run_group.py"):
        raise RuntimeError("host test hash mismatch")
    if contract.get("execution_default") != "inert" or contract.get("group_invocation") != "--execute --group N":
        raise RuntimeError("driver execution contract changed")

    source_files = source_integrity.get("sources", {})
    volatile = set(contract.get("volatile_predecessor_snapshot_paths", []))
    for rel, digest in source_files.items():
        # HF9 progress is a live execution record and can advance while this candidate waits.
        # Its frozen-at-preparation digest stays pinned in the source inventory; execution instead
        # requires a separate root-signed terminal predecessor gate that pins the final completion.
        if rel in volatile:
            continue
        verify_hash(root / rel, digest, "frozen source")
    runner_rel = "models/autoports/google_gemma_4_31b_it/tools/precision_repro_hf.py"
    if driver_integrity.get("runner_sha256") != source_files.get(runner_rel):
        raise RuntimeError("driver/runner source pin mismatch")
    roster_review_path = base / "root-roster-review.json"
    if driver_integrity.get("roster_review_sha256") != sha(roster_review_path):
        raise RuntimeError("root roster review hash mismatch")
    roster_review = read(roster_review_path)
    if (
        roster_review.get("status") != "root_roster_review_complete_no_execution_admission"
        or roster_review.get("reviewer") != "root"
    ):
        raise RuntimeError("root roster review status mismatch")
    if roster_review.get("proposal_sha256") != sha(base / "proposal.json") or roster_review.get(
        "forecast_sha256"
    ) != sha(base / "forecast.json"):
        raise RuntimeError("root roster review candidate pins mismatch")
    reviewed = roster_review.get("checks", [])
    if (
        roster_review.get("observations") != 20
        or len(reviewed) != 20
        or any(not row.get("source_and_fixture_hashes_verified") for row in reviewed)
    ):
        raise RuntimeError("root roster review did not verify every observation")
    if len(commands["groups"]) != 8:
        raise RuntimeError("group count changed")
    orders = []
    cases = proposal.get("workload", {}).get("observations", [])
    case_by_name = {case["name"]: case for case in cases}
    if len(cases) != 20 or len(case_by_name) != 20:
        raise RuntimeError("proposal no longer contains 20 unique observations")
    total_cases = 0
    for group, forecast_row in zip(commands["groups"], forecast["groups"]):
        order = group.get("order")
        orders.append(order)
        if (
            forecast_row.get("group") != order
            or forecast_row.get("seed") != group.get("seed")
            or forecast_row.get("cap") != group.get("cap")
        ):
            raise RuntimeError(f"forecast/group identity mismatch at {order}")
        if group.get("timeout_seconds") != forecast_row.get("timeout_seconds") or group["timeout_seconds"] != 18000:
            raise RuntimeError(f"timeout mismatch at group {order}")
        if group.get("manifest_sha256") != sha(base / group["manifest"]):
            raise RuntimeError(f"manifest hash mismatch at group {order}")
        jobs = read(base / group["manifest"])
        if len(jobs) != group["case_count"] or len(jobs) != len(group["cases"]):
            raise RuntimeError(f"manifest case count mismatch at group {order}")
        argv = group.get("command", [])
        if argv[:2] != ["timeout", "18000"] or "--compare-dir" in argv:
            raise RuntimeError(f"command contract mismatch at group {order}")
        if argv[argv.index("--cases") + 1] != str((base / group["manifest"]).resolve()):
            raise RuntimeError(f"command manifest path mismatch at group {order}")
        if argv[argv.index("--output") + 1] != group["output_dir"]:
            raise RuntimeError(f"command output path mismatch at group {order}")
        if argv[argv.index("--tokens") + 1] != str(group["cap"]):
            raise RuntimeError(f"command cap mismatch at group {order}")
        if argv[argv.index("--seeds") + 1] != str(group["seed"]):
            raise RuntimeError(f"command seed mismatch at group {order}")
        if argv[argv.index("--prefill-chunk") + 1] != "512":
            raise RuntimeError(f"prefill chunk mismatch at group {order}")
        if "--validate-only" in argv:
            raise RuntimeError("saved generation command must not contain --validate-only")
        for env in (
            "HF_HUB_OFFLINE=1",
            "HF_DATASETS_OFFLINE=1",
            "OMP_NUM_THREADS=16",
            "MKL_NUM_THREADS=16",
            "OPENBLAS_NUM_THREADS=16",
        ):
            if env not in argv:
                raise RuntimeError(f"missing command environment {env}")
        for job in jobs:
            named = case_by_name.get(job["name"])
            if named is None or job["case"] != named["case_path"]:
                raise RuntimeError(f"manifest case mapping mismatch: {job['name']}")
            if job["branches"] != ["chosen"] or job["free_run_tokens"] != group["cap"] or job["free_run_trials"] != 1:
                raise RuntimeError(f"manifest sampling/cap mismatch: {job['name']}")
            case_path = Path(job["case"])
            case = read(case_path)
            if sha(case_path) != named["case_sha256"] or len(case["prompt_token_ids"]) != named["prompt_tokens"]:
                raise RuntimeError(f"case fixture identity mismatch: {job['name']}")
            ids_hash = hashlib.sha256(json.dumps(case["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest()
            identity = hashlib.sha256(f"{ids_hash}:{group['seed']}:{group['cap']}".encode()).hexdigest()
            if (
                ids_hash != named["prompt_token_ids_sha256"]
                or identity != named["exact_prompt_seed_cap_identity_sha256"]
            ):
                raise RuntimeError(f"exact input/seed/cap identity mismatch: {job['name']}")
            if len(case.get("continuations", {}).get("chosen", [])) != named["source_chosen_tokens"]:
                raise RuntimeError(f"chosen source length mismatch: {job['name']}")
            if named["chosen_teacher_forcing_positions"] != min(64, len(case["continuations"]["chosen"])):
                raise RuntimeError(f"chosen position count mismatch: {job['name']}")
        total_cases += len(jobs)
    if orders != list(range(1, 9)) or total_cases != 20:
        raise RuntimeError("group ordering or total case count mismatch")
    return proposal, commands, forecast, source_integrity, driver_integrity, contract


def verify_admission(base: Path, group, bundle, now=None):
    proposal, commands, forecast, source_integrity, driver_integrity, contract = bundle
    admission_path = base / "admission.json"
    admission = read(admission_path)
    if admission.get("status") != "admitted" or admission.get("reviewer") != "root":
        raise RuntimeError("root admission for this exact group is required")
    if admission.get("driver_integrity_sha256") != sha(base / "driver-integrity.json"):
        raise RuntimeError("admission driver-integrity pin mismatch")
    if admission.get("driver_sha256") != sha(Path(__file__).resolve()):
        raise RuntimeError("admission driver pin mismatch")
    for key, path in (
        ("proposal_sha256", base / "proposal.json"),
        ("commands_sha256", base / "commands.json"),
        ("forecast_sha256", base / "forecast.json"),
        ("source_integrity_sha256", base / "source-integrity.json"),
        ("driver_contract_sha256", base / "driver-contract.json"),
        ("tests_sha256", base / "test_run_group.py"),
    ):
        if admission.get(key) != sha(path):
            raise RuntimeError(f"admission {key} mismatch")
    authz = admission.get("group_authorizations", {})
    if set(authz) != {str(group["order"])}:
        raise RuntimeError("admission must contain exactly this group's authorization")
    auth = authz.get(str(group["order"]))
    if not auth or auth.get("status") != "authorized" or auth.get("reviewer") != "root":
        raise RuntimeError(f"root has not authorized group {group['order']}")
    if admission.get("authorized_groups") != [group["order"]]:
        raise RuntimeError("admission must authorize exactly this single group")
    forecast_row = next(x for x in forecast["groups"] if x["group"] == group["order"])
    if auth.get("forecast_sha256") != sha(base / "forecast.json"):
        raise RuntimeError("group authorization does not pin current forecast")
    if auth.get("forecast_range_seconds") != forecast_row["sensitivity_range_seconds"]:
        raise RuntimeError("root group forecast range acceptance mismatch")
    ceiling = auth.get("accepted_runtime_ceiling_seconds")
    deadline = auth.get("deadline_unix")
    if not isinstance(ceiling, (int, float)) or ceiling <= 0 or not isinstance(deadline, (int, float)):
        raise RuntimeError("root group authorization needs accepted runtime ceiling and deadline")
    now = time.time() if now is None else now
    if now + ceiling > deadline:
        raise RuntimeError("accepted group runtime ceiling exceeds root-authorized deadline")
    if ceiling < forecast_row["sensitivity_range_seconds"][0]:
        raise RuntimeError("accepted runtime ceiling is below the group's forecast lower bound")
    if admission.get("predecessor_gate_sha256") != sha(base / "predecessor-gate.json"):
        raise RuntimeError("admission does not pin terminal predecessor gate")
    return admission, sha(admission_path), auth


def proc_table():
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            if cmd:
                rows.append((int(entry.name), cmd, entry))
        except OSError:
            continue
    return rows


def check_resource_snapshot(hf, owners, mem_bytes):
    if hf:
        raise RuntimeError(f"active HF process blocks execution: {hf}")
    if owners:
        raise RuntimeError(f"TT device owners block execution: {owners}")
    if mem_bytes < MIN_MEM:
        raise RuntimeError(f"MemAvailable below 100GiB: {mem_bytes} bytes")
    return {
        "active_hf_processes": hf,
        "device_owners": owners,
        "mem_available_bytes": mem_bytes,
        "required_mem_available_bytes": MIN_MEM,
    }


def preflight_resources():
    pid = os.getpid()
    hf = [{"pid": p, "cmdline": cmd[:400]} for p, cmd, _ in proc_table() if p != pid and "precision_repro_hf.py" in cmd]
    owners = []
    for p, cmd, entry in proc_table():
        if p == pid:
            continue
        for fd in (entry / "fd").glob("*"):
            try:
                if os.readlink(fd).startswith("/dev/tenstorrent/"):
                    owners.append({"pid": p, "cmdline": cmd[:400]})
                    break
            except OSError:
                continue
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    available = fields.get("MemAvailable", "")
    if not available.endswith("kB"):
        raise RuntimeError("cannot read MemAvailable")
    mem_bytes = int(available[:-2].strip()) * 1024
    return check_resource_snapshot(hf, owners, mem_bytes)


def verify_predecessors(base: Path):
    gate_path = base / "predecessor-gate.json"
    gate = read(gate_path)
    if gate.get("status") != "root_accepted_predecessors_terminal" or gate.get("reviewer") != "root":
        raise RuntimeError("HF9, SKHF, and P3 require root-accepted terminal predecessor evidence")
    items = gate.get("predecessors", {})
    for name in ("hf9", "skhf", "p3"):
        item = items.get(name)
        if not item or item.get("status") not in ("completed", "terminal") or item.get("exit_code") != 0:
            raise RuntimeError(f"predecessor {name} is missing or not root-accepted terminal success")
        path = Path(item["completion_path"])
        if not path.is_absolute():
            path = ROOT / path
        verify_hash(path, item.get("completion_sha256", ""), f"{name} completion")
    return {"gate_sha256": sha(gate_path), "predecessors": items}


def verify_prior_group(base: Path, previous_group, forecast):
    n = previous_group["order"]
    launch_path = base / f"launch-group-{n:02d}.json"
    completion_path = base / f"completion-group-{n:02d}.json"
    review_path = base / f"root-review-group-{n:02d}.json"
    launch, completion, review = read(launch_path), read(completion_path), read(review_path)
    if (
        launch.get("status") != "launched"
        or completion.get("status") != "completed"
        or completion.get("driver_exit_code") != 0
    ):
        raise RuntimeError(f"prior group {n} lacks successful terminal completion")
    prior_admission_sha = launch.get("admission_sha256")
    if not isinstance(prior_admission_sha, str) or len(prior_admission_sha) != 64:
        raise RuntimeError(f"prior group {n} has no admission hash")
    if (
        completion.get("admission_sha256") != prior_admission_sha
        or review.get("candidate_admission_sha256") != prior_admission_sha
    ):
        raise RuntimeError(f"prior group {n} admission lineage mismatch")
    if launch.get("group") != n or completion.get("group") != n:
        raise RuntimeError(f"prior group {n} record identity mismatch")
    if completion.get("output_validation_status") != "passed":
        raise RuntimeError(f"prior group {n} outputs were not validated")
    if completion.get("launch_sha256") != sha(launch_path):
        raise RuntimeError(f"prior group {n} launch pin mismatch")
    forecast_row = next(x for x in forecast["groups"] if x["group"] == n)
    if completion.get("forecast_range_seconds") != forecast_row["sensitivity_range_seconds"]:
        raise RuntimeError(f"prior group {n} forecast range mismatch")
    measured = completion.get("elapsed_seconds")
    if not isinstance(measured, (int, float)):
        raise RuntimeError(f"prior group {n} missing measured wall")
    in_range = forecast_row["sensitivity_range_seconds"][0] <= measured <= forecast_row["sensitivity_range_seconds"][1]
    forecast_gate = "hit"
    miss_sha = None
    if not in_range:
        miss_path = base / f"forecast-miss-accepted-group-{n:02d}.json"
        miss = read(miss_path)
        expected = {
            "status": "root_accepted_forecast_miss",
            "reviewer": "root",
            "accepted": True,
            "completion_sha256": sha(completion_path),
            "admission_sha256": prior_admission_sha,
            "measured_seconds": measured,
            "forecast_range_seconds": forecast_row["sensitivity_range_seconds"],
            "group": n,
        }
        for key, value in expected.items():
            if miss.get(key) != value:
                raise RuntimeError(f"forecast-miss record {key} mismatch")
        miss_sha = sha(miss_path)
        forecast_gate = "root_accepted_miss"
    expected_review = {
        "status": "root_reviewed_forecast_and_outputs_next_group_authorized",
        "reviewer": "root",
        "candidate_admission_sha256": prior_admission_sha,
        "launch_sha256": sha(launch_path),
        "completion_sha256": sha(completion_path),
        "forecast_gate": forecast_gate,
        "result_sha256s": completion.get("result_sha256s"),
    }
    for key, value in expected_review.items():
        if review.get(key) != value:
            raise RuntimeError(f"prior group root review mismatch: {key}")
    if miss_sha and review.get("forecast_miss_acceptance_sha256") != miss_sha:
        raise RuntimeError("root review did not pin forecast miss acceptance")
    if not completion.get("result_sha256s"):
        raise RuntimeError(f"prior group {n} missing bound output hashes")
    for rel, digest in completion["result_sha256s"].items():
        verify_hash(ROOT / rel, digest, "prior group result")
    return {
        "group": n,
        "completion_sha256": sha(completion_path),
        "root_review_sha256": sha(review_path),
        "forecast_gate": forecast_gate,
        "forecast_miss_acceptance_sha256": miss_sha,
    }


def verify_outputs(base: Path, root: Path, group):
    output = Path(group["output_dir"])
    if not output.is_absolute():
        output = root / output
    jobs = read(base / group["manifest"])
    records = {x["name"]: x for x in read(base / "proposal.json")["workload"]["observations"]}
    hashes = {}
    for job in jobs:
        name = job["name"]
        rec = records[name]
        case_path = Path(job["case"])
        case = read(case_path)
        result_path = output / (name + ".hf.private.json")
        if not result_path.is_file():
            raise RuntimeError(f"missing HF result: {result_path}")
        result = read(result_path)
        if result.get("status") != "complete":
            raise RuntimeError(f"incomplete result: {name}")
        if (
            result.get("revision") != REVISION
            or result.get("device") != "cpu"
            or result.get("weights_dtype") != "bfloat16"
        ):
            raise RuntimeError(f"wrong revision/device/dtype: {name}")
        if (
            result.get("prompt_source") != "Saved prompt_token_ids, no chat rerender"
            or result.get("prefill_chunk") != 512
        ):
            raise RuntimeError(f"prompt source/prefill contract mismatch: {name}")
        if Path(result.get("case", "")).resolve() != case_path.resolve():
            raise RuntimeError(f"result case path mismatch: {name}")
        if result.get("case_sha256") != sha(case_path) or result.get("prompt_tokens") != rec["prompt_tokens"]:
            raise RuntimeError(f"result prompt source mismatch: {name}")
        positions = result.get("teacher_forced_chosen", {}).get("positions", [])
        if len(positions) != rec["chosen_teacher_forcing_positions"]:
            raise RuntimeError(f"teacher-forcing positions mismatch: {name}")
        trials = result.get("results", [])
        if len(trials) != 1 or trials[0].get("seed") != group["seed"]:
            raise RuntimeError(f"trial seed mismatch: {name}")
        if trials[0].get("requested_tokens") != group["cap"]:
            raise RuntimeError(f"requested generation cap mismatch: {name}")
        generated = trials[0].get("generated_tokens")
        token_ids = trials[0].get("generated_token_ids", [])
        if not isinstance(generated, int) or generated < 1 or generated > group["cap"] or len(token_ids) != generated:
            raise RuntimeError(f"generated output length/cap mismatch: {name}")
        if result.get("tt_comparisons", {}) not in ({}, None):
            raise RuntimeError(f"unexpected TT comparison payload: {name}")
        hashes[result_path.relative_to(root).as_posix()] = sha(result_path)
    return hashes


def _run_group(group_order, base=OUT, root=ROOT, hooks=None):
    """Execute one group; hooks exist only to host-mock gate/run behavior in tests."""
    hooks = hooks or {}
    launch_path = base / f"launch-group-{group_order:02d}.json"
    completion_path = base / f"completion-group-{group_order:02d}.json"
    if launch_path.exists() or completion_path.exists():
        raise RuntimeError("refusing overwrite of existing launch/completion record")
    start = time.time()
    terminal = {
        "status": "preflight_failed",
        "group": group_order,
        "started_unix": start,
        "candidate_admission_sha256": None,
        "child_exit_code": None,
        "driver_exit_code": 2,
    }
    try:
        bundle = hooks.get("verify_bundle", verify_static_bundle)(base, root)
        proposal, commands, forecast, source_integrity, driver_integrity, contract = bundle
        group = next((x for x in commands["groups"] if x["order"] == group_order), None)
        if group is None:
            raise RuntimeError(f"unknown group {group_order}")
        admission, admission_sha, auth = hooks.get("verify_admission", verify_admission)(base, group, bundle)
        terminal["candidate_admission_sha256"] = admission_sha
        if (base / "STOP_BEFORE_LAUNCH").exists():
            raise RuntimeError("STOP_BEFORE_LAUNCH is set")
        predecessor = hooks.get("verify_predecessors", verify_predecessors)(base)
        prior = None
        if group_order > 1:
            prior_group = commands["groups"][group_order - 2]
            prior = hooks.get("verify_prior_group", verify_prior_group)(base, prior_group, forecast)
        resources = hooks.get("preflight_resources", preflight_resources)()
        output = Path(group["output_dir"])
        if not output.is_absolute():
            output = root / output
        if output.exists():
            raise RuntimeError(f"refusing pre-existing output directory {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        launch = {
            "status": "launched",
            "group": group_order,
            "started_unix": start,
            "command": group["command"],
            "admission_sha256": admission_sha,
            "authorized_group": group_order,
            "forecast_sha256": sha(base / "forecast.json"),
            "forecast_range_seconds": next(x for x in forecast["groups"] if x["group"] == group_order)[
                "sensitivity_range_seconds"
            ],
            "predecessor_gate": predecessor,
            "prior_group": prior,
            "preflight": resources,
        }
        write_exclusive(launch_path, launch)
        terminal.update(
            {
                "status": "failed",
                "launch_sha256": sha(launch_path),
                "admission_sha256": admission_sha,
                "command": group["command"],
                "preflight": resources,
                "predecessor_gate": predecessor,
                "prior_group": prior,
            }
        )
        log_path = base / f"group-{group_order:02d}.log"
        child_code = None
        start_mono = time.monotonic()
        try:
            with open(log_path, "xb") as log:
                completed = hooks.get("subprocess_run", subprocess.run)(
                    group["command"],
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=group["timeout_seconds"] + 60,
                    check=False,
                )
            child_code = completed.returncode
        except subprocess.TimeoutExpired:
            child_code = 124
            terminal["child_timeout"] = True
        except Exception as exc:
            terminal["launch_error"] = f"{type(exc).__name__}: {exc}"
            child_code = 127
        elapsed = time.monotonic() - start_mono
        terminal["child_exit_code"] = child_code
        terminal["elapsed_seconds"] = elapsed
        terminal["log_path"] = log_path.relative_to(root).as_posix()
        if log_path.exists():
            terminal["log_sha256"] = sha(log_path)
        if child_code == 0:
            try:
                result_hashes = hooks.get("verify_outputs", verify_outputs)(base, root, group)
                terminal.update(
                    {
                        "status": "completed",
                        "output_validation_status": "passed",
                        "result_sha256s": result_hashes,
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
        else:
            terminal.update(
                {
                    "status": "child_failed",
                    "output_validation_status": "not_run_child_failed",
                    "driver_exit_code": child_code or 1,
                }
            )
    except Exception as exc:
        terminal["error"] = f"{type(exc).__name__}: {exc}"
    terminal["finished_unix"] = time.time()
    if not completion_path.exists():
        write_exclusive(completion_path, terminal)
    return terminal


def validate_only():
    verify_static_bundle()
    print("Static candidate validation passed; no admission check, runner, model load, or device access was performed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--group", type=int, choices=range(1, 9))
    args = parser.parse_args(argv)
    if args.validate_only:
        if args.group is not None:
            parser.error("--group is only valid with --execute")
        validate_only()
        return 0
    if args.execute and args.group is None:
        parser.error("--execute requires one --group N; no automatic advance")
    outcome = _run_group(args.group)
    print(
        json.dumps(
            {
                k: outcome.get(k)
                for k in ("status", "group", "child_exit_code", "driver_exit_code", "error", "output_validation_error")
            },
            indent=2,
        )
    )
    return int(outcome.get("driver_exit_code", 2))


if __name__ == "__main__":
    sys.exit(main())
