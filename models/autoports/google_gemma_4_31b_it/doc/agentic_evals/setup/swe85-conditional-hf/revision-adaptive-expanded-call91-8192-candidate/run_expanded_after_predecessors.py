#!/usr/bin/env python3
"""Inert-by-default serial HF diagnostics, gated on P and suffix8192 terminal success."""

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
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
OUT = D / "setup/swe85-conditional-hf/revision-adaptive-expanded-call91-8192-candidate"
P = D / "setup/swe85-precision-bisect/head-bfp8-hifi2-first-panel-after-w"
E8192 = D / "setup/swe85-adaptive-canonicalprefix-suffix8192-p1pinfix-candidate"
P_ADMISSION_SHA256 = "91dc764e6a3a6b1e8a1be51e6eee5fc958c19d1dfcfa3d057b989aa325618cdd"
P_ADMISSION_INVENTORY_PLACEHOLDER_SHA256 = "428bea328fae8fa6e4f6fe21802f3763985d6b5ee819f5beac6c6df3ff1f2b97"
P_SOURCE_HASHES_SHA256 = "e1254ae95f87360bc214b01757062a8910b5fed95f6e6cfd13244b024cce01ad"
E8192_ADMISSION_SHA256 = "300d69403a28112bc5f2a7225df5327bdb9e577ad9ca96cf754ab304f3538205"
P2 = D / "setup/swe85-precision-bisect/head-bf16-hifi4-fixed108-proposal"
P2_ADMISSION_SHA256 = "74bef162c97e23e882612eb85dae365f32c9da1f13ba4b9b8e1e1f8f582ad1be"
STOP_BEFORE = OUT / "STOP_BEFORE_LAUNCH"
STOP_BETWEEN = OUT / "STOP_BETWEEN_GROUPS"
ADMISSION = OUT / "admission.json"
PROPOSAL = OUT / "proposal.json"
COMMANDS = OUT / "commands.json"
INTEGRITY = OUT / "source-integrity.json"
FORECAST = OUT / "forecast.json"
RUNNER = ROOT / "models/autoports/google_gemma_4_31b_it/tools/precision_repro_hf.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_file(path, expected, label):
    path = Path(path)
    assert path.is_file(), f"missing {label}: {path}"
    actual = sha(path)
    assert actual == expected, f"hash mismatch for {label}: expected {expected}, got {actual}"
    return actual


def terminal_pin(pins, key, fallback_path):
    """Return a captured immutable pin without evaluating an unused file fallback."""
    return pins[key] if key in pins else sha(fallback_path)


def verify_inventory():
    proposal = read(PROPOSAL)
    commands = read(COMMANDS)
    integrity = read(INTEGRITY)
    assert proposal["status"] == "expanded_candidate_unadmitted_no_launch"
    assert proposal["selection"]["total_cases"] == 9
    assert proposal["manifest_groups"] == 8
    assert proposal.get("source_integrity") == {"path": "source-integrity.json", "sha256": sha(INTEGRITY)}
    assert commands.get("source_integrity") == {"path": "source-integrity.json", "sha256": sha(INTEGRITY)}
    driver_pin = integrity["execution_driver"]
    verify_file(ROOT / driver_pin["path"], driver_pin["sha256"], "execution driver")
    assert driver_pin["path"] == Path(__file__).resolve().relative_to(ROOT).as_posix()
    assert commands["runner_sha256"] == integrity["runner_sha256"] == sha(RUNNER)
    assert commands["revision"] == integrity["revision"] == proposal["workload"]["hf_revision"]
    assert len(commands["groups"]) == 8

    for name, digest in proposal["manifest_sha256s"].items():
        verify_file(OUT / name, digest, f"candidate manifest {name}")
    for name, item in integrity["candidate_manifests"].items():
        verify_file(ROOT / item["path"], item["sha256"], f"inventory manifest {name}")
        assert item["sha256"] == proposal["manifest_sha256s"][name]
    for name, item in integrity["fixtures"].items():
        verify_file(ROOT / item["path"], item["sha256"], f"fixture {name}")
    for name, item in integrity["tt_outputs"].items():
        for field, digest_field in (("path", "sha256"), ("served_path", "served_sha256"), ("full_path", "full_sha256")):
            if field in item:
                verify_file(ROOT / item[field], item[digest_field], f"TT source {name}:{field}")
    for section in ("root_reviews", "candidate_manifests", "semantic_selection_evidence"):
        for name, item in integrity.get(section, {}).items():
            verify_file(ROOT / item["path"], item["sha256"], f"{section} {name}")
    for name, item in integrity.get("native_positive_sources", {}).items():
        verify_file(ROOT / item["path"], item["sha256"], f"native source {name}")
    for row in integrity["new_candidate_sources"]:
        verify_file(D / row["fixture"], row["fixture_sha256"], f"new positive fixture {row['case_name']}")
        verify_file(D / row["served_source"], row["served_sha256"], f"new positive served source {row['case_name']}")
        verify_file(D / row["full_source"], row["full_sha256"], f"new positive full source {row['case_name']}")
    for item in integrity["semantic_selection_evidence"].values():
        for path_key, digest_key in (
            ("paired_comparison_path", "paired_comparison_sha256"),
            ("completion_summary_path", "completion_summary_sha256"),
        ):
            if path_key in item:
                verify_file(ROOT / item[path_key], item[digest_key], "semantic evidence " + path_key)
    cost_basis = integrity.get("call91_8192_cost_basis")
    if cost_basis:
        verify_file(ROOT / cost_basis["path"], cost_basis["sha256"], "call91 8192 cost basis")
        verify_file(
            ROOT / cost_basis["source_cost_basis"], cost_basis["source_cost_basis_sha256"], "HF timing baseline"
        )
        verify_file(ROOT / cost_basis["prefill_log"], cost_basis["prefill_log_sha256"], "HF prefill log")

    seen_orders = []
    case_total = 0
    for group in commands["groups"]:
        seen_orders.append(group["serial_order"])
        verify_file(group["manifest"], group["manifest_sha256"], "group manifest")
        argv = group["command"]
        assert argv[0] == "timeout" and int(argv[1]) == group["timeout_seconds"]
        assert "HF_HUB_OFFLINE=1" in argv and "HF_DATASETS_OFFLINE=1" in argv
        for env in ("OMP_NUM_THREADS=16", "MKL_NUM_THREADS=16", "OPENBLAS_NUM_THREADS=16"):
            assert env in argv, f"missing thread bound {env}"
        assert RUNNER.as_posix() in argv
        assert str(group["seed_group"]) in argv[argv.index("--seeds") + 1 : argv.index("--prefill-chunk")]
        assert argv[argv.index("--tokens") + 1] == "512"
        assert argv[argv.index("--prefill-chunk") + 1] == "512"
        assert argv.count("--compare-dir") == 2
        manifest_jobs = read(group["manifest"])
        case_total += len(manifest_jobs)
        for job in manifest_jobs:
            case = read(ROOT / job["case"])
            assert job["branches"] == ["chosen"]
            assert job["free_run_tokens"] == 512 and job["free_run_trials"] == 1
            assert case["prompt_token_ids"] and case["continuations"]["chosen"]
    assert seen_orders == list(range(1, 9)), f"unexpected serial group order: {seen_orders}"
    assert case_total == 9, f"candidate group manifests contain {case_total} cases, expected nine"
    assert integrity["semantic_selection_evidence"]["requests5414_call10_seed42_full_suffix4096"]["sha256"]
    assert integrity["semantic_selection_evidence"]["matplotlib_call86_and_sphinx_call70_native_family"]["sha256"]
    return proposal, commands, integrity


def candidate_admission():
    admission = read(ADMISSION)
    assert admission.get("status") == "admitted", "root admission required; HF candidate remains inert"
    proposal, commands, integrity = verify_inventory()
    for key, path in (
        ("proposal_sha256", PROPOSAL),
        ("driver_sha256", Path(__file__)),
        ("commands_sha256", COMMANDS),
        ("source_integrity_sha256", INTEGRITY),
        ("forecast_sha256", FORECAST),
        ("runner_sha256", RUNNER),
    ):
        verify_file(path, admission.get(key, ""), f"admitted candidate {key}")
    assert admission.get("p_admission_sha256") == P_ADMISSION_SHA256
    assert admission.get("suffix8192_admission_sha256") == E8192_ADMISSION_SHA256
    assert admission.get("p2_admission_sha256") == P2_ADMISSION_SHA256
    return admission, proposal, commands, integrity


def active_hf_processes():
    hits = []
    for ent in Path("/proc").iterdir():
        if not ent.name.isdigit() or int(ent.name) == os.getpid():
            continue
        try:
            args = (ent / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except (OSError, PermissionError):
            continue
        if "precision_repro_hf.py" in args:
            hits.append({"pid": int(ent.name), "cmdline": args[:400]})
    return hits


def available_memory_bytes():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    value = fields.get("MemAvailable", "")
    assert value.endswith("kB"), "cannot parse /proc/meminfo MemAvailable"
    return int(value[:-2].strip()) * 1024


def device_owners():
    sys.path.insert(0, str(ROOT / "models/autoports/google_gemma_4_31b_it/tools"))
    import benchmark_server

    return benchmark_server.device_owners()


def p_sources():
    admission_path = P / "admission.json"
    verify_file(admission_path, P_ADMISSION_SHA256, "P admission")
    admission = read(admission_path)
    assert admission.get("status") == "admitted"
    verify_file(P / "proposal.json", admission["proposal_sha256"], "P proposal")
    verify_file(P / "run_panel_after_w.py", admission["driver_sha256"], "P driver")
    inventory_path = P / "source-hashes.json"
    verify_file(inventory_path, P_SOURCE_HASHES_SHA256, "P frozen source inventory")
    inventory = read(inventory_path)
    admission_row = inventory["files"].get("admission")
    assert admission_row is not None, "P source inventory lacks expected admission row"
    assert (
        admission_row.get("path") == admission_path.relative_to(ROOT).as_posix()
    ), "P admission exception path changed"
    assert (
        admission_row.get("sha256") == P_ADMISSION_INVENTORY_PLACEHOLDER_SHA256
    ), "P admission exception is not the known frozen placeholder"
    for key, item in inventory["files"].items():
        if key == "admission":
            # Inventory was frozen before final admission; current bytes/status were independently verified above.
            continue
        verify_file(ROOT / item["path"], item["sha256"], "P source " + item["path"])
    return {
        "admission_sha256": sha(admission_path),
        "proposal_sha256": sha(P / "proposal.json"),
        "driver_sha256": sha(P / "run_panel_after_w.py"),
        "source_hashes_sha256": sha(P / "source-hashes.json"),
        "admission_inventory_exception": {
            "row_key": "admission",
            "stale_placeholder_sha256": P_ADMISSION_INVENTORY_PLACEHOLDER_SHA256,
            "verified_current_admission_sha256": P_ADMISSION_SHA256,
        },
    }


def e8192_sources():
    admission_path = E8192 / "admission.json"
    verify_file(admission_path, E8192_ADMISSION_SHA256, "suffix8192 admission")
    admission = read(admission_path)
    assert admission.get("status") == "admitted"
    proposal = read(E8192 / "proposal.json")
    verify_file(E8192 / "proposal.json", admission["proposal_sha256"], "suffix8192 proposal")
    verify_file(E8192 / "run_after_p.py", admission["driver_sha256"], "suffix8192 driver")
    for name, digest in proposal["candidate_hashes"].items():
        verify_file(E8192 / name, digest, "suffix8192 candidate " + name)
    provenance = read(E8192 / "provenance.json")
    for row in provenance["fixture_records"]:
        verify_file(ROOT / row["path"], row["sha256"], "suffix8192 fixture " + row["path"])
    return {
        "admission_sha256": sha(admission_path),
        "proposal_sha256": sha(E8192 / "proposal.json"),
        "driver_sha256": sha(E8192 / "run_after_p.py"),
    }


def p2_sources():
    admission_path = P2 / "admission.json"
    verify_file(admission_path, P2_ADMISSION_SHA256, "P2 admission")
    admission = read(admission_path)
    assert admission.get("status") == "admitted"
    proposal_path = P2 / "proposal.json"
    verify_file(proposal_path, admission.get("proposal_sha256", ""), "P2 proposal")
    driver_path = P2 / "run_after_p1_x.py"
    verify_file(driver_path, admission.get("driver_sha256", ""), "P2 driver")
    inventory_path = P2 / "source-hashes.json"
    verify_file(inventory_path, admission.get("source_inventory_sha256", ""), "P2 source inventory")
    inventory = read(inventory_path)
    for item in inventory["files"].values():
        verify_file(ROOT / item["path"], item["sha256"], "P2 source " + item["path"])
    proposal = read(proposal_path)
    assert proposal.get("driver_sha256") == sha(driver_path), "P2 proposal driver pin mismatch"
    assert proposal.get("source_inventory_sha256") == sha(inventory_path), "P2 proposal inventory pin mismatch"
    return {
        "admission_sha256": sha(admission_path),
        "proposal_sha256": sha(proposal_path),
        "driver_sha256": sha(driver_path),
        "source_inventory_sha256": sha(inventory_path),
    }


def validate_p2_terminal_records(queue, completion, launch, command, admission_sha256):
    assert queue.get("status") == "completed" and queue.get("exit_code") == 0, "P2 queue is not terminal-success"
    assert queue.get("candidate_admission_sha256") == admission_sha256, "P2 queue admission pin mismatch"
    assert (
        completion.get("exit_code") == 0 and completion.get("status") == "completed"
    ), "P2 completion is not successful"
    assert completion.get("candidate_admission_sha256") == admission_sha256, "P2 completion admission pin mismatch"
    assert completion.get("command") == command, "P2 completion command mismatch"
    assert launch.get("candidate_admission_sha256") == admission_sha256, "P2 launch admission pin mismatch"
    assert launch.get("command") == command, "P2 launch command mismatch"


def p2_terminal_and_forecast_gate():
    pins = p2_sources()
    queue_path, completion_path, launch_path = P2 / "queue.json", P2 / "completion.json", P2 / "launch.json"
    queue, completion, launch = read(queue_path), read(completion_path), read(launch_path)
    command = read(P2 / "command.json")["argv"]
    validate_p2_terminal_records(queue, completion, launch, command, P2_ADMISSION_SHA256)
    proposal = read(P2 / "proposal.json")
    forecast_path = P2 / "runtime-estimate.json"
    verify_file(forecast_path, proposal["forecast_source_sha256"], "P2 frozen forecast")
    forecast = read(forecast_path)
    assert launch.get("forecast") == forecast, "P2 launch forecast differs from frozen runtime estimate"
    bounds_obj = forecast["forecast_seconds"]
    bounds = [bounds_obj["low"], bounds_obj["high"]]
    acceptance_path = P2 / "forecast-miss-accepted.json"
    gate = forecast_gate(
        completion["elapsed_seconds"], bounds, completion_path, P2_ADMISSION_SHA256, acceptance_path, "P2"
    )
    return {
        "queue_sha256": sha(queue_path),
        "completion_sha256": sha(completion_path),
        "launch_sha256": sha(launch_path),
        **pins,
        "forecast_gate": gate,
    }


def forecast_gate(
    measured, forecast_range, completion_path, admission_sha256, acceptance_path, label, extra_binding=None
):
    assert len(forecast_range) == 2 and forecast_range[0] <= forecast_range[1], f"invalid {label} forecast range"
    low, high = forecast_range
    if low <= measured <= high:
        return {"status": "forecast_range_hit", "measured_seconds": measured, "forecast_range_seconds": forecast_range}
    assert acceptance_path.is_file(), f"{label} forecast miss requires a root hash-bound acceptance record"
    accepted = read(acceptance_path)
    expected = {
        "status": "root_accepted_forecast_miss",
        "reviewer": "root",
        "accepted": True,
        "completion_sha256": sha(completion_path),
        "measured_seconds": measured,
        "forecast_range_seconds": forecast_range,
        "admission_sha256": admission_sha256,
    }
    expected.update(extra_binding or {})
    for key, value in expected.items():
        assert accepted.get(key) == value, f"{label} forecast-miss acceptance mismatch: {key}"
    return {
        "status": "root_accepted_forecast_miss",
        "measured_seconds": measured,
        "forecast_range_seconds": forecast_range,
        "acceptance_record_sha256": sha(acceptance_path),
    }


def predecessor_forecast_gates():
    pcompletion_path, plaunch_path = P / "completion.json", P / "launch.json"
    pc, pl = read(pcompletion_path), read(plaunch_path)
    pf = pl.get("forecast", {})
    prange = pf.get("forecast_range_seconds")
    assert prange, "P launch lacks frozen forecast range"
    verify_file(ROOT / pf["forecast_path"], pf["forecast_sha256"], "P forecast source")
    pgate = forecast_gate(
        pc["elapsed_seconds"], prange, pcompletion_path, P_ADMISSION_SHA256, P / "forecast-miss-accepted.json", "P"
    )

    ef = read(E8192 / "forecast.json")
    egates = {}
    for mode in ("served", "full"):
        completion_path = E8192 / f"completion-{mode}.json"
        completion = read(completion_path)
        mode_forecast = ef["modes"][mode]
        egates[mode] = forecast_gate(
            completion["elapsed_seconds"],
            mode_forecast["range_s"],
            completion_path,
            E8192_ADMISSION_SHA256,
            E8192 / f"forecast-miss-accepted-{mode}.json",
            f"suffix8192 {mode}",
        )
    return {"P": pgate, "suffix8192": egates}


def wait_for_predecessors():
    deadline = time.monotonic() + 24 * 3600
    while True:
        if STOP_BEFORE.exists() or STOP_BETWEEN.exists():
            raise RuntimeError("STOP file set while waiting for predecessor closure")
        p_ready = (P / "queue.json").is_file() and (P / "completion.json").is_file()
        e_ready = (
            (E8192 / "queue.json").is_file()
            and (E8192 / "completion-served.json").is_file()
            and (E8192 / "completion-full.json").is_file()
        )
        p2_ready = (P2 / "queue.json").is_file() and (P2 / "completion.json").is_file()
        if p_ready and e_ready and p2_ready:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("P, suffix8192, and P2 did not reach terminal closure within 24 hours")
        time.sleep(10)

    pq, pc = read(P / "queue.json"), read(P / "completion.json")
    pproposal = read(P / "proposal.json")
    plaunch = read(P / "launch.json")
    assert pq.get("status") == "completed" and pq.get("exit_code") == 0, "P queue not terminal-success"
    assert pc.get("exit_code") == 0, "P completion failed"
    assert pc.get("command") == pproposal["execution"]["command"], "P completion command differs from proposal"
    assert pq.get("admission_sha256") == P_ADMISSION_SHA256
    assert pc.get("admission_sha256") == P_ADMISSION_SHA256 and plaunch.get("admission_sha256") == P_ADMISSION_SHA256
    ppins = p_sources()

    eq = read(E8192 / "queue.json")
    ec_s, ec_f = read(E8192 / "completion-served.json"), read(E8192 / "completion-full.json")
    ecommand = read(E8192 / "command.json")
    assert (
        eq.get("status") == "all_suffix8192_modes_completed" and eq.get("exit_code") == 0
    ), "suffix8192 queue not terminal-success"
    assert eq.get("candidate_admission_sha256") == E8192_ADMISSION_SHA256
    for mode, completion in (("served", ec_s), ("full", ec_f)):
        assert completion.get("exit_code") == 0, f"suffix8192 {mode} completion failed"
        assert (
            completion.get("command") == ecommand["modes"][mode]["command"]
        ), f"suffix8192 {mode} completion command mismatch"
        assert completion.get("candidate_admission_sha256") == E8192_ADMISSION_SHA256
    epins = e8192_sources()
    comparison_path = E8192 / "comparison-summary.json"
    comparison = read(comparison_path)
    assert comparison.get("status") == "passed", "suffix8192 matched-mode comparison missing or failed"
    assert eq.get("comparison_sha256") == sha(comparison_path), "suffix8192 queue comparison hash mismatch"
    gates = predecessor_forecast_gates()
    p_terminal = {
        "queue_sha256": sha(P / "queue.json"),
        "completion_sha256": sha(P / "completion.json"),
        "launch_sha256": sha(P / "launch.json"),
        **ppins,
        "forecast_gates": gates,
    }
    e_terminal = {
        "queue_sha256": sha(E8192 / "queue.json"),
        "served_completion_sha256": sha(E8192 / "completion-served.json"),
        "full_completion_sha256": sha(E8192 / "completion-full.json"),
        "served_launch_sha256": sha(E8192 / "launch-served.json"),
        "full_launch_sha256": sha(E8192 / "launch-full.json"),
        "comparison_sha256": sha(comparison_path),
        **epins,
        "forecast_gates": gates["suffix8192"],
    }
    p2_terminal = p2_terminal_and_forecast_gate()
    return p_terminal, e_terminal, p2_terminal


def check_runtime_preflight():
    hits = active_hf_processes()
    assert not hits, f"HF reference process is active: {hits}"
    owners = device_owners()
    assert not owners, f"device owners are active: {owners}"
    mem = available_memory_bytes()
    assert mem >= 100 * 1024**3, f"MemAvailable below100GiB: {mem} bytes"
    return {
        "active_hf_processes": hits,
        "device_owners": owners,
        "mem_available_bytes": mem,
        "required_mem_available_bytes": 100 * 1024**3,
    }


def verify_terminal_pin_map(current_pins, expected_pins, base, names, label):
    """Recheck captured returned pins; only hash a file when no captured pin exists."""
    for key, expected in expected_pins.items():
        if key in current_pins:
            current = current_pins[key]
        else:
            assert key in names, f"no fallback path for {label} pin {key}"
            current = sha(base / names[key])
        assert current == expected, f"{label} terminal pin changed: {key}"


def verify_predecessors(p_terminal, e_terminal, p2_terminal):
    assert predecessor_forecast_gates() == {
        "P": p_terminal["forecast_gates"]["P"],
        "suffix8192": e_terminal["forecast_gates"],
    }, "predecessor forecast status/acceptance changed before HF group"
    ppins = p_sources()
    epins = e8192_sources()
    p_names = {"queue_sha256": "queue.json", "completion_sha256": "completion.json", "launch_sha256": "launch.json"}
    e_names = {
        "queue_sha256": "queue.json",
        "served_completion_sha256": "completion-served.json",
        "full_completion_sha256": "completion-full.json",
        "served_launch_sha256": "launch-served.json",
        "full_launch_sha256": "launch-full.json",
        "comparison_sha256": "comparison-summary.json",
    }
    p_expected = {key: value for key, value in p_terminal.items() if key != "forecast_gates"}
    e_expected = {key: value for key, value in e_terminal.items() if key != "forecast_gates"}
    verify_terminal_pin_map(ppins, p_expected, P, p_names, "P")
    verify_terminal_pin_map(epins, e_expected, E8192, e_names, "suffix8192")
    current_p2 = p2_terminal_and_forecast_gate()
    assert current_p2 == p2_terminal, "P2 terminal, source, or forecast gate changed before HF group"


def load_progress(admission_sha, candidate_pins, resume):
    path = OUT / "progress.json"
    if path.exists():
        assert resume, "existing progress: use --resume; refusing implicit restart"
        state = read(path)
        assert state.get("candidate_admission_sha256") == admission_sha
        assert state.get("candidate_pins") == candidate_pins
        assert state.get("status") in {
            "ready",
            "paused_between_groups",
            "group_completed",
            "forecast_miss_pending_root",
            "failed",
        }
        return state
    assert not resume, "--resume requested without existing progress record"
    state = {
        "status": "ready",
        "candidate_admission_sha256": admission_sha,
        "candidate_pins": candidate_pins,
        "groups": [],
        "started_unix": time.time(),
    }
    write_json(path, state)
    return state


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def verify_group_outputs(group):
    outdir = Path(group["command"][group["command"].index("--output") + 1])
    jobs = read(group["manifest"])
    result_hashes = {}
    for job in jobs:
        result_path = outdir / (job["name"] + ".hf.private.json")
        assert result_path.is_file(), f"missing HF result {result_path}"
        result = read(result_path)
        assert result.get("status") == "complete", f"incomplete HF result: {result_path}"
        assert result.get("revision") == "842da3794eaa0b77d5f08bae87a17459d91ff475"
        assert result.get("prompt_tokens") == len(read(ROOT / job["case"])["prompt_token_ids"])
        assert len(result.get("teacher_forced_chosen", {}).get("positions", [])) == min(
            64, len(read(ROOT / job["case"])["continuations"]["chosen"])
        )
        trials = result.get("results", [])
        assert len(trials) == 1 and trials[0].get("seed") == group["seed_group"]
        assert trials[0].get("requested_tokens") == 512
        assert set(result.get("tt_comparisons", {})) == {str(Path(x)) for x in _compare_dirs(group["command"])}
        result_hashes[str(result_path)] = sha(result_path)
    return result_hashes


def _compare_dirs(command):
    result = []
    for i, arg in enumerate(command[:-1]):
        if arg == "--compare-dir":
            result.append(command[i + 1])
    return result


def execute_group(group, index, admission, candidate_pins, p_terminal, e_terminal, p2_terminal, state):
    if STOP_BEFORE.exists() or STOP_BETWEEN.exists():
        state["status"] = "paused_between_groups"
        state["pause_reason"] = "stop_file"
        write_json(OUT / "progress.json", state)
        return False
    current_admission = read(ADMISSION)
    assert current_admission == admission and current_admission.get("status") == "admitted"
    proposal, commands, _ = verify_inventory()
    for previous in state["groups"]:
        accepted_sha = previous.get("forecast_miss_acceptance_sha256")
        if accepted_sha:
            verify_file(
                OUT / f"forecast-miss-accepted-group-{previous['serial_order']:02d}.json",
                accepted_sha,
                f"group{previous['serial_order']} forecast acceptance",
            )
    for name, expected in candidate_pins.items():
        path = ROOT / name if name.startswith("models/") else OUT / name
        verify_file(path, expected, f"candidate pin {name}")
    verify_predecessors(p_terminal, e_terminal, p2_terminal)
    preflight = check_runtime_preflight()
    verify_predecessors(p_terminal, e_terminal, p2_terminal)
    outdir = Path(group["command"][group["command"].index("--output") + 1])
    forecast = read(FORECAST)
    forecast_rows = [
        row
        for row in forecast["groups"]
        if row["seed_group"] == group["seed_group"] and row["suffix_source"] == group["suffix_source"]
    ]
    assert len(forecast_rows) == 1, f"expected one forecast row for serial group {index}"
    forecast_row = forecast_rows[0]
    assert forecast_row["timeout_seconds"] == group["timeout_seconds"]
    launch_path = OUT / f"launch-group-{index:02d}.json"
    completion_path = OUT / f"completion-group-{index:02d}.json"
    if launch_path.exists() or completion_path.exists():
        raise RuntimeError(f"group {index} already has launch/completion artifacts; inspect, do not retry")
    assert not outdir.exists() or not any(outdir.iterdir()), f"refusing non-empty HF output directory {outdir}"
    outdir.mkdir(parents=True, exist_ok=True)
    launch = {
        "serial_order": index,
        "group": group,
        "command": group["command"],
        "start_unix": time.time(),
        "candidate_admission_sha256": sha(ADMISSION),
        "candidate_pins": candidate_pins,
        "p_terminal_hashes": p_terminal,
        "suffix8192_terminal_hashes": e_terminal,
        "p2_terminal_hashes": p2_terminal,
        "preflight": preflight,
        "forecast": forecast_row,
        "forecast_source_sha256": sha(FORECAST),
        "timeout_seconds": group["timeout_seconds"],
    }
    write_json(launch_path, launch)
    state["status"] = "group_running"
    state["active_group"] = index
    state["active_launch_sha256"] = sha(launch_path)
    state["groups"].append({"serial_order": index, "status": "running", "launch_sha256": sha(launch_path)})
    write_json(OUT / "progress.json", state)
    log_path = OUT / f"group-{index:02d}.log"
    try:
        with log_path.open("w") as log:
            result = subprocess.run(
                group["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=group["timeout_seconds"] + 30
            )
        completion = {
            **launch,
            "end_unix": time.time(),
            "elapsed_seconds": time.time() - launch["start_unix"],
            "exit_code": result.returncode,
            "log_sha256": sha(log_path),
        }
    except subprocess.TimeoutExpired as exc:
        completion = {
            **launch,
            "end_unix": time.time(),
            "elapsed_seconds": time.time() - launch["start_unix"],
            "exit_code": 124,
            "timeout": True,
            "log_sha256": sha(log_path),
        }
        write_json(completion_path, completion)
        state["status"] = "failed"
        state["failure"] = f"group {index} hit timeout; no automatic retry"
        write_json(OUT / "progress.json", state)
        raise RuntimeError(state["failure"]) from exc
    write_json(completion_path, completion)
    if result.returncode != 0:
        state["status"] = "failed"
        state["failure"] = f"group {index} exited {result.returncode}; no automatic retry"
        state["groups"][-1].update(status="failed", completion_sha256=sha(completion_path))
        write_json(OUT / "progress.json", state)
        raise RuntimeError(state["failure"])
    forecast_range = forecast_row["range_seconds"]
    elapsed = completion["elapsed_seconds"]
    in_range = forecast_range[0] <= elapsed <= forecast_range[1]
    completion["forecast_gate"] = {
        "status": "forecast_range_hit" if in_range else "root_acceptance_required_before_next_group",
        "measured_seconds": elapsed,
        "forecast_range_seconds": forecast_range,
        "acceptance_record": None if in_range else f"forecast-miss-accepted-group-{index:02d}.json",
    }
    write_json(completion_path, completion)
    result_hashes = verify_group_outputs(group)
    row_status = "completed" if in_range else "forecast_miss_pending_root"
    state["groups"][-1].update(status=row_status, completion_sha256=sha(completion_path), results=result_hashes)
    state["active_group"] = None
    state["active_launch_sha256"] = None
    state["last_completed_group"] = index
    state["status"] = row_status
    if not in_range:
        state["groups"][-1]["forecast_miss"] = completion["forecast_gate"]
    write_json(OUT / "progress.json", state)
    return True


def resolve_pending_forecast_misses(state, commands, admission_sha):
    for row in state["groups"]:
        if row.get("status") != "forecast_miss_pending_root":
            continue
        index = row["serial_order"]
        group = next(item for item in commands["groups"] if item["serial_order"] == index)
        forecast_rows = [
            item
            for item in read(FORECAST)["groups"]
            if item["seed_group"] == group["seed_group"] and item["suffix_source"] == group["suffix_source"]
        ]
        assert len(forecast_rows) == 1
        completion_path = OUT / f"completion-group-{index:02d}.json"
        completion = read(completion_path)
        accepted = forecast_gate(
            completion["elapsed_seconds"],
            forecast_rows[0]["range_seconds"],
            completion_path,
            admission_sha,
            OUT / f"forecast-miss-accepted-group-{index:02d}.json",
            f"HF group {index}",
            {"serial_order": index},
        )
        assert accepted["status"] == "root_accepted_forecast_miss"
        row["status"] = "completed"
        row["forecast_miss_acceptance_sha256"] = accepted["acceptance_record_sha256"]
        state["status"] = "group_completed"
        write_json(OUT / "progress.json", state)


def run(admission, proposal, commands, integrity, resume):
    candidate_pins = {
        "proposal.json": sha(PROPOSAL),
        "commands.json": sha(COMMANDS),
        "source-integrity.json": sha(INTEGRITY),
        "forecast.json": sha(FORECAST),
        RUNNER.relative_to(ROOT).as_posix(): sha(RUNNER),
    }
    for name, digest in proposal["manifest_sha256s"].items():
        candidate_pins[name] = digest
    admission_sha = sha(ADMISSION)
    state = load_progress(admission_sha, candidate_pins, resume)
    p_terminal, e_terminal, p2_terminal = wait_for_predecessors()
    state["predecessors_pinned"] = {"P": p_terminal, "suffix8192": e_terminal, "P2": p2_terminal}
    write_json(OUT / "progress.json", state)

    resolve_pending_forecast_misses(state, commands, admission_sha)
    completed_orders = {row["serial_order"] for row in state["groups"] if row.get("status") == "completed"}
    for group in sorted(commands["groups"], key=lambda x: x["serial_order"]):
        index = group["serial_order"]
        if index in completed_orders:
            # Verify the prior output and launch/completion records before trusting a resume checkpoint.
            row = next(x for x in state["groups"] if x["serial_order"] == index)
            verify_file(OUT / f"launch-group-{index:02d}.json", row["launch_sha256"], f"group{index} launch")
            verify_file(
                OUT / f"completion-group-{index:02d}.json", row["completion_sha256"], f"group{index} completion"
            )
            for path, digest in row["results"].items():
                verify_file(path, digest, f"group{index} result")
            if row.get("forecast_miss_acceptance_sha256"):
                verify_file(
                    OUT / f"forecast-miss-accepted-group-{index:02d}.json",
                    row["forecast_miss_acceptance_sha256"],
                    f"group{index} forecast acceptance",
                )
            continue
        if any(row.get("serial_order") == index and row.get("status") == "running" for row in state["groups"]):
            raise RuntimeError(
                f"group {index} has a launch without a successful checkpoint; inspect manually, no retry"
            )
        if not execute_group(group, index, admission, candidate_pins, p_terminal, e_terminal, p2_terminal, state):
            return
        if state.get("status") == "forecast_miss_pending_root":
            return
        if STOP_BEFORE.exists() or STOP_BETWEEN.exists():
            state["status"] = "paused_between_groups"
            state["pause_reason"] = "stop_file_after_completed_group"
            write_json(OUT / "progress.json", state)
            return
    state["status"] = "all_groups_completed"
    state["ended_unix"] = time.time()
    write_json(OUT / "progress.json", state)


def self_test():
    assert available_memory_bytes() > 0
    assert len(_compare_dirs(["x", "--compare-dir", "/a", "--compare-dir", "/b"])) == 2
    assert ROOT == Path("/home/mvasiljevic/tt-metal-gemma4-31b-pipeline")
    # Mocked terminal dictionaries contain captured admission/proposal pins; the absent
    # fallback paths must never be evaluated for keys already present in those dictionaries.
    p_terminal_mock = {"admission_sha256": "p-admission", "proposal_sha256": "p-proposal"}
    e_terminal_mock = {"admission_sha256": "e-admission", "proposal_sha256": "e-proposal"}
    for pins in (p_terminal_mock, e_terminal_mock):
        for key, expected in pins.items():
            assert terminal_pin(pins, key, OUT / "deliberately-missing-fallback") == expected
    p_pins = {
        "queue_sha256": "p-queue",
        "completion_sha256": "p-completion",
        "launch_sha256": "p-launch",
        "admission_sha256": "p-admission",
        "proposal_sha256": "p-proposal",
        "driver_sha256": "p-driver",
        "source_hashes_sha256": "p-inventory",
        "admission_inventory_exception": {"row_key": "admission", "stale_placeholder_sha256": "frozen-stale-row"},
    }
    p_fallback_names = {
        "queue_sha256": "queue.json",
        "completion_sha256": "completion.json",
        "launch_sha256": "launch.json",
    }
    verify_terminal_pin_map(p_pins, p_pins, OUT, p_fallback_names, "mock P")
    e_pins = {
        "queue_sha256": "e-queue",
        "served_completion_sha256": "e-served-completion",
        "full_completion_sha256": "e-full-completion",
        "served_launch_sha256": "e-served-launch",
        "full_launch_sha256": "e-full-launch",
        "comparison_sha256": "e-comparison",
        "admission_sha256": "e-admission",
        "proposal_sha256": "e-proposal",
        "driver_sha256": "e-driver",
    }
    e_fallback_names = {
        "queue_sha256": "queue.json",
        "served_completion_sha256": "completion-served.json",
        "full_completion_sha256": "completion-full.json",
        "served_launch_sha256": "launch-served.json",
        "full_launch_sha256": "launch-full.json",
        "comparison_sha256": "comparison-summary.json",
    }
    verify_terminal_pin_map(e_pins, e_pins, OUT, e_fallback_names, "mock suffix8192")
    q = {"status": "completed", "exit_code": 0, "candidate_admission_sha256": "p2"}
    c = {"status": "completed", "exit_code": 0, "candidate_admission_sha256": "p2", "command": ["diagnostic"]}
    l = {"candidate_admission_sha256": "p2", "command": ["diagnostic"]}
    validate_p2_terminal_records(q, c, l, ["diagnostic"], "p2")
    try:
        validate_p2_terminal_records(q, c, l, ["other"], "p2")
    except AssertionError:
        pass
    else:
        raise AssertionError("P2 command mismatch was accepted")
    print("host self-test passed; no HF model load, TTNN import, predecessor wait, or device query")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--execute", action="store_true", help="requires root admission and all live gates")
    parser.add_argument("--resume", action="store_true", help="resume only from a successful group-boundary checkpoint")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.validate_only:
        verify_inventory()
        print("candidate inventory valid; no predecessor access, model load, network access, or device query")
        return
    if not args.execute:
        print(
            "candidate-only inert: pass --validate-only for host checks; execution requires --execute and root admission"
        )
        return
    admission, proposal, commands, integrity = candidate_admission()
    run(admission, proposal, commands, integrity, args.resume)


if __name__ == "__main__":
    main()
