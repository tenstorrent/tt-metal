# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bind full host completion phases and source-derived work to vLLM results."""

import argparse
import hashlib
import json
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

from benchmark_control import query
from benchmark_phases import reduce_phases
from benchmark_work import WorkAccounting


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def export_idle(server):
    deadline = time.monotonic() + 30
    while True:
        data = query(server["phase_control"], "export")
        if data["errors"]:
            raise ValueError("Observer errors: " + repr(data["errors"]))
        if not data["pending"]:
            return data
        if time.monotonic() >= deadline:
            raise TimeoutError("Existing async submissions did not naturally complete: " + repr(data["pending"]))
        time.sleep(0.1)


def decode_execution_counts(events, measured_ids):
    """Infer model executions from the immutable, complete observed dispatch stream.

    Initial prefill is a required anchor: configure_sampling releases any old
    trace. A new logical batch then warms once before recording and replaying.
    Trace recording bypasses the device issue queue; it is not a third pass.
    Scope is this non-DP, compact-slot, unpenalized greedy performance profile.
    """
    measured = set(measured_ids)
    bound_batch = None
    anchored = False
    result = {}
    for event in sorted((e for e in events if e["event"] == "dispatch"), key=lambda e: e["timestamp_ns"]):
        ids = event["request_ids"]
        if event["phase"] == "prefill":
            bound_batch = None
            anchored = True
            continue
        if not event["device_sampling"]:
            if set(ids) & measured:
                raise ValueError("Performance request fell back to host sampling")
            # The next device-sampling prefill must establish a new anchor.
            anchored = False
            bound_batch = None
            continue
        batch = event["batch_slots"]
        if batch != len(ids) or len(event["positions"]) != batch:
            raise ValueError("Cannot infer warm work for noncompact active slots")
        if set(ids) & measured:
            if not anchored:
                raise ValueError("Decode warm accounting lacks a preceding observed prefill anchor")
            result[event["submission_id"]] = 2 if bound_batch != batch else 1
        bound_batch = batch
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 32), required=True)
    parser.add_argument("--action", choices=("check", "collect"), required=True)
    parser.add_argument(
        "--events-file", type=Path, help="Reaccount an unchanged pre-stop export without querying a server"
    )
    args = parser.parse_args()
    if args.events_file and args.action != "collect":
        parser.error("--events-file is collect-only")
    root, n = args.run_dir, args.concurrency
    server_path = root / f"perf-b{n}-server.json"
    server = json.loads(server_path.read_text())
    work = WorkAccounting()
    saved = json.loads(args.events_file.read_text()) if args.events_file else None
    if saved is not None:
        configuration_path = root / server["configuration_evidence"]
        if hashlib.sha256(configuration_path.read_bytes()).hexdigest() != server["configuration_evidence_sha256"]:
            raise ValueError("Recorded configuration hash mismatch")
        configuration = json.loads(configuration_path.read_text())
        if saved["identity"] != configuration["phase_observer"] or saved["pending"] or saved["errors"]:
            raise ValueError("Saved export identity, completion or observer errors mismatch")
    initial = saved if saved is not None else query(server["phase_control"])
    identity = initial["identity"]
    if identity["layer_count"] != 30 or identity["max_num_seqs"] != n or identity["mesh"] != [1, 4]:
        raise ValueError("Running phase collector identity mismatch")
    if args.action == "check":
        # A small normal request proves both existing completion paths emit; it
        # is deliberately a different ID family from benchmark warmups/results.
        request_id = f"gemma4-check-b{n}-" + uuid.uuid4().hex
        body = dict(model=server["model"], prompt=[42] * 32, max_tokens=2, temperature=0, ignore_eos=True, stream=False)
        request = Request(
            server["base_url"] + "/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        )
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
        data = export_idle(server)
        records = [r for r in data["events"] if any(request_id in rid for rid in r["request_ids"])]
        phases = {r["phase"] for r in records if r["event"] == "completion"}
        if phases != {"prefill", "decode"} or result["usage"]["completion_tokens"] != 2:
            raise ValueError("Readiness request did not prove both actual completion phases")
        readiness = {
            "identity": identity,
            "events": records,
            "usage": result["usage"],
            "work_source_sha256": work.source_sha256,
        }
        if n == 32:
            accuracy_id = "gemma4-check-accuracy-" + uuid.uuid4().hex
            accuracy_body = dict(
                model=server["model"],
                messages=[{"role": "user", "content": "What is two plus two?"}],
                max_tokens=2,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                logprobs=True,
                top_logprobs=0,
                stream=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            accuracy_request = Request(
                server["base_url"] + "/v1/chat/completions",
                data=json.dumps(accuracy_body).encode(),
                headers={"Content-Type": "application/json", "X-Request-Id": accuracy_id},
            )
            with urlopen(accuracy_request, timeout=180) as response:
                accuracy_response = json.load(response)
            accuracy_data = export_idle(server)
            accuracy_states = {rid: state for rid, state in accuracy_data["requests"].items() if accuracy_id in rid}
            accuracy_events = [
                event for event in accuracy_data["events"] if any(accuracy_id in rid for rid in event["request_ids"])
            ]
            readiness["accuracy_host_route_probe"] = {
                "request": accuracy_body,
                "response": accuracy_response,
                "states": accuracy_states,
                "events": accuracy_events,
                "scope": "Two-token transport/sampling-route readiness probe; not benchmark accuracy.",
            }
            # Preserve the response before rejecting an integration failure.
            (root / f"phase-readiness-b{n}.json").write_text(json.dumps(readiness, indent=2) + "\n")
            if len(accuracy_states) != 1 or any(
                state["top_k"] != 64 or state["temperature"] != 1.0 for state in accuracy_states.values()
            ):
                raise ValueError("Accuracy readiness did not retain exact top-64 temperature-1 sampling")
            accuracy_dispatch = [event for event in accuracy_events if event["event"] == "dispatch"]
            if not accuracy_dispatch or any(event["device_sampling"] for event in accuracy_dispatch):
                raise ValueError("Accuracy readiness did not use exact host sampling route")
            if not any(event["event"] == "completion" for event in accuracy_events):
                raise ValueError("Accuracy readiness completion was not observed")
        (root / f"phase-readiness-b{n}.json").write_text(json.dumps(readiness, indent=2) + "\n")
        print(f"Collector ready for{n} slots; both normal completion paths observed")
        return
    raw_path = root / f"perf-b{n}.json"
    raw = json.loads(raw_path.read_text())
    request_map = json.loads((root / f"perf-b{n}-request-map.json").read_text())
    prefix = request_map["request_id_prefix"]
    expression = re.compile(re.escape(prefix) + r"(\d+)-0(?:_|-|$)")
    data = saved if saved is not None else export_idle(server)
    # Retain all raw observer records for explicit warmup/probe exclusion audit.
    raw_events_path = root / f"phase-events-b{n}.json"
    if saved is None:
        raw_events_path.write_text(json.dumps(data, indent=2) + "\n")
    elif args.events_file.resolve() != raw_events_path.resolve():
        raise ValueError("Offline reaccounting must use the original profile export path")
    cohort = {}
    for rid, state in data["requests"].items():
        if match := expression.search(rid):
            index = int(match.group(1))
            if index in cohort:
                raise ValueError("Measured request index appeared more than once")
            cohort[index] = rid
            if state["prompt_tokens"] != 4096 or state["max_tokens"] != 128 or state["temperature"] != 0:
                raise ValueError("Observed request violates performance workload")
    expected = int(raw["completed"])
    if set(cohort) != set(range(expected)) or expected != max(8, n * 3):
        raise ValueError(f"Measured request mapping incomplete: {len(cohort)}/{expected}; inspect {raw_events_path}")
    ids = list(cohort.values())
    if len({data["requests"][rid]["prompt_sha256"] for rid in ids}) != expected:
        raise ValueError("Performance prompts were not distinct")
    timeline = reduce_phases(data["events"], ids)
    dispatch = [
        event for event in data["events"] if event["event"] == "dispatch" and set(event["request_ids"]) <= set(ids)
    ]
    execution_counts = decode_execution_counts(data["events"], ids)
    counts = Counter()
    sums = {"prefill": 0, "decode": 0}
    details = []
    for event in dispatch:
        if not event["device_sampling"]:
            raise ValueError("Performance request fell back to host sampling")
        counts.update(event["request_ids"])
        if event["phase"] == "prefill":
            calculation = work.prefill(event["prompt_lens"])
            sums["prefill"] += calculation["useful_flops"]
        else:
            calculation = work.decode(event["positions"], batch_slots=event["batch_slots"])
            executions = execution_counts[event["submission_id"]]
            calculation["dram_bytes_per_execution"] = calculation["dram_bytes"]
            calculation["model_executions"] = executions
            calculation["warm_model_executions"] = executions - 1
            calculation["terms"] = {key: value * executions for key, value in calculation["terms"].items()}
            calculation["dram_bytes"] *= executions
            sums["decode"] += calculation["dram_bytes"]
        details.append({"submission_id": event["submission_id"], **calculation})
    if any(counts[rid] < 128 for rid in ids):
        raise ValueError("Missing output-step completion evidence")
    if raw["total_input_tokens"] != expected * 4096 or raw["total_output_tokens"] != expected * 128:
        raise ValueError("Actual token totals differ from4096/128 workload")
    if timeline["first_dispatch_ns"] / 1e9 < min(raw["start_times"]):
        raise ValueError("Measured observer span precedes client requests")
    peaks = work.peaks()
    evidence = root / f"phase-accounting-b{n}.json"
    evidence.write_text(
        json.dumps(
            {
                "timeline": timeline,
                "offline_reaccounting": saved is not None,
                "original_export_time_ns": data["export_time_ns"],
                "work_by_submission": details,
                "output_steps_per_request": dict(counts),
                "client_request_map": request_map,
                "raw_events_file": raw_events_path.name,
                "raw_events_sha256": hashlib.sha256(raw_events_path.read_bytes()).hexdigest(),
                "source_sha256": work.source_sha256,
                "warm_execution_inference": "Observed prefill releases model trace; first following decode and each logical-batch-size change warms one model pass before recording/replaying. Compact DP1 greedy rows only; capture records to host bypass buffer. Sampler setup traffic remains approximate.",
                "peaks": peaks,
                "assumptions_file": "models/autoports/google_gemma_4_26b_a4b_it/doc/benchmark/WORK_ACCOUNTING.md",
            },
            indent=2,
        )
        + "\n"
    )
    row = {
        "performance_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "server_identity_sha256": digest(server),
        "requests": expected,
        "input_tokens": raw["total_input_tokens"],
        "output_tokens": raw["total_output_tokens"],
    }
    for phase, key, peak in [
        ("prefill", "flops", "peak_flops_per_second"),
        ("decode", "dram_bytes", "peak_dram_bytes_per_second"),
    ]:
        row[phase] = {
            key: sums[phase],
            "seconds": timeline["phase_seconds"][phase],
            peak: peaks["peak_flops_per_s" if phase == "prefill" else "peak_dram_bytes_per_s"],
            "timing_scope": "full_phase_wall_time",
            "timing_method": "Full normal dispatch through actual existing async completion/output construction; same-phase overlap counted once, gaps retained; transition gaps assigned to following phase. No added device waits.",
            "work_method": "Source-derived full30-layer TP4/EP4 actual invocation shapes and selected stored precision; active MoE weight reads per serial row, shared head per invocation; material activation estimates detailed in WORK_ACCOUNTING.md.",
            "peak_source": peaks["reference"] + "; " + peaks["peak_basis"],
            "evidence": evidence.name,
            "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        }
    path = root / "roofline.json"
    all_rows = json.loads(path.read_text()) if path.exists() else {}
    all_rows[str(n)] = row
    path.write_text(json.dumps(all_rows, indent=2) + "\n")
    print(json.dumps({"profile": n, "phase_seconds": timeline["phase_seconds"], "work": sums}))


if __name__ == "__main__":
    main()
