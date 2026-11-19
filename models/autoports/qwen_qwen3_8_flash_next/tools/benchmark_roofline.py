#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collect buffered full-phase host timing before an owned server stops."""

import argparse
import hashlib
import json
import re
import struct
import sys
import time
from pathlib import Path

from benchmark_accounting import account, validate_identity
from benchmark_server import get_text, idle


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    from benchmark_stage.subsets import digest as canonical_digest

    return canonical_digest(value)


def flush(path):
    path = Path(path)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("errors"):
            raise RuntimeError(f"Phase observer stopped with retained errors: {previous['errors']}")
    request = Path(str(path) + ".flush")
    request.touch()
    deadline = time.monotonic() + 30
    while request.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if request.exists() or not path.is_file():
        raise RuntimeError("Phase observer did not acknowledge export; inspect server before any reload")
    result = json.loads(path.read_text())
    if result["errors"] or result["pending_tickets"]:
        raise RuntimeError(f"Phase collector errors/pending: {result['errors']}, {result['pending_tickets']}")
    return result


def select_events(snapshot, perf):
    # CPython perf_counter and monotonic share the Linux CLOCK_MONOTONIC clock.
    # The vLLM client stores each request's perf_counter start, not process start.
    starts = perf["start_times"]
    ends = [s + f + sum(itls) for s, f, itls in zip(starts, perf["ttfts"], perf["itls"], strict=True)]
    low, high = int(min(starts) * 1e9), int(max(ends) * 1e9)
    requests = {}
    for event in snapshot["events"]:
        if event["phase"] != "prefill":
            continue
        for row in event["metadata"]["requests"]:
            if row["start"] == 0 and low <= row["first_submission_ns"] <= high:
                requests[row["request"]] = dict(row)
    if len(requests) != perf["completed"]:
        raise ValueError(f"Measured request mapping has {len(requests)} requests, expected {perf['completed']}")
    if len({row["prompt_sha256"] for row in requests.values()}) != len(requests):
        raise ValueError("Measured performance prompts do not have distinct first chunks")
    selected = []
    for event in snapshot["events"]:
        identities = {r.get("request") for r in event["metadata"]["requests"]}
        if identities & requests.keys():
            if not identities <= requests.keys():
                raise ValueError("Measured and unmeasured requests share a phase; cannot separate work")
            if "completion_observed_ns" not in event:
                raise ValueError("Missing asynchronous completion")
            selected.append(event)
    selected.sort(key=lambda row: row["end_order"])
    # Require a contiguous stream: no probe, warmup or unrelated phase can hide
    # between included completion boundaries.
    orders = [e["end_order"] for e in selected]
    if orders != list(range(orders[0], orders[-1] + 1)):
        raise ValueError("Measured phase stream contains unrelated work")
    for request, row in requests.items():
        chunks = [
            r for e in selected if e["phase"] == "prefill" for r in e["metadata"]["requests"] if r["request"] == request
        ]
        if max(r["end"] for r in chunks) != 4096:
            raise ValueError("Measured prompt is not exactly4096 tokens")
        cursor = 0
        for chunk in chunks:
            if chunk["start"] != cursor:
                raise ValueError("Missing or repeated scheduler prefill span")
            cursor = chunk["end"]
        row["logical_prompt_tokens"] = max(r["end"] for r in chunks)
        row["decode_positions"] = [
            r["position"]
            for e in selected
            if e["phase"] == "decode"
            for r in e["metadata"]["requests"]
            if r["request"] == request
        ]
        if len(row["decode_positions"]) < 127:
            raise ValueError("Missing decode steps for measured output")
        if row["decode_positions"] != list(range(4096, 4096 + len(row["decode_positions"]))):
            raise ValueError("Decode positions are not consecutive from the prompt boundary")
        row["queued_extra_decode_steps"] = len(row["decode_positions"]) - 127
    if any(e["phase"] == "decode" and "host_complete_ns" not in e for e in selected):
        raise ValueError("A selected decode lacks its existing host readback completion")
    return selected, list(requests.values()), dict(client_start_ns=low, client_final_token_ns=high)


def verify_client_prompts(events, requests, artifact_path):
    """Bind selected runtime requests to exact upstream client token sequences.

    Pinned TT InputBatch stores complete prompts in a CPU int32 tensor. Both
    prefill builders slice every submitted row through max(prompt_lens), and
    generator_vllm hashes that whole row's native-endian numpy byte sequence.
    Thus a row's hash width is the submission's maximum end, even when that
    row's scheduled chunk ends earlier. No padding is guessed: widths extending
    beyond a client's complete prompt are explicitly unsupported.
    """
    artifact_path = Path(artifact_path)
    client = json.loads(artifact_path.read_text())
    rows = client["requests"]
    if len(rows) != len(requests):
        raise ValueError("Client token artifact request count differs from measured requests")
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("Client token artifact request IDs are not unique")
    for row in rows:
        tokens = row["prompt_token_ids"]
        if len(tokens) != row["input_tokens"] or not tokens:
            raise ValueError("Client token artifact has an invalid prompt length")
        if any(type(token) is not int or not 0 <= token < 2**31 for token in tokens):
            raise ValueError("Client prompt token is outside the runtime int32 contract")
        expected = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        if row["prompt_token_ids_sha256"] != expected:
            raise ValueError("Client prompt token digest changed")
    initial = {}
    for event in events:
        if event["phase"] == "prefill":
            submitted = event["metadata"]["requests"]
            width = max(row["end"] for row in submitted)
            for row in submitted:
                if row["start"] == 0:
                    if row["request"] in initial:
                        raise ValueError("Runtime request has multiple initial prefill submissions")
                    initial[row["request"]] = width
    matched = set()
    for request in requests:
        width = initial[request["request"]]
        candidates = []
        for index, row in enumerate(rows):
            tokens = row["prompt_token_ids"]
            if width > len(tokens):
                raise ValueError("Cannot reconstruct initial buffer padding beyond the client prompt")
            encoded = struct.pack(f"={width}i", *tokens[:width])
            if hashlib.sha256(encoded).hexdigest() == request["prompt_sha256"]:
                candidates.append(index)
        if len(candidates) != 1 or candidates[0] in matched:
            raise ValueError("Runtime prompt hash lacks a unique unmatched client token sequence")
        index = candidates[0]
        matched.add(index)
        row = rows[index]
        if request["logical_prompt_tokens"] != row["input_tokens"]:
            raise ValueError("Runtime complete prompt length differs from matched client request")
        request.update(
            client_request_id=row["request_id"],
            client_prompt_token_ids_sha256=row["prompt_token_ids_sha256"],
            initial_buffer_tokens=width,
        )
    return dict(
        artifact=artifact_path.name,
        artifact_sha256=sha(artifact_path),
        matched_requests=len(matched),
        method=verify_client_prompts.__doc__,
        runtime_dtype="torch.int32",
        byte_order=sys.byteorder,
    )


def validate_native_fifo(events):
    """Confirm callback order agrees with native event order on one CQ0 mesh.

    MeshEvent's C++ stream formatter exposes its queue, root device and range.
    Event IDs are allocated monotonically by that queue; intervening unrecorded
    events may create gaps. All rows belong to one recorder/mesh lifetime.
    """
    if not events:
        raise ValueError("Cannot validate an empty native completion stream")
    pattern = re.compile(
        r"MeshEvent\(id=(\d+), device_id=(\d+), mesh_cq_id=(\d+), "
        r"device_range=(MeshCoordinateRange\(start=MeshCoordinate\(\[[0-9, ]+\]\), "
        r"end=MeshCoordinate\(\[[0-9, ]+\]\)\))\)"
    )
    native = []
    common = None
    previous_id = previous_order = -1
    for event in events:
        match = pattern.fullmatch(event.get("event_repr", ""))
        if match is None:
            raise ValueError("Missing or unrecognized native MeshEvent representation")
        event_id, device_id, cq_id = map(int, match.group(1, 2, 3))
        identity = dict(device_id=device_id, cq_id=cq_id, device_range=match.group(4))
        if cq_id != 0 or (common is not None and identity != common):
            raise ValueError("Native completion events do not share the same CQ0 mesh/range")
        order = event["end_order"]
        if type(order) is not int or order <= previous_order or event_id <= previous_id:
            raise ValueError("Native event IDs must strictly increase in completion end_order")
        common = identity
        native.append(dict(ticket=event["ticket"], end_order=order, native_event_id=event_id))
        previous_id, previous_order = event_id, order
    return dict(
        **common,
        events=native,
        checked_count=len(native),
        event_id_gaps_allowed=True,
        method=validate_native_fifo.__doc__,
        representation_source="tt_metal/distributed/mesh_event.cpp:51",
    )


def partition(events):
    """Exclusive FIFO completion intervals on the one CQ0 mesh timeline.

    Each event fences all prior queue work. A completion-to-completion interval
    belongs to the phase whose queued work it closes. This includes host gaps
    between launches; overlapping submission latencies are never summed. Host
    postprocessing overlaps following in-flight work and is already inside that
    elapsed timeline. Extend the final boundary to final token processing.
    """
    cursor = min(e["begin_ns"] for e in events)
    first = cursor
    totals = {"prefill": 0, "decode": 0}
    intervals = []
    previous_lower = cursor
    final_host = max(e.get("host_complete_ns", 0) for e in events)
    for index, event in enumerate(events):
        end = event["completion_observed_ns"]
        # Existing final readback gives an independent completion observation.
        # Use the polling upper bound consistently; expose both raw markers.
        if index == len(events) - 1:
            end = max(end, final_host)
        if end < cursor:
            raise ValueError("Completion order regressed")
        totals[event["phase"]] += end - cursor
        current_lower = event["completion_lower_ns"]
        if index == len(events) - 1:
            current_lower = max(current_lower, final_host)
        intervals.append(
            dict(
                ticket=event["ticket"],
                phase=event["phase"],
                start_ns=cursor,
                end_ns=end,
                duration_lower_ns=max(0, current_lower - cursor),
                duration_upper_ns=max(0, end - previous_lower),
            )
        )
        cursor = end
        previous_lower = current_lower
    if sum(totals.values()) != cursor - first or min(totals.values()) <= 0:
        raise ValueError("Phase partition does not cover the full elapsed timeline")
    return {key: value / 1e9 for key, value in totals.items()}, intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--action", required=True, choices=("check", "collect"))
    opts = parser.parse_args()
    run, batch = opts.run_dir.resolve(), str(opts.concurrency)
    server_path = run / f"perf-b{batch}-server.json"
    server = json.loads(server_path.read_text())
    get_text(server["base_url"] + "/health")
    idle(server["base_url"])
    phase_path = Path(server["phase_log"])
    runtime = json.loads(Path(str(phase_path) + ".identity.json").read_text())
    if runtime["layers"] != list(range(48)) or runtime["capacity"] != 262144 or runtime["slots"] != opts.concurrency:
        raise ValueError("Instrumentation runtime identity differs from configured full model")
    expected_file = Path(__file__).resolve().parents[1] / "tt/generator_vllm.py"
    if Path(runtime["module_file"]).resolve() != expected_file:
        raise ValueError("Server did not import the generated autoport")
    snapshot = flush(phase_path)
    if opts.action == "check":
        # Work accounting must validate the selected precision before accuracy.
        validate_identity(runtime)
        print(f"B{batch}: full48/C262144 phase observer and precision accounting ready; no added device waits")
        return
    perf_path = run / f"perf-b{batch}.json"
    perf = json.loads(perf_path.read_text())
    events, requests, window = select_events(snapshot, perf)
    prompt_mapping = verify_client_prompts(events, requests, Path(str(perf_path) + ".input-token-ids.json"))
    native_fifo = validate_native_fifo(events)
    seconds, intervals = partition(events)
    work = account(events, runtime)
    artifact = run / f"phase-accounting-b{batch}.json"
    evidence = dict(
        runtime=runtime,
        recorder=snapshot["identity"],
        events=events,
        requests=requests,
        client_window=window,
        client_prompt_mapping=prompt_mapping,
        native_fifo_validation=native_fifo,
        intervals=intervals,
        seconds=seconds,
        work=work,
        timing_method=partition.__doc__,
        max_completion_uncertainty_seconds=max(e["completion_uncertainty_ns"] for e in events) / 1e9,
        max_observer_poll_gap_seconds=snapshot["max_observer_poll_gap_ns"] / 1e9,
    )
    evidence["phase_duration_bounds_seconds"] = {
        phase: {
            bound: sum(row[f"duration_{bound}_ns"] for row in intervals if row["phase"] == phase) / 1e9
            for bound in ("lower", "upper")
        }
        for phase in ("prefill", "decode")
    }
    artifact.write_text(json.dumps(evidence, indent=2) + "\n")
    raw = run / f"phase-raw-b{batch}.json"
    raw.write_text(json.dumps(snapshot, separators=(",", ":")) + "\n")
    profile = dict(
        performance_sha256=sha(perf_path),
        server_identity_sha256=digest(server),
        requests=perf["completed"],
        input_tokens=perf["total_input_tokens"],
        output_tokens=perf["total_output_tokens"],
    )
    for phase in ("prefill", "decode"):
        profile[phase] = dict(
            work[phase],
            seconds=seconds[phase],
            timing_scope="full_phase_wall_time",
            timing_method="Exclusive ordered CQ0 completion intervals including intervening host gaps; nonblocking EventQuery observations; final host readback tail included",
            evidence=artifact.name,
            evidence_sha256=sha(artifact),
        )
    roofline = run / "roofline.json"
    saved = json.loads(roofline.read_text()) if roofline.exists() else {}
    saved[batch] = profile
    roofline.write_text(json.dumps(saved, indent=2) + "\n")
    print(
        f"B{batch}: {len(requests)} requests mapped; prefill {seconds['prefill']:.3f}s; decode {seconds['decode']:.3f}s; accounting retained"
    )


if __name__ == "__main__":
    main()
