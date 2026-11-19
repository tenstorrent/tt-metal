# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Collect buffered completion events from an idle, exclusively owned server."""

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path
from urllib.request import urlopen

from benchmark_stage.subsets import digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def snapshot(server):
    # Flush only after all HTTP requests complete; no device commands are issued.
    for _ in range(2):
        with urlopen(server["base_url"] + "/metrics", timeout=10) as response:
            metrics = response.read().decode()
        rows = [
            line
            for line in metrics.splitlines()
            if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))
        ]
        if len(rows) < 2 or any(float(line.rsplit(" ", 1)[1]) != 0 for line in rows):
            raise RuntimeError("Collector requires an idle server and no other client traffic")
        time.sleep(0.2)
    path = Path(server["phase_dir"]) / "snapshot.json"
    before = path.stat().st_mtime_ns
    os.kill(server["engine_pid"], signal.SIGUSR1)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.stat().st_mtime_ns != before:
            data = json.loads(path.read_text())
            if data["pid"] != server["engine_pid"] or data["pending"]:
                raise RuntimeError("Snapshot belongs to another engine or has unfinished outputs")
            return data
        time.sleep(0.05)
    raise RuntimeError("Phase recorder did not flush at idle")


def reconcile(data, perf):
    starts = perf["start_times"]
    latencies = (
        perf["latencies"]
        if "latencies" in perf
        else [ttft + sum(itls) for ttft, itls in zip(perf["ttfts"], perf["itls"])]
    )
    begin, end = min(starts), max(start + latency for start, latency in zip(starts, latencies))
    events = [e for e in data["events"] if begin <= e["time"] <= end]
    calls = {}
    for e in events:
        if e["kind"].endswith("_start"):
            calls[e["call"]] = dict(phase=e["kind"].split("_")[0], start=e["time"], chunks=e.get("chunks"))
        elif e["kind"] == "decode_accepted":
            if e["call"] not in calls:
                raise RuntimeError("Unmeasured decode crosses measured cohort boundary")
            calls[e["call"]].update(
                members=e["members"], trace_key=e["trace_key"], generation=e["generation"], is_tokens=e["is_tokens"]
            )
        elif e["kind"].endswith("_done"):
            if e["call"] not in calls:
                raise RuntimeError("Unmeasured completion contaminates cohort")
            calls[e["call"]]["end"] = e["time"]
    if not calls or any("end" not in call for call in calls.values()):
        raise RuntimeError("Incomplete measured phase call")
    chunks, decodes = [], []
    for call, row in calls.items():
        if row["phase"] == "prefill":
            chunks.extend(row["chunks"])
        else:
            if not row["is_tokens"]:
                raise RuntimeError("Performance phase accounting expects device sampling")
            decodes.append(dict(call=call, **row))
    lifetimes = {}
    for chunk in chunks:
        lifetimes.setdefault(chunk["request"], []).append(chunk)
    if len(lifetimes) != perf["completed"]:
        raise RuntimeError(f"Request lifetime mismatch: {len(lifetimes)} vs {perf['completed']}")
    for request, pieces in lifetimes.items():
        cursor = 0
        for piece in pieces:
            if piece["start"] != cursor:
                raise RuntimeError("Noncontiguous or replayed prefill in measured cohort")
            cursor = piece["end"]
        if cursor != 4096:
            raise RuntimeError("Measured lifetime is not the required 4096-token prompt")
    if sum(c["end"] - c["start"] for c in chunks) != perf["total_input_tokens"]:
        raise RuntimeError("Input work differs from measured workload")
    positions = {request: [] for request in lifetimes}
    for row in decodes:
        for member in row["members"]:
            if member["request"] not in positions:
                raise RuntimeError("Decode batch mixes measured and unmeasured requests")
            positions[member["request"]].append(member["position"])
    for request, values in positions.items():
        if values != list(range(4096, 4096 + len(values))) or not 127 <= len(values) <= 129:
            raise RuntimeError(f"Invalid decode positions/count for {request}: {len(values)}")
    # Union same-phase async intervals. Cross-phase overlap cannot be attributed
    # from these hooks and is rejected. Gaps go to the next actual phase.
    intervals = []
    for row in sorted(calls.values(), key=lambda r: (r["start"], r["end"])):
        if intervals and row["start"] < intervals[-1]["end"]:
            if row["phase"] != intervals[-1]["phase"]:
                raise RuntimeError("Cross-phase overlap needs stronger completion evidence")
            intervals[-1]["end"] = max(intervals[-1]["end"], row["end"])
        else:
            intervals.append(dict(phase=row["phase"], start=row["start"], end=row["end"]))
    seconds = dict(prefill=0.0, decode=0.0)
    cursor = intervals[0]["start"]
    for row in intervals:
        row["host_gap_before"] = row["start"] - cursor
        seconds[row["phase"]] += row["end"] - cursor
        cursor = row["end"]
    return dict(
        events=events,
        calls=calls,
        chunks=chunks,
        decodes=decodes,
        request_lifetimes=lifetimes,
        decode_positions=positions,
        intervals=intervals,
        seconds=seconds,
        measured_http_envelope=[begin, end],
        model_phase_envelope=[intervals[0]["start"], intervals[-1]["end"]],
        outside_phase_envelope_seconds=dict(lead=intervals[0]["start"] - begin, tail=end - intervals[-1]["end"]),
        cohort_method="Exclusive owned server; all requests complete; client monotonic send/completion window excludes initial probe and warmups; every contiguous 4096-token slot lifetime and all decode members reconciled. No other traffic permitted.",
    )


def main():
    from benchmark_work import decode_work, peak_rates, prefill_work

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--action", choices=("check", "collect"), required=True)
    args = parser.parse_args()
    directory, b = args.run_dir.resolve(), args.concurrency
    server = json.loads((directory / f"perf-b{b}-server.json").read_text())
    data = snapshot(server)
    assert server["layer_count"] == 40 and server["precision"]["config_id"] == "baseline_bfp8"
    grid = [int(x) for x in server.get("worker_grid", "11-10").split("-")]
    assert grid == [11, 10], "Re-audit observed hardware geometry"
    peaks = peak_rates(worker_cores=grid[0] * grid[1])
    if args.action == "check":
        write(
            directory / f"roofline-b{b}-ready.json",
            dict(
                pid=data["pid"],
                instance=data["instance"],
                peaks=peaks,
                buffered_events=len(data["events"]),
                collector_sha256=sha(Path(__file__)),
            ),
        )
        print("READY: buffered host completion recorder and accounting inputs")
        return
    perf_path = directory / f"perf-b{b}.json"
    perf = json.loads(perf_path.read_text())
    evidence = reconcile(data, perf)
    prefill = prefill_work(evidence["chunks"])
    peaks = peak_rates(prefill, worker_cores=grid[0] * grid[1])
    decode = [decode_work(row["members"], {"key": row["trace_key"], "server_slots": b}) for row in evidence["decodes"]]
    evidence.update(
        prefill_accounting=prefill,
        decode_accounting=decode,
        peaks=peaks,
        performance_sha256=sha(perf_path),
        server_identity_sha256=digest(server),
        instance=data["instance"],
        work_source_sha256=sha(Path(__file__).with_name("benchmark_work.py")),
    )
    # Retain all detailed events/accounting as bounded JSON pieces; keep the
    # required evidence index small enough for the repository artifact limit.
    detail_files = {}
    for field in ("events", "calls", "chunks", "decodes", "decode_accounting", "request_lifetimes", "decode_positions"):
        value = evidence.pop(field)
        items = list(value.items()) if isinstance(value, dict) else value
        pieces, current, size = [], [], 0
        for item in items:
            encoded_size = len(json.dumps(item).encode())
            if current and size + encoded_size > 150000:
                pieces.append(current)
                current, size = [], 0
            current.append(item)
            size += encoded_size
        if current:
            pieces.append(current)
        detail_files[field] = []
        for index, piece in enumerate(pieces):
            path = directory / f"roofline-b{b}-{field}-{index:03}.json"
            write(path, dict(piece) if isinstance(value, dict) else piece)
            detail_files[field].append(dict(path=path.name, sha256=sha(path), rows=len(piece)))
    evidence["detail_files"] = detail_files
    artifact = directory / f"roofline-b{b}-evidence.json"
    write(artifact, evidence)
    common = dict(
        timing_scope="full_phase_wall_time",
        timing_method="Union of accepted submit-to-existing-host-completion intervals on CLOCK_MONOTONIC; queued decode overlaps counted once; intervening host gaps assigned to next phase; cross-phase overlaps rejected; no added device synchronization",
        work_method="Source-derived useful-work model; see per-operation terms, MoE routing/reuse assumptions and bounds in linked evidence. Estimated bytes, not hardware counters.",
        peak_source=peaks["source"],
        evidence=artifact.name,
        evidence_sha256=sha(artifact),
    )
    row = dict(
        performance_sha256=sha(perf_path),
        server_identity_sha256=digest(server),
        requests=perf["completed"],
        input_tokens=perf["total_input_tokens"],
        output_tokens=perf["total_output_tokens"],
        prefill=dict(
            common,
            flops=prefill["flops"],
            seconds=evidence["seconds"]["prefill"],
            peak_flops_per_second=peaks["prefill_flops_per_second"],
        ),
        decode=dict(
            common,
            dram_bytes=sum(r["dram_bytes"] for r in decode),
            seconds=evidence["seconds"]["decode"],
            peak_dram_bytes_per_second=peaks["dram_bytes_per_second"],
        ),
    )
    path = directory / "roofline.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    result[str(b)] = row
    write(path, result)
    print(f"COLLECTED b={b}, requests={row['requests']}, full-phase seconds={evidence['seconds']}")


if __name__ == "__main__":
    main()
