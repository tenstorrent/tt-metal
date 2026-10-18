"""Fail-closed audit of the same-server mixed-shape trace workload."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def correctness_basis(args, client):
    if client["status"] == "pass":
        return dict(scope="Exact original lifecycle comparisons", original_status="pass")
    # A native numerical difference needs a measured same-physical-chunk
    # reference. Preserve the original failure; never relabel its raw result.
    assert client["status"] == "failed" and args.canonical_reference is not None
    reference = json.loads(args.canonical_reference.read_text())
    assert reference["status"] == "verified"
    original = reference["original_cross_shape_result"]
    assert original["status"] == "failed" and Path(original["artifact"]).resolve() == args.client.resolve()
    assert original["differences"] == client["exact_greedy_differences"]
    assert len(reference["canonical_references"]) == 2
    assert all(row["exact_output_tokens"] == 96 for row in reference["canonical_references"])
    artifacts = {Path(path).resolve(): digest for path, digest in reference["artifacts"].items()}
    assert {args.client.resolve(), args.events.resolve(), args.runtime.resolve()} <= artifacts.keys()
    for path, digest in artifacts.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Changed reference artifact: {path}"
    return dict(
        scope=reference["scope"],
        original_status="failed",
        original_differences=original["differences"],
        canonical_reference=str(args.canonical_reference),
        canonical_reference_sha256=hashlib.sha256(args.canonical_reference.read_bytes()).hexdigest(),
        provenance_limit=reference["provenance_limit"],
    )


def main(args):
    events = [json.loads(line) for line in args.events.read_text().splitlines()]
    client = json.loads(args.client.read_text())
    runtime = json.loads(args.runtime.read_text())
    prepared = [e for e in events if e["event"] == "prepared"]
    correctness = correctness_basis(args, client)
    assert len(prepared) == 1
    baseline = prepared[0]
    workload_start = min(row["start"] for row in client["requests"])
    workload_end = max(row["end"] for row in client["requests"])
    assert baseline["time"] < workload_start
    assert runtime["batch"] == baseline["batch"]
    captures = [e for e in events if e["event"] == "capture"]
    counts = Counter(e["signature"] for e in captures)
    assert all(n == 1 for n in counts.values())
    assert all(e["time"] < baseline["time"] for e in captures)
    ids = {e["signature"]: e["trace_id"] for e in captures}
    preparations = {e["signature"]: e for e in events if e["event"] == "preparation"}
    assert set(ids) <= set(preparations)
    assert all(e["elapsed_s"] >= 0 and e["time"] < baseline["time"] for e in preparations.values())
    assert ids == baseline["traces"]
    forbidden = {"retirement", "retire", "invalidation", "eviction", "evict", "recapture"}
    assert not any(e["event"] in forbidden for e in events)
    assert not any(
        e["event"] in {"release", "shutdown_release"} and workload_start <= e["time"] <= workload_end for e in events
    )
    replays = [e for e in events if e["event"] == "replay"]
    assert all(ids[e["signature"]] == e["trace_id"] for e in replays)
    schedule, cursor, covered = None, None, []
    for e in events:
        if e["event"] == "prefill_schedule":
            assert schedule is None
            schedule, cursor = e, e["starts"].copy()
        elif e["event"] == "prefill_chunk":
            assert schedule is not None
            for i in range(len(cursor)):
                if e["active"][i]:
                    assert e["start_positions"][i] == cursor[i]
                    cursor[i] += e["chunk_length"]
                    assert cursor[i] <= schedule["ends"][i]
        elif e["event"] == "prefill_complete":
            assert cursor == schedule["ends"]
            covered.append(schedule)
            schedule, cursor = None, None
    assert schedule is None and covered
    assert any(any(start > 0 for start in e["starts"]) for e in covered)
    assert any(any(n % 32 for n in e["suffix_lengths"]) for e in covered)
    decode = [e for e in events if e["event"] == "decode_submit"]
    assert max(sum(e["active"]) for e in decode) > 1
    page_only = [e for e in decode if e["reload_page_table"]]
    assert len(page_only) >= 2 and all(not e["reload_inputs"] for e in page_only)
    assert {"greedy", "sample"} <= {e["sampling"] for e in decode}
    # A continuation occurs between decode submissions on this same generator.
    continuations = [e for e in covered if any(e["starts"])]
    assert any(
        any(d["time"] < e["time"] for d in decode) and any(d["time"] > e["time"] for d in decode) for e in continuations
    )
    memories = [e["memory"]["total_bytes_allocated_per_bank"] for e in events if e["event"] == "prefill_complete"]
    assert memories and len(set(memories)) == 1, memories
    mixed_requests = [r for r in client["requests"] if r["label"].startswith("mixed_")]
    mixed_start = min(r["start"] for r in mixed_requests)
    mixed_end = max(r["end"] for r in mixed_requests)
    mixed_decode = [e for e in decode if mixed_start <= e["time"] <= mixed_end]
    submission_gaps = [(b["time"] - a["time"]) * 1000 for a, b in zip(mixed_decode, mixed_decode[1:])]
    assert submission_gaps
    output = dict(
        status="pass",
        correctness=correctness,
        events=str(args.events),
        client=str(args.client),
        batch=baseline["batch"],
        context=runtime["context"],
        runtime=str(args.runtime),
        prepared_buckets=baseline["buckets"],
        preparation_events=list(preparations.values()),
        preparation_memory=[e for e in events if e["event"] == "preparation_memory"],
        execution_signatures=[
            dict(
                signature=e["signature"],
                trace_id=e["trace_id"],
                fixed_batch=baseline["batch"],
                context=runtime["context"],
                capture_s=e["elapsed_s"],
                preparation_s=preparations[e["signature"]]["elapsed_s"],
                preparation_scope=preparations[e["signature"]]["scope"],
                cumulative_trace_memory=e.get("memory", {}).get("trace"),
                captures=counts[e["signature"]],
                replays=sum(r["signature"] == e["signature"] for r in replays),
            )
            for e in captures
        ],
        workload_captures=0,
        retirements=0,
        invalidations=0,
        evictions=0,
        recaptures=0,
        shutdown_releases=sum(e["event"] == "shutdown_release" for e in events),
        prefill_schedules=covered,
        maximum_active_decode_rows=max(sum(e["active"]) for e in decode),
        page_only_refreshes=len(page_only),
        unchanged_decode_submissions=sum(not e["reload_inputs"] and not e["reload_page_table"] for e in decode),
        stable_allocated_bytes_per_bank=memories[0],
        ttft_ms=client["ttft_ms"],
        max_stream_pause_ms=client["max_stream_pause_ms"],
        mixed_decode_submission_gaps_ms=submission_gaps,
        maximum_mixed_decode_submission_gap_ms=max(submission_gaps),
        pause_timing_scope=(
            "Host monotonic decode submissions during mixed arrivals, including prefill-induced pauses; "
            "not device kernel timing or token ITL. HTTP text-chunk pauses additionally include hidden special tokens."
        ),
        completed_requests_per_s=client["completed_requests_per_s"],
        workload_request_shapes=[
            dict(prompt=r["prompt_length"], output=r["output_tokens"], sampling=r["sampling"])
            for r in client["requests"]
        ],
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k not in ("prefill_schedules", "workload_request_shapes")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-reference", type=Path)
    main(parser.parse_args())
