# SPDX-License-Identifier: Apache-2.0
"""Targeted HTTP residency/mixed-shape workload; separate from headline benchmarks."""

import argparse
import hashlib
import json
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def log_events(data):
    events = []
    decoder = json.JSONDecoder()
    for number, line in enumerate(data.decode(errors="replace").splitlines(), 1):
        start = line.find('{"event":')
        if start >= 0:
            try:
                event, _ = decoder.raw_decode(line[start:])
                events.append(dict(event, log_line=number))
            except (ValueError, TypeError):
                pass
    return events


def trace_summary(baseline, workload):
    signatures = {}
    missing = []
    preparation_timings = []
    for phase, events in (("baseline", baseline), ("workload", workload)):
        for event in events:
            if event.get("event") != "generator_trace_prepare":
                continue
            warmup = event.get("warmup_seconds", {})
            capture = event.get("capture_seconds", {})
            timings_available = all(key in warmup for key in ("prefill", "decode", "sample")) and all(
                key in capture for key in ("prefill", "decode", "sample", "buffered_sample")
            )
            if not timings_available:
                missing.append(f"{phase} trace preparation lacks per-phase inclusive warmup/capture timings.")
            preparation_timings.append(
                {
                    "phase": phase,
                    "log_line": event.get("log_line"),
                    "warmup_seconds_including_compilation": warmup or None,
                    "capture_seconds": capture or None,
                    "pure_compilation_seconds": None,
                    "pure_compilation_missing_reason": "Runtime event does not separate compilation from warmup execution.",
                    "timing_basis": event.get("timing_basis"),
                    "prefill_execution": event.get("prefill_execution"),
                    "programs_before": event.get("programs_before"),
                    "programs_warmed": event.get("programs_warmed"),
                    "programs_after_capture": event.get("programs_after_capture"),
                }
            )
            for kind, trace_id in event.get("traces", {}).items():
                signature = {
                    "kind": kind,
                    "batch": event.get("batch"),
                    "layers": event.get("layers"),
                    "cache_shapes": event.get("cache_shapes"),
                    "prefill_buckets": event.get("prefill_buckets"),
                }
                key = json.dumps(signature, sort_keys=True)
                row = signatures.setdefault(
                    key, dict(execution_signature=signature, baseline_captures=0, workload_captures=0, trace_ids=[])
                )
                row[f"{phase}_captures"] += 1
                row["trace_ids"].append(str(trace_id))
                row["preparation_seconds"] = event.get("preparation_seconds")
                row["capture_seconds"] = capture.get(kind)
                row["warmup_seconds_including_compilation"] = warmup.get(
                    "sample" if kind == "buffered_sample" else kind
                )
                row["timing_basis"] = event.get("timing_basis")
    if not signatures:
        missing.append("No generator_trace_prepare baseline or workload event; missing counts are not zero.")
    retirements = [
        event
        for event in workload
        if event.get("event") in ("generator_trace_release", "generator_trace_retire", "generator_trace_invalidate")
    ]
    sampling = [event for event in workload if event.get("event") == "sampling_mode"]
    chunks = [event for event in workload if event.get("event") == "generator_prefill_chunk"]
    prompt_tokens = [event for event in workload if event.get("event") == "generator_prefill_prompt_token"]
    partial_pages = [event for event in workload if event.get("event") == "generator_prefill_partial_page"]
    scheduler_prefill = [event for event in sampling if event.get("phase") == "prefill"]
    decoder_lines = [event["log_line"] for event in sampling if event.get("phase") == "decode"]
    continuation_lines = [
        event["log_line"] for event in scheduler_prefill if any(start > 0 for start in event.get("starts", []))
    ]
    interleaved = any(
        left < decode < right
        for left, right in zip(continuation_lines, continuation_lines[1:])
        for decode in decoder_lines
    )
    counts = Counter(f'{event.get("phase")}:{event.get("mode")}' for event in sampling)
    decode_steps = [event for event in sampling if event.get("phase") == "decode"]
    lifetimes_pass = (
        bool(signatures)
        and not retirements
        and all(row["baseline_captures"] + row["workload_captures"] == 1 for row in signatures.values())
    )
    return {
        "execution_signatures": list(signatures.values()),
        "preparation_timings": preparation_timings,
        "workload_retirements": retirements,
        "shutdown_releases": [event for event in retirements if event.get("reason") == "shutdown"],
        "trace_lifetime_pass": lifetimes_pass,
        "sampling_step_counts": dict(counts),
        "scheduler_prefill_steps": scheduler_prefill,
        "actual_generator_chunks": chunks,
        "actual_batched_prefill_groups": [
            event for event in workload if event.get("event") == "generator_batched_prefill"
        ],
        "actual_unaligned_prompt_positions": prompt_tokens,
        "actual_partial_pages": partial_pages,
        "decode_step_count": len(decoder_lines),
        "active_decode_row_counts": dict(Counter(event.get("active_rows") for event in decode_steps)),
        "page_only_reload_steps": sum(bool(event.get("reload_page_table")) for event in decode_steps),
        "full_input_reload_steps": sum(bool(event.get("reload_inputs")) for event in decode_steps),
        "unchanged_state_decode_steps": sum(
            not event.get("reload_inputs") and not event.get("reload_page_table") for event in decode_steps
        ),
        "memory_snapshots": [event for event in workload if event.get("event") == "serving_memory"],
        "decode_between_continuation_steps": interleaved,
        "missing_reasons": missing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=("synchronous", "async"), required=True)
    parser.add_argument(
        "--require-batched-prefill",
        action="store_true",
        help="Add short-row concurrent controls and require batched events for each physical bucket",
    )
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--revision", default="842da3794eaa0b77d5f08bae87a17459d91ff475")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    vocabulary_size = len(tokenizer)
    records, checks, errors = [], [], []
    baseline_log = args.server_log.read_bytes()
    log_offset = len(baseline_log)
    started = time.monotonic()
    wall_started = time.time()
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference and reference.get("execution_mode") == args.execution_mode:
        raise ValueError("Reference must use the other synchronous/async execution mode")
    prompts = {}

    def prompt(length, topic):
        key = (length, topic)
        if key not in prompts:
            base = tokenizer.encode("Explain " + topic + " using a concrete example. ", add_special_tokens=False)
            prompts[key] = (base * ((length + len(base) - 1) // len(base)))[:length]
        return prompts[key]

    def request(label, length, topic, temperature=0.0, output_tokens=128, first_token=None):
        prefix = prompt(length, topic)
        body = dict(
            model=args.model,
            prompt=prefix,
            max_tokens=output_tokens,
            temperature=temperature,
            top_k=32 if temperature else 1,
            top_p=0.95 if temperature else 1.0,
            seed=42,
            ignore_eos=True,
            stream=True,
            return_token_ids=True,
        )
        began = time.monotonic()
        events, token_ids, pieces = [], [], []
        req = urllib.request.Request(
            args.url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        finish_reason = None
        with urllib.request.urlopen(req, timeout=600) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    raise RuntimeError(f"{label}: {chunk['error']}")
                for choice in chunk.get("choices", []):
                    assert choice.get("index", 0) == 0
                    ids = choice.get("token_ids") or []
                    if ids:
                        events.append((time.monotonic() - began, len(ids)))
                        token_ids.extend(ids)
                        if first_token is not None:
                            first_token.set()
                    pieces.append(choice.get("text", ""))
                    finish_reason = choice.get("finish_reason") or finish_reason
        ended = time.monotonic()
        assert len(token_ids) == output_tokens, (label, len(token_ids), finish_reason)
        assert all(0 <= token < vocabulary_size for token in token_ids)
        gaps = [(right[0] - left[0]) * 1000 for left, right in zip(events, events[1:])]
        return dict(
            label=label,
            input_tokens=length,
            output_tokens=len(token_ids),
            prompt_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
            temperature=temperature,
            top_k=body["top_k"],
            top_p=body["top_p"],
            seed=42,
            token_ids=token_ids,
            text="".join(pieces),
            finish_reason=finish_reason,
            start_seconds=began - started,
            end_seconds=ended - started,
            first_token_seconds=began - started + events[0][0],
            ttft_ms=events[0][0] * 1000,
            elapsed_seconds=ended - began,
            token_events=events,
            streamed_chunk_gap_ms_p50=percentile(gaps, 0.5),
            streamed_chunk_gap_ms_p99=percentile(gaps, 0.99),
            max_decode_pause_ms=max(gaps, default=0),
        )

    def check(name, actual, expected):
        checks.append(dict(name=name, passed=actual == expected))

    shapes = [(129, "rainbows"), (130, "prime numbers"), (131, "plant growth")]
    try:
        # First encounters are included in metrics; no per-request warmup.
        for i, (length, topic) in enumerate(shapes):
            records.append(request(f"serial_{i}", length, topic))
        baseline = {r["input_tokens"]: r["token_ids"] for r in records}
        if reference:
            expected = {
                r["input_tokens"]: r["token_ids"] for r in reference["requests"] if r["label"].startswith("serial_")
            }
            check("sync_async_serial_control", baseline, expected)
        for wave in range(2):
            jobs = shapes if wave == 0 else list(reversed(shapes))
            long_length = 8193 + wave * 130
            prompt(long_length, "how tides work")
            anchor_started = threading.Event()
            with ThreadPoolExecutor(max_workers=4) as pool:
                length, topic = jobs[0]
                anchor = pool.submit(request, f"anchor{wave}", length, topic, 0.0, 384, anchor_started)
                if not anchor_started.wait(timeout=180):
                    raise RuntimeError("Anchor did not begin decoding before concurrent arrivals")
                # Long prefill arrives only after a live request has produced a token.
                futures = [anchor, pool.submit(request, f"long{wave}", long_length, "how tides work")]
                for i, (length, topic) in enumerate(jobs[1:]):
                    futures.append(pool.submit(request, f"wave{wave}_{i}", length, topic, 0.0, (32, 128)[i]))
                wave_records = [future.result() for future in futures]
                anchor_record, long_record = wave_records[:2]
                check(
                    f"wave{wave}_long_arrived_during_anchor_decode",
                    anchor_record["first_token_seconds"] <= long_record["start_seconds"] < anchor_record["end_seconds"],
                    True,
                )
                for record in wave_records:
                    if record["input_tokens"] in baseline:
                        count = min(128, record["output_tokens"])
                        check(
                            record["label"] + "_greedy_prefix",
                            record["token_ids"][:count],
                            baseline[record["input_tokens"]][:count],
                        )
                    records.append(record)
        if args.require_batched_prefill:
            short_shapes = [
                (length, f"length {length} case {replica} about rainbows")
                for length in (32, 64, 96, 127, 128)
                for replica in range(4)
            ]
            short_controls = {}
            for index, (length, topic) in enumerate(short_shapes):
                row = request(f"short_serial_{index}", length, topic, output_tokens=32)
                short_controls[length, topic] = row["token_ids"]
                records.append(row)
            if reference:
                expected_short = {
                    r["label"]: r["token_ids"] for r in reference["requests"] if r["label"].startswith("short_serial_")
                }
                actual_short = {r["label"]: r["token_ids"] for r in records if r["label"].startswith("short_serial_")}
                check("sync_async_short_serial_control", actual_short, expected_short)
            for wave in range(2):
                # Reset only while idle: prior solo controls must not turn these
                # short-prefill requests into prefix-cache hits.
                reset = urllib.request.Request(args.url + "/reset_prefix_cache", data=b"", method="POST")
                with urllib.request.urlopen(reset, timeout=30) as response:
                    if response.status != 200:
                        raise RuntimeError("Short-wave prefix cache reset failed")
                jobs = (
                    short_shapes
                    if wave == 0
                    else list(reversed([shape for index, shape in enumerate(short_shapes) if index % 4 < 2]))
                )
                anchor_started = threading.Event()
                with ThreadPoolExecutor(max_workers=len(jobs) + 2) as pool:
                    anchor = pool.submit(request, f"short_anchor{wave}", 131, "plant growth", 0.0, 384, anchor_started)
                    if not anchor_started.wait(timeout=180):
                        raise RuntimeError("Short-wave anchor did not begin decoding")
                    futures = [pool.submit(request, f"short_long{wave}", 8193 + wave * 130, "how tides work", 0.0, 128)]
                    futures.extend(
                        pool.submit(request, f"short_wave{wave}_{index}", length, topic, 0.0, 32)
                        for index, (length, topic) in enumerate(jobs)
                    )
                    rows = [future.result() for future in futures]
                    anchor_row = anchor.result()
                records.extend([anchor_row, *rows])
                check(
                    f"short_wave{wave}_arrived_during_decode",
                    all(
                        anchor_row["first_token_seconds"] <= row["start_seconds"] < anchor_row["end_seconds"]
                        for row in rows
                    ),
                    True,
                )
                for row, shape in zip(rows[1:], jobs):
                    check(row["label"] + "_solo_equal", row["token_ids"], short_controls[shape])
                if reference:
                    expected_wave = {
                        r["label"]: r["token_ids"]
                        for r in reference["requests"]
                        if r["label"].startswith(f"short_wave{wave}_")
                    }
                    check(f"sync_async_short_wave{wave}", {r["label"]: r["token_ids"] for r in rows[1:]}, expected_wave)
        for repeat in range(2):
            records.append(request(f"sampled_{repeat}", 129, "rainbows", 1.0))
        check("seeded_repeat", records[-1]["token_ids"], records[-2]["token_ids"])
        if reference:
            expected_sampled = next((r for r in reference["requests"] if r["label"] == "sampled_0"), None)
            if expected_sampled is not None:
                check("sync_async_seeded_control", records[-1]["token_ids"], expected_sampled["token_ids"])
        last = request("return_to_greedy", 129, "rainbows")
        check("sampling_switch_return_to_greedy", last["token_ids"], baseline[129])
        records.append(last)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        elapsed = time.monotonic() - started
        log_after = args.server_log.read_bytes()
        if len(log_after) < log_offset:
            errors.append("Server log truncated during workload; baseline cannot be correlated")
        evidence = trace_summary(log_events(baseline_log), log_events(log_after[log_offset:]))
        checks.append(dict(name="trace_lifetime", passed=evidence["trace_lifetime_pass"]))
        checks.append(
            dict(name="decode_interleaved_between_continuations", passed=evidence["decode_between_continuation_steps"])
        )
        if args.require_batched_prefill:
            groups = evidence["actual_batched_prefill_groups"]
            check("short_batched_physical_buckets", {e.get("physical_size") for e in groups}, {32, 64, 96, 128})
            check(
                "short_batched_groups_have_multiple_rows",
                bool(groups) and all(len(e.get("rows", [])) >= 2 for e in groups),
                True,
            )
        failed = bool(errors) or any(not item["passed"] for item in checks)
        report = dict(
            status="failed" if failed else "incomplete_evidence" if evidence["missing_reasons"] else "pass",
            scope="mixed-shape HTTP control, not headline performance",
            execution_mode=args.execution_mode,
            require_batched_prefill=args.require_batched_prefill,
            reference=str(args.reference) if args.reference else None,
            server_log=str(args.server_log),
            server_log_byte_range=[log_offset, len(log_after)],
            workload_start_unix_seconds=wall_started,
            completed_requests=len(records),
            elapsed_seconds=elapsed,
            completed_requests_per_second=len(records) / elapsed,
            ttft_ms={f"p{int(p*100)}": percentile([r["ttft_ms"] for r in records], p) for p in (0.5, 0.95, 0.99)},
            checks=checks,
            errors=errors,
            trace_evidence=evidence,
            requests=records,
            limitations=[
                "Stream event gaps are chunk-arrival pauses, not inferred device latency or per-token ITL.",
                "HTTP overlap plus interleaved server step events demonstrate concurrent traffic; no request IDs in adapter events.",
                "Execution mode is operator-declared; retain the actual server command as independent evidence.",
            ],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        raise AssertionError(f"Served-state checks failed; inspect {args.output}")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
