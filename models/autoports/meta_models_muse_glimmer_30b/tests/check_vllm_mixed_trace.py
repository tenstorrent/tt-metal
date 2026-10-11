# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check mixed logical shapes against one already prepared, live vLLM server.

This client never imports TTNN, starts a server, resets counters, or warms a
signature separately. Run alone against the server so event attribution is
unambiguous. The output is trace/lifecycle evidence, not a quality verdict or
the primary serving benchmark. Inspect the separate server log for the trace
allocation-safety gate; this script cannot infer worker environment variables.
"""

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL = "meta-models/Muse-Glimmer-30B"
REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
SNAPSHOT = Path("/mnt/models/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots") / REVISION


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def parse_events(data):
    if data and not data.endswith(b"\n"):
        raise ValueError("Event file ends in a partial JSONL record; retry when the server is idle")
    return [dict(json.loads(line), source_line=i + 1) for i, line in enumerate(data.splitlines()) if line]


def values(value):
    # Older scheduler events serialized NumPy ends with json default=str.
    return [int(x) for x in re.findall(r"-?\d+", value)] if isinstance(value, str) else list(value)


def key_id(key):
    return json.dumps(key, separators=(",", ":"))


def execution_signature(key):
    if isinstance(key, list):
        return dict(kind=key[0], physical_rows=key[1], start_alignment=key[2] if len(key) > 2 else 0)
    return dict(kind=key, physical_rows="fixed generator batch")


def last_counters(events):
    return next((event["counters"] for event in reversed(events) if "counters" in event), {})


def percentiles(items):
    data = sorted(x for x in items if x is not None and math.isfinite(x))
    if not data:
        return dict(count=0, mean=None, p50=None, p95=None, p99=None, maximum=None)

    def percentile(p):
        offset = (len(data) - 1) * p
        lo, hi = math.floor(offset), math.ceil(offset)
        return data[lo] + (data[hi] - data[lo]) * (offset - lo)

    return dict(
        count=len(data),
        mean=sum(data) / len(data),
        p50=percentile(0.50),
        p95=percentile(0.95),
        p99=percentile(0.99),
        maximum=data[-1],
    )


def make_prompt(tokenizer, name, length, topic):
    """Vary only user content; never trim an HF chat envelope to hit a length."""

    def render(padding):
        content = (
            f"Explain {topic} in English using a clear numbered list. "
            f"Stay on the topic of {topic}. Background notes:" + " detail" * padding + "\nNow give the explanation."
        )
        messages = [{"role": "user", "content": content}]
        if tokenizer.chat_template:
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        else:
            ids = tokenizer.encode(content, add_special_tokens=True)
        return messages, list(ids)

    # Repeated ' detail' is normally one token. Search on actual rendered
    # lengths and fail instead of truncating or inventing special tokens.
    lo, hi = 0, length * 2
    while lo <= hi:
        mid = (lo + hi) // 2
        messages, ids = render(mid)
        if len(ids) == length:
            return dict(
                name=name,
                requested_length=length,
                messages=messages,
                prompt_token_ids=ids,
                rendered_prompt=tokenizer.decode(ids, skip_special_tokens=False),
            )
        if len(ids) < length:
            lo = mid + 1
        else:
            hi = mid - 1
    raise ValueError(f"Cannot render an intact HF prompt with exactly {length} tokens")


def baseline_check(events):
    starts = [i for i, event in enumerate(events) if event.get("event") == "adapter_loaded"]
    if not starts:
        raise ValueError("Existing events must identify the running adapter before workload submission")
    active = events[starts[-1] :]
    prepared = [event for event in active if event.get("event") == "prepared"]
    captures = [event for event in active if event.get("event") == "capture"]
    if len(prepared) != 1 or not captures:
        raise ValueError("The running generator must already have one complete preparation and captured traces")
    if active[0].get("revision") != REVISION:
        raise ValueError("The live adapter does not report the required checkpoint revision")
    if any(event.get("event") == "teardown_release" for event in active):
        raise ValueError("The latest recorded generator has already shut down")
    counts = Counter(key_id(event["key"]) for event in captures)
    if any(count != 1 for count in counts.values()):
        raise ValueError(f"Baseline already recaptured an execution signature: {dict(counts)}")
    kinds = {execution_signature(event["key"])["kind"] for event in captures}
    if not {"prefill", "head", "decode", "sample", "argmax"} <= kinds:
        raise ValueError(f"Baseline lacks prepared prefill/decode/sampling signatures: {kinds}")
    expected_keys = {key_id([kind, size]) for kind in ("prefill", "head") for size in prepared[0]["buckets"]}
    expected_keys.update(key_id(["prefill", 32, alignment]) for alignment in (32, 64, 96))
    expected_keys.update(key_id(kind) for kind in ("decode", "sample", "argmax", "admission_save", "admission_restore"))
    if expected_keys - set(counts):
        raise ValueError(
            f"Generator-wide preparation has not captured all signatures: {sorted(expected_keys - set(counts))}"
        )
    seen = set()
    for event in active:
        if event.get("event") == "prefill_chunk" and event.get("start") == 0:
            seen.add(event["length"])
        if event.get("event") == "scheduler_prefill":
            for start, end, intermediate in zip(values(event["starts"]), values(event["ends"]), event["intermediate"]):
                if start == 0 and not intermediate:
                    seen.add(end)
    if seen & {129, 130, 131}:
        raise ValueError(f"Required unseen lengths already occurred on this server: {sorted(seen & {129, 130, 131})}")
    return active


async def request(client, args, prompt, label, phase, tokens, sampled, raw, first_token=None):
    request_id = f"{args.run_id}-{label}"
    payload = dict(
        model=args.model,
        request_id=request_id,
        prompt=prompt["prompt_token_ids"],
        max_tokens=tokens,
        temperature=0.8 if sampled else 0.0,
        top_k=16 if sampled else 1,
        top_p=0.9 if sampled else 1.0,
        seed=47,
        ignore_eos=True,
        stream=True,
        stream_options={"include_usage": True},
        return_token_ids=True,
    )
    start, wall = time.perf_counter(), time.time()
    row = dict(
        label=label,
        phase=phase,
        prompt=prompt["name"],
        request_id=request_id,
        request=payload,
        started_unix=wall,
        sampled=sampled,
        status="running",
        output_token_ids=[],
        output_text="",
        stream_token_arrivals=[],
        finish_reason=None,
    )
    try:
        async with client.stream("POST", args.url.rstrip("/") + "/v1/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data, elapsed = line[5:].strip(), time.perf_counter() - start
                raw.write(json.dumps(dict(label=label, elapsed_seconds=elapsed, sse=data)) + "\n")
                raw.flush()
                if data == "[DONE]":
                    row["saw_done"] = True
                    break
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(event["error"])
                row["response_id"] = event.get("id", row.get("response_id"))
                if event.get("usage"):
                    row["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    ids = choice.get("token_ids") or []
                    row["output_text"] += choice.get("text", "")
                    row["output_token_ids"].extend(ids)
                    if ids:
                        row["stream_token_arrivals"].append(dict(seconds=elapsed, count=len(ids)))
                        if first_token is not None:
                            first_token.set()
                    if choice.get("finish_reason"):
                        row["finish_reason"] = choice["finish_reason"]
        if not row.get("saw_done") or row["finish_reason"] != "length":
            raise RuntimeError(f"Incomplete fixed-length request: finish={row['finish_reason']}")
        if len(row["output_token_ids"]) != tokens:
            raise RuntimeError(f"Expected {tokens} exact output IDs, got {len(row['output_token_ids'])}")
        usage = row.get("usage", {})
        if usage.get("prompt_tokens") != len(prompt["prompt_token_ids"]) or usage.get("completion_tokens") != tokens:
            raise RuntimeError(f"Server usage does not match untruncated request: {usage}")
        row["status"] = "pass"
    except Exception as error:
        row.update(status="fail", error=f"{type(error).__name__}: {error}")
    row["elapsed_seconds"] = time.perf_counter() - start
    arrivals = row["stream_token_arrivals"]
    first = arrivals[0]["seconds"] if arrivals else None
    last = arrivals[-1]["seconds"] if arrivals else None
    gaps = [(b["seconds"] - a["seconds"]) * 1000 for a, b in zip(arrivals, arrivals[1:])]
    # Keep chunk boundaries visible. Coalesced token arrivals are not exact ITL.
    exact_itl = [gap for gap, a, b in zip(gaps, arrivals, arrivals[1:]) if a["count"] == b["count"] == 1]
    row["metrics"] = dict(
        ttft_ms=first * 1000 if first is not None else None,
        tpot_ms=(last - first) * 1000 / (tokens - 1) if len(row["output_token_ids"]) == tokens else None,
        itl_ms=exact_itl,
        stream_gap_ms=gaps,
        coalesced_chunks=sum(item["count"] > 1 for item in arrivals),
        max_decode_pause_ms=max(gaps, default=None),
    )
    return row


async def workload(args, prompts, raw, rows):
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), trust_env=False) as client:

        async def send(name, label, phase, tokens=24, sampled=False, first_token=None):
            result = await request(client, args, prompts[name], label, phase, tokens, sampled, raw, first_token)
            rows.append(result)
            write_json(args.output / "requests.json", rows)
            print(f"{result['status']} {label}: {len(result['output_token_ids'])} tokens", flush=True)
            return result

        # These are measured first encounters, not warmup calls.
        for name in ("p129", "p130", "p131"):
            await send(name, f"first-{name}", "first-encounters")
        first = asyncio.Event()
        anchor = asyncio.create_task(send("p129", "anchor", "overlap", args.anchor_tokens, first_token=first))
        waiter = asyncio.create_task(first.wait())
        done, _ = await asyncio.wait({anchor, waiter}, timeout=args.timeout, return_when=asyncio.FIRST_COMPLETED)
        if not first.is_set():
            waiter.cancel()
            if not anchor.done():
                anchor.cancel()
            await asyncio.gather(anchor, waiter, return_exceptions=True)
            raise RuntimeError(f"Anchor did not produce its first token; completed tasks={len(done)}")
        await waiter

        async def delayed(name, delay):
            await asyncio.sleep(delay)
            return await send(name, f"overlap-{name}", "overlap")

        await asyncio.gather(anchor, delayed("long_a", 0), delayed("long_b", 0.03), delayed("p131", 0.06))
        # Change arrival order, batch combination, and sampling strategy.
        await asyncio.gather(
            send("long_b", "sample-long_b", "sampling-switch", sampled=True),
            send("p130", "sample-p130", "sampling-switch", sampled=True),
            send("long_a", "sample-long_a", "sampling-switch", sampled=True),
        )
        await asyncio.gather(
            *(send(name, f"repeat-{name}", "greedy-return") for name in ("p131", "long_b", "p129", "long_a", "p130"))
        )


def analyze(baseline, events, rows, args):
    failures = []
    captures = [event for event in baseline + events if event.get("event") == "capture"]
    signature_rows = {}
    for event in captures:
        key = key_id(event["key"])
        entry = signature_rows.setdefault(
            key,
            dict(
                key=event["key"],
                signature=execution_signature(event["key"]),
                captures=[],
                retirements=[],
                prefill_replays=0,
            ),
        )
        entry["captures"].append(
            dict(
                trace_id=event["trace_id"],
                seconds=event.get("seconds"),
                phase="baseline" if event in baseline else "workload",
            )
        )
    if any(len(entry["captures"]) != 1 for entry in signature_rows.values()):
        failures.append("An execution signature was captured more than once across baseline and workload")
    baseline_keys = {key_id(event["key"]) for event in baseline if event.get("event") == "capture"}
    if any(key_id(event["key"]) not in baseline_keys for event in events if event.get("event") == "capture"):
        failures.append("Capture used a signature absent from Muse's complete generator-wide prepared baseline")
    bad_lifecycle = [
        event
        for event in events
        if event.get("event")
        in {"adapter_loaded", "prepared", "teardown_release", "release", "retire", "invalidate", "evict", "recapture"}
    ]
    if bad_lifecycle:
        failures.append("Generator restart, preparation, or trace retirement occurred during the workload")
    for event in bad_lifecycle:
        if "key" in event and key_id(event["key"]) in signature_rows:
            signature_rows[key_id(event["key"])]["retirements"].append(event)
    for entry in signature_rows.values():
        entry["baseline_capture_count"] = sum(item["phase"] == "baseline" for item in entry["captures"])
        entry["workload_capture_count"] = sum(item["phase"] == "workload" for item in entry["captures"])
        entry["workload_retirement_count"] = len(entry["retirements"])
    scheduler, chunks = [], []
    current = None
    for event in events:
        if event.get("event") == "scheduler_prefill":
            current = event
            current = dict(current, starts=values(current["starts"]), ends=values(current["ends"]))
            current["chunk_lengths"] = [end - start for start, end in zip(current["starts"], current["ends"])]
            scheduler.append(current)
        elif event.get("event") == "prefill_chunk":
            entry = dict(
                event,
                scheduler_source_line=current["source_line"] if current else None,
                active_decode_requests=current.get("active_decode_requests", []) if current else [],
            )
            chunks.append(entry)
            key = key_id(event["key"])
            if key not in signature_rows:
                failures.append(f"Prefill replay has no recorded captured signature: {key}")
            else:
                signature_rows[key]["prefill_replays"] += 1
    if not scheduler or not chunks:
        failures.append("Missing actual scheduler_prefill or prefill_chunk events")
    if any(sum(event["chunk_lengths"]) > args.scheduler_budget for event in scheduler):
        failures.append("Actual scheduled token count exceeds the declared workload scheduler budget")
    our_ids = {row["request_id"] for row in rows}
    seen_ids = {req for event in scheduler for req in event["requests"]}
    for req in our_ids:
        if not any(req in seen for seen in seen_ids):
            failures.append(f"No scheduler events matched request {req}")
    foreign = [req for req in seen_ids if not any(ours in req for ours in our_ids)]
    if foreign:
        failures.append(f"Other workload requests make trace attribution ambiguous: {foreign}")
    intermediate = [event for event in scheduler if any(event["intermediate"])]
    offsets = sorted({start for event in scheduler for start in event["starts"] if start > 0})
    simultaneous = [event for event in scheduler if event.get("active_decode_requests") and any(event["intermediate"])]
    partial = [event for event in chunks if event["length"] < event["bucket"]]
    if not intermediate or len(offsets) < 2:
        failures.append("Did not exercise intermediate chunks with multiple continuation start positions")
    if not simultaneous:
        failures.append("No intermediate prefill event recorded concurrently live decode requests")
    if not partial:
        failures.append("No partial physical prefill bucket was executed")
    if not any(start == 0 for event in scheduler for start in event["starts"]):
        failures.append("No first prefill chunks were observed")
    final_rows = [
        (req, start, end)
        for event in scheduler
        for req, start, end, mid in zip(event["requests"], event["starts"], event["ends"], event["intermediate"])
        if not mid
    ]
    request_schedules, batch_combinations = [], []
    for row in rows:
        prompt_length = len(row["request"]["prompt"])
        spans = [
            (start, min(end, prompt_length))
            for event in scheduler
            for req, start, end in zip(event["requests"], event["starts"], event["ends"])
            if row["request_id"] in req and start < prompt_length
        ]
        expected_start = 0
        for start, end in spans:
            if start != expected_start or end <= start:
                failures.append(f"Noncontiguous scheduler prompt spans for {row['label']}: {spans}")
                break
            expected_start = end
        if expected_start != prompt_length:
            failures.append(f"Incomplete scheduler prompt span for {row['label']}: {expected_start}/{prompt_length}")
        request_schedules.append(dict(label=row["label"], prompt=row["prompt"], spans=spans))
        if not any(row["request_id"] in req and end == len(row["request"]["prompt"]) for req, start, end in final_rows):
            failures.append(f"Missing final logical prompt boundary for {row['label']}")
    for event in scheduler:
        labels = [next((row["label"] for row in rows if row["request_id"] in req), req) for req in event["requests"]]
        batch_combinations.append(dict(source_line=event["source_line"], rows=labels))
    if not any(len(event["requests"]) > 1 for event in scheduler):
        failures.append("No actual multi-request scheduler batch was recorded")
    if not (
        any(
            row["prompt"] == "long_b" and any(start > 0 and end > 4096 for start, end in row["spans"])
            for row in request_schedules
        )
        and any(event["start"] >= 4096 for event in chunks)
    ):
        failures.append("No >4096 prompt continuation executed for the hybrid-retirement case")
    comparisons = []
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] != "pass":
            failures.append(f"Request failed: {row['label']}: {row.get('error')}")
        if not row["sampled"] and row["status"] == "pass":
            grouped[row["prompt"]].append(row)
    for name, group in grouped.items():
        reference = group[0]
        for other in group[1:]:
            count = min(len(reference["output_token_ids"]), len(other["output_token_ids"]))
            equal = reference["output_token_ids"][:count] == other["output_token_ids"][:count]
            comparisons.append(
                dict(
                    prompt=name,
                    reference=reference["label"],
                    repeat=other["label"],
                    compared_tokens=count,
                    exact_match=equal,
                )
            )
            if not equal:
                failures.append(
                    f"Greedy repeat differs across request shape/order: {reference['label']} vs {other['label']}"
                )
    for name in ("p129", "p130", "p131", "long_a", "long_b"):
        if not any(row["prompt"] == name for row in comparisons):
            failures.append(f"No greedy contamination control comparison for {name}")
    before, after = last_counters(baseline), last_counters(events)
    counter_delta = {name: after.get(name, 0) - value for name, value in before.items()}
    counter_delta.update({name: value for name, value in after.items() if name not in before})
    if any(value < 0 for value in counter_delta.values()):
        failures.append("Generator counters decreased or disappeared during workload")
    if counter_delta.get("full_logits_readbacks", 0) != 0 or any(
        event.get("event") == "host_sampling_compatibility" for event in events
    ):
        failures.append("The supported device-sampling workload used host compatibility or full-logit reads")
    # Serving's canonical split sampler also handles greedy k=1, p=0. The
    # standalone optional argmax trace remains live but need not be replayed.
    for kind in ("sample", "decode", "prefill"):
        if counter_delta.get("replay_" + kind, 0) <= 0:
            failures.append(f"No recorded replay counter increase for {kind}")
    if counter_delta.get("captures", 0) != sum(event.get("event") == "capture" for event in events):
        failures.append("Capture counter change does not match captured-signature events")
    return dict(
        status="pass" if not failures else "fail",
        failures=failures,
        signatures=list(signature_rows.values()),
        lifecycle_events=bad_lifecycle,
        scheduler_events=scheduler,
        physical_chunks=chunks,
        request_schedules=request_schedules,
        batch_combinations=batch_combinations,
        coverage=dict(
            intermediate_scheduler_events=len(intermediate),
            continuation_offsets=offsets,
            concurrent_decode_events=len(simultaneous),
            partial_physical_chunks=len(partial),
        ),
        greedy_repeat_comparisons=comparisons,
        counter_baseline=before,
        counter_final=after,
        preparation_timing=[event for event in baseline if event.get("event") == "prepared"],
        compilation_timing_note="Included in measured generator-wide preparation; not separately instrumented",
        counter_delta=counter_delta,
        workload_capture_count=sum(event.get("event") == "capture" for event in events),
        workload_retirement_count=len(bad_lifecycle),
        shutdown_releases="outside workload; server remains running",
        quality_verdict="not evaluated; greedy repeat equality is a contamination regression check only",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default=MODEL, help="Served model name")
    parser.add_argument("--tokenizer", type=Path, default=SNAPSHOT)
    parser.add_argument("--events", required=True, type=Path, help="Existing live server MUSE_VLLM_EVENTS JSONL")
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory; must not exist")
    parser.add_argument("--scheduler-budget", type=int, default=512)
    parser.add_argument("--anchor-tokens", type=int, default=192)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.scheduler_budget < 32 or args.anchor_tokens < 32:
        parser.error("Use scheduler-budget >=32 and anchor-tokens >=32")
    args.run_id = "mixed-trace-" + uuid.uuid4().hex[:12]
    # Validate evidence before issuing any HTTP work.
    initial = args.events.read_bytes()
    file_identity = (args.events.stat().st_dev, args.events.stat().st_ino)
    baseline = baseline_check(parse_events(initial))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompts = {
        name: make_prompt(tokenizer, name, length, topic)
        for name, length, topic in (
            ("p129", 129, "how rain forms"),
            ("p130", 130, "the phases of the Moon"),
            ("p131", 131, "how seeds grow"),
            ("long_a", args.scheduler_budget * 2 + 163, "how ocean tides work"),
            ("long_b", max(4259, args.scheduler_budget * 3 + 131), "how birds migrate"),
        )
    }
    if max(len(prompt["prompt_token_ids"]) for prompt in prompts.values()) + args.anchor_tokens > 131072:
        parser.error("Workload exceeds the recorded context contract")
    args.output.mkdir(parents=True, exist_ok=False)
    with urlopen(args.url.rstrip("/") + "/metrics", timeout=10) as response:
        (args.output / "metrics_before.txt").write_bytes(response.read())
    (args.output / "baseline.events.jsonl").write_bytes(initial)
    write_json(args.output / "prompts.json", prompts)
    template = tokenizer.chat_template
    metadata = dict(
        command=sys.argv,
        model_id=MODEL,
        revision=REVISION,
        served_model=args.model,
        tokenizer_path=str(args.tokenizer),
        tokenizer_class=type(tokenizer).__name__,
        chat_template_present=bool(template),
        chat_template_sha256=hashlib.sha256(str(template).encode()).hexdigest() if template else None,
        prompt_mode="chat" if template else "completion",
        endpoint="/v1/completions",
        rendering=(
            "HF apply_chat_template with add_generation_prompt; exact token IDs submitted"
            if template
            else "Plain continuation token IDs; no chat wrapper"
        ),
        event_source=str(args.events),
        baseline_bytes=len(initial),
        baseline_sha256=hashlib.sha256(initial).hexdigest(),
        baseline_adapter=baseline[0],
        scheduler_budget=args.scheduler_budget,
        workload="unseen 129/130/131, decode anchor + long arrivals, stochastic switch, reordered greedy repeats",
        generation_caps=dict(ordinary=24, anchor=args.anchor_tokens),
        trace_allocation_safety="inspect server log separately",
        memory_evidence="HTTP KV-cache/request counters in metrics_before/after.txt; device allocation ownership audited in source",
    )
    write_json(args.output / "manifest.json", metadata)
    rows, errors = [], []
    started = time.perf_counter()
    with (args.output / "http_stream.jsonl").open("w") as raw:
        try:
            asyncio.run(workload(args, prompts, raw, rows))
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    elapsed = time.perf_counter() - started
    with urlopen(args.url.rstrip("/") + "/metrics", timeout=10) as response:
        (args.output / "metrics_after.txt").write_bytes(response.read())
    try:
        final = args.events.read_bytes()
        if file_identity != (args.events.stat().st_dev, args.events.stat().st_ino) or not final.startswith(initial):
            raise ValueError("Event source was replaced, truncated, or its baseline changed")
        suffix = final[len(initial) :]
        (args.output / "workload.events.jsonl").write_bytes(suffix)
        events = parse_events(final)[len(parse_events(initial)) :]
        report = analyze(baseline, events, rows, args)
    except Exception as error:
        report = dict(status="fail", failures=[f"{type(error).__name__}: {error}"])
    report["failures"].extend(errors)
    if errors:
        report["status"] = "fail"
    complete = [row for row in rows if row["status"] == "pass"]
    report["metrics"] = dict(
        elapsed_seconds=elapsed,
        submitted_requests=len(rows),
        completed_requests=len(complete),
        completed_requests_per_second=len(complete) / elapsed,
        output_tokens_per_second=sum(len(row["output_token_ids"]) for row in complete) / elapsed,
        ttft_ms=percentiles([row["metrics"]["ttft_ms"] for row in complete]),
        tpot_ms=percentiles([row["metrics"]["tpot_ms"] for row in complete]),
        itl_ms=percentiles([gap for row in complete for gap in row["metrics"]["itl_ms"]]),
        stream_gap_ms=percentiles([gap for row in complete for gap in row["metrics"]["stream_gap_ms"]]),
        itl_note="Exact only for adjacent single-token chunks; coalesced arrivals retained in requests.json",
        scope="Entire measured mixed traffic, including first logical shapes, offsets and intentional arrival gaps",
    )
    write_json(args.output / "requests.json", rows)
    write_json(args.output / "summary.json", report)
    print(json.dumps(dict(status=report["status"], failures=report["failures"], metrics=report["metrics"]), indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
