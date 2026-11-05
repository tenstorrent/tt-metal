# SPDX-License-Identifier: Apache-2.0
"""Growing conversation cache traffic; not a headline performance benchmark."""

import argparse
import hashlib
import json
import math
import re
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from .check_served_state import log_events, trace_summary

METRIC_KEYS = ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total", "vllm:num_preemptions_total")


def parse_metrics(text):
    result = {}
    for line in text.splitlines():
        match = re.match(
            r"(vllm:(?:prefix_cache_hits_total|prefix_cache_queries_total|num_preemptions_total))(?:\{.*\})?\s+([\d.eE+-]+)$",
            line,
        )
        if match:
            value = float(match[2])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid cumulative counter {match[1]}")
            result[match[1]] = result.get(match[1], 0.0) + value
    missing = set(METRIC_KEYS) - set(result)
    if missing:
        raise ValueError(f"Required serving counters absent: {sorted(missing)}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--contexts", default="4096,8192,16384,24576,32768")
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision="842da3794eaa0b77d5f08bae87a17459d91ff475")
    vocabulary_size = len(tokenizer)
    contexts = list(map(int, args.contexts.split(",")))
    assert args.concurrency > 0 and args.output_tokens > 0 and len(contexts) >= 2
    assert all(length > 0 for length in contexts)
    run_nonce = uuid.uuid4().hex
    assert all(b > a + args.output_tokens for a, b in zip(contexts, contexts[1:]))
    baseline = args.server_log.read_bytes()
    histories = [
        tokenizer.encode(f"{run_nonce}:{i}: Explain careful experimental reasoning. ") for i in range(args.concurrency)
    ]
    append = tokenizer.encode(
        " A measurement should distinguish the proposed explanation from alternatives. ", add_special_tokens=False
    )
    rounds = []

    def metrics():
        with urllib.request.urlopen(args.url + "/metrics", timeout=30) as response:
            text = response.read().decode()
        return parse_metrics(text)

    def completed_metrics(before, expected_queries):
        # Metrics consume scheduler statistics asynchronously. Wait for this
        # round's token queries instead of attributing a delayed earlier wave.
        deadline = time.monotonic() + 30
        while True:
            after = metrics()
            delta = {key: after[key] - before[key] for key in METRIC_KEYS}
            if any(value < 0 for value in delta.values()):
                raise ValueError("Serving counters reset during the workload")
            if delta["vllm:prefix_cache_queries_total"] > expected_queries:
                raise ValueError("Unexpected prefix-cache queries; concurrent external traffic invalidates attribution")
            if delta["vllm:prefix_cache_queries_total"] == expected_queries:
                return after
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Prefix metrics did not account for {expected_queries} queried tokens")
            time.sleep(0.25)

    def request(index, length):
        tokens = histories[index]
        assert len(tokens) < length
        remaining = length - len(tokens)
        tokens = tokens + (append * ((remaining + len(append) - 1) // len(append)))[:remaining]
        body = dict(
            model=args.model,
            prompt=tokens,
            max_tokens=args.output_tokens,
            ignore_eos=True,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            return_token_ids=True,
        )
        started = time.monotonic()
        req = urllib.request.Request(
            args.url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=1800) as response:
            result = json.load(response)
        generated = result["choices"][0]["token_ids"]
        assert len(generated) == args.output_tokens
        assert all(0 <= token < vocabulary_size for token in generated)
        usage = result.get("usage", {})
        assert usage.get("prompt_tokens") == length, usage
        assert usage.get("completion_tokens") == args.output_tokens, usage
        histories[index] = tokens + generated
        return dict(
            conversation=index,
            prompt_tokens=len(tokens),
            output_tokens=len(generated),
            generated_token_ids=generated,
            prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            elapsed_seconds=time.monotonic() - started,
        )

    report = dict(
        scope="synthetic growing conversation cache traffic; raw continuation stress, not a qualitative verdict",
        concurrency=args.concurrency,
        context_schedule=contexts,
        output_tokens_per_turn=args.output_tokens,
        run_nonce=run_nonce,
        prefix_ceiling_basis="Prior prompt plus generated tokens except the final unprocessed sampled token; block alignment may lower attainable hits.",
        pool_bound=False,
        rounds=rounds,
        errors=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        for turn, length in enumerate(contexts):
            # The final sampled token has not yet entered the cache.
            ceiling = sum(max(0, len(history) - 1) for history in histories) if turn else 0
            before = metrics()
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                rows = list(executor.map(lambda index: request(index, length), range(args.concurrency)))
            expected_queries = args.concurrency * length
            after = completed_metrics(before, expected_queries)
            delta = {key: after[key] - before[key] for key in METRIC_KEYS}
            hits = delta.get("vllm:prefix_cache_hits_total")
            rounds.append(
                dict(
                    turn=turn,
                    context_tokens=length,
                    prior_cached_token_ceiling=ceiling,
                    metrics_delta=delta,
                    expected_queried_tokens=expected_queries,
                    hit_fraction_of_ceiling=hits / ceiling if hits is not None and ceiling else None,
                    requests=rows,
                )
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: value for key, value in rounds[-1].items() if key != "requests"}), flush=True)
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        log_after = args.server_log.read_bytes()
        if len(log_after) < len(baseline):
            report["errors"].append("Server log truncated during workload")
        events = log_events(log_after[len(baseline) :])
        report["elapsed_seconds"] = time.monotonic() - started
        report["trace_evidence"] = trace_summary(log_events(baseline), events)
        report["prefilled_tokens"] = sum(
            int(event["valid"]) for event in events if event.get("event") == "generator_prefill_chunk"
        )
        report["prefilled_tokens"] += sum(event.get("event") == "generator_prefill_prompt_token" for event in events)
        report["prefilled_tokens"] += sum(
            int(event["valid"]) for event in events if event.get("event") == "generator_prefill_partial_page"
        )
        report["memory_snapshots"] = [event for event in events if event.get("event") == "serving_memory"]
        measured = [row for row in rounds if row["prior_cached_token_ceiling"]]
        hits = [row["metrics_delta"].get("vllm:prefix_cache_hits_total") for row in measured]
        report["hit_fraction_of_ceiling"] = (
            sum(hits) / sum(row["prior_cached_token_ceiling"] for row in measured)
            if hits and all(hit is not None for hit in hits)
            else None
        )
        late = measured[-max(1, math.ceil(len(rounds) / 4)) :]
        late_hits = [row["metrics_delta"].get("vllm:prefix_cache_hits_total") for row in late]
        report["late_hit_fraction"] = (
            sum(late_hits) / sum(row["prior_cached_token_ceiling"] for row in late)
            if late_hits and all(hit is not None for hit in late_hits)
            else None
        )
        report[
            "late_fraction_basis"
        ] = "Final quarter of all requests, conservatively rounded up to whole concurrent rounds; per-round gates also enforced"
        report["late_observed_requests"] = sum(len(row["requests"]) for row in late)
        report["final_context_token_sum"] = args.concurrency * rounds[-1]["context_tokens"] if rounds else None
        report["prefilled_to_final_context_ratio"] = (
            report["prefilled_tokens"] / report["final_context_token_sum"]
            if report["final_context_token_sum"]
            else None
        )
        report["final_context_tokens"] = rounds[-1]["context_tokens"] if rounds else None
        preemptions = [row["metrics_delta"].get("vllm:num_preemptions_total") for row in rounds]
        report["preemptions"] = (
            sum(preemptions) if preemptions and all(value is not None for value in preemptions) else None
        )
        trace = report["trace_evidence"]
        captures_after_warmup = sum(row["workload_captures"] for row in trace["execution_signatures"])
        report["trace_captures_after_warmup"] = captures_after_warmup
        ratios = [row["hit_fraction_of_ceiling"] for row in measured]
        report["checks"] = {
            "all_rounds_completed": len(rounds) == len(contexts),
            "prefix_reuse_at_least_90_percent": bool(ratios)
            and all(value is not None and 0.9 <= value <= 1 for value in ratios),
            "late_reuse_at_least_90_percent": report["late_hit_fraction"] is not None
            and 0.9 <= report["late_hit_fraction"] <= 1,
            "trace_lifetime": trace["trace_lifetime_pass"],
            "no_captures_after_warmup": captures_after_warmup == 0,
            "trace_timing_evidence_present": not trace["missing_reasons"],
        }
        report["status"] = "pass" if not report["errors"] and all(report["checks"].values()) else "fail"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "pass":
        raise AssertionError(f"Growing traffic failed: {args.output}")


if __name__ == "__main__":
    main()
