# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Same-server nonaligned/chunked traffic and greedy request-isolation check.

Raw-token continuation stress coverage, not a model-quality verdict. Reads
generator events from the supplied server log, including pre-workload captures.
--latency-only preserves the workload/isolation checks but cannot establish
trace, allocator, or program-cache acceptance; use the default tracked run too.
"""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import numpy as np
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--control-result", type=Path)
    parser.add_argument(
        "--latency-only",
        action="store_true",
        help="Measure identical traffic without telemetry contract checks; never acceptance-ready",
    )
    args = parser.parse_args()
    baseline_log = args.server_log.read_text(errors="replace")
    results = []
    begin = time.perf_counter()

    def request(name, length, *, generated=64, sampled=False, penalties=False):
        # Distinct valid histories; exact integer prompt lengths are observable
        # in HTTP usage and in the adapter's actual chunk start/end events.
        prompt = [1000 + (i % 200) for i in range(length)]
        body = dict(
            model=args.model,
            prompt=prompt,
            max_tokens=generated,
            temperature=0.0,
            ignore_eos=True,
            return_token_ids=True,
            stream=True,
            stream_options={"include_usage": True},
        )
        if sampled:
            body.update(temperature=0.7, top_k=4, top_p=0.9, seed=42)
        if penalties:
            body.update(presence_penalty=0.5, frequency_penalty=0.25, repetition_penalty=1.1)
        submitted = time.perf_counter()
        response = requests.post(args.server_url + "/v1/completions", json=body, stream=True, timeout=900)
        response.raise_for_status()
        ids, chunks, text, usage = [], [], "", None
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                new_ids = choice.get("token_ids") or []
                if new_ids:
                    chunks.append(dict(seconds=time.perf_counter() - submitted, count=len(new_ids)))
                    ids.extend(new_ids)
                text += choice.get("text", "")
        result = dict(
            name=name,
            prompt_length=length,
            generated=generated,
            sampled=sampled,
            penalties=penalties,
            submitted_seconds=submitted - begin,
            duration_seconds=time.perf_counter() - submitted,
            token_ids=ids,
            chunks=chunks,
            text=text,
            usage=usage,
        )
        assert usage and usage["prompt_tokens"] == length, result
        assert len(ids) == generated == usage["completion_tokens"], result
        result["ttft_ms"] = chunks[0]["seconds"] * 1000
        result["itl_ms"] = [(b["seconds"] - a["seconds"]) * 1000 for a, b in zip(chunks, chunks[1:])]
        return result

    try:
        # First encounters with these logical tails, with no per-request warmup.
        for length in (129, 130, 131):
            results.append(request(f"solo_{length}", length))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            # A decoder remains live while several long prefills cross the
            # scheduler's512-token budget. The log must prove interleaving.
            jobs = [pool.submit(request, "concurrent_decode", 129, generated=256)]
            for length in (1027, 1539, 131):
                jobs.append(pool.submit(request, f"mixed_{length}", length))
            results.extend(job.result() for job in jobs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(request, f"reordered_{length}", length) for length in (131, 129, 130)]
            results.extend(job.result() for job in jobs)
        results.append(request("sampled_switch", 130, sampled=True))
        results.append(request("penalty_trace_switch", 130, penalties=True))
        results.append(request("greedy_after_sample", 129))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(request, f"same_prompt_{index}", 129) for index in range(4)]
            results.extend(job.result() for job in jobs)
        by_name = {result["name"]: result for result in results}
        comparisons = [(f"solo_{n}", f"reordered_{n}") for n in (129, 130, 131)]
        comparisons += [("solo_129", "greedy_after_sample"), ("solo_131", "mixed_131")]
        comparisons += [("solo_129", f"same_prompt_{index}") for index in range(4)]
        for left, right in comparisons:
            assert by_name[left]["token_ids"] == by_name[right]["token_ids"], (left, right)
        log = args.server_log.read_text(errors="replace")
        events = {}
        for tag in (
            "QWEN_TRACE",
            "QWEN_PREFILL_CHUNK",
            "QWEN_VLLM_PREFILL",
            "QWEN_VLLM_DECODE",
            "QWEN_VLLM_PAGE_UPDATE",
        ):
            events[tag] = [json.loads(line.split(tag + " ", 1)[1]) for line in log.splitlines() if tag + " " in line]
        workload_lines = log[len(baseline_log) :].splitlines()
        chunks = [
            json.loads(line.split("QWEN_PREFILL_CHUNK ", 1)[1])
            for line in workload_lines
            if "QWEN_PREFILL_CHUNK " in line
        ]
        assert any(chunk["start"] > 0 and chunk["length"] % 32 for chunk in chunks), "No continuation tail"
        # Telemetry-disabled latency runs must not infer zero retirements,
        # unchanged allocation state, or stable program counts from absent logs.
        signatures = interleaved = consumed_growth = unchanged_steps = page_only = None
        if not args.latency_only:
            assert any("QWEN_VLLM_DECODE " in line for line in workload_lines), "No decode execution"
            # Decode must execute while an actual intermediate scheduler prefill
            # remains unfinished, not merely between two independent requests.
            pending_slots, interleaved = set(), False
            for line in workload_lines:
                if "QWEN_VLLM_PREFILL " in line:
                    event = json.loads(line.split("QWEN_VLLM_PREFILL ", 1)[1])
                    for slot, final in zip(event["slots"], event["sampling_mask"]):
                        if final:
                            pending_slots.discard(slot)
                        else:
                            pending_slots.add(slot)
                if "QWEN_VLLM_DECODE " in line and pending_slots:
                    event = json.loads(line.split("QWEN_VLLM_DECODE ", 1)[1])
                    interleaved |= event["active_rows"] > 0
            assert interleaved, "No decode while scheduler prefill continuation was pending"
            captures = [event for event in events["QWEN_TRACE"] if event["event"] == "capture"]
            signatures = [
                (event["key"], event.get("logical_batch", event["batch"]), event["cache_pages"]) for event in captures
            ]
            assert len(signatures) == len(set(signatures)), "Repeated trace capture"
            assert not any(event["event"] == "shutdown_release" for event in events["QWEN_TRACE"])
            # Reconstruct execution positions from authoritative reloads plus
            # exactly one advance per traced decode. Page-only reloads retain this
            # state; their scheduler positions may lag due to pending async output.
            positions, new_pages, consumed_growth = [], {}, []
            unchanged_steps, page_only = 0, 0
            for line in workload_lines:
                if "QWEN_VLLM_PAGE_UPDATE " in line:
                    event = json.loads(line.split("QWEN_VLLM_PAGE_UPDATE ", 1)[1])
                    if event["reload_inputs"]:
                        positions = event["scheduler_positions"][:]
                        new_pages = {}
                    else:
                        page_only += 1
                        for row, col, old, new in event["changed"]:
                            if old == 0 and new > 0:
                                new_pages[(row, col)] = new
                if "QWEN_VLLM_DECODE " in line:
                    event = json.loads(line.split("QWEN_VLLM_DECODE ", 1)[1])
                    unchanged_steps += not event["reload_inputs"] and not event["reload_page_table"]
                    for row, position in enumerate(positions):
                        if position < 0:
                            continue
                        key = (row, position // 32)
                        if key in new_pages:
                            consumed_growth.append(
                                dict(row=row, position=position, column=key[1], page=new_pages.pop(key))
                            )
                        positions[row] += 1
            assert len(consumed_growth) >= 2, "No automatic new blocks consumed across multiple page boundaries"
            assert unchanged_steps > 0, "No unchanged-table resident decode"
        control_comparisons = []
        if args.control_result:
            control = {row["name"]: row for row in json.loads(args.control_result.read_text())["requests"]}
            for result in results:
                if not result["sampled"]:
                    assert result["token_ids"] == control[result["name"]]["token_ids"], result["name"]
                    control_comparisons.append(result["name"])
        program_counts = {
            event["program_cache_entries"]
            for tag in ("QWEN_TRACE", "QWEN_VLLM_PREFILL", "QWEN_VLLM_DECODE")
            for event in events[tag]
            if "program_cache_entries" in event
        }
        if not args.latency_only:
            assert len(program_counts) <= 1, ("Program cache grew after preparation", sorted(program_counts))
        total = time.perf_counter() - begin
        itls = [itl for result in results for itl in result["itl_ms"]]
        ttfts = [result["ttft_ms"] for result in results]
        report = dict(
            passed=True,
            acceptance_ready=not args.latency_only,
            latency_only=args.latency_only,
            verdict_scope="request/isolation/latency checks only" if args.latency_only else "full lifecycle contract",
            contract_checks={
                name: "NOT RUN" if args.latency_only else "passed"
                for name in (
                    "trace_reuse",
                    "decode_prefill_interleaving",
                    "allocator_page_growth",
                    "unchanged_table_decode",
                    "program_cache_stability",
                )
            },
            acceptance_note=(
                "Latency evidence only. A separate default tracked lifecycle run remains mandatory."
                if args.latency_only
                else "Existing lifecycle contract checks passed."
            ),
            mode="raw-token continuation stress, not qualitative",
            requests=results,
            events=events,
            workload_prefill_chunks=chunks,
            captures_before_workload=None if args.latency_only else baseline_log.count('"event": "capture"'),
            trace_signatures=signatures,
            retirements=None if args.latency_only else 0,
            program_cache_entries=None if args.latency_only else sorted(program_counts),
            concurrent_decode_between_prefills=interleaved,
            ttft_ms=dict(zip(("p50", "p95", "p99"), np.percentile(ttfts, [50, 95, 99]).tolist())),
            itl_ms=dict(zip(("p50", "p99", "max"), np.percentile(itls, [50, 99, 100]).tolist())),
            completed_request_throughput=len(results) / total,
            duration_seconds=total,
            greedy_comparisons=comparisons,
            consumed_allocator_growth=consumed_growth,
            unchanged_table_decode_steps=unchanged_steps,
            page_only_updates=page_only,
            nonoverlapped_control=str(args.control_result) if args.control_result else None,
            control_comparisons=control_comparisons,
        )
    except BaseException as error:
        report = dict(
            passed=False,
            acceptance_ready=False,
            latency_only=args.latency_only,
            contract_checks={
                name: "NOT RUN" if args.latency_only else "not completed"
                for name in (
                    "trace_reuse",
                    "decode_prefill_interleaving",
                    "allocator_page_growth",
                    "unchanged_table_decode",
                    "program_cache_stability",
                )
            },
            error=repr(error),
            requests=results,
        )
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({key: value for key, value in report.items() if key not in ("events", "requests")}), flush=True
        )


if __name__ == "__main__":
    main()
