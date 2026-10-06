# SPDX-License-Identifier: Apache-2.0
"""Same-server logical-tail, concurrent-growth and trace-lifetime evidence."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
from transformers import AutoTokenizer


async def main(args):
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = tok.encode("The lighthouse stands beside the ocean. ", add_special_tokens=False)
    events = Path(args.events)
    before = events.read_text().splitlines() if events.exists() else []
    workload_started = time.monotonic()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        async with session.get(args.url + "/metrics") as response:
            metrics_before = await response.text()

        async def request(length, label, sampled=False):
            ids = (base * ((length + len(base) - 1) // len(base)))[:length]
            if label.endswith("b"):
                ids[-1] = tok.encode(" blue", add_special_tokens=False)[0]
            payload = dict(
                model=args.model,
                prompt=ids,
                max_tokens=args.output,
                temperature=0.7 if sampled else 0.0,
                top_k=16 if sampled else 1,
                ignore_eos=True,
                stream=True,
                return_token_ids=True,
                stream_options={"include_usage": True},
            )
            started = time.monotonic()
            stamps, chunks, usage = [], [], None
            output_ids = []
            async with session.post(args.url + "/v1/completions", json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(await response.text())
                async for line in response.content:
                    if not line.startswith(b"data: ") or b"[DONE]" in line:
                        continue
                    data = json.loads(line[6:])
                    if data.get("choices"):
                        output_ids.extend(data["choices"][0].get("token_ids") or [])
                        text = data["choices"][0]["text"]
                        if text:
                            chunks.append(text)
                            stamps.append(time.monotonic())
                    if data.get("usage"):
                        usage = data["usage"]
            assert usage["prompt_tokens"] == length, usage
            assert usage["completion_tokens"] == args.output, usage
            assert len(output_ids) == args.output, (len(output_ids), args.output)
            return dict(
                label=label,
                prompt_ids=ids,
                output_token_ids=output_ids,
                length=length,
                sampled=sampled,
                usage=usage,
                text="".join(chunks),
                elapsed=time.monotonic() - started,
                ttft_ms=1000 * (stamps[0] - started),
                intervals_ms=[1000 * (b - a) for a, b in zip(stamps, stamps[1:])],
            )

        results = []
        for length in (129, 130, 131, 511, 512, 513):
            results.append(await request(length, f"first-{length}"))
        results.extend(
            await asyncio.gather(
                request(1537, "concurrent-a"),
                request(1025, "concurrent-b"),
                request(129, "concurrent-c"),
                request(131, "concurrent-d"),
            )
        )
        results.append(await request(130, "sampling-switch", True))
        results.extend(
            await asyncio.gather(request(131, "repeat-131"), request(129, "repeat-129"), request(130, "repeat-130"))
        )
        async with session.get(args.url + "/metrics") as response:
            metrics_after = await response.text()
    elapsed = time.monotonic() - workload_started
    after = events.read_text().splitlines()
    new = [json.loads(x) for x in after[len(before) :]]
    ttfts = [r["ttft_ms"] for r in results]
    intervals = [v for r in results for v in r["intervals_ms"]]
    report = dict(
        status="recorded-not-validated",
        baseline_events=len(before),
        baseline_trace_events=[json.loads(x) for x in before if json.loads(x)["event"] in ("capture", "prepared")],
        results=results,
        events=new,
        elapsed_seconds=elapsed,
        completed_requests_per_second=len(results) / elapsed,
        ttft_percentiles_ms=dict(zip(("p50", "p95", "p99"), np.percentile(ttfts, [50, 95, 99]).tolist())),
        streaming_pause_percentiles_ms=dict(
            zip(("p50", "p95", "p99", "max"), np.percentile(intervals, [50, 95, 99, 100]).tolist())
        ),
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        prompt_format="Raw completion continuation stress; qualitative verdict uses separate chat-template suite",
    )
    Path(args.output_file).write_text(json.dumps(report, indent=2) + "\n")
    for length in (129, 130, 131):
        a = next(x for x in results if x["label"] == f"first-{length}")
        b = next(x for x in results if x["label"] == f"repeat-{length}")
        assert a["output_token_ids"] == b["output_token_ids"], (length, a["output_token_ids"], b["output_token_ids"])
    assert not any(x["event"] == "shutdown_release" for x in new)
    active_pids = {x["pid"] for x in new}
    all_events = [json.loads(x) for x in after if json.loads(x)["pid"] in active_pids]
    ids = [(x["pid"], x["key"]) for x in all_events if x["event"] == "capture"]
    assert len(ids) == len(set(ids)), ids
    report["status"] = "pass"
    Path(args.output_file).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(completed=len(results), output=args.output, trace_count=len(ids))))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--output", type=int, default=96)
    p.add_argument("--output-file", required=True)
    asyncio.run(main(p.parse_args()))
