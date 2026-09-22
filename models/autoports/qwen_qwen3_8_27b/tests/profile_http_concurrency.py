# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Equal-work client-concurrency probe against an already-running B1 server.

This does not change server capacity. Launch with max_num_seqs=1 and prefix
caching disabled, and retain the server log as independent capacity evidence.
Every case submits the same number of identical fixed-token requests; only
the client semaphore changes. Request latency excludes client-semaphore wait.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp


async def run(args):
    payload = dict(
        model=args.model,
        prompt=[100 + i % 256 for i in range(args.isl)],
        max_tokens=args.osl,
        temperature=0,
        ignore_eos=True,
        stream=True,
        stream_options={"include_usage": True},
    )
    report = dict(num_requests=args.num_requests, isl=args.isl, osl=args.osl, cases=[])
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:

        async def request(index):
            start = time.perf_counter()
            first, usage = None, None
            events, fragments = [], []
            async with session.post(args.server_url.rstrip("/") + "/v1/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.content:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    usage = data.get("usage") or usage
                    for choice in data.get("choices", []):
                        fragment = choice.get("text", "")
                        if fragment:
                            now = time.perf_counter()
                            first = now if first is None else first
                            events.append(now - start)
                            fragments.append(fragment)
            end = time.perf_counter()
            if first is None or not usage:
                raise RuntimeError("A completion did not provide text and final token usage")
            if usage["prompt_tokens"] != args.isl or usage["completion_tokens"] != args.osl:
                raise RuntimeError(f"Request token counts differ from the fixed workload: {usage}")
            return dict(
                index=index,
                start=start,
                end=end,
                ttft=first - start,
                e2el=end - start,
                tpot=(end - first) / (args.osl - 1),
                events=events,
                usage=usage,
                output="".join(fragments),
            )

        report["warmup"] = await request(-1)
        for concurrency in args.concurrencies:
            semaphore = asyncio.Semaphore(concurrency)

            async def guarded(index):
                async with semaphore:
                    return await request(index)

            start = time.perf_counter()
            rows = await asyncio.gather(*(guarded(i) for i in range(args.num_requests)))
            wall = time.perf_counter() - start
            case = dict(
                concurrency=concurrency,
                wall=wall,
                output_tps=args.num_requests * args.osl / wall,
                mean_ttft=sum(row["ttft"] for row in rows) / len(rows),
                mean_e2el=sum(row["e2el"] for row in rows) / len(rows),
                mean_tpot=sum(row["tpot"] for row in rows) / len(rows),
                requests=rows,
            )
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: value for key, value in case.items() if key != "requests"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--concurrencies", default="1,8,16,1")
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=252)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.concurrencies = [int(value) for value in args.concurrencies.split(",")]
    if args.num_requests < 1 or args.isl < 1 or args.osl < 2 or any(c < 1 for c in args.concurrencies):
        parser.error("Request count, input length, and concurrency must be positive; output length must exceed one")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
