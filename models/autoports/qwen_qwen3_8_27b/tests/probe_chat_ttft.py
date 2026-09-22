# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: retain first-event and first-content timings for the same requests."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


async def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    text = tokenizer.decode([100 + i % 256 for i in range(2 * args.isl)])
    report = dict(isl=args.isl, osl=args.osl, concurrency=args.concurrency, cohorts=[])
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:

        async def request(index):
            payload = dict(
                model="Qwen/Qwen3.8-27B",
                messages=[dict(role="user", content=text)],
                temperature=0,
                max_completion_tokens=args.osl,
                truncate_prompt_tokens=args.isl,
                stream=True,
                stream_options=dict(include_usage=True),
            )
            start = time.perf_counter()
            first_event = first_content = headers_at = None
            events, fragments, usage = [], [], None
            async with session.post(args.url + "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                headers_at = time.perf_counter()
                async for line in response.content:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    now = time.perf_counter()
                    data = json.loads(line[6:])
                    usage = data.get("usage") or usage
                    for choice in data.get("choices", []):
                        delta = choice.get("delta", {})
                        if first_event is None:
                            first_event = now
                        content = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                        if content:
                            first_content = now if first_content is None else first_content
                            fragments.append(content)
                        if len(events) < 3:
                            events.append(dict(seconds=now - start, delta=delta))
            if first_event is None or not usage or usage["prompt_tokens"] != args.isl:
                raise RuntimeError(f"Missing streamed event/usage or unexpected input count: {usage}")
            return dict(
                index=index,
                start=start,
                end=time.perf_counter(),
                headers_s=headers_at - start,
                first_event_s=first_event - start,
                first_content_s=None if first_content is None else first_content - start,
                events=events,
                usage=usage,
                output="".join(fragments),
            )

        report["single_request_control"] = await request(-1)
        for repeat in range(args.repeats):
            semaphore = asyncio.Semaphore(args.concurrency)

            async def guarded(index):
                async with semaphore:
                    return await request(index)

            rows = await asyncio.gather(*(guarded(i) for i in range(args.requests)))
            cohort = dict(
                repeat=repeat,
                requests=rows,
                mean_first_event_s=sum(x["first_event_s"] for x in rows) / len(rows),
                mean_first_content_s=sum(x["first_content_s"] for x in rows) / len(rows),
            )
            report["cohorts"].append(cohort)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print("CHAT_TTFT", json.dumps({k: v for k, v in cohort.items() if k != "requests"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=252)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
