"""Live-server nonaligned/chunked/concurrent traffic with trace evidence.

Client timing covers first encounters; benchmark headlines use the shared runner.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
from transformers import AutoTokenizer


async def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    topics = {
        129: "Explain why leaves are green in one sentence.",
        130: "Describe how rain forms in two sentences.",
        131: "Explain what a bicycle does in two sentences.",
        513: "Explain how bread is baked in a short paragraph.",
        645: "Describe the orbit of Earth around the Sun in a short paragraph.",
    }
    prompts = {}
    for length, content in topics.items():
        text = tokenizer.apply_chat_template(
            [dict(role="user", content=content)], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer.encode(text)
        end = ids.index(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        ids[end:end] = [198] * (length - len(ids))
        assert len(ids) == length
        prompts[length] = ids
    rows, differences = [], []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal = args.output.with_suffix(".jsonl")
    before = time.monotonic()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as session:

        async def request(label, length, sampling="greedy", delay=0):
            await asyncio.sleep(delay)
            start = time.monotonic()
            times, fragments, token_ids, usage = [], [], [], None
            payload = dict(
                model="Qwen/Qwen3.8-Flash-Next",
                prompt=prompts[length],
                temperature=0 if sampling == "greedy" else 0.8,
                top_k=1 if sampling == "greedy" else 8,
                top_p=1,
                max_tokens=args.tokens,
                ignore_eos=True,
                return_token_ids=True,
                stream=True,
                stream_options=dict(include_usage=True),
            )
            async with session.post(args.url + "/v1/completions", json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(await response.text())
                async for line in response.content:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    if "error" in data:
                        raise RuntimeError(data)
                    if data.get("usage"):
                        usage = data["usage"]
                    for choice in data.get("choices", []):
                        token_ids.extend(choice.get("token_ids") or [])
                        if choice.get("text"):
                            times.append(time.monotonic())
                            fragments.append(choice["text"])
            end = time.monotonic()
            assert times and usage and usage["completion_tokens"] == args.tokens, (label, usage)
            assert usage["prompt_tokens"] == length, usage
            assert len(token_ids) == args.tokens, (label, len(token_ids))
            row = dict(
                label=label,
                prompt_length=length,
                output_tokens=args.tokens,
                sampling=sampling,
                start=start,
                end=end,
                ttft_ms=(times[0] - start) * 1000,
                stream_chunk_gaps_ms=np.diff(times).tolist(),
                max_stream_pause_ms=max(np.diff(times), default=0) * 1000,
                text="".join(fragments),
                prompt_token_ids=prompts[length],
                token_ids=token_ids,
                usage=usage,
            )
            row["stream_chunk_gaps_ms"] = [v * 1000 for v in row["stream_chunk_gaps_ms"]]
            rows.append(row)
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {k: row[k] for k in ("label", "prompt_length", "sampling", "ttft_ms", "max_stream_pause_ms")}
                ),
                flush=True,
            )
            return row

        controls = {}
        for length in (129, 130, 131) if args.smoke else (129, 130, 131, 513, 645):
            controls[length] = await request(f"first_{length}", length)
        if not args.smoke:
            # Long requests arrive while the first request is decoding. Their
            # lengths exceed the scheduler's512-token budget and end in tails.
            first = await asyncio.gather(
                request("mixed_129", 129),
                request("mixed_645", 645, delay=0.25),
                request("mixed_513", 513, delay=0.5),
                request("mixed_131", 131, delay=0.75),
            )
            second = await asyncio.gather(
                request("repeat_131", 131),
                request("sample_130", 130, "sample", delay=0.1),
                request("repeat_129", 129, delay=0.2),
            )
            final = await request("return_greedy_130", 130)
            for row in first + second + [final]:
                if row["sampling"] == "greedy" and row["prompt_length"] in controls:
                    control = controls[row["prompt_length"]]["token_ids"]
                    if row["token_ids"] != control:
                        position = next(i for i, (a, b) in enumerate(zip(control, row["token_ids"])) if a != b)
                        differences.append(
                            dict(
                                label=row["label"],
                                first_difference=position,
                                control_token=control[position],
                                candidate_token=row["token_ids"][position],
                            )
                        )
    elapsed = time.monotonic() - before
    summary = dict(
        status="failed" if differences else "pass",
        exact_greedy_differences=differences,
        model="Qwen/Qwen3.8-Flash-Next",
        url=args.url,
        workload="nonaligned sequential controls, concurrent chunked prefill/decode, mode switches and repeats",
        timing_scope="HTTP stream chunks; separate from canonical benchmark token ITL",
        requests=rows,
        completed_requests=len(rows),
        duration_s=elapsed,
        completed_requests_per_s=len(rows) / elapsed,
        ttft_ms=dict(zip(("p50", "p95", "p99"), np.percentile([r["ttft_ms"] for r in rows], [50, 95, 99]).tolist())),
        max_stream_pause_ms=max(r["max_stream_pause_ms"] for r in rows),
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if differences:
        raise AssertionError(f"Exact greedy lifecycle comparisons failed: {differences}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8019")
    parser.add_argument(
        "--checkpoint", default="/home/vkovacevic/qwen38-flash/hf-local/de4b8e4d43b917e7706784d8bb445c9af86a3540"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--smoke", action="store_true")
    asyncio.run(main(parser.parse_args()))
