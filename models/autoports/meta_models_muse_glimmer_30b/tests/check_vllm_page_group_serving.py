# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare allocator growth and reordered requests to pre-optimization serving."""

import argparse
import asyncio
import json
from pathlib import Path

from check_vllm_mixed_trace import MODEL, SNAPSHOT, make_prompt
from openai import AsyncOpenAI
from transformers import TokenizersBackend


async def run(args):
    tokenizer = TokenizersBackend.from_pretrained(SNAPSHOT, local_files_only=True)
    prompts = [
        make_prompt(tokenizer, str(n), n, topic)
        for n, topic in zip([129, 130, 131, 4259], ["the water cycle", "solar energy", "ocean tides", "photosynthesis"])
    ]
    results = []
    async with AsyncOpenAI(base_url=args.url.rstrip("/") + "/v1", api_key="empty") as client:

        async def send(index, phase):
            response = await client.completions.create(
                model=MODEL,
                prompt=prompts[index]["prompt_token_ids"],
                max_tokens=100,
                temperature=0,
                extra_body={"ignore_eos": True, "return_token_ids": True},
            )
            raw = response.model_dump()
            if (
                raw["usage"]["prompt_tokens"] != prompts[index]["requested_length"]
                or raw["usage"]["completion_tokens"] != 100
            ):
                raise AssertionError(raw["usage"])
            results.append(dict(index=index, phase=phase, response=raw))

        for index in range(4):
            await send(index, "sequential")
        for order in ([0, 1, 2], [2, 0, 1]):
            await asyncio.gather(*(send(index, str(order)) for index in order))
    reference = json.loads(args.control.read_text()) if args.control else None
    if reference:
        if reference["prompts"] != prompts:
            raise AssertionError("Control prompts changed")
        expected = {(r["phase"], r["index"]): r["response"]["choices"][0]["token_ids"] for r in reference["results"]}
        for row in results:
            row["exact_control_match"] = (
                row["response"]["choices"][0]["token_ids"] == expected[(row["phase"], row["index"])]
            )
    passed = all(r.get("exact_control_match", True) for r in results)
    args.output.write_text(
        json.dumps(
            dict(
                status="pass" if passed else "fail",
                control=str(args.control) if args.control else None,
                model=MODEL,
                layers=52,
                max_num_seqs=32,
                context=131072,
                prompts=prompts,
                results=results,
                boundary="HTTP token outputs; three native page boundaries per short request; 4259-token sliding retirement",
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"{'PASS' if passed else 'FAIL'}: {len(results)} requests, 100 tokens each")
    if not passed:
        raise AssertionError("Serving tokens differ from pre-optimization control")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--control", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
