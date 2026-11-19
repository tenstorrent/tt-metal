# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One full-context serving acceptance, with a non-aligned logical prompt."""

import argparse
import json
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context, output_tokens = 262144, 33
    prompt_length = context - output_tokens
    body = dict(
        model="Qwen/Qwen3.6-27B",
        prompt=[1000 + i % 200 for i in range(prompt_length)],
        max_tokens=output_tokens,
        temperature=0.0,
        ignore_eos=True,
        return_token_ids=True,
        stream=True,
        stream_options={"include_usage": True},
    )
    begin = time.perf_counter()
    result = dict(
        passed=False,
        workload="Raw-token capacity stress, not qualitative",
        max_model_len=context,
        prompt_length=prompt_length,
        output_tokens=output_tokens,
        prompt_mod32=prompt_length % 32,
        temperature=0.0,
        ignore_eos=True,
    )
    try:
        models = requests.get(args.server_url + "/v1/models", timeout=60).json()
        result["models"] = models
        response = requests.post(args.server_url + "/v1/completions", json=body, stream=True, timeout=7200)
        response.raise_for_status()
        ids, times, usage = [], [], None
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                new = choice.get("token_ids") or []
                if new:
                    times.append(time.perf_counter() - begin)
                    ids.extend(new)
        assert usage["prompt_tokens"] == prompt_length
        assert len(ids) == usage["completion_tokens"] == output_tokens
        assert usage["total_tokens"] == context
        result.update(
            passed=True,
            usage=usage,
            token_ids=ids,
            token_chunk_seconds=times,
            ttft_seconds=times[0],
            duration_seconds=time.perf_counter() - begin,
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
