# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Upstream vLLM bench with exact token-ID transport for its random prompts.

The pinned client's tokenizer and the older TT server can segment the same text
differently. Preserve the upstream random generator's validated sequence instead
of retokenizing its decoded text on the server. HTTP timing/scoring stay upstream.
"""

import hashlib
import importlib.metadata
import json
import sys
from dataclasses import replace
from pathlib import Path


def token_id_requests(requests, tokenizer):
    converted = []
    for request in requests:
        if not isinstance(request.prompt, str):
            raise ValueError("Exact-token transport requires unbatched upstream random text prompts")
        tokens = tokenizer.encode(request.prompt, add_special_tokens=False)
        if len(tokens) != request.prompt_len:
            raise ValueError(f"Upstream random prompt mismatch: {len(tokens)} != {request.prompt_len}")
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Random prompt must contain nonnegative integer token IDs")
        converted.append(replace(request, prompt=tokens))
    if len({tuple(request.prompt) for request in converted}) != len(converted):
        raise ValueError("Performance prompts must be distinct")
    return converted


def option(name, default=None):
    for index, argument in enumerate(sys.argv[1:], 1):
        if argument == name:
            return sys.argv[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return default


def main():
    if sys.argv[1:3] != ["bench", "serve"]:
        raise SystemExit("This wrapper only runs upstream vllm bench serve")
    if option("--dataset-name", "random") != "random" or option("--backend", "vllm") != "vllm":
        raise SystemExit("Exact-token wrapper requires random dataset and vllm completions backend")
    if option("--endpoint", "/v1/completions") != "/v1/completions":
        raise SystemExit("Token-ID transport requires /v1/completions")
    from vllm.benchmarks.datasets.datasets import RandomDataset
    from vllm.entrypoints.cli.main import main as upstream_main

    original_sample = RandomDataset.sample

    def sample(self, tokenizer, *args, **kwargs):
        requests = token_id_requests(original_sample(self, tokenizer, *args, **kwargs), tokenizer)
        result_dir = Path(option("--result-dir", "."))
        result_name = option("--result-filename", "benchmark.json")
        result_dir.mkdir(parents=True, exist_ok=True)
        artifact = result_dir / (result_name + ".input-token-ids.json")
        artifact.write_text(
            json.dumps(
                {
                    "method": "Unmodified upstream RandomDataset.sample; encode using its tokenizer, validate count, transport IDs without server retokenization",
                    "upstream_vllm_version": importlib.metadata.version("vllm"),
                    "tokenizer_class": type(tokenizer).__name__,
                    "tokenizer_path": option("--tokenizer", option("--model")),
                    "random_seed": option("--seed", "0"),
                    "requests": [
                        {
                            "request_id": request.request_id,
                            "input_tokens": request.prompt_len,
                            "output_tokens": request.expected_output_len,
                            "prompt_token_ids": request.prompt,
                            "prompt_token_ids_sha256": hashlib.sha256(json.dumps(request.prompt).encode()).hexdigest(),
                        }
                        for request in requests
                    ],
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        print(f"EXACT_TOKEN_TRANSPORT {len(requests)} distinct prompts; evidence={artifact}", flush=True)
        return requests

    RandomDataset.sample = sample
    upstream_main()


if __name__ == "__main__":
    main()
