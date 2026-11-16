# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run upstream vLLM bench serve with exact, preserved random token-ID inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np

REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"


def install_adapter():
    from vllm.benchmarks import serve
    from vllm.benchmarks.datasets.datasets import RandomDataset

    upstream_samples = serve.get_samples
    result_paths = []

    def exact_samples(args, tokenizer):
        assert args.dataset_name == "random" and args.endpoint == "/v1/completions"
        assert args.random_prefix_len == 0 and float(args.random_range_ratio) == 0
        assert args.random_batch_size == 1 and not args.enable_multimodal_chat
        assert args.backend == "vllm" and not args.extra_body
        assert REVISION in str(args.tokenizer)
        assert tokenizer.num_special_tokens_to_add() == 0
        expected = args.random_input_len

        def generate(self, *, prefix_token_ids, input_len, offset, index, allowed_tokens, **kwargs):
            assert not prefix_token_ids and input_len == expected
            ids = allowed_tokens[(offset + index + np.arange(input_len)) % len(allowed_tokens)].tolist()
            assert len(ids) == expected and all(type(token) is int for token in ids)
            return ids, len(ids), 0

        original = RandomDataset.generate_token_sequence
        RandomDataset.generate_token_sequence = generate
        try:
            samples = upstream_samples(args, tokenizer)
        finally:
            RandomDataset.generate_token_sequence = original
        prohibited = set(tokenizer.all_special_ids)
        hashes = []
        records = []
        for sample in samples:
            ids = sample.prompt
            assert isinstance(ids, list) and len(ids) == sample.prompt_len == expected
            assert not prohibited.intersection(ids)
            digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            hashes.append(digest)
            records.append({"request_id": sample.request_id, "sha256": digest, "token_ids": ids})
        assert len(set(hashes)) == len(samples), "Performance prompts must be distinct"
        output = Path(args.result_dir) / (args.result_filename + ".inputs.json")
        result_paths.append(Path(args.result_dir) / args.result_filename)
        output.write_text(
            json.dumps(
                {"transport": "integer_token_ids", "tokenizer": args.tokenizer, "seed": args.seed, "requests": records}
            )
            + "\n"
        )
        return samples

    serve.get_samples = exact_samples
    return result_paths


if __name__ == "__main__":
    result_paths = install_adapter()
    from vllm.entrypoints.cli.main import main

    main()
    # Normalize before the collector hashes the result, so repository hooks
    # cannot invalidate byte-level evidence links by adding the final newline.
    for result_path in result_paths:
        payload = result_path.read_bytes()
        if not payload.endswith(b"\n"):
            result_path.write_bytes(payload + b"\n")
