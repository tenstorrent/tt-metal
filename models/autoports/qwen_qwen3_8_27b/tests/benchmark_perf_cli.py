#!/usr/bin/env python3
"""Upstream vLLM bench CLI with random prompts transported as exact token IDs.

The upstream decode/encode round trip is not length preserving for this
tokenizer. Keep its seeded sampling and all serving/timing code, but pass the
sampled IDs directly to the completions API, which supports token-array input.
"""
import numpy as np
import hashlib
import json
from pathlib import Path
import sys
from vllm.benchmarks.datasets.datasets import RandomDataset


def exact_token_sequence(self, *, tokenizer, prefix_token_ids, prefix_len,
                         vocab_size, input_len, offset, index, allowed_tokens):
    tokens = prefix_token_ids + allowed_tokens[
        (offset + index + np.arange(input_len)) % len(allowed_tokens)
    ].tolist()
    assert len(tokens) == prefix_len + input_len == 4096
    assert all(isinstance(t, int) and 0 <= t < vocab_size for t in tokens)
    if '--result-dir' in sys.argv and '--result-filename' in sys.argv:
        result_dir=Path(sys.argv[sys.argv.index('--result-dir')+1])
        name=sys.argv[sys.argv.index('--result-filename')+1].removesuffix('.json')
        path=result_dir/(name+'-inputs.jsonl')
        with path.open('w' if index==0 else 'a') as stream:
            stream.write(json.dumps(dict(index=index,tokens=tokens,
                prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()))+'\n')
    return tokens, len(tokens), 0


RandomDataset.generate_token_sequence = exact_token_sequence

if __name__ == '__main__':
    from vllm.entrypoints.cli.main import main
    main()
