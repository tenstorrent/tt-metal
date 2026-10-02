# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run HF Gemma4 in fp32 on the CPU and keep its final (pre-norm) hidden states for the likelihood check.

The prompt is the GPU capture's token prefix, prefilled in chunks through HF's own KV cache, so memory stays
linear in context. Every position is kept; compare against it with ``likelihood.py``.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger

from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4PrefillAdapter
from models.demos.gemma4_d_p.tt.runners.likelihood import HIDDEN_SAMPLES, next_tokens, save_samples


@torch.no_grad()
def prepare_cpu_reference(trace_dir, destination, context_len, chunk_size):
    from transformers import AutoModelForCausalLM

    from models.demos.gemma4_d_p.utils.partial_weights import resolve_checkpoint_dir

    token_ids = json.loads((Path(trace_dir) / "metadata.json").read_text())["token_ids"]
    if not 0 < context_len <= len(token_ids):
        raise ValueError(f"The capture holds {len(token_ids)} tokens, not {context_len}")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        resolve_checkpoint_dir(Gemma4PrefillAdapter().hf_model_id), dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    logger.info(f"Loaded {type(model).__name__} in fp32 in {time.perf_counter() - started:.0f} s")

    # The final norm's input is the decoder output the TT model and the GPU capture both return.
    hidden = []
    hook = model.model.language_model.norm.register_forward_hook(lambda _, args, __: hidden.append(args[0][0].clone()))
    cache = None
    try:
        for start in range(0, context_len, chunk_size):
            started = time.perf_counter()
            end = min(start + chunk_size, context_len)
            output = model(
                input_ids=torch.tensor([token_ids[start:end]]), past_key_values=cache, use_cache=True, logits_to_keep=1
            )
            cache = output.past_key_values
            logger.info(f"CPU reference rows [{start}, {end}) in {time.perf_counter() - started:.0f} s")
    finally:
        hook.remove()

    positions = list(range(context_len))
    path = destination / HIDDEN_SAMPLES
    save_samples(
        path,
        torch.tensor(positions),
        torch.cat(hidden),
        next_tokens(token_ids, positions),
        source="hf_cpu_fp32",
        trace_dir=Path(trace_dir).resolve(),
        chunk_size=chunk_size,
    )
    logger.info(f"CPU reference: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--trace-dir", type=Path, default=Path(Gemma4PrefillAdapter.prefill_trace_default))
    parser.add_argument("--context-len", type=int, default=32768)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    prepare_cpu_reference(args.trace_dir, args.destination, args.context_len, args.chunk_size)


if __name__ == "__main__":
    main()
