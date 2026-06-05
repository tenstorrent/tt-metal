#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Generate a deterministic, metadata-rich CPU reference ``.refpt`` for Llama-3.3-70B-Instruct.

PERF.md workload: prefill = 512 tokens, 511 decode iterations for accuracy.
Pass ``--target-prompt-len 512`` (default) so the saved ``prompt_len`` matches the
perf benchmark's prefill budget.

This script emits:
    - reference_tokens: [prompt_len + num_target_tokens]
    - top5_tokens: [num_target_tokens, 5], aligned to target positions
    - prompt_len: int
    - metadata: provenance + deterministic generation settings

Usage::

    ./python_env/bin/python \\
        models/common/tests/demos/llama33_70b/generate_controlled_refpt.py \\
        --hf-model meta-llama/Llama-3.3-70B-Instruct

See ``dev-tools/agentic-bringup/skills/reference-sanity.md``.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.tt_transformers.tt.common import encode_prompt_hf

DEFAULT_PROMPT = (
    "Write a detailed explanation of how transformer attention mechanisms work, "
    "including the mathematical formulation of scaled dot-product attention, "
    "multi-head attention, positional encodings, and how these components interact "
    "in a full encoder-decoder or decoder-only architecture. Explain the role of "
    "the key, query, and value projections and why they are beneficial."
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic CPU Llama-3.3-70B reference .refpt")
    parser.add_argument("--hf-model", required=True, help="HF model id, e.g. meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument(
        "--output",
        default="models/tt_transformers/tests/reference_outputs/Llama-3.3-70B-Instruct.refpt",
        help="Output .refpt path",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-target-tokens", type=int, default=511, help="Number of continuation tokens")
    parser.add_argument(
        "--target-prompt-len",
        type=int,
        default=512,
        help="Pad/truncate tokenized prompt to this length (PERF.md prefill budget)",
    )
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT, help="Prompt text for chat-template encoding")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16", help="CPU model dtype")
    return parser


def _dtype_from_arg(name: str) -> torch.dtype:
    return torch.float32 if name == "float32" else torch.bfloat16


def main() -> None:
    args = _build_parser().parse_args()
    _seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=_dtype_from_arg(args.dtype),
    )
    model.eval()

    raw_ids = encode_prompt_hf(tokenizer, args.prompt_text)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Pad or truncate to target_prompt_len for PERF.md workload alignment.
    if len(raw_ids) < args.target_prompt_len:
        prefix = [pad_id] * (args.target_prompt_len - len(raw_ids))
        prompt_ids = prefix + list(raw_ids)
    else:
        prompt_ids = list(raw_ids)[-args.target_prompt_len :]

    prompt_len = len(prompt_ids)
    assert prompt_len == args.target_prompt_len, f"prompt_len={prompt_len} != target={args.target_prompt_len}"

    full_sequence: list[int] = list(prompt_ids)
    top5_rows: list[torch.Tensor] = []

    with torch.no_grad():
        model_input = torch.tensor([prompt_ids], dtype=torch.long)
        outputs = model(model_input, use_cache=True)
        past_key_values = outputs.past_key_values

        for step in range(args.num_target_tokens):
            logits = outputs.logits[0, -1, :]
            top5 = torch.topk(logits, k=5, dim=-1).indices.to(torch.long).cpu()
            top5_rows.append(top5)
            next_token = int(top5[0].item())
            full_sequence.append(next_token)
            if step < args.num_target_tokens - 1:
                next_input = torch.tensor([[next_token]], dtype=torch.long)
                outputs = model(next_input, use_cache=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values

    reference_tokens = torch.tensor(full_sequence, dtype=torch.long)
    top5_tokens = torch.stack(top5_rows, dim=0)
    target_tokens = reference_tokens[prompt_len:]

    top1_consistency = (top5_tokens[:, 0] == target_tokens).float().mean().item()
    top5_contains = (top5_tokens == target_tokens.unsqueeze(1)).any(dim=1).float().mean().item()

    created_at = datetime.now(timezone.utc).isoformat()
    revision = getattr(model.config, "_commit_hash", None) or getattr(model.config, "revision", None)
    metadata = {
        "hf_model_id": args.hf_model,
        "revision": revision,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "seed": args.seed,
        "generation_mode": "teacher_forcing_greedy_cpu",
        "created_at": created_at,
        "prompt_text": args.prompt_text,
        "num_target_tokens": args.num_target_tokens,
        "target_prompt_len": args.target_prompt_len,
        "dtype": args.dtype,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "reference_tokens": reference_tokens,
            "top5_tokens": top5_tokens,
            "prompt_len": prompt_len,
            "metadata": metadata,
        },
        out_path,
    )

    print(f"Saved controlled reference to: {out_path}")
    print(f"prompt_len={prompt_len}, total_len={reference_tokens.numel()}, target_len={target_tokens.numel()}")
    print(f"top1 consistency: {top1_consistency * 100:.2f}%")
    print(f"top5 containment: {top5_contains * 100:.2f}%")
    if top1_consistency < 0.99:
        print("WARNING: top1 < 99% — regenerate with different prompt or check model revision")
    print("metadata:")
    for key, value in metadata.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
