#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Generate a deterministic, metadata-rich CPU reference ``.refpt`` for Qwen2-7B-Instruct.

This script emits:
    - reference_tokens: [prompt_len + num_target]
    - top5_tokens: [num_target, 5], aligned to target positions
    - prompt_len: int
    - metadata: provenance + deterministic generation settings

Usage::

    python models/common/tests/demos/qwen2_7b/generate_controlled_refpt.py \\
        --hf-model Qwen/Qwen2-7B-Instruct \\
        --output models/tt_transformers/tests/reference_outputs/Qwen2-7B-Instruct.refpt

Always verify intrinsic self-consistency (top-1 ≥ 95%) before using a ``.refpt`` for
accuracy thresholding — see the reference-sanity guide.
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

DEFAULT_PROMPT = "Write a short paragraph explaining why deterministic model references are important for debugging."


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic CPU Qwen2-7B reference .refpt")
    parser.add_argument("--hf-model", required=True, help="HF model id, e.g. Qwen/Qwen2-7B-Instruct")
    parser.add_argument(
        "--output",
        default="models/tt_transformers/tests/reference_outputs/Qwen2-7B-Instruct.refpt",
        help="Output .refpt path",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-target-tokens", type=int, default=512, help="Number of continuation tokens")
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT, help="Prompt text for chat-template encoding")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16", help="CPU model dtype")
    return parser


def _dtype_from_arg(name: str) -> torch.dtype:
    return torch.float32 if name == "float32" else torch.bfloat16


def main() -> None:
    args = _build_parser().parse_args()
    _seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
        torch_dtype=_dtype_from_arg(args.dtype),
    )
    model.eval()

    prompt_tokens = encode_prompt_hf(tokenizer, args.prompt_text)
    prompt_len = len(prompt_tokens)

    full_sequence: list[int] = list(prompt_tokens)
    top5_rows: list[torch.Tensor] = []

    with torch.no_grad():
        model_input = torch.tensor([prompt_tokens], dtype=torch.long)
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
    print("metadata:")
    for key, value in metadata.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
