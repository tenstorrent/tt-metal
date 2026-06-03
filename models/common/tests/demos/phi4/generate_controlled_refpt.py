#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generate a deterministic, metadata-rich CPU reference ``.refpt`` for Phi-4 (microsoft/phi-4).

Mirrors the DeepSeek-R1-14B / Qwen2.5-7B controlled refpt pattern (see
``dev-tools/agentic-bringup/skills/reference-sanity.md``). Output ``.refpt`` contains:

    - reference_tokens: [prompt_len + num_target]
    - top5_tokens:      [num_target, 5], aligned to target positions
    - prompt_len:       int
    - metadata:         provenance + deterministic generation settings

The script prints intrinsic top-1 / top-5 self-checks before writing.
Intrinsic top-1 must be >= 95% for the refpt to be a valid accuracy reference.

Usage::

    HF_HUB_OFFLINE=1 ./python_env/bin/python \\
      models/common/tests/demos/phi4/generate_controlled_refpt.py --hf-model microsoft/phi-4

PERF.md's ``ci-token-matching`` workload prefills 512 tokens; ``--num-target-tokens 512``
keeps the continuation in the same regime. The default prompt is padded/truncated by the
demo so ``prompt_len`` need not be exactly 512 here — ``prompt_len`` is recorded in metadata
and the demo aligns to it.
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

# Full-weights snapshot pinned for microsoft/phi-4 (matches Phi4Transformer.DEFAULT_HF_REVISION).
DEFAULT_HF_REVISION = "187ef0342fff0eb3333be9f00389385e95ef0b61"

DEFAULT_PROMPT = "Write a short paragraph explaining why deterministic model references are important for debugging."


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic CPU Phi-4 reference .refpt")
    parser.add_argument("--hf-model", default="microsoft/phi-4", help="HF model id (default: microsoft/phi-4)")
    parser.add_argument(
        "--output",
        default="models/tt_transformers/tests/reference_outputs/phi-4.refpt",
        help="Output .refpt path",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-target-tokens", type=int, default=512, help="Number of continuation tokens")
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT, help="Prompt text for chat-template encoding")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16", help="CPU model dtype")
    parser.add_argument(
        "--revision",
        default=DEFAULT_HF_REVISION,
        help="Pin a specific HF revision (commit SHA). Recorded in metadata.",
    )
    return parser


def _dtype_from_arg(name: str) -> torch.dtype:
    return torch.float32 if name == "float32" else torch.bfloat16


def main() -> None:
    args = _build_parser().parse_args()
    _seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, revision=args.revision)
    load_kwargs: dict = {"torch_dtype": _dtype_from_arg(args.dtype)}
    if args.revision:
        load_kwargs["revision"] = args.revision
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, **load_kwargs)
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

    revision = args.revision or getattr(model.config, "_commit_hash", None) or getattr(model.config, "revision", None)
    metadata = {
        "hf_model_id": args.hf_model,
        "revision": revision,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "seed": args.seed,
        "generation_mode": "teacher_forcing_greedy_cpu",
        "created_at": datetime.now(timezone.utc).isoformat(),
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

    if top1_consistency < 0.95:
        print(
            f"\nWARNING: intrinsic top-1 consistency {top1_consistency * 100:.1f}% < 95%. "
            "This refpt has a low ceiling; TT accuracy thresholds will need to be lowered. "
            "See dev-tools/agentic-bringup/skills/reference-sanity.md."
        )


if __name__ == "__main__":
    main()
