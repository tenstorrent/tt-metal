#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generate a **book-methodology** CPU reference ``.refpt`` for DeepSeek-R1-Distill-Qwen-14B.

Book methodology (identical in spirit to TTTv1
``models/tt_transformers/tests/generate_reference_outputs.py`` and the committed
Llama/Qwen/Mistral book references): teacher-force the HF model over ground-truth
tokens from a real corpus (``tale-of-two-cities.txt.bz2``) in a single forward pass
and record, per position, the model's top-5 predicted tokens for the *next* corpus
token. Targets come from the real text — **not** the model's own greedy output — so
the reference is a genuine accuracy yardstick, not a tautology.

This deliberately loads the model with its **native** HF config (no YaRN rope
injection, no second ``ModelArgs`` model), so the reference is faithful to the
shipped distill.

Output ``.refpt`` matches the committed sibling book refpts (bare, 2-D):

    - reference_tokens: LongTensor ``[1, total_length]``      (corpus token ids)
    - top5_tokens:      LongTensor ``[total_length - 1, 5]``  (HF top-5 for next token)

The script prints the HF model's intrinsic top-1 / top-5 accuracy against the corpus
as a health check before writing.

Usage::

    ./python_env/bin/python models/common/tests/demos/deepseek_r1_distill_qwen_14b/generate_book_refpt.py \\
        --hf-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

    # Pin a specific revision for reproducibility:
    ./python_env/bin/python models/common/tests/demos/deepseek_r1_distill_qwen_14b/generate_book_refpt.py \\
        --hf-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
        --revision 1df8507178afcc1bef68cd8c393f61a886323761
"""

from __future__ import annotations

import argparse
import bz2
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# tale-of-two-cities corpus, shared with the TTTv1 book-reference generator.
DEFAULT_CORPUS = "models/tt_transformers/tests/tale-of-two-cities.txt.bz2"


def _dtype_from_arg(name: str) -> torch.dtype:
    return torch.float32 if name == "float32" else torch.bfloat16


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a book-methodology CPU DeepSeek-R1-Distill-Qwen-14B reference .refpt"
    )
    parser.add_argument(
        "--hf-model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        help="HF model id (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)",
    )
    parser.add_argument(
        "--output",
        default="models/tt_transformers/tests/reference_outputs/DeepSeek-R1-Distill-Qwen-14B.refpt",
        help="Output .refpt path (shared reference_outputs dir, same as the sibling book refpts)",
    )
    parser.add_argument("--total-length", type=int, default=1024, help="Number of corpus tokens to score")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help="bz2-compressed corpus text file")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="CPU model dtype (float32 matches the TTTv1/family reference convention)",
    )
    parser.add_argument("--revision", default=None, help="Pin a specific HF revision (commit SHA)")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    load_kwargs: dict = {"trust_remote_code": True, "torch_dtype": _dtype_from_arg(args.dtype)}
    if args.revision:
        load_kwargs["revision"] = args.revision
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, **load_kwargs)
    model.eval()

    with bz2.open(args.corpus, "rt", encoding="utf-8") as f:
        text = f.read()

    total_length = args.total_length
    encoded = tokenizer(text, return_tensors="pt").input_ids[:, :total_length]  # [1, T]
    actual_len = encoded.shape[1]
    if actual_len < total_length:
        raise ValueError(f"Corpus only yields {actual_len} tokens (< {total_length}); use a longer corpus.")

    with torch.no_grad():
        logits = model(encoded).logits  # [1, T, V]

    # Position j predicts token j+1; drop the last position (it has no next-token target).
    # ``.clone()`` on the corpus slice is essential: without it the saved tensor is a view into the
    # full ~190k-token book tokenization and torch.save serializes the entire backing storage (~1.5 MB
    # vs the intended ~50 KB). Mirrors TTTv1 generate_reference_outputs.py.
    top5_tokens = torch.topk(logits[0, :-1, :].float(), k=5, dim=-1).indices.to(torch.long).clone()  # [T-1, 5]
    reference_tokens = encoded[:, :total_length].to(torch.long).clone().contiguous()  # [1, T]

    # Intrinsic health check: the HF model's own accuracy against the ground-truth corpus.
    targets = reference_tokens[0, 1:total_length]  # [T-1]
    top1 = (top5_tokens[:, 0] == targets).float().mean().item()
    top5 = (top5_tokens == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"top5_tokens": top5_tokens, "reference_tokens": reference_tokens}, out_path)

    print(f"Saved book reference to: {out_path}")
    print(
        f"total_length={total_length}, "
        f"top5_tokens={tuple(top5_tokens.shape)}, reference_tokens={tuple(reference_tokens.shape)}"
    )
    print(f"HF intrinsic top-1 vs corpus: {top1 * 100:.2f}%")
    print(f"HF intrinsic top-5 vs corpus: {top5 * 100:.2f}%")
    if top1 < 0.5:
        print(
            f"\nWARNING: HF intrinsic top-1 {top1 * 100:.1f}% < 50%. A healthy book reference for a strong "
            "model on natural English text is typically ~60-75% top-1; a low value points at a "
            "tokenizer / corpus / config problem — investigate before committing."
        )


if __name__ == "__main__":
    main()
