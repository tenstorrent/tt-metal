# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate the AIME24 chat-template readiness reference for Laguna-XS-2.1.

Reuses the shared readiness scoring/serialisation (``_generate_one_entry``, ``save_reference``,
``Reference``, the AIME24 prompt loader and stop-id logic) verbatim so the produced .refpt is
schema- and semantics-identical to ``models.common.readiness_check.generate``. The only override:
this model's tokenizer returns a dict from ``apply_chat_template(tokenize=True)`` (an ``input_ids``
key), which the shared CLI's list-comprehension can't consume — we extract ``input_ids`` here.

Usage:
  python -m \
    models.autoports.poolside_laguna_xs_2_1.tests.gen_aime_reference --output <path> --gen-len 100 --top-k 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.common.readiness_check.generate import (
    DEFAULT_AIME24_PROMPTS_FILE,
    _generate_continuation_tokens,
    _generate_one_entry,
    _generation_stop_ids,
    _load_aime24_prompt,
    _safe_pad_id,
)
from models.common.readiness_check.schema import Reference, save_reference

MODEL_ID = "poolside/Laguna-XS-2.1"


def chat_template_tokens(tokenizer, prompt_text):
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=True
    )
    if isinstance(out, dict) or hasattr(out, "get"):  # BatchEncoding / dict
        out = out["input_ids"]
    if len(out) and isinstance(out[0], (list, tuple)):  # batched [[...]]
        out = out[0]
    return [int(t) for t in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gen-len", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--aime24-prompt-index", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).eval().to(device)

    stop_ids = _generation_stop_ids(tokenizer, model)
    eos_id = stop_ids[0]
    bos_id = tokenizer.bos_token_id
    pad_id = _safe_pad_id(tokenizer, stop_ids)

    prompt_text = _load_aime24_prompt(DEFAULT_AIME24_PROMPTS_FILE, args.aime24_prompt_index)
    prompt_tokens = torch.tensor(chat_template_tokens(tokenizer, prompt_text), dtype=torch.long)
    print(f"AIME24[{args.aime24_prompt_index}] chat-template prompt: {len(prompt_tokens)} tokens")

    gen_tokens = _generate_continuation_tokens(model, tokenizer, prompt_tokens, args.gen_len, device)
    print(f"HF greedy continuation: {len(gen_tokens)} tokens")

    entry = _generate_one_entry(model, tokenizer, prompt_tokens, gen_tokens, args.top_k, device)
    reference = Reference(
        k=args.top_k,
        hf_model_id=MODEL_ID,
        entries=[entry],
        token_ids_meta={
            "bos_id": int(bos_id) if bos_id is not None else None,
            "eos_id": int(eos_id),
            "pad_id": int(pad_id) if pad_id is not None else None,
        },
    )
    path = save_reference(reference, args.output)
    print(f"Reference saved to: {path}")


if __name__ == "__main__":
    main()
