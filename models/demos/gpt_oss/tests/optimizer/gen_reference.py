# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF fp32 reference for the optimizer accuracy gate (test_optimizer_pcc.py) on gpt-oss-20b.

Sequence: the first prompt of demo/sample_prompts/input_data_questions_prefill_128.json, encoded the way the demo
encodes it (ModelArgs.encode_prompt: chat template with add_generation_prompt), then a greedy 100-token HF
continuation. Logits: one fp32 teacher-forced forward over prompt + continuation[:-1]; row i is the
distribution that predicts continuation[i] (positions P-1 .. P+98).

Run once on the host (CPU, ~90 GB RAM for fp32):
    python models/demos/gpt_oss/tests/optimizer/gen_reference.py --out generated/optimizer_reference/gpt-oss-20b-logits.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = Path(__file__).resolve().parents[2] / "demo" / "sample_prompts" / "input_data_questions_prefill_128.json"


def encode(tokenizer, prompt: str) -> list[int]:
    """Same steps as ModelArgs.encode_prompt for a plain user prompt."""
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
    )
    if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "ids"):
        encoded = list(encoded.ids)
    return [int(t) for t in encoded]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("HF_MODEL") or "openai/gpt-oss-20b")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    prompt = json.loads(PROMPTS.read_text())[0]["prompt"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = encode(tokenizer, prompt)
    print(f"prompt {prompt!r}: {len(prompt_ids)} tokens", flush=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    print(f"loaded fp32 in {time.time() - t0:.0f} s; dtype {next(model.parameters()).dtype}", flush=True)

    with torch.no_grad():
        t1 = time.time()
        out = model.generate(
            torch.tensor([prompt_ids]), max_new_tokens=args.tokens, do_sample=False, eos_token_id=None, pad_token_id=0
        )
        cont = out[0, len(prompt_ids) :].tolist()
        assert len(cont) == args.tokens, len(cont)
        print(f"greedy continuation in {time.time() - t1:.0f} s: {tokenizer.decode(cont)!r}", flush=True)

        t2 = time.time()
        full = torch.tensor([prompt_ids + cont[:-1]])
        logits = model(full).logits[0, len(prompt_ids) - 1 :].float()
        assert logits.shape[0] == args.tokens
        agree = int((logits.argmax(-1) == torch.tensor(cont)).sum())
        print(f"teacher-forced forward in {time.time() - t2:.0f} s; argmax == greedy at {agree}/{args.tokens}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": args.model,
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "continuation_ids": cont,
            "logits": logits.contiguous(),
            "dtype": "float32",
            "attn_implementation": "eager",
        },
        args.out,
    )
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
