# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runnable text-generation demo for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.

Loads a real prompt via the HF tokenizer, runs the SHARED chained TTNN pipeline
(tt/pipeline.py — the exact same forward the e2e test asserts), greedily decodes
on device, and prints the generated text.

Run:
  ./python_env/bin/python -m models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.demo.demo_text_generation \
      --prompt "The capital of France is" --max-new-tokens 5
"""
from __future__ import annotations

import argparse
import sys

import torch

import ttnn
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import pipeline as pl
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt._hf_compat import install_hf_compat

install_hf_compat()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="NemotronH-3-Nano-30B-A3B TTNN text-generation demo")
    ap.add_argument("--prompt", default="The capital of France is", help="input prompt text")
    ap.add_argument("--max-new-tokens", type=int, default=5, help="number of tokens to generate")
    ap.add_argument(
        "--compose",
        type=int,
        default=1,
        choices=[0, 1],
        help="1=compose graduated child stubs (Gate 2); 0=monolith backbone",
    )
    ap.add_argument("--device-id", type=int, default=0, help="TT device id to open")
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(pl.HF_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pl.HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    eos = int(getattr(model.config, "eos_token_id", 2))

    input_ids = tok(args.prompt, return_tensors="pt")["input_ids"]

    device = ttnn.open_device(device_id=args.device_id)
    try:
        pipe = pl.build_pipeline(device, model, compose=bool(args.compose))
        new_ids, _ = pipe.generate(input_ids, args.max_new_tokens, eos_token_id=eos)
    finally:
        ttnn.close_device(device)

    completion = tok.decode(new_ids)
    print("=" * 60)
    print(f"PROMPT     : {args.prompt!r}")
    print(f"GENERATED  : {completion!r}")
    print(f"FULL       : {args.prompt + completion!r}")
    print(f"NEW_IDS    : {new_ids}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
