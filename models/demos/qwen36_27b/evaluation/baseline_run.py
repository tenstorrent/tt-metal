# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Baseline runner for Qwen3.6-27B on a single Blackhole (P150a path).

Reports, for a handful of prompts:
  - coherence (decoded text, NaN/inf check)
  - prefill tokens/s and decode tokens/s
  - which DeltaNet kernel path is active

Honors DISABLE_FUSED_KERNEL / DISABLE_FULL_FUSED_KERNEL env vars (read by tt.deltanet).

Usage (inside container via run_qwen36.sh):
  python -m models.demos.qwen36_27b.evaluation.baseline_run
  python -m models.demos.qwen36_27b.evaluation.baseline_run --max-tokens 64
"""
import argparse
import os
import time

import torch
import ttnn
from transformers import AutoTokenizer

from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import load_state_dict
from models.demos.qwen36_27b.tt.model import TtQwen36Model
from models.demos.qwen36_27b.tt.generator import Qwen36Generator
from models.demos.qwen36_27b.tt import deltanet as dn

DEFAULT_MODEL_PATH = (
    "/home/yito/work/hf_cache/models--Qwen--Qwen3.6-27B/"
    "snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

PROMPTS = [
    ("factual", "What is the capital of France?", 24),
    ("reasoning", "Explain why the sky is blue in one sentence.", 48),
    ("code", "Write a Python function to reverse a string.", 64),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--max-tokens", type=int, default=None, help="override per-prompt max tokens")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    print(f"[kernel] USE_FUSED_KERNEL={dn.USE_FUSED_KERNEL} "
          f"USE_FULL_FUSED_KERNEL={dn.USE_FULL_FUSED_KERNEL} "
          f"USE_PREFILL_FUSED_KERNEL={dn.USE_PREFILL_FUSED_KERNEL}")

    config = Qwen36ModelConfig()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("[weights] loading...")
    t0 = time.time()
    state_dict = load_state_dict(config, model_path=args.model_path)
    print(f"[weights] loaded {len(state_dict)} tensors in {time.time()-t0:.1f}s")

    device = ttnn.open_device(device_id=args.device_id)
    try:
        t0 = time.time()
        model = TtQwen36Model(device, state_dict, config)
        generator = Qwen36Generator(model, config, tokenizer=tokenizer)
        del state_dict
        print(f"[model] built in {time.time()-t0:.1f}s")

        for name, prompt, mnt in PROMPTS:
            mnt = args.max_tokens or mnt
            generator.reset()
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(formatted, return_tensors="pt")
            plen = input_ids.shape[1]

            t_pf = time.perf_counter()
            last_logits = generator.prefill(input_ids)
            pf_t = time.perf_counter() - t_pf

            logits_cpu = ttnn.to_torch(last_logits).float().reshape(-1)
            has_bad = bool(torch.isnan(logits_cpu).any() or torch.isinf(logits_cpu).any())
            next_token = torch.argmax(logits_cpu[:config.vocab_size]).item()
            gen = [next_token]

            t_dec = time.perf_counter()
            for _ in range(mnt - 1):
                tok = torch.tensor([[next_token]], dtype=torch.long)
                _, nt = generator.decode_one_token(tok)
                next_token = nt.item()
                gen.append(next_token)
                if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
                    break
            dec_t = time.perf_counter() - t_dec
            dec_n = len(gen) - 1

            text = tokenizer.decode(gen, skip_special_tokens=True)
            print(f"\n=== {name} ===")
            print(f"  prompt_tokens={plen}  prefill={plen/pf_t:.1f} t/s ({pf_t:.2f}s)  "
                  f"decode={dec_n/dec_t:.2f} t/s ({dec_t:.2f}s, {dec_n} tok)  nan/inf={has_bad}")
            print(f"  output: {text!r}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
