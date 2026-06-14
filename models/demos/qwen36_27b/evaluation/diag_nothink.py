# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Validate accuracy on chat prompts with thinking DISABLED (direct answers),
and also let one thinking-enabled prompt run long enough to reach an answer.
Default config (CPU prefill + full-fused decode).
"""
import argparse, faulthandler, time
import torch, ttnn
from transformers import AutoTokenizer

import models.demos.qwen36_27b.tt.deltanet as dn
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import load_state_dict
from models.demos.qwen36_27b.tt.model import TtQwen36Model
from models.demos.qwen36_27b.tt.generator import Qwen36Generator

faulthandler.dump_traceback_later(1200, exit=False)
MP = "/home/yito/work/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

QUESTIONS = [
    "What is the capital of France?",
    "What is 17 + 25?",
    "Write a Python function to reverse a string.",
]


def run(gen, cfg, tok, ids, n):
    gen.reset()
    tpf = time.perf_counter()
    pf_logits = gen.prefill(ids)
    pf_dt = time.perf_counter() - tpf
    print(f"    [prefill {ids.shape[1]/pf_dt:.1f} t/s, {ids.shape[1]} tok in {pf_dt:.2f}s]", flush=True)
    logits = ttnn.to_torch(pf_logits).float().reshape(-1)
    nt = torch.argmax(logits[:cfg.vocab_size]).item()
    out = [nt]
    t0 = time.perf_counter()
    steps = 0
    for _ in range(n):
        _, ntt = gen.decode_one_token(torch.tensor([[nt]], dtype=torch.long))
        nt = ntt.item()
        out.append(nt)
        steps += 1
        if tok.eos_token_id is not None and nt == tok.eos_token_id:
            break
    dt = time.perf_counter() - t0
    print(f"    [decode {steps/dt:.2f} tok/s, {steps} tok in {dt:.2f}s]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--think-tokens", type=int, default=200)
    args = ap.parse_args()
    print(f"[kernel] FULL={dn.USE_FULL_FUSED_KERNEL} PREFILL={dn.USE_PREFILL_FUSED_KERNEL}", flush=True)
    cfg = Qwen36ModelConfig()
    tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
    sd = load_state_dict(cfg, model_path=MP)
    dev = ttnn.open_device(device_id=0)
    try:
        print("building model...", flush=True)
        model = TtQwen36Model(dev, sd, cfg)
        gen = Qwen36Generator(model, cfg, tokenizer=tok)
        del sd
        print("model built", flush=True)

        print("\n########## enable_thinking=False (direct answers) ##########", flush=True)
        for q in QUESTIONS:
            s = tok.apply_chat_template([{"role": "user", "content": q}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(s, return_tensors="pt")
            out = run(gen, cfg, tok, ids, args.n)
            print(f"\nQ: {q}", flush=True)
            print(f"A: {tok.decode(out, skip_special_tokens=True)!r}", flush=True)

        print("\n########## thinking enabled, long generation ##########", flush=True)
        q = QUESTIONS[0]
        s = tok.apply_chat_template([{"role": "user", "content": q}],
                                    tokenize=False, add_generation_prompt=True)
        ids = tok.encode(s, return_tensors="pt")
        out = run(gen, cfg, tok, ids, args.think_tokens)
        print(f"\nQ(think): {q}", flush=True)
        print(f"A(think): {tok.decode(out, skip_special_tokens=True)!r}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
