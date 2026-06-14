# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Is the residual garbage caused by the chat template / longer sequence, or the
prefill path? Build once (fixed default config = CPU prefill + full-fused decode)
and run, for both a RAW prompt and a CHAT-TEMPLATED prompt:
  - prefill -> next token + 16-token greedy answer
  - sequential decode over the same tokens -> next token   (consistency check)
"""
import argparse
import faulthandler
import torch
import ttnn
from transformers import AutoTokenizer

import models.demos.qwen36_27b.tt.deltanet as dn
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import load_state_dict
from models.demos.qwen36_27b.tt.model import TtQwen36Model
from models.demos.qwen36_27b.tt.generator import Qwen36Generator

faulthandler.dump_traceback_later(900, exit=False)
MP = "/home/yito/work/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def gen_greedy(gen, cfg, ids, n):
    gen.reset()
    logits = ttnn.to_torch(gen.prefill(ids)).float().reshape(-1)
    nt = torch.argmax(logits[:cfg.vocab_size]).item()
    out = [nt]
    for _ in range(n):
        _, ntt = gen.decode_one_token(torch.tensor([[nt]], dtype=torch.long))
        nt = ntt.item()
        out.append(nt)
    return out, logits


def seq_next(gen, cfg, ids):
    gen.reset()
    allg = gen.get_logits_for_sequence(ids)
    return torch.argmax(allg[-1][:cfg.vocab_size]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=16)
    args = ap.parse_args()
    print(f"[kernel] FUSED={dn.USE_FUSED_KERNEL} FULL={dn.USE_FULL_FUSED_KERNEL} PREFILL={dn.USE_PREFILL_FUSED_KERNEL}", flush=True)
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

        # ---- RAW prompt ----
        raw = "The capital of France is"
        ids_raw = torch.tensor([tok.encode(raw, add_special_tokens=False)], dtype=torch.long)
        print(f"\n=== RAW: {raw!r} ids={ids_raw.tolist()} ===", flush=True)
        out, _ = gen_greedy(gen, cfg, ids_raw, args.n)
        print(f"  prefill answer: {tok.decode(out)!r}", flush=True)
        print(f"  seq next: {tok.decode([seq_next(gen, cfg, ids_raw)])!r}", flush=True)

        # ---- CHAT-TEMPLATED prompt ----
        q = "What is the capital of France?"
        formatted = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        ids_chat = tok.encode(formatted, return_tensors="pt")
        print(f"\n=== CHAT templated string ===\n{formatted!r}", flush=True)
        print(f"  ids({ids_chat.shape[1]})={ids_chat.tolist()}", flush=True)
        out, _ = gen_greedy(gen, cfg, ids_chat, args.n)
        print(f"  prefill answer: {tok.decode(out)!r}", flush=True)
        print(f"  seq next: {tok.decode([seq_next(gen, cfg, ids_chat)])!r}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
