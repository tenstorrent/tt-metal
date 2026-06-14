# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Isolate whether the prefill path encodes prompt context correctly.

Method A: prefill(prompt) -> argmax next token, then greedy continue.
Method B: feed the SAME prompt token-by-token in decode mode
          (get_logits_for_sequence) -> argmax next token, then greedy continue.

If A is wrong but B is right -> the prefill path is broken.
If both wrong            -> decode / attention / rope / embedding bug.
"""
import argparse
import faulthandler
import torch
import ttnn

# If anything hangs > 600s, dump all thread tracebacks so we can see WHERE
# (device wedges are common on this host under heavy/thermal load).
faulthandler.dump_traceback_later(600, exit=False)
from transformers import AutoTokenizer

from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import load_state_dict
from models.demos.qwen36_27b.tt.model import TtQwen36Model
from models.demos.qwen36_27b.tt.generator import Qwen36Generator
from models.demos.qwen36_27b.tt import deltanet as dn

MP = "/home/yito/work/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def greedy_continue_from_decode(gen, cfg, first_tok, n, tag=""):
    out = [first_tok]
    nt = first_tok
    for i in range(n):
        print(f"    {tag} decode step {i+1}/{n}...", flush=True)
        _, ntt = gen.decode_one_token(torch.tensor([[nt]], dtype=torch.long))
        nt = ntt.item()
        out.append(nt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("-n", type=int, default=8)
    args = ap.parse_args()

    print(f"[kernel] FUSED={dn.USE_FUSED_KERNEL} FULL={dn.USE_FULL_FUSED_KERNEL} PREFILL={dn.USE_PREFILL_FUSED_KERNEL}")
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

        ids = torch.tensor([tok.encode(args.prompt, add_special_tokens=False)], dtype=torch.long)
        print(f"prompt={args.prompt!r}  ids={ids.tolist()}", flush=True)

        # --- Method A: prefill ---
        gen.reset()
        print("  [A] running prefill...", flush=True)
        logitsA = ttnn.to_torch(gen.prefill(ids)).float().reshape(-1)
        tokA = torch.argmax(logitsA[:cfg.vocab_size]).item()
        print(f"  [A] prefill next={tokA!r} {tok.decode([tokA])!r}", flush=True)
        contA = greedy_continue_from_decode(gen, cfg, tokA, args.n, tag="[A]")
        print(f"\n[A prefill]    next={tokA!r} {tok.decode([tokA])!r}", flush=True)
        print(f"[A continue]   {tok.decode(contA)!r}", flush=True)

        # --- Method B: token-by-token decode over the prompt ---
        gen.reset()
        print("  [B] running sequential decode over prompt...", flush=True)
        all_logits = gen.get_logits_for_sequence(ids)  # [S, vocab]
        tokB = torch.argmax(all_logits[-1][:cfg.vocab_size]).item()
        print(f"  [B] seq next={tokB!r} {tok.decode([tokB])!r}", flush=True)
        # gen.position is now S, deltanet state advanced over prompt -> continue
        contB = greedy_continue_from_decode(gen, cfg, tokB, args.n, tag="[B]")
        print(f"\n[B seq-decode] next={tokB!r} {tok.decode([tokB])!r}")
        print(f"[B continue]   {tok.decode(contB)!r}")

        # top-5 of each for visibility
        def top5(lg):
            v, i = torch.topk(lg[:cfg.vocab_size], 5)
            return [(int(ix), tok.decode([int(ix)]), round(float(vv), 2)) for vv, ix in zip(v, i)]
        print(f"\n[A top5] {top5(logitsA)}")
        print(f"[B top5] {top5(all_logits[-1])}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
