# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Pinpoint which fused DeltaNet kernel breaks accuracy, building the model ONCE
and switching kernel paths by monkey-patching the dn.* module flags.

Prefill is kept on the CPU path for ALL configs (the fused-prefill kernel
wedged a board on short prompts), so this isolates the DECODE kernels:
  - cfg1: full=False -> partial-fused decode (deltanet_decode kernel)   [known-good]
  - cfg2: full=True  -> full-fused decode    (deltanet_decode_full)     [suspect]

Reports next-token, an 8-token greedy continuation, and decode tok/s for each.
"""
import argparse
import faulthandler
import time

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


def run_cfg(gen, cfg, tok, ids, name, full_decode, n=8):
    # Prefill always CPU (safe); toggle only the decode kernel.
    dn.USE_PREFILL_FUSED_KERNEL = False
    dn.USE_FULL_FUSED_KERNEL = full_decode
    dn.USE_FUSED_KERNEL = True
    gen.reset()
    print(f"\n--- {name}: USE_FULL_FUSED_KERNEL={dn.USE_FULL_FUSED_KERNEL} (prefill=CPU) ---", flush=True)
    logits = ttnn.to_torch(gen.prefill(ids)).float().reshape(-1)
    nt = torch.argmax(logits[:cfg.vocab_size]).item()
    out = [nt]
    t0 = time.perf_counter()
    for _ in range(n):
        _, ntt = gen.decode_one_token(torch.tensor([[nt]], dtype=torch.long))
        nt = ntt.item()
        out.append(nt)
    dt = time.perf_counter() - t0
    print(f"  prefill_next={tok.decode([out[0]])!r}", flush=True)
    print(f"  continue={tok.decode(out)!r}", flush=True)
    print(f"  decode={n/dt:.2f} tok/s ({dt:.2f}s for {n} tok)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()
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
        print(f"prompt={args.prompt!r} ids={ids.tolist()}", flush=True)

        run_cfg(gen, cfg, tok, ids, "cfg1 partial-fused decode", full_decode=False)
        run_cfg(gen, cfg, tok, ids, "cfg2 full-fused decode", full_decode=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
