# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
HF-reference oracle for MiniMax-M2 — ground truth for verifying the TT device run.

Runs the dequantized bf16 checkpoint through HuggingFace on CPU for one or more
prompts and saves, per prompt: the token ids, the last-position argmax + top-5
(decoded), and the full last-position logit vector (fp32 .npy). Compare a TT
device run's logits against these for argmax-match + logit PCC.

CPU forward of the full 230B model is I/O-bound (mmaps ~426GB of weights) — expect
several to ~20 min per prompt. Run once; results persist.

Run:
  cd /data/vmelnykov/tt-metal
  export HF_MODEL=/data/vmelnykov/MiniMax-M2
  source python_env/bin/activate
  python3 models/demos/minimax_m2/tests/hf_reference_oracle.py \
      --out /data/vmelnykov/minimax_m2_ref --prompt "The capital of France is"
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="/data/vmelnykov/minimax_m2_ref")
    ap.add_argument("--prompt", action="append", help="repeatable; defaults to one prompt")
    ap.add_argument("--no-chat-template", action="store_true")
    args = ap.parse_args()
    prompts = args.prompt or ["The capital of France is"]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.environ["HF_MODEL"]
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(os.cpu_count() or 32)
    print(f"[oracle] loading HF model from {model_path} (bf16, cpu, mmap) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    print("[oracle] model loaded", flush=True)

    results = []
    for prompt in prompts:
        if args.no_chat_template:
            ids = tok(prompt)["input_ids"]
        else:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
            )
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        last = len(ids) - 1
        print(f"[oracle] prompt={prompt!r} n_tokens={len(ids)} -> forward (slow) ...", flush=True)
        with torch.no_grad():
            out = model(input_ids=input_ids)
        logits = out.logits[0, last, :].float()  # [vocab]
        top = torch.topk(logits, 5)
        top5 = [(int(i), tok.decode([int(i)]), float(logits[int(i)])) for i in top.indices]
        argmax = int(top.indices[0])

        key = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        np.save(outdir / f"logits_{key}.npy", logits.numpy())
        rec = {
            "prompt": prompt,
            "chat_template": not args.no_chat_template,
            "token_ids": list(map(int, ids)),
            "n_tokens": len(ids),
            "last_pos": last,
            "argmax_id": argmax,
            "argmax_text": tok.decode([argmax]),
            "top5": [{"id": i, "text": t, "logit": l} for i, t, l in top5],
            "logits_file": f"logits_{key}.npy",
            "vocab_size": int(logits.shape[0]),
        }
        results.append(rec)
        print(f"[oracle] argmax={argmax} {tok.decode([argmax])!r}  top5={[(i,t) for i,t,_ in top5]}", flush=True)

    with (outdir / "ref_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[oracle] saved {len(results)} record(s) to {outdir}/ref_results.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
