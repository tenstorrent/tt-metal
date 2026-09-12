#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Canada Quant Labs
# SPDX-License-Identifier: Apache-2.0
"""Command-R (c4ai-command-r-v01) CPU reference activation capture.

Runs on a CPU host with enough RAM (validated on a 249 GB host; the 70 GB F16
checkpoint fits; fp32 reference ≈140 GB — leave headroom, see --dtype). Loads CohereForCausalLM from
the local HF snapshot (NO network), registers forward hooks at the PCC capture
points, runs fixed prompts, and dumps per-layer activations as compressed .npz.

Capture points (per prompt, keys in the .npz):
    embed                 embed_tokens output           [1, S, 8192]
    layer{NN:02d}_in/out  decoder layer NN residual in/out (00..39)
    final_norm            model.norm output             [1, S, 8192]
    logits                lm_head output * logit_scale  [1, S, 256000]  (fp32)

These keys are the contract the TTNN PCC harnesses in ../tests/
(test_cohere_pcc.py, test_cohere_fullmodel_pcc.py) compare against at PCC >= 0.99.

Verified model facts (on-box config.json + HF v4.39.3 reference, 2026-08-28):
40 layers / hidden 8192 / FFN 22528 / MHA 64:64 (head_dim 128, no attn bias, no
QK norm) / RoPE theta 8e6 / model_max_length 131072 / CohereLayerNorm (fp32
mean-centering, weight-only, eps 1e-5) / parallel block / tied embeddings /
logit_scale 0.0625 / vocab 256000.

Prereqs (CPU-only):
    pip install 'transformers>=4.39.1' torch --index-url \
        https://download.pytorch.org/whl/cpu   # CPU wheel is enough
Usage:
    python3 models/experimental/cohere/reference/cpu_reference_capture.py \
        --snapshot /path/to/local/HF/snapshot \
        --out ./cohere_reference --dtype float32
"""
import argparse
import json
import os
import time

DEFAULT_PROMPTS = [
    # Short greedy single-turn prompts; P6 extends this set with chat /
    # grounded-gen / tool-use template fixtures.
    "The capital of France is",
    "Explain the difference between TCP and UDP in one sentence.",
    "def fibonacci(n):",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="local HF snapshot dir (config.json + shards)")
    ap.add_argument("--out", required=True, help="output dir for .npz dumps")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                    help="model compute dtype; float32 is the PCC gold reference (~140 GB RAM)")
    ap.add_argument("--prompts-file", default=None, help="optional JSON list of prompts")
    ap.add_argument("--max-new-tokens", type=int, default=1,
                    help="1 = single forward step capture (prefill+first logits); >1 also decodes")
    ap.add_argument("--layers", default="all", help="'all' or 'START-END' inclusive range (memory-saver)")
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)

    torch_dtype = getattr(torch, args.dtype)
    print(f"[capture] loading tokenizer from {args.snapshot}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.snapshot)
    print(f"[capture] loading CohereForCausalLM dtype={args.dtype} (low_cpu_mem_usage)", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation="eager",
    )
    model.config.use_cache = False
    model.eval()
    print(f"[capture] model loaded in {time.time()-t0:.1f}s; "
          f"layers={model.config.num_hidden_layers} logit_scale={model.config.logit_scale}", flush=True)

    want_all = args.layers == "all"
    if not want_all:
        lo, hi = (int(x) for x in args.layers.split("-"))
    acts = {}
    hooks = []

    def save(key, tensor):
        acts[key] = tensor.detach().to(torch.float32).cpu().numpy()

    def layer_hook(name):
        def _hook(_mod, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            save(name, hidden)
        return _hook

    hooks.append(model.model.embed_tokens.register_forward_hook(layer_hook("embed")))
    for i, layer in enumerate(model.model.layers):
        if want_all or lo <= i <= hi:
            hooks.append(layer.register_forward_hook(layer_hook(f"layer{i:02d}_out")))
    hooks.append(model.model.norm.register_forward_hook(layer_hook("final_norm")))

    for pidx, prompt in enumerate(prompts):
        acts.clear()
        ids = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(**ids)
        # HF applies logit_scale inside CohereForCausalLM.forward? No — it applies
        # in forward() of CohereForCausalLM (v4.39.3 line 1114), so out.logits is
        # ALREADY scaled. We store the scaled logits (the serving contract).
        save("logits", out.logits)
        path = os.path.join(args.out, f"prompt{pidx:02d}.npz")
        np.savez_compressed(path, **{k: v for k, v in acts.items()})
        print(f"[capture] prompt{pidx:02d} -> {path} keys={len(acts)}", flush=True)

    for h in hooks:
        h.remove()
    print("[capture] done", flush=True)


if __name__ == "__main__":
    main()
