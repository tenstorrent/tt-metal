# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
HF-reference oracle for GPT-OSS — ground truth for verifying the TT device prefill run.

Runs the bf16 checkpoint through HuggingFace on CPU for one or more prompts and saves,
per prompt: the token ids, the last-position argmax + top-5 (decoded), and the full
last-position logit vector (fp32 .npy). Compare a TT device run's logits against these
for argmax-match + logit PCC.

With --save-layers, also captures the hidden-state output of each decoder layer (the
post-residual tensor after attention + MLP) as float32 .npy files.  These are used by
layer_activations_prefill.py to do per-layer PCC comparisons against the TT model.

CPU forward of the full 120B model is I/O-bound — expect several minutes per prompt.
Run once; results persist in --out.

Run (logits only):
  cd /path/to/tt-metal
  export HF_MODEL=/data/jmalone/.cache/huggingface/hub/models--openai--gpt-oss-120b/gpt-oss-120b
  python3 models/demos/gpt_oss_d_p/tests/accuracy/hf_reference_oracle.py \\
      --out /data/jmalone/gpt_oss_ref \\
      --prompt "What are the prime factors of 1?"

Run (logits + per-layer activations for first 5 layers):
  python3 models/demos/gpt_oss_d_p/tests/accuracy/hf_reference_oracle.py \\
      --out /data/jmalone/gpt_oss_ref \\
      --prompt "What are the prime factors of 1?" \\
      --save-layers --max-layers 5
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
    ap.add_argument("--out", type=str, default="/data/jmalone/gpt_oss_ref")
    ap.add_argument("--prompt", action="append", help="repeatable; defaults to one prompt")
    ap.add_argument(
        "--chat-template",
        action="store_true",
        help="apply chat template (off by default; GPT-OSS demos use plain tokenization)",
    )
    ap.add_argument(
        "--save-layers",
        action="store_true",
        help="also save per-layer hidden-state activations for layer_activations_prefill.py",
    )
    ap.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="limit layer capture to first N layers (default: all layers)",
    )
    args = ap.parse_args()
    prompts = args.prompt or ["What are the prime factors of 1?"]

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

    num_model_layers = len(model.model.layers)
    if args.save_layers:
        num_capture = args.max_layers if args.max_layers is not None else num_model_layers
        num_capture = min(num_capture, num_model_layers)
        print(f"[oracle] will capture hidden states for layers 0..{num_capture - 1}", flush=True)

    # Load any existing results so we can append without overwriting.
    results_path = outdir / "ref_results.json"
    if results_path.exists():
        with results_path.open() as f:
            existing = json.load(f)
        existing_by_key = {hashlib.sha256(r["prompt"].encode()).hexdigest()[:12]: r for r in existing}
    else:
        existing = []
        existing_by_key = {}

    results = list(existing)
    for prompt in prompts:
        if args.chat_template:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
            )
        else:
            ids = tok(prompt)["input_ids"]

        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        last = len(ids) - 1
        key = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        print(f"[oracle] prompt={prompt!r} n_tokens={len(ids)} key={key} -> forward (slow) ...", flush=True)

        layer_outputs: dict[int, np.ndarray] = {}
        hooks = []
        if args.save_layers:

            def make_hook(idx):
                def hook(module, inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    # Save as float32; shape [1, seq_len, hidden_size]
                    layer_outputs[idx] = hs.detach().float().numpy()

                return hook

            for i in range(num_capture):
                hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))

        with torch.no_grad():
            out = model(input_ids=input_ids)

        for h in hooks:
            h.remove()

        logits = out.logits[0, last, :].float()  # [vocab]
        top = torch.topk(logits, 5)
        top5 = [(int(i), tok.decode([int(i)]), float(logits[int(i)])) for i in top.indices]
        argmax = int(top.indices[0])

        np.save(outdir / f"logits_{key}.npy", logits.numpy())

        layer_files: dict[str, str] = {}
        for i, hs in layer_outputs.items():
            fname = f"layer_{i}_{key}.npy"
            np.save(outdir / fname, hs)
            layer_files[str(i)] = fname
        if layer_files:
            print(f"[oracle] saved {len(layer_files)} layer activation file(s)", flush=True)

        rec = {
            "prompt": prompt,
            "chat_template": args.chat_template,
            "token_ids": list(map(int, ids)),
            "n_tokens": len(ids),
            "last_pos": last,
            "argmax_id": argmax,
            "argmax_text": tok.decode([argmax]),
            "top5": [{"id": i, "text": t, "logit": l} for i, t, l in top5],
            "logits_file": f"logits_{key}.npy",
            "vocab_size": int(logits.shape[0]),
        }
        if layer_files:
            rec["layer_files"] = layer_files

        # Replace existing record for the same prompt, or append.
        if key in existing_by_key:
            results = [rec if hashlib.sha256(r["prompt"].encode()).hexdigest()[:12] == key else r for r in results]
        else:
            results.append(rec)

        print(f"[oracle] argmax={argmax} {tok.decode([argmax])!r}  top5={[(i, t) for i, t, _ in top5]}", flush=True)

    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[oracle] saved {len(results)} record(s) to {results_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
