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

With --save-kv, captures the K and V tensors from past_key_values for each decoder
layer as float32 .npy files (shape [num_kv_heads, seq_len, head_dim]).  These are used
by kv_cache_prefill.py to do per-layer KV cache PCC comparisons against the TT model.

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

Run (logits + per-layer KV cache for first 5 layers):
  python3 models/demos/gpt_oss_d_p/tests/accuracy/hf_reference_oracle.py \\
      --out /data/jmalone/gpt_oss_ref \\
      --prompt "What are the prime factors of 1?" \\
      --save-kv --max-layers 5
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
        "--save-kv",
        action="store_true",
        help="also save per-layer K and V cache tensors for kv_cache_prefill.py",
    )
    ap.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="limit layer capture to first N layers (default: all layers)",
    )
    ap.add_argument(
        "--check-kv-rope",
        action="store_true",
        help=(
            "diagnostic: for layer 0, compute K manually from the weight projection and compare "
            "against past_key_values[0][0] to determine whether HF stores pre- or post-RoPE K. "
            "Prints cosine similarity; exits after the check."
        ),
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
    num_capture = args.max_layers if args.max_layers is not None else num_model_layers
    num_capture = min(num_capture, num_model_layers)
    if args.save_layers:
        print(f"[oracle] will capture hidden states for layers 0..{num_capture - 1}", flush=True)
    if args.save_kv:
        print(f"[oracle] will capture KV cache for layers 0..{num_capture - 1}", flush=True)

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

        # Capture the post-layernorm hidden state that feeds into layer-0 self_attn,
        # so we can manually reproduce pre-RoPE K and compare against past_key_values.
        layer0_attn_input: list[torch.Tensor] = []
        if args.check_kv_rope:

            def _capture_l0_attn_input(module, args, kwargs=None):
                if args:
                    hs = args[0]
                else:
                    hs = (kwargs or {}).get("hidden_states")
                if hs is None:
                    raise RuntimeError("Could not capture hidden_states from self_attn pre-hook")
                layer0_attn_input.append(hs.detach().float())

            _h0 = model.model.layers[0].self_attn.register_forward_pre_hook(_capture_l0_attn_input)

        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)

        for h in hooks:
            h.remove()

        if args.check_kv_rope:
            _h0.remove()
            # Manually compute pre-RoPE K for layer 0.
            attn0 = model.model.layers[0].self_attn
            hs_in = layer0_attn_input[0]  # [1, seq_len, hidden_size], float32 (post-layernorm)
            with torch.no_grad():
                k_manual = (
                    torch.nn.functional.linear(
                        hs_in.bfloat16(),
                        attn0.k_proj.weight,
                        attn0.k_proj.bias,
                    )
                    .float()
                    .squeeze(0)
                )  # [seq_len, num_kv_heads * head_dim]
            head_dim = attn0.head_dim
            num_kv = attn0.num_key_value_heads
            k_manual = k_manual.reshape(-1, num_kv, head_dim).permute(1, 0, 2)  # [num_kv, seq, head_dim]

            k_cached = out.past_key_values[0][0][0].float()  # [num_kv, seq, head_dim]

            def _cosim(a, b):
                a, b = a.flatten(), b.flatten()
                return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))

            cos = _cosim(k_manual, k_cached)
            print(
                f"[oracle] --check-kv-rope layer 0: cosine_sim(manual_pre_rope_K, past_key_values_K) = {cos:.4f}",
                flush=True,
            )
            if cos > 0.9:
                print("[oracle] => past_key_values stores PRE-RoPE K (cos≈1 means no rotation applied yet)", flush=True)
            elif cos < 0.1:
                print("[oracle] => past_key_values stores POST-RoPE K (cos≈0 means K has been rotated)", flush=True)
            else:
                print(f"[oracle] => ambiguous (cos={cos:.4f}); investigate further", flush=True)
            return 0

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

        kv_files: dict[str, dict[str, str]] = {}
        if args.save_kv and out.past_key_values is not None:
            # SANITY CHECK: save pre-RoPE K to match TT fill_cache (which also now saves
            # pre-RoPE K).  Revert both sides once QKV/fill correctness is confirmed.
            for i, layer_kv in enumerate(out.past_key_values):
                if i >= num_capture:
                    break
                k, v = layer_kv[0], layer_kv[1]
                # k: [batch, num_kv_heads, seq_len, head_dim] bfloat16, pre-RoPE
                k_float = k[0].float()  # [num_kv_heads, seq_len, head_dim]
                v_float = v[0].float()  # V is not RoPE-rotated
                k_fname = f"kv_k_{i}_{key}.npy"
                v_fname = f"kv_v_{i}_{key}.npy"
                np.save(outdir / k_fname, k_float.numpy())
                np.save(outdir / v_fname, v_float.numpy())
                kv_files[str(i)] = {"k": k_fname, "v": v_fname}
            print(f"[oracle] saved {len(kv_files)} KV cache layer(s)", flush=True)

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
        if kv_files:
            rec["kv_files"] = kv_files

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
