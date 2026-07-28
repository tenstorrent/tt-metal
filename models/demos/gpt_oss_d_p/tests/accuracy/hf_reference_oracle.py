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
        "--prompt-file",
        type=str,
        default=None,
        help="read a single prompt from a file (mutually exclusive with --prompt)",
    )
    ap.add_argument(
        "--num-tokens",
        type=int,
        default=None,
        help="truncate the tokenized prompt to this many tokens before running the model",
    )
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
        "--zero-sinks",
        action="store_true",
        help="DIAGNOSTIC: zero every attention layer's `sinks` parameter before running "
        "the reference forward pass.  Matches the --zero-sinks flag on kv_cache_prefill.py.",
    )
    ap.add_argument(
        "--disable-sliding-window",
        action="store_true",
        help="DIAGNOSTIC: force every attention layer to use full causal attention "
        "in the reference (no sliding window).  Matches --disable-sliding-window on "
        "kv_cache_prefill.py.",
    )
    ap.add_argument(
        "--gen-tokens",
        type=int,
        default=0,
        help="after the initial forward, autoregressively generate this many tokens "
        "(greedy) reusing past_key_values.  For each generated position, save the "
        "argmax id/text and top-5 (id, text, logit) into rec['generation'].  Used by "
        "decode_from_kv.py to compare TT decode outputs against this HF ground truth.",
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
    if args.prompt_file is not None and args.prompt:
        ap.error("--prompt-file and --prompt are mutually exclusive")
    if args.prompt_file is not None:
        with open(args.prompt_file) as f:
            prompts = [f.read()]
    else:
        prompts = args.prompt or ["What are the prime factors of 1?"]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

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

    if args.zero_sinks:
        n_zeroed = 0
        with torch.no_grad():
            for layer in model.model.layers:
                attn = layer.self_attn
                if hasattr(attn, "sinks") and attn.sinks is not None:
                    attn.sinks.zero_()
                    n_zeroed += 1
        print(f"[oracle] DIAGNOSTIC: zeroed attention sinks on {n_zeroed} layer(s)", flush=True)

    if args.disable_sliding_window:
        # Flip every layer to full_attention: rewrite both the module-level
        # sliding_window attribute (used by HF's eager/sdpa attention) and the
        # config.layer_types list (used to select the attention implementation).
        n_disabled = 0
        if hasattr(model.config, "layer_types") and model.config.layer_types is not None:
            model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
        for layer in model.model.layers:
            attn = layer.self_attn
            if hasattr(attn, "sliding_window"):
                attn.sliding_window = None
            if hasattr(attn, "is_sliding"):
                attn.is_sliding = False
            n_disabled += 1
        print(f"[oracle] DIAGNOSTIC: disabled sliding window on {n_disabled} layer(s)", flush=True)

    class FullKVCapture(DynamicCache):
        """DynamicCache subclass that snapshots the full pre-truncation K,V per layer.

        GPT-OSS uses sliding_window=128 for some layers, so past_key_values only retains
        the last 127 positions after a long prefill.  By intercepting update() here we
        capture the complete [1, num_kv_heads, seq_len, head_dim] tensors before the
        sliding-window layer discards earlier positions, while still returning the
        truncated view so that attention computation remains correct.
        """

        def __init__(self, config):
            super().__init__(config=config)
            self.full_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            self.full_kv[layer_idx] = (key_states.detach().cpu(), value_states.detach().cpu())
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

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

        if args.num_tokens is not None:
            ids = ids[: args.num_tokens]

        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        last = len(ids) - 1
        key = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        prompt_preview = repr(prompt[:80] + "..." if len(prompt) > 80 else prompt)
        print(f"[oracle] prompt={prompt_preview} n_tokens={len(ids)} key={key} -> forward (slow) ...", flush=True)

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
        layer0_position_embeddings: list[tuple] = []
        if args.check_kv_rope:

            def _capture_l0_attn_input(module, args, kwargs):
                if args:
                    hs = args[0]
                else:
                    hs = kwargs.get("hidden_states")
                if hs is None:
                    raise RuntimeError("Could not capture hidden_states from self_attn pre-hook")
                layer0_attn_input.append(hs.detach())
                pe = kwargs.get("position_embeddings")
                if pe is not None:
                    layer0_position_embeddings.append((pe[0].detach(), pe[1].detach()))

            _h0 = model.model.layers[0].self_attn.register_forward_pre_hook(_capture_l0_attn_input, with_kwargs=True)

        capture_cache = FullKVCapture(config=model.config) if args.save_kv else None
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True, past_key_values=capture_cache)

        for h in hooks:
            h.remove()

        if args.check_kv_rope:
            _h0.remove()
            attn0 = model.model.layers[0].self_attn
            hs_in = layer0_attn_input[0]  # [1, seq_len, hidden_size], bfloat16
            with torch.no_grad():
                k_pre = torch.nn.functional.linear(
                    hs_in,
                    attn0.k_proj.weight,
                    attn0.k_proj.bias,
                ).squeeze(
                    0
                )  # [seq_len, num_kv_heads * head_dim]
            head_dim = attn0.head_dim
            num_kv = attn0.k_proj.weight.shape[0] // head_dim
            k_pre = k_pre.reshape(-1, num_kv, head_dim).permute(1, 0, 2)  # [num_kv, seq, head_dim]

            # Apply RoPE manually to get post-RoPE K.
            k_post = None
            if layer0_position_embeddings:
                rope_cos, rope_sin = layer0_position_embeddings[0]  # [1, seq, rope_dim]
                rope_cos = rope_cos.squeeze(0)  # [seq, rope_dim]
                rope_sin = rope_sin.squeeze(0)
                # Expand to cover all kv heads.
                rope_cos = rope_cos.unsqueeze(0).expand(num_kv, -1, -1)  # [num_kv, seq, rope_dim]
                rope_sin = rope_sin.unsqueeze(0).expand(num_kv, -1, -1)
                rope_dim = rope_cos.shape[-1]
                k_rope_part = k_pre[..., :rope_dim]
                k_pass_part = k_pre[..., rope_dim:]
                # Standard rotate_half RoPE.
                half = rope_dim // 2
                k_rot = torch.cat([-k_rope_part[..., half:], k_rope_part[..., :half]], dim=-1)
                k_rope_rotated = k_rope_part * rope_cos + k_rot * rope_sin
                k_post = torch.cat([k_rope_rotated, k_pass_part], dim=-1)

            k_cached = out.past_key_values.layers[0].keys[0]  # [num_kv, seq, head_dim]

            def _cosim(a, b):
                a, b = a.float().flatten(), b.float().flatten()
                return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))

            cos_pre = _cosim(k_pre, k_cached)
            print(f"[oracle] cosine_sim(pre_rope_K,  cached_K) = {cos_pre:.4f}", flush=True)
            if k_post is not None:
                cos_post = _cosim(k_post, k_cached)
                print(f"[oracle] cosine_sim(post_rope_K, cached_K) = {cos_post:.4f}", flush=True)
                if cos_post > cos_pre:
                    print("[oracle] => cached K is POST-RoPE (post_rope closer to cached)", flush=True)
                else:
                    print("[oracle] => cached K is PRE-RoPE (pre_rope closer to cached)", flush=True)
            else:
                print("[oracle] => position_embeddings not captured; cannot compute post-RoPE comparison", flush=True)
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
        if args.save_kv and capture_cache is not None:
            # Use full_kv captured before any sliding-window truncation.
            # GPT-OSS has sliding_window=128 for some layers; past_key_values for those
            # layers only retains the last 127 positions after a long prefill, while the
            # TT model fills its cache with all positions.  capture_cache.full_kv holds
            # the complete post-RoPE K,V tensors ([1, num_kv_heads, seq_len, head_dim])
            # for every layer, captured in FullKVCapture.update() before truncation.
            for i in range(num_capture):
                if i not in capture_cache.full_kv:
                    break
                k_full, v_full = capture_cache.full_kv[i]
                # k_full: [1, num_kv_heads, seq_len, head_dim]
                k_float = k_full[0].float()  # [num_kv_heads, seq_len, head_dim]
                v_float = v_full[0].float()
                k_fname = f"kv_k_{i}_{key}.npy"
                v_fname = f"kv_v_{i}_{key}.npy"
                np.save(outdir / k_fname, k_float.numpy())
                np.save(outdir / v_fname, v_float.numpy())
                kv_files[str(i)] = {"k": k_fname, "v": v_fname}
            print(f"[oracle] saved {len(kv_files)} KV cache layer(s)", flush=True)

        # Generation must come AFTER --save-kv so capture_cache.full_kv still
        # reflects only prefill positions.  During autoregressive steps HF will
        # call FullKVCapture.update() with new K/V for each generated position,
        # which would overwrite the prefill snapshot.
        generation: list[dict] = []
        if args.gen_tokens > 0:
            # generation[i] is the token at position (last+1+i) in the extended
            # sequence.  Element 0 is the argmax of the last-prompt-position
            # logits (already computed above); we prepend it so the list is the
            # full generated stream starting from the first output token.
            generation.append(
                {
                    "pos": last + 1,
                    "argmax_id": argmax,
                    "argmax_text": tok.decode([argmax]),
                    "top5": [{"id": i, "text": t, "logit": l} for i, t, l in top5],
                }
            )
            gen_cache = out.past_key_values
            next_tok = argmax
            print(f"[oracle] generating {args.gen_tokens} tokens greedily...", flush=True)
            with torch.no_grad():
                for step in range(1, args.gen_tokens):
                    step_out = model(
                        input_ids=torch.tensor([[next_tok]], dtype=torch.long),
                        past_key_values=gen_cache,
                        use_cache=True,
                    )
                    step_logits = step_out.logits[0, -1, :].float()
                    step_top = torch.topk(step_logits, 5)
                    step_top5 = [
                        {"id": int(i), "text": tok.decode([int(i)]), "logit": float(step_logits[int(i)])}
                        for i in step_top.indices
                    ]
                    step_argmax = int(step_top.indices[0])
                    generation.append(
                        {
                            "pos": last + 1 + step,
                            "argmax_id": step_argmax,
                            "argmax_text": tok.decode([step_argmax]),
                            "top5": step_top5,
                        }
                    )
                    next_tok = step_argmax
                    gen_cache = step_out.past_key_values
            print(
                f"[oracle] generated: {tok.decode([g['argmax_id'] for g in generation])!r}",
                flush=True,
            )

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
        if generation:
            rec["generation"] = generation
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
