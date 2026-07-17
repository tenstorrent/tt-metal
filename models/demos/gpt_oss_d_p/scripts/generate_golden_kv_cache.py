#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate a golden KV cache for GPT-OSS prefill validation.

Adapted from ``minimax_m3/scripts/generate_golden_kv_cache.py``. GPT-OSS is a stock HuggingFace
model (``GptOssForCausalLM``), so — unlike M3 — the reference is just an HF forward pass with
``use_cache=True``: HF stores exactly what we want in ``past_key_values`` — the **post-RoPE K** and
**raw V** per layer, shaped ``[1, num_kv_heads, seq_len, head_dim]`` (GQA; there is NO sparse
lightning indexer, so NO ``index_k``).

We may instead consume goldens produced by another team (e.g. Lukasz). Either way the on-disk trace
format below is the contract the KV-PCC harness (``tests/galaxy_prefill_kv_pcc.py`` /
``TtPrefillRuntime.kv_cache_pcc_check``) reads.

Output format (identical to M3 minus index_k):
    {trace_dir}/
        metadata.json           - prompt, token_ids, model info (n_tokens, num_layers, num_kv_heads, head_dim)
        kv_cache/
            layer_0.safetensors - key_cache_layer_0  [1, num_kv_heads, seq_len, head_dim]  (post-RoPE K, HF layout)
            layer_1.safetensors   value_cache_layer_0 [1, num_kv_heads, seq_len, head_dim]  (raw V)
            ...                   (NO index_k_cache_* — GPT-OSS is dense GQA)

  K CONVENTION: HF stores K in its "half-split" RoPE convention. The device stores K Meta-RoPE
  swizzled over the head_dim, so the PCC harness permutes the golden's rotary slice HF->Meta before
  comparing (see kv_cache_pcc_check). V is raw and compared directly.

Usage (from tt-metal root):
    export HF_MODEL=/path/to/gpt-oss-120b   # or gpt-oss-20b for a smaller golden
    python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \\
        --prompt "The capital of France is" --out /tmp/gpt_oss_golden
    python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \\
        --prompt-json prompt.json --out /mnt/models/gpt-oss-cache/golden/longbook_8192 --max-tokens 8192
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate golden KV cache for GPT-OSS prefill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    prompt_group = ap.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-json", type=Path, help='JSON file with {"prompt": "..."}')
    prompt_group.add_argument("--prompt", type=str, help="Direct prompt text")
    ap.add_argument("--out", type=Path, required=True, help="Output trace directory (creates kv_cache/ subdir)")
    ap.add_argument("--model-path", type=str, default=None, help="HF model dir (default: $HF_MODEL)")
    ap.add_argument("--max-tokens", type=int, default=None, help="Truncate prompt to this many tokens")
    ap.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Use raw prompt (no chat template). MUST match the tokenization the tests use.",
    )
    ap.add_argument("--dtype", choices=["bfloat16", "float32", "float16"], default="bfloat16", help="Stored KV dtype")
    ap.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float32", "float16"],
        default="float32",
        help="Model compute dtype for the reference forward (default: float32 high-precision golden)",
    )
    ap.add_argument("--device", type=str, default="cpu", help="torch device for the reference forward (cpu/cuda)")
    return ap.parse_args()


def load_prompt(args) -> str:
    if args.prompt_json:
        with open(args.prompt_json) as f:
            data = json.load(f)
        if isinstance(data, dict) and "prompt" in data:
            return data["prompt"]
        if isinstance(data, str):
            return data
        raise ValueError(f"{args.prompt_json}: expected dict with 'prompt' key or string")
    return args.prompt


def tokenize_prompt(tokenizer, prompt, max_tokens, use_chat_template):
    if use_chat_template:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        )
    else:
        ids = tokenizer(prompt)["input_ids"]
    if max_tokens and len(ids) > max_tokens:
        print(f"[tokenize] truncating {len(ids)} tokens -> {max_tokens}")
        ids = ids[:max_tokens]
    return ids, len(ids)


def _legacy_kv(past_key_values):
    """Return a list of (key, value) tuples per layer, whatever cache type HF returned.

    Newer transformers hand back a ``Cache`` object (DynamicCache); older ones the legacy tuple.
    """
    if past_key_values is None:
        raise RuntimeError("model returned no past_key_values — run with use_cache=True")
    if hasattr(past_key_values, "to_legacy_cache"):
        return list(past_key_values.to_legacy_cache())
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    return list(past_key_values)  # already a tuple of (k, v)


def main():
    args = parse_args()
    model_path = args.model_path or os.environ.get("HF_MODEL")
    if not model_path:
        print("ERROR: Must provide --model-path or set $HF_MODEL", file=sys.stderr)
        return 1

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
    store_dtype = dtype_map[args.dtype]
    compute_dtype = dtype_map[args.compute_dtype]

    out_dir = args.out
    kv_cache_dir = out_dir / "kv_cache"
    kv_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] loading tokenizer from {model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompt = load_prompt(args)
    use_chat_template = not args.no_chat_template
    token_ids, seq_len = tokenize_prompt(tokenizer, prompt, args.max_tokens, use_chat_template)
    print(f"[load] tokenized to {seq_len} tokens (chat_template={use_chat_template})", flush=True)

    print(f"[load] building GptOssForCausalLM ({model_path}, compute_dtype={compute_dtype})...", flush=True)
    torch.set_num_threads(os.cpu_count() or 32)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=compute_dtype, trust_remote_code=True)
    model = model.to(args.device).eval()
    cfg = getattr(model.config, "text_config", model.config)
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print(f"[load] ready — {num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}", flush=True)

    input_ids = torch.tensor(token_ids, dtype=torch.long, device=args.device).unsqueeze(0)

    print(f"[forward] prefill forward for {seq_len} tokens (use_cache=True) ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    kv = _legacy_kv(out.past_key_values)
    forward_time = time.time() - t0
    print(f"[forward] done in {forward_time:.1f}s ({seq_len / max(forward_time, 1e-9):.2f} tok/s)", flush=True)

    expected_shape = (1, num_kv_heads, seq_len, head_dim)
    saved = {}
    for layer_idx, (k, v) in enumerate(kv):
        k = k.detach().float().to(store_dtype).contiguous()
        v = v.detach().float().to(store_dtype).contiguous()
        if tuple(k.shape) != expected_shape or tuple(v.shape) != expected_shape:
            print(
                f"[save] WARNING: layer {layer_idx} KV shape K={tuple(k.shape)} V={tuple(v.shape)} "
                f"!= expected {expected_shape}",
                flush=True,
            )
        save_file(
            {f"key_cache_layer_{layer_idx}": k, f"value_cache_layer_{layer_idx}": v},
            str(kv_cache_dir / f"layer_{layer_idx}.safetensors"),
        )
        saved["key_shape"] = list(k.shape)
        saved["value_shape"] = list(v.shape)
    print(f"[save] wrote {len(kv)} layer files to {kv_cache_dir}/", flush=True)

    metadata = {
        "model_path": str(model_path),
        "reference": "transformers.GptOssForCausalLM forward (use_cache) — post-RoPE K + raw V",
        "prompt_source": str(args.prompt_json) if args.prompt_json else "direct",
        "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
        "prompt_length_chars": len(prompt),
        "token_ids": token_ids,
        "n_tokens": seq_len,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": args.dtype,
        "compute_dtype": args.compute_dtype,
        "chat_template": use_chat_template,
        "forward_time_seconds": forward_time,
        "kv_cache_format": "separate_k_v",  # key_cache_layer_N / value_cache_layer_N, NO index_k (GQA)
        "key_cache_shape": saved.get("key_shape"),  # [1, num_kv_heads, seq_len, head_dim]
        "value_cache_shape": saved.get("value_shape"),
        "index_k_cache_shape": None,  # GPT-OSS is dense GQA — no sparse index key
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Golden KV cache written to {out_dir}", flush=True)
    print(f"   Use in tests: export PREFILL_TRACE_DIR={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
