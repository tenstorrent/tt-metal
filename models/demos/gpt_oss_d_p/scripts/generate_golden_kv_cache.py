#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate golden KV cache for GPT-OSS prefill validation.

Runs the HF golden reference (models/demos/gpt_oss_d_p/reference/model.py) and extracts
per-layer post-RoPE K and raw V, saving in the separate K/V format used by the MiniMax-style
golden trace layout and ``scripts/verify_golden_kv.py``.

GPT-OSS uses alternating sliding/full attention; the reference captures full-length K/V via
``FullKVCapture`` before sliding-window truncation.

Memory approach:
- Model weights mmap'd via ``low_cpu_mem_usage=True``
- Computes in bfloat16 (matches device cache dtype)
- Saves layer-by-layer via streaming callback during the forward pass

Output format:
    {trace_dir}/
        metadata.json
        kv_cache/
            layer_0.safetensors - key_cache_layer_0 / value_cache_layer_0  [1, num_kv_heads, seq_len, head_dim]
            ...

Usage:
    export HF_MODEL=/path/to/gpt-oss-120b
    python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \\
        --prompt-json prompt.json \\
        --out /mnt/models/gpt-oss-cache/golden/longbook_full \\
        --max-tokens 56320

    python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \\
        --prompt "What are the prime factors of 1?" \\
        --out /tmp/gpt_oss_golden
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from models.demos.gpt_oss_d_p.reference.model import load_golden_model  # noqa: E402


def _raise_cpu_time_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
            print(f"[limit] raised RLIMIT_CPU soft {soft}s -> {hard}")
        except (ValueError, OSError) as e:
            print(
                f"[limit] WARNING: could not raise RLIMIT_CPU (soft={soft}s); "
                f"if the run dies with 'CPU time limit exceeded', run `ulimit -t unlimited` first: {e}",
                file=sys.stderr,
            )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate golden KV cache for GPT-OSS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    prompt_group = ap.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-json", type=Path, help='JSON file with {"prompt": "..."}')
    prompt_group.add_argument("--prompt", type=str, help="Direct prompt text")

    ap.add_argument("--out", type=Path, required=True, help="Output trace directory (creates kv_cache/)")
    ap.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="HF model directory (default: $HF_MODEL or $DEEPSEEK_V3_HF_MODEL)",
    )
    ap.add_argument("--max-tokens", type=int, default=None, help="Truncate prompt to this many tokens")
    ap.add_argument(
        "--chat-template",
        action="store_true",
        help="Apply chat template (off by default; GPT-OSS demos use plain tokenization)",
    )
    ap.add_argument(
        "--dtype",
        choices=["bfloat16", "float32", "float16"],
        default="bfloat16",
        help="Stored KV cache dtype (default: bfloat16)",
    )
    ap.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers to capture (default: all, 36 for gpt-oss-120b)",
    )
    ap.add_argument(
        "--zero-sinks",
        action="store_true",
        help="Zero attention sinks before forward (diagnostic; matches kv_cache_prefill.py --zero-sinks)",
    )
    ap.add_argument(
        "--disable-sliding-window",
        action="store_true",
        help="Force full causal attention on all layers (diagnostic)",
    )

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


def tokenize_prompt(tokenizer, prompt: str, max_tokens: int | None, use_chat_template: bool) -> tuple[list[int], int]:
    if use_chat_template:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    else:
        ids = tokenizer(prompt)["input_ids"]

    if max_tokens and len(ids) > max_tokens:
        print(f"[tokenize] truncating {len(ids)} tokens -> {max_tokens}")
        ids = ids[:max_tokens]

    return ids, len(ids)


def main():
    args = parse_args()
    _raise_cpu_time_limit()

    model_path = args.model_path or os.environ.get("HF_MODEL") or os.environ.get("DEEPSEEK_V3_HF_MODEL")
    if not model_path:
        print("ERROR: Must provide --model-path or set $HF_MODEL / $DEEPSEEK_V3_HF_MODEL", file=sys.stderr)
        return 1

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    kv_cache_dir = out_dir / "kv_cache"
    kv_cache_dir.mkdir(exist_ok=True)

    print("[load] reading prompt...", flush=True)
    prompt = load_prompt(args)
    print(f"[load] prompt length: {len(prompt)} characters", flush=True)

    print(f"[load] loading tokenizer from {model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    token_ids, seq_len = tokenize_prompt(tokenizer, prompt, args.max_tokens, args.chat_template)
    print(f"[load] tokenized to {seq_len} tokens (chat_template={args.chat_template})", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"[load] Building golden reference from {model_path}", flush=True)
    print(f"[load] compute dtype={dtype}; weights mmap'd (low_cpu_mem_usage)", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    torch.set_num_threads(os.cpu_count() or 32)

    t0 = time.time()
    model = load_golden_model(
        model_path,
        num_layers=args.num_layers,
        compute_dtype=dtype,
        zero_sinks=args.zero_sinks,
        disable_sliding_window=args.disable_sliding_window,
    )
    num_layers = model.cfg.num_hidden_layers
    num_kv_heads = model.cfg.num_key_value_heads
    head_dim = model.cfg.head_dim
    print(
        f"[load] reference ready in {time.time() - t0:.1f}s — "
        f"{num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}",
        flush=True,
    )
    if model.cfg.sliding_window and not args.disable_sliding_window:
        print(
            f"[forward] sliding_window={model.cfg.sliding_window} on alternating layers; "
            f"FullKVCapture preserves full-seq K/V",
            flush=True,
        )

    print(f"\n{'=' * 70}", flush=True)
    print(f"[forward] Running prefill forward pass for {seq_len} tokens", flush=True)
    print("[forward] WARNING: CPU inference is SLOW — can take many minutes for long prompts!", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    saved_key_shape: list[int] | None = None
    saved_val_shape: list[int] | None = None

    t0 = time.time()
    progress = tqdm(total=num_layers, desc="Prefill (save KV per layer)", unit="layer")

    def save_layer(layer_idx: int, key_cache: torch.Tensor, value_cache: torch.Tensor):
        nonlocal saved_key_shape, saved_val_shape
        key_cache = key_cache.to(dtype)
        value_cache = value_cache.to(dtype)
        expected_shape = (1, num_kv_heads, seq_len, head_dim)
        if tuple(key_cache.shape) != expected_shape or tuple(value_cache.shape) != expected_shape:
            tqdm.write(
                f"[save] WARNING: layer {layer_idx} KV shape mismatch! "
                f"K: {tuple(key_cache.shape)}, V: {tuple(value_cache.shape)}, expected: {expected_shape}"
            )
        tensors = {
            f"key_cache_layer_{layer_idx}": key_cache.contiguous(),
            f"value_cache_layer_{layer_idx}": value_cache.contiguous(),
        }
        save_file(tensors, str(kv_cache_dir / f"layer_{layer_idx}.safetensors"))
        saved_key_shape = list(key_cache.shape)
        saved_val_shape = list(value_cache.shape)
        progress.update(1)

    with torch.no_grad():
        model.prefill(input_ids, kv_callback=save_layer)

    progress.close()
    forward_time = time.time() - t0
    mins = int(forward_time // 60)
    secs = int(forward_time % 60)
    print(f"\n[forward] completed in {mins}m {secs}s ({seq_len / forward_time:.2f} tok/s)", flush=True)
    print(f"[save] saved all {num_layers} layers to {kv_cache_dir}/", flush=True)

    metadata = {
        "model_path": str(model_path),
        "reference": "models.demos.gpt_oss_d_p.reference.model (HF AutoModelForCausalLM golden)",
        "prompt_source": str(args.prompt_json) if args.prompt_json else "direct",
        "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
        "prompt_length_chars": len(prompt),
        "token_ids": token_ids,
        "n_tokens": seq_len,
        "n_layers": num_layers,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "sliding_window": model.cfg.sliding_window,
        "disable_sliding_window": args.disable_sliding_window,
        "zero_sinks": args.zero_sinks,
        "dtype": args.dtype,
        "chat_template": args.chat_template,
        "forward_time_seconds": forward_time,
        "tokens_per_second": seq_len / forward_time,
        "kv_cache_format": "separate_k_v",
        "key_cache_shape": saved_key_shape,
        "value_cache_shape": saved_val_shape,
    }

    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    total_size_gb = sum(f.stat().st_size for f in kv_cache_dir.glob("*.safetensors")) / (1024**3)

    print(f"\n{'=' * 70}", flush=True)
    print("Golden KV cache generation complete!", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"Metadata:         {metadata_path}", flush=True)
    print(f"KV cache:         {kv_cache_dir}/ ({num_layers} layer files)", flush=True)
    print(f"Total size:       {total_size_gb:.2f} GB", flush=True)
    print(f"Each layer:       K {saved_key_shape}, V {saved_val_shape}", flush=True)
    print("Next steps:", flush=True)
    print(f"  1. Verify: python3 models/demos/gpt_oss_d_p/scripts/verify_golden_kv.py {out_dir}", flush=True)
    print(f"  2. Use in tests: export PREFILL_TRACE_DIR={out_dir}", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
