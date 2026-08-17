#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate golden KV cache for MiniMax M3 prefill validation.

This script runs the self-contained torch reference (models/demos/minimax_m3/reference/model.py)
forward pass and extracts the KV cache (post-RoPE K, raw V) for each layer, saving it in the
format the golden KV-cache PCC check (scripts/verify_golden_kv.py) expects.

The reference is used instead of HuggingFace AutoModel because the shipped MiniMax-M3 checkpoint
is a vision-language package (model_type=minimax_m3_vl) that ships no modeling code, so
AutoModelForCausalLM cannot load it. The reference rebuilds just the text tower from the raw
language_model.* weights, with math composed from the team's PCC-validated unit-test references.

The reference uses the real attention schedule: dense causal attention for layers 0-2 and MSA
sparse attention for layers 3-59. Pass --dense-only to force full causal attention on every layer
(identical to the default at <=2048 tokens; above 2048 it will NOT match the real sparse model —
pipeline testing only, while MSA prefill is not yet device-tested).

Memory-efficient approach for 500GB RAM machines:
- Streams language_model.* weights per layer via mmap (never fully resident)
- Computes in fp32 (high-precision golden); saves K/V at --dtype (bf16 by default)
- Processes prompt in one shot (prefill-only, no decode)
- Saves layer-by-layer (streaming callback) to avoid keeping all KV in memory at once

Output format:
    {trace_dir}/
        metadata.json           - prompt, token_ids, model info
        kv_cache/
            layer_0.safetensors - key_cache_layer_0 / value_cache_layer_0  [1, num_kv_heads, seq_len, head_dim]
            layer_1.safetensors   (post-RoPE K, raw V; HF layout). MSA (sparse) layers ALSO carry
            ...                   index_k_cache_layer_N [1, 1, seq_len, sparse_index_dim] (post-norm/
            layer_N.safetensors   post-RoPE shared index key); dense layers / --dense-only store K/V only.

Usage:
    # From tt-metal root
    export HF_MODEL=/path/to/MiniMax-M3-dequantized
    python3 models/demos/minimax_m3/scripts/generate_golden_kv_cache.py \\
        --prompt-json prompt.json \\
        --out /mnt/models/minimax-m3-cache/golden/longbook_full \\
        --max-tokens 65536

    # Or with direct prompt text
    python3 models/demos/minimax_m3/scripts/generate_golden_kv_cache.py \\
        --prompt "The capital of France is" \\
        --out /tmp/minimax_m3_golden
"""

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

# Make `models.demos.minimax_m3...` importable when run as a script (sys.path[0] is the script dir).
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from models.demos.minimax_m3.reference.model import load_text_model  # noqa: E402


def _raise_cpu_time_limit():
    """The fp32 CPU reference runs matmuls across all cores, and RLIMIT_CPU counts CPU-seconds
    summed over every thread. With N threads busy the budget drains ~N× wall-clock, so a default
    soft cap (e.g. 86400 = 24 CPU-hr) is exhausted after only ~86400/N wall-seconds — the process
    then dies mid-run with SIGXCPU ("CPU time limit exceeded", core dumped), which looks like a
    crash but is just the limit. Raise the soft limit to the hard limit (allowed without
    privileges) so a long multi-layer prefill isn't killed partway through."""
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
            print(f"[limit] raised RLIMIT_CPU soft {soft}s -> {hard} (fp32 CPU golden burns CPU-time × nthreads)")
        except (ValueError, OSError) as e:
            print(
                f"[limit] WARNING: could not raise RLIMIT_CPU (soft={soft}s); if the run dies with "
                f"'CPU time limit exceeded', run `ulimit -t unlimited` first: {e}",
                file=sys.stderr,
            )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate golden KV cache for MiniMax M3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    prompt_group = ap.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt-json",
        type=Path,
        help='JSON file with {"prompt": "..."}',
    )
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="Direct prompt text",
    )

    # Output
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output trace directory (will create kv_cache/ subdir)",
    )

    # Model
    ap.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="HF model directory (default: $HF_MODEL env var). "
        "Works with official MiniMaxAI/MiniMax-M3 weights downloaded from HuggingFace.",
    )

    # Processing options
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Truncate prompt to this many tokens (default: no truncation)",
    )
    ap.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Skip chat template wrapping (use raw prompt). "
        "IMPORTANT: Must match tokenization used in your tests - if tests use chat "
        "template, generate golden trace WITH chat template (default). If tests use "
        "raw prompts, generate golden trace WITHOUT chat template (--no-chat-template).",
    )
    ap.add_argument(
        "--dtype",
        choices=["bfloat16", "float32", "float16"],
        default="bfloat16",
        help="Stored KV cache dtype (default: bfloat16). The reference always computes in fp32; "
        "K/V are cast to this dtype on save (bf16 matches the device cache).",
    )
    ap.add_argument(
        "--dense-only",
        action="store_true",
        help="Force full causal attention on ALL layers, skipping the MSA sparse path. Default is "
        "the real schedule (dense layers 0-2 + MSA sparse 3-59). At <=2048 tokens dense-only is "
        "identical to the default; above 2048 it does NOT match the real sparse model — use it "
        "to exercise the larger-sequence pipeline while MSA prefill is not yet device-tested.",
    )

    return ap.parse_args()


def load_prompt(args) -> str:
    """Load prompt from JSON or direct text."""
    if args.prompt_json:
        with open(args.prompt_json) as f:
            data = json.load(f)
        if isinstance(data, dict) and "prompt" in data:
            return data["prompt"]
        elif isinstance(data, str):
            return data
        else:
            raise ValueError(f"{args.prompt_json}: expected dict with 'prompt' key or string")
    return args.prompt


def tokenize_prompt(tokenizer, prompt: str, max_tokens: int, use_chat_template: bool) -> tuple[list[int], int]:
    """Tokenize and optionally truncate prompt.

    Returns:
        (token_ids, actual_length) where actual_length <= max_tokens
    """
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
    _raise_cpu_time_limit()  # fp32 CPU golden burns CPU-time × nthreads; avoid SIGXCPU mid-run (see fn)

    # Setup
    model_path = args.model_path or os.environ.get("HF_MODEL")
    if not model_path:
        print("ERROR: Must provide --model-path or set $HF_MODEL", file=sys.stderr)
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

    # Load prompt
    print(f"[load] reading prompt...", flush=True)
    prompt = load_prompt(args)
    print(f"[load] prompt length: {len(prompt)} characters", flush=True)

    # Load tokenizer
    print(f"[load] loading tokenizer from {model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Tokenize
    use_chat_template = not args.no_chat_template
    token_ids, seq_len = tokenize_prompt(tokenizer, prompt, args.max_tokens, use_chat_template)
    print(f"[load] tokenized to {seq_len} tokens", flush=True)

    # Build the self-contained torch reference (streams language_model.* weights via mmap).
    print(f"\n{'='*70}", flush=True)
    print(f"[load] Building reference text model from {model_path}", flush=True)
    print(f"[load] compute dtype=float32 (high-precision golden); weights streamed per layer (mmap)", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Set number of threads for CPU inference
    torch.set_num_threads(os.cpu_count() or 32)

    t0 = time.time()
    model = load_text_model(model_path, compute_dtype=torch.float32)
    num_layers = model.cfg.num_hidden_layers
    num_kv_heads = model.cfg.num_key_value_heads
    head_dim = model.cfg.head_dim
    print(
        f"[load] ✓ Reference ready in {time.time()-t0:.1f}s — "
        f"{num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}",
        flush=True,
    )

    # Run forward pass, saving each layer's KV cache as it is produced (streaming, low memory).
    print(f"\n{'='*70}", flush=True)
    print(f"[forward] Running prefill forward pass for {seq_len} tokens", flush=True)
    if args.dense_only:
        print(f"[forward] --dense-only: full causal attention on ALL layers (MSA path skipped)", flush=True)
        if seq_len > 2048:
            print(
                f"[forward] WARNING: seq_len={seq_len} > 2048 with --dense-only — output will NOT "
                f"match the real MSA sparse model (pipeline testing only).",
                flush=True,
            )
    else:
        print(f"[forward] real schedule: dense attn (layers 0-2) + MSA sparse attn (layers 3-59)", flush=True)
    print(f"[forward] WARNING: CPU inference is SLOW - can take 10-60+ minutes!", flush=True)
    print(f"{'='*70}\n", flush=True)

    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

    t0 = time.time()
    progress = tqdm(total=num_layers, desc="Prefill (save KV per layer)", unit="layer")
    saved = {}

    def save_layer(layer_idx, key_cache, value_cache, index_k=None):
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
        # MSA layers also emit index_k (post-norm/post-RoPE single shared index head
        # [1, 1, seq_len, sparse_index_dim]); dense layers pass None and store only K/V.
        if index_k is not None:
            index_k = index_k.to(dtype)
            tensors[f"index_k_cache_layer_{layer_idx}"] = index_k.contiguous()
            saved["index_k_shape"] = list(index_k.shape)
        save_file(tensors, str(kv_cache_dir / f"layer_{layer_idx}.safetensors"))
        saved["key_shape"] = list(key_cache.shape)
        saved["value_shape"] = list(value_cache.shape)
        progress.update(1)

    with torch.no_grad():
        model.prefill(input_ids, dense_only=args.dense_only, kv_callback=save_layer)

    progress.close()
    forward_time = time.time() - t0
    mins = int(forward_time // 60)
    secs = int(forward_time % 60)
    print(f"\n[forward] ✓ Completed in {mins}m {secs}s ({seq_len/forward_time:.2f} tok/s)", flush=True)
    print(f"[save] ✓ Saved all {num_layers} layers to {kv_cache_dir}/", flush=True)

    # Save metadata
    metadata = {
        "model_path": str(model_path),
        "reference": "models.demos.minimax_m3.reference.model (dense+MSA torch golden, fp32 compute)",
        "dense_only": args.dense_only,
        # default (dense+MSA) matches the real model at any length; dense_only matches only <=2048.
        "attention_matches_real_model": (not args.dense_only) or seq_len <= 2048,
        "prompt_source": str(args.prompt_json) if args.prompt_json else "direct",
        "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,  # Truncate for readability
        "prompt_length_chars": len(prompt),
        "token_ids": token_ids,
        "n_tokens": seq_len,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": args.dtype,
        "chat_template": use_chat_template,
        "forward_time_seconds": forward_time,
        "tokens_per_second": seq_len / forward_time,
        "kv_cache_format": "separate_k_v",
        "key_cache_shape": saved.get("key_shape"),  # [1, num_kv_heads, seq_len, head_dim]
        "value_cache_shape": saved.get("value_shape"),
        # MSA (sparse) layers also carry index_k_cache_layer_{i}; absent on dense layers / dense_only.
        "index_k_cache_shape": saved.get("index_k_shape"),  # [1, 1, seq_len, sparse_index_dim] or None
    }

    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Calculate storage size
    total_size_gb = sum(f.stat().st_size for f in kv_cache_dir.glob("*.safetensors")) / (1024**3)

    print(f"\n{'='*70}", flush=True)
    print(f"✅ Golden KV cache generation complete!", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"Metadata:         {metadata_path}", flush=True)
    print(f"KV cache:         {kv_cache_dir}/ ({num_layers} layer files)", flush=True)
    print(f"Total size:       {total_size_gb:.2f} GB", flush=True)
    print(f"", flush=True)
    print(f"Each layer contains:", flush=True)
    print(f"  - key_cache_layer_{{N}}:   {saved.get('key_shape')}", flush=True)
    print(f"  - value_cache_layer_{{N}}: {saved.get('value_shape')}", flush=True)
    print(f"", flush=True)
    print(f"Performance:", flush=True)
    print(f"  - Forward time: {forward_time:.1f}s ({seq_len/forward_time:.2f} tok/s)", flush=True)
    print(f"", flush=True)
    print(f"Next steps:", flush=True)
    print(f"  1. Verify: python3 models/demos/minimax_m3/scripts/verify_golden_kv.py {out_dir}", flush=True)
    print(f"  2. Use in tests: export PREFILL_TRACE_DIR={out_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
