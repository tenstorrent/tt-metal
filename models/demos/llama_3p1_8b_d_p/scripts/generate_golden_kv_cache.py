#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the golden per-layer KV cache for Llama-3.1-8B prefill (tt-blaze#4147).

Runs the torch reference (``reference/model.py``) on CPU and saves post-RoPE K and raw V for every
layer, in the trace layout ``verify_golden_kv.py`` and the full-model KV PCC check consume:

    {trace_dir}/
        metadata.json                 prompt + token_ids + shapes
        kv_cache/layer_N.safetensors  key_cache_layer_N / value_cache_layer_N

A trace dir therefore carries **both** the tokens and the golden KV, so a per-slot trace gives a
per-slot prompt *and* a per-slot golden — which is what the Gate 2 cross-talk check needs.

Two things about this golden are easy to get wrong and both produce a golden that makes a broken
prefill look correct:

**Frame.** ``--frame meta`` (the default) converts K into the Meta-interleaved frame before saving.
The reference computes in the HF frame, because that is what makes it comparable to HF logits, but
**blaze decode writes K in the Meta-interleaved frame** and the golden has to match what decode
writes, not what prefill happens to do. Saving the HF frame would still pass a device-vs-golden
comparison if prefill also had the frame wrong — the two errors cancel — while decode would read a
permutation of the cache. ``--frame hf`` exists only for debugging that divergence.

**Dtype.** The device cache is ``bfloat8_b``. K/V are stored here in bfloat16 (bf8_b has no torch
equivalent), so **the consumer must round-trip the golden through bf8_b before computing PCC**. A
full-precision golden leaves a spurious ~0.94-0.96 gap that reads as a real bug. ``verify_golden_kv.py``
reports the bf16-vs-bf8 self-PCC so the size of that effect is visible up front.

Usage:
    python3 models/demos/llama_3p1_8b_d_p/scripts/generate_golden_kv_cache.py \\
        --prompt "The capital of France is" --out /data/$USER/llama31_golden_short

    python3 models/demos/llama_3p1_8b_d_p/scripts/generate_golden_kv_cache.py \\
        --prompt-json prompt.json --out /data/$USER/llama31_golden_8k --max-tokens 8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig  # noqa: E402
from models.demos.llama_3p1_8b_d_p.reference.model import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    load_reference_model,
    to_meta_frame,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the golden KV cache for Llama-3.1-8B prefill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-json", type=Path, help='JSON file with {"prompt": "..."}')
    prompt_group.add_argument("--prompt", type=str, help="Direct prompt text")

    parser.add_argument("--out", type=Path, required=True, help="Output trace directory (creates kv_cache/)")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("PREFILL_HF_MODEL", str(DEFAULT_CHECKPOINT)),
        help="HF checkpoint dir (default: $PREFILL_HF_MODEL, else the served checkpoint)",
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="Truncate the prompt to this many tokens")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=Llama31_8BConfig.NUM_LAYERS,
        help="Layers to capture (default: all 32). Fewer is for debugging only — the full-model "
        "check needs every layer.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Prompt chunk size for the reference forward. Only bounds peak memory: the chunked "
        "and single-shot results are identical (pinned by test_chunked_prefill_equals_single_shot).",
    )
    parser.add_argument(
        "--frame",
        choices=["meta", "hf"],
        default="meta",
        help="Saved K frame. 'meta' (default) is what blaze decode writes and the only correct "
        "choice for a golden; 'hf' is for debugging a frame divergence.",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Apply the chat template (off by default, matching the prefill demos)",
    )
    return parser.parse_args()


def load_prompt(args) -> str:
    if args.prompt_json:
        data = json.loads(Path(args.prompt_json).read_text())
        if isinstance(data, dict) and "prompt" in data:
            return data["prompt"]
        if isinstance(data, str):
            return data
        raise ValueError(f"{args.prompt_json}: expected a string or a dict with a 'prompt' key")
    return args.prompt


def main() -> int:
    args = parse_args()
    torch.set_num_threads(os.cpu_count() or 32)

    out_dir: Path = args.out
    kv_dir = out_dir / "kv_cache"
    kv_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    print(f"[load] tokenizer from {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt = load_prompt(args)

    if args.chat_template:
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        )
    else:
        token_ids = tokenizer(prompt)["input_ids"]
    if args.max_tokens is not None and len(token_ids) > args.max_tokens:
        print(f"[load] truncating {len(token_ids)} -> {args.max_tokens} tokens", flush=True)
        token_ids = token_ids[: args.max_tokens]
    seq_len = len(token_ids)
    print(f"[load] {seq_len} tokens", flush=True)

    print(f"[load] building reference ({args.num_layers} layers) — CPU, this takes a few minutes", flush=True)
    start = time.time()
    model = load_reference_model(args.model_path, num_layers=args.num_layers, dtype=torch.float32)
    print(f"[load] reference ready in {time.time() - start:.1f}s", flush=True)

    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

    print(f"[forward] prefilling {seq_len} tokens in chunks of {args.chunk_size} — CPU inference is SLOW", flush=True)
    start = time.time()
    past_kvs = None
    with torch.no_grad():
        for chunk_start in range(0, seq_len, args.chunk_size):
            chunk = input_ids[:, chunk_start : chunk_start + args.chunk_size]
            _, past_kvs = model(chunk, start_pos=chunk_start, past_kvs=past_kvs, return_kv=True)
            done = min(chunk_start + args.chunk_size, seq_len)
            print(f"[forward] {done}/{seq_len} tokens ({time.time() - start:.0f}s)", flush=True)
    forward_time = time.time() - start

    head_dim = Llama31_8BConfig.HEAD_DIM
    n_kv_heads = Llama31_8BConfig.NUM_KEY_VALUE_HEADS
    expected = (1, n_kv_heads, seq_len, head_dim)

    key_shape = value_shape = None
    for layer_idx, (k, v) in enumerate(past_kvs):
        if tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError(f"layer {layer_idx}: K={tuple(k.shape)} V={tuple(v.shape)}, expected {expected}")
        # V is frame-independent: RoPE only touches K. Converting V too would be a silent corruption
        # that no shape check catches.
        k_out = to_meta_frame(k) if args.frame == "meta" else k
        k_out = k_out.to(torch.bfloat16).contiguous()
        v_out = v.to(torch.bfloat16).contiguous()
        layer_path = kv_dir / f"layer_{layer_idx}.safetensors"
        save_file(
            {f"key_cache_layer_{layer_idx}": k_out, f"value_cache_layer_{layer_idx}": v_out},
            str(layer_path),
        )
        # save_file creates 0600, ignoring the umask that gave metadata.json 0664 next to it. A
        # trace written to the shared /mnt/models store is then readable only by whoever generated
        # it, and CI runs as a different user: the KV-accuracy stage failed on a golden that was
        # present and correct. Worth the explicit chmod because of how it fails -- safetensors
        # reports any failed open as "No such file or directory", so the symptom names a missing
        # file and sends you looking for a share that was never the problem.
        layer_path.chmod(0o644)
        key_shape, value_shape = list(k_out.shape), list(v_out.shape)
        print(f"[save] layer {layer_idx}", flush=True)

    metadata = {
        "model_path": str(args.model_path),
        "reference": "models.demos.llama_3p1_8b_d_p.reference.model (fresh torch reference)",
        "prompt_source": str(args.prompt_json) if args.prompt_json else "direct",
        "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
        "prompt_length_chars": len(prompt),
        "token_ids": token_ids,
        "n_tokens": seq_len,
        "n_layers": args.num_layers,
        "num_layers": args.num_layers,
        "num_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "sliding_window": Llama31_8BConfig.SLIDING_WINDOW,
        # The two fields a consumer must not guess. rope_frame says which permutation K is in;
        # device_cache_dtype says what the golden has to be round-tripped through before PCC.
        "rope_frame": args.frame,
        "stored_dtype": "bfloat16",
        "device_cache_dtype": "bfloat8_b",
        "chat_template": args.chat_template,
        "chunk_size": args.chunk_size,
        "kv_cache_format": "separate_k_v",
        "key_cache_shape": key_shape,
        "value_cache_shape": value_shape,
        "forward_time_seconds": forward_time,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    total_gb = sum(f.stat().st_size for f in kv_dir.glob("*.safetensors")) / (1024**3)
    print(f"\n{'=' * 70}")
    print(f"Golden KV written to {out_dir} ({total_gb:.2f} GB, {args.num_layers} layers)")
    print(f"K frame: {args.frame} (decode reads meta); stored bfloat16, device cache is bfloat8_b")
    print(f"Verify:  python3 models/demos/llama_3p1_8b_d_p/scripts/verify_golden_kv.py {out_dir}")
    print(f"Use:     export PREFILL_TRACE_DIR={out_dir}")
    print(f"{'=' * 70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
