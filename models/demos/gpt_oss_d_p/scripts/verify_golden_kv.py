#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify a GPT-OSS golden KV cache trace has the expected structure and properties.

GPT-OSS uses GQA with separate post-RoPE K and raw V per layer:
  key_cache_layer_{N}   [1, num_kv_heads, seq_len, head_dim]
  value_cache_layer_{N} [1, num_kv_heads, seq_len, head_dim]

Quick sanity check before using a trace for prefill validation:
- metadata.json complete (token_ids, shapes, layer count)
- All layer files exist with matching K/V tensors
- Data is finite (no NaNs/Infs)

Usage:
    python3 verify_golden_kv.py /path/to/trace_dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def verify_trace(trace_dir: Path) -> bool:
    success = True

    print(f"[verify] checking {trace_dir}/metadata.json...")
    metadata_path = trace_dir / "metadata.json"
    if not metadata_path.exists():
        print("  metadata.json not found")
        return False

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"  failed to parse metadata.json: {e}")
        return False

    required_keys = ["token_ids", "n_tokens", "key_cache_shape", "value_cache_shape"]
    missing = [k for k in required_keys if k not in metadata]
    if missing:
        print(f"  metadata missing required keys: {missing}")
        return False

    n_tokens = metadata["n_tokens"]
    num_layers = metadata.get("num_layers") or metadata.get("n_layers")
    if num_layers is None:
        print("  metadata missing num_layers / n_layers")
        return False

    expected_key_shape = tuple(metadata["key_cache_shape"])
    expected_val_shape = tuple(metadata["value_cache_shape"])

    print(f"  metadata OK: {n_tokens} tokens, {num_layers} layers")
    print(f"    K shape: {expected_key_shape}, V shape: {expected_val_shape}")

    kv_cache_dir = trace_dir / "kv_cache"
    if not kv_cache_dir.is_dir():
        print("  kv_cache/ directory not found")
        return False

    print(f"[verify] checking {num_layers} layer files...")
    for layer_idx in range(num_layers):
        layer_file = kv_cache_dir / f"layer_{layer_idx}.safetensors"
        if not layer_file.exists():
            print(f"  layer_{layer_idx}.safetensors not found")
            success = False
            continue

        key_name = f"key_cache_layer_{layer_idx}"
        val_name = f"value_cache_layer_{layer_idx}"
        try:
            with safe_open(str(layer_file), framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if key_name not in keys or val_name not in keys:
                    print(f"  layer {layer_idx}: expected {key_name!r} and {val_name!r}, found: {keys}")
                    success = False
                    continue

                key_tensor = f.get_tensor(key_name)
                val_tensor = f.get_tensor(val_name)

                if key_tensor.shape != expected_key_shape:
                    print(
                        f"  layer {layer_idx}: K shape {key_tensor.shape} != expected {expected_key_shape}"
                    )
                    success = False
                    continue
                if val_tensor.shape != expected_val_shape:
                    print(
                        f"  layer {layer_idx}: V shape {val_tensor.shape} != expected {expected_val_shape}"
                    )
                    success = False
                    continue

                for name, tensor in (("K", key_tensor), ("V", val_tensor)):
                    if tensor.dtype != torch.bfloat16:
                        print(f"  layer {layer_idx}: {name} dtype {tensor.dtype} (expected bfloat16)")
                    sample = tensor.flatten()[:1000]
                    if not torch.isfinite(sample).all():
                        print(f"  layer {layer_idx}: {name} contains NaN or Inf values")
                        success = False
        except Exception as e:
            print(f"  layer {layer_idx}: failed to load - {e}")
            success = False
            continue

        if (layer_idx + 1) % 10 == 0:
            print(f"  ... checked {layer_idx + 1}/{num_layers} layers")

    if success:
        print(f"  all {num_layers} layer files OK")
    else:
        print("  some layer files had errors")

    print("[verify] checking file sizes...")
    k_size = (
        expected_key_shape[0]
        * expected_key_shape[1]
        * expected_key_shape[2]
        * expected_key_shape[3]
        * 2
    )
    v_size = (
        expected_val_shape[0]
        * expected_val_shape[1]
        * expected_val_shape[2]
        * expected_val_shape[3]
        * 2
    )
    expected_size_per_layer_mb = (k_size + v_size) / (1024 * 1024)

    total_size_mb = 0.0
    for layer_idx in range(num_layers):
        layer_file = kv_cache_dir / f"layer_{layer_idx}.safetensors"
        if layer_file.exists():
            size_mb = layer_file.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            if size_mb < expected_size_per_layer_mb * 0.8 or size_mb > expected_size_per_layer_mb * 1.5:
                print(
                    f"  layer {layer_idx}: size {size_mb:.1f}MB unexpected "
                    f"(expected ~{expected_size_per_layer_mb:.1f}MB)"
                )

    print(f"  total trace size: {total_size_mb:.1f} MB ({total_size_mb / 1024:.2f} GB)")

    print(f"\n{'=' * 70}")
    if success:
        print("Trace verification PASSED")
        print(f"   Trace: {trace_dir}")
        print(f"   Tokens: {n_tokens}")
        print(f"   Layers: {num_layers}")
        print(f"   K shape: {expected_key_shape}")
        print(f"   V shape: {expected_val_shape}")
        print(f"   Size: {total_size_mb / 1024:.2f} GB")
        print(f"\n   Ready to use with PREFILL_TRACE_DIR={trace_dir}")
    else:
        print("Trace verification FAILED - see errors above")
    print(f"{'=' * 70}\n")

    return success


def main():
    ap = argparse.ArgumentParser(
        description="Verify a GPT-OSS golden KV cache trace directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "trace_dir",
        type=Path,
        help="Path to trace directory (contains metadata.json and kv_cache/)",
    )
    args = ap.parse_args()

    if not args.trace_dir.is_dir():
        print(f"ERROR: {args.trace_dir} is not a directory", file=sys.stderr)
        return 1

    success = verify_trace(args.trace_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
