#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify a generated golden KV cache trace has the expected structure and properties.

Quick sanity check before using a trace for validation:
- All layer files exist
- Shapes are consistent
- Data is finite (no NaNs/Infs)
- Metadata is complete

Usage:
    python3 verify_golden_kv.py /path/to/trace_dir
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def verify_trace(trace_dir: Path) -> bool:
    """Run verification checks on a golden KV trace directory.

    Returns:
        True if all checks pass, False otherwise
    """
    success = True

    # 1. Check metadata.json exists and is valid
    print(f"[verify] checking {trace_dir}/metadata.json...")
    metadata_path = trace_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"  ❌ metadata.json not found")
        return False

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"  ❌ failed to parse metadata.json: {e}")
        return False

    required_keys = ["token_ids", "n_tokens", "num_layers"]
    missing = [k for k in required_keys if k not in metadata]
    if missing:
        print(f"  ❌ metadata missing required keys: {missing}")
        return False

    n_tokens = metadata["n_tokens"]
    num_layers = metadata["num_layers"]

    # Check if using new format (separate K/V) or old format (kv_post_transform)
    if "key_cache_shape" in metadata and "value_cache_shape" in metadata:
        expected_key_shape = tuple(metadata["key_cache_shape"])
        expected_val_shape = tuple(metadata["value_cache_shape"])
        cache_format = "separate K/V"
    elif "kv_cache_shape_per_layer" in metadata:
        expected_key_shape = tuple(metadata["kv_cache_shape_per_layer"])
        expected_val_shape = None
        cache_format = "kv_post_transform (legacy)"
    else:
        print(f"  ❌ metadata missing cache shape information")
        return False

    print(f"  ✓ metadata OK: {n_tokens} tokens, {num_layers} layers")
    print(f"    Format: {cache_format}")
    if expected_val_shape:
        print(f"    K shape: {expected_key_shape}, V shape: {expected_val_shape}")
    else:
        print(f"    KV shape: {expected_key_shape}")

    # 2. Check kv_cache/ directory exists
    kv_cache_dir = trace_dir / "kv_cache"
    if not kv_cache_dir.is_dir():
        print(f"  ❌ kv_cache/ directory not found")
        return False

    # 3. Check all layer files exist and have correct structure
    print(f"[verify] checking {num_layers} layer files...")
    for layer_idx in range(num_layers):
        layer_file = kv_cache_dir / f"layer_{layer_idx}.safetensors"

        if not layer_file.exists():
            print(f"  ❌ layer_{layer_idx}.safetensors not found")
            success = False
            continue

        # Open and verify
        try:
            with safe_open(str(layer_file), framework="pt", device="cpu") as f:
                keys = list(f.keys())

                # Check for new format (separate K/V) or legacy format (kv_post_transform)
                has_separate_kv = f"key_cache_layer_{layer_idx}" in keys and f"value_cache_layer_{layer_idx}" in keys
                has_legacy = f"kv_post_transform_layer_{layer_idx}" in keys

                if not has_separate_kv and not has_legacy:
                    print(f"  ❌ layer {layer_idx}: no recognized keys found (found: {keys})")
                    success = False
                    continue

                if has_separate_kv:
                    # New format: separate K and V
                    key_tensor = f.get_tensor(f"key_cache_layer_{layer_idx}")
                    val_tensor = f.get_tensor(f"value_cache_layer_{layer_idx}")

                    # Check shapes
                    if key_tensor.shape != expected_key_shape:
                        print(f"  ❌ layer {layer_idx}: K shape {key_tensor.shape} != expected {expected_key_shape}")
                        success = False
                        continue

                    if val_tensor.shape != expected_val_shape:
                        print(f"  ❌ layer {layer_idx}: V shape {val_tensor.shape} != expected {expected_val_shape}")
                        success = False
                        continue

                    # Check dtype
                    if key_tensor.dtype != torch.bfloat16:
                        print(f"  ⚠️  layer {layer_idx}: K dtype {key_tensor.dtype} (expected bfloat16)")
                    if val_tensor.dtype != torch.bfloat16:
                        print(f"  ⚠️  layer {layer_idx}: V dtype {val_tensor.dtype} (expected bfloat16)")

                    # Check for NaN/Inf
                    k_sample = key_tensor.flatten()[:1000]
                    v_sample = val_tensor.flatten()[:1000]
                    if not torch.isfinite(k_sample).all():
                        print(f"  ❌ layer {layer_idx}: K contains NaN or Inf values")
                        success = False
                        continue
                    if not torch.isfinite(v_sample).all():
                        print(f"  ❌ layer {layer_idx}: V contains NaN or Inf values")
                        success = False
                        continue

                else:
                    # Legacy format: kv_post_transform
                    tensor = f.get_tensor(f"kv_post_transform_layer_{layer_idx}")

                    if tensor.shape != expected_key_shape:
                        print(f"  ❌ layer {layer_idx}: shape {tensor.shape} != expected {expected_key_shape}")
                        success = False
                        continue

                    if tensor.dtype != torch.bfloat16:
                        print(f"  ⚠️  layer {layer_idx}: dtype {tensor.dtype} (expected bfloat16)")

                    sample = tensor.flatten()[:1000]
                    if not torch.isfinite(sample).all():
                        print(f"  ❌ layer {layer_idx}: contains NaN or Inf values")
                        success = False
                        continue

        except Exception as e:
            print(f"  ❌ layer {layer_idx}: failed to load - {e}")
            success = False
            continue

        # Progress update
        if (layer_idx + 1) % 10 == 0:
            print(f"  ... checked {layer_idx + 1}/{num_layers} layers")

    if success:
        print(f"  ✓ all {num_layers} layer files OK")
    else:
        print(f"  ❌ some layer files had errors")

    # 4. Check file sizes are reasonable
    print(f"[verify] checking file sizes...")
    total_size_mb = 0

    # Calculate expected size based on format
    if expected_val_shape:
        # Separate K/V: sum of both shapes
        k_size = expected_key_shape[0] * expected_key_shape[1] * expected_key_shape[2] * expected_key_shape[3] * 2
        v_size = expected_val_shape[0] * expected_val_shape[1] * expected_val_shape[2] * expected_val_shape[3] * 2
        expected_size_per_layer_mb = (k_size + v_size) / (1024 * 1024)
    else:
        # Legacy format
        expected_size_per_layer_mb = (expected_key_shape[0] * expected_key_shape[1] * 2) / (1024 * 1024)

    for layer_idx in range(num_layers):
        layer_file = kv_cache_dir / f"layer_{layer_idx}.safetensors"
        if layer_file.exists():
            size_mb = layer_file.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb

            # Check if size is approximately correct (within 20% due to safetensors overhead)
            if size_mb < expected_size_per_layer_mb * 0.8 or size_mb > expected_size_per_layer_mb * 1.5:
                print(
                    f"  ⚠️  layer {layer_idx}: size {size_mb:.1f}MB unexpected (expected ~{expected_size_per_layer_mb:.1f}MB)"
                )

    print(f"  ✓ total trace size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")

    # 5. Final summary
    print(f"\n{'='*70}")
    if success:
        print(f"✅ Trace verification PASSED")
        print(f"   Trace: {trace_dir}")
        print(f"   Format: {cache_format}")
        print(f"   Tokens: {n_tokens}")
        print(f"   Layers: {num_layers}")
        if expected_val_shape:
            print(f"   K shape: {expected_key_shape}")
            print(f"   V shape: {expected_val_shape}")
        else:
            print(f"   KV shape: {expected_key_shape}")
        print(f"   Size: {total_size_mb/1024:.2f} GB")
        print(f"\n   Ready to use with PREFILL_TRACE_DIR={trace_dir}")
    else:
        print(f"❌ Trace verification FAILED - see errors above")

    print(f"{'='*70}\n")

    return success


def main():
    ap = argparse.ArgumentParser(
        description="Verify a golden KV cache trace directory",
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
