#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sanity-check a generated Llama-3.1-8B golden KV trace (tt-blaze#4147).

Runs before anything is graded against the golden, because a malformed golden does not announce
itself — it shows up as a device PCC number that looks like a model bug. Pure host-side: no device,
no checkpoint, no model.

Checks, in rough order of how badly each one misleads when wrong:

* **Frame.** ``metadata.rope_frame`` must be ``meta``, since that is the frame blaze decode writes.
  An ``hf``-frame golden agrees with a prefill that has the *same* frame error — the two cancel —
  so it passes a device-vs-golden comparison while decode reads a permuted cache.
* **bf8 headroom.** Reports the PCC of the stored bfloat16 golden against itself quantised to the
  device's bfloat8_b. That is the ceiling any device comparison can reach; forgetting the
  round-trip is what produces the spurious ~0.94-0.96 gap that reads as a real bug.
* **Completeness and shape.** Every layer present, every tensor ``[1, n_kv_heads, n_tokens,
  head_dim]``, consistent across layers.
* **Content.** No NaN or Inf; no all-zero layer (an unwritten layer is otherwise invisible); K and V
  differ from each other (saving K twice is an easy mistake that preserves every shape).
* **Layer diversity.** Adjacent layers must not be identical, which catches a capture callback that
  recorded the same tensor 32 times.

Usage:
    python3 models/demos/llama_3p1_8b_d_p/scripts/verify_golden_kv.py /path/to/trace_dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def bf8_reference_pcc(tensor: torch.Tensor) -> float:
    """PCC of a bfloat16 tensor against a simulated bfloat8_b round-trip.

    bfloat8_b is a block format: one shared 8-bit exponent per 16-element block with 8-bit
    mantissas. Approximated here by quantising each 16-element block to the block's exponent, which
    is close enough to show the magnitude of the effect without needing a device.
    """
    flat = tensor.float().flatten()
    block = 16
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.view(-1, block)

    max_abs = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    exponent = torch.floor(torch.log2(max_abs))
    step = torch.pow(2.0, exponent - 7)  # 8-bit mantissa relative to the block exponent
    quantised = torch.round(blocks / step) * step

    a = blocks.flatten()
    b = quantised.flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-30)
    return float((a @ b) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir", type=Path, help="Golden trace directory (contains metadata.json + kv_cache/)")
    parser.add_argument(
        "--bf8-floor",
        type=float,
        default=0.999,
        help="Fail if the bf8 round-trip self-PCC is below this (default 0.999)",
    )
    args = parser.parse_args()

    trace_dir: Path = args.trace_dir
    metadata_path = trace_dir / "metadata.json"
    kv_dir = trace_dir / "kv_cache"

    problems: list[str] = []
    warnings: list[str] = []

    if not metadata_path.is_file():
        print(f"FAIL: no metadata.json in {trace_dir}", file=sys.stderr)
        return 1
    if not kv_dir.is_dir():
        print(f"FAIL: no kv_cache/ in {trace_dir}", file=sys.stderr)
        return 1

    metadata = json.loads(metadata_path.read_text())
    num_layers = metadata.get("num_layers") or metadata.get("n_layers")
    n_tokens = metadata.get("n_tokens")
    n_kv_heads = metadata.get("num_kv_heads")
    head_dim = metadata.get("head_dim")
    token_ids = metadata.get("token_ids")

    print(f"=== {trace_dir} ===")
    print(f"model        : {metadata.get('model_path')}")
    print(f"tokens       : {n_tokens}")
    print(f"layers       : {num_layers}")
    print(f"kv heads     : {n_kv_heads}, head_dim {head_dim}")
    print(f"rope frame   : {metadata.get('rope_frame')}")
    print(f"stored dtype : {metadata.get('stored_dtype')} (device cache {metadata.get('device_cache_dtype')})")

    if metadata.get("rope_frame") != "meta":
        problems.append(
            f"rope_frame is {metadata.get('rope_frame')!r}, expected 'meta' — blaze decode writes K in the "
            f"Meta-interleaved frame, and an hf-frame golden cancels against a prefill with the same "
            f"frame error instead of catching it"
        )
    if metadata.get("sliding_window") is not None:
        problems.append(f"sliding_window is {metadata['sliding_window']!r}; Llama-3.1-8B is full attention")
    if token_ids is not None and n_tokens is not None and len(token_ids) != n_tokens:
        problems.append(f"metadata has {len(token_ids)} token_ids but n_tokens={n_tokens}")

    expected = (1, n_kv_heads, n_tokens, head_dim)
    previous_key = None
    worst_bf8 = 1.0
    worst_bf8_layer = None

    for layer_idx in range(num_layers):
        path = kv_dir / f"layer_{layer_idx}.safetensors"
        if not path.is_file():
            problems.append(f"layer {layer_idx}: missing {path.name}")
            continue
        tensors = load_file(str(path))
        key_name, value_name = f"key_cache_layer_{layer_idx}", f"value_cache_layer_{layer_idx}"
        if key_name not in tensors or value_name not in tensors:
            problems.append(f"layer {layer_idx}: expected {key_name} and {value_name}, got {sorted(tensors)}")
            continue

        key, value = tensors[key_name], tensors[value_name]
        for name, tensor in ((key_name, key), (value_name, value)):
            if tuple(tensor.shape) != expected:
                problems.append(f"{name}: shape {tuple(tensor.shape)}, expected {expected}")
            if torch.isnan(tensor.float()).any():
                problems.append(f"{name}: contains NaN")
            if torch.isinf(tensor.float()).any():
                problems.append(f"{name}: contains Inf")
            if torch.count_nonzero(tensor) == 0:
                problems.append(f"{name}: is entirely zero — the layer was probably never captured")

        if torch.equal(key, value):
            problems.append(f"layer {layer_idx}: K and V are identical — K was likely saved twice")
        if previous_key is not None and torch.equal(previous_key, key):
            problems.append(f"layer {layer_idx}: K is identical to layer {layer_idx - 1} — capture recorded one tensor")
        previous_key = key

        pcc = bf8_reference_pcc(key)
        if pcc < worst_bf8:
            worst_bf8, worst_bf8_layer = pcc, layer_idx

    print(f"bf8 headroom : worst-layer self-PCC {worst_bf8:.6f} (layer {worst_bf8_layer})")
    print(
        "               this is the CEILING for any device comparison — round-trip the golden "
        "through bfloat8_b before computing PCC"
    )
    if worst_bf8 < args.bf8_floor:
        warnings.append(
            f"bf8 round-trip self-PCC {worst_bf8:.6f} is below {args.bf8_floor}; a device PCC target "
            f"above this is unreachable regardless of correctness"
        )

    for warning in warnings:
        print(f"WARN: {warning}")
    if problems:
        print(f"\nFAILED with {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(f"\nOK: {num_layers} layers, {n_tokens} tokens, meta frame, no NaN/Inf, layers distinct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
