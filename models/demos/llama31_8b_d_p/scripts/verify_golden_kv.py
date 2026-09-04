#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify a Llama-3.1-8B golden KV-cache trace. Gate: `G-GOLDEN`. **Imports no ttnn.**

Checks the trace `scripts/generate_golden_kv_cache.py` wrote is structurally sound and *carries
content*, over every layer, and prints one row per layer. Exit 0 iff every check passes.

**It scores nothing against the device.** `BRINGUP_RECIPE.md:1595-1598` describes this file as
comparing "a device KV read-back against the golden ... reporting min/mean PCC per layer", while
`:1588-1590` — the `G-GOLDEN` gate that owns it — says it "imports no ttnn" and that "the
device-vs-golden scoring lives in `G-CHUNK`". The two cannot both hold. This file follows the gate:
device-vs-golden PCC is `tests/unit/test_attention_chunked_vs_ref.py`'s
(`G-CHUNK`), and both in-repo templates
(`models/demos/gpt_oss_d_p/scripts/verify_golden_kv.py`,
`models/demos/minimax_m3/scripts/verify_golden_kv.py`) are host-only structural checkers too.
`DEC-060` records the reading.

**Four checks the templates do not make.** `G-GOLDEN`'s negative controls are "a zeroed layer and a
deleted layer must both make it exit non-zero", and the template verifier
(`models/demos/gpt_oss_d_p/scripts/verify_golden_kv.py:111-130`) catches the deleted layer but
**passes a zeroed one**: it validates shape, dtype and finiteness of the first 1000 elements and
nothing else. So:

1. **Content, not just shape** — every layer's K and V must have a non-zero norm and a non-zero
   spread, and **no all-zero token row**. This is what makes the zeroed-layer control fail.
2. **Finiteness over the whole tensor**, not a 1000-element sample. A NaN at position 300 of 512 is
   exactly the kind of thing a leading sample misses.
3. **fp32 is required, not merely recorded.** The golden is the reference and recipe §2.1(a) is
   about precisely this: a bf16 golden shares the device's rounding and reports a flattered PCC.
4. **No two layers may be bit-identical** — a streaming driver that wrote the same layer twice would
   otherwise pass every per-layer check.

Run:
    python3 models/demos/llama31_8b_d_p/scripts/verify_golden_kv.py $PREFILL_TRACE_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REQUIRED_METADATA = ("token_ids", "n_tokens", "num_layers", "num_kv_heads", "head_dim", "dtype", "key_cache_shape")
# The golden must be fp32 (`DEC-059`, recipe P7 step 1, `BRINGUP_RECIPE.md:1587`).
REQUIRED_DTYPE = torch.float32


class Report:
    """Collects failures instead of raising, so one run names every problem in the trace."""

    def __init__(self):
        self.failures: list[str] = []

    def check(self, ok: bool, message: str) -> bool:
        if not ok:
            self.failures.append(message)
            print(f"  FAIL: {message}")
        return ok

    @property
    def ok(self) -> bool:
        return not self.failures


def _stats(tensor: torch.Tensor) -> dict:
    """Per-layer statistics, over the **whole** tensor."""
    flat = tensor.reshape(-1)
    row_norms = tensor.reshape(-1, tensor.shape[-1]).norm(dim=-1)
    return {
        "rms": float(flat.pow(2).mean().sqrt()),
        "absmax": float(flat.abs().max()),
        "std": float(flat.std()),
        "min_row_norm": float(row_norms.min()),
        "finite": bool(torch.isfinite(flat).all()),
        "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()[:16],
    }


def verify_trace(trace_dir: Path) -> bool:
    report = Report()
    print(f"[verify] {trace_dir}")

    metadata_path = trace_dir / "metadata.json"
    if not metadata_path.is_file():
        print(f"  FAIL: metadata.json not found in {trace_dir}")
        return False
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  FAIL: could not parse metadata.json: {e}")
        return False

    missing = [k for k in REQUIRED_METADATA if k not in metadata]
    if missing:
        print(f"  FAIL: metadata.json is missing {missing}")
        return False

    n_tokens = int(metadata["n_tokens"])
    num_layers = int(metadata["num_layers"])
    num_kv_heads = int(metadata["num_kv_heads"])
    head_dim = int(metadata["head_dim"])
    expected_shape = (1, num_kv_heads, n_tokens, head_dim)

    report.check(
        metadata["dtype"] == "float32",
        f"metadata dtype is {metadata['dtype']!r}; the golden is the reference and must be fp32 (recipe P7 step 1)",
    )
    report.check(
        len(metadata["token_ids"]) == n_tokens,
        f"metadata has {len(metadata['token_ids'])} token_ids but claims n_tokens={n_tokens}",
    )
    report.check(
        tuple(metadata["key_cache_shape"]) == expected_shape,
        f"metadata key_cache_shape {tuple(metadata['key_cache_shape'])} != derived {expected_shape}",
    )
    loop = metadata.get("llamamodel_loop_check")
    report.check(
        loop is not None and loop.get("layers_compared") == num_layers,
        "metadata has no llamamodel_loop_check over every layer: G-GOLDEN requires the streamed "
        "driver to equal LlamaModel's own loop bit-exactly (re-run the generator without --no-verify-loop)",
    )
    if loop is not None:
        for field in ("max_abs_delta_k", "max_abs_delta_v", "max_abs_delta_post_norm_hidden"):
            report.check(
                float(loop.get(field, 1.0)) == 0.0,
                f"llamamodel_loop_check.{field} = {loop.get(field)}; rtol=atol=0 agreement is required",
            )
    report.check(
        metadata.get("zeroed_layer") is None,
        f"metadata records zeroed_layer={metadata.get('zeroed_layer')} — this trace is a negative control",
    )

    kv_dir = trace_dir / "kv_cache"
    if not kv_dir.is_dir():
        print(f"  FAIL: {kv_dir} not found")
        return False

    print(
        f"[verify] {num_layers} layers, {n_tokens} tokens, {num_kv_heads} KV heads, "
        f"head_dim={head_dim}, dtype={metadata['dtype']}"
    )
    header = f"  {'layer':>5} | {'K rms':>10} {'K absmax':>10} {'K min|row|':>10} | {'V rms':>10} {'V absmax':>10} {'V min|row|':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    seen: dict[str, tuple[int, str]] = {}
    for layer_idx in range(num_layers):
        layer_file = kv_dir / f"layer_{layer_idx}.safetensors"
        if not layer_file.is_file():
            report.check(False, f"layer {layer_idx}: {layer_file.name} is missing")
            continue
        key_name, val_name = f"key_cache_layer_{layer_idx}", f"value_cache_layer_{layer_idx}"
        try:
            with safe_open(str(layer_file), framework="pt", device="cpu") as h:
                keys = list(h.keys())
                if key_name not in keys or val_name not in keys:
                    report.check(False, f"layer {layer_idx}: expected {key_name!r} and {val_name!r}, found {keys}")
                    continue
                tensors = {"K": h.get_tensor(key_name), "V": h.get_tensor(val_name)}
        except Exception as e:  # a corrupt file is a trace failure, not a crash
            report.check(False, f"layer {layer_idx}: could not read {layer_file.name}: {e}")
            continue

        stats = {}
        for name, tensor in tensors.items():
            report.check(
                tuple(tensor.shape) == expected_shape,
                f"layer {layer_idx} {name}: shape {tuple(tensor.shape)} != {expected_shape}",
            )
            report.check(
                tensor.dtype == REQUIRED_DTYPE,
                f"layer {layer_idx} {name}: dtype {tensor.dtype} != {REQUIRED_DTYPE} (the golden must be fp32)",
            )
            stats[name] = _stats(tensor)
            report.check(stats[name]["finite"], f"layer {layer_idx} {name}: holds a NaN or an Inf")
            # Content checks — what makes the zeroed-layer negative control fail.
            report.check(stats[name]["rms"] > 0.0, f"layer {layer_idx} {name}: RMS is 0 — the tensor is all zeros")
            report.check(
                stats[name]["std"] > 0.0, f"layer {layer_idx} {name}: std is 0 — every element is the same value"
            )
            report.check(
                stats[name]["min_row_norm"] > 0.0,
                f"layer {layer_idx} {name}: some token row is all zeros (min row norm 0)",
            )
            digest = stats[name]["sha256"]
            if digest in seen:
                other_layer, other_name = seen[digest]
                report.check(
                    False,
                    f"layer {layer_idx} {name} is bit-identical to layer {other_layer} {other_name} "
                    f"(sha256 {digest}) — the streamed driver wrote the same tensor twice",
                )
            else:
                seen[digest] = (layer_idx, name)

        print(
            f"  {layer_idx:>5} | {stats['K']['rms']:>10.5f} {stats['K']['absmax']:>10.4f} "
            f"{stats['K']['min_row_norm']:>10.5f} | {stats['V']['rms']:>10.5f} "
            f"{stats['V']['absmax']:>10.4f} {stats['V']['min_row_norm']:>10.5f}"
        )

    bytes_per_layer = 2 * 1 * num_kv_heads * n_tokens * head_dim * REQUIRED_DTYPE.itemsize
    actual = sum(f.stat().st_size for f in kv_dir.glob("*.safetensors"))
    print(
        f"[verify] {len(list(kv_dir.glob('*.safetensors')))} layer files, {actual / 1024**2:.1f} MB total "
        f"(expected ~{num_layers * bytes_per_layer / 1024**2:.1f} MB at fp32)"
    )

    print("=" * 72)
    if report.ok:
        print(f"G-GOLDEN: trace verification PASSED — {num_layers} layers, {n_tokens} tokens, fp32")
        print(f"  export PREFILL_TRACE_DIR={trace_dir}")
    else:
        print(f"G-GOLDEN: trace verification FAILED — {len(report.failures)} problem(s):")
        for failure in report.failures:
            print(f"  - {failure}")
    print("=" * 72)
    return report.ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify a Llama-3.1-8B golden KV-cache trace (G-GOLDEN)")
    ap.add_argument(
        "trace_dir",
        type=Path,
        nargs="?",
        default=None,
        help="trace directory (default: $PREFILL_TRACE_DIR)",
    )
    args = ap.parse_args(argv)

    trace_dir = args.trace_dir or (
        Path(os.environ["PREFILL_TRACE_DIR"]) if os.environ.get("PREFILL_TRACE_DIR") else None
    )
    if trace_dir is None:
        print("ERROR: pass a trace directory or set $PREFILL_TRACE_DIR", file=sys.stderr)
        return 2
    if not Path(trace_dir).is_dir():
        print(f"ERROR: {trace_dir} is not a directory", file=sys.stderr)
        return 2
    return 0 if verify_trace(Path(trace_dir)) else 1


if __name__ == "__main__":
    sys.exit(main())
