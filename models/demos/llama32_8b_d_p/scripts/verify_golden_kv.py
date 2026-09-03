#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify a ``llama32_8b_d_p`` golden KV trace, and PCC a device read-back against it per layer.

Two jobs, in this order (``BRINGUP_RECIPE.md`` P7 step 2, gate ``G-GOLDEN``):

1. **Structure** — every ``layer_<i>.safetensors`` exists, carries both
   ``key_cache_layer_<i>`` / ``value_cache_layer_<i>`` at ``[1, num_kv_heads, seq_len, head_dim]``,
   is finite, and is **not constant**. The constant check matters: an all-zero cache passes
   "no NaNs" and then scores a *nan* PCC rather than a low one, so a broken producer looks like a
   broken comparison.
2. **Device comparison** (``--device-dump DIR``) — per-layer K and V PCC against the golden, with
   the min and the mean over all layers, printed as a table and returned to the caller.

Run it with no ``--device-dump`` to validate a freshly generated trace; run it with one to score a
device dump written by
``tests/unit/test_attention_chunked_vs_ref.py`` / ``tt/tt_prefill_runtime.py::dump_slot_kv``.

**The K convention is the whole subtlety.** The golden stores HF's *half-split* rotary
(``[x0..x63, x64..x127]`` rotated by ``rotate_half``); the device stores Meta's *interleaved*
(``[c0, c0, c1, c1, ...]``, see the ``tt/rope.py`` module docstring). Both encode the same
``[S, head_dim/2]`` frequency table, and comparing them without the lane permutation gives a
plausible-but-wrong ~0.5-0.9 — the classic RoPE-convention signature from ``Appendix B``.
:func:`hf_to_meta_lane_permutation` is the single definition of that permutation in this package;
it mirrors ``models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:552-556``. A device dump declares
its own convention in ``metadata.json`` (``"meta"`` or ``"hf"``) and the permutation is applied
only for ``"meta"``.

Usage::

    # structure only
    python3 models/demos/llama32_8b_d_p/scripts/verify_golden_kv.py /path/to/trace

    # structure + device read-back PCC per layer
    python3 models/demos/llama32_8b_d_p/scripts/verify_golden_kv.py /path/to/trace \\
        --device-dump /path/to/device_dump --pcc-k 0.99 --pcc-v 0.98
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

# Appendix A / BRINGUP_RECIPE.md P7: V is consistently the weaker of the two at a bf8_b cache dtype.
DEFAULT_PCC_K = 0.99
DEFAULT_PCC_V = 0.98


def hf_to_meta_lane_permutation(head_dim: int, rotary_dim: int | None = None) -> torch.Tensor:
    """Lane index map turning an **HF half-split** rotary layout into **Meta interleaved**.

    ``golden_k[..., perm]`` is directly comparable with a device K tensor.

    HF puts frequency ``i`` at lanes ``(i, i + rotary_dim/2)``; Meta puts it at ``(2i, 2i + 1)``.
    So Meta lane ``m`` must read HF lane ``half*(m % 2) + (m // 2)``. Lanes at or above
    ``rotary_dim`` (none for Llama, which is full rotary: ``rotary_dim == head_dim == 128``) are
    identity — kept so the helper is correct for a partial-rotary model too.
    """
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    assert rotary_dim % 2 == 0 and rotary_dim <= head_dim, f"bad rotary_dim {rotary_dim} for head_dim {head_dim}"
    half = rotary_dim // 2
    lanes = list(range(head_dim))
    for m in range(rotary_dim):
        lanes[m] = half * (m % 2) + (m // 2)
    return torch.tensor(lanes, dtype=torch.long)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two tensors, flattened, in fp64.

    Local rather than ``models.common.utility_functions.comp_pcc`` on purpose: this script must run
    as a plain host script with no ``ttnn`` import (``comp_pcc``'s module pulls in the device
    stack), which is what makes ``G-GOLDEN`` a host gate.
    """
    x = a.reshape(-1).double()
    y = b.reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom == 0:
        # Two constant tensors correlate perfectly only if they are the same constant.
        return 1.0 if torch.allclose(a.double(), b.double()) else 0.0
    return float(torch.dot(x, y) / denom)


def _read_layer(kv_dir: Path, layer_idx: int, *, n_tokens=None):
    path = kv_dir / f"layer_{layer_idx}.safetensors"
    assert path.exists(), f"missing {path}"
    with safe_open(str(path), framework="pt") as handle:
        keys = set(handle.keys())
        k_name, v_name = f"key_cache_layer_{layer_idx}", f"value_cache_layer_{layer_idx}"
        assert k_name in keys and v_name in keys, f"{path}: has {sorted(keys)}, want {k_name} and {v_name}"
        k = handle.get_tensor(k_name).float()
        v = handle.get_tensor(v_name).float()
    if n_tokens is not None:
        k, v = k[:, :, :n_tokens, :], v[:, :, :n_tokens, :]
    return k, v


def verify_structure(trace_dir: Path, *, out=print) -> tuple[bool, dict]:
    """Structural checks over ``metadata.json`` and every layer file. Returns ``(ok, metadata)``."""
    ok = True
    meta_path = trace_dir / "metadata.json"
    if not meta_path.exists():
        out(f"  FAIL metadata.json not found at {meta_path}")
        return False, {}
    try:
        with open(meta_path) as fh:
            metadata = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        out(f"  FAIL could not parse metadata.json: {exc}")
        return False, {}

    required = ["token_ids", "n_tokens", "num_layers", "num_kv_heads", "head_dim"]
    missing = [k for k in required if k not in metadata]
    if missing:
        out(f"  FAIL metadata is missing {missing}")
        return False, metadata

    n_tokens = int(metadata["n_tokens"])
    num_layers = int(metadata["num_layers"])
    expected = (1, int(metadata["num_kv_heads"]), n_tokens, int(metadata["head_dim"]))
    if len(metadata["token_ids"]) != n_tokens:
        out(f"  FAIL len(token_ids)={len(metadata['token_ids'])} != n_tokens={n_tokens}")
        ok = False
    out(f"  metadata OK: {num_layers} layers, {n_tokens} tokens, K/V shape {expected}")

    kv_dir = trace_dir / "kv_cache"
    for layer_idx in range(num_layers):
        path = kv_dir / f"layer_{layer_idx}.safetensors"
        if not path.exists():
            out(f"  FAIL layer {layer_idx}: {path} missing")
            ok = False
            continue
        try:
            k, v = _read_layer(kv_dir, layer_idx)
        except AssertionError as exc:
            out(f"  FAIL layer {layer_idx}: {exc}")
            ok = False
            continue
        for name, tensor in (("K", k), ("V", v)):
            if tuple(tensor.shape) != expected:
                out(f"  FAIL layer {layer_idx} {name}: shape {tuple(tensor.shape)} != {expected}")
                ok = False
            if not torch.isfinite(tensor).all():
                out(f"  FAIL layer {layer_idx} {name}: non-finite values")
                ok = False
            # A zeroed / constant tensor scores a *nan* PCC rather than a low one, so it must be
            # caught here or a dead producer masquerades as a broken comparison.
            if float(tensor.std()) == 0.0:
                out(f"  FAIL layer {layer_idx} {name}: constant tensor (std == 0)")
                ok = False
    if ok:
        out(f"  all {num_layers} layer files present, correctly shaped, finite and non-constant")
    return ok, metadata


def compare_device_dump(trace_dir: Path, dump_dir: Path, *, pcc_k=DEFAULT_PCC_K, pcc_v=DEFAULT_PCC_V, out=print):
    """Per-layer K/V PCC of a device dump against the golden. Returns ``(ok, rows, summary)``.

    ``rows`` is ``[(layer_idx, pcc_k, pcc_v), ...]``; ``summary`` carries the min and mean of each.
    """
    ok, metadata = verify_structure(trace_dir, out=out)
    if not ok:
        return False, [], {}

    dump_meta_path = dump_dir / "metadata.json"
    assert dump_meta_path.exists(), f"device dump has no metadata.json at {dump_meta_path}"
    with open(dump_meta_path) as fh:
        dump_meta = json.load(fh)
    convention = dump_meta.get("convention", "meta")
    assert convention in ("meta", "hf"), f"device dump declares convention={convention!r}, want 'meta' or 'hf'"
    n_tokens = int(dump_meta.get("n_tokens", metadata["n_tokens"]))
    layer_indices = list(dump_meta.get("layers", range(int(metadata["num_layers"]))))
    head_dim = int(metadata["head_dim"])
    rotary_dim = int(dump_meta.get("rotary_dim", head_dim))
    perm = hf_to_meta_lane_permutation(head_dim, rotary_dim) if convention == "meta" else None

    out("")
    out(f"[device] dump {dump_dir}")
    out(f"[device] convention={convention} n_tokens={n_tokens} layers={len(layer_indices)}")
    out(f"[device] thresholds: K >= {pcc_k}, V >= {pcc_v}")
    out("")
    out("  layer |        K PCC |        V PCC | verdict")
    out("  ------+--------------+--------------+--------")

    rows = []
    all_ok = True
    for layer_idx in layer_indices:
        g_k, g_v = _read_layer(trace_dir / "kv_cache", layer_idx, n_tokens=n_tokens)
        d_k, d_v = _read_layer(dump_dir, layer_idx, n_tokens=n_tokens)
        if perm is not None:
            g_k = g_k[..., perm]
        assert g_k.shape == d_k.shape, f"layer {layer_idx}: golden K {tuple(g_k.shape)} vs device {tuple(d_k.shape)}"
        assert g_v.shape == d_v.shape, f"layer {layer_idx}: golden V {tuple(g_v.shape)} vs device {tuple(d_v.shape)}"
        k_score, v_score = pcc(g_k, d_k), pcc(g_v, d_v)
        good = k_score >= pcc_k and v_score >= pcc_v
        all_ok = all_ok and good
        rows.append((layer_idx, k_score, v_score))
        out(f"  {layer_idx:>5} | {k_score:>12.5f} | {v_score:>12.5f} | {'PASS' if good else 'FAIL'}")

    ks = [r[1] for r in rows]
    vs = [r[2] for r in rows]
    summary = {
        "n_layers": len(rows),
        "min_k": min(ks),
        "mean_k": sum(ks) / len(ks),
        "min_v": min(vs),
        "mean_v": sum(vs) / len(vs),
        "argmin_k": rows[ks.index(min(ks))][0],
        "argmin_v": rows[vs.index(min(vs))][0],
    }
    out("  ------+--------------+--------------+--------")
    out(
        f"  min   | {summary['min_k']:>12.5f} | {summary['min_v']:>12.5f} | (layers {summary['argmin_k']}/{summary['argmin_v']})"
    )
    out(f"  mean  | {summary['mean_k']:>12.5f} | {summary['mean_v']:>12.5f} |")
    out("")
    out(
        f"[device] {'PASS' if all_ok else 'FAIL'}: min K = {summary['min_k']:.5f} (>= {pcc_k}), min V = {summary['min_v']:.5f} (>= {pcc_v})"
    )
    return all_ok, rows, summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Verify a golden KV trace and optionally PCC a device dump against it",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("trace_dir", type=Path, help="Golden trace directory (from generate_golden_kv_cache.py)")
    ap.add_argument("--device-dump", type=Path, default=None, help="Device KV dump directory to score")
    ap.add_argument("--pcc-k", type=float, default=DEFAULT_PCC_K)
    ap.add_argument("--pcc-v", type=float, default=DEFAULT_PCC_V)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    trace_dir = args.trace_dir
    if not trace_dir.is_dir():
        print(f"ERROR: {trace_dir} is not a directory", file=sys.stderr)
        return 2

    print(f"[verify] golden trace {trace_dir}")
    if args.device_dump is None:
        ok, _ = verify_structure(trace_dir)
        print(f"[verify] {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    ok, _rows, _summary = compare_device_dump(trace_dir, args.device_dump, pcc_k=args.pcc_k, pcc_v=args.pcc_v)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
