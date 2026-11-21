# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of nlp_create_qkv_heads_decode (verifier, Phase 0).

The op only moves values (spec rule R1), and its numeric requirement is `exact` for each of q, k and v. This
test measures, per shape and per output tensor: PCC (comp_pcc), max / mean absolute error (comp_allclose and
the difference), relative RMS (RMS error over the reference's std, as eval.metrics.check_output reads it), the
largest ULP distance (bit patterns of the bfloat16 values), and the got/true ratio spread (the scale-bug
detector: a tight cluster around a non-1.0 constant would mean a structural bug, not rounding).

It gates on bit-exactness, which implies PCC = 1 and relative RMS = 0. The measurements are written to
`generated/precision_baseline/nlp_create_qkv_heads_decode.jsonl` (a test's own output is hidden under the
simulator's workers) and recorded with `record_property`.
"""

import json
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.nlp_create_qkv_heads_decode import nlp_create_qkv_heads_decode

HEIGHT_SHARDED_L1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
RESULTS = Path("generated/precision_baseline/nlp_create_qkv_heads_decode.jsonl")

# (shape, num_heads, num_kv_heads): small, the two traced calls, the widest first-phase row.
SHAPES = [
    ((1, 1, 1, 96), 1, 1),
    ((1, 1, 1, 3072), 32, 8),  # Llama 3.2 1B
    ((1, 1, 1, 6144), 32, 8),  # Qwen3-VL
    ((1, 1, 1, 8192), 16, 8),
]


def _expected(x, nh, nkv):
    users, width = x.shape[2], x.shape[3]
    hd = width // (nh + 2 * nkv)
    rows = x[0, 0]
    return [rows[:, a * hd:(a + n) * hd].reshape(1, users, n, hd) for a, n in ((0, nh), (nh, nkv), (nh + nkv, nkv))]


def _ulp_max(got, want):
    """Largest distance in units in the last place between two bfloat16 tensors (bit patterns, sign-magnitude)."""

    def ordered(t):
        bits = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return int((ordered(got) - ordered(want)).abs().max())


def _measure(got, want):
    g, w = got.double(), want.double()
    _, pcc = comp_pcc(w, g)
    _, allclose = comp_allclose(w, g)
    err = g - w
    rel_rms = float(err.pow(2).mean().sqrt() / w.std()) if w.numel() > 1 and float(w.std()) > 0 else 0.0
    mask = torch.isfinite(w) & (w != 0)
    ratio = (g[mask] / w[mask]).float()
    return {
        "pcc": float(pcc),
        "max_abs": float(err.abs().max()),
        "mean_abs": float(err.abs().mean()),
        "rel_rms": rel_rms,
        "ulp_max": _ulp_max(got, want),
        "ratio_median": float(ratio.median()),
        "ratio_p5": float(torch.quantile(ratio, 0.05)),
        "ratio_p95": float(torch.quantile(ratio, 0.95)),
        "allclose": allclose,
        "bit_exact": bool(torch.equal(got, want)),
    }


@pytest.mark.parametrize("entry", SHAPES, ids=lambda e: "x".join(map(str, e[0])) + f"_h{e[1]}_kv{e[2]}")
def test_precision_baseline(device, record_property, entry):
    shape, nh, nkv = entry
    torch.manual_seed(1234)
    x = torch.randn(shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = nlp_create_qkv_heads_decode(t, num_heads=nh, num_kv_heads=nkv, memory_config=HEIGHT_SHARDED_L1)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    for name, got, want in zip("qkv", out, _expected(x, nh, nkv)):
        got = ttnn.to_torch(got)
        assert list(got.shape) == list(want.shape), name
        m = _measure(got, want)
        record_property(f"{name}_metrics", json.dumps(m))
        with RESULTS.open("a") as f:
            f.write(json.dumps({"shape": list(shape), "num_heads": nh, "num_kv_heads": nkv, "tensor": name, **m}) + "\n")
        # The requirement is `exact`: every element is one input element, unchanged.
        assert m["bit_exact"], f"{name}: {m}"
        assert m["pcc"] >= 0.9999 and m["rel_rms"] == 0.0 and m["ulp_max"] == 0, f"{name}: {m}"
