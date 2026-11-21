# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of nlp_create_qkv_heads (verifier, Phase 0).

The op moves values and computes nothing (rule R1): every output element must equal its input element bit for
bit, so the expected baseline is PCC = 1, every error 0, every got/true ratio exactly 1. The test measures it on
four first-phase shapes (small, the two traced Qwen3-VL calls, one long) and gates on PCC and relative RMS
together, as eval.metrics.check_output does, plus bit equality.

Under pytest-xdist / --forked a test's output is not shown, so each measurement is recorded with
record_property and appended to the JSON file named by NLPQKV_BASELINE_OUT (default under /tmp).
"""

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.nlp_create_qkv_heads import nlp_create_qkv_heads

from eval.specs.nlp_create_qkv_heads.spec import reference

OUT = os.environ.get("NLPQKV_BASELINE_OUT", "/tmp/nlp_create_qkv_heads_precision_baseline.jsonl")

SHAPES = [
    ((1, 1, 32, 96), 1, 1),  # small: one head each, one tile per head
    ((1, 1, 128, 3072), 16, 16),  # traced: Qwen3-VL vision tower
    ((1, 1, 128, 6144), 32, 8),  # traced: Qwen3-VL text model (grouped-query)
    ((1, 1, 4096, 96), 1, 1),  # larger: 4096 tokens
]


def _id(entry):
    shape, nh, nkv = entry
    return "x".join(map(str, shape)) + f"_h{nh}_kv{nkv}"


def _ulp_max(got, want):
    """Largest distance in bfloat16 units in the last place (sign-magnitude ordered bit patterns)."""

    def ordered(t):
        bits = t.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return int((ordered(got) - ordered(want)).abs().max())


def _measure(got, want):
    got64, want64 = got.double(), want.double()
    _, pcc = comp_pcc(want64, got64)
    _, allclose = comp_allclose(want64, got64)
    diff = (got64 - want64).abs()
    rel_rms = float(torch.sqrt((diff**2).mean()) / want64.std())
    mask = torch.isfinite(want64) & (want64 != 0)
    ratio = got64[mask] / want64[mask]
    return {
        "pcc": float(pcc),
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "rel_rms": rel_rms,
        "ulp_max": _ulp_max(got, want),
        "ratio_median": float(ratio.median()),
        "ratio_p5": float(torch.quantile(ratio, 0.05)),
        "ratio_p95": float(torch.quantile(ratio, 0.95)),
        "allclose": allclose,
        "bit_exact": bool(torch.equal(got64, want64)),
    }


@pytest.mark.parametrize("entry", SHAPES, ids=_id)
def test_precision_baseline(device, entry, record_property):
    shape, nh, nkv = entry
    torch.manual_seed(42)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    tx = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    outputs = nlp_create_qkv_heads(
        tx, num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    rows = []
    for name, out, want in zip("qkv", outputs, reference(x, nh, nkv, transpose_k_heads=False)):
        got = ttnn.to_torch(out)
        assert list(got.shape) == list(want.shape), (name, got.shape, want.shape)
        m = _measure(got, want)
        m.update({"shape": _id(entry), "tensor": name})
        rows.append(m)
        record_property(f"{name}_metrics", json.dumps(m))
    with open(OUT, "a") as f:
        for m in rows:
            f.write(json.dumps(m) + "\n")
    for m in rows:
        # check_output's gate (PCC and relative RMS together), at this op's legacy default (0.9999, 0.0),
        # then the movement class's own requirement: bit exactness.
        assert m["pcc"] >= 0.9999 and m["rel_rms"] <= 0.0, m
        assert m["bit_exact"] and m["ulp_max"] == 0, m
