# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of paged_scaled_dot_product_attention_decode (first phase: bfloat16, both DEST widths).

Four shapes (one tile wide; Llama's per-head call; Qwen's two-key/value-head call, D = 128; Llama's captured call,
4096 positions on 8 clusters), each at a fixed position near the end of what the page table reaches (most keys
attended) and at both accumulators. Measured against the specification's float64 reference: PCC and max / mean
absolute error (comp_pcc / comp_allclose), relative RMS (RMS error / reference std, as eval/metrics.py), and the
got/true ratio spread (median, p5, p95 over elements whose reference is not near 0) — the scale-bug detector.

Gated on the cell's (PCC, relative RMS) row of NUMERIC["legacy_tolerances"]. Each case also writes its numbers to
generated/precision_baseline/paged_scaled_dot_product_attention_decode/<id>.json (a test's output is not shown under
the simulator's workers).
"""

import json
import os
import re
from pathlib import Path

import pytest
import torch
import ttnn

from eval import golden_runner
from eval.specs.paged_scaled_dot_product_attention_decode import spec as S
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.paged_scaled_dot_product_attention_decode import paged_scaled_dot_product_attention_decode

OP = "paged_scaled_dot_product_attention_decode"
CELL = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT, "memory_config": "L1", "sharding": "height",
        "output_memory": "DRAM"}
OUT_DIR = Path(os.environ.get("TT_METAL_HOME", ".")) / "generated/precision_baseline" / OP

SHAPES = [
    ("one_tile_wide", S.ONE_TILE_WIDE, 63),
    ("llama_split", S.LLAMA_SPLIT, 250),
    ("qwen_two_kv_heads", S.QWEN_TWO_KV_HEADS, 250),
    ("llama_captured", S.LLAMA_CAPTURED, 4000),
]


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32acc", "bf16acc"])
@pytest.mark.parametrize("name,inputs,position", SHAPES, ids=[s[0] for s in SHAPES])
def test_precision_baseline(device, name, inputs, position, fp32_dest_acc_en):
    axes = {**CELL, "fp32_dest_acc_en": fp32_dest_acc_en}
    spec = golden_runner.bound(OP).spec
    cell, plan = golden_runner.build_cell(OP, inputs, axes, device, {"positions": [position]}, seed=1234)
    dev = golden_runner.device_tensors(plan, cell.tensors, device)
    call = golden_runner.bind_call(spec, dev, cell.kwargs)
    output = paged_scaled_dot_product_attention_decode(*call.args, **call.kwargs)
    expected = spec.reference(**{p: t.to(torch.float64) for p, t in cell.tensors.items()},
                              **{k: v for k, v in cell.kwargs.items() if k == "scale"})

    heads = inputs[0][2]
    got = ttnn.to_torch(output).to(torch.float64)[..., :heads, :]
    want = expected[..., :heads, :]

    _, pcc = comp_pcc(want, got)
    _, allclose = comp_allclose(got, want)
    max_abs = float(re.search(r"Max ATOL Delta: ([0-9.eE+-]+|nan|inf)", allclose).group(1))
    diff = got - want
    mean_abs = diff.abs().mean().item()
    rel_rms = (diff.pow(2).mean().sqrt() / want.std()).item()
    keep = torch.isfinite(got) & (want.abs() > 0.05 * want.std())
    ratio = (got[keep] / want[keep]).flatten()
    r_med, r_p5, r_p95 = (torch.quantile(ratio, q).item() for q in (0.5, 0.05, 0.95))

    pcc_min, rms_max = golden_runner.legacy_tolerance(spec, {**axes, "kv_heads_mode": "grouped"})
    record = {"shape": name, "inputs": [list(x) if isinstance(x, tuple) else x for x in inputs],
              "position": position, "fp32_dest_acc_en": fp32_dest_acc_en, "pcc": pcc, "max_abs": max_abs,
              "mean_abs": mean_abs, "rel_rms": rel_rms, "ratio_median": r_med, "ratio_p5": r_p5,
              "ratio_p95": r_p95, "tolerance": [pcc_min, rms_max]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}-{'fp32acc' if fp32_dest_acc_en else 'bf16acc'}.json").write_text(json.dumps(record))

    assert torch.isfinite(got).all()
    assert pcc >= pcc_min, record
    assert rms_max is None or rel_rms <= rms_max, record
    # A uniform scale error keeps PCC high: the ratio's median must sit on 1.
    assert abs(r_med - 1.0) < 0.02, record
