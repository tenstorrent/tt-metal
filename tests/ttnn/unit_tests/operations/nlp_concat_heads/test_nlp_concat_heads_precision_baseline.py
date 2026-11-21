# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of nlp_concat_heads.

The op only moves values (spec R1, NUMERIC: every cell graded `exact`), so the baseline is exactness:
PCC 1, every absolute error 0, relative RMS 0, 0 ULP, and a got/true ratio of exactly 1 everywhere (the
scale-bug detector: a tight cluster around a constant other than 1 would mean a structural bug).

- Supported cells (bfloat16, DRAM -> DRAM, tile-aligned, one sequence), through the public entry point.
- float32, through the program builder (validate() bypassed: float32 is not yet in SUPPORTED). This is
  evidence for rule R1 ahead of the refinement that widens the dtype: no element passes through DEST.

Metrics go to `record_property` and, when NCH_PRECISION_OUT names a directory, to one JSON file per case
(test output is not shown under the simulator's workers).
"""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.nlp_concat_heads import nlp_concat_heads
from ttnn.operations.nlp_concat_heads.nlp_concat_heads_program_artifacts import create_program_artifacts

SHAPES = [(1, 4, 32, 64), (1, 4, 32, 128), (1, 32, 128, 64), (1, 64, 32, 128)]
_INT_VIEW = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


def _reference(x):
    b, h, s, d = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, 1, s, h * d)


def _ulp_distance(got, expected):
    """Max ULP distance, on the sign-magnitude integers of the stored format."""
    iv = _INT_VIEW[expected.dtype]
    bits = 8 * expected.element_size()

    def ordered(t):
        i = t.contiguous().view(iv).to(torch.int64)
        return torch.where(i < 0, -(i & ((1 << (bits - 1)) - 1)), i)

    return int((ordered(got) - ordered(expected)).abs().max())


def _metrics(got, expected):
    _, pcc = comp_pcc(expected.float(), got.float(), pcc=1.0)
    _, allclose_msg = comp_allclose(expected.float(), got.float(), rtol=0.0, atol=0.0)
    diff = (got.double() - expected.double()).abs()
    rms = float(diff.pow(2).mean().sqrt())
    rel_rms = rms / float(expected.double().std())
    finite = torch.isfinite(expected) & (expected != 0)
    ratio = got.double()[finite] / expected.double()[finite]
    return {
        "pcc": float(pcc),
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "rel_rms": rel_rms,
        "ulp_max": _ulp_distance(got, expected),
        "ratio_median": float(ratio.median()),
        "ratio_p5": float(ratio.quantile(0.05)) if ratio.numel() <= 2**24 else float(ratio.min()),
        "ratio_p95": float(ratio.quantile(0.95)) if ratio.numel() <= 2**24 else float(ratio.max()),
        "ratio_std": float(ratio.std()),
        "allclose": allclose_msg,
    }


def _report(record_property, case, m):
    for k, v in m.items():
        record_property(k, v)
    out = os.environ.get("NCH_PRECISION_OUT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"{case}.json").write_text(json.dumps({"case": case, **m}, indent=1))


def _assert_exact(m):
    assert m["pcc"] == pytest.approx(1.0, abs=0.0) or m["pcc"] >= 1.0 - 1e-12, m
    assert m["max_abs_err"] == 0.0 and m["rel_rms"] == 0.0 and m["ulp_max"] == 0, m
    assert m["ratio_median"] == 1.0 and m["ratio_std"] == 0.0, m


def _case_id(shape):
    return "x".join(map(str, shape))


@pytest.mark.parametrize("shape", SHAPES, ids=_case_id)
def test_precision_baseline_bfloat16(device, shape, record_property):
    torch.manual_seed(1234)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    x_dev = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    got = ttnn.to_torch(nlp_concat_heads(x_dev, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    m = _metrics(got, _reference(x))
    _report(record_property, f"bf16_{_case_id(shape)}", m)
    _assert_exact(m)


@pytest.mark.parametrize("shape", SHAPES[:2] + SHAPES[3:], ids=_case_id)
def test_precision_float32_program(device, shape, record_property):
    """float32 through the program (not the entry point): R1 evidence for the dtype refinement."""
    torch.manual_seed(4321)
    x = torch.randn(*shape, dtype=torch.float32)
    x_dev = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    b, h, s, d = shape
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape([b, 1, s, h * d]), ttnn.float32, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        device,
    )
    spec, run_args, idx = create_program_artifacts(x_dev, out)
    got = ttnn.to_torch(ttnn.generic_op([x_dev, out], spec, run_args, idx))
    m = _metrics(got, _reference(x))
    _report(record_property, f"fp32_{_case_id(shape)}", m)
    _assert_exact(m)
