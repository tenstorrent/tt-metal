# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of to_memory_config (verifier, Phase 0).

The op moves bytes and computes nothing (spec rule R1), so the expected baseline is exact: PCC 1, every
abs error 0, relative RMS 0 (the spec grades every cell at relative RMS <= 0), 0 ULP, and a got/true
ratio of exactly 1 everywhere. One case per regime (R-bal, the ragged row-major slot cap, R-out, R-in),
small to large. Results are appended as JSON lines to $TMC_BASELINE_OUT (default
/tmp/to_memory_config_precision_baseline.jsonl), since a test's output is not shown under forked workers.
"""

import json
import os

import pytest
import torch

import ttnn
from eval import memory
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.to_memory_config import to_memory_config

OUT = os.environ.get("TMC_BASELINE_OUT", "/tmp/to_memory_config_precision_baseline.jsonl")


def _config(where, shape, layout, device):
    if where == "L1/height":
        return memory.place(memory.build("L1", "height"), shape, layout, device)
    return {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}[where]


CASES = [
    # (id, shape, layout, from, to)
    ("small_tile_bal", (32, 64), ttnn.TILE_LAYOUT, "DRAM", "L1"),
    ("ragged_rm_bal", (1, 1, 17, 50), ttnn.ROW_MAJOR_LAYOUT, "L1", "DRAM"),
    ("medium_tile_out", (1, 1, 400, 512), ttnn.TILE_LAYOUT, "DRAM", "L1/height"),
    ("large_tile_in", (1, 1, 1600, 256), ttnn.TILE_LAYOUT, "L1/height", "DRAM"),
]


def _ulp_bf16(a, b):
    """Max distance in bfloat16 units in the last place (sign-magnitude bit patterns made monotonic)."""
    def key(x):
        bits = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)
    return int((key(a) - key(b)).abs().max())


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_precision_baseline(device, case, record_property):
    name, shape, layout, src, dst = case
    torch.manual_seed(0)
    expected = torch.randn(shape).to(torch.bfloat16)
    x = ttnn.from_torch(expected, dtype=ttnn.bfloat16, layout=layout, device=device,
                        memory_config=_config(src, shape, layout, device))
    actual = ttnn.to_torch(to_memory_config(x, _config(dst, shape, layout, device)))

    _, pcc = comp_pcc(expected, actual, pcc=0.9999)
    _, allclose = comp_allclose(expected, actual)
    diff = (actual.double() - expected.double()).abs()
    rel_rms = float(torch.sqrt((diff ** 2).mean()) / expected.double().std())
    finite = torch.isfinite(expected) & (expected != 0)
    ratio = actual.double()[finite] / expected.double()[finite]
    result = {
        "case": name, "shape": list(shape), "layout": str(layout), "from": src, "to": dst,
        "pcc": float(pcc), "max_abs_err": float(diff.max()), "mean_abs_err": float(diff.mean()),
        "rel_rms": rel_rms, "max_ulp": _ulp_bf16(actual, expected), "allclose": allclose,
        "ratio_median": float(ratio.median()), "ratio_p5": float(ratio.quantile(0.05)),
        "ratio_p95": float(ratio.quantile(0.95)),
    }
    for k, v in result.items():
        record_property(k, v)
    with open(OUT, "a") as f:
        f.write(json.dumps(result) + "\n")

    # The op's requirement is exact (spec NUMERIC legacy_default = (0.9999, 0.0)): gate PCC and RMS together.
    assert pcc >= 0.9999, result
    assert rel_rms <= 0.0, result
    assert result["max_ulp"] == 0 and torch.equal(actual, expected), result
