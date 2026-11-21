# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of softmax over its supported cells (bfloat16, fp32 DEST, TILE, rank 4, dim=-2).

Gates PCC and relative RMS together, as eval/metrics.py check_output does (relative RMS = RMS error / reference
std), at NUMERIC["legacy_tolerances"][(bfloat16, True)] of the specification. Also records the max / mean abs
error, the bfloat16 ULP distance from the correctly rounded result, and the got/true ratio spread (the scale-bug
detector: a tight cluster around a non-1.0 constant is a structural bug, a spread centered on 1.0 is rounding).

Output of a test is not shown under workers / forking on the simulator: each case writes its numbers to
$SOFTMAX_PRECISION_DIR (default /tmp/softmax_precision_baseline)/<case>.json and records them as properties.
"""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.softmax import softmax

from eval.specs.softmax.spec import NUMERIC

PCC_MIN, REL_RMS_MAX = NUMERIC["legacy_tolerances"][(ttnn.bfloat16, True)]
OUT_DIR = Path(os.environ.get("SOFTMAX_PRECISION_DIR", "/tmp/softmax_precision_baseline"))

# (shape, memory_config, input scale): the two YOLOv8 calls, a small aligned one, a medium batched one, the tall
# resident column, and the scale-10 distribution of rule R1.
CASES = [
    ((1, 4, 16, 400), "L1", 1.0),
    ((1, 4, 16, 8400), "DRAM", 1.0),
    ((1, 1, 64, 128), "DRAM", 1.0),
    ((2, 4, 128, 512), "DRAM", 1.0),
    ((1, 1, 2048, 256), "DRAM", 1.0),
    ((1, 4, 16, 400), "DRAM", 10.0),
]


def _bf16_ulp_distance(got_bf16, ref64):
    """ULP distance in bfloat16 between the device result and the float64 reference rounded once to bfloat16.
    Softmax outputs are >= 0, so the int16 bit patterns are monotonic in the value."""
    a = got_bf16.to(torch.bfloat16).view(torch.int16).to(torch.int64)
    b = ref64.to(torch.bfloat16).view(torch.int16).to(torch.int64)
    return (a - b).abs().double()


@pytest.mark.parametrize(
    "shape,memory,scale", CASES, ids=["x".join(map(str, s)) + f"-{m}-s{sc:g}" for s, m, sc in CASES]
)
def test_precision_baseline(device, record_property, shape, memory, scale):
    torch.manual_seed(42)
    t = (torch.randn(shape, dtype=torch.float32) * scale).to(torch.bfloat16)
    mem = ttnn.L1_MEMORY_CONFIG if memory == "L1" else ttnn.DRAM_MEMORY_CONFIG
    x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=mem)
    got_bf16 = ttnn.to_torch(softmax(x, dim=-2))
    got = got_bf16.double()
    ref = torch.softmax(t.double(), dim=-2)

    _, pcc = comp_pcc(ref, got)
    _, allclose_msg = comp_allclose(ref, got)
    abs_err = (got - ref).abs()
    rel_rms = ((got - ref).pow(2).mean().sqrt() / ref.std()).item()
    ulp = _bf16_ulp_distance(got_bf16, ref)

    nz = ref.abs() > 1e-6
    ratio = got[nz] / ref[nz]
    q = torch.quantile(ratio.float()[: 1 << 24], torch.tensor([0.05, 0.5, 0.95])).tolist()

    result = {
        "shape": list(shape),
        "memory": memory,
        "scale": scale,
        "pcc": float(pcc),
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "rel_rms": rel_rms,
        "allclose": allclose_msg,
        "ulp_median": ulp.median().item(),
        "ulp_p99": torch.quantile(ulp.float()[: 1 << 24], 0.99).item(),
        "ulp_max": ulp.max().item(),
        "ratio_p5": q[0],
        "ratio_median": q[1],
        "ratio_p95": q[2],
        "ratio_std": ratio.std().item(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / ("x".join(map(str, shape)) + f"-{memory}-s{scale:g}.json")).write_text(json.dumps(result, indent=1))
    for k, v in result.items():
        record_property(k, v)

    assert pcc >= PCC_MIN and rel_rms <= REL_RMS_MAX, result
    # Scale-bug detector: the bulk of the elements must sit around 1.0, not around another constant.
    assert abs(result["ratio_median"] - 1.0) < 0.01, result
