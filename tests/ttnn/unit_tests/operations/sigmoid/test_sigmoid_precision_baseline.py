# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of sigmoid (verifier): PCC, max / mean abs error, relative RMS, bf16 ULP distance to
the correctly rounded float64 reference, and the got/true ratio spread (the scale-bug detector).

Each case appends one JSON line to generated/sigmoid_precision_baseline.jsonl (test output is not shown under
the simulator's forked workers) and records the same numbers with record_property.

Gates: the shared check's (PCC, relative RMS) for (bfloat16, fp32_dest_acc_en=True), DEFAULT_TOLERANCES in
eval/metrics.py, and the specification's numeric entry for bfloat16 (NUMERIC_RESOLVED): <= 2 ULP everywhere,
<= 1 ULP when every input lies in [0, 1].
"""

import json
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.sigmoid import sigmoid

PCC_MIN, REL_RMS_MAX = 0.995, 0.04  # DEFAULT_TOLERANCES[(bfloat16, True)]
OUT = Path(__file__).resolve().parents[5] / "generated" / "sigmoid_precision_baseline.jsonl"

SHAPES = [(1, 1, 32, 64), (1, 80, 400), (2, 4, 128, 512), (1, 80, 8400)]
# name -> (generator, ULP bound of the numeric entry)
DISTRIBUTIONS = {
    "normal": (lambda shape: torch.randn(shape), 2),
    "normal_x10": (lambda shape: 10.0 * torch.randn(shape), 2),  # both saturated tails
    "uniform01": (lambda shape: torch.rand(shape), 1),  # the entry's [0, 1] case
}


def _ordered_bf16(x):
    """bf16 bit patterns mapped onto a monotone integer line, so a difference counts ULPs."""
    bits = x.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def _metrics(got, ref64):
    got64 = got.double()
    err = got64 - ref64
    _, pcc = comp_pcc(got64, ref64)
    _, allclose = comp_allclose(got64, ref64)
    rel_rms = (err.pow(2).mean().sqrt() / ref64.std()).item()
    ulp = (_ordered_bf16(got) - _ordered_bf16(ref64)).abs()
    nz = ref64.abs() > 0
    ratio = got64[nz] / ref64[nz]
    q = torch.quantile(ratio.float()[: 1 << 24], torch.tensor([0.05, 0.5, 0.95])).tolist()
    return {
        "pcc": float(pcc),
        "max_abs_err": err.abs().max().item(),
        "mean_abs_err": err.abs().mean().item(),
        "rel_rms": rel_rms,
        "allclose": allclose,
        "ulp_max": int(ulp.max().item()),
        "ulp_hist": {str(k): int((ulp == k).sum().item()) for k in range(int(ulp.max().item()) + 1)},
        "ratio_p05": q[0],
        "ratio_median": q[1],
        "ratio_p95": q[2],
    }


def _record(record_property, row):
    for k, v in row.items():
        record_property(k, v)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")


@pytest.mark.parametrize("dist", list(DISTRIBUTIONS))
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_precision_baseline(device, shape, dist, record_property):
    gen, ulp_bound = DISTRIBUTIONS[dist]
    torch.manual_seed(42)
    x = gen(shape).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(sigmoid(t))
    m = _metrics(got, torch.sigmoid(x.double()))
    _record(record_property, {"shape": list(shape), "dist": dist, "config": "default", **m})

    assert m["pcc"] >= PCC_MIN, m
    assert m["rel_rms"] <= REL_RMS_MAX, m
    assert m["ulp_max"] <= ulp_bound, m


# The compute config's other fields are passed through, not gated (R3; spec GUIDANCE "pass it as-is"):
# dst_full_sync_en reaches double_buffer_dest and doubles block_tiles, the fidelity and the approximation flag
# reach the hardware config, and the accurate sigmoid body holds the same bound under each.
CONFIGS = {
    "full_sync": dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True),
    "lofi_approx": dict(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=True),
}


@pytest.mark.parametrize("config", list(CONFIGS))
def test_config_pass_through(device, config, record_property):
    shape = (1, 80, 400)
    torch.manual_seed(42)
    x = torch.randn(shape).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(sigmoid(t, compute_kernel_config=ttnn.ComputeConfigDescriptor(**CONFIGS[config])))
    m = _metrics(got, torch.sigmoid(x.double()))
    _record(record_property, {"shape": list(shape), "dist": "normal", "config": config, **m})
    assert m["ulp_max"] <= 2, m
