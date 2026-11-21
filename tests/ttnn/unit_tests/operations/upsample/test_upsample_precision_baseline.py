# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of upsample (verifier, Phase 0).

The op moves values and computes nothing (spec rule R1); every cell's numeric requirement is `exact`
(NUMERIC_RESOLVED, class `movement`). The baseline therefore records the usual metrics (PCC, max / mean
absolute error, relative RMS, got/true ratio spread) and gates on bit-exactness, the requirement itself.

Under pytest workers and forking a test's stdout is not shown, so each case writes its metrics to
$UPSAMPLE_PRECISION_OUT (default /tmp/upsample_precision_baseline/) and records them with record_property.
"""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.upsample import upsample

OUT_DIR = Path(os.environ.get("UPSAMPLE_PRECISION_OUT", "/tmp/upsample_precision_baseline"))

# small / medium / the model's calls (YOLOv8) / batched with factor 3; DRAM and L1.
CASES = [
    ((1, 16, 16, 32), 2, ttnn.DRAM_MEMORY_CONFIG),
    ((2, 8, 8, 128), 3, ttnn.L1_MEMORY_CONFIG),
    ((1, 20, 20, 512), (2, 2), ttnn.DRAM_MEMORY_CONFIG),
    ((1, 40, 40, 512), (2, 2), ttnn.DRAM_MEMORY_CONFIG),
]


def _id(case):
    shape, sf, mc = case
    s = str(sf) if isinstance(sf, int) else "x".join(map(str, sf))
    return "x".join(map(str, shape)) + f"_s{s}_" + ("L1" if mc == ttnn.L1_MEMORY_CONFIG else "DRAM")


@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_upsample_precision_baseline(device, case, record_property):
    shape, sf, mc = case
    sh, sw = (sf, sf) if isinstance(sf, int) else sf
    torch.manual_seed(1234)
    t = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)  # spec NUMERIC: normal, stored in the cell dtype
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    y = upsample(x, sf)
    got = ttnn.to_torch(y).double()
    ref = t.double().repeat_interleave(sh, dim=1).repeat_interleave(sw, dim=2)  # spec reference(): float64

    assert tuple(got.shape) == tuple(ref.shape)
    _, pcc = comp_pcc(ref, got, pcc=0.9999)
    _, allclose_str = comp_allclose(ref, got)
    err = (got - ref).abs()
    rel_rms = (torch.sqrt(((got - ref) ** 2).mean()) / ref.std()).item()
    nz = (ref != 0) & torch.isfinite(ref)
    ratio = (got[nz] / ref[nz]).float()
    metrics = {
        "case": _id(case),
        "pcc": float(pcc),
        "max_abs_err": err.max().item(),
        "mean_abs_err": err.mean().item(),
        "rel_rms": rel_rms,
        "allclose": allclose_str,
        "ratio_median": ratio.median().item(),
        "ratio_p5": torch.quantile(ratio[:100000], 0.05).item(),
        "ratio_p95": torch.quantile(ratio[:100000], 0.95).item(),
        "bit_exact": bool(torch.equal(got, ref)),
        "output_buffer": str(y.memory_config().buffer_type),
    }
    for k, v in metrics.items():
        record_property(k, v)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{_id(case)}.json").write_text(json.dumps(metrics, indent=1))

    # The requirement (NUMERIC_RESOLVED: `exact`) and the gate check_output applies, PCC with relative RMS.
    assert metrics["bit_exact"], metrics
    assert metrics["pcc"] >= 0.9999 and metrics["rel_rms"] == 0.0, metrics
    # R5: the result lives in the input's buffer.
    assert y.memory_config().buffer_type == mc.buffer_type
