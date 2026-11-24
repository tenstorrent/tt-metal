# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""EXPERIMENT gather_raw_first: precision menu. The op and the raw-first copy on identical inputs (same seed): worst
(device, slice) PCC / relative RMS vs the golden reference for each, and the fraction of bit-identical output elements
between the two. Prints one RESULT line per case; asserts only the golden gate (correctness)."""

import os
import re

import pytest
import torch
import ttnn

from eval.golden_tests.all_gather_rms_norm.helpers import _pcc_rel_rms, prepare_all_gather_rms_norm
from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases, _parametrize_cases
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm as base_op
from ttnn.operations.all_gather_rms_norm.perf_experiments.gather_raw_first.agrn_raw import all_gather_rms_norm as raw_op


def _cases():
    pats = [re.compile(p) for p in os.environ.get("AGRN_CELLS", "").split(";") if p]
    if pats:
        return [pytest.param(*c.values, {}, id=c.id) for c in _parametrize_cases() if all(p.search(c.id) for p in pats)]
    return _loose_cases()


def _worst(out, expected, G):
    worst = (2.0, 0.0)
    devs = ttnn.get_device_tensors(out)
    cols = expected.shape[1]
    for idx, d in enumerate(devs):
        r, c = divmod(idx, cols)
        a = ttnn.to_torch(d).to(torch.float64)
        e = expected[r, c].to(torch.float64)
        w = a.shape[-1] // G
        for p in range(G):
            pcc, rms = _pcc_rel_rms(a[..., p * w : (p + 1) * w], e[..., p * w : (p + 1) * w])
            if os.environ.get("AGRN_DBG") and pcc < 0.999:
                sl = a[..., p * w : (p + 1) * w].reshape(-1, w)
                el = e[..., p * w : (p + 1) * w].reshape(-1, w)
                bad = ((sl - el).abs() > 0.1) | sl.isnan()
                rows = bad.any(dim=1).nonzero().flatten()
                colsb = bad.any(dim=0).nonzero().flatten()
                print(
                    f"DBG dev{idx} slice{p} pcc={pcc:.4f} bad rows {rows.min().item() if len(rows) else -1}.."
                    f"{rows.max().item() if len(rows) else -1} n={len(rows)} cols {colsb.min().item() if len(colsb) else -1}.."
                    f"{colsb.max().item() if len(colsb) else -1} n={len(colsb)} nbad={int(bad.sum())}"
                )
            worst = (min(worst[0], pcc), max(worst[1], rms))
    return worst, [ttnn.to_torch(d) for d in devs]


@pytest.mark.parametrize("inputs,axes,extras", _cases())
def test_raw_precision(inputs, axes, extras, mesh_device, request):
    from eval.golden_tests.all_gather_rms_norm import helpers

    captured = {}
    orig = helpers.pytorch_all_gather_rms_norm

    def capture(*a, **k):
        captured["expected"] = orig(*a, **k)
        return captured["expected"]

    helpers.pytorch_all_gather_rms_norm = capture
    try:
        call_b, check_b = prepare_all_gather_rms_norm(
            inputs, mesh_device=mesh_device, extras=extras, op=base_op, **axes
        )
        out_b = call_b()
        exp = captured["expected"]
        call_r, check_r = prepare_all_gather_rms_norm(inputs, mesh_device=mesh_device, extras=extras, op=raw_op, **axes)
        out_r = call_r()
    finally:
        helpers.pytorch_all_gather_rms_norm = orig
    G = tuple(mesh_device.shape)[axes["cluster_axis"]]
    (pb, rb), tb = _worst(out_b, exp, G)
    (pr, rr), tr = _worst(out_r, exp, G)
    same = sum(
        int(
            (
                x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)
                == y.view(torch.int16 if y.dtype == torch.bfloat16 else torch.int32)
            ).sum()
        )
        for x, y in zip(tb, tr)
    )
    total = sum(x.numel() for x in tb)
    maxdiff = max(float((x.float() - y.float()).abs().max()) for x, y in zip(tb, tr))
    print(
        f"\nRESULT {request.node.callspec.id[:60]} base pcc={pb:.6f} rel_rms={rb:.6f} | raw pcc={pr:.6f} "
        f"rel_rms={rr:.6f} | bit-identical {same}/{total} max|diff|={maxdiff:.3g}"
    )
    check_b(out_b)
    check_r(out_r)
