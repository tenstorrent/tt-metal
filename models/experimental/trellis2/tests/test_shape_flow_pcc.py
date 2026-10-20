# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""G2 (shape stage): the TRELLIS.2 shape SLat flow DiT on one device against the CPU fp32 goldens.

Gate (project VALIDATION.md, accepted 2026-10-15):
- single pass: every recorded call, fed its golden x_t / t / cond, PCC >= 0.999;
- full sample: 12 free-running steps from the golden noise, last call PCC >= 0.988 and final latent >= 0.995.
Padding: the same call at two bucket lengths must agree on the real rows (windowed self-attention).

TRELLIS2_RESULTS_DIR, if set, receives a JSON table per test.
"""

from __future__ import annotations

import json
import os
import time

import pytest
from loguru import logger

from models.experimental.trellis2.reference import slat_flow_host as slat
from models.experimental.trellis2.reference import structure_flow_host as host
from models.experimental.trellis2.tests import goldens
from models.experimental.trellis2.tests.test_structure_flow_pcc import precision_from_env
from models.experimental.trellis2.tt.slat_flow import TtSLatFlow
from models.experimental.trellis2.tt.structure_flow import Precision

SINGLE_PASS_PCC = 0.999
LAST_CALL_PCC = 0.988
FINAL_LATENT_PCC = 0.995
PAD_AGREEMENT_PCC = 0.9999

pytestmark = pytest.mark.skipif(goldens.missing_reason() is not None, reason=goldens.missing_reason() or "")


def _model(mesh_device, precision: Precision | None = None) -> TtSLatFlow:
    precision = precision or precision_from_env()
    logger.info(f"precision {precision}")
    model = TtSLatFlow(mesh_device=mesh_device, in_channels=32, precision=precision)
    model.load_torch_state_dict(host.load_state_dict(f"{goldens.weights_dir()}/{slat.SHAPE_CKPT}"))
    return model


def _save(name: str, result: dict) -> None:
    out_dir = os.environ.get("TRELLIS2_RESULTS_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/{name}.json", "w") as f:
            json.dump(result, f, indent=1)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_shape_flow_single_pass(mesh_device):
    model = _model(mesh_device)
    _, coords, _, cond, _, _ = goldens.shape_call_inputs(0)
    tables = model.prepare(coords, cond.shape[1])
    rows = []
    for i in range(len(goldens.shape_stage()["calls"])):
        feats, _, t, cond, gold, branch = goldens.shape_call_inputs(i)
        start = time.time()
        out = model(feats, t, cond, tables)
        p = goldens.pcc(out, gold)
        rows.append(
            {
                "call": i,
                "branch": branch,
                "t": float(t[0]) / 1000,
                "pcc": p,
                "max_abs_err": (out - gold).abs().max().item(),
                "seconds": time.time() - start,
            }
        )
        logger.info(f"call {i:2d} ({branch}) t={rows[-1]['t']:.3f} pcc={p:.6f} {rows[-1]['seconds']:.2f} s")
    result = {
        "n_tokens": coords.shape[0],
        "bucket": tables.cos.shape[2],
        "min_pcc": min(r["pcc"] for r in rows),
        "threshold": SINGLE_PASS_PCC,
        "rows": rows,
    }
    _save("shape_flow_single_pass", result)
    logger.info(f"min single-pass PCC {result['min_pcc']:.6f} (N {result['n_tokens']}, bucket {result['bucket']})")
    assert result["min_pcc"] >= SINGLE_PASS_PCC


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_shape_flow_padding(mesh_device):
    model = _model(mesh_device)
    feats, coords, t, cond, gold, _ = goldens.shape_call_inputs(0)
    n = coords.shape[0]
    outs = {}
    for n_pad in (slat.bucket_len(n), slat.bucket_len(n) + 512):
        outs[n_pad] = model(feats, t, cond, model.prepare(coords, cond.shape[1], n_pad))
    a, b = outs.values()
    result = {
        "buckets": list(outs),
        "agreement_pcc": goldens.pcc(a, b),
        "pcc_vs_golden": [goldens.pcc(o, gold) for o in outs.values()],
    }
    _save("shape_flow_padding", result)
    logger.info(
        f"padding {result['buckets']}: agreement {result['agreement_pcc']:.6f}, vs golden {result['pcc_vs_golden']}"
    )
    assert result["agreement_pcc"] >= PAD_AGREEMENT_PCC
    assert min(result["pcc_vs_golden"]) >= SINGLE_PASS_PCC


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_shape_flow_full_sample(mesh_device):
    model = _model(mesh_device)
    stage, cond = goldens.shape_stage(), goldens.conditions()
    calls, steps = stage["calls"], stage["steps"]
    coords = steps[0]["x_t"]["coords"]
    tables = model.prepare(coords, cond["cond"].shape[1])
    start = time.time()
    final, outs = slat.sample(
        lambda x, t, c: model(x, t, c, tables),
        steps[0]["x_t"]["feats"],
        cond["cond"],
        cond["neg_cond"],
        slat.SamplerParams(),
    )
    seconds = time.time() - start
    assert len(outs) == len(calls)
    rows = [
        {"call": i, "branch": c["branch"], "pcc": goldens.pcc(o, c["out"]["feats"])}
        for i, (o, c) in enumerate(zip(outs, calls))
    ]
    for r in rows:
        logger.info(f"call {r['call']:2d} ({r['branch']}) free-running pcc={r['pcc']:.6f}")
    gold_final = steps[-1]["x_prev"]["feats"]
    result = {
        "last_call_pcc": rows[-1]["pcc"],
        "final_sample_pcc": goldens.pcc(final, gold_final),
        "thresholds": {"last_call": LAST_CALL_PCC, "final_sample": FINAL_LATENT_PCC},
        "final_sample_max_abs_err": (final - gold_final).abs().max().item(),
        "seconds": seconds,
        "rows": rows,
    }
    _save("shape_flow_full_sample", result)
    logger.info(
        f"last call {result['last_call_pcc']:.6f}, final sample {result['final_sample_pcc']:.6f}, {seconds:.1f} s"
    )
    assert result["last_call_pcc"] >= LAST_CALL_PCC
    assert result["final_sample_pcc"] >= FINAL_LATENT_PCC
