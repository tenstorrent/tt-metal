# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""G2 (texture stage): the TRELLIS.2 texture SLat flow DiT on one device against the CPU fp32 goldens.

The input is the noise latent concatenated with the normalized shape latent (concat_cond, fixed for the stage);
texture CFG strength is 1, so every step is one pass. Gate: single pass >= 0.999 (accepted 2026-10-15); the
free-running thresholds sit just under the CPU bf16 floor (last call 0.999893, final 0.999982; S03) and are
proposed, not yet accepted by the user.

TRELLIS2_RESULTS_DIR, if set, receives a JSON table per test.
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch
from loguru import logger

from models.experimental.trellis2.reference import slat_flow_host as slat
from models.experimental.trellis2.reference import structure_flow_host as host
from models.experimental.trellis2.tests import goldens
from models.experimental.trellis2.tests.test_structure_flow_pcc import precision_from_env
from models.experimental.trellis2.tt.slat_flow import TtSLatFlow
from models.experimental.trellis2.tt.structure_flow import Precision

SINGLE_PASS_PCC = 0.999
LAST_CALL_PCC = 0.9998
FINAL_LATENT_PCC = 0.9999

pytestmark = pytest.mark.skipif(goldens.missing_reason() is not None, reason=goldens.missing_reason() or "")


def _model(mesh_device, precision: Precision | None = None) -> TtSLatFlow:
    precision = precision or precision_from_env()
    logger.info(f"precision {precision}")
    model = TtSLatFlow(mesh_device=mesh_device, in_channels=64, precision=precision)
    model.load_torch_state_dict(host.load_state_dict(f"{goldens.weights_dir()}/{slat.TEX_CKPT}"))
    return model


def _save(name: str, result: dict) -> None:
    out_dir = os.environ.get("TRELLIS2_RESULTS_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/{name}.json", "w") as f:
            json.dump(result, f, indent=1)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_tex_flow_single_pass(mesh_device):
    model = _model(mesh_device)
    _, coords, _, cond, _, _ = goldens.tex_call_inputs(0)
    tables = model.prepare(coords, cond.shape[1])
    rows = []
    for i in range(len(goldens.tex_stage()["calls"])):
        feats, _, t, cond, gold, branch = goldens.tex_call_inputs(i)
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
    _save("tex_flow_single_pass", result)
    logger.info(f"min single-pass PCC {result['min_pcc']:.6f} (N {result['n_tokens']}, bucket {result['bucket']})")
    assert result["min_pcc"] >= SINGLE_PASS_PCC


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_tex_flow_full_sample(mesh_device):
    model = _model(mesh_device)
    stage, cond, concat = goldens.tex_stage(), goldens.conditions(), goldens.tex_concat()
    calls, steps = stage["calls"], stage["steps"]
    coords = steps[0]["x_t"]["coords"]
    tables = model.prepare(coords, cond["cond"].shape[1])
    start = time.time()
    final, outs = slat.sample(
        lambda x, t, c: model(torch.cat([x, concat], dim=-1), t, c, tables),
        steps[0]["x_t"]["feats"],
        cond["cond"],
        cond["neg_cond"],
        slat.TEX_SAMPLER,
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
    _save("tex_flow_full_sample", result)
    logger.info(
        f"last call {result['last_call_pcc']:.6f}, final sample {result['final_sample_pcc']:.6f}, {seconds:.1f} s"
    )
    assert result["last_call_pcc"] >= LAST_CALL_PCC
    assert result["final_sample_pcc"] >= FINAL_LATENT_PCC
