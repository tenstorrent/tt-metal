# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""G1: the TRELLIS.2 structure flow DiT on one device against the CPU fp32 goldens.

Gate (project VALIDATION.md, accepted 2026-10-15):
- single pass: every recorded call, fed its golden x_t / t / cond, PCC >= 0.999 (CPU bf16 floor 0.99987);
- full sample: 12 free-running steps from the golden noise, last call PCC >= 0.985 (CPU bf16 floor 0.9854).

TRELLIS2_RESULTS_DIR, if set, receives a JSON table per test.
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch
from loguru import logger

from models.experimental.trellis2.reference import structure_flow_host as host
from models.experimental.trellis2.tests import goldens
from models.experimental.trellis2.tt.structure_flow import TtStructureFlow

SINGLE_PASS_PCC = 0.999
LAST_CALL_PCC = 0.985

pytestmark = pytest.mark.skipif(goldens.missing_reason() is not None, reason=goldens.missing_reason() or "")


def _model(mesh_device) -> TtStructureFlow:
    model = TtStructureFlow(mesh_device=mesh_device)
    model.load_torch_state_dict(host.load_state_dict(f"{goldens.weights_dir()}/{goldens.SS_CKPT}"))
    return model


def _save(name: str, result: dict) -> None:
    out_dir = os.environ.get("TRELLIS2_RESULTS_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/{name}.json", "w") as f:
            json.dump(result, f, indent=1)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_structure_flow_single_pass(mesh_device):
    model = _model(mesh_device)
    rows = []
    for i in range(len(goldens.structure_calls())):
        x, t, cond, gold, branch = goldens.call_inputs(i)
        start = time.time()
        out = model(x, t, cond)
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
    result = {"min_pcc": min(r["pcc"] for r in rows), "threshold": SINGLE_PASS_PCC, "rows": rows}
    _save("structure_flow_single_pass", result)
    logger.info(f"min single-pass PCC {result['min_pcc']:.6f}")
    assert result["min_pcc"] >= SINGLE_PASS_PCC


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_structure_flow_full_sample(mesh_device):
    model = _model(mesh_device)
    steps, calls, cond = goldens.structure_steps(), goldens.structure_calls(), goldens.conditions()
    start = time.time()
    final, outs = host.sample(model, steps[0]["x_t"], cond["cond"], cond["neg_cond"])
    seconds = time.time() - start
    assert len(outs) == len(calls)
    rows = [
        {"call": i, "branch": c["branch"], "pcc": goldens.pcc(o, c["out"])} for i, (o, c) in enumerate(zip(outs, calls))
    ]
    for r in rows:
        logger.info(f"call {r['call']:2d} ({r['branch']}) free-running pcc={r['pcc']:.6f}")
    gold_final = steps[-1]["x_prev"]
    result = {
        "last_call_pcc": rows[-1]["pcc"],
        "threshold": LAST_CALL_PCC,
        "final_sample_pcc": goldens.pcc(final, gold_final),
        "final_sample_max_abs_err": (final - gold_final).abs().max().item(),
        "seconds": seconds,
        "rows": rows,
    }
    _save("structure_flow_full_sample", result)
    logger.info(
        f"last call {result['last_call_pcc']:.6f}, final sample {result['final_sample_pcc']:.6f}, {seconds:.1f} s"
    )
    assert result["last_call_pcc"] >= LAST_CALL_PCC
