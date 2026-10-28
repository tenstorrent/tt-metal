# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Full-model PCC on the test goldens (stage C), both precisions; per-layer PCC on `single`.
The gate is tests/perf/check_gates.py; these are the tt-metal tiers (0.99 normalized / 0.98 final)."""

import json
import os

import pytest
import torch

from models.experimental.flowstate_forecast.reference import weights
from models.experimental.flowstate_forecast.tests.pcc import metrics
from models.experimental.flowstate_forecast.tests.golden.test_dataflow import GOLDENS, TEST_CASES
from models.experimental.flowstate_forecast.tt.model import FlowStateTT

OUT = os.environ.get("FLOWSTATE_PCC_OUT", "/tmp/flowstate_pcc")


@pytest.fixture(scope="module")
def ref():
    return weights.load()


@pytest.fixture(params=["accuracy", "performance"])
def tt_model(request, device, ref):
    return FlowStateTT(ref, device, request.param)


def score(g, out):
    m = metrics.gate_metrics(
        out["combined_normalized"],
        g["combined_normalized"],
        out["quantile_outputs"],
        g["quantile_outputs"],
        out["prediction_outputs"],
        g["prediction_outputs"],
    )
    gh, gq = metrics.head(g["combined_normalized"]), metrics.head(g["quantile_outputs"])
    col = metrics.gate_metrics(metrics.collapsed(gh), gh, metrics.collapsed(gq), gq)
    return m, col


def _record(entry):
    path = os.environ.get("FLOWSTATE_SUITE_JSON")
    if path:
        rec = json.loads(open(path).read()) if os.path.isfile(path) else []
        rec.append(entry)
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)


@pytest.mark.parametrize("case", TEST_CASES)
def test_model_case(tt_model, case):
    g = torch.load(GOLDENS / f"{case}.pt", weights_only=True)
    keep = case == "single"
    out = tt_model(g["inputs"].clone(), g["s"], g["horizon"], keep=keep)
    assert out["quantile_outputs"].shape == g["quantile_outputs"].shape
    m, col = score(g, out)
    if keep:
        m["layer_pcc"] = [metrics.pearson(out["layers"][i + 1], g[f"layer{i}"]) for i in range(6)]
        m["embed_pcc"] = metrics.pearson(out["layers"][0], g["embed"])
    print(tt_model.p.name, case, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}))
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{tt_model.p.name}_{case}.json", "w") as f:
        json.dump(m, f, indent=1)
    from models.experimental.flowstate_forecast.tests.perf.check_gates import git_stamp

    _record(
        {
            "test": "suite",
            "name": f"suite:{case}",
            "precision": tt_model.p.name,
            **m,
            **{f"collapsed_{k}": v for k, v in col.items()},
            **git_stamp(),
        }
    )
    assert m["pcc_normalized"] > 0.99 and m["pcc_final"] > 0.98, m
