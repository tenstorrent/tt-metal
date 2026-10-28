# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Full-model PCC on the test goldens (stage C), both precisions; per-layer PCC on `single`.
The gate is tests/perf/check_gates.py; these are the tt-metal tiers (0.99 normalized / 0.98 final)."""

import json
import os

import pytest
import torch

from models.experimental.flowstate_forecast.reference import metrics, weights
from models.experimental.flowstate_forecast.tests.golden.test_dataflow import GOLDENS, TEST_CASES
from models.experimental.flowstate_forecast.tt.model import FlowStateTT

OUT = os.environ.get("FLOWSTATE_PCC_OUT", "/tmp/flowstate_pcc")
MEDIAN = 4


def as_rows(q):
    """(B, 9, H, 1) -> (B, H, 9), the metrics' [rows, horizon, n_quantiles]."""
    return q[..., 0].transpose(1, 2)


@pytest.fixture(scope="module")
def ref():
    return weights.load()


@pytest.fixture(params=["accuracy", "performance"])
def tt_model(request, device, ref):
    return FlowStateTT(ref, device, request.param)


def score(g, out):
    m = metrics.gate_metrics(
        as_rows(g["combined_normalized"]),
        as_rows(out["combined_normalized"]),
        as_rows(g["quantile_outputs"]),
        as_rows(out["quantile_outputs"]),
        MEDIAN,
    )
    m["collapsed_pcc_spread"] = metrics.pcc(
        metrics.spread(as_rows(g["combined_normalized"]), MEDIAN),
        metrics.spread(metrics.collapsed(as_rows(out["combined_normalized"])), MEDIAN),
    )
    return m


@pytest.mark.parametrize("case", TEST_CASES)
def test_model_case(tt_model, case):
    g = torch.load(GOLDENS / f"{case}.pt", weights_only=True)
    keep = case == "single"
    out = tt_model(g["inputs"].clone(), g["s"], g["horizon"], keep=keep)
    assert out["quantile_outputs"].shape == g["quantile_outputs"].shape
    m = score(g, out)
    if keep:
        m["layer_pcc"] = [metrics.pcc(out["layers"][i + 1], g[f"layer{i}"]) for i in range(6)]
        m["embed_pcc"] = metrics.pcc(out["layers"][0], g["embed"])
    print(tt_model.p.name, case, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}))
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{tt_model.p.name}_{case}.json", "w") as f:
        json.dump(m, f, indent=1)
    assert m["pcc_normalized"] > 0.99 and m["pcc_final"] > 0.98, m
