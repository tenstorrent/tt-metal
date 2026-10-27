# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Real-weight full-model PCC on the test goldens (CONTRACT §6C), both precisions."""

import pytest
import torch

from models.experimental.tafsut_forecast.tests.golden.goldens import load_golden
from models.experimental.tafsut_forecast.tests.helpers import case_metrics, check_floors
from models.experimental.tafsut_forecast.tt.common import Precision
from models.experimental.tafsut_forecast.tt.model import TtTafsut

CASES = ["single", "single_long", "ragged4", "model16", "fallback", "horizon4"]


def get_model(device, prec):
    return TtTafsut(device, precision=Precision.by_name(prec))


@pytest.mark.parametrize("prec", ["accuracy", "performance"])
@pytest.mark.parametrize("case", CASES)
def test_model(device, case, prec):
    g = load_golden(case)
    m = get_model(device, prec)
    heads = []
    q = m.forecast(g["x"], g["horizon"], heads=heads)
    met = case_metrics(heads, g, q)
    print(f"[pcc] {case}-{prec}: " + " ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in met.items()))
    check_floors(met, prec, module=True)


@pytest.mark.parametrize("case", ["single", "single_long", "ragged4"])
def test_block_outputs(device, case):
    """Embedding and every block against the reference's block outputs (accuracy; module floor 0.999)."""
    from models.experimental.tafsut_forecast.reference import dataflow as df
    from models.experimental.tafsut_forecast.tests.helpers import pcc

    g = load_golden(case)
    m = get_model(device, "accuracy")
    prep = m.prepare_inputs(g["x"])
    taps = []
    m.forward(prep, taps=taps)
    f = g["forwards"][0]
    P = prep.tokens.shape[1]
    P0 = f["ctx_emb"].shape[1]
    off = P - P0  # bucket left padding
    emb = taps[0][:, off:P]
    valid = prep.key_valid[:, off:]
    worst = 1.0
    for i, b in enumerate([None] + list(f["blocks"][: df.NL - 1])):
        ref = f["ctx_emb"] if b is None else b[:, :P0]
        dev = taps[i][:, off:P]
        p = pcc(dev[valid], ref[valid])
        worst = min(worst, p)
        print(f"[pcc] {case} {'embed' if b is None else f'block {i - 1}'}: {p:.6f}")
    assert worst > 0.999
