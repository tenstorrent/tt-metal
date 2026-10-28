# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Real-weight full-model PCC on the test goldens (CONTRACT §6C), both precisions."""

import os

import pytest
import torch

from models.experimental.tafsut_forecast.tests.golden.goldens import load_golden
from models.experimental.tafsut_forecast.tests.helpers import case_metrics, check_floors
from models.experimental.tafsut_forecast.tests.perf.test_forward_perf import _record
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
    print(
        f"[pcc] {case}-{prec}: "
        + " ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in met.items())
    )
    # suite record for check_gates (MODEL_PERF_JSON), and the raw outputs for row-level analysis (TAFSUT_DUMP dir)
    _record({"test": "suite", "name": case, "precision": prec, "rows": int(g["x"].shape[0]), **met})
    if os.environ.get("TAFSUT_DUMP"):
        torch.save({"heads": heads, "q": q}, os.path.join(os.environ["TAFSUT_DUMP"], f"{case}-{prec}.pt"))
    check_floors(met, prec, module=True)


@pytest.mark.parametrize("case", ["single", "single_long", "ragged4"])
def test_block_outputs(device, case):
    """Embedding, every block and the head against the reference (accuracy; module floor 0.999 on context rows).

    Context rows of the embedding and blocks 0-12; the future rows (and REG) of every block, the last one included
    (it runs on the 32 future rows only, the read-out cut); and the head output (D2 item 5: where does
    single_long's error first appear). Each line prints the PCC and the worst absolute error (normalized units)."""
    from models.experimental.tafsut_forecast.reference import dataflow as df
    from models.experimental.tafsut_forecast.tests.helpers import pcc

    g = load_golden(case)
    m = get_model(device, "accuracy")
    prep = m.prepare_inputs(g["x"])
    taps = []
    head = m.forward(prep, taps=taps)
    f = g["forwards"][0]
    P = prep.tokens.shape[1]
    P0 = f["ctx_emb"].shape[1]
    off = P - P0  # bucket left padding
    valid = prep.key_valid[:, off:]
    nf = df.NFUT
    line = lambda what, d, r: (pcc(d, r), float((d - r).abs().max()), what)
    rows = [line("embed context", taps[0][:, off:P][valid], f["ctx_emb"][valid])]
    rows.append(line("embed future", taps[0][:, P : P + nf], f["fut_emb"]))
    for i, b in enumerate(f["blocks"]):
        dev = taps[i + 1]
        if i < df.NL - 1:
            rows.append(line(f"block {i} context", dev[:, off:P][valid], b[:, :P0][valid]))
            rows.append(line(f"block {i} REG", dev[:, P + nf], b[:, P0]))
            rows.append(line(f"block {i} future", dev[:, P : P + nf], b[:, P0 + 1 : P0 + 1 + nf]))
        else:  # read-out cut: the last block's output is the 32 future rows
            rows.append(line(f"block {i} future (read-out)", dev[:, :nf], b[:, P0 + 1 : P0 + 1 + nf]))
    rows.append(line("head", head, f["head"]))
    for p, err, what in rows:
        print(f"[pcc] {case} {what}: pcc {p:.6f} max_err {err:.4e}")
    ctx = [p for p, _, w in rows if "context" in w]
    fut = [p for p, _, w in rows if "future" in w or "REG" in w]
    assert min(ctx) > 0.999
    assert min(fut) > 0.998
    assert rows[-1][0] > 0.997  # single_long's case floor
