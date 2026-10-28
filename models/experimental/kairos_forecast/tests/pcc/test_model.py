# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TtKairos forward vs the reference on the test goldens (captured host front-ends): encoder output and the head
output of every forward, in both precisions."""
import pytest
import torch

from models.experimental.kairos_forecast.reference import dataflow as DF
from models.experimental.kairos_forecast.tests.pcc.test_dataflow import CASES, GOLD, fronts
from models.experimental.kairos_forecast.tt.common import Precision
from models.experimental.kairos_forecast.tt.model import N_TOK, S_PAD, TtKairos


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.fixture(scope="module")
def W():
    return DF.load_weights()


def enc_rows(enc_dev, B, N):
    """Device encoder output (B * 96, 512) -> the reference's (B, N + 1, 512) order ([REG] at index N)."""
    e = enc_dev.view(B, S_PAD, -1)
    return torch.cat([e[:, :N], e[:, N_TOK : N_TOK + 1]], 1)


@pytest.mark.parametrize("prec", ["accuracy", "performance"])
@pytest.mark.parametrize("case", CASES)
def test_model_head(device, W, case, prec):
    import ttnn

    rec = torch.load(GOLD / f"{case}.pt", weights_only=False)
    fs = fronts(rec)
    enc_ref = [o for n, _a, _k, o in rec["firings"] if n == "encoder.final_layer_norm"]
    m = TtKairos(device, W, getattr(Precision, prec)())
    floor = 0.999 if prec == "accuracy" else 0.995
    for i, f in enumerate(fs[:4]):
        inp = m.prepare(f)
        enc = ttnn.to_torch(m.encode(inp)).float().view(-1, 512)
        B, N = f["attn_mask"].shape
        e_pcc = pcc(enc_rows(enc, B, N), enc_ref[i])
        got = ttnn.to_torch(m.head(m.decode(m.encode(inp), inp))).float().view(B, -1, 576)[:, :2]
        h_pcc = pcc(got, f["head"])
        rel = float((got - f["head"]).abs().max() / f["head"].abs().max())
        print(f"{case}/{prec} fwd {i}: B={B} N={N} enc pcc {e_pcc:.6f} head pcc {h_pcc:.6f} max rel {rel:.3e}")
        assert h_pcc >= floor, (case, i, h_pcc)
