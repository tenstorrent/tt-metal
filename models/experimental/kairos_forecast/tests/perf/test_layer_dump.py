# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""D5 R1-0c: per-layer host-vs-device dump of one ragged series (s110, rows 110 / 238), both flip halves, both
precisions. Opt-in: KAIROS_LAYER_DUMP=<out.json> (KAIROS_DUMP_CASE / KAIROS_DUMP_SERIES change the case / series).

Both sides take the same front (the generator's: x half, router update, -x half), so routing is identical and
every difference is device numerics. Per tap and per row, rel = |dev - ref| / |ref| over the row's valid tokens
(encoder: unmasked tokens + [REG]; decoder: the 2 tokens), and the signed mean error over |ref| rms. The row of
interest is reported next to the median and p99 over all rows and its rank."""

from __future__ import annotations

import json
import os

import pytest
import torch

from models.experimental.kairos_forecast.reference import dataflow as DF
from models.experimental.kairos_forecast.tests.perf.suite import load
from models.experimental.kairos_forecast.tt.common import D_PAD, S_PAD, Precision
from models.experimental.kairos_forecast.tt.generator import _cat_fronts
from models.experimental.kairos_forecast.tt.host_frontend import front_end
from models.experimental.kairos_forecast.tt.model import N_SEG, N_TOK, TtKairos
from models.experimental.kairos_forecast.tt.router_state import RouterState

OUT = os.environ.get("KAIROS_LAYER_DUMP")
CASE = os.environ.get("KAIROS_DUMP_CASE", "ragged")
SERIES = int(os.environ.get("KAIROS_DUMP_SERIES", "110"))


def ref_taps(W, front):
    """The reference forward with the residual stream after every block (fp32, CPU)."""
    taps = {}
    emb = DF.embed(W, front["x_final"], front["weights"], front["indices"])
    B, s = emb.shape[0], front["s_features"]
    h = torch.cat([emb, W["shared.weight"][1].expand(B, 1, DF.D_MODEL)], 1)
    taps["embed"] = h.clone()
    m = torch.cat([front["attn_mask"].float(), torch.ones(B, 1)], 1)
    mask_add = ((1.0 - m) * DF.NEG)[:, None, None, :]
    for i in range(DF.N_LAYERS):
        p = f"encoder.block.{i}.layer"
        n = DF.rms_norm(h, W[f"{p}.0.layer_norm.weight"])
        h = h + DF.attention(W, f"{p}.0.SelfAttention", n, n, mask_add, s)
        h = h + DF.ff(W, f"{p}.1.DenseReluDense", DF.rms_norm(h, W[f"{p}.1.layer_norm.weight"]))
        taps[f"enc{i}"] = h.clone()
    enc = DF.rms_norm(h, W["encoder.final_layer_norm.weight"])
    taps["enc_norm"] = enc
    h = W["shared.weight"][0].expand(B, N_SEG, DF.D_MODEL).clone()
    causal = torch.triu(torch.full((N_SEG, N_SEG), DF.NEG), 1)[None, None]
    for i in range(DF.N_LAYERS):
        p = f"decoder.block.{i}.layer"
        n = DF.rms_norm(h, W[f"{p}.0.layer_norm.weight"])
        h = h + DF.attention(W, f"{p}.0.SelfAttention", n, n, causal, s)
        n = DF.rms_norm(h, W[f"{p}.1.layer_norm.weight"])
        h = h + DF.attention(W, f"{p}.1.EncDecAttention", n, enc, mask_add, s)
        h = h + DF.ff(W, f"{p}.2.DenseReluDense", DF.rms_norm(h, W[f"{p}.2.layer_norm.weight"]))
        taps[f"dec{i}"] = h.clone()
    out = DF.rms_norm(h, W["decoder.final_layer_norm.weight"])
    taps["dec_norm"] = out
    taps["head"] = DF.head(W, out)
    return taps, m.bool()


def dev_taps(model, front):
    import ttnn

    inp = model.prepare(front)
    B = inp["B"]
    host = lambda t, rows: ttnn.to_torch(t).float().reshape(B, rows, -1)  # noqa: E731
    taps = {}
    e = model.embed(inp["xs"])
    e = ttnn.concat([ttnn.reshape(e, (B, N_TOK, DF.D_MODEL)), inp["reg"]], dim=1)
    h = ttnn.reshape(e, (1, 1, B * S_PAD, DF.D_MODEL))
    taps["embed"] = host(h, S_PAD)
    for i, (blk, cs) in enumerate(zip(model.enc, inp["enc_rope"])):
        h = model.enc_block(h, blk, cs, inp["enc_mask"], B)
        taps[f"enc{i}"] = host(h, S_PAD)
    enc = model._norm(h, model.enc_norm)
    taps["enc_norm"] = host(enc, S_PAD)
    h = inp["dec_in"]
    for i, (blk, rope) in enumerate(zip(model.dec, inp["dec_rope"])):
        h = model.dec_block(h, enc, blk, rope, inp["cross_mask"], B)
        taps[f"dec{i}"] = host(h, D_PAD)
    out = model._norm(h, model.dec_norm)
    taps["dec_norm"] = host(out, D_PAD)
    taps["head"] = host(model.head(out), D_PAD)
    return taps


def compare(ref, dev, valid):
    """Per row: relative error and signed mean error / ref rms over the valid tokens."""
    if ref.shape[1] == N_TOK + 1:  # encoder: dev rows 0..63 tokens, 64 [REG]
        dev = dev[:, : N_TOK + 1]
    else:
        dev, valid = dev[:, :N_SEG], torch.ones(ref.shape[:2], dtype=torch.bool)
    v = valid[..., None].double()
    d, r = (dev.double() - ref.double()) * v, ref.double() * v
    n = v.sum((1, 2)) * ref.shape[-1] / 1.0
    rms = (r.square().sum((1, 2)) / n).sqrt()
    rel = d.square().sum((1, 2)).sqrt() / r.square().sum((1, 2)).sqrt().clamp_min(1e-30)
    signed = d.sum((1, 2)) / n / rms.clamp_min(1e-30)
    return rel, signed


@pytest.mark.skipif(not OUT, reason="opt-in: KAIROS_LAYER_DUMP=<out.json>")
def test_layer_dump(device):
    rec = load(CASE)
    W = DF.load_weights()
    ctx = rec["context"].float()
    S = ctx.shape[0]
    router = RouterState(rec["bias0"])
    fa, _ = front_end(ctx, router, W)
    fb, _ = front_end(-ctx, router, W)
    front = _cat_fronts(fa, fb)
    ref, valid = ref_taps(W, front)
    gold = torch.cat(rec["head"])
    res = {
        "case": CASE,
        "series": SERIES,
        "rows": [SERIES, S + SERIES],
        "ref_vs_golden_head_max": float((ref["head"][:, :N_SEG].reshape(gold.shape) - gold).abs().max()),
    }
    for prec in ("accuracy", "performance"):
        model = TtKairos(device, W, getattr(Precision, prec)())
        dev = dev_taps(model, front)
        out = {}
        for tap in ref:
            rel, signed = compare(ref[tap], dev[tap], valid)
            row = {}
            for half, i in (("x", SERIES), ("-x", S + SERIES)):
                row[half] = {"rel": float(rel[i]), "signed": float(signed[i]), "rank": int((rel > rel[i]).sum()) + 1}
            row["rel_p50"] = float(rel.median())
            row["rel_p99"] = float(torch.quantile(rel, 0.99))
            out[tap] = row
        res[prec] = out
        del model
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    for prec in ("accuracy", "performance"):
        for tap, row in res[prec].items():
            print(
                prec,
                tap,
                {
                    k: (round(v["rel"], 5), round(v["signed"], 5), v["rank"]) if isinstance(v, dict) else round(v, 5)
                    for k, v in row.items()
                },
            )
