# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.20 (owner 2026-11-01): does the device's hidden-state error matter for the tokens? Token-level accuracy of the
device and of DeepSeek's own reference at its shipped precision, both against the full-precision golden.

Per rung (``s10240``: chunk 1, positions 5120-10239, every chunk from position 0; ``last``: s56320's last chunk,
51200-56319, the golden's state prefix loaded), as testing/ladder.py runs it:

    device     the all-device model's final norm of every position of the chunk, shipped settings (one device run)
    wkv16      the same with DS41_WINDOW_KV_FORMAT=bf16_tiles (P.26's switch; recorded, not gated)
    official   DeepSeek's reference at numerics official (bf16, its fp8 / fp4 quantizers), fed the same way (CPU, made
               once and cached beside the golden: tests/bringup/official_final_norm.py)
    golden     the native fp32 golden's final norm (chunk model.safetensors)

logits = final_norm @ head.weight (fp32, CPU in row blocks), log-softmax in float64. Against the golden, per position:
KL(p_golden || p_x), top-1 agreement, top-5 overlap (|top5_g & top5_x| / 5), and the next-token NLL of the real text
(tokens shifted by one; the rung's last position has no next token). Metrics (x = device; suffix _official, _wkv16):
kl_mean_<rung>, kl_p99_<rung>, top1_agree_<rung>, top5_ovl_<rung>, nll_<rung>; nll_<rung>_golden. Gated:
kl_mean_<rung> <= kl_mean_<rung>_limit = e2e.calib^2 x the official's kl_mean (KL ~ rel^2, F135's PCC convention; no
fixed number, F138). The 20 worst positions are printed (and kept in generated/fix20_logit_kl/).
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden, hf_path
from models.demos.common.bringup.testing import e2e_limits as E
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec
from models.demos.common.bringup.testing.ladder import rel
from models.demos.deepseek_v41_flash_d_p.tests.bringup import official_final_norm as OFN

S = spec()
RUNGS = ["s10240", "last"]
VARIANTS = {"device": None, "wkv16": "bf16_tiles"}  # variant -> DS41_WINDOW_KV_FORMAT (None: the shipped default)
WKV_ENV = "DS41_WINDOW_KV_FORMAT"
BLOCK = 256  # logit rows per CPU block
OUT = Path(__file__).resolve().parents[5] / "generated" / "fix20_logit_kl"
_HEAD: dict = {}
_OFFICIAL: dict = {}  # rung -> per-position stats of the official and the golden (computed once per process)


def head_weight() -> torch.Tensor:
    if "w" not in _HEAD:
        from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint

        _HEAD["w"] = Checkpoint(hf_path(S)).weight("head.weight", torch.float32).contiguous()
    return _HEAD["w"]


def device_final_norm(mesh, rung_name: str) -> tuple[torch.Tensor, dict]:
    """The device's final norm [chunk, H] of the rung's last chunk, run as testing/ladder.py runs it."""
    rung = S.rung(rung_name)
    g = Golden.for_rung(S, rung_name)
    layers = list(range(S.num_layers))
    assert all(i in g.layers for i in layers), "the golden must hold every layer"
    tokens, chunk, n_chunks = g.tokens(), rung["chunk"], rung["seq"] // rung["chunk"]
    last = n_chunks - 1
    t0 = time.time()
    model = S.hooks().device_model(mesh, S, layers, lm_head=True)
    info = {"model_load_s": round(time.time() - t0, 1)}
    state = model.new_state(rung["seq"])
    info["formats"] = {k: str(v) for k, v in (state.formats() if hasattr(state, "formats") else {}).items()}
    first = 0
    if rung.get("prefix_from_golden"):
        t1 = time.time()
        for i in layers:
            state.load_prefix(i, g.state(i, at=last * chunk), last * chunk)
        info["prefix_load_s"] = round(time.time() - t1, 1)
        first = last
    final = None
    for c in range(first, n_chunks):
        s0, t1 = c * chunk, time.time()
        begin_chunk(model, tokens[s0 : s0 + chunk], s0)
        h = model.embed(tokens[s0 : s0 + chunk])
        for i in layers:
            h2 = model.layer(i, h, s0, state)
            model.free(h)
            h = h2
        hidden = model.final_norm(h)
        model.free(h)
        model.sync()
        if c == last:
            final = model.to_host(hidden).float().reshape(chunk, -1).clone()
        model.free(hidden)
        print(f"chunk {c} [{s0},{s0 + chunk}) {time.time() - t1:.2f}s", flush=True)
    del state, model
    gc.collect()
    return final, info


def official_final_norm(rung_name: str) -> tuple[torch.Tensor, dict]:
    g = Golden.for_rung(S, rung_name)
    st, _ = OFN.paths(g.dir, rung_name)
    if not st.exists():  # made once (CPU); normally made beforehand: python -m ...official_final_norm <rung>
        OFN.generate(rung_name)
    return OFN.load(g, rung_name)


def token_stats(x: torch.Tensor, gold: torch.Tensor, targets: torch.Tensor) -> dict:
    """Per position vs the golden: kl, top1 (0/1), top5 overlap, nll of x and of the golden on ``targets`` (-1: none),
    and both top-1 ids. CPU, fp32 logits in row blocks, log-softmax in float64."""
    w = head_weight()
    n = gold.shape[0]
    out = {k: torch.empty(n, dtype=torch.float64) for k in ("kl", "top1", "top5", "nll", "nll_golden")}
    out["id_golden"], out["id_x"] = torch.empty(n, dtype=torch.long), torch.empty(n, dtype=torch.long)
    for r0 in range(0, n, BLOCK):
        r1 = min(n, r0 + BLOCK)
        lg = torch.log_softmax((gold[r0:r1].float() @ w.T).double(), dim=-1)
        lx = torch.log_softmax((x[r0:r1].float() @ w.T).double(), dim=-1)
        out["kl"][r0:r1] = (lg.exp() * (lg - lx)).sum(-1)
        tg, tx = lg.topk(5, dim=-1).indices, lx.topk(5, dim=-1).indices
        out["top1"][r0:r1] = (tg[:, 0] == tx[:, 0]).double()
        out["top5"][r0:r1] = (tg.unsqueeze(-1) == tx.unsqueeze(-2)).any(-1).double().mean(-1)
        out["id_golden"][r0:r1], out["id_x"][r0:r1] = tg[:, 0], tx[:, 0]
        t = targets[r0:r1]
        ok = t >= 0
        tt = t.clamp_min(0).unsqueeze(-1)
        out["nll"][r0:r1] = torch.where(ok, -lx.gather(-1, tt).squeeze(-1), torch.nan)
        out["nll_golden"][r0:r1] = torch.where(ok, -lg.gather(-1, tt).squeeze(-1), torch.nan)
    return out


def summary(st: dict) -> dict:
    nll = st["nll"][~st["nll"].isnan()]
    return {
        "kl_mean": float(st["kl"].mean()),
        "kl_p99": float(torch.quantile(st["kl"], 0.99)),
        "kl_max": float(st["kl"].max()),
        "top1_agree": float(st["top1"].mean()),
        "top5_ovl": float(st["top5"].mean()),
        "nll": float(nll.mean()),
    }


def rung_inputs(rung_name: str):
    rung = S.rung(rung_name)
    g = Golden.for_rung(S, rung_name)
    chunk, last = rung["chunk"], rung["seq"] // rung["chunk"] - 1
    s0 = last * chunk
    gm = g.model(last)
    toks = g.tokens()
    nxt = torch.full((chunk,), -1, dtype=torch.long)
    avail = min(chunk, toks.numel() - s0 - 1)
    nxt[:avail] = toks[s0 + 1 : s0 + 1 + avail]
    assert torch.equal(gm["tokens"].long(), toks[s0 : s0 + chunk]), "golden chunk tokens disagree with the token ids"
    return g, gm["final_norm"].float().reshape(chunk, -1), toks[s0 : s0 + chunk], nxt, s0


@mesh_parametrize
@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
@pytest.mark.parametrize("rung", RUNGS, ids=RUNGS)
def test_logit_kl(mesh_device, rung, variant, monkeypatch):
    if VARIANTS[variant] is None:
        monkeypatch.delenv(WKV_ENV, raising=False)
    else:
        monkeypatch.setenv(WKV_ENV, VARIANTS[variant])
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    settings.thaw()
    wkv = settings.get("WINDOW_KV_FORMAT")
    print(f"[logit_kl] rung {rung} variant {variant}: WINDOW_KV_FORMAT={wkv}", flush=True)
    if variant == "device":
        assert wkv == "bfp8_tiles", f"shipped WINDOW_KV_FORMAT is {wkv}, owner 2026-11-01: bfp8_tiles"

    g, gold, toks, nxt, s0 = rung_inputs(rung)
    dev, info = device_final_norm(mesh_device, rung)
    settings.thaw()

    if rung not in _OFFICIAL:
        off, meta = official_final_norm(rung)
        ost = token_stats(off.reshape(gold.shape), gold, nxt)
        _OFFICIAL[rung] = (ost, summary(ost), meta, rel(off, gold))
    ost, osum, ometa, orel = _OFFICIAL[rung]
    dst = token_stats(dev, gold, nxt)
    dsum = summary(dst)
    gnll = float(dst["nll_golden"][~dst["nll_golden"].isnan()].mean())
    drel = rel(dev, gold)

    sfx = "" if variant == "device" else "_wkv16"
    for k, v in dsum.items():
        metrics.record(f"{k}_{rung}{sfx}", v)
    metrics.record(f"err_final_norm_{rung}{sfx}", drel)  # rel L2 of the hidden (diagnostic)
    for k, v in osum.items():
        metrics.record(f"{k}_{rung}_official", v)
    metrics.record(f"err_final_norm_{rung}_official", orel)
    metrics.record(f"nll_{rung}_golden", gnll)
    metrics.record(f"official_seconds_{rung}", ometa.get("total_seconds"))
    limit = E.CALIB**2 * osum["kl_mean"]
    if variant == "device":
        metrics.record(f"kl_mean_{rung}_limit", limit)

    worst = torch.argsort(dst["kl"], descending=True)[:20]
    rows = [
        {
            "pos": s0 + int(p),
            "token": int(toks[p]),
            "next": int(nxt[p]),
            "kl": round(float(dst["kl"][p]), 5),
            "kl_official": round(float(ost["kl"][p]), 5),
            "golden_top1": int(dst["id_golden"][p]),
            "x_top1": int(dst["id_x"][p]),
            "official_top1": int(ost["id_x"][p]),
        }
        for p in worst
    ]
    print(
        f"[logit_kl] {rung} {variant} (wkv {wkv}): final_norm rel {drel:.4f} (official {orel:.4f}); "
        f"golden nll {gnll:.4f}\n"
        f"  {'':10s} {'kl_mean':>9s} {'kl_p99':>9s} {'kl_max':>8s} {'top1':>7s} {'top5':>7s} {'nll':>7s}\n"
        + "\n".join(
            f"  {name:10s} {s['kl_mean']:9.5f} {s['kl_p99']:9.5f} {s['kl_max']:8.4f} {s['top1_agree']:7.4f} "
            f"{s['top5_ovl']:7.4f} {s['nll']:7.4f}"
            for name, s in ((variant, dsum), ("official", osum))
        )
        + f"\n  kl_mean limit (calib^2 {E.CALIB**2:g} x official) {limit:.5f}\n  20 worst positions:\n"
        + "\n".join(
            f"    pos {r['pos']} tok {r['token']} -> {r['next']}: kl {r['kl']:.4f} (official {r['kl_official']:.4f}) "
            f"golden top1 {r['golden_top1']} {variant} {r['x_top1']} official {r['official_top1']}"
            for r in rows
        ),
        flush=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{rung}-{variant}.json").write_text(
        json.dumps(
            {
                "rung": rung,
                "variant": variant,
                "window_kv_format": wkv,
                "device": dsum,
                "official": osum,
                "official_meta": ometa,
                "nll_golden": gnll,
                "err_final_norm": drel,
                "err_final_norm_official": orel,
                "limit": limit,
                "info": info,
                "worst": rows,
            },
            indent=1,
        )
    )
    if variant == "device":
        assert dsum["kl_mean"] <= limit, f"kl_mean_{rung} {dsum['kl_mean']:.5g} > limit {limit:.5g}"
