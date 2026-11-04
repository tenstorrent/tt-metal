# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer decode gate: every layer runs on the GOLDEN's input stream (x and the mHC pre weights at the top of the layer), so only that layer's own numerics show: no compounding
through the 8 layers and no amplification by routing flips of earlier layers.

TT_OPT_GOLDEN=write (all levers off): records the verify block (X, base), the layer inputs (x, pre), the layer / engram outputs, the routing ids (top-6 experts per token) and the logits.
TT_OPT_GOLDEN=check (levers on): the same round with the golden's X and with every layer's input replaced by the golden's; per layer the min PCC over the mesh rows of the output stream and
the fraction of tokens whose expert set differs from the golden's at that layer (routing flips on identical input); the logits of the last layer's output.
Pass: per-layer min PCC >= TT_OPT_LAYER_PCC (default 0.999) and logits PCC >= 0.999.
"""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.tt_opt import test_tt_opt_decode as T


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", T.PARAMS, indirect=True)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_decode_gate_perlayer(mesh_device):
    md = mesh_device
    _, runner, X, base, snaps = T._setup(md, minimal=True, tracy_layers=False) if T.GATE_MINIMAL else T._setup(md)
    dec = runner.dec
    cols = md.shape[1]
    path = os.path.join(T.GOLDEN_DIR, "perlayer.pt")
    mode = os.environ.get("TT_OPT_GOLDEN", "check")
    gold = None
    if mode == "check":
        gold = torch.load(path)
        X, base = gold["X"].clone(), gold["base"].clone()
        dec.gate_golden = gold["store"]
        dec.gate_mode = "inject"
    else:
        dec.gate_mode = "write"
    dec.gate_store = {}
    dec.dbg_store = []
    T._feed_eager(runner, X, base, -1)
    dec.forward()
    ttnn.synchronize_device(md)
    out = {"X": X.clone(), "base": base.clone(), "streams": {}}
    for tag, rows in dec.dbg_store:
        out["streams"][tag] = [r.clone() for r in rows]
    dec.dbg_store = None
    dec.gate_mode = None
    lg = dec.dbg_logits
    out["logits"] = torch.cat(
        [
            torch.cat(
                [
                    ttnn.to_torch(ttnn.get_device_tensors(lg)[r * cols + c]).float().reshape(dec.U * dec.n, -1)
                    for c in range(cols)
                ],
                dim=1,
            )
            for r in range(md.shape[0])
        ]
    )
    out["store"] = dec.gate_store
    if snaps is not None:
        dec.restore_states(snaps)
    if mode == "write":
        os.makedirs(T.GOLDEN_DIR, exist_ok=True)
        torch.save(out, path)
        rep = {k[1]: v[2] for k, v in out["store"].items() if k[0] == "in"}
        T._log(
            f"PERLAYER golden written: {path} streams {len(out['streams'])} routing layers {sum(1 for k in out['store'] if k[0] == 'route')} x/pre replicated over the mesh columns: {rep}"
        )
        return
    thr = float(os.environ.get("TT_OPT_LAYER_PCC", "0.999"))
    bad = []
    lines = []
    for tag, rows in gold["streams"].items():
        pccs = [T._pcc(g, c) for g, c in zip(rows, out["streams"][tag])]
        lid = int(tag.split()[1])
        flips, tot = 0, 0
        rk = ("route", lid)
        clean, flipped = [], []  # per-token PCC of the tokens whose expert set did / did not change at this layer
        if tag.startswith("layer") and rk in gold["store"] and rk in out["store"]:
            for g, c, sg, sc in zip(gold["store"][rk], out["store"][rk], rows, out["streams"][tag]):
                g2 = torch.sort(g.reshape(g.shape[0], -1).to(torch.int64), dim=-1)[0]
                c2 = torch.sort(c.reshape(c.shape[0], -1).to(torch.int64), dim=-1)[0]
                diff = (g2 != c2).any(dim=-1)
                flips += int(diff.sum())
                tot += g2.shape[0]
                for i in range(g2.shape[0]):
                    (flipped if bool(diff[i]) else clean).append(T._pcc(sg[i], sc[i]))
        else:
            clean = list(pccs)
        fl = f"{flips / tot:.3f}" if tot else "n/a"
        cmin = min(clean) if clean else 1.0
        fmin = min(flipped) if flipped else 1.0
        lines.append(
            f"PERLAYER {tag:11s} min PCC {min(pccs):.6f} rows {[round(p, 5) for p in pccs]} route flips {fl} ({flips}/{tot}) | no-flip tokens min PCC {cmin:.6f} (n={len(clean)}) flipped tokens min PCC {fmin:.6f} (n={len(flipped)})"
        )
        if cmin < thr:  # a defect: a token whose experts did not change but whose layer output differs
            bad.append((tag, round(cmin, 6)))
    for ln in lines:
        T._log(ln)
    t2 = T._pcc(gold["logits"], out["logits"])
    T._log(
        f"PERLAYER logits PCC {t2:.6f}; worst layer {min([float(l.split('min PCC ')[1].split()[0]) for l in lines]):.6f}; threshold {thr}"
    )
    print(
        f"PERLAYER_SCORES min_layer_pcc={min([float(l.split('min PCC ')[1].split()[0]) for l in lines]):.6f} logits_pcc={t2:.6f}",
        flush=True,
    )
    assert not bad, f"layers below {thr}: {bad}"
    assert t2 >= 0.999, f"logits PCC {t2}"
