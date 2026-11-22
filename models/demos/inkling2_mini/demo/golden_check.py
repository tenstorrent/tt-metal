# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced check of the TT model against the CPU golden (reference.md layout).

Every application gets the golden input of that application (HF dataflow, see model.golden_app_inputs); stack and
messages follow from those inputs. Prints per-app PCC over all positions, then logits PCC / top-1 / top-5.

    flock /tmp/tt-card.lock python -m models.demos.inkling2_mini.demo.golden_check [--prompts 0 1] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.demos.inkling2_mini.tt.model import APPS, VOCAB_REAL, Inkling2Model, golden_app_inputs

GOLDEN = Path("/work/inkling-bringup/golden")


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def row0(t):
    return ttnn.to_torch(t).float().reshape(-1, t.shape[-1])[0]


def run_prompt(model, k, force=True, tag="", gdir=GOLDEN, save=None):
    """k: golden prompt name (e.g. "chat") or index n for "prompt{n}". save: dir for the raw TT tensors."""
    name = k if isinstance(k, str) else f"prompt{k}"
    d = Path(gdir) / name
    ids = torch.load(d / "input_ids.pt")[0].tolist()
    embed = torch.load(d / "embed.pt")[0].float()
    hid = [torch.load(d / f"hidden_{a:02d}.pt")[0].float() for a in range(len(APPS))]
    ref_logits = torch.load(d / "logits.pt")[0].float()
    T = len(ids)
    tt_apps = torch.zeros(len(APPS), T, embed.shape[-1])
    tt_embed = torch.zeros(T, embed.shape[-1])
    tt_logits = torch.zeros(T, VOCAB_REAL)
    tt_tok = []
    model.reset()
    t0 = time.time()
    for t in range(T):
        inputs, keys = golden_app_inputs([h[t] for h in hid], embed[t])
        model.set_inputs(ids[t], t)
        r = model.step(force={"inputs": inputs, "msg_keys": keys} if force else None, collect=True)
        tt_embed[t] = row0(r["embed"])
        for a, h in enumerate(r["apps"]):
            tt_apps[a, t] = row0(h)
        tt_logits[t] = ttnn.to_torch(r["logits"]).float().reshape(-1)[:VOCAB_REAL]
        tt_tok.append(int(ttnn.to_torch(r["token"]).reshape(-1)[0]))
    secs = time.time() - t0
    if save is not None:
        torch.save(
            {"apps": tt_apps, "embed": tt_embed, "logits": tt_logits},
            Path(save) / f"{'tf' if force else 'free'}{tag}_{name}.pt",
        )
    ref_top1 = ref_logits.argmax(-1)
    tt_top5 = tt_logits.topk(5, dim=-1).indices
    out = {
        "prompt": k,
        "T": T,
        "seconds": round(secs, 1),
        "embed_pcc": pcc(tt_embed, embed),
        "app_pcc": [pcc(tt_apps[a], hid[a]) for a in range(len(APPS))],
        "app_pcc_min_pos": [min(pcc(tt_apps[a, t], hid[a][t]) for t in range(T)) for a in range(len(APPS))],
        "logits_pcc": pcc(tt_logits, ref_logits),
        "logits_pcc_min_pos": min(pcc(tt_logits[t], ref_logits[t]) for t in range(T)),
        "kl_mean": float(
            torch.nn.functional.kl_div(
                tt_logits.log_softmax(-1), ref_logits.log_softmax(-1), log_target=True, reduction="none"
            )
            .sum(-1)
            .mean()
        ),
        "top1_agree": float((tt_logits.argmax(-1) == ref_top1).float().mean()),
        "top5_has_ref_top1": float((tt_top5 == ref_top1[:, None]).any(-1).float().mean()),
        "device_argmax_eq_host": float((torch.tensor(tt_tok) == tt_logits.argmax(-1)).float().mean()),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", default=["0", "1"], help="indices (prompt{n}) or golden names")
    ap.add_argument("--max-ctx", type=int, default=2048)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--b8-layers", type=int, nargs="*", default=[], help="layers loaded from --b8-root")
    ap.add_argument("--b8-root", type=str, default="/work/inkling-bringup/cache/v1-b8")
    ap.add_argument("--trace-mib", type=int, default=512)
    ap.add_argument("--no-force", action="store_true", help="feed golden tokens only (no hidden forcing)")
    args = ap.parse_args()
    dev = ttnn.open_device(device_id=0, trace_region_size=args.trace_mib << 20)
    try:
        model = Inkling2Model(
            dev, max_ctx=args.max_ctx, log=lambda s: None, layer_roots={L: args.b8_root for L in args.b8_layers}
        )
        res = [
            run_prompt(
                model,
                int(k) if k.isdigit() else k,
                not args.no_force,
                "_b8" + "_".join(map(str, args.b8_layers)) if args.b8_layers else "",
                save="/work/inkling-bringup/A1-scratch",
            )
            for k in args.prompts
        ]
    finally:
        ttnn.close_device(dev)
    for r in res:
        print(f"{r['prompt']} T={r['T']} {r['seconds']}s embed {r['embed_pcc']:.5f}")
        for a, (L, p) in enumerate(APPS):
            print(f"  app {a:2d} L{L:02d} p{p} pcc {r['app_pcc'][a]:.5f} min/pos {r['app_pcc_min_pos'][a]:.5f}")
        print(
            f"  logits pcc {r['logits_pcc']:.5f} (min/pos {r['logits_pcc_min_pos']:.5f}) top1 {r['top1_agree']:.3f} "
            f"top5 {r['top5_has_ref_top1']:.3f} dev-argmax==host {r['device_argmax_eq_host']:.3f} kl {r['kl_mean']:.4f}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
