# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Offline analysis of device chain runs (tests/test_chain_steps_device.py -> .pt) against the bf16 and fp32 CPU reference chains.

    python -m ...reference.chain_analyze --dev a.pt [--dev b.pt ...] --ref /mnt/tt-data/ssinghal/dsv4-chain-m --ref32 /mnt/tt-data/ssinghal/dsv4-chain-f32
Per layer (step 0 and mean over steps) it prints stream PCC, relative L2 error of the stream EXCLUDING the top-k massive channels (k=--topk,
chosen by mean |ref| per channel), mean/min per-token cosine, the router flip rate (device top-6 set != reference set) and the share of the
layer-output error energy carried by flipped tokens. With --ref32 it also gives the reference's own bf16-vs-fp32 noise floor.
"""
import argparse
import os

import torch


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def rel_excl(got, ref, topk):
    """relative L2 error of [N, 4, D] streams excluding the ``topk`` channels with the largest mean |ref|"""
    g, r = got.float().reshape(-1, got.shape[-1]), ref.float().reshape(-1, ref.shape[-1])
    if topk:
        mass = r.abs().mean(0)
        keep = torch.ones(r.shape[-1], dtype=torch.bool)
        keep[mass.topk(topk).indices] = False
        g, r = g[:, keep], r[:, keep]
    return float((g - r).norm() / (r.norm() + 1e-30))


def tok_cos(got, ref):
    g, r = got.float().reshape(got.shape[0], -1), ref.float().reshape(ref.shape[0], -1)
    return torch.nn.functional.cosine_similarity(g, r, dim=-1)


def set_flip(dev_idx, ref_idx):
    """[B] bool: the top-k expert SETS differ"""
    d, r = dev_idx.long(), ref_idx.long()
    return ~((d[:, :, None] == r[:, None, :]).any(-1).all(-1))


def load_ref(chain, L):
    return torch.load(os.path.join(chain, f"layer_{L}.pt"), mmap=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="append", required=True)
    ap.add_argument("--ref", default="/mnt/tt-data/ssinghal/dsv4-chain-m")
    ap.add_argument("--ref32", default=None)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--table", action="store_true", help="print the per-layer table")
    a = ap.parse_args()
    lo, hi = [int(x) for x in a.layers.split("-")]
    layers = list(range(lo, hi + 1))
    devs = {os.path.basename(p): torch.load(p) for p in a.dev}
    summary = {}
    for name, d in devs.items():
        have = sorted({k[0] for k in d["x"]})
        steps = sorted({k[1] for k in d["x"]})
        print(
            f"==== {name}  teacher={d['teacher']} layers {have[0]}..{have[-1]} steps {steps} env={ {k: v for k, v in d['env'].items() if k not in ('DSV41_OUT',)} }"
        )
        rows = []
        for L in [l for l in layers if l in have]:
            ref = load_ref(a.ref, L)
            r32 = load_ref(a.ref32, L) if a.ref32 and os.path.exists(os.path.join(a.ref32, f"layer_{L}.pt")) else None
            line = {"L": L}
            pc, re_, rx, cs, cmin, fl, share, p32, rr, flip32 = [], [], [], [], [], [], [], [], [], []
            for s in steps:
                xo = d["x"][(L, s)].float()
                ro = ref["dec_out_steps"][s][0].float().reshape(-1, 4, 5120)
                pc.append(pcc(xo, ro))
                re_.append(rel_excl(xo, ro, 0))
                rx.append(rel_excl(xo, ro, a.topk))
                c = tok_cos(xo, ro)
                cs.append(float(c.mean())), cmin.append(float(c.min()))
                if r32 is not None:
                    o32 = r32["dec_out_steps"][s][0].float().reshape(-1, 4, 5120)
                    p32.append(pcc(xo, o32))
                    rr.append(pcc(ro, o32))
                # expert-set flips vs the reference routing of this step
                ridx = None
                if s == 0 and ref.get("routing_idx") is not None:
                    ridx = ref["routing_idx"]
                if r32 is not None and r32.get("routing_steps"):
                    ridx32 = r32["routing_steps"][s][1]
                    flip32.append(float(set_flip(d["idx"][(L, s)], ridx32).float().mean())) if (L, s) in d[
                        "idx"
                    ] else None
                if ridx is not None and (L, s) in d["idx"]:
                    f = set_flip(d["idx"][(L, s)], ridx)
                    fl.append(float(f.float().mean()))
                    e = (xo - ro).pow(2).sum((1, 2))
                    share.append(float(e[f].sum() / (e.sum() + 1e-30)))
            line.update(
                pcc0=pc[0],
                pcc=sum(pc) / len(pc),
                rel=sum(re_) / len(re_),
                relx=sum(rx) / len(rx),
                cos=sum(cs) / len(cs),
                cosmin=min(cmin),
                flip=(sum(fl) / len(fl) if fl else float("nan")),
                share=(sum(share) / len(share) if share else float("nan")),
                pcc32=(sum(p32) / len(p32) if p32 else float("nan")),
                refnoise=(sum(rr) / len(rr) if rr else float("nan")),
                flip32=(sum(flip32) / len(flip32) if flip32 else float("nan")),
                pcc_by_step=pc,
            )
            rows.append(line)
        if a.table:
            print(
                " L  pcc@0  pcc(mean) relErr  relErr-topk cos  cosmin | flip@0 errShare | pcc_vs_fp32 ref_bf16_vs_fp32 flip_vs_fp32"
            )
            for r in rows:
                print(
                    f"{r['L']:2d} {r['pcc0']:.4f} {r['pcc']:.4f}   {r['rel']:.4f} {r['relx']:.4f}     {r['cos']:.4f} {r['cosmin']:.3f} | {r['flip']:.3f} {r['share']:.3f} | {r['pcc32']:.4f} {r['refnoise']:.4f} {r['flip32']:.3f}"
                )
        summary[name] = rows
        last = rows[-1]
        print(
            f"layer {last['L']}: pcc@0 {last['pcc0']:.4f}  mean-over-steps {last['pcc']:.4f}  per-step {[round(x, 4) for x in last['pcc_by_step']]}"
        )
        # final logits per step
        if last["L"] == 39:
            from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostHead

            head = HostHead()
            fin = torch.load(os.path.join(a.ref, "final.pt"))
            fin32 = (
                torch.load(os.path.join(a.ref32, "final.pt"))
                if a.ref32 and os.path.exists(os.path.join(a.ref32, "final.pt"))
                else None
            )
            for s in steps:
                lg = head(d["x"][(39, s)].float().reshape(-1, 1, 4, 5120), d["pre"][(39, s)].float().reshape(-1, 1, 4))
                ref_lg = fin["logits_steps"][s]
                m = int((lg.argmax(-1) == ref_lg.argmax(-1)).sum())
                pu = [round(pcc(lg[u], ref_lg[u]), 3) for u in range(lg.shape[0])]
                extra = ""
                if fin32 is not None:
                    extra = f" | vs fp32-ref: logits PCC {pcc(lg, fin32['logits_steps'][s]):.4f}, match {int((lg.argmax(-1) == fin32['logits_steps'][s].argmax(-1)).sum())}/{lg.shape[0]}; bf16-ref vs fp32-ref PCC {pcc(ref_lg, fin32['logits_steps'][s]):.4f} match {int((ref_lg.argmax(-1) == fin32['logits_steps'][s].argmax(-1)).sum())}"
                print(
                    f"  LOGITS step {s}: PCC {pcc(lg, ref_lg):.4f}  top-1 match {m}/{lg.shape[0]}  per-user PCC {pu}{extra}"
                )


if __name__ == "__main__":
    main()
