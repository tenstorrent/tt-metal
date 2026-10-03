# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Offline metrics of the end-to-end GSM8K harness (tests/test_e2e_gsm8k_device.py) against the cached CPU reference generation.

    python -m ...reference.e2e_analyze --dev /mnt/tt-data/ssinghal/dsv4-e2e-dev/base [--ref /mnt/tt-data/ssinghal/dsv4-e2e-ref] [--kv]
Closed loop : first-divergence position vs the reference greedy text, token agreement before divergence, GSM8K exact match (device vs reference vs gold).
Teacher forced: per-step logits PCC, top-1 agreement, top-5 containment, KL(ref||dev) (full logits for the first 64 steps, top-50 approximation after).
KV          : per-layer PCC / relative error of the device-written window + compressed KV vs the reference's, by age of the entry.
"""
import argparse
import glob
import json
import os
import re

import torch


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def extract(text):
    """final numeric answer of a model completion: '#### x' / \\boxed{x} / the last number"""
    for pat in (r"####\s*\$?(-?[\d,]*\.?\d+)", r"\\boxed\{\s*\\?\$?\s*(-?[\d,]*\.?\d+)"):
        m = re.findall(pat, text)
        if m:
            return norm(m[-1])
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return norm(m[-1]) if m else None


def norm(s):
    s = s.replace(",", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def gold(ans):
    return norm(ans.split("####")[-1].strip().replace(",", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--ref", default="/mnt/tt-data/ssinghal/dsv4-e2e-ref")
    ap.add_argument("--kv", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R

    tok = AutoTokenizer.from_pretrained(R.CKPT_DIR)
    meta = torch.load(os.path.join(a.ref, "meta.pt"))
    refres = torch.load(os.path.join(a.ref, "results.pt"))
    S = meta["S"]
    rs = refres["stream"]  # [Btot, S+n]
    EOS = 1
    files = sorted(glob.glob(os.path.join(a.dev, "res_b*.pt")), key=lambda p: int(re.findall(r"b(\d+)", p)[-1]))
    out = {"cl": [], "tf": {}}
    cl_rows = []
    tf_steps = {}
    for f in files:
        b = int(re.findall(r"b(\d+)", f)[-1])
        res = torch.load(f)
        sl = slice(16 * b, 16 * b + 16)
        refs = rs[sl, S:]  # reference tokens after the prompt
        # ---- first token (device prefill logits vs reference prefill logits) ----
        if res.get("logits0_topk") is not None:
            rv, ri = refres["topk_val"][0][sl], refres["topk_idx"][0][sl]
            dt = res["first_tok"]
            print(
                f"b{b}: first token match {int((dt == refs[:, 0]).sum())}/16; dev top1 in ref top5 {sum(int(dt[u] in ri[u, :5]) for u in range(16))}/16"
            )
        # ---- closed loop ----
        if "cl_tokens" in res:
            dtok = res["cl_tokens"]  # [16, n+1]
            n = min(dtok.shape[1], refs.shape[1])
            for u in range(16):
                g = b * 16 + u
                d, r = dtok[u, :n], refs[u, :n]
                # truncate both at their EOS
                dl = (d == EOS).nonzero()
                rl = (r == EOS).nonzero()
                d_end = int(dl[0]) + 1 if len(dl) else n
                r_end = int(rl[0]) + 1 if len(rl) else n
                neq = (d != r).nonzero()
                div = int(neq[0]) if len(neq) else n
                dtext = tok.decode(d[:d_end].tolist(), skip_special_tokens=True)
                rtext = tok.decode(r[:r_end].tolist(), skip_special_tokens=True)
                gd = gold(meta["gold"][g])
                cl_rows.append(
                    dict(
                        g=g,
                        div=div,
                        n=n,
                        d_done=bool(len(dl)),
                        r_done=bool(len(rl)),
                        d_ans=extract(dtext) if len(dl) else None,
                        r_ans=extract(rtext) if len(rl) else None,
                        gold=gd,
                        d_ans_any=extract(dtext),
                        r_ans_any=extract(rtext),
                        same_text=bool(div >= min(d_end, r_end) and d_end == r_end),
                    )
                )
        # ---- teacher forced ----
        if "tf_topk_idx" in res:
            tv, ti, tl = res["tf_topk_val"], res["tf_topk_idx"], res["tf_lse"]  # [N,16,50], [N,16,50], [N,16]
            N = tv.shape[0]
            full = res.get("tf_full")
            ref_full = None
            lf = os.path.join(a.ref, "logits_tf.pt")
            if os.path.exists(lf) and full is not None and full.shape[0]:
                ref_full = torch.load(lf, mmap=True)
            done_at = refres["done_at"][sl]
            for j in range(N):
                m = j + 1  # reference logits index
                valid = (done_at < 0) | (m <= done_at)
                if m >= refres["topk_val"].shape[0]:
                    break
                rv, ri, rl_ = refres["topk_val"][m][sl].float(), refres["topk_idx"][m][sl], refres["lse"][m][sl].float()
                dv, di, dl_ = tv[j].float(), ti[j], tl[j].float()
                st = tf_steps.setdefault(j, dict(pcc=[], top1=[], top5=[], rtop1_in_d5=[], kl=[], klfull=[]))
                for u in range(16):
                    if not valid[u]:
                        continue
                    st["top1"].append(float(di[u, 0] == ri[u, 0]))
                    st["top5"].append(float(di[u, 0] in ri[u, :5]))
                    st["rtop1_in_d5"].append(float(ri[u, 0] in di[u, :5]))
                    if ref_full is not None and j < full.shape[0] and m < ref_full.shape[0]:
                        rf, df = ref_full[m][16 * b + u].float(), full[j][u].float()
                        st["pcc"].append(pcc(rf, df))
                        lp_r, lp_d = torch.log_softmax(rf, -1), torch.log_softmax(df, -1)
                        st["klfull"].append(float((lp_r.exp() * (lp_r - lp_d)).sum()))
                    # top-50 KL approximation: p_ref over its top-50 vs the device's probability of those tokens
                    pr = (rv[u] - rl_[u]).exp()
                    dmap = dict(zip(di[u].tolist(), (dv[u] - dl_[u]).tolist()))
                    floor = float(dv[u, -1] - dl_[u])
                    lq = torch.tensor([dmap.get(int(t), floor) for t in ri[u]])
                    st["kl"].append(float((pr * ((rv[u] - rl_[u]) - lq)).sum()))
    # ---------------- report ----------------
    if cl_rows:
        n_q = len(cl_rows)
        divs = torch.tensor([r["div"] for r in cl_rows], dtype=torch.float)
        print(
            f"\nCLOSED LOOP over {n_q} questions (compared horizon {min(r['n'] for r in cl_rows)}-{max(r['n'] for r in cl_rows)} tokens)"
        )
        print(
            f"  first-divergence position: mean {divs.mean():.1f} median {divs.median():.0f} min {divs.min():.0f}; identical full text: {sum(r['same_text'] for r in cl_rows)}/{n_q}"
        )
        print(
            "  divergence histogram (tokens): "
            + str(
                {
                    k: int(((divs >= lo) & (divs < hi)).sum())
                    for k, (lo, hi) in {
                        "0-4": (0, 5),
                        "5-15": (5, 16),
                        "16-39": (16, 40),
                        "40-99": (40, 100),
                        "100+": (100, 1e9),
                    }.items()
                }
            )
        )
        bothdone = [r for r in cl_rows if r["d_done"] and r["r_done"]]
        refok = sum(r["r_ans_any"] == r["gold"] for r in cl_rows)
        devok = sum(r["d_ans_any"] == r["gold"] for r in cl_rows)
        agree = sum(r["d_ans_any"] == r["r_ans_any"] for r in cl_rows)
        print(
            f"  GSM8K exact match: reference {refok}/{n_q} = {100 * refok / n_q:.1f}%  device {devok}/{n_q} = {100 * devok / n_q:.1f}%  (final answers equal dev vs ref: {agree}/{n_q}); finished (EOS) within horizon: ref {sum(r['r_done'] for r in cl_rows)} dev {sum(r['d_done'] for r in cl_rows)}"
        )
        if bothdone:
            rok = sum(r["r_ans"] == r["gold"] for r in bothdone)
            dok = sum(r["d_ans"] == r["gold"] for r in bothdone)
            print(f"  restricted to {len(bothdone)} questions finished by both: reference {rok} device {dok}")
        json.dump(cl_rows, open(os.path.join(a.dev, "cl_rows.json"), "w"))
    if tf_steps:
        print("\nTEACHER FORCED (step j = j-th decode step, i.e. context length S+j; averages over valid users)")
        print("  step  logitsPCC  top1   top5   ref1in_dev5  KL(ref||dev) [full | top50-approx]")
        buckets = [
            (0, 1),
            (1, 2),
            (2, 4),
            (4, 8),
            (8, 16),
            (16, 32),
            (32, 64),
            (64, 96),
            (96, 128),
            (128, 192),
            (192, 256),
        ]
        for lo, hi in buckets:
            ks = [k for k in tf_steps if lo <= k < hi]
            if not ks:
                continue
            g = (
                lambda key: sum((x for k in ks for x in tf_steps[k][key]), [])
                if False
                else [x for k in ks for x in tf_steps[k][key]]
            )
            mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
            print(
                f"  {lo:3d}-{hi - 1:3d}  {mean(g('pcc')):.4f}   {mean(g('top1')):.3f}  {mean(g('top5')):.3f}  {mean(g('rtop1_in_d5')):.3f}       {mean(g('klfull')):.4f} | {mean(g('kl')):.4f}   (n={len(g('top1'))})"
            )
    if a.kv:
        kvf = sorted(glob.glob(os.path.join(a.dev, "kv_b*.pt")))
        for f in kvf:
            b = int(re.findall(r"kv_b(\d+)", f)[0])
            dev = torch.load(f)
            # res file holds n_written
            n_dev = torch.load(os.path.join(a.dev, f"res_b{b}.pt"))["kv_n_written"]
            print(
                f"\nKV error, batch {b} (device wrote positions < {n_dev}); window: PCC / rel err by age bucket; comp: PCC / rel err"
            )
            for L in sorted(dev):
                kvr = torch.load(os.path.join(a.ref, "kv_final", f"layer_{L}.pt"))
                n_ref = kvr["n_written"]
                d = dev[L][:, :, :].float()  # [16, W+C, 512]
                w_ref = kvr["window"][16 * b : 16 * b + 16].float()
                ring = w_ref.shape[1]  # 128
                if (
                    d.shape[1] > ring + 1 and d.shape[1] - ring < 1000 and d.shape[1] != 256
                ):  # compressed layer: ring + latents
                    dw = d[:, :ring]
                    lo = max(n_ref - ring, n_dev - ring, 0)
                    ps = [p for p in range(lo, min(n_ref, n_dev))]
                    ages = [n_dev - p for p in ps]
                    sel = lambda xs, sl_: torch.stack([xs[:, p % ring] for p in sl_], 1)
                else:  # ratio-0 layer: linear cache, position p at slot p
                    dw = d
                    lo = max(n_ref - ring, 0)
                    ps = [p for p in range(lo, min(n_ref, n_dev))]
                    ages = [n_dev - p for p in ps]
                if not ps:
                    continue
                dsel = torch.stack([dw[:, p % ring] if dw.shape[1] == ring else dw[:, p] for p in ps], 1)
                rsel = torch.stack([w_ref[:, p % ring] for p in ps], 1)
                rel = lambda x, y: float((x - y).norm() / (y.norm() + 1e-30))
                line = f"  L{L:2d} window pos {ps[0]}..{ps[-1]}: PCC {pcc(dsel, rsel):.4f} rel {rel(dsel, rsel):.4f}"
                if "comp" in kvr:
                    ref_c = kvr["comp"][16 * b : 16 * b + 16].float()
                    nc = min(
                        ref_c.shape[1], (n_dev) // max(1, int(round((n_dev) / max(ref_c.shape[1], 1))))
                    )  # entries both have
                    dc = d[:, ring:] if d.shape[1] > ring else None
                    if dc is not None:
                        nc = min(ref_c.shape[1], dc.shape[1], max(1, n_dev // max(1, round(n_ref / ref_c.shape[1]))))
                        line += f" | comp[0:{nc}]: PCC {pcc(dc[:, :nc], ref_c[:, :nc]):.4f} rel {rel(dc[:, :nc], ref_c[:, :nc]):.4f}"
                print(line)


if __name__ == "__main__":
    main()
