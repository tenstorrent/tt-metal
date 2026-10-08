# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""L2 prefill accuracy gate: compare a candidate's KV / logits dump against the baseline build's dump.

Both dumps come from kagent_prefill_bench.py (BENCH_DUMP_DIR) on the same tokens, prefix and request:
kv_{k,v,ik}.pt are bf16 [layers, heads, tokens, 128] in natural token order; logits_top5.pt holds the top-5
LM-head ids/values for every request row.

Gate (lead's L2 prefill rule, docs/tenstorrent/m3/05-accuracy.md §6.2c adapted to baseline-relative):
  * per layer and per cache (K, V, index_k): PCC(candidate, baseline) over the whole [0, tokens) range and over
    the request rows [prefix, tokens) separately; 1 - PCC <= 0.05 and PCC >= 0.88 for every layer
    (bit-exact -> class A, passes trivially)
  * teacher-forced next-token top-1 agreement of the candidate vs the baseline over the request rows (reported;
    >= 99% expected for a same-precision change), top-1 flips at positions where the baseline is confident
    (top1 - top2 logit >= 1.0) and top-1 == next prompt token for both

Usage: python kagent_prefill_compare.py <baseline_dump> <candidate_dump> [--prefix 51200] [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    den = (a.norm() * b.norm()).item()
    if den == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return (a @ b).item() / den


def rel_l2(a, b):
    a = a.double()
    b = b.double()
    d = b.norm().item()
    return (a - b).norm().item() / d if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--prefix", type=int, default=51200)
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--floor", type=float, default=0.88)
    ap.add_argument("--json")
    ap.add_argument(
        "--noise-floor",
        help="compare.json of a BENIGN perturbation (e.g. single-RS vs its bit-exact twin, or chunk 4096 vs 5120): "
        "calibrated gate = every layer/cache PCC >= that arm's PCC - --floor-margin (the literal 5%%/0.88 rule "
        "only passes bit-exact candidates because prefill numerics are chaotic over 60 layers)",
    )
    ap.add_argument("--floor-margin", type=float, default=0.02)
    ap.add_argument("--nll-tol", type=float, default=0.02, help="calibrated gate: upper 95%% CI of mean NLL delta")
    a = ap.parse_args()
    base, cand = Path(a.base), Path(a.cand)
    out = {"base": str(base), "cand": str(cand), "layers": {}}
    ok = True
    bitexact = True
    worst = {}
    has_kv = all((d / f"kv_{n}.pt").exists() for d in (base, cand) for n in ("k", "v", "ik"))
    if not has_kv:
        print("[compare] no KV dump in one of the dirs: logits / NLL comparison only")
    for name in ("k", "v", "ik") if has_kv else ():
        kb = torch.load(base / f"kv_{name}.pt")
        kc = torch.load(cand / f"kv_{name}.pt")
        assert kb.shape == kc.shape, (name, kb.shape, kc.shape)
        n_tok = kb.shape[2]
        worst[name] = (1.0, 1.0, -1)
        for L in range(kb.shape[0]):
            b, c = kb[L], kc[L]
            if b.abs().sum() == 0 and c.abs().sum() == 0:
                continue  # index_k on dense layers
            exact = torch.equal(b, c)
            bitexact &= exact
            p_all = pcc(c, b)
            p_req = pcc(c[:, a.prefix : n_tok], b[:, a.prefix : n_tok])
            r_all = rel_l2(c, b)
            out["layers"].setdefault(L, {})[name] = {
                "exact": exact,
                "pcc_all": p_all,
                "pcc_request": p_req,
                "rel_l2_all": r_all,
            }
            m = min(p_all, p_req)
            if m < worst[name][0]:
                worst[name] = (m, r_all, L)
            if (1 - m) > a.budget or m < a.floor:
                ok = False
    out["worst"] = {k: {"pcc": v[0], "rel_l2": v[1], "layer": v[2]} for k, v in worst.items()}
    out["kv_bitexact"] = bitexact
    lb, lc = base / "logits_top5.pt", cand / "logits_top5.pt"
    if lb.exists() and lc.exists():
        db, dc = torch.load(lb), torch.load(lc)
        assert torch.equal(db["positions"], dc["positions"])
        tb, tc = db["top_ids"][:, 0], dc["top_ids"][:, 0]
        agree = (tb == tc).float().mean().item()
        margin = db["top_vals"][:, 0] - db["top_vals"][:, 1]
        conf = margin >= 1.0
        flips_conf = ((tb != tc) & conf).sum().item()
        nxt = db["next_tokens"]
        valid = nxt >= 0
        acc_b = (tb[valid] == nxt[valid]).float().mean().item()
        acc_c = (tc[valid] == nxt[valid]).float().mean().item()
        top5_c_has_b = (dc["top_ids"] == tb[:, None]).any(dim=1).float().mean().item()
        out["logits"] = {
            "rows": int(tb.numel()),
            "top1_agreement": agree,
            "baseline_top1_in_candidate_top5": top5_c_has_b,
            "confident_rows": int(conf.sum().item()),
            "flips_at_confident_rows": int(flips_conf),
            "next_token_acc_base": acc_b,
            "next_token_acc_cand": acc_c,
            "nll_base": float(db["nll"][~db["nll"].isnan()].mean()) if "nll" in db else None,
            "nll_cand": float(dc["nll"][~dc["nll"].isnan()].mean()) if "nll" in dc else None,
            "logits_bitexact": bool(
                torch.equal(db["top_vals"], dc["top_vals"]) and torch.equal(db["top_ids"], dc["top_ids"])
            ),
        }
    if "logits" in out and "nll" in db and "nll" in dc:
        # paired bootstrap of mean NLL(cand) - NLL(base) over request rows (teacher-forced perplexity on the text)
        m = ~(db["nll"].isnan() | dc["nll"].isnan())
        diff = (dc["nll"][m] - db["nll"][m]).double()
        # moving-block bootstrap: neighbouring rows share context, so per-row resampling would understate the CI
        g = torch.Generator().manual_seed(0)
        blk = 256
        nb = max(1, diff.numel() // blk)
        starts_max = diff.numel() - blk + 1
        boots = torch.stack(
            [
                torch.cat([diff[s : s + blk] for s in torch.randint(0, starts_max, (nb,), generator=g).tolist()]).mean()
                for _ in range(2000)
            ]
        )
        out["logits"]["nll_delta"] = diff.mean().item()
        out["logits"]["nll_delta_ci95"] = [boots.quantile(0.025).item(), boots.quantile(0.975).item()]
    out["gate_pass"] = ok
    print(f"[compare] {cand} vs {base}")
    print(f"[compare] KV bit-exact: {bitexact}")
    for k, v in out["worst"].items():
        print(
            f"[compare]   worst {k:>2}: PCC {v['pcc']:.6f} (1-PCC {1 - v['pcc']:.2e}) relL2 {v['rel_l2']:.4f} at layer {v['layer']}"
        )
    if "logits" in out:
        g = out["logits"]
        print(
            f"[compare]   logits: top-1 agreement {g['top1_agreement']*100:.2f}% over {g['rows']} rows; flips at "
            f"confident rows {g['flips_at_confident_rows']}/{g['confident_rows']}; base-top1 in cand-top5 "
            f"{g['baseline_top1_in_candidate_top5']*100:.2f}%; next-token acc base {g['next_token_acc_base']*100:.2f}% "
            f"cand {g['next_token_acc_cand']*100:.2f}%; logits bit-exact {g['logits_bitexact']}"
        )
    if "logits" in out and out["logits"].get("nll_delta") is not None:
        g = out["logits"]
        print(
            f"[compare]   teacher-forced NLL base {g['nll_base']:.4f} cand {g['nll_cand']:.4f}; paired delta "
            f"{g['nll_delta']:+.4f} nats/token, 95% CI [{g['nll_delta_ci95'][0]:+.4f}, {g['nll_delta_ci95'][1]:+.4f}]"
        )
    if a.noise_floor:
        nf = json.load(open(a.noise_floor))["layers"]
        worst_gap, cal_ok = (1.0, None), True
        for L, caches in out["layers"].items():
            for name, r in caches.items():
                ref = nf.get(str(L), {}).get(name)
                if ref is None or r["exact"]:
                    continue
                m = min(r["pcc_all"], r["pcc_request"])
                fl = min(ref["pcc_all"], ref["pcc_request"]) - a.floor_margin
                if m - fl < worst_gap[0]:
                    worst_gap = (m - fl, f"layer {L} {name}: {m:.5f} vs floor {fl:.5f}")
                cal_ok &= m >= fl
        g = out.get("logits", {})
        nll_ok = g.get("nll_delta_ci95") is None or g["nll_delta_ci95"][1] <= a.nll_tol
        out["calibrated_gate"] = {"kv_ok": cal_ok, "nll_ok": nll_ok, "tightest": worst_gap[1]}
        print(
            f"[compare] calibrated gate (KV >= noise floor - {a.floor_margin} every layer; NLL delta CI95 upper <= "
            f"{a.nll_tol}): KV {'PASS' if cal_ok else 'FAIL'} (tightest {worst_gap[1]}), NLL "
            f"{'PASS' if nll_ok else 'FAIL'}"
        )
    print(f"[compare] L2 KV gate (1-PCC <= {a.budget}, PCC >= {a.floor} every layer/cache): {'PASS' if ok else 'FAIL'}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
