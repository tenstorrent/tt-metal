#!/usr/bin/env python3
# Mt=8 characterization + comparable baselines (Parts 2-3 of the Mt=8 objective).
#   - 4 primary M=256 shapes: config=None (product) + a PRACTICAL exhaustive sweep over all 5 levers
#     (documented sub-domain: kb in {1,2,4,8}, nsb<=8; deep-kb is covered separately by the K-depth
#     experiment). Ranked by MEDIAN steady-state kernel us (min + spread retained).
#   - M-scaling series {32,64,128,256} at K,N = (2048,1024) small-N and (6144,4608) wide-N: config=None
#     + best-manual, to separate an Mt-dependent cost from small-N fixed overhead.
# Emits mt8_characterize.json (+ per-shape regime_a_sweep_*.json via the sweep) and MT8_CHARACTERIZATION.md.
# Resumable via its own CacheStore (mt8_characterize_cache.json). Compare median-with-median.
import json, os, sys

import regime_a_bench as b

HERE = os.path.dirname(__file__)
CACHE = f"{HERE}/mt8_characterize_cache.json"
OUTJSON = f"{HERE}/mt8_characterize.json"
OUTMD = f"{HERE}/MT8_CHARACTERIZATION.md"

PRIMARY = [(256, 2048, 1024), (256, 6144, 768), (256, 6144, 2304), (256, 6144, 4608)]
MSCALE_SMALLN = [(32, 2048, 1024), (64, 2048, 1024), (128, 2048, 1024), (256, 2048, 1024)]
MSCALE_WIDEN = [(32, 6144, 4608), (64, 6144, 4608), (128, 6144, 4608), (256, 6144, 4608)]

KB_SET = (1, 2, 4, 8)
NSB_MAX = 8
DOMAIN_DOC = (
    f"practical planner-valid sub-domain: Pk 1..min(Kt,13), Ns 1..6, Sm 1..Mt (cores<=104, L1<=1440KB), "
    f"kb in {KB_SET}, nsb 1..min(N_own,{NSB_MAX}). NOT full: kb>8 and nsb>{NSB_MAX} are excluded here "
    f"(deep-kb handled by the K-depth experiment)."
)


def sweep_shape(M, K, N, store):
    cfgs = b.enumerate_feasible(M, K, N, kb_set=KB_SET, nsb_max=NSB_MAX)
    print(f"[sweep] {M}x{K}x{N}: {len(cfgs)} practical configs", flush=True)
    res = []
    for i, c in enumerate(cfgs):
        r = b.run_cfg(M, K, N, c, store)
        if b._ok(r):
            res.append(r)
    res.sort(key=lambda r: r["us_med"])
    # persist a per-shape sweep file too
    swout = f"{HERE}/regime_a_sweep_{M}x{K}x{N}.json"
    json.dump(
        {"M": M, "K": K, "N": N, "domain": DOMAIN_DOC, "n_valid": len(cfgs), "results": res}, open(swout, "w"), indent=2
    )
    return res


def enrich(M, K, N, rec):
    """Attach plan_metrics + eff-BW reference points to an ok record."""
    if not b._ok(rec):
        return None
    pm = b.plan_metrics(M, K, N, tuple(rec["cfg"]))
    return {
        "cfg": rec["cfg"],
        "cores": pm["cores"],
        "us_med": rec["us_med"],
        "us_min": rec["us_min"],
        "spread_pct": rec["us_spread_pct"],
        "pct512_med": rec["pct512"],
        "eff_gbps": rec["eff_gbps"],
        "pcc": rec["pcc"],
        "logical_bytes": pm["logical_bytes"],
        "ideal_us_512": pm["ideal_us_512"],
        "Ktl": pm["Ktl"],
        "Mblk": pm["Mblk"],
        "N_own": pm["N_own"],
        "N_bpc": pm["N_bpc"],
        "N_slice": pm["N_slice"],
        "k_pad_tiles": pm["k_pad_tiles"],
        "cb0_in1": pm["cb0_in1"],
        "cb1_in0": pm["cb1_in0"],
        "cb2_interm": pm["cb2_interm"],
        "cb3_out": pm["cb3_out"],
        "cb7_reduce": pm["cb7_reduce"],
        "l1_bytes": pm["l1_bytes"],
        "l1_pct": pm["l1_pct"],
    }


def main():
    store = b.CacheStore(CACHE)
    out = {"domain": DOMAIN_DOC, "primary": [], "mscale_smallN": [], "mscale_wideN": []}

    for M, K, N in PRIMARY:
        prod = b.run_cfg(M, K, N, None, store)
        res = sweep_shape(M, K, N, store)
        best = res[0] if res else None
        best_sm1 = next((r for r in res if r["cfg"][2] == 1), None)
        best_smgt1 = next((r for r in res if r["cfg"][2] > 1), None)
        rec = {
            "M": M,
            "K": K,
            "N": N,
            "Mt": b.cdiv(M, 32),
            "n_ok": len(res),
            "auto": enrich(M, K, N, prod),
            "best_overall": enrich(M, K, N, best),
            "best_sm1": enrich(M, K, N, best_sm1),
            "best_smgt1": enrich(M, K, N, best_smgt1),
            "top8": [enrich(M, K, N, r) for r in res[:8]],
        }
        out["primary"].append(rec)
        bu = best["us_med"] if best else float("nan")
        print(
            f"[primary] {M}x{K}x{N} auto={prod.get('cfg')} {prod.get('us_med',0):.1f}us "
            f"best={best['cfg'] if best else None} {bu:.1f}us",
            flush=True,
        )
        json.dump(out, open(OUTJSON, "w"), indent=2)

    for series, shapes in (("mscale_smallN", MSCALE_SMALLN), ("mscale_wideN", MSCALE_WIDEN)):
        for M, K, N in shapes:
            prod = b.run_cfg(M, K, N, None, store)
            man = b.best_manual(M, K, N, json.load(open(b.SWEEP)), store, b.auto_config)
            out[series].append(
                {
                    "M": M,
                    "K": K,
                    "N": N,
                    "Mt": b.cdiv(M, 32),
                    "auto": enrich(M, K, N, prod),
                    "best_manual": enrich(M, K, N, man),
                }
            )
            print(
                f"[{series}] {M}x{K}x{N} auto {prod.get('us_med',0):.1f}us "
                f"best {man.get('us_med',0) if b._ok(man) else float('nan'):.1f}us",
                flush=True,
            )
            json.dump(out, open(OUTJSON, "w"), indent=2)

    write_md(out)
    print(f"WROTE {OUTJSON} + {OUTMD}")


def write_md(out):
    L = ["# Mt=8 characterization + comparable baselines\n"]
    L.append(f"Domain: {out['domain']}\n")
    L.append(
        "Ranked by **median** steady-state kernel us (min + spread retained). op %512 = of 512 GB/s. "
        "`ideal_us_512` = logical_bytes / 512 GB/s (the pure-DRAM floor).\n"
    )
    L.append("## Primary M=256 shapes\n")
    hdr = (
        "| shape | Mt | which | cfg(Ns,Pk,Sm,kb,nsb) | cores | med us | min us | spread% | eff %512 | "
        "ideal us@512 | med/ideal | Ktl | Mblk | N_own | k_pad | L1 % | cb0/cb1/cb2/cb3/cb7 |"
    )
    L += [hdr, "|" + "---|" * 19]

    def row(M, K, N, label, e):
        if not e:
            return f"| {M}x{K}x{N} | {b.cdiv(M,32)} | {label} | - | - | - | - | - | - | - | - | - | - | - | - | - |"
        r = e["us_med"] / e["ideal_us_512"]
        cbs = f"{e['cb0_in1']}/{e['cb1_in0']}/{e['cb2_interm']}/{e['cb3_out']}/{e['cb7_reduce']}"
        return (
            f"| {M}x{K}x{N} | {b.cdiv(M,32)} | {label} | {tuple(e['cfg'])} | {e['cores']} | "
            f"{e['us_med']:.1f} | {e['us_min']:.1f} | {e['spread_pct']:.1f} | {e['pct512_med']:.1f} | "
            f"{e['ideal_us_512']:.1f} | {r:.2f}x | {e['Ktl']} | {e['Mblk']} | {e['N_own']} | "
            f"{e['k_pad_tiles']} | {e['l1_pct']:.0f} | {cbs} |"
        )

    for rec in out["primary"]:
        M, K, N = rec["M"], rec["K"], rec["N"]
        L.append(row(M, K, N, "auto", rec["auto"]))
        L.append(row(M, K, N, "best", rec["best_overall"]))
        L.append(row(M, K, N, "best Sm=1", rec["best_sm1"]))
        L.append(row(M, K, N, "best Sm>1", rec["best_smgt1"]))
    L.append("")
    for series, title in (
        ("mscale_smallN", "M-scaling small-N (K=2048,N=1024)"),
        ("mscale_wideN", "M-scaling wide-N (K=6144,N=4608)"),
    ):
        L.append(f"## {title}\n")
        L += ["| shape | Mt | which | cfg | cores | med us | eff %512 | ideal us@512 | med/ideal |", "|" + "---|" * 9]
        for rec in out[series]:
            M, K, N = rec["M"], rec["K"], rec["N"]
            for label, key in (("auto", "auto"), ("best", "best_manual")):
                e = rec[key]
                if e:
                    L.append(
                        f"| {M}x{K}x{N} | {rec['Mt']} | {label} | {tuple(e['cfg'])} | {e['cores']} | "
                        f"{e['us_med']:.1f} | {e['pct512_med']:.1f} | {e['ideal_us_512']:.1f} | "
                        f"{e['us_med']/e['ideal_us_512']:.2f}x |"
                    )
                else:
                    L.append(f"| {M}x{K}x{N} | {rec['Mt']} | {label} | - | - | - | - | - | - |")
        L.append("")
    open(OUTMD, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
