#!/usr/bin/env python3
# Focused confirmation of the in0-ring C=1 chunk win: 2 independent runs x 10 interleaved relaunches, CW vs
# C1 (+C2 on the winners), keyed by structural params (W, N_sub, N_bpc, Mblk). Decides whether the win is a
# robust structural effect or shape-specific/noise. NO production change.
import json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import regime_a_bench as rb
import regime_a_diag_suite as ds

C1, C2, C4 = 1 << 28, 1 << 27, 1 << 26
# (group, M, K, N). winners(profile) + profile-but-neutral discriminator + wide-Nsub + W1 controls.
SHAPES = [
    ("winner", 256, 15360, 768),
    ("winner", 128, 15360, 768),
    ("winner", 256, 2304, 6144),
    ("profile_neutral", 128, 15360, 1536),  # matches W>=3/nsb3/Mblk4 but was neutral -> the discriminator
    ("ctl_wideNsub", 256, 15360, 1536),  # nsb6
    ("ctl_W1", 256, 2048, 512),
    ("ctl_W1", 256, 6144, 4608),
]


def struct(M, K, N):
    Ns, Pk, Sm, kb, nsb = rb.auto_config(M, K, N)
    Kt, Nt = rb.cdiv(K, 32), rb.cdiv(N, 32)
    Ktl = rb.rup(rb.cdiv(Kt, Pk), kb * 8)
    W = Ktl // (8 * kb)
    Nband = rb.cdiv(Nt, 8)
    Nown = rb.cdiv(Nband, Ns)
    Nbpc = rb.cdiv(Nown, nsb)
    return dict(cfg=(Ns, Pk, Sm, kb, nsb), W=W, Nsub=nsb, Nbpc=Nbpc, Mblk=rb.cdiv(rb.cdiv(M, 32), Sm), kb=kb)


def run(M, K, N, cfg, masks, relaunches):
    walls = {m: [] for m in masks}
    for _ in range(relaunches):
        for m in masks:  # interleaved
            x = ds.run_one(M, K, N, cfg, m)
            if x.get("ok") and x.get("wall_us"):
                walls[m].append(x["wall_us"])
    return walls


if __name__ == "__main__":
    RELAUNCH = 10
    out = []
    for grp, M, K, N in SHAPES:
        s = struct(M, K, N)
        cfg = s["cfg"]
        masks = [0, C1] + ([C2] if grp == "winner" else [])
        run_res = []
        for r in range(2):  # two independent runs
            walls = run(M, K, N, cfg, masks, RELAUNCH)
            med = {m: (statistics.median(v) if v else None) for m, v in walls.items()}
            spr = {m: ((max(v) - min(v)) / min(v) * 100 if v else None) for m, v in walls.items()}
            d1 = ((med[C1] / med[0] - 1) * 100) if (med.get(0) and med.get(C1)) else None
            d2 = ((med[C2] / med[0] - 1) * 100) if (C2 in med and med.get(0) and med.get(C2)) else None
            run_res.append(
                {
                    "cw": med[0],
                    "c1": med.get(C1),
                    "c1_delta": d1,
                    "c2_delta": d2,
                    "cw_spread": spr[0],
                    "c1_spread": spr.get(C1),
                    "cw_walls": sorted(walls[0]),
                    "c1_walls": sorted(walls[C1]),
                }
            )
        rec = {"group": grp, "M": M, "K": K, "N": N, **s, "runs": run_res}
        out.append(rec)
        json.dump(out, open("confirm_chunk.json", "w"), indent=2)
        d = [rr["c1_delta"] for rr in run_res]
        c2 = [rr["c2_delta"] for rr in run_res if rr["c2_delta"] is not None]
        print(
            f"[confirm/{grp}] {M}x{K}x{N} W{s['W']} nsub{s['Nsub']} nbpc{s['Nbpc']} Mblk{s['Mblk']}: "
            f"C1 run1={d[0] and round(d[0],1)}% run2={d[1] and round(d[1],1)}% "
            f"(cw {run_res[0]['cw'] and round(run_res[0]['cw'],1)}/{run_res[1]['cw'] and round(run_res[1]['cw'],1)}, "
            f"spread {run_res[0]['cw_spread'] and round(run_res[0]['cw_spread'],1)}%)"
            + (f" C2={[round(x,1) for x in c2]}%" if c2 else ""),
            flush=True,
        )
    print("CONFIRM DONE", flush=True)
