#!/usr/bin/env python3
# FLUX/LTX real-model shape comparison: the regime_a_matmul op (LATEST auto-picker, config=None) vs the
# historical minimal_matmul "main baseline" (base) and "branch" (production auto), from bh_skinny_results.json.
#
# Model labels + the regime-A real-model set are from SMALL_MT_IMPL_PLAN.md (FLUX.2 1024px SP4/TP8, LTX).
# BW conventions kept SEPARATE (never mixed): op = % of 512 GB/s; base/branch (minimal_matmul) = % of
# 500 GB/s. The ONLY cross-source comparison is kernel us (convention-independent) -> speedup columns.
import json, os

HERE = os.path.dirname(__file__)
corpus = json.load(open(f"{HERE}/regime_a_bench.json"))["corpus"]
skinny = {(s["M"], s["K"], s["N"]): s for s in json.load(open(f"{HERE}/bh_skinny_results.json"))["shapes"]}
op = {(r["M"], r["K"], r["N"]): r for r in corpus}

# Real-model regime-A FLUX/LTX shapes (SMALL_MT_IMPL_PLAN.md REAL-MODEL SHAPE CLASSIFICATION).
FLUXLTX = [
    ("FLUX", 32, 256, 6144),
    ("FLUX", 32, 6144, 1536),
    ("FLUX", 32, 6144, 2304),
    ("FLUX", 32, 6144, 4608),
    ("FLUX", 32, 6144, 6144),
    ("FLUX", 512, 6144, 768),
    ("FLUX", 512, 15360, 768),
    ("FLUX", 512, 6144, 2304),
    ("FLUX", 512, 2304, 6144),
    ("FLUX", 512, 3072, 6144),
    ("FLUX", 512, 6144, 4608),
    ("LTX", 32, 2048, 512),
    ("LTX", 32, 2048, 1536),
    ("LTX", 32, 2048, 2048),
    ("LTX", 256, 2048, 1024),
]


def okp(r):
    p = r.get("product") if r else None
    return p if p and p.get("cls") == "ok" else None


def f(x, fmt="{:.1f}"):
    return fmt.format(x) if isinstance(x, (int, float)) else "-"


lines = []
lines.append("# FLUX / LTX real-model shapes — regime_a_matmul (latest picker) vs main & branch\n")
lines.append(
    "Op = `ttnn.experimental.regime_a_matmul` **auto-picker (config=None)**, latest table "
    "(commits up to ab1acb7c871). `main` and `branch` are the historical minimal_matmul numbers from "
    "`bh_skinny_results.json`: **main** = the *main optimized baseline* (plain unicast, best-swept blocks, "
    "all branch levers gated + TT_MM_NO_LARGE_LEVERS=1 — verified == main bit-for-bit on the dataflow "
    "path); **branch** = the minimal_matmul production auto path.\n"
)
lines.append(
    "> BW conventions are NOT mixed: **op % is of 512 GB/s**, **main/branch % is of 500 GB/s**. Compare "
    "across sources by **kernel µs only** (the speedup columns). Model labels + the regime-A set are from "
    "SMALL_MT_IMPL_PLAN.md.\n"
)
hdr = (
    "| model | shape | Mt | op cfg (Ns,Pk,Sm,kb,nsb) | op us | op %512 | branch us | branch %500 | "
    "main us | main %500 | op vs branch | op vs main |"
)
lines += [hdr, "|" + "---|" * 12]

rows = []
for model, M, K, N in FLUXLTX:
    r = op.get((M, K, N))
    p = okp(r)
    Mt = r["Mt"] if r else (M + 31) // 32
    ou = p.get("us_med") if p else None
    opc = p.get("pct512") if p else None
    cfg = p.get("cfg") if p else None
    sk = skinny.get((M, K, N))
    bu = sk["branch"]["us"] if sk else None
    bp = sk["branch"]["bw_util"] if sk else None
    mu = sk["base"]["us"] if sk else None
    mp = sk["base"]["bw_util"] if sk else None
    vb = (f"{(bu/ou-1)*100:+.0f}%") if (ou and bu) else "-"
    vm = (f"{(mu/ou-1)*100:+.0f}%") if (ou and mu) else "-"
    rows.append((model, M, K, N, Mt, cfg, ou, opc, bu, bp, mu, mp, vb, vm))
    lines.append(
        f"| {model} | {M}x{K}x{N} | {Mt} | {cfg} | {f(ou)} | {f(opc)} | {f(bu)} | {f(bp,'{:.0f}')} | "
        f"{f(mu)} | {f(mp,'{:.0f}')} | {vb} | {vm} |"
    )

# aggregate speedups on shapes that HAVE historical numbers
import math


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


vb_r = [bu / ou for (_, _, _, _, _, _, ou, _, bu, _, _, _, _, _) in rows if ou and bu]
vm_r = [mu / ou for (_, _, _, _, _, _, ou, _, _, _, mu, _, _, _) in rows if ou and mu]
lines.append("")
lines.append(
    f"**Aggregate (shapes with historical numbers, by µs):** op is **{geomean(vb_r):.2f}x** vs branch, "
    f"**{geomean(vm_r):.2f}x** vs main, over the {len(vb_r)} Mt=1 FLUX/LTX shapes.\n"
)
lines.append(
    "**No historical main/branch exist for the large-Mt set** (FLUX M=512 Mt16 ×6, LTX 256x2048x1024 "
    "Mt8): these were the OOM/GAP shapes not in the minimal_matmul bh_skinny sweep, so main/branch are '-'. "
    "The op RUNS all of them correctly; 256x2048x1024 is at 37% (structural ceiling, see the main report), "
    "and the Mt16 FLUX set is diagnostic-only (out of the Mt<=8 acceptance scope) at 33-59%.\n"
)

out = f"{HERE}/FLUXLTX_COMPARE.md"
open(out, "w").write("\n".join(lines) + "\n")
print("WROTE", out)
print(f"op vs branch geomean {geomean(vb_r):.2f}x ({len(vb_r)} shapes); op vs main {geomean(vm_r):.2f}x")
