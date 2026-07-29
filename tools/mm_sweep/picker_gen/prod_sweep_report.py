#!/usr/bin/env python3
"""Comprehensive default-configuration sweep report, sorted ascending by effective DRAM bandwidth.

BANDWIDTH ACCOUNTING (why these are the right byte counts):
  in0  Ns * M*K*2   -- each of the Ns n-slice groups reads ALL of in0 once. Within one 8-bank ring the 8 cores
                       read DIFFERENT shards of the same k-slice, so there is no duplication across a ring, and
                       none across Pk (distinct k-slices) or Sm (distinct M rows). Only Ns duplicates.
  in1  K*N*2        -- in1 is DRAM width-sharded; every (kk,nn) group reads its own k-slice x n-band exactly
                       once. Under M-split the reader reads once and forwards over the NoC, so still once.
  out  M*N*2        -- written once; only valid_m x valid_n positions are ever written.
The in0 ring FORWARD traffic is NoC-only and is deliberately excluded: this is a DRAM bandwidth metric.
Padded/zero-fill positions are never read from DRAM (balanced tails), so these effective bytes are also the
bytes physically moved. sched/valid shows how much the schedule's capacity exceeds the logical shape, i.e. the
padding the op carries in compute/L1 even though it does not pay DRAM for it.

usage: prod_sweep_report.py [jsonl] [--md]
"""
import json
import re
import statistics
import sys

S = "/tmp/claude-1211402837/-localdev-cglagovich-tt-metal/0d2ade65-06b4-46ea-a732-5d8b776f32c7/scratchpad"
path = next((a for a in sys.argv[1:] if not a.startswith("--")), f"{S}/prod_sweep.jsonl")
AS_MD = "--md" in sys.argv
PEAK = 512.0  # GB/s, measured BH DRAM ceiling used throughout this campaign

CFG_RE = re.compile(
    r"regime_a_cfg M=(\d+) K=(\d+) N=(\d+) pick=\((\d+),(\d+),(\d+),(\d+),(\d+)\) cores=(\d+) "
    r"reduction=(\S+) placement=(\S+)"
)


def cd(v):
    return -(-v // 32)


rows, bad = [], []
for line in open(path):
    js, _, cfgpart = line.partition("||CFG||")
    i = js.find("{")
    if i < 0:
        continue
    r = json.loads(js[i:])
    M, K, N = r["M"], r["K"], r["N"]
    if r.get("outcome") != "ok":
        bad.append((M, K, N, r.get("outcome"), str(r.get("err"))[:70]))
        continue
    m = CFG_RE.search(cfgpart)
    if not m:
        bad.append((M, K, N, "no-cfg", "config log line missing"))
        continue
    g = m.groups()
    Pk, Ns, Sm, kb, nsb = (int(x) for x in g[3:8])
    cores, red, place = int(g[8]), g[9], g[10]
    wall = r["median_us"]

    eff = (Ns * M * K * 2) + (K * N * 2) + (M * N * 2)
    eff_gbps = eff / (wall * 1e-6) / 1e9

    Mt, Kt, Nt = cd(M), cd(K), cd(N)
    K_slice = -(-(-(-Kt // Pk)) // (kb * 8)) * (kb * 8)
    M_block = -(-Mt // Sm)
    N_band = -(-Nt // 8)
    N_own = -(-N_band // Ns)
    N_sub = nsb if nsb else N_own
    N_bpc = -(-N_own // N_sub)
    sMt, sKt, sNt = M_block * Sm, K_slice * Pk, N_sub * N_bpc * Ns * 8
    sched = (Ns * sMt * sKt + sKt * sNt + sMt * sNt) * 32 * 32 * 2
    bm = r.get("block_medians", [])
    spread = 100.0 * (max(bm) - min(bm)) / statistics.median(bm) if len(bm) > 1 else 0.0
    iter_spread = 100.0 * (r["max_us"] - r["min_us"]) / wall

    rows.append(
        dict(
            name="{}x{}x{}".format(M, K, N),
            Mt=Mt,
            wall=wall,
            eff=eff_gbps,
            pct=100 * eff_gbps / PEAK,
            cfg="{},{},{},{},{}".format(Pk, Ns, Sm, kb, nsb),
            cores=cores,
            red=red,
            place=place,
            pcc=r.get("pcc", float("nan")),
            spread=spread,
            ispread=iter_spread,
            sov=sched / eff,
            ok=r.get("pcc", 0) >= 0.999,
        )
    )

rows.sort(key=lambda z: z["eff"])

hdr = [
    "shape",
    "Mt",
    "Pk,Ns,Sm,kb,nsb",
    "core",
    "reduction",
    "placement",
    "dev us",
    "eff GB/s",
    "%pk",
    "sch/val",
    "PCC",
    "blk%",
    "it%",
]
if AS_MD:
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        print(
            "| {name} | {Mt} | {cfg} | {cores} | {red} | {place} | {wall:.2f} | {eff:.1f} | {pct:.0f}% | "
            "{sov:.2f} | {pcc:.5f} | {spread:.1f} | {ispread:.1f} |".format(**r)
        )
else:
    print("{:16s} {:>3s} {:>16s} {:>5s} {:>15s} {:>11s} {:>8s} {:>9s} {:>5s} {:>8s} {:>9s} {:>6s} {:>6s}".format(*hdr))
    print("-" * 140)
    for r in rows:
        print(
            "{name:16s} {Mt:3d} {cfg:>16s} {cores:5d} {red:>15s} {place:>11s} {wall:8.2f} {eff:9.1f} "
            "{pct:4.0f}% {sov:8.2f} {pcc:9.5f} {spread:6.1f} {ispread:6.1f}".format(**r)
        )
    print("-" * 140)

n = len(rows)
if n:
    print("\n{} shapes measured, all at DEFAULTS (config=None, no diagnostic mask).".format(n))
    print(
        "effective DRAM BW: min {:.1f}  median {:.1f}  max {:.1f} GB/s   (peak {:.0f})".format(
            rows[0]["eff"], statistics.median([r["eff"] for r in rows]), rows[-1]["eff"], PEAK
        )
    )
    print(
        "%peak: median {:.0f}%   under 50%: {}   over 80%: {}".format(
            statistics.median([r["pct"] for r in rows]),
            sum(1 for r in rows if r["pct"] < 50),
            sum(1 for r in rows if r["pct"] > 80),
        )
    )
    print(
        "correctness: {}/{} PCC >= 0.999   (min PCC {:.5f})".format(
            sum(1 for r in rows if r["ok"]), n, min(r["pcc"] for r in rows)
        )
    )
    print(
        "stability: median block-to-block spread {:.1f}%, worst {:.1f}% ({}); "
        "median iteration spread {:.1f}%".format(
            statistics.median([r["spread"] for r in rows]),
            max(r["spread"] for r in rows),
            max(rows, key=lambda z: z["spread"])["name"],
            statistics.median([r["ispread"] for r in rows]),
        )
    )
    rs = sum(1 for r in rows if r["red"] == "reduce-scatter")
    print(
        "reduction: {} reduce-scatter / {} chain     placement: {} mesh / {} in1-near / {} bank-local".format(
            rs,
            n - rs,
            sum(1 for r in rows if r["place"] == "mesh"),
            sum(1 for r in rows if r["place"] == "in1-near"),
            sum(1 for r in rows if r["place"] == "bank-local"),
        )
    )
    pad = [r for r in rows if r["sov"] > 1.15]
    if pad:
        print(
            "schedule padding > 1.15x on {}: {}".format(
                len(pad), ", ".join("{} ({:.2f})".format(r["name"], r["sov"]) for r in pad[:8])
            )
        )
if bad:
    print("\nFAILED / not measured ({}):".format(len(bad)))
    for b in bad:
        print("  {}x{}x{}  {}  {}".format(*b))
