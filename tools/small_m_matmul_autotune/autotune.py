#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""AUTOTUNER for ttnn.experimental.small_m_matmul: rank a shortlist, MEASURE it, cache the winner.

USER-FACING DOCUMENTATION lives at the top of the entry point that wraps this script:
tests/ttnn/unit_tests/operations/matmul/test_small_m_matmul_autotune.py -- how to run it, the env vars, what it
guarantees, why measuring beats a better cost formula, and a worked example. Kept in ONE place so the two
cannot drift; what follows is only this script's own internals.

WHAT THE CACHE IS
    kTable in small_m_matmul_config.cpp. This tool emits ready-to-apply entries; apply_table.py writes them.
    That keeps the runtime picker unchanged (no file I/O, no global state, no first-call measurement inside
    an op) and reuses the cache mechanism that already ships.

RANKING
    Uses the component-wise PHYSICAL model, which beat the shipped cost model as a ranker at every shortlist
    size on held-out shapes (top-4: 3.14% vs 5.02% mean; top-4 worst 13.2% vs 31.7%). It is NOT used as a
    chooser -- as a chooser it LOSES to the shipped picker, which is why this tool measures.

USAGE
    python3 tools/small_m_matmul_autotune/autotune.py 256x2048x1024 512x6144x768 ...
    python3 tools/small_m_matmul_autotune/autotune.py --shapes-file shapes.txt [--topk 8] [--relaunches 2] [--apply]
    Emits, per shape, the measured winner and an apply_table.py argument line (or applies it with --apply).
    Must run from the repo root with TT_METAL_HOME set (the profiler worker needs it).
"""
import argparse, json, os, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from autotune_feas import enumerate_full  # noqa: E402  (exact mirror of the C++ feasibility rules)

WORKER = os.path.join(HERE, "prod_sweep_worker.py")
TILE, CLK = 2048.0, 1350.0
BW_DRAM, BW_CORE, BW_NOC, CYC_MAC = 512.0e3, 21.0e3, 25.0e3, 46.0
LAT_REQ, BLK_OVH, EPS_OVL = 0.3795, 14.45, 0.01671
cd = lambda v: -(-v // 32)


def phys_cost(Mt, Kt, Nt, cfg, g):
    """Component-wise physical cost in microseconds (ranking only)."""
    Pk, Ns, Sm, kb, nsb = cfg
    area, out_t = g["sub_area"], g["out_tiles"]
    t_agg = (Ns * Mt * Kt + Kt * Nt + Mt * Nt) * TILE / BW_DRAM
    in1 = g["Ktl"] * g["Nown"] * TILE
    t_in1 = in1 / BW_CORE + LAT_REQ * (g["Ktl"] * g["Nbpc"] / float(kb))
    t_in0 = (g["Mblk"] * g["Ktl"] * TILE * 0.875 + (Sm - 1) * in1) / BW_NOC
    t_comp = CYC_MAC * (g["Mblk"] * g["Nown"] * g["Ktl"]) * (1.0 + BLK_OVH / (area * kb)) / CLK
    gg = Pk - 1.0 if Pk > 1 else 0.0
    t_red = (gg / Pk * out_t * TILE / BW_NOC) if g["rs"] else (gg * area * TILE / BW_NOC)
    s = t_agg + t_in1 + t_in0 + t_comp
    mx = max(t_agg, t_in1, t_in0, t_comp)
    return mx + EPS_OVL * (s - mx) + t_red


def shipped_cost(Mt, Kt, Nt, cfg, g):
    """The cost model that currently ships (used as a SECOND ranker, not as an authority)."""
    Pk, Ns, Sm, kb, nsb = cfg
    readT = Kt * Nt / min(g["cores"], 24)
    area = min(g["sub_area"], 6)
    kbe = min(kb, 2)
    compT = g["Mblk"] * g["Nown"] * g["Ktl"] / ((kbe / (kbe + 0.5)) * (area / (area + 2.0)))
    ovlT = g["Mblk"] * g["Nown"] * g["Ktl"] / g["Nbpc"]
    return (max(readT, compT) + ovlT) * (1.0 + 0.5 * (g["wasteK"] + g["wasteN"])) + 0.8 * max(0, Pk - 1) * g[
        "Mblk"
    ] * g["Nown"]


def shortlist(Mt, Kt, Nt, k):
    """UNION of the top-k under BOTH rankers.

    The physical model is the better ranker on average (held out, top-4: 3.1% vs 5.0% mean regret) but it has
    a real tail -- measured on three shapes at Kt=192 its top-8 did NOT contain the shipped pick, which was
    4-6% FASTER than anything it shortlisted. Neither ranker dominates, and measuring the union costs only a
    few more runs than measuring one, so take both. The shipped pick is measured separately during
    confirmation, so the effective candidate set is (physical top-k) U (shipped-cost top-k) U {shipped pick}.
    """
    cands = enumerate_full(Mt, Kt, Nt)
    if not cands:
        return []
    out, seen = [], set()
    pools = [
        (phys_cost, cands),
        (shipped_cost, cands),
        # The guarded fallback picks an Sm=1 ANCHOR for non-table shapes; ranking over all Sm buries it
        # under Sm>1 candidates, so rank the Sm==1 subset separately to make sure it is covered.
        (shipped_cost, [cg for cg in cands if cg[0][2] == 1]),
    ]
    for costf, pool in pools:
        if not pool:
            continue
        ranked = sorted(
            pool,
            key=lambda cg: (costf(Mt, Kt, Nt, cg[0], cg[1]), -cg[1]["sub_area"] * cg[0][3], -cg[1]["cores"], cg[0]),
        )
        for cfg, _ in ranked[:k]:
            if cfg not in seen:
                seen.add(cfg)
                out.append(cfg)
    return out


def measure(M, K, N, cfg, env):
    """Median device us for one config, or None. cfg=None measures the shipped pick (config=None)."""
    a = [sys.executable, WORKER, str(M), str(K), str(N), "2"] + ([",".join(map(str, cfg))] if cfg else [])
    try:
        r = subprocess.run(a, capture_output=True, text=True, env=env, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    line = next((x for x in r.stdout.splitlines() if x.startswith("SWEEP_JSON")), None)
    if not line:
        return None
    d = json.loads(line[11:])
    # CORRECTNESS BEFORE TIMING. The worker computes PCC on its first (untimed) call, so a config that is
    # wrong is discarded before its timing is ever considered. Both gates matter and they catch different
    # things: PCC catches "numerics are wrong", the finite count catches a handful of NaN/Inf among millions
    # of elements, which barely moves PCC but is a hard failure (a reduce-scatter CB wrap bug was exactly that).
    if d.get("outcome") != "ok" or d.get("pcc", 0) < 0.999:
        return None
    if d.get("n_nonfinite", 0) != 0 or d.get("finite") is False:
        return None
    return d["median_us"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shapes", nargs="*", help="MxKxN")
    ap.add_argument("--shapes-file")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument(
        "--relaunches", type=int, default=2, help="fresh relaunches used to CONFIRM the winner beats the shipped pick"
    )
    ap.add_argument("--min-gain", type=float, default=1.5, help="percent; below this, keep the shipped pick")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="write the confirmed winners into kTable via apply_table.py, then show the diff. "
        "Tuning stays OFFLINE either way -- this edits a source table that the runtime "
        "picker already consults; it never adds measurement to the operator.",
    )
    a = ap.parse_args()
    shapes = list(a.shapes)
    if a.shapes_file:
        shapes += [l.split()[0] for l in open(a.shapes_file) if l.strip() and not l.startswith("#")]
    if not shapes:
        ap.error("give shapes as MxKxN, or --shapes-file")
    env = dict(os.environ, TT_METAL_DEVICE_PROFILER="1")
    env.pop("TT_SMALL_M_LOG_CFG", None)
    apply_args = []
    for nm in shapes:
        M, K, N = (int(x) for x in nm.lower().split("x"))
        Mt, Kt, Nt = cd(M), cd(K), cd(N)
        sl = shortlist(Mt, Kt, Nt, a.topk)
        if not sl:
            print("[%s] no feasible config -- the small-M matmul cannot serve this shape" % nm, flush=True)
            continue
        # Measure the SHIPPED pick alongside the shortlist so the reported best is the best of everything
        # tried, and so the tool can never propose something slower than what already ships.
        shipped_us = measure(M, K, N, None, env)
        results = []
        for cfg in sl:
            us = measure(M, K, N, cfg, env)
            if us is not None:
                results.append((us, cfg))
        if not results:
            print("[%s] every shortlisted config failed to measure" % nm, flush=True)
            continue
        results.sort()
        best_us, best_cfg = results[0]
        if shipped_us is not None and shipped_us <= best_us:
            print(
                "[%s] %d shortlisted; shipped pick %.2f us already best (shortlist best %s @ %.2f us)"
                % (nm, len(results), shipped_us, ",".join(map(str, best_cfg)), best_us),
                flush=True,
            )
            continue
        # CONFIRM against the shipped pick with fresh relaunches -- a single reading is not enough on this
        # hardware (this gate rejected 6 of 32 apparent wins during the campaign).
        gains = []
        for _ in range(max(1, a.relaunches)):
            base = measure(M, K, N, None, env)
            cand = measure(M, K, N, best_cfg, env)
            if base and cand:
                gains.append(100.0 * (cand - base) / base)
        cs = ",".join(map(str, best_cfg))
        if gains and all(g < -a.min_gain for g in gains):
            verdict = "CONFIRMED %s" % "/".join("%+.1f%%" % g for g in gains)
            apply_args.append('"%s=%s:AUTOTUNE %+.1f%%"' % (nm, cs, statistics.mean(gains)))
        else:
            verdict = "keep shipped pick (%s)" % ("/".join("%+.1f%%" % g for g in gains) if gains else "no baseline")
        print(
            "[%s] shortlist %d measured, best %s @ %.2f us -- %s" % (nm, len(results), cs, best_us, verdict), flush=True
        )
    if not apply_args:
        print("\n# nothing to apply: the shipped pick was within %.1f%% on every shape" % a.min_gain)
        return
    cmd = "python3 tools/small_m_matmul_autotune/apply_table.py \\\n  " + " \\\n  ".join(apply_args)
    if not a.apply:
        print("\n# apply with:\n" + cmd)
        return
    # Strip the shell quoting the printed form carries; argv here needs the bare strings.
    bare = [x.strip('"') for x in apply_args]
    r = subprocess.run([sys.executable, os.path.join(HERE, "apply_table.py")] + bare, capture_output=True, text=True)
    print(r.stdout + r.stderr, end="")
    if r.returncode != 0:
        print("apply_table.py FAILED; kTable not modified")
        return
    tbl = "ttnn/cpp/ttnn/operations/experimental/small_m_matmul/device/small_m_matmul_config.cpp"
    print("\n# kTable patch (review before committing):")
    print(subprocess.run(["git", "diff", "--", tbl], capture_output=True, text=True).stdout, end="")


if __name__ == "__main__":
    main()
