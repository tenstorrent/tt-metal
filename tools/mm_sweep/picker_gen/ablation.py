#!/usr/bin/env python3
"""in0-read ablation supervisor (baseline / skip-redundant / skip-all) + report.

3 fresh persistent-session relaunches per shape, reversing mode block order between relaunches; resumable
+ hang-safe (per-relaunch atomic checkpoint; tt-smi -r on hang). Preserves all raw samples. Computes paired
median deltas, IQR/spread, per-RISC spans, %improvement, the theoretical DRAM time of all/redundant in0
reads at 512 GB/s, and an exposure classification (hidden / partial / full). Skip deltas are UPPER BOUNDS
that EXCLUDE the future cross-Ns NoC-copy cost.

Usage: python3 ablation.py run     (measure, resumable)
       python3 ablation.py report  (regenerate report + raw JSON from checkpoint)
"""
import argparse, json, os, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("TT_METAL_HOME", os.path.abspath(f"{HERE}/../.."))
WORKER = f"{HERE}/ablation_worker.py"
CKPT = f"{HERE}/results_v2/ablation.jsonl"
RAW = f"{HERE}/ablation_raw.json"
MD = f"{HERE}/ABLATION_IN0_READ.md"
TILE = 32
PEAK_GBS = 512.0
RELAUNCHES = 3
SHAPES = [  # (M,K,N): (Ns,Pk,Sm,kb,nsb)
    ((256, 2048, 2048), (2, 2, 3, 4, 4)),
    ((256, 2048, 6144), (3, 2, 2, 2, 4)),
    ((512, 6144, 2304), (2, 6, 1, 2, 1)),
    ((512, 6144, 4608), (2, 6, 1, 4, 1)),
]
os.makedirs(os.path.dirname(CKPT), exist_ok=True)


def key(M, K, N):
    return f"{M}x{K}x{N}"


def load():
    out = {}
    if os.path.exists(CKPT):
        for line in open(CKPT):
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["shape"], []).append(r)
    return out


def device_reset():
    subprocess.run(["pkill", "-9", "-f", "ablation_worker"], capture_output=True)
    time.sleep(2)
    subprocess.run(["tt-smi", "-r"], capture_output=True, timeout=180)
    time.sleep(10)


def relaunch(M, K, N, cfg, order, verify, timeout=400):
    Ns, Pk, Sm, kb, nsb = cfg
    env = dict(os.environ)
    env.update(TT_METAL_DEVICE_PROFILER="1", TT_METAL_HOME=ROOT, ARCH_NAME="blackhole")
    args = [
        sys.executable,
        WORKER,
        str(M),
        str(K),
        str(N),
        str(Ns),
        str(Pk),
        str(Sm),
        str(kb),
        str(nsb),
        ",".join(order),
        str(verify),
    ]
    try:
        r = subprocess.run(args, env=env, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"outcome": "hang", "err": "timeout"}
    line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
    return json.loads(line) if line else {"outcome": "runtime", "err": (r.stderr or r.stdout)[-300:]}


def run(args):
    base = ["baseline", "skip_redundant", "skip_all"]
    for (M, K, N), cfg in SHAPES:
        sh = key(M, K, N)
        have = [r for r in load().get(sh, []) if r.get("outcome") == "ok"]
        if len(have) >= RELAUNCHES:
            print(f"{sh}: {len(have)} relaunches (complete)", flush=True)
            continue
        print(f"{sh} {cfg}: {len(have)}/{RELAUNCHES} relaunches", flush=True)
        attempts = 0
        while (
            len([r for r in load().get(sh, []) if r.get("outcome") == "ok"]) < RELAUNCHES and attempts < RELAUNCHES * 3
        ):
            idx = len([r for r in load().get(sh, []) if r.get("outcome") == "ok"])
            order = base if idx % 2 == 0 else list(reversed(base))  # reverse mode order on alternate relaunches
            res = relaunch(M, K, N, cfg, order, verify=1 if idx == 0 else 0)
            res.update(shape=sh, cfg=list(cfg), relaunch_idx=idx, attempt=attempts)
            with open(CKPT, "a") as f:
                f.write(json.dumps(res) + "\n")
                f.flush()
                os.fsync(f.fileno())
            attempts += 1
            if res["outcome"] == "hang":
                print(f"  {sh}: HANG; reset+retry", flush=True)
                device_reset()
            elif res["outcome"] != "ok":
                print(f"  {sh}: {res['outcome']}: {res.get('err','')[:160]}", flush=True)
            else:
                md = {m: res["modes"][m]["median_us"] for m in res["modes"]}
                print(f"  {sh}: relaunch {idx} order={order[0]}.. medians={md}", flush=True)
    print("run complete", flush=True)
    report(args)


def report(args):
    ck = load()
    rows = []
    for (M, K, N), cfg in SHAPES:
        sh = key(M, K, N)
        oks = [r for r in ck.get(sh, []) if r.get("outcome") == "ok"]
        if not oks:
            continue
        Ns, Pk, Sm, kb, nsb = cfg
        agg = {}
        for m in ("baseline", "skip_redundant", "skip_all"):
            samples = [s for r in oks if m in r.get("modes", {}) for s in r["modes"][m]["samples_us"]]
            if not samples:
                continue
            risc_meds = {}
            for rn in ("BRISC", "NCRISC", "TRISC"):
                xs = [
                    r["modes"][m]["risc_spans_us"][rn]
                    for r in oks
                    if m in r.get("modes", {}) and r["modes"][m]["risc_spans_us"].get(rn) is not None
                ]
                risc_meds[rn] = round(statistics.median(xs), 2) if xs else None
            q = statistics.quantiles(samples, n=4) if len(samples) >= 4 else [samples[0]] * 3
            agg[m] = {
                "median": round(statistics.median(samples), 3),
                "n": len(samples),
                "iqr": round(q[2] - q[0], 3),
                "spread_pct": round((max(samples) - min(samples)) / min(samples) * 100, 2),
                "risc": risc_meds,
                "samples": [round(s, 3) for s in samples],
            }
        b = agg["baseline"]["median"]
        d_red = b - agg["skip_redundant"]["median"]
        d_all = b - agg["skip_all"]["median"]
        # theoretical DRAM time at 512 GB/s (bytes = 2 * elements)
        MK = M * K
        t_all_in0_us = (2 * Ns * MK) / (PEAK_GBS * 1e9) * 1e6  # all Ns groups' in0 reads
        t_redundant_us = (2 * (Ns - 1) * MK) / (PEAK_GBS * 1e9) * 1e6  # redundant duplicate reads

        def classify(delta, theo):
            if theo <= 0:
                return "n/a"
            f = delta / theo
            if f < 0.15:
                return "hidden"
            if f > 0.85:
                return "fully-exposed"
            return "partially-exposed"

        rows.append(
            {
                "shape": sh,
                "cfg": list(cfg),
                "Ns": Ns,
                "baseline_pcc": next((r.get("baseline_pcc") for r in oks if r.get("baseline_pcc") is not None), None),
                "cached_replay_pcc": next(
                    (r.get("cached_replay_pcc") for r in oks if r.get("cached_replay_pcc") is not None), None
                ),
                "cached_replay_matches": next(
                    (r.get("cached_replay_matches") for r in oks if "cached_replay_matches" in r), None
                ),
                "n_relaunch": len(oks),
                "modes": agg,
                "baseline_us": b,
                "skip_redundant_us": agg["skip_redundant"]["median"],
                "skip_all_us": agg["skip_all"]["median"],
                "delta_redundant_us": round(d_red, 3),
                "delta_all_us": round(d_all, 3),
                "pct_redundant": round(d_red / b * 100, 2),
                "pct_all": round(d_all / b * 100, 2),
                "theo_all_in0_us": round(t_all_in0_us, 2),
                "theo_redundant_us": round(t_redundant_us, 2),
                "exposure_all": classify(d_all, t_all_in0_us),
                "exposure_redundant": classify(d_red, t_redundant_us),
                "exposed_frac_all": round(d_all / t_all_in0_us, 3) if t_all_in0_us else None,
                "exposed_frac_redundant": round(d_red / t_redundant_us, 3) if t_redundant_us else None,
            }
        )
    json.dump({"rows": rows}, open(RAW, "w"), indent=2)
    write_md(rows)
    for r in rows:
        print(
            f"{r['shape']:16s} base={r['baseline_us']:.1f} skipR={r['skip_redundant_us']:.1f}"
            f"({r['pct_redundant']:+.1f}%) skipA={r['skip_all_us']:.1f}({r['pct_all']:+.1f}%) "
            f"theoR={r['theo_redundant_us']:.1f} theoA={r['theo_all_in0_us']:.1f} "
            f"exp_R={r['exposure_redundant']} exp_A={r['exposure_all']}",
            flush=True,
        )
    print(f"wrote {RAW} and {MD}", flush=True)


def write_md(rows):
    with open(MD, "w") as f:
        f.write("# in0-read ablation (baseline / skip-redundant / skip-all)\n\n")
        f.write(
            "Test-only compile-gated diagnostic (`diag_in0_read_mask`, TT_REGIME_A_DIAG_IN0 env; reflection "
            "program-cache-hashed; public API + mask-0 binaries unchanged). Skipped reads preserve CB "
            "reserve/push/pop, in0 ring forwarding, semaphores, compute, reduction, output writes exactly "
            "(no zero-fill, no removed downstream) — outputs for masks 1/2 are intentionally invalid; PCC "
            "asserted only for mask 0.\n\n"
        )
        f.write(
            "One persistent device session per relaunch; 2 warmup + 16 timed resident-input iters/mode; "
            "3 relaunches/shape with mode block order reversed on alternate relaunches; kernel wall + "
            "per-RISC spans from the device profiler (run-host-id demux). Tuples `(Ns,Pk,Sm,kb,nsb)`. "
            "Commit `ce79cca7f79` + this diagnostic; BH p150b, 1.35 GHz, fw 19.5.0, KMD 2.4.1; peak "
            "DRAM ref 512 GB/s.\n\n"
        )
        f.write(
            "> **Skip deltas are UPPER BOUNDS** on any cross-Ns in0-sharing win: they remove the DRAM reads "
            "entirely and **exclude the NoC-copy cost** that real sharing (one Ns group reads, then "
            "distributes to the other rings) would add.\n\n"
        )
        f.write("## Summary\n\n")
        cols = [
            "shape",
            "cfg",
            "baseline us",
            "skip-redundant us (Δ, %)",
            "skip-all us (Δ, %)",
            "per-RISC B/N/T (base→skipAll)",
            "theo redundant / all-in0 DRAM us",
            "exposed frac R / A",
            "exposure R / A",
            "baseline PCC (replay)",
        ]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            bm, sr, sa = r["modes"]["baseline"], r["modes"]["skip_redundant"], r["modes"]["skip_all"]
            risc = (
                f"{bm['risc']['BRISC']}/{bm['risc']['NCRISC']}/{bm['risc']['TRISC']} → "
                f"{sa['risc']['BRISC']}/{sa['risc']['NCRISC']}/{sa['risc']['TRISC']}"
            )
            f.write(
                "| "
                + " | ".join(
                    [
                        r["shape"],
                        str(tuple(r["cfg"])),
                        f"{r['baseline_us']:.2f}",
                        f"{r['skip_redundant_us']:.2f} ({r['delta_redundant_us']:+.2f}, {r['pct_redundant']:+.1f}%)",
                        f"{r['skip_all_us']:.2f} ({r['delta_all_us']:+.2f}, {r['pct_all']:+.1f}%)",
                        risc,
                        f"{r['theo_redundant_us']:.1f} / {r['theo_all_in0_us']:.1f}",
                        f"{r['exposed_frac_redundant']} / {r['exposed_frac_all']}",
                        f"{r['exposure_redundant']} / {r['exposure_all']}",
                        f"{r['baseline_pcc']:.5f} ({r['cached_replay_pcc']:.5f})",
                    ]
                )
                + " |\n"
            )
        f.write("\n## Raw spread\n\n| shape | mode | median us | IQR us | spread% | n |\n|---|---|---|---|---|---|\n")
        for r in rows:
            for m in ("baseline", "skip_redundant", "skip_all"):
                a = r["modes"][m]
                f.write(
                    f"| {r['shape']} | {m} | {a['median']:.2f} | {a['iqr']:.2f} | {a['spread_pct']:.1f} | {a['n']} |\n"
                )
        f.write("\n## Interpretation\n\n")
        f.write(
            "- **exposed fraction** = measured skip delta ÷ theoretical DRAM time of the removed reads "
            "(all-in0 for skip-all, redundant duplicates for skip-redundant). ~0 ⇒ the reads are hidden "
            "behind compute/in1/ring (removing them frees no wall time); ~1 ⇒ fully exposed on the "
            "critical path.\n"
        )
        for r in rows:
            f.write(
                f"- **{r['shape']}**: skip-redundant {r['pct_redundant']:+.1f}% "
                f"(exposed {r['exposed_frac_redundant']} ⇒ {r['exposure_redundant']}); skip-all "
                f"{r['pct_all']:+.1f}% (exposed {r['exposed_frac_all']} ⇒ {r['exposure_all']}). The "
                f"cross-Ns dedup opportunity here removes {r['theo_redundant_us']:.1f} us of redundant "
                f"DRAM traffic; realizable upside is **at most** the skip-redundant delta "
                f"({r['delta_redundant_us']:+.2f} us) minus NoC-copy cost.\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "report"])
    args = ap.parse_args()
    {"run": run, "report": report}[args.phase](args)


if __name__ == "__main__":
    main()
