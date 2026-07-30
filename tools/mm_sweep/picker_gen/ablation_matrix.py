#!/usr/bin/env python3
"""Critical-path ablation MATRIX supervisor + report (baseline + 6 singles + 15 pairs = 22 modes).

Per shape: >=2 fresh persistent-session relaunches (forward mode order on even relaunches, reverse on odd),
each measuring all 22 modes (2 warmup + 12 timed iters/mode). A 3rd relaunch is added only when a delta is
near noise or relaunch distributions overlap (see trigger below). Resumable + hang-safe (per-relaunch
atomic checkpoint; tt-smi -r on hang). Preserves every raw sample + per-RISC span.

Analysis: gain(S)=Tbase-T(S); interaction(A,B)=gain(A|B)-gain(A)-gain(B). Emits a 6x6 interaction matrix,
fastest combos, critical-RISC transitions, and evidence-backed per-stage exposure. Single-ablation
percentages are NOT summed as a forecast.

Usage: python3 ablation_matrix.py run | report
"""
import argparse, itertools, json, os, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("TT_METAL_HOME", os.path.abspath(f"{HERE}/../.."))
WORKER = f"{HERE}/ablation_matrix_worker.py"
CKPT = f"{HERE}/results_v2/ablation_matrix.jsonl"
RAW = f"{HERE}/ablation_matrix_raw.json"
MD = f"{HERE}/ABLATION_MATRIX.md"
TILE = 32
PEAK_GBS = 512.0

BITS = [1, 2, 4, 8, 16, 32]
BITNAME = {1: "in0_all", 2: "in0_redun", 4: "ring_fwd", 8: "compute", 16: "reduction", 32: "output"}
SINGLES = list(BITS)
PAIRS = [a | b for a, b in itertools.combinations(BITS, 2)]  # 15 pairs (includes 1|2=3 identity)
MODES = [0] + SINGLES + PAIRS  # 22
SHAPES = [  # (M,K,N): (Ns,Pk,Sm,kb,nsb)
    ((256, 2048, 2048), (2, 2, 3, 4, 4)),
    ((256, 2048, 6144), (3, 2, 2, 2, 4)),
    ((512, 6144, 2304), (2, 6, 1, 2, 1)),
    ((512, 6144, 4608), (2, 6, 1, 4, 1)),
]
os.makedirs(os.path.dirname(CKPT), exist_ok=True)


def key(M, K, N):
    return f"{M}x{K}x{N}"


def maskname(m):
    if m == 0:
        return "baseline"
    return "+".join(BITNAME[b] for b in BITS if m & b)


def load():
    out = {}
    if os.path.exists(CKPT):
        for line in open(CKPT):
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["shape"], []).append(r)
    return out


def device_reset():
    subprocess.run(["pkill", "-9", "-f", "ablation_matrix_worker"], capture_output=True)
    time.sleep(2)
    subprocess.run(["tt-smi", "-r"], capture_output=True, timeout=180)
    time.sleep(10)


def relaunch(M, K, N, cfg, masks, verify, timeout=600):
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
        ",".join(str(m) for m in masks),
        str(verify),
    ]
    try:
        r = subprocess.run(args, env=env, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"outcome": "hang", "err": "timeout"}
    line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
    return json.loads(line) if line else {"outcome": "runtime", "err": (r.stderr or r.stdout)[-300:]}


def relaunch_medians(oks, m):
    """per-relaunch median list for mask m."""
    out = []
    for r in oks:
        md = r.get("modes", {}).get(str(m))
        if md:
            out.append(md["median_us"])
    return out


def needs_third(oks):
    """Trigger a 3rd relaunch if any mode's cross-relaunch median spread > 2% (distributions overlap) OR any
    single-ablation |gain| < 2% of baseline (near noise). Covers the 'delta below 2% / distributions overlap
    / interaction ambiguous' criteria (ambiguous interactions arise when component gains are near noise)."""
    if len(oks) < 2:
        return True
    base = [relaunch_medians(oks, 0)]
    if not base or not base[0]:
        return True
    tbase = statistics.median(base[0])
    for m in MODES:
        rms = relaunch_medians(oks, m)
        if len(rms) >= 2 and min(rms) > 0 and (max(rms) - min(rms)) / min(rms) * 100 > 2.0:
            return True
    for m in SINGLES:
        rms = relaunch_medians(oks, m)
        if rms:
            gain_pct = (tbase - statistics.median(rms)) / tbase * 100
            if abs(gain_pct) < 2.0:
                return True
    return False


def run(args):
    for (M, K, N), cfg in SHAPES:
        sh = key(M, K, N)
        oks = [r for r in load().get(sh, []) if r.get("outcome") == "ok"]
        # target: 2 relaunches, then a conditional 3rd
        while True:
            oks = [r for r in load().get(sh, []) if r.get("outcome") == "ok"]
            idx = len(oks)
            if idx >= 3:
                break
            if idx >= 2 and not needs_third(oks):
                break
            masks = MODES if idx % 2 == 0 else list(reversed(MODES))  # forward even, reverse odd
            print(f"{sh} {cfg}: relaunch {idx} ({'fwd' if idx % 2 == 0 else 'rev'}, {len(masks)} modes)", flush=True)
            res = relaunch(M, K, N, cfg, masks, verify=1 if idx == 0 else 0)
            res.update(shape=sh, cfg=list(cfg), relaunch_idx=idx)
            with open(CKPT, "a") as f:
                f.write(json.dumps(res) + "\n")
                f.flush()
                os.fsync(f.fileno())
            if res["outcome"] == "hang":
                print(f"  {sh}: HANG; reset+retry", flush=True)
                device_reset()
            elif res["outcome"] != "ok":
                print(f"  {sh}: {res['outcome']}: {res.get('err','')[:160]}", flush=True)
                break  # avoid infinite loop on persistent error
            else:
                print(f"  {sh}: relaunch {idx} ok (baseline {res['modes']['0']['median_us']}us)", flush=True)
    print("run complete", flush=True)
    report(args)


def report(args):
    ck = load()
    shapes_out = []
    for (M, K, N), cfg in SHAPES:
        sh = key(M, K, N)
        oks = [r for r in ck.get(sh, []) if r.get("outcome") == "ok"]
        if not oks:
            continue
        Ns, Pk, Sm, kb, nsb = cfg
        # aggregate samples per mask across relaunches
        agg = {}
        for m in MODES:
            samples = [s for r in oks if str(m) in r.get("modes", {}) for s in r["modes"][str(m)]["samples_us"]]
            if not samples:
                continue
            risc = {}
            for rn in ("BRISC", "NCRISC", "TRISC"):
                xs = [
                    r["modes"][str(m)]["risc_spans_us"][rn]
                    for r in oks
                    if str(m) in r.get("modes", {}) and r["modes"][str(m)]["risc_spans_us"].get(rn) is not None
                ]
                risc[rn] = round(statistics.median(xs), 2) if xs else None
            q = statistics.quantiles(samples, n=4) if len(samples) >= 4 else [samples[0]] * 3
            crit = max(risc, key=lambda k: (risc[k] if risc[k] is not None else -1))
            agg[m] = {
                "median": round(statistics.median(samples), 3),
                "n": len(samples),
                "iqr": round(q[2] - q[0], 3),
                "spread_pct": round((max(samples) - min(samples)) / min(samples) * 100, 2),
                "risc": risc,
                "crit_risc": crit,
                "relaunch_medians": [round(x, 3) for x in relaunch_medians(oks, m)],
            }
        tbase = agg[0]["median"]
        gain = {m: round(tbase - agg[m]["median"], 3) for m in agg}
        # pair-interaction matrix
        inter = {}
        for a, b in itertools.combinations(BITS, 2):
            ab = a | b
            if a in gain and b in gain and ab in gain:
                inter[(a, b)] = round(gain[ab] - gain[a] - gain[b], 3)
        # theoretical DRAM times (bytes = 2*elements)
        MK, KN, MN = M * K, K * N, M * N
        theo = {
            "in0_all": round(2 * Ns * MK / (PEAK_GBS * 1e9) * 1e6, 2),
            "in0_redun": round(2 * (Ns - 1) * MK / (PEAK_GBS * 1e9) * 1e6, 2),
            "output": round(2 * MN / (PEAK_GBS * 1e9) * 1e6, 2),
            "in1": round(2 * KN / (PEAK_GBS * 1e9) * 1e6, 2),
        }
        fastest = sorted(agg, key=lambda m: agg[m]["median"])[:6]
        shapes_out.append(
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
                "tbase": tbase,
                "agg": agg,
                "gain": gain,
                "inter": inter,
                "theo": theo,
                "fastest": fastest,
            }
        )
    json.dump(
        {
            "shapes": [
                {
                    k: (v if k not in ("agg", "gain", "inter") else {str(kk): vv for kk, vv in v.items()})
                    for k, v in s.items()
                }
                for s in shapes_out
            ]
        },
        open(RAW, "w"),
        indent=2,
        default=str,
    )
    write_md(shapes_out)
    for s in shapes_out:
        print(
            f"{s['shape']:16s} base={s['tbase']:.1f}us relaunch={s['n_relaunch']} "
            f"pcc={s['baseline_pcc']} fastest={maskname(s['fastest'][0])}({s['agg'][s['fastest'][0]]['median']:.1f})",
            flush=True,
        )
    print(f"wrote {RAW} and {MD}", flush=True)


def write_md(shapes_out):
    with open(MD, "w") as f:
        f.write("# Regime-A critical-path ablation matrix\n\n")
        f.write(
            "Test-only compile-gated, program-cache-hashed 6-bit diagnostic (`diag_mask`, "
            "TT_REGIME_A_DIAG_MASK env). Public API + mask-0 binaries unchanged. Bits: "
            "1=in0_all, 2=in0_redun, 4=ring_fwd, 8=compute, 16=reduction, 32=output. Skips preserve all "
            "unaffected CB reserve/push/pop, pointers, waits, semaphores, loop structure; outputs for "
            "masks!=0 are intentionally invalid (PCC only for baseline).\n\n"
        )
        f.write(
            "Modes: baseline + 6 singles + 15 pairs (22). One persistent session/relaunch; 2 warmup + 12 "
            "timed iters/mode; >=2 relaunches with mode order reversed on odd relaunches; a 3rd added when a "
            "delta is near noise or relaunch distributions overlap. Kernel wall + per-RISC via run-host-id "
            "demux. `gain(S)=Tbase-T(S)`; `interaction(A,B)=gain(A+B)-gain(A)-gain(B)`. Commit + diagnostic; "
            "BH p150b, 1.35 GHz, fw 19.5.0; peak DRAM 512 GB/s. **Single-ablation %s are NOT summed as a "
            "forecast.**\n\n"
        )
        for s in shapes_out:
            ag, gn, tb = s["agg"], s["gain"], s["tbase"]
            f.write(
                f"## {s['shape']}  cfg (Ns,Pk,Sm,kb,nsb)={tuple(s['cfg'])}  "
                f"baseline {tb:.2f} us  (PCC {s['baseline_pcc']}, replay {s['cached_replay_matches']}, "
                f"{s['n_relaunch']} relaunches)\n\n"
            )
            f.write("### Baseline + singles\n\n")
            f.write("| mode | median us | gain us | gain % | IQR | spread% | crit RISC | B/N/T us |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for m in [0] + SINGLES:
                if m not in ag:
                    continue
                a = ag[m]
                r = a["risc"]
                f.write(
                    f"| {maskname(m)} | {a['median']:.2f} | {gn[m]:+.2f} | {gn[m]/tb*100:+.1f}% | "
                    f"{a['iqr']:.2f} | {a['spread_pct']:.1f} | {a['crit_risc']} | "
                    f"{r['BRISC']}/{r['NCRISC']}/{r['TRISC']} |\n"
                )
            f.write(
                f"\nTheoretical DRAM us @512GB/s: in0_all={s['theo']['in0_all']}, "
                f"in0_redun={s['theo']['in0_redun']}, in1={s['theo']['in1']}, output={s['theo']['output']}.\n\n"
            )
            # 6x6 interaction matrix
            f.write("### Pair-interaction matrix (us)  `interaction=gain(A+B)-gain(A)-gain(B)`\n\n")
            f.write("| A\\B | " + " | ".join(BITNAME[b] for b in BITS) + " |\n")
            f.write("|" + "---|" * (len(BITS) + 1) + "\n")
            for a in BITS:
                cells = []
                for b in BITS:
                    if a == b:
                        cells.append("·")
                    else:
                        key_ab = (min(a, b), max(a, b))
                        v = s["inter"].get(key_ab)
                        cells.append(f"{v:+.2f}" if v is not None else "n/a")
                f.write(f"| {BITNAME[a]} | " + " | ".join(cells) + " |\n")
            f.write("\n### Fastest combinations\n\n| rank | mode | median us | vs baseline |\n|---|---|---|---|\n")
            for i, m in enumerate(s["fastest"], 1):
                f.write(f"| {i} | {maskname(m)} | {ag[m]['median']:.2f} | {gn[m]/tb*100:+.1f}% |\n")
            # critical-RISC transitions
            f.write("\n### Critical-RISC transitions (which RISC bounds the wall)\n\n")
            base_crit = ag[0]["crit_risc"]
            trans = [(m, ag[m]["crit_risc"]) for m in [0] + SINGLES if m in ag and ag[m]["crit_risc"] != base_crit]
            f.write(
                f"- baseline critical RISC: **{base_crit}**"
                + (
                    f" (B/N/T = {ag[0]['risc']['BRISC']}/{ag[0]['risc']['NCRISC']}/{ag[0]['risc']['TRISC']} us)\n"
                    if ag[0]["risc"]
                    else "\n"
                )
            )
            if trans:
                for m, c in trans:
                    f.write(f"- {maskname(m)} shifts the critical RISC to **{c}**\n")
            else:
                f.write("- no single ablation changes the critical RISC.\n")
            # interpretation
            f.write("\n### Interpretation\n\n")
            for m in SINGLES:
                if m not in gn:
                    continue
                nm = BITNAME[m]
                gp = gn[m] / tb * 100
                th = s["theo"].get("in0_all" if m == 1 else "in0_redun" if m == 2 else "output" if m == 32 else None)
                exp = ""
                if th:
                    exp = f" (theo DRAM {th} us; exposed ~{gn[m]/th:.2f})" if th > 0 else ""
                tag = (
                    "hidden"
                    if abs(gp) < 2
                    else ("exposed" if gn[m] > 0 else "NEGATIVE (removing work worsened phasing)")
                )
                f.write(f"- **{nm}**: gain {gn[m]:+.2f} us ({gp:+.1f}%){exp} -> {tag}.\n")
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "report"])
    args = ap.parse_args()
    {"run": run, "report": report}[args.phase](args)


if __name__ == "__main__":
    main()
