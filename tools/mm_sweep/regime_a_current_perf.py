#!/usr/bin/env python3
# READ-ONLY performance characterization of the CURRENT production regime_a_matmul (config=None -> Picker v3;
# mask 0 -> all accepted optimizations). No picker/kernel/placement/API change. Fresh result cache, resident
# in0/in1, 1 warmup + 8 timed iters per launch, 3 INTERLEAVED relaunches/shape (suspicious -> 2nd run).
# Records full per-shape metrics (raw JSON) and emits REGIME_A_CURRENT_PERF_REPORT.md.
#
# modes: measure | report | all
import json, os, sys, statistics, math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import regime_a_bench as rb
import regime_a_campaign as cp

cdiv = rb.cdiv
JSON = f"{HERE}/regime_a_current_perf.json"
REPORT = f"{HERE}/REGIME_A_CURRENT_PERF_REPORT.md"
MT16 = [(512, 2304, 6144), (512, 3072, 6144), (512, 6144, 768), (512, 6144, 2304), (512, 6144, 4608), (512, 15360, 768)]


def ai_flops_per_byte(M, K, N):
    # bf16 (2 bytes/elem): 2*M*K*N flops / (2*(M*K+K*N+M*N) bytes) = M*K*N / (M*K+K*N+M*N)
    return (M * K * N) / (M * K + K * N + M * N)


def shard_W(M, K, N, cfg):
    Ns, Pk, Sm, kb, nsb = cfg
    Ktl = rb.rup(cdiv(cdiv(K, 32), Pk), kb * 8)
    return Ktl // (8 * kb)


def measure_shape(M, K, N, relaunches):
    """`relaunches` interleaved fresh launches of the config=None product path (each = 1 warmup + 8 iters).
    Returns per-launch us_med + per-RISC + core-spread + PCC (fresh worker each -> no result-cache hit).
    cfg may be None if the python picker mirror can't configure the shape (some Mt=16 diagnostic shapes)."""
    try:
        cfg = tuple(rb.auto_config(M, K, N))
    except Exception:
        return None, [{"cls": "picker_infeasible"}]  # python-mirror can't pick (diagnostic Mt16); skip
    launches = []
    for _ in range(relaunches):
        r = rb.run_cfg(M, K, N, None, {})  # fresh dict => no cache; worker does warmup(compile)+8 cached iters
        launches.append(r)
    return cfg, launches


def _agg(M, K, N, cfg, launches):
    oks = [r for r in launches if rb._ok(r)]
    pm = rb.plan_metrics(M, K, N, cfg)
    Mt, Kt, Nt = cdiv(M, 32), cdiv(K, 32), cdiv(N, 32)
    ideal = pm["ideal_us_512"]
    meds = sorted(r["us_med"] for r in oks)
    med = statistics.median(meds) if meds else None
    Ns, Pk, Sm, kb, nsb = cfg
    Kt_s, Mt_s = pm["Ktl"] * Pk, pm["Mblk"] * Sm
    Nt_s = pm["N_slice"] * Ns * 8
    delivered_bytes = (Mt_s * Kt_s + Kt_s * Nt_s + Mt_s * Nt_s) * rb.TB
    logical = pm["logical_bytes"]
    # per-RISC (max across cores) + core spread: take from the first ok launch's detailed parse
    risc = next((r.get("per_risc_us") for r in oks if r.get("per_risc_us")), None)
    cspread = next((r.get("core_spread_pct") for r in oks if r.get("core_spread_pct") is not None), None)
    pcc = min((r["pcc"] for r in oks), default=None)
    return {
        "M": M,
        "K": K,
        "N": N,
        "Mt": Mt,
        "family": f"K{K}",
        "ai": round(ai_flops_per_byte(M, K, N), 1),
        "cfg": list(cfg),
        "cores": pm["cores"],
        "W": shard_W(M, K, N, cfg),
        "us_med": med,
        "us_all": meds,
        "spread_pct": ((max(meds) - min(meds)) / min(meds) * 100 if len(meds) > 1 and min(meds) else 0.0),
        "eff_gbps": (logical / (med / 1e6) / 1e9 if med else None),
        "delivered_gbps": (delivered_bytes / (med / 1e6) / 1e9 if med else None),
        "sched_over_valid": (delivered_bytes / logical if logical else None),
        "ideal_us": ideal,
        "wall_over_ideal": (med / ideal if (med and ideal) else None),
        "excess_us": (med - ideal if med else None),
        "per_risc_us": ({k: round(v, 1) for k, v in risc.items()} if risc else None),
        "core_spread_pct": (round(cspread, 0) if cspread is not None else None),
        "pcc": pcc,
        "n_ok": len(oks),
        "cls": ("ok" if oks else launches[0].get("cls", "?")),
        "cache": ("warmup-fresh + 8 cached-program iters" if oks else "n/a"),
    }


def measure(sections=("mt8", "mt16")):
    corpus = list(cp.corpus().keys())
    # resume: keep existing sections not being re-measured
    out = {"mt8": [], "mt16": []}
    if os.path.exists(JSON):
        try:
            out = json.load(open(JSON))
        except Exception:
            out = {"mt8": [], "mt16": []}
    for tag, shapes in (("mt8", corpus), ("mt16", MT16)):
        if tag not in sections:
            continue
        out[tag] = []
        for i, (M, K, N) in enumerate(sorted(shapes, key=lambda s: (cdiv(s[0], 32), s[1], s[2]))):
            cfg, launches = measure_shape(M, K, N, 3)
            if cfg is None:  # python picker mirror couldn't configure (diagnostic Mt16 only)
                rec = {
                    "M": M,
                    "K": K,
                    "N": N,
                    "Mt": cdiv(M, 32),
                    "family": f"K{K}",
                    "cls": "picker_infeasible",
                    "cfg": None,
                    "W": None,
                    "ai": round(ai_flops_per_byte(M, K, N), 1),
                }
                out[tag].append(rec)
                json.dump(out, open(JSON, "w"), indent=2)
                print(f"[{tag} {i+1}] {M}x{K}x{N} Mt{rec['Mt']} [picker_infeasible]", flush=True)
                continue
            rec = _agg(M, K, N, cfg, launches)
            # suspicious (relaunch spread > 5%) -> a 2nd independent run of 3 more, merge
            if rec["cls"] == "ok" and rec["spread_pct"] > 5.0:
                _, more = measure_shape(M, K, N, 3)
                rec = _agg(M, K, N, cfg, launches + more)
                rec["rechecked"] = True
            out[tag].append(rec)
            json.dump(out, open(JSON, "w"), indent=2)
            s = (
                f"{rec['us_med']:.1f}us {rec.get('wall_over_ideal',0):.2f}w/i exc={rec.get('excess_us',0):.1f} "
                f"{rec['eff_gbps']:.0f}GB/s pcc={rec['pcc']:.4f}"
                if rec["cls"] == "ok"
                else f"[{rec['cls']}]"
            )
            print(f"[{tag} {i+1}] {M}x{K}x{N} Mt{rec['Mt']} W{rec['W']} cfg={rec['cfg']} {s}", flush=True)
    print("MEASURE DONE", flush=True)
    return out


# ---------------------------------------------------------------- report
def _row(r):
    if r.get("cls") != "ok":
        return (
            f"| {r['M']}x{r['K']}x{r['N']} | {r['Mt']} | {r.get('ai','-')} | {r.get('cfg') or '-'} | - | "
            f"{r.get('W') or '-'} | - | - | - | - | - | - | - | - | - | **{r.get('cls','?')}** |"
        )
    pr = r.get("per_risc_us") or {}
    prs = "/".join(f"{k[0]}{pr[k]}" for k in ("BRISC", "NCRISC", "TRISC") if k in pr) or "-"
    return (
        f"| {r['M']}x{r['K']}x{r['N']} | {r['Mt']} | {r['ai']} | {tuple(r['cfg'])} | {r['cores']} | {r['W']} | "
        f"{_f(r['us_med'])} | {r.get('spread_pct',0):.1f} | {_f(r['eff_gbps'],0)}/{_f(r['delivered_gbps'],0)} | "
        f"{_f(r['ideal_us'])} | {_f(r['wall_over_ideal'],2)} | {_f(r['excess_us'])} | {prs} | "
        f"{_f(r.get('core_spread_pct'),0)} | {_f(r['pcc'],4)} | {r['cls']} |"
    )


def _f(x, d=1):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else "-"


HDR = (
    "| shape | Mt | AI | cfg(Ns,Pk,Sm,kb,nsb) | cores | W | us | spread% | eff/deliv GB/s | ideal us | "
    "w/ideal | excess us | per-RISC(B/N/T) | cspread% | pcc | cls |\n"
    "|---|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|"
)


def _classify(r):
    """Heuristic bottleneck label from AI, wall/ideal, ideal_us, core-spread, per-RISC."""
    if (r.get("ideal_us") or 0) < 8 and (r.get("excess_us") or 0) < 4:
        return "dispatch/short-kernel (few-us ideal; judge by excess)"
    tags = []
    pr = r.get("per_risc_us") or {}
    b, n, t = pr.get("BRISC"), pr.get("NCRISC"), pr.get("TRISC")
    if b and n and t:
        mx = max(b, n, t)
        if b >= mx * 0.98:
            tags.append("in1-read-bound (BRISC critical)")
        if t >= mx * 0.98:
            tags.append("compute-bound (TRISC critical)")
        if n >= mx * 0.98:
            tags.append("in0-fwd/reduce (NCRISC critical)")
    if (r.get("core_spread_pct") or 0) > 40:
        tags.append(f"load-imbalance (cspread {r['core_spread_pct']:.0f}%)")
    if (r.get("sched_over_valid") or 1) > 1.05:
        tags.append(f"bank-quant/tail (sched/valid {r['sched_over_valid']:.2f})")
    return ", ".join(tags) if tags else "balanced"


def report():
    d = json.load(open(JSON))
    ok = [r for r in d["mt8"] if r["cls"] == "ok"]
    L = [
        "# Regime-A current production perf characterization (config=None, mask 0)\n",
        "Read-only. Picker v3 + all accepted optimizations. Fresh cache, resident in0/in1, 1 warmup + 8 timed "
        "iters, 3 interleaved relaunches (suspicious spread>5% re-run in a 2nd independent run). 512 GB/s "
        "convention. Mt=16 FLUX shapes are a SEPARATE diagnostic section (excluded from Mt<=8 stats).\n",
    ]
    if ok:
        L.append(
            f"**{len(ok)}/{len(d['mt8'])} Mt<=8 shapes ok, {sum(1 for r in ok if r['pcc']<0.999)} PCC<0.999. "
            f"median %512={statistics.median([r['ideal_us']/r['us_med']*100 for r in ok]):.0f}%, "
            f"median wall/ideal={statistics.median([r['wall_over_ideal'] for r in ok]):.2f}.**\n"
        )

    # 1. full table by family (K)
    L.append("## 1. Full corpus (sorted by family = K, then N, then M)\n")
    L.append(HDR)
    for r in sorted(d["mt8"], key=lambda r: (r["K"], r["N"], r["M"])):
        L.append(_row(r))
    L.append("")

    # 2. worst underperformers
    L.append("## 2. Worst underperformers\n")
    L.append("### by wall/ideal (top 10)")
    L.append(HDR)
    for r in sorted(ok, key=lambda r: -(r["wall_over_ideal"] or 0))[:10]:
        L.append(_row(r))
    L.append("\n### by absolute excess us (top 10)")
    L.append(HDR)
    for r in sorted(ok, key=lambda r: -(r["excess_us"] or 0))[:10]:
        L.append(_row(r))
    L.append("\n### lowest effective GB/s among shapes with ideal>=20us (top 10)")
    L.append(HDR)
    for r in sorted([r for r in ok if (r["ideal_us"] or 0) >= 20], key=lambda r: (r["eff_gbps"] or 0))[:10]:
        L.append(_row(r))
    L.append("")

    # 3. per-Mt tables
    for mt in (1, 2, 4, 8):
        rows = [r for r in d["mt8"] if r["Mt"] == mt]
        if not rows:
            continue
        okm = [r for r in rows if r["cls"] == "ok"]
        med512 = statistics.median([r["ideal_us"] / r["us_med"] * 100 for r in okm]) if okm else 0
        L.append(f"## 3. Mt={mt} ({len(rows)} shapes, median %512={med512:.0f}%)\n")
        L.append(HDR)
        for r in sorted(rows, key=lambda r: (r["K"], r["N"])):
            L.append(_row(r))
        L.append("")

    # 4. LTX/FLUX table
    import re

    flux = set()
    if os.path.exists(f"{HERE}/FLUXLTX_COMPARE.md"):
        for m in re.finditer(r"(\d+)x(\d+)x(\d+)", open(f"{HERE}/FLUXLTX_COMPARE.md").read()):
            flux.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    fl = [r for r in d["mt8"] if (r["M"], r["K"], r["N"]) in flux]
    L.append(f"## 4. Real-model LTX/FLUX shapes (Mt<=8, {len(fl)})\n")
    L.append(HDR)
    for r in sorted(fl, key=lambda r: (r["Mt"], r["K"], r["N"])):
        L.append(_row(r))
    L.append("")

    # 5. interpretation
    L.append("## 5. Interpretation (bottleneck classification of the worst shapes)\n")
    L.append("| shape | Mt | w/ideal | excess us | eff GB/s | likely bottleneck |")
    L.append("|---|--|--|--|--|--|")
    worst = sorted(ok, key=lambda r: -(r["wall_over_ideal"] or 0))[:12]
    for r in worst:
        L.append(
            f"| {r['M']}x{r['K']}x{r['N']} | {r['Mt']} | {_f(r['wall_over_ideal'],2)} | {_f(r['excess_us'])} | "
            f"{_f(r['eff_gbps'],0)} | {_classify(r)} |"
        )
    L.append(
        "\n_Categories: dispatch/short-kernel (few-us ideal), in1/read-bandwidth (BRISC critical), "
        "in0-forward/sync (NCRISC critical), split-K/compute (TRISC critical + high core-spread), "
        "bank-quant/tail (sched/valid > 1.05). Judged by wall/ideal AND absolute excess, not eff-BW alone._\n"
    )

    # 6. matched-M scaling
    L.append("## 6. Matched-M scaling (fixed K,N; Mt=1->2->4->8)\n")
    import collections

    byKN = collections.defaultdict(dict)
    for r in ok:
        byKN[(r["K"], r["N"])][r["Mt"]] = r
    L.append("| K,N | Mt1 us(%512) | Mt2 | Mt4 | Mt8 | trend |")
    L.append("|---|--|--|--|--|--|")
    for (K, N), bymt in sorted(byKN.items()):
        if len(bymt) < 2:
            continue

        def cell(mt):
            r = bymt.get(mt)
            return f"{r['us_med']:.1f}({r['ideal_us']/r['us_med']*100:.0f}%)" if r else "-"

        pcts = {mt: bymt[mt]["ideal_us"] / bymt[mt]["us_med"] * 100 for mt in bymt}
        trend = f"{min(pcts.values()):.0f}->{max(pcts.values()):.0f}%512" if pcts else ""
        L.append(f"| {K},{N} | {cell(1)} | {cell(2)} | {cell(4)} | {cell(8)} | {trend} |")
    L.append("")

    # Mt=16 diagnostic (separate)
    if d.get("mt16"):
        L.append("## Diagnostic: Mt=16 FLUX shapes (NOT in Mt<=8 stats)\n")
        L.append(HDR)
        for r in sorted(d["mt16"], key=lambda r: (r["K"], r["N"])):
            L.append(_row(r))
        L.append("")

    open(REPORT, "w").write("\n".join(L) + "\n")
    print(f"REPORT DONE -> {REPORT}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "mt16":  # re-measure only the Mt=16 diagnostic section, merge into existing JSON, then report
        measure(("mt16",))
        report()
    else:
        if mode in ("measure", "all"):
            measure()
        if mode in ("report", "all"):
            report()
