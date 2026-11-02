"""Per-op tables of one decode round from a tests/test_fusedec_round_prof.py run.
python3 -I fusedec_roundtab.py <run dir> [--top N] [--trace M,A,B,S_greedy,...] [--json out.json]

Joins the digest rows of the replays of every trace (<run>/<name>_cpp.csv, trace id from prof_map.json) with the OpRecorder rows of the capture (oprows.json: op, caller, tensor shapes,
generic_op kernel names, op-id range; GLOBAL CALL COUNT >> 10 = op id, & 1023 = chip index) and the phase marks (embedding, engramN, layerN, head, ...). Per op / group: count per round,
kernel time on the mean chip, the slowest chip and the fastest chip (per-chip SUM over the ops of the group, mean over the replay sessions), mean op-to-op gap.
"""
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict

K = "DEVICE KERNEL DURATION [ns]"
G = "OP TO OP LATENCY [ns]"
SC, EC = "DEVICE KERNEL START CYCLE", "DEVICE KERNEL END CYCLE"
CLK = 1.35  # cycles per ns


def f(x):
    try:
        return float(x)
    except Exception:  # noqa
        return float("nan")


def shape_of(d):
    m = re.match(r"\[([0-9x]+)\]", d or "")
    return m.group(1) if m else ""


def category(r):
    """(category, name) of one recorded call"""
    op, who, kern = r["op"], r["caller"], r.get("kernels", "")
    first = who.split("<")[0]
    fl, _, rest = first.partition(":")
    fn = rest.split(":")[-1] if rest else ""
    low = op.lower()
    ins = r.get("ins", [])
    sh = [shape_of(i) for i in ins]
    if "all_gather" in low or "reduce_scatter" in low or "all_reduce" in low or "all_to_all" in low:
        if "dispatch" in low:
            return "MoE dispatch (CCL)", op.split(".")[-1]
        if fl.startswith("moe_block") or fn in ("forward",) and "moe" in fl:
            return "MoE reduce_scatter (CCL)", op.split(".")[-1]
        if fl.startswith("ccl_lean"):
            if fn == "_allreduce":
                return "attention all-reduce (CCL)", op.split(".")[-1]
            if fn == "_allgather":
                return "MoE all_gather (CCL)", op.split(".")[-1]
        if fl.startswith("engram"):
            return "Engram CCL", op.split(".")[-1]
        return "other CCL", f"{op.split('.')[-1]}@{fl}:{fn}"
    if "moe_compute" in low:
        return "MoE moe_compute", "moe_compute"
    if fl.startswith("moe_tail") or "moe_tail" in kern:
        return "MoE tail/fold", "moe_tail"
    if fl.startswith("router") or "router" in kern or fl.startswith("router_mm"):
        return "router", f"{op.split('.')[-1]}:{kern or fn}"
    if fl.startswith("moe_block") or fl.startswith("tt_moe_decode"):
        return "MoE glue", f"{op.split('.')[-1]}@{fn}"
    if fl.startswith("shared_expert"):
        return "shared expert", f"{op.split('.')[-1]}@{fn}"
    if fl.startswith("mhc") or "mhc" in kern:
        return "mHC", f"{kern or op.split('.')[-1]}@{fl}:{fn}"
    if fl.startswith("engram") or fn.startswith("_engram"):
        return "Engram", f"{op.split('.')[-1]}@{fl}:{fn}"
    if "sparse_sdpa" in low or "scaled_dot_product" in low:
        return "attention sdpa", op.split(".")[-1]
    if fl.startswith("idx_") or fl.startswith("indexer") or "idx_" in kern or "indexer" in fl:
        return "attention indexer", f"{kern or op.split('.')[-1]}@{fl}:{fn}"
    if fl.startswith("paged_ops") or fl.startswith("paged_kernels") or "paged" in kern:
        return "attention kv/paged ops", f"{kern or op.split('.')[-1]}@{fl}:{fn}"
    if low.endswith("linear") or low.endswith("matmul"):
        if fl.startswith(("attention", "paged_attention", "spec_attention", "spec_paged", "indexer")):
            return "attention linears", f"linear {' '.join(sh[:2])}"
        if fl.startswith(("mtp", "spec_decoder")):
            return "drafter/accept", f"linear {' '.join(sh[:2])}@{fl}"
        if fl.startswith(("device_head", "dsv41_model")):
            return "head", f"linear {' '.join(sh[:2])}"
        return "other matmul", f"{op.split('.')[-1]} {' '.join(sh[:2])}@{fl}:{fn}"
    if fl.startswith(
        ("attention", "paged_attention", "spec_attention", "spec_paged", "attn_fused", "attn_pre", "qk_norm", "rope")
    ):
        return "attention glue", f"{kern or op.split('.')[-1]}@{fl}:{fn}"
    if fl.startswith(("device_head", "shared_sampler")):
        return "head", f"{kern or op.split('.')[-1]}@{fl}:{fn}"
    if fl.startswith("device_sampler"):
        return "sampler", f"{kern or op.split('.')[-1]}@{fl}:{fn}"
    if fl.startswith("mtp"):
        return "drafter", f"{kern or op.split('.')[-1]}@{fl}:{fn}"
    if fl.startswith(("spec_decoder", "spec_model")):
        return "accept/commit/glue", f"{kern or op.split('.')[-1]}@{fl}:{fn}"
    if fl.startswith("layer"):
        return "layer glue", f"{kern or op.split('.')[-1]}@{fn}"
    if fl.startswith("dsv41_model") or fl.startswith("model"):
        return "embedding/model glue", f"{kern or op.split('.')[-1]}@{fl}:{fn}"
    return "other", f"{kern or op.split('.')[-1]}@{fl}:{fn}"


def load(run):
    O = json.load(open(os.path.join(run, "oprows.json")))
    pm = (
        json.load(open(os.path.join(run, "prof_map.json")))
        if os.path.exists(os.path.join(run, "prof_map.json"))
        else {}
    )
    walls = json.load(open(os.path.join(run, "walls.json"))) if os.path.exists(os.path.join(run, "walls.json")) else {}
    return O, pm, walls


def trace_table(run, O, name, tid):
    path = os.path.join(run, f"{name}_cpp.csv")
    if not os.path.exists(path):
        path = os.path.join(run, "prof", ".logs", "cpp_device_perf_report.csv")
    if not os.path.exists(path):
        return None
    rows = [r for r in O["rows"] if r.get("tid") == tid]
    byid = {}
    for r in rows:
        for i in range(r["id0"], r["id1"]):
            byid[i] = r
    tinfo = O["traces"][str(tid)]
    marks = O["marks"][tinfo["marks"][0] : tinfo["marks"][1]]
    marks = sorted(marks, key=lambda m: m[1])
    mid = [m[1] for m in marks]

    def phase(i):
        p = "start"
        for m in marks:
            if m[1] <= i:
                p = m[0]
            else:
                break
        return p

    dev = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # oid -> dev -> field -> [values over sessions]
    sess = set()
    for r in csv.DictReader(open(path)):
        if r["METAL TRACE ID"] != str(tid):
            continue
        g = int(float(r["GLOBAL CALL COUNT"]))
        oid, d = g >> 10, g & 1023
        s = r["METAL TRACE REPLAY SESSION ID"]
        sess.add(s)
        x = dev[oid][d]
        x["k"].append(f(r[K]))
        x["g"].append(f(r[G]))
        x["s"].append(f(r[SC]))
        x["e"].append(f(r[EC]))
        for key, col in (
            ("br", "DEVICE BRISC KERNEL DURATION [ns]"),
            ("nc", "DEVICE NCRISC KERNEL DURATION [ns]"),
            ("t0", "DEVICE TRISC0 KERNEL DURATION [ns]"),
            ("t1", "DEVICE TRISC1 KERNEL DURATION [ns]"),
            ("t2", "DEVICE TRISC2 KERNEL DURATION [ns]"),
            ("cc", "CORE COUNT"),
        ):
            x[key].append(f(r.get(col, "")))
    devs = sorted({d for o in dev.values() for d in o})
    ops = []
    for oid in sorted(dev):
        row = byid.get(oid)
        per = {}
        for d, x in dev[oid].items():
            ks = [v for v in x["k"] if v == v]
            gs = [v for v in x["g"] if v == v]
            per[d] = (
                statistics.mean(ks) if ks else 0.0,
                statistics.mean(gs) if gs else 0.0,
                x["s"][-1],
                x["e"][-1],
            )
        ops.append((oid, row, phase(oid), per))
    risc = {}
    for oid, dd in dev.items():
        risc[oid] = {
            k: statistics.mean([v for d in dd.values() for v in d[k] if v == v] or [0.0])
            for k in ("br", "nc", "t0", "t1", "t2", "cc")
        }
    return dict(ops=ops, devs=devs, sessions=sorted(sess), rows=len(rows), risc=risc)


def summarize(T, top=40):
    devs = T["devs"]
    nd = len(devs)
    chip_sum = {d: 0.0 for d in devs}
    chip_first = {d: 1e30 for d in devs}
    chip_last = {d: 0.0 for d in devs}
    groups = defaultdict(lambda: dict(n=0, dev={d: 0.0 for d in devs}, gap=0.0, rows=set()))
    cats = defaultdict(lambda: dict(n=0, dev={d: 0.0 for d in devs}, gap=0.0))
    phases = defaultdict(
        lambda: dict(
            n=0,
            gap=0.0,
            dev={d: 0.0 for d in devs},
            cats=defaultdict(float),
            catsmax=defaultdict(float),
            subs=defaultdict(float),
        )
    )
    unk = 0
    for oid, row, ph, per in T["ops"]:
        cat, nm = category(row) if row else ("unrecorded", f"op {oid}")
        if not row:
            unk += 1
        gaps = []
        for d, (k, g, s, e) in per.items():
            chip_sum[d] += k
            chip_first[d] = min(chip_first[d], s)
            chip_last[d] = max(chip_last[d], e)
            gaps.append(g)
        mg = statistics.mean(gaps) if gaps else 0.0
        for store, key in ((groups, (cat, nm)), (cats, cat), (phases, ph)):
            x = store[key]
            x["n"] += 1
            for d, (k, g, s, e) in per.items():
                x["dev"][d] += k
            if "gap" in x:
                x["gap"] += mg
            if store is phases:
                x["cats"][cat] += statistics.mean([v[0] for v in per.values()])
                x["catsmax"][cat] += max(v[0] for v in per.values())
                x["subs"][(cat, nm)] += statistics.mean([v[0] for v in per.values()])
    return dict(
        chip_sum=chip_sum,
        wall={d: (chip_last[d] - chip_first[d]) / CLK for d in devs},
        groups=groups,
        cats=cats,
        phases=phases,
        unk=unk,
        nops=len(T["ops"]),
    )


def stat(devd):
    v = [x for x in devd.values()]
    return statistics.mean(v), max(v), min(v)


def fmt_table(S, top):
    out = []
    cs = S["chip_sum"]
    m, mx, mn = stat(cs)
    wl = S["wall"]
    out.append(
        f"ops {S['nops']} (unrecorded {S['unk']}); kernel sum per chip: mean {m/1e3:.1f} us, slowest {mx/1e3:.1f}, fastest {mn/1e3:.1f}; replay wall per chip (first start..last end): mean {statistics.mean(wl.values())/1e3:.1f}, max {max(wl.values())/1e3:.1f} us"
    )
    out.append("")
    out.append(
        "| category | ops | mean-chip us | slowest-chip us | fastest-chip us | gaps us | % of mean-chip kernel |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for cat, x in sorted(S["cats"].items(), key=lambda kv: -stat(kv[1]["dev"])[0]):
        a, b, c = stat(x["dev"])
        out.append(
            f"| {cat} | {x['n']} | {a/1e3:.1f} | {b/1e3:.1f} | {c/1e3:.1f} | {x['gap']/1e3:.1f} | {100*a/m:.1f} |"
        )
    out.append("")
    out.append(f"| top {top} op groups | n | mean-chip us | slowest | fastest | gap us |")
    out.append("|---|---|---|---|---|---|")
    for (cat, nm), x in sorted(S["groups"].items(), key=lambda kv: -stat(kv[1]["dev"])[0])[:top]:
        a, b, c = stat(x["dev"])
        out.append(f"| {cat}: {nm} | {x['n']} | {a/1e3:.1f} | {b/1e3:.1f} | {c/1e3:.1f} | {x['gap']/1e3:.1f} |")
    out.append("")
    out.append("| phase | ops | mean-chip us | slowest | fastest | top categories |")
    out.append("|---|---|---|---|---|---|")
    for ph, x in S["phases"].items():
        a, b, c = stat(x["dev"])
        tc = ", ".join(f"{k} {v/1e3:.0f}" for k, v in sorted(x["cats"].items(), key=lambda kv: -kv[1])[:4])
        out.append(f"| {ph} | {x['n']} | {a/1e3:.1f} | {b/1e3:.1f} | {c/1e3:.1f} | {tc} |")
    for ph, x in S["phases"].items():
        if not ph.startswith(
            ("layer", "engram", "start", "head", "commit", "drafter", "accept", "embedding", "sampler")
        ):
            continue
        out.append("")
        out.append(
            f"### phase {ph}: {x['n']} ops, kernel mean-chip {stat(x['dev'])[0]/1e3:.1f} us, gaps {x['gap']/1e3:.1f} us (mean us / slowest-chip us per category; top sub-ops mean us)"
        )
        for c, v in sorted(x["cats"].items(), key=lambda kv: -kv[1]):
            subs = sorted(((n, t) for (cc, n), t in x["subs"].items() if cc == c), key=lambda kv: -kv[1])[:3]
            out.append(
                f"- {c}: {v/1e3:.1f} / {x['catsmax'][c]/1e3:.1f} :: "
                + "; ".join(f"{n[:60]} {t/1e3:.1f}" for n, t in subs)
            )
    return "\n".join(out)


def main():
    run = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 40
    which = sys.argv[sys.argv.index("--trace") + 1].split(",") if "--trace" in sys.argv else None
    O, pm, walls = load(run)
    print(
        f"# run {run}\nmeta {json.dumps(O['meta'])}\nwalls (unprofiled replay, ms) {json.dumps(walls.get('walls'))}\nsessions/prof_map {json.dumps(pm)}\n"
    )
    res = {}
    for name, tid in pm.items():
        if which and name not in which:
            continue
        T = trace_table(run, O, name, tid)
        if T is None:
            print(f"## {name}: no digest\n")
            continue
        S = summarize(T, top)
        print(f"## trace {name} (id {tid}, sessions {T['sessions']})\n")
        print(fmt_table(S, top))
        print()
        res[name] = dict(
            chip_sum_mean=stat(S["chip_sum"])[0],
            chip_sum_max=stat(S["chip_sum"])[1],
            cats={c: stat(x["dev"]) for c, x in S["cats"].items()},
            phases={p: stat(x["dev"]) for p, x in S["phases"].items()},
            phase_cats={p: {c: [v, x["catsmax"][c]] for c, v in x["cats"].items()} for p, x in S["phases"].items()},
            phase_subs={p: {f"{c}: {n}": v for (c, n), v in x["subs"].items()} for p, x in S["phases"].items()},
            groups={f"{c}: {n}": (x["n"],) + stat(x["dev"]) + (x["gap"],) for (c, n), x in S["groups"].items()},
        )
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"))


if __name__ == "__main__":
    main()
