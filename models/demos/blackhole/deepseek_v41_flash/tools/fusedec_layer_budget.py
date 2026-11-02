"""python3 -I fusedec_layer_budget.py <run dir> [--routes routes_round.json] [--md out.md]
Single-layer loop (tests/test_fusedec_layer.py, EAGER ops under the device profiler): reads the profiler digest (cpp_device_perf_report.csv of the run = the repo's ops_perf_results), joins the rows of every arm /
repetition (op-id ranges of layer_arms.json) with the OpRecorder rows (oprows.json) and prints per arm:
  METRIC (explicit): layer kernel time = sum over the layer's ops of DEVICE KERNEL DURATION [ns]; reported (a) on the slowest chip = max over chips of the per-chip sum, (b) on the mean chip. Averaged over the repetitions.
  Eager span (not the metric): (max DEVICE FW END - min DEVICE FW START) of the layer's ops per chip, includes the host launch gaps of eager execution; the on-device gap = span - kernel sum is reported for information.
  Per op: measured (mean / slowest / fastest chip) vs ideal (tools/fusedec_budget.py op model), ratio, cause class; per category; and the delta of every arm against the first arm."""
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusedec_budget as B  # noqa: E402

RT = B.RT
K = "DEVICE KERNEL DURATION [ns]"
FS, FE = "DEVICE FW START CYCLE", "DEVICE FW END CYCLE"


def main():
    run = sys.argv[1]
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    A = json.load(open(os.path.join(run, "layer_arms.json")))
    O = json.load(open(os.path.join(run, "oprows.json")))
    layer = A["layer"]
    byid = {}
    for r in O["rows"]:
        for i in range(r["id0"], r["id1"]):
            byid[i] = r
    routes = B.routes_per_layer(arg("--routes", os.path.join(run, "routes_round.json")))
    rows_T = O["meta"].get("T", 8) * 4
    ranges = {(n, ri): tuple(rg) for n, a in A["arms"].items() for ri, rg in enumerate(a["ranges"])}
    want = {}
    for (n, ri), (i0, i1) in ranges.items():
        for i in range(i0, i1):
            want[i] = (n, ri)
    dig = defaultdict(dict)  # oid -> dev -> (k, fs, fe)
    path = arg("--csv", os.path.join(run, "prof", ".logs", "cpp_device_perf_report.csv"))
    for r in csv.DictReader(open(path)):
        if r["METAL TRACE ID"]:
            continue
        oid = int(float(r["GLOBAL CALL COUNT"])) >> 10
        if oid not in want:
            continue
        try:
            dig[oid][int(r["DEVICE ID"])] = (float(r[K] or 0), float(r[FS] or 0), float(r[FE] or 0))
        except ValueError:
            pass
    out = []
    P = out.append
    summ = {}
    perop = {}
    for name in A["arms"]:
        reps = []
        for ri in range(len(A["arms"][name]["ranges"])):
            i0, i1 = ranges[(name, ri)]
            ids = [i for i in range(i0, i1) if i in dig and i in byid]
            chip = defaultdict(float)
            starts, ends = defaultdict(lambda: 1e30), defaultdict(float)
            ops = []
            for i in ids:
                per = dig[i]
                for d, (k, fs, fe) in per.items():
                    chip[d] += k
                    starts[d] = min(starts[d], fs)
                    ends[d] = max(ends[d], fe)
                ops.append((i, byid[i], per))
            span = statistics.mean((ends[d] - starts[d]) / RT.CLK for d in chip) if chip else 0
            reps.append(
                dict(slow=max(chip.values()), mean=statistics.mean(chip.values()), span=span, ops=ops, n=len(ids))
            )
        summ[name] = dict(
            slow=statistics.mean(r["slow"] for r in reps) * 1e-3,
            mean=statistics.mean(r["mean"] for r in reps) * 1e-3,
            span=statistics.mean(r["span"] for r in reps) * 1e-3,
            n=reps[0]["n"],
            spread=(max(r["slow"] for r in reps) - min(r["slow"] for r in reps)) * 1e-3,
        )
        # per op, aligned by order across the repetitions
        nops = min(len(r["ops"]) for r in reps)
        table = []
        for j in range(nops):
            kk = [[v[0] for v in r["ops"][j][2].values()] for r in reps]
            mean = statistics.mean(statistics.mean(k) for k in kk)
            slow = statistics.mean(max(k) for k in kk)
            fast = statistics.mean(min(k) for k in kk)
            row = reps[0]["ops"][j][1]
            mod = B.op_model(row, None, layer, routes, rows_T)
            dram_t, comp_t, fab_t = (
                mod["dram"] / B.DRAM_BW,
                (mod["flops"] / mod["peak"] if mod["peak"] else 0),
                mod["fab"] / B.FAB_BW,
            )
            ideal = max(dram_t, comp_t, fab_t)
            mean_s = mean * 1e-9
            if ideal > mean_s:
                ideal = mean_s
            cat, nm = RT.category(row)
            is_ccl = (
                "CCL" in cat or "all_gather" in row["op"] or "reduce_scatter" in row["op"] or "all_to_all" in row["op"]
            )
            cls = B.classify(mean_s, fast * 1e-9, ideal, dram_t, comp_t, fab_t, is_ccl)
            table.append(
                dict(
                    cat=cat,
                    name=nm,
                    mean=mean * 1e-3,
                    slow=slow * 1e-3,
                    fast=fast * 1e-3,
                    ideal=ideal * 1e6,
                    cls=cls,
                    dram=mod["dram"],
                    note=mod["note"],
                )
            )
        perop[name] = table
    P(
        f"# Single-layer loop: layer {layer}, B=8 k=3 T=8, eager under the device profiler, run {os.path.basename(run)}\n"
    )
    P(
        "Metric: layer kernel time = sum over the layer's ops of DEVICE KERNEL DURATION on the slowest chip (max over chips of the per-chip sum) and on the mean chip; mean of the repetitions. The eager span (FW start..end, host launch gaps included) is shown but is not the metric.\n"
    )
    P(
        "| arm | ops | kernel sum slowest chip us | kernel sum mean chip us | spread of the reps (slowest chip) us | eager span us (info) | PCC vs first arm | bit equal |"
    )
    P("|---|---|---|---|---|---|---|---|")
    first = None
    for name, s in summ.items():
        a = A["arms"][name]
        P(
            f"| {name} | {s['n']} | {s['slow']:.1f} | {s['mean']:.1f} | {s['spread']:.1f} | {s['span']:.1f} | {a.get('pcc', 'ref')} | {a.get('bit_equal', 'ref')} |"
        )
        first = first or name
    base = summ[first]
    for name, s in summ.items():
        if name != first:
            P(
                f"\narm {name}: slowest-chip sum {s['slow']-base['slow']:+.1f} us ({100*(s['slow']-base['slow'])/base['slow']:+.1f} %), mean-chip sum {s['mean']-base['mean']:+.1f} us vs {first}"
            )
    for name in summ:
        t = perop[name]
        cats = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
        for r in t:
            c = cats[r["cat"]]
            c[0] += 1
            c[1] += r["mean"]
            c[2] += r["slow"]
            c[3] += r["ideal"]
        P(f"\n## arm {name}: by category (us, mean chip / slowest-chip sum of the op times / ideal)\n")
        P("| category | ops | measured us | slowest-chip us | ideal us | above ideal us |")
        P("|---|---|---|---|---|---|")
        for k, v in sorted(cats.items(), key=lambda kv: -(kv[1][1] - kv[1][3])):
            P(f"| {k} | {v[0]} | {v[1]:.1f} | {v[2]:.1f} | {v[3]:.1f} | {v[1]-v[3]:.1f} |")
        if name == first or "--all-ops" in sys.argv:
            P(f"\n### arm {name}: every op, ranked by time above ideal\n")
            P("| op | measured us (mean / slowest / fastest chip) | ideal us | ratio | cause | note |")
            P("|---|---|---|---|---|---|")
            for r in sorted(t, key=lambda r: -(r["mean"] - r["ideal"]))[:45]:
                P(
                    f"| {r['cat']}: {r['name'][:60]} | {r['mean']:.1f} / {r['slow']:.1f} / {r['fast']:.1f} | {r['ideal']:.1f} | {r['mean']/max(r['ideal'],1e-3):.1f} | {r['cls']} | {r['note']} |"
                )
    text = "\n".join(out) + "\n"
    if "--md" in sys.argv:
        open(arg("--md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
