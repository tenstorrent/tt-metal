"""Parse merge-8 perf logs: {(op, model, layout, isl): (ns, pcc)} per run tag and mode."""

import glob
import re
import statistics
import sys

LOGS = "/localdev/mbezulj/logs"
RT = re.compile(r'RT-CAL (\w+) \("(\w+)", (\d+)\): ([\d_]+) ns')
PCC = re.compile(r"PCC over active slice \((\d+) rows\): ([-0-9.einfa]+)")


def parse(run, mode):
    out = {}
    for p in sorted(glob.glob(f"{LOGS}/merge8_{run}_*_{mode}_*.log")):
        name = p.split("/")[-1]
        op = "routed" if f"{run}_routed_" in name else "swiglu" if f"{run}_swiglu_" in name else None
        if op is None:
            continue
        pcc = None
        for line in open(p, errors="replace"):
            m = PCC.search(line)
            if m:
                pcc = (int(m.group(1)), float(m.group(2)))
            m = RT.search(line)
            if m:
                isl = int(m.group(3))
                # Only a PCC line graded over this case's own row count belongs to it.
                val = pcc[1] if pcc is not None and pcc[0] == isl else None
                out[(op, m.group(2), m.group(1), isl)] = (int(m.group(4).replace("_", "")), val)
                pcc = None
    return out


def rows(a_runs, a_mode, b_runs, b_mode):
    a = [parse(r, a_mode) for r in a_runs]
    b = [parse(r, b_mode) for r in b_runs]
    keys = sorted(set().union(*a) | set().union(*b))
    res = []
    for k in keys:
        av = [x[k][0] for x in a if k in x]
        bv = [x[k][0] for x in b if k in x]
        ap = next((x[k][1] for x in a if k in x), None)
        bp = next((x[k][1] for x in b if k in x), None)
        am = statistics.median(av) if av else None
        bm = statistics.median(bv) if bv else None
        res.append(
            dict(
                op=k[0],
                model=k[1],
                layout=k[2],
                isl=k[3],
                a_us=None if am is None else round(am / 1e3, 2),
                b_us=None if bm is None else round(bm / 1e3, 2),
                delta=None if am is None or bm is None else round((bm - am) / 1e3, 2),
                delta_pct=None if am is None or bm is None else round(100 * (bm - am) / am, 1),
                pcc_a=ap,
                pcc_b=bp,
                a_runs_us=[round(v / 1e3, 2) for v in av],
                b_runs_us=[round(v / 1e3, 2) for v in bv],
            )
        )
    return res


if __name__ == "__main__":
    # parse.py runsA modeA runsB modeB
    for x in rows(sys.argv[1].split(","), sys.argv[2], sys.argv[3].split(","), sys.argv[4]):
        print(
            f"{x['op']:6s} {x['model']:9s} {x['layout'][2:]:12s} {x['isl']:5d} {x['a_us']} -> {x['b_us']} "
            f"d={x['delta']} ({x['delta_pct']}%) pcc {x['pcc_a']} -> {x['pcc_b']} a{x['a_runs_us']} b{x['b_runs_us']}"
        )
