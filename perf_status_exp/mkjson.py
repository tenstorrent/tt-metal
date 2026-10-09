"""Build perf_status_exp/merge-8.json and final_sweep.json from the merge8_* logs.

usage: mkjson.py <final_mode>   (final_mode: the kernel tree of the final build, e.g. new)
Metadata (commits, tests, notes) comes from meta.json next to this script.
"""

import json
import statistics
import sys

sys.path.insert(0, "/localdev/mbezulj/tt-metal/.claude/worktrees/agent-abb2aba9472191522/perf_status_exp")
from parse import parse, rows  # noqa: E402

D = "/localdev/mbezulj/tt-metal/.claude/worktrees/agent-abb2aba9472191522/perf_status_exp"
FINAL = sys.argv[1] if len(sys.argv) > 1 else "new"
WIN = json.load(open("/localdev/mbezulj/tt-metal/.claude/worktrees/moe-ffn-isl64-zones/perf_status/winners.json"))
RSILU = json.load(
    open("/localdev/mbezulj/tt-metal/.claude/worktrees/agent-ae766552516edf96b/perf_status_exp/r-silu.json")
)


def wdelta(name, m, lay, isl):
    if name == "r-silu":
        for r in RSILU["rows"]:
            if r["model"] == m and r["layout"] == lay and r["isl"] == isl:
                return round(r["delta_ns"] / 1e3, 2)
        return None
    for r in WIN[name]:
        if r["model"] == m and r["layout"] == lay and r["isl"] == isl:
            return round(r["new_us"] - r["base_us"], 2)
    return None


def rename(rs, a, b):
    out = []
    for r in rs:
        out.append(
            dict(
                op=r["op"],
                model=r["model"],
                layout=r["layout"],
                isl=r["isl"],
                **{
                    f"{a}_us": r["a_us"],
                    f"{b}_us": r["b_us"],
                    "delta": r["delta"],
                    "delta_pct": r["delta_pct"],
                    f"pcc_{a}": r["pcc_a"],
                    f"pcc_{b}": r["pcc_b"],
                    f"{a}_runs_us": r["a_runs_us"],
                    f"{b}_runs_us": r["b_runs_us"],
                },
            )
        )
    return out


# Final sweep: base (untouched 768453d91be + main build) vs final.
fin = rename(rows(["fb1", "fb2"], "base", ["ff1", "ff2"], FINAL), "base", "final")
json.dump(
    dict(
        name="final_sweep",
        base="untouched git archive of 768453d91be (pytest CWD) + main build host",
        final=f"branch moe-ffn-opt-11x8 build + kernels ({FINAL})",
        method="RT device time, median of 3 per run; runs fb1,ff1,fb2,ff2 alternating; value = median of the 2 runs",
        card="TT_VISIBLE_DEVICES=0 (PCIe 2)",
        rows=fin,
    ),
    open(f"{D}/final_sweep.json", "w"),
    indent=1,
)

# merge-8 rows: the requested matrix, taken from the final sweep.
R_ISL = [0, 256, 320, 384, 512, 544, 640, 800, 1024]
S_ISL = [0, 64, 128, 192, 256, 288, 320]
m8rows = []
for r in fin:
    if (r["op"] == "routed" and r["isl"] in R_ISL) or (r["op"] == "swiglu" and r["isl"] in S_ISL):
        x = dict(
            op=r["op"],
            model=r["model"],
            layout=r["layout"],
            isl=r["isl"],
            base_us=r["base_us"],
            new_us=r["final_us"],
            delta=r["delta"],
            delta_pct=r["delta_pct"],
            pcc_base=r["pcc_base"],
            pcc_new=r["pcc_final"],
        )
        if r["op"] == "routed":
            a = wdelta("r-silu", r["model"], r["layout"], r["isl"])
            b = wdelta("r-tilize-e2e", r["model"], r["layout"], r["isl"])
            x["win_rsilu_delta"] = a
            x["win_rtilize_delta"] = b
            x["win_sum_delta"] = None if a is None or b is None else round(a + b, 2)
        else:
            x["win_scombo_delta"] = wdelta("s-combo", r["model"], r["layout"], r["isl"])
        m8rows.append(x)

step5 = rename(rows(["ab1", "ab2"], "m7", ["ab1", "ab2"], "m8"), "without", "with")
# Step 6 final (NEED_START-only 2-page scratch): sx = ndshard, si = interleaved recheck runs.
S6 = ["sx1", "sx2", "si1", "si2"]
step6 = rename(rows(S6, "m8", S6, "new"), "without", "with")
# First try (always-2-page scratch) incl. the routed control: ab1/ab2.
step6_first = rename(rows(["ab1", "ab2"], "m8", ["ab1", "ab2"], "new"), "without", "with")

th = [parse("th1", FINAL), parse("th2", FINAL)]
thresh = []
for m in ["glm_53", "kimi_k2_7"]:
    for isl in [192, 224, 256, 272, 288, 320, 352, 384]:
        ks, kr = ("swiglu", m, "w_ndshard", isl), ("routed", m, "w_ndshard", isl)
        s = [x[ks][0] for x in th if ks in x]
        r = [x[kr][0] for x in th if kr in x]
        su = round(statistics.median(s) / 1e3, 2) if s else None
        ru = round(statistics.median(r) / 1e3, 2) if r else None
        thresh.append(
            dict(
                model=m,
                layout="w_ndshard",
                isl=isl,
                swiglu_us=su,
                routed_us=ru,
                routed_minus_swiglu=None if su is None or ru is None else round(ru - su, 2),
                faster=None if su is None or ru is None else ("swiglu" if su < ru else "routed"),
                swiglu_runs_us=[round(v / 1e3, 2) for v in s],
                routed_runs_us=[round(v / 1e3, 2) for v in r],
                pcc_swiglu=next((x[ks][1] for x in th if ks in x), None),
                pcc_routed=next((x[kr][1] for x in th if kr in x), None),
            )
        )

doc = json.load(open(f"{D}/meta.json"))
doc.update(dict(rows=m8rows, step5_ab=step5, step6_ab=step6, step6_first_try_ab=step6_first, threshold=thresh))
json.dump(doc, open(f"{D}/merge-8.json", "w"), indent=1)

f = lambda v: "-" if v is None else f"{v:.2f}"
print("== merge-8 rows")
for r in m8rows:
    extra = (
        f"silu={f(r.get('win_rsilu_delta'))} tilize={f(r.get('win_rtilize_delta'))} sum={f(r.get('win_sum_delta'))}"
        if r["op"] == "routed"
        else f"scombo={f(r.get('win_scombo_delta'))}"
    )
    print(
        f"{r['op']:6s} {r['model']:9s} {r['layout'][2:]:12s} {r['isl']:5d} {f(r['base_us'])} {f(r['new_us'])} "
        f"{f(r['delta'])} ({r['delta_pct']}%) pcc {r['pcc_base']}/{r['pcc_new']} {extra}"
    )
for name, rs in (("step5", step5), ("step6", step6)):
    print(f"== {name}")
    for r in rs:
        print(
            f"{r['op']:6s} {r['model']:9s} {r['layout'][2:]:12s} {r['isl']:5d} {f(r['without_us'])} {f(r['with_us'])} "
            f"{f(r['delta'])} {r['without_runs_us']} {r['with_runs_us']}"
        )
print("== threshold")
for t in thresh:
    print(t["model"], t["isl"], t["swiglu_us"], t["routed_us"], t["routed_minus_swiglu"], t["faster"])
