"""Assemble perf_data.json, VARIANT-KEYED (deepseek_v32 + glm_5_1).

Per-op tables / totals / cold-by-iter are re-derived from the BRANCH report dirs (discover.py's
totals_<branch>.json) via merge_device_rows — NOT the top-level summary CSVs, which get clobbered when a
later sweep (e.g. the baseline, or the other variant) writes the same {variant}_{mode}_mla_perf dir.
Baseline totals come from totals_<baseline>.json. build_html folds percall.json in as block_timing/expanded.
"""
import glob, json, os, sys
import pandas as pd

sys.path.insert(0, os.getcwd())
from models.tt_transformers.tests.test_utils import merge_device_rows

SPdir = os.path.dirname(os.path.abspath(__file__))
DUR = "DEVICE KERNEL DURATION [ns]"
SCEN = ["warm", "cold", "long"]
BRANCH = json.load(open(SPdir + "/totals_e71a1b0a666.json"))  # "v/m/s" -> {dir,total_ns,calls,iters}
BASELINE = json.load(open(SPdir + "/totals_dc50ed81dc5.json"))
VARIANTS = sorted({k.split("/")[0] for k in BRANCH})
CHUNK = 1280  # LoudBox per-box global chunk (both variants)

# Per-variant model config (from reference/{deepseek_v3_2,glm_5_1}_config.py; the lean manifest omits it).
CONFIG = {
    "deepseek_v32": {
        "hidden": 7168,
        "heads": 128,
        "heads_per_chip": 32,
        "q_lora": 1536,
        "kv_lora": 512,
        "qk_rope": 64,
        "qk_nope": 128,
        "v_head": 128,
        "index_heads": 64,
        "index_head_dim": 128,
        "index_topk": 2048,
        "kvpe": 576,
        "rope_interleave": False,
    },
    "glm_5_1": {
        "hidden": 6144,
        "heads": 64,
        "heads_per_chip": 16,
        "q_lora": 2048,
        "kv_lora": 512,
        "qk_rope": 64,
        "qk_nope": 192,
        "v_head": 256,
        "index_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "kvpe": 576,
        "rope_interleave": True,
    },
}
BASELINE_META = {
    "commit": "dc50ed8",
    "branch": "mvasilijevic/sparse_test_improvements",
    "label": "TP-head-sharded indexer (pre-SP×TP)",
    "desc": "merge-base 'before': the DSA indexer with the full-width TP logit all-reduce, identical "
    "harness/config to the branch. Dense is unchanged (indexer-independent) — a control.",
}


def region_of(rp):
    df0 = pd.read_csv(glob.glob(rp + "/ops_perf_results_*.csv")[0], low_memory=False)
    mk = df0[df0["OP TYPE"] == "signpost"]["OP CODE"]
    a, b = mk[mk == "start"].index[0], mk[mk == "stop"].index[0]
    r = df0.iloc[a + 1 : b].copy()
    r[DUR] = pd.to_numeric(r[DUR], errors="coerce")
    return r


def by_op(df):
    tot = float(df[DUR].sum())
    g = (
        df.groupby("OP CODE")[DUR]
        .agg(count="count", total_ns="sum", avg_ns="mean")
        .sort_values("total_ns", ascending=False)
    )
    ops = [
        {
            "op": op,
            "count": int(r["count"]),
            "total_ns": float(r["total_ns"]),
            "avg_ns": float(r["avg_ns"]),
            "pct": 100.0 * float(r["total_ns"]) / tot,
        }
        for op, r in g.iterrows()
    ]
    return ops, tot, int(g["count"].sum())


data = {"variants": {}, "default_variant": "deepseek_v32" if "deepseek_v32" in VARIANTS else VARIANTS[0]}
for v in VARIANTS:
    modes = {"sparse": {}, "dense": {}}
    cold_by_iter = {"sparse": None, "dense": None}
    for mode in ("sparse", "dense"):
        for s in SCEN:
            rp = BRANCH.get(f"{v}/{mode}/{s}", {}).get("dir")
            if not rp:
                modes[mode][s] = None
                continue
            region = region_of(rp)
            ops, tot, calls = by_op(merge_device_rows(region))
            modes[mode][s] = {"ops": ops, "total_ns": tot, "total_calls": calls}
            if s == "cold":
                ri = region.reset_index(drop=True)
                starts = list(ri.index[ri["OP CODE"] == "MLA_START"])
                bounds = starts + [len(ri)]
                per = []
                for i in range(len(starts)):
                    o2, t2, c2 = by_op(merge_device_rows(ri.iloc[bounds[i] + 1 : bounds[i + 1]]))
                    per.append(
                        {"iteration": i, "cache_depth_tokens": i * CHUNK, "ops": o2, "total_ns": t2, "op_count": c2}
                    )
                cold_by_iter[mode] = per
    bl = {"sparse": {}, "dense": {}}
    for mode in ("sparse", "dense"):
        for s in SCEN:
            bv = BASELINE.get(f"{v}/{mode}/{s}")
            bl[mode][s] = {"total_ns": bv["total_ns"], "total_calls": bv["calls"]} if bv else None
    data["variants"][v] = {
        "modes": modes,
        "cold_by_iter": cold_by_iter,
        "baseline": bl,
        "baseline_meta": BASELINE_META,
        "config": CONFIG.get(v),
    }

json.dump(data, open(SPdir + "/perf_data.json", "w"), indent=1)
for v in VARIANTS:
    for mode in ("sparse", "dense"):
        for s in SCEN:
            e = data["variants"][v]["modes"][mode][s]
            b = data["variants"][v]["baseline"][mode][s]
            if e:
                d = (1 - e["total_ns"] / b["total_ns"]) * 100 if b else None
                print(
                    f"{v:12} {mode}/{s:4}: {e['total_ns']/1e6:8.3f}ms ({e['total_calls']:4d})  baseline {b['total_ns']/1e6 if b else 0:8.3f}  Δ{('%+.1f%%'%d) if d is not None else 'n/a'}"
                )
print("wrote perf_data.json")
