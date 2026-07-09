#!/usr/bin/env python3
"""Manifest-driven dump discovery (the run-store idea): scan every reports/<ts>/run_manifest.json under
the deepseek_v32 profiler dirs, group by (commit, scenario, mode), and compute the device-collapsed
signposted total per group via merge_device_rows. No dependence on the clobber-prone top-level summary
CSVs — the raw reports/<ts>/ accumulate and the manifest says exactly what each one is.

Usage: python discover.py <commit_prefix>   -> prints/writes totals for that commit's dumps.
"""
import glob, json, os, sys
import pandas as pd

sys.path.insert(0, os.getcwd())
from models.tt_transformers.tests.test_utils import merge_device_rows

DUR = "DEVICE KERNEL DURATION [ns]"
BASE = "generated/profiler"


def region_of(rpt_dir):
    csvs = glob.glob(rpt_dir + "/ops_perf_results_*.csv")
    if not csvs:
        return None
    df0 = pd.read_csv(csvs[0], low_memory=False)
    mk = df0[df0["OP TYPE"] == "signpost"]["OP CODE"]
    a, b = mk[mk == "start"].index[0], mk[mk == "stop"].index[0]
    r = df0.iloc[a + 1 : b].copy()
    r[DUR] = pd.to_numeric(r[DUR], errors="coerce")
    return r


def total_and_iters(region):
    df = merge_device_rows(region)
    ri = region.reset_index(drop=True)
    iters = len(list(ri.index[ri["OP CODE"] == "MLA_START"]))
    return float(df[DUR].sum()), int(len(df)), iters


def discover():
    out = {}  # commit -> {(mode,scenario): {dir,total_ns,calls,iters,branch}}
    for mode in ("sparse", "dense"):
        for man in glob.glob(f"{BASE}/deepseek_v32_{mode}_mla_perf/reports/*/run_manifest.json"):
            m = json.load(open(man))
            d = os.path.dirname(man)
            key = (m.get("commit") or "?")[:11]
            out.setdefault(key, {"branch": m.get("branch"), "runs": {}})
            # keep the latest dir per (mode,scenario) for this commit (dir name sorts by timestamp)
            slot = out[key]["runs"].setdefault(f"{mode}/{m['scenario']}", {"dir": None})
            if slot["dir"] is None or d > slot["dir"]:
                slot["dir"] = d
    return out


if __name__ == "__main__":
    want = sys.argv[1][:11] if len(sys.argv) > 1 else None
    disc = discover()
    result = {}
    for commit, info in disc.items():
        if want and commit != want:
            continue
        for k, slot in sorted(info["runs"].items()):
            reg = region_of(slot["dir"])
            if reg is None:
                continue
            tot, calls, iters = total_and_iters(reg)
            result[k] = {"total_ns": tot, "calls": calls, "iters": iters, "dir": slot["dir"]}
            print(f"{commit} {info['branch']:38} {k:14} {tot/1e6:9.3f}ms  {calls:4d} calls  {iters:2d} it")
    if want:
        json.dump(
            result, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"totals_{want}.json"), "w"), indent=1
        )
