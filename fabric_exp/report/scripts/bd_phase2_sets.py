#!/usr/bin/env python3
"""bd_phase2_sets.py: turn phase-2 runs into sample sets and points for build_breakdown.py (BREAKDOWN_PLAN.md).

  bd_phase2_sets.py p150 TAG NEW_VARIANT --job "a09u32 job N" [--date YYYY-MM-DD]
      the runs of consolidation/bd_p150_runs.sh NEW_VARIANT TAG (labels bd_TAG_<pass>_<router>_<row>)
  bd_phase2_sets.py galaxy MANIFEST.json
      the Galaxy agent's runs, one record per sampled run (format: BREAKDOWN_PLAN.md, "Galaxy samples: what to save")

Writes (merging by id) fabric_exp/report/breakdown/sets_phase2.json, points_phase2.json (the five points: passes a and
b per router; points not yet run keep their provisional phase-1 entries) and points_costliest.json (the costliest-rows
profile, new router only). Then:
  build_breakdown.py --sets breakdown/sets_phase2.json --points breakdown/points_phase2.json --md
  build_breakdown.py --sets breakdown/sets_phase2.json --points breakdown/points_costliest.json \
      --out data/breakdown_costliest.json --md"""
import argparse
import copy
import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BD = os.path.normpath(os.path.join(HERE, "..", "breakdown"))
CACHE = "/home/aagarwal/.tt-metal-cache-bd-{tag}-{router}"
P150_POINTS = {1: ("ew4352", "PcsEW4352", 4352), 3: ("mcast", "BdMcast", 4352)}
COSTLIEST = [  # row, test, topology, send type, why (memory: project-pending-costliest-rows-profile)
    ("mcast_scatter", "BdMcastScatter", "2D mesh", "scatter_write", "MeshMulticast scatter (169.9 on v4pl6)"),
    ("halfring_scatter", "BdHalfRingScatter", "ring", "scatter_write", "HalfRingMulticast scatter (234.7 on v4pl6)"),
    ("ring1_scatter", "BdRing1Scatter", "ring", "scatter_write", "Ring1Hop scatter (189.5)"),
    ("unidirlin_write", "BdUnidirLinWrite", "1D line", "write", "UnidirLinearMulticast write (186.9)"),
    ("unidirlin_scatter", "BdUnidirLinScatter", "1D line", "scatter_write", "UnidirLinearMulticast scatter (227.3)"),
    ("torusns_scatter", "torusns_scatter", "Y-torus", "scatter_write", "TorusY1HopNS scatter (200.8)"),
]


def load(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def merge_sets(new):
    p = os.path.join(BD, "sets_phase2.json")
    have = {s["id"]: s for s in load(p, [])}
    have.update({s["id"]: s for s in new})
    json.dump(list(have.values()), open(p, "w"), indent=1)
    return p


def points_doc():
    p = os.path.join(BD, "points_phase2.json")
    if os.path.exists(p):
        return p, json.load(open(p))
    d = copy.deepcopy(json.load(open(os.path.join(BD, "points.json"))))
    d["version"] = "phase2"
    d["note"] = ("Phase 2: each point's own runs, passes a and b per router, replace the phase-1 provisional entries as "
                 "they arrive (bd_phase2_sets.py). Entries still marked provisional are phase-1 proxies.")
    return p, d


def set_point(doc, point, router, entries):
    for pt in doc["points"]:
        if pt["point"] == point:
            pt[router] = entries


def p150(a):
    date = a.date or datetime.date.today().isoformat()
    L = f"bd_{a.tag}"
    sets, entries = [], {}
    for router in ("main", "new"):
        build = "main (kernel v1, flags 0, firmware keepalive)" if router == "main" else a.variant
        base = dict(cache=CACHE.format(tag=a.tag, router=router),
                    logs=[f"{L}_c_{router}_*.log", f"{L}_?_{router}_*.log"],
                    sources=[f"tgz:fabric_exp/results/{L}_src_{router}.tgz"], map=router, router=router, build=build,
                    job=a.job, date=date, system="P150x8", runner="bd_p150_runs.sh")
        for point, (row, test, size) in P150_POINTS.items():
            for pas in ("a", "b"):
                sid = f"{L}_{pas}_{router}_{row}"
                sets.append(dict(id=sid, prefix=f"{sid}_samples", rate_csv=f"{sid}.csv", test=test,
                                 packet_size_B=size, **base))
                entries.setdefault((point, router), []).append(
                    {"set": sid, "provisional": False, "notes": f"pass {pas} ({'main then new' if pas == 'a' else 'new then main'}), {build}."})
        if router == "new":
            for row, test, topo, st, why in COSTLIEST:
                sid = f"{L}_x_new_{row}"
                sets.append(dict(id=sid, prefix=f"{sid}_samples", rate_csv=f"{sid}.csv", test=test,
                                 packet_size_B=4352, **base))
    print(merge_sets(sets))
    p, doc = points_doc()
    for (point, router), e in entries.items():
        set_point(doc, point, router, e)
    json.dump(doc, open(p, "w"), indent=1)
    print(p)
    cost = {"version": "phase2-costliest", "note": "Costliest two-ERISC rows on the new router only, both ERISCs "
            "(memory project-pending-costliest-rows-profile); P150, 4352 B. Descriptive only.", "points": []}
    for k, (row, test, topo, st, why) in enumerate(COSTLIEST):
        cost["points"].append({
            "point": 101 + k, "label": f"{test} ({why})", "link": "400G", "system": "P150x8", "topology": topo,
            "config": "two ERISCs", "send_type": st, "packet_size_B": 4352, "why": why, "slack_test": None,
            "main": [], "new": [{"set": f"{L}_x_new_{row}", "provisional": False, "notes": f"{a.variant}, one pass."}]})
    cost["missing_notes"] = {"main": "Not sampled for main (the costliest-rows profile is the new router's)."}
    cp = os.path.join(BD, "points_costliest.json")
    json.dump(cost, open(cp, "w"), indent=1)
    print(cp)


def galaxy(a):
    runs = json.load(open(a.manifest))
    sets, entries = [], {}
    for r in runs:
        sid = f"glx_{r['job'].split()[-1]}_{r['pass']}_{r['router']}_p{r['point']}"
        sets.append(dict(
            id=sid, prefix=r["samples_prefix"], cache=r["cache"], logs=r["logs"], sources=r["sources"],
            map=r["router"], router=r["router"], build=r["build"], job=r["job"], date=r["date"], system="BH Galaxy",
            test=r["test"], packet_size_B=r["packet_size_B"], rate_csv=r["rate_csv"], node=r.get("node"),
            runner="Galaxy agent"))
        note = f"pass {r['pass']}, {r['build']}, node {r.get('node')}."
        if r["router"] == "new" and r["build"].startswith("v4rxg3"):
            note += (" The sending ERISC (E0) is the sender before the sender pass: rerun on the final router "
                     "(the receiver is the lead's).")
        elif r["router"] == "new":  # v4txa11 (2026-11-03): the final router, sender pass and receiver both in
            note += " The final router: both ERISCs (sender and receiver) are the final code."
        entry = {"set": sid, "provisional": False, "notes": note}
        if r.get("ab_targets"):  # only an A/B that names the router overrides the busiest-router rule
            entry.update(ab_targets=r["ab_targets"], ab_source=r.get("ab_source", "an A/B (manifest)"))
        entries.setdefault((r["point"], r["router"]), []).append(entry)
    print(merge_sets(sets))
    p, doc = points_doc()
    for (point, router), e in entries.items():
        set_point(doc, point, router, sorted(e, key=lambda x: x["set"]))
    json.dump(doc, open(p, "w"), indent=1)
    print(p)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("p150"); s.add_argument("tag"); s.add_argument("variant")
    s.add_argument("--job", required=True); s.add_argument("--date")
    g = sub.add_parser("galaxy"); g.add_argument("manifest")
    a = ap.parse_args()
    (p150 if a.mode == "p150" else galaxy)(a)


if __name__ == "__main__":
    main()
