#!/usr/bin/env python3
"""rf_bd_ingest.py LB2_JOB LB2_NODE GLX_JOB GLX_NODE: the report data refresh's cycle-breakdown records (2026-11-05):
main as shipped (d2bd12fcf1f) against the port (PR tip d51e42cd78a), PC samples over the debug bus (pc_sampler.py:
both ERISCs of every router while one long row runs; no measured perturbation), mapped by source line through
bd_map.py with the refresh's maps (refresh/map_main_d2bd.json: map_main.json re-anchored on main d2bd's own kernel;
refresh/map_port.json: map_new.json with the port's file names), reduced by build_breakdown.py.

Points: 1 (2D 1-hop E/W write 4352 B, 400G) and 3 (2D multicast write 4352 B, 400G) from the lb2 job
(refresh/rf_lb_stock.sh, labels rf1<a|b>_<main|port>_bd_<row>); point 2 at 400G (the same row at 1024 B) as a separate
output; points 2, 4, 5 at 200G (1-hop E/W 1024 B, 7 hops N/S 1024 B, full-ring multicast 4352 B) from the Galaxy job
(refresh/rf_glx.sbatch, labels <a|b>_<main|port>_s_<row>). Kernel caches: the router builds the jobs kept as tarballs
(<label>_routers.tgz), unpacked under /home/aagarwal/.tt-metal-rf-bdcache. Sets and points go to refresh/rf_bd_sets.json
and rf_bd_points*.json (the study's breakdown/*.json files are not touched); records are appended to data/breakdown.json
with ids bdrf-..., router "port" for the new router, and the job's records replaced on a rerun.
Descriptive only: a split within one ERISC, not which ERISC bounds a row (repo CLAUDE.md)."""
import copy, glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FE = os.path.dirname(os.path.dirname(HERE))
REPORT = os.path.join(FE, "report")
R = os.path.join(FE, "results")
CACHES = "/home/aagarwal/.tt-metal-rf-bdcache"
DATE = "2026-11-05"
lbjob, lbnode, gjob, gnode = sys.argv[1:5]
GR = f"/data/aagarwal/frs-glx/jobs/{gjob}/results"
MAPS = {"main": os.path.join(HERE, "map_main_d2bd.json"), "new": os.path.join(HERE, "map_port.json")}
BUILD = {"main": "main as shipped (d2bd12fcf1f; kernel v1, firmware keepalive)",
         "new": "port d51e42cd78a (PR tip)"}


def unpack(tgz, name):
    d = os.path.join(CACHES, name)
    if not os.path.isdir(d) and os.path.exists(tgz):
        os.makedirs(d)
        subprocess.run(["tar", "xzf", tgz, "-C", d], check=True)
    return d


sets, pts400, pts1024, ptsglx = [], {}, {}, {}
for r, lab in (("main", "main"), ("new", "port")):
    for p in "ab":
        c = unpack(f"{R}/rf1{p}_{lab}_bd_routers.tgz", f"lb{lbjob}_{p}_{lab}")
        for row, test, size, point in (("ew4352", "PcsEW4352", 4352, 1), ("mcast", "BdMcast", 4352, 3),
                                       ("ew1024", "PcsEW1024", 1024, 2)):
            sid = f"rf1{p}_{lab}_bd_{row}"
            if not os.path.exists(f"{R}/{sid}_samples.json"):
                continue
            sets.append(dict(id=sid, prefix=f"{sid}_samples", rate_csv=f"{sid}.csv", test=test, packet_size_B=size,
                             cache=c, logs=[f"rf1{p}_{lab}_bd_*.log"], sources=[f"tgz:fabric_exp/results/rf1_{lab}_bd_src.tgz"],
                             map=MAPS[r], router=r, build=BUILD[r], job=f"lb2 {lbjob} ({lbnode})", date=DATE,
                             system="P150x8", runner="refresh/rf_lb_stock.sh"))
            (pts1024 if point == 2 else pts400).setdefault((point, r), []).append(
                {"set": sid, "provisional": False,
                 "notes": f"pass {p} ({'main then port' if p == 'a' else 'port then main'}), {BUILD[r]}; same lb2 job."})
        # Galaxy
        gc = unpack(f"{GR}/bdcache_{lab}_{p}_routers.tgz", f"glx{gjob}_{p}_{lab}")
        for row, test, size, point in (("ew1024", "BdGlxEW1024", 1024, 2), ("ns7", "BdGlxNS7", 1024, 4),
                                       ("fullring", "BdGlxFullRing4352", 4352, 5)):
            sid = f"{p}_{lab}_s_{row}"
            if not os.path.exists(f"{GR}/{sid}_samples.json"):
                continue
            sets.append(dict(id=f"glxrf{gjob}_{sid}", prefix=f"{GR}/{sid}_samples", rate_csv=f"{GR}/{sid}.csv",
                             test=test, packet_size_B=size, cache=gc,
                             logs_unused=sorted(glob.glob(f"{GR}/{p}_{lab}_[cs]_*.log")),
                             sources=[f"tgz:{GR}/src_{lab}.tgz"], map=MAPS[r], router=r, build=BUILD[r],
                             job=f"Slurm {gjob} ({gnode})", date=DATE, system="BH Galaxy", runner="refresh/rf_glx.sbatch",
                             notes="Builds matched by the launch layout's text head (as the earlier Galaxy sets)."))
            ptsglx.setdefault((point, r), []).append(
                {"set": f"glxrf{gjob}_{sid}", "provisional": False,
                 "notes": f"pass {p} ({'main then port' if p == 'a' else 'port then main'}), {BUILD[r]}; node {gnode}."})
json.dump(sets, open(os.path.join(HERE, "rf_bd_sets.json"), "w"), indent=1)
base = json.load(open(os.path.join(REPORT, "breakdown", "points_phase2.json")))


def points_file(name, entries, keep, relabel=None):
    d = copy.deepcopy(base)
    d["version"] = "refresh-2026-11-05"
    d["note"] = "Report data refresh: main as shipped d2bd12fcf1f against the port d51e42cd78a, same job, passes a and b."
    d["points"] = [p for p in d["points"] if p["point"] in keep]
    for p in d["points"]:
        p["main"] = entries.get((p["point"], "main"), [])
        p["new"] = entries.get((p["point"], "new"), [])
        if relabel:
            p.update(relabel)
    path = os.path.join(HERE, name)
    json.dump(d, open(path, "w"), indent=1)
    return path


outs = []
for name, entries, keep, relabel in (
        ("rf_bd_points_400g.json", pts400, (1, 3), None),
        # built as point 6 (build_breakdown.py's EXCLUDE of main's E0 at point 2 is the Galaxy row's ambiguity, job
        # 170096), renumbered to 2 below with link 400G
        ("rf_bd_points_400g_1024.json", pts1024, (2,), {"point": 6, "label": "2D mesh 1-hop write, 1,024 B (400G)",
                                                         "link": "400G", "system": "P150x8",
                                                         "phase2": {"P150": "refresh/yamls/rf_bd_ew1024.yaml"}}),
        ("rf_bd_points_glx.json", ptsglx, (2, 4, 5), None)):
    if not entries:
        print("no sets for", name)
        continue
    pf = points_file(name, entries, keep, relabel)
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"rf_bd_{name}")
    subprocess.run([sys.executable, os.path.join(REPORT, "scripts", "build_breakdown.py"), "--points", pf, "--sets",
                    os.path.join(HERE, "rf_bd_sets.json"), "--out", out, "--md"], check=True)
    outs.append((name, out))

recs = []
for name, out in outs:
    for r in json.load(open(out)):
        tag = "glx" if "glx" in name else ("400g1024" if "1024" in name else "400g")
        if tag == "400g1024":
            r["point"] = 2
            r["id"] = r["id"].replace("-p6-", "-p2-")
            r["pair_key"] = (r.get("pair_key") or "").replace("-p6-", "-p2-") or None
        r["id"] = "bdrf-" + tag + "-" + r["id"].split("-", 1)[1]
        if r.get("pair_key"):
            r["pair_key"] = "bdrf-" + tag + "-" + r["pair_key"].split("-", 1)[1]
        if r.get("router") == "new":
            r["router"] = "port"
        r["same_job_with_main"] = True
        r["notes"] = (r.get("notes") or "") + (" Report data refresh 2026-11-05: main and the port sampled in the same "
                                               "job, passes a (main first) and b (port first); telemetry off.")
        recs.append(r)
path = os.path.join(REPORT, "data", "breakdown.json")
old = json.load(open(path))
keep = [r for r in old if not str(r.get("id", "")).startswith("bdrf-")]
json.dump(keep + recs, open(path, "w"), indent=1)
print(f"breakdown.json: {len(old) - len(keep)} refresh records replaced, {len(recs)} written, {len(keep) + len(recs)} total")
