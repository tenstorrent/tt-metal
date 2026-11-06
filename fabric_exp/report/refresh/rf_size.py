#!/usr/bin/env python3
"""rf_size.py OPT [CASE ...]: code size and stack of the port's router images at the PR tip (report data refresh,
2026-11-05), for the coverage cases of data/size.json's v4txa11 rows that the port's offline tooling can build.

Builds: CPU-only mock runs of port_tune/cov/cov.py (TT_COV_TREE=/home/aagarwal/tt-metal-rf-port: d51e42cd78a + the
test harness's telemetry switch, its own host build) with TT_METAL_FABRIC_OPT_LEVEL=OPT; the JIT compiles every router
image the case's fabric needs into the case's cache (COV=/home/aagarwal/.tt-metal-rf-cov-<OPT>). Measured as
consolidation/size_report.py measures (its one(): loadable bytes; the deepest static stack from _start against the
ERISC's entry stack pointer: free stack), per case as build_size.py reduces (largest code per ERISC, least free stack;
one-ERISC routers have only the subordinate binary). Writes fabric_exp/report/refresh/size_port_d51e_<OPT>.json, then
drops the case caches (disk). Cases added here beyond cov.py's: bh_torus_vc2_1erisc (cov.py's bh_torus_vc2 on one ERISC)."""
import collections, concurrent.futures, glob, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FE = os.path.dirname(os.path.dirname(HERE))
OPT = sys.argv[1]
os.environ["TT_COV_TREE"] = "/home/aagarwal/tt-metal-rf-port"
os.environ["TT_COV_DIR"] = f"/home/aagarwal/.tt-metal-rf-cov-{OPT}"
os.environ["TT_COV_EXTRA_ENV"] = json.dumps({"TT_METAL_FABRIC_OPT_LEVEL": OPT})
sys.path.insert(0, os.path.join(FE, "port_tune", "cov"))
sys.path.insert(0, os.path.join(FE, "consolidation"))
import cov  # noqa: E402
import size_report as sr  # noqa: E402

cov.CASES["bh_torus_vc2_1erisc"] = (cov.BH, cov.test_yaml("Torus", cov.TORUS_Y + cov.VC2), cov.ONE)
DEFAULT = ["bh_1d", "bh_1d_1erisc", "bh_1d_mux", "bh_1d_telemetry", "bh_1d_trim", "bh_2d", "bh_2d_1erisc",
           "bh_2d_bwtelemetry", "bh_2d_mux", "bh_2d_telemetry", "bh_2d_trim", "bh_2d_udm_gtest", "bh_2d_vc2",
           "bh_2d_vc2_1erisc", "bh_neighbor", "bh_neighbor_1erisc", "bh_ring", "bh_ring_1erisc", "bh_ring_mux", "bh_torus",
           "bh_torus_1erisc", "bh_torus_vc2", "bh_torus_vc2_1erisc", "bh_trim_gtest", "bh_vc1", "bh_vc1_vc2", "glx_1d",
           "glx_2d", "glx_express", "glx_express_1erisc", "glx_ring", "glx_torus"]
cases = sys.argv[2:] or DEFAULT
cov.ensure_root()
with concurrent.futures.ThreadPoolExecutor(int(os.environ.get("RF_JOBS", "4"))) as ex:
    list(ex.map(cov.run_mock, cases))
out = {"opt": OPT, "tree": "d51e42cd78a (/home/aagarwal/tt-metal-rf-port)", "cases": {}}
for c in cases:
    d = os.path.join(cov.COV, "cases", c, "cache")
    elfs = []
    for proc in sr.ENTRY_SP:
        elfs += [(e, proc) for e in glob.glob(f"{d}/**/kernels/fabric_erisc_router/*/{proc}/{proc}.elf", recursive=True)]
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda x: (x[0].split("/")[-3], sr.ERISC[x[1]], sr.one(*x)), elfs))
    per = collections.defaultdict(dict)
    for h, e, r in res:
        per[h][e] = {"code": r["code"], "free": r["free"], "stack": r["stack"], "flags": r["flags"]}
    log = open(os.path.join(cov.COV, "cases", c, "mock.log"), errors="replace").read()
    fatal = [l.strip()[:300] for l in log.splitlines() if "TT_FATAL" in l or "TT_THROW" in l][:3]
    red = {}
    for e in ("E0", "E1"):
        xs = [v[e] for v in per.values() if e in v]
        if xs:
            red[e] = {"code": max(x["code"] for x in xs), "free": min(x["free"] for x in xs)}
    out["cases"][c] = {"images": len(res), "builds": len(per), "per_erisc": red, "fatal": fatal,
                       "flags": sorted({f for v in per.values() for x in v.values() for f in x["flags"]})}
    print(c, len(res), "images", {e: (v["code"], v["free"]) for e, v in red.items()}, ("FATAL: " + fatal[0]) if fatal else "")
    shutil.rmtree(d, ignore_errors=True)
json.dump(out, open(os.path.join(HERE, f"size_port_d51e_{OPT}.json"), "w"), indent=1)
