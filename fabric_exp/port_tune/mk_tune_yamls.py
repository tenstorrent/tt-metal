#!/usr/bin/env python3
"""mk_tune_yamls.py: the receiver tuning sweep's own yamls (RECEIVER_TUNING.md), into yamls/ here, and
yamls/manifest.json (test counts of every suite tune_runs.sh runs, these and the study's).
  pert.yaml   the delay-perturbation rows: bursts of 1 / 4 / 8 / 32 packets with a 500-cycle gap (E/W), 8 with a
              2000-cycle gap (E/W), 8 with a 500-cycle gap (N/S), 6 with a 500-cycle gap (E/W; not a multiple of the
              completion group), and the steady 1-hop and 3-hop E/W writes; each at 4352 and 16 B (as their sources).
  val.yaml    data-checked rows run once per arm (benchmark off): 2D 1-hop E/W and N/S and 3-hop E/W writes, 4352 B
              (60 packets: the receiver's L1 chunk, as port_slots/mk_port_yamls.py) and 16 B (2000 packets).
  val1d.yaml  the same for 1D: Linear 1-hop and 3-hop E/W.
"""
import copy
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "../.."))
OUT = os.path.join(HERE, "yamls")
PERT = {
    "fabric_exp/exp_burst.yaml": ["BurstEW_N1_G500", "BurstEW_N4_G500", "BurstEW_N8_G500", "BurstEW_N32_G500",
                                  "BurstEW_N8_G2000", "BurstNS_N8_G500"],
    "fabric_exp/exp_burst612.yaml": ["BurstEW_N6_G500"],
    "fabric_exp/exp_2d_4352.yaml": ["Mesh1HopEW", "Mesh3HopEW"],
}
VAL = {"fabric_exp/exp_2d_4352.yaml": ("val", ["Mesh1HopEW", "Mesh1HopNS", "Mesh3HopEW"]),
       "fabric_exp/exp_1d_4352.yaml": ("val1d", ["Linear1HopEW", "Linear3HopEW"])}
# Every suite tune_runs.sh runs: name -> yaml (vc1: two ranks through tt-run).
SUITES = {
    "2d": "fabric_exp/exp_2d_4352.yaml",
    "1d": "fabric_exp/exp_1d_4352.yaml",
    "mix": "fabric_exp/streams/p3_mixes.yaml",
    "burst": "fabric_exp/exp_burst.yaml",
    "burst612": "fabric_exp/exp_burst612.yaml",
    "vc1": "fabric_exp/vc1/vc1_alloc.yaml",
    "vc1i": "fabric_exp/vc1/vc1_inter.yaml",
    "pert": "fabric_exp/port_tune/yamls/pert.yaml",
    # census-only suites (tune_runs.sh census; their extra env is in tune_runs.sh suite_env)
    "2d_1e": "fabric_exp/exp_2d_4352.yaml",
    "1d_1e": "fabric_exp/exp_1d_4352.yaml",
    "mux": "fabric_exp/v4/exp_mux.yaml",
    "vc2ew": "fabric_exp/pcs3/vc2ew4352.yaml",
    "vc02ew": "fabric_exp/pcs3/vc02ew4352.yaml",
    "vc02bew": "fabric_exp/pcs3/vc02bew4352.yaml",
    "val": "fabric_exp/port_tune/yamls/val.yaml",
    "val1d": "fabric_exp/port_tune/yamls/val1d.yaml",
    # the IDLE_PASSES sweep on e7fe1899049 (RECEIVER_TUNING.md section 8): NX 4096 B at 1 link, the ring control and latency
    "nx4096": "fabric_exp/port_tune/yamls/nx4096.yaml",
    "ring": "fabric_exp/v4/exp_ring.yaml",
    "lat": "fabric_exp/exp_latency.yaml",
}


def n_tests(t):
    n = 1
    for v in (t.get("parametrization_params") or {}).values():
        n *= len(v) if isinstance(v, list) else 1
    return n


def main():
    os.makedirs(OUT, exist_ok=True)
    tests, policies = [], None
    for src, names in PERT.items():
        d = yaml.safe_load(open(os.path.join(TREE, src)))
        policies = policies or d.get("allocation_policies")
        by = {t["name"]: t for t in d["Tests"]}
        tests += [copy.deepcopy(by[n]) for n in names]
    with open(os.path.join(OUT, "pert.yaml"), "w") as f:
        f.write("# Delay-perturbation rows of the receiver tuning sweep (mk_tune_yamls.py; RECEIVER_TUNING.md).\n")
        out = {"allocation_policies": policies} if policies else {}
        out["Tests"] = tests
        yaml.safe_dump(out, f, sort_keys=False)
    for src, (name, rows) in VAL.items():
        d = yaml.safe_load(open(os.path.join(TREE, src)))
        by = {t["name"]: t for t in d["Tests"]}
        out = []
        for r in rows:
            for size, packets in ((4352, 60), (16, 2000)):
                t = copy.deepcopy(by[r])
                t.pop("benchmark_mode", None)
                t["name"] = f"Val{r}_{size}"
                t["parametrization_params"] = {"ntype": ["unicast_write"], "size": [size]}
                t.setdefault("defaults", {})["num_packets"] = packets
                out.append(t)
        with open(os.path.join(OUT, f"{name}.yaml"), "w") as f:
            f.write("# Data-checked rows of the receiver tuning sweep (mk_tune_yamls.py; RECEIVER_TUNING.md).\n")
            o = {"allocation_policies": d["allocation_policies"]} if d.get("allocation_policies") else {}
            o["Tests"] = out
            yaml.safe_dump(o, f, sort_keys=False)
    # NX 4096 B, 1 link: pr_perf/pr_size_sweep.yaml's NeighborExchange row at one size.
    d = yaml.safe_load(open(os.path.join(TREE, "fabric_exp/pr_perf/pr_size_sweep.yaml")))
    nx = copy.deepcopy(next(t for t in d["Tests"] if t["name"] == "NeighborExchange"))
    nx["parametrization_params"]["size"] = [4096]
    with open(os.path.join(OUT, "nx4096.yaml"), "w") as f:
        f.write("# NeighborExchange write, 1 link, 4096 B (mk_tune_yamls.py; RECEIVER_TUNING.md section 8).\n")
        yaml.safe_dump({"allocation_policies": d["allocation_policies"], "Tests": [nx]}, f, sort_keys=False)
    manifest = {}
    for s, y in SUITES.items():
        d = yaml.safe_load(open(os.path.join(TREE, y)))
        manifest[s] = {"yaml": y, "tests": sum(n_tests(t) for t in d["Tests"])}
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    for s, m in manifest.items():
        print(f"{s:9} {m['tests']:3} tests  {m['yaml']}")


if __name__ == "__main__":
    main()
