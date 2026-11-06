#!/usr/bin/env python3
"""mk_pvs_yamls.py: the port-against-study comparison's yamls (SLOT_ALLOCATOR.md section 12.12; RECONCILE_HOST.md),
into yamls/ here, with their test counts in yamls/manifest.json. 4352 B slots throughout (the study's perf rows).
  pvs_ns          P150 2x4 mesh, one hop N/S (chip 0 -> 4) and E/W (0 -> 1): unicast writes of 16 and 4352 B, atomic
                  increments (the decode A/B's rows, LEDGER E1-18/E1-19, as plain test_tt_fabric rows: the port's
                  test binary has no stream support)
  pvs_t42ns       the P150 as a 4x2 XY torus (fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto): its N/S rows
                  (1-hop, bidirectional, 3-hop, wrap 1-hop) and 1-hop E/W, writes and scatter writes, 256-4352 B
                  (torusxy_4x2_sizes.yaml without its multicast rows)
  pvs_c2d         the 2D census suite (exp_2d_4352.yaml), as is
  pvs_c1lin / pvs_c1ring / pvs_c1ne   the 1D census suite (exp_1d_4352.yaml) split by topology (Linear, Ring,
                  NeighborExchange), so each part can get its own slot row on the port
"""
import copy
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "../.."))
OUT = os.path.join(HERE, "yamls")


def n_tests(t):
    n = 1
    for v in (t.get("parametrization_params") or {}).values():
        n *= len(v) if isinstance(v, list) else 1
    return n


def main():
    man_path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    load = lambda p: yaml.safe_load(open(os.path.join(TREE, p)))  # noqa: E731
    p2d = load("fabric_exp/slotstudy/sls_2d_p150.yaml")
    pol = p2d["allocation_policies"]

    def emit(name, header, tests, policies=pol):
        path = os.path.join(OUT, f"{name}.yaml")
        with open(path, "w") as f:
            f.write("".join(f"# {line}\n" for line in header))
            yaml.safe_dump({"allocation_policies": policies, "Tests": tests}, f, sort_keys=False)
        manifest[name] = {"yaml": os.path.relpath(path, TREE), "tests": sum(n_tests(t) for t in tests)}

    base = {t["name"]: t for t in p2d["Tests"]}["Sls1HopNS"]
    ns = []
    for d, dst in (("NS", 4), ("EW", 1)):
        for kind, ntype, sizes in (("Write", "unicast_write", [16, 4352]), ("Atomic", "atomic_inc", [4352])):
            t = copy.deepcopy(base)
            t["name"] = f"{d}_{kind}"
            t["parametrization_params"] = {"ntype": [ntype], "size": sizes}
            t["senders"][0]["patterns"][0]["destination"]["device"] = [0, dst]
            ns.append(t)
    head = "Port against study (fabric_exp/port_slots/mk_pvs_yamls.py)"
    emit("pvs_ns", [head, "P150 2x4 mesh: N/S and E/W 1-hop writes (16, 4352 B) and atomic increments."], ns)
    t42 = load("fabric_exp/torus/torusxy_4x2_sizes.yaml")
    keep = ["TorusXY4x2_1HopNS", "TorusXY4x2_1HopNSBidir", "TorusXY4x2_3HopNS", "TorusXY4x2_Wrap1HopNS",
            "TorusXY4x2_1HopEW"]
    emit("pvs_t42ns", [head, "P150 4x2 XY torus (needs its descriptor): N/S rows and 1-hop E/W, no multicast."],
         [t for t in t42["Tests"] if t["name"] in keep], t42.get("allocation_policies", pol))
    c2 = load("fabric_exp/exp_2d_4352.yaml")
    emit("pvs_c2d", [head, "fabric_exp/exp_2d_4352.yaml (the 2D census suite)."], c2["Tests"], c2["allocation_policies"])
    c1 = load("fabric_exp/exp_1d_4352.yaml")
    for name, topo in (("pvs_c1lin", "Linear"), ("pvs_c1ring", "Ring"), ("pvs_c1ne", "NeighborExchange")):
        emit(name, [head, f"fabric_exp/exp_1d_4352.yaml's {topo} rows (the 1D census suite)."],
             [t for t in c1["Tests"] if t["fabric_setup"]["topology"] == topo], c1["allocation_policies"])
    json.dump(manifest, open(man_path, "w"), indent=1)
    for k, v in manifest.items():
        if k.startswith("pvs"):
            print(k, v["tests"])


if __name__ == "__main__":
    main()
