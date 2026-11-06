#!/usr/bin/env python3
"""mk_port_yamls.py: the yamls of the port slot measurements (SLOT_ALLOCATOR.md section 12.12), into yamls/ here, and
yamls/manifest.json (test counts). Rows follow section 12's forwarding and receiver conventions (knee/mk_knee_yamls.py):
  pf2d_3072     2D forwarding rows (Sls3HopEW, SlsTurn4, SlsRowMerge, SlsMulticast) at 256-3072 B, 3072 B slots
  pf2dns_3072   N/S multi-hop on the P150 as a 4x2 mesh (knee/merge/p150_x8_4x2_mesh_mgd.textproto): chip 0 -> 6
  pf1d_3072     1D forwarding rows (SlsLin3HopEW, SlsLinRowMerge) at 256-3072 B, 3072 B slots
  pw2d_3072 / pw2d_4352   2D 1-hop rows (Sls1HopEW, Sls1HopNS, Sls1HopEWBidir) at 256 / 1024 / 2048 B
  pf2ds_3072 / pf2ds_4352 2D forwarding rows at 256 / 1024 / 2048 B
  pval2d_3072 / pval1d_3072 / pval2d_4352  data-checked 1-hop writes (benchmark off), packets <= the receiver ring
"""
import copy
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "../.."))
OUT = os.path.join(HERE, "yamls")
L1 = 0x100000
SRC = {"p2d": "fabric_exp/slotstudy/sls_2d_p150.yaml", "p1d": "fabric_exp/slotstudy/sls_1d_p150.yaml"}
FWD2D = ["Sls3HopEW", "SlsTurn4", "SlsRowMerge", "SlsMulticast"]
HOP1 = ["Sls1HopEW", "Sls1HopNS", "Sls1HopEWBidir"]
FWD1D = ["SlsLin3HopEW", "SlsLinRowMerge"]


def n_tests(t):
    n = 1
    for v in (t.get("parametrization_params") or {}).values():
        n *= len(v) if isinstance(v, list) else 1
    return n


def main():
    os.makedirs(OUT, exist_ok=True)
    src = {k: yaml.safe_load(open(os.path.join(TREE, v))) for k, v in SRC.items()}
    by = {k: {t["name"]: t for t in d["Tests"]} for k, d in src.items()}
    pol = src["p2d"]["allocation_policies"]
    manifest = {}

    def emit(name, header, tests):
        path = os.path.join(OUT, f"{name}.yaml")
        with open(path, "w") as f:
            f.write("".join(f"# {line}\n" for line in header))
            yaml.safe_dump({"allocation_policies": pol, "Tests": tests}, f, sort_keys=False)
        manifest[name] = {"yaml": os.path.relpath(path, TREE), "tests": sum(n_tests(t) for t in tests)}

    def rows(s, names, sizes, slot, bench=True, packets=None):
        out = []
        for r in names:
            t = copy.deepcopy(by[s][r])
            pp = t.setdefault("parametrization_params", {})
            pp["size"] = list(sizes)
            if "ntype" not in pp and "ntype" not in (t.get("defaults") or {}):
                pp["ntype"] = ["unicast_write"]
            if slot != 4352:
                t["fabric_setup"] = dict(t.get("fabric_setup") or {}, max_packet_size=slot)
            if not bench:
                t.pop("benchmark_mode", None)
                t["name"] = f"Val{r}"
                t.setdefault("defaults", {})["num_packets"] = packets
            out.append(t)
        return out

    head = "Port slot measurements (SLOT_ALLOCATOR.md section 12.12; fabric_exp/port_slots/mk_port_yamls.py)"
    emit("pf2d_3072", [head, f"2D forwarding rows {FWD2D}, 3072 B slots."], rows("p2d", FWD2D, [256, 1024, 2048, 3072], 3072))
    emit("pf1d_3072", [head, f"1D forwarding rows {FWD1D}, 3072 B slots."], rows("p1d", FWD1D, [256, 1024, 2048, 3072], 3072))
    # N/S multi-hop: the P150 as a 4x2 mesh (chip = row * 2 + column): chip 0 (row 0) to chip 6 (row 3), 3 N/S hops.
    ns = copy.deepcopy(by["p2d"]["Sls3HopEW"])
    ns["name"] = "Sls3HopNS4x2"
    ns["senders"][0]["device"] = [0, 0]
    ns["senders"][0]["patterns"][0]["destination"]["device"] = [0, 6]
    ns["parametrization_params"]["size"] = [256, 1024, 2048, 3072]
    ns["fabric_setup"] = dict(ns.get("fabric_setup") or {}, max_packet_size=3072)
    emit("pf2dns_3072", [head, "N/S 3-hop on the P150 as a 4x2 mesh (needs TT_MESH_GRAPH_DESC_PATH=4x2 mesh), 3072 B slots."],
         [ns])
    for slot in (3072, 4352):
        emit(f"pw2d_{slot}", [head, f"2D 1-hop rows {HOP1} at 256-2048 B, {slot} B slots."],
             rows("p2d", HOP1, [256, 1024, 2048], slot))
        emit(f"pf2ds_{slot}", [head, f"2D forwarding rows {FWD2D} at 256-2048 B, {slot} B slots."],
             rows("p2d", FWD2D, [256, 1024, 2048], slot))
    chunk = L1 // pol["receiver"]["max_configs_per_core"]
    emit("pval2d_3072", [head, f"Data-checked Sls1HopEW at 3072 B, {chunk // 3072} packets."],
         rows("p2d", ["Sls1HopEW"], [3072], 3072, bench=False, packets=chunk // 3072))
    emit("pval1d_3072", [head, f"Data-checked SlsLin1HopEW at 3072 B, {chunk // 3072} packets."],
         rows("p1d", ["SlsLin1HopEW"], [3072], 3072, bench=False, packets=chunk // 3072))
    emit("pval2d_4352", [head, f"Data-checked Sls1HopEW at 4352 B, {chunk // 4352} packets."],
         rows("p2d", ["Sls1HopEW"], [4352], 4352, bench=False, packets=chunk // 4352))
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    for k, v in manifest.items():
        print(k, v["tests"])


if __name__ == "__main__":
    main()
