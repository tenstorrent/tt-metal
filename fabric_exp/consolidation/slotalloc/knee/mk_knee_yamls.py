#!/usr/bin/env python3
"""mk_knee_yamls.py: the slot-need sweep's suites (SLOT_ALLOCATOR.md section 12), from the slot-study suites.

Each suite keeps only the rows on one channel kind's path, at the packet sizes of its slot size
(max_packet_size = the slot's payload, set in fabric_setup when it is not 4352 B):
  4352 B: 256, 1024, 2048, 4352   3072 B: 256, 1024, 2048, 3072   8704 B: 256, 1024, 4352, 8704   15232 B: 256, 4352, 15232
Suites: p2w / p2f (P150 2D worker-receiver / forwarding rows), p1w / p1f (P150 1D), pvc1 (P150 VC1 2x2 split),
p2sq / p1sq (15232 B squeeze rows), pval (data-checked 1-hop writes at large slots, packets <= the receiver ring's
slots), g2w / g2f (Galaxy 2D), gvc1 (Galaxy 2x2 split), gvc1x44 (Galaxy 4x4 split) and gvc1x44san (its data-checked
multi-mesh sanity, packet-capped). Writes knee/yamls/<suite>_<slot>.yaml and knee/yamls/manifest.json (tests per
suite, after parametrization: the runners' expected "Tests: N executed")."""
import copy
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "../../../.."))
OUT = os.path.join(HERE, "yamls")
SIZES = {4352: [256, 1024, 2048, 4352], 3072: [256, 1024, 2048, 3072], 8704: [256, 1024, 4352, 8704],
         15232: [256, 4352, 15232]}
SRC = {
    "p2d": "fabric_exp/slotstudy/sls_2d_p150.yaml", "p1d": "fabric_exp/slotstudy/sls_1d_p150.yaml",
    "pvc1": "fabric_exp/vc1/sls_vc1_all.yaml", "g2d": "fabric_exp/slotstudy/sls_2d_glx.yaml",
    "gvc1": "fabric_exp/slotstudy/sls_glx_vc1.yaml", "gx44": "fabric_exp/slotstudy/sls_glx_vc1_4x4.yaml",
}
SUITES = {  # name: (source, rows, slot sizes)
    "p2w": ("p2d", ["Sls1HopEW", "Sls1HopNS", "Sls1HopEWBidir"], [4352, 3072, 8704]),
    "p2f": ("p2d", ["Sls3HopEW", "SlsTurn4", "SlsRowMerge", "SlsMulticast"], [4352]),
    "p1w": ("p1d", ["SlsLin1HopEW", "Lin1HopEWBidir"], [4352, 8704]),
    "p1f": ("p1d", ["SlsLin3HopEW", "SlsLinRowMerge"], [4352, 8704]),
    "pvc1": ("pvc1", ["VC1Inter_0_0_to_1_0", "VC1Inter_0_0_to_1_1", "VC1Inter_0_0_to_1_2", "VC1Inter_0_0_to_1_3",
                      "VC1MixWorker", "VC1MixForward", "VC1SoloWorker", "VC1SoloForward", "VC1Mesh2Hop",
                      "VC1MeshMulticast"], [4352, 3072]),
    "p2sq": ("p2d", ["Sls1HopEW", "Sls1HopEWBidir", "Sls3HopEW", "SlsMulticast"], [15232]),
    "p1sq": ("p1d", ["SlsLin1HopEW", "SlsLin3HopEW", "SlsLinRowMerge"], [15232]),
    "g2w": ("g2d", ["Sls1HopEW", "Sls1HopNS", "Sls1HopEWBidir"], [4352]),
    "g2f": ("g2d", ["Sls3HopEW", "Sls3HopNS", "Sls7HopNS", "SlsTurn4", "SlsMulticast"], [4352]),
    "gvc1": ("gvc1", [f"GlxInter_0_{a}_to_1_{b}" for a in (0, 1) for b in range(4)] +
             ["GlxMixWorker", "GlxMixForward", "GlxSoloWorker", "GlxSoloForward", "VC1Mesh2Hop", "VC1MeshMulticast"],
             [4352, 3072]),
    "gvc1x44": ("gx44", [f"X44Inter_0_{a}_to_1_{b}" for a in (0, 15) for b in (0, 3, 12, 15)] +
                ["X44Mesh3HopEW", "X44Mesh3HopNS", "X44MeshCorner6"], [4352, 3072]),
}
L1 = 0x100000


def n_tests(t):
    n = 1
    for v in (t.get("parametrization_params") or {}).values():
        n *= len(v) if isinstance(v, list) else 1
    return n


def write(name, slot, header, tests, policies):
    d = {"allocation_policies": policies, "Tests": tests}
    path = os.path.join(OUT, f"{name}_{slot}.yaml")
    with open(path, "w") as f:
        f.write("".join(f"# {line}\n" for line in header))
        yaml.safe_dump(d, f, sort_keys=False)
    return os.path.relpath(path, TREE), sum(n_tests(t) for t in tests)


def main():
    os.makedirs(OUT, exist_ok=True)
    src = {k: yaml.safe_load(open(os.path.join(TREE, v))) for k, v in SRC.items()}
    by_name = {k: {t["name"]: t for t in d["Tests"]} for k, d in src.items()}
    # 1D bidirectional 1-hop: SlsLin1HopEW plus the reverse sender.
    lin = copy.deepcopy(by_name["p1d"]["SlsLin1HopEW"])
    s0 = lin["senders"][0]
    rev = copy.deepcopy(s0)
    rev["device"], rev["patterns"][0]["destination"]["device"] = s0["patterns"][0]["destination"]["device"], s0["device"]
    lin["name"], lin["senders"] = "Lin1HopEWBidir", [s0, rev]
    by_name["p1d"]["Lin1HopEWBidir"] = lin
    manifest = {}
    for name, (s, rows, slots) in SUITES.items():
        for slot in slots:
            tests = []
            for r in rows:
                t = copy.deepcopy(by_name[s][r])
                pp = t.setdefault("parametrization_params", {})
                pp["size"] = list(SIZES[slot])
                if "ntype" not in pp and "ntype" not in (t.get("defaults") or {}):
                    pp["ntype"] = ["unicast_write"]
                if slot != 4352:
                    t["fabric_setup"] = dict(t.get("fabric_setup") or {}, max_packet_size=slot)
                tests.append(t)
            hdr = [f"Slot-need sweep (SLOT_ALLOCATOR.md section 12; mk_knee_yamls.py): {', '.join(rows)} from",
                   f"{SRC[s]} at {SIZES[slot]} B packets, {slot} B slots."]
            manifest[f"{name}_{slot}"] = write(name, slot, hdr, tests, src[s]["allocation_policies"])
    # Composition test (SLOT_ALLOCATOR.md section 12.7): P150 2D rows at 256 and 4352 B only.
    for name, rows, sizes in (("q1h", ["Sls1HopEW", "Sls1HopEWBidir"], [256, 4352]),
                              ("q3h", ["Sls3HopEW", "SlsTurn4"], [256, 4352]),
                              ("qall", ["Sls1HopEW", "Sls1HopEWBidir", "Sls3HopEW", "SlsTurn4"], [256, 4352]),
                              ("x1h", ["Sls1HopEW", "Sls1HopEWBidir"], [256, 2048, 4352])):
        tests = []
        for r in rows:
            t = copy.deepcopy(by_name["p2d"][r])
            t.setdefault("parametrization_params", {})["size"] = list(sizes)
            tests.append(t)
        manifest[f"{name}_4352"] = write(name, 4352, [f"Sections 12.7-12.9: {', '.join(rows)} from "
                                         f"{SRC['p2d']} at {sizes} B packets, 4352 B slots."], tests,
                                         src["p2d"]["allocation_policies"])
    # Data-checked 1-hop writes at large slots (benchmark_mode off), packets <= the receiver ring's slots.
    for topo, row in (("p2d", "Sls1HopEW"), ("p1d", "SlsLin1HopEW")):
        for slot in (4352, 8704, 15232):
            t = copy.deepcopy(by_name[topo][row])
            t.pop("benchmark_mode", None)
            t["name"] = f"Val{row}"
            t["parametrization_params"] = {"ntype": ["unicast_write"], "size": [slot]}
            chunk = L1 // src[topo]["allocation_policies"]["receiver"]["max_configs_per_core"]
            t.setdefault("defaults", {})["num_packets"] = chunk // slot
            if slot != 4352:
                t["fabric_setup"] = dict(t.get("fabric_setup") or {}, max_packet_size=slot)
            name = f"pval{topo[1:]}"
            manifest[f"{name}_{slot}"] = write(name, slot, [f"Data-checked {row} at {slot} B packets and slots, "
                                               f"{chunk // slot} packets (= the receiver ring's slots)."], [t],
                                               src[topo]["allocation_policies"])
    # The 4x4 split's data-checked multi-mesh sanity (packet-capped), first in G2.
    san = [copy.deepcopy(by_name["gx44"][n]) for n in ("MultiMeshAllToAll", "MultiMeshNeighborExchangeInter")]
    manifest["gvc1x44san_4352"] = write("gvc1x44san", 4352, ["Data-checked multi-mesh sanity of the 4x4 split "
                                        "(sls_glx_vc1_4x4.yaml's first two tests, packets capped)."], san,
                                        src["gx44"]["allocation_policies"])
    # The same at 3072 B slots: payloads above 3072 B capped to it, packets <= the receiver ring's slots.
    san3 = copy.deepcopy(san)
    for t in san3:
        pp = t.setdefault("parametrization_params", {})
        if "size" in pp:
            pp["size"] = sorted({min(x, 3072) for x in pp["size"]})
        t["fabric_setup"] = dict(t.get("fabric_setup") or {}, max_packet_size=3072)
        chunk = L1 // src["gx44"]["allocation_policies"]["receiver"]["max_configs_per_core"]
        n = (t.get("defaults") or {}).get("num_packets")
        if isinstance(n, int) and "size" in pp and n > chunk // max(pp["size"]):
            t["defaults"]["num_packets"] = chunk // max(pp["size"])
    manifest["gvc1x44san_3072"] = write("gvc1x44san", 3072, ["Data-checked multi-mesh sanity of the 4x4 split at "
                                        "3072 B slots (payloads capped to 3072 B, packets <= the ring's slots)."],
                                        san3, src["gx44"]["allocation_policies"])
    json.dump({k: {"yaml": v[0], "tests": v[1]} for k, v in manifest.items()},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    for k, (p, n) in manifest.items():
        print(f"{k}: {n} tests  {p}")


if __name__ == "__main__":
    main()
