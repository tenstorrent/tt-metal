#!/usr/bin/env python3
"""mk_p1d_yamls.py: the data-checked rows of the 1D leftover-order A/B (SLOT_ALLOCATOR.md section 12.15), run on the
staged arm before any benchmark row. Writes to yamls/ and adds them to yamls/manifest.json:
  pval1d_4352   pval1d_3072.yaml at 4352 B (one 1-hop unicast flow of full-size packets on a line)
  pvline_<S>    bfc/yamls/val1d_m1.yaml's Linear tests (multicast all-to-all, unidirectional, one-to-all unicast)
  pvring_<S>    bfc/yamls/val_ring_p150.yaml (full ring multicast and unicast, half ring, the quick deadlock row)
num_links 1 and 2 only (the P150 has 2 routing planes; the binary skips 3 and 4); at 3072 B every fabric_setup gets
max_packet_size 3072 (the payloads are 1024 B and under). Tests per yaml: the product of the parametrization lists,
times a pattern's iterations (LinearMulticastMicroLooped's 20 iterations are 20 tests)."""
import copy
import json
import os

import yaml

D = os.path.dirname(os.path.abspath(__file__))
BFC = os.path.join(D, "../consolidation/bfc/yamls")
REL = "fabric_exp/port_slots/yamls"


def count(tests):
    """Tests the binary reports as executed: each parametrization, times a pattern's iterations (each is a test)."""
    n = 0
    for t in tests:
        k = 1
        for v in t.get("parametrization_params", {}).values():
            k *= len(v)
        k *= max([p.get("iterations", 1) for p in t.get("patterns", [])] or [1])
        n += k
    return n


def sized(tests, size):
    out = []
    for t in copy.deepcopy(tests):
        pp = t.get("parametrization_params", {})
        if "num_links" in pp:
            pp["num_links"] = [x for x in pp["num_links"] if x <= 2]
        if size == 3072:
            t.setdefault("fabric_setup", {})["max_packet_size"] = 3072
        out.append(t)
    return out


def write(name, head, tests, manifest):
    path = os.path.join(D, "yamls", name + ".yaml")
    with open(path, "w") as f:
        f.write(head)
        yaml.safe_dump({"Tests": tests}, f, sort_keys=False)
    manifest[name] = {"yaml": f"{REL}/{name}.yaml", "tests": count(tests)}


manifest_path = os.path.join(D, "yamls", "manifest.json")
manifest = json.load(open(manifest_path))
line = [t for t in yaml.safe_load(open(os.path.join(BFC, "val1d_m1.yaml")))["Tests"]
        if t["fabric_setup"]["topology"] == "Linear"]
ring = yaml.safe_load(open(os.path.join(BFC, "val_ring_p150.yaml")))["Tests"]
for s in (4352, 3072):
    write(f"pvline_{s}", f"# mk_p1d_yamls.py: val1d_m1.yaml's Linear tests, num_links 1-2, {s} B slots. Data-checked.\n",
          sized(line, s), manifest)
    write(f"pvring_{s}", f"# mk_p1d_yamls.py: val_ring_p150.yaml, num_links 1-2, {s} B slots. Data-checked (the "
          "deadlock-quick row is a benchmark row).\n", sized(ring, s), manifest)
one_hop = yaml.safe_load(open(os.path.join(D, "yamls", "pval1d_3072.yaml")))
for t in one_hop["Tests"]:
    t["fabric_setup"].pop("max_packet_size", None)
    t["parametrization_params"]["size"] = [4352]
    t["defaults"]["num_packets"] = 51
write("pval1d_4352", "# mk_p1d_yamls.py: pval1d_3072.yaml at 4352 B (default slots), 51 packets. Data-checked "
      "SlsLin1HopEW.\n", one_hop["Tests"], manifest)
json.dump(manifest, open(manifest_path, "w"), indent=1)
for k in ("pval1d_4352", "pvline_4352", "pvline_3072", "pvring_4352", "pvring_3072"):
    print(k, manifest[k])
