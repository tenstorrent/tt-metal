#!/usr/bin/env python3
"""topo_dump.py OUT.yaml: write this host's UMD cluster descriptor (tt_umd TopologyDiscovery, read-only: opens the
devices and reads ARC and Ethernet-core status, no reset or firmware load), then print the chip -> PCIe map, the
Ethernet links per chip pair, and whether chips {0,1,2,3} and {4,5,6,7} form the two 2x2 meshes the VC1 split's rank
binding expects (each chip linked to two others in its group, and links between the groups)."""
import collections
import sys

import tt_umd

out = sys.argv[1]
desc = tt_umd.TopologyDiscovery.create_cluster_descriptor()
desc.serialize_to_file(out)
chips = sorted(desc.get_all_chips())
print(f"chips {chips}; chip -> PCIe bus id {desc.get_chip_to_bus_id()}")
print(f"unhealthy: {desc.get_unhealthy_devices()}")
links = collections.Counter()
for chip, chans in desc.get_ethernet_connections().items():
    for chan, (other, _other_chan) in chans.items():
        links[tuple(sorted((chip, other)))] += 1
pairs = {p: n // 2 for p, n in links.items()}  # each link is listed from both ends
for p in sorted(pairs):
    print(f"  chips {p[0]}-{p[1]}: {pairs[p]} links")
groups = ({0, 1, 2, 3}, {4, 5, 6, 7})
for g in groups:
    ok = all(sum(1 for p in pairs if c in p and (set(p) - {c}) <= g) == 2 for c in g)
    print(f"group {sorted(g)}: {'a 2x2 (each chip linked to two others in the group)' if ok else 'NOT a 2x2'}")
between = sum(n for p, n in pairs.items() if (p[0] in groups[0]) != (p[1] in groups[0]))
print(f"links between the groups: {between}")
