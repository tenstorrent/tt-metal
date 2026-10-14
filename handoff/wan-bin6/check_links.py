# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Report the ethernet link state of a single BH Galaxy, using UMD only (no tt-metal device open).

    python -I handoff/wan-bin6/check_links.py           # inactive ports per chip (empty = all trained)
    python -I handoff/wan-bin6/check_links.py --map     # full neighbour map: tray/location -> eth channels
    python -I handoff/wan-bin6/check_links.py --dram    # DRAM harvest mask histogram

Chips are named T<tray>L<asic_location>. On this machine a healthy board prints nothing in the
default mode: every chip has ports 0-9 trained. Run it after every `tt-smi -glx_reset`; resets on this
board often leave a different set of links down.
"""

import collections
import os
import sys
import tempfile

import tt_umd
import yaml

cd = tt_umd.TopologyDiscovery.create_cluster_descriptor()
conns = cd.get_ethernet_connections()
loc = {c: f"T{cd.get_tray_id(c)}L{cd.get_asic_location(c)}" for c in cd.get_all_chips()}
order = sorted(cd.get_all_chips(), key=lambda c: (cd.get_tray_id(c), cd.get_asic_location(c)))

if "--map" in sys.argv:
    for c in order:
        nb = collections.defaultdict(list)
        for ch, (r, _) in conns.get(c, {}).items():
            nb[loc[r]].append(ch)
        print(loc[c], {k: sorted(v) for k, v in sorted(nb.items())})
elif "--dram" in sys.argv:
    path = tempfile.mktemp(suffix=".yaml")
    cd.serialize_to_file(path)
    harvesting = yaml.safe_load(open(path)).get("harvesting", {})
    os.remove(path)
    hist = collections.Counter(h["dram_harvesting_mask"] for h in harvesting.values())
    print("dram_harvesting_mask -> number of chips:", dict(sorted(hist.items())))
else:
    for c in order:
        active = set(cd.get_active_eth_channels(c))
        connected = set(conns.get(c, {}))
        inactive = sorted(set(range(10)) - active)
        if inactive or active != connected:
            print(loc[c], "inactive ports:", inactive, "active-but-unconnected:", sorted(active - connected))
