#!/usr/bin/env python3
"""Generate a Factory System Descriptor (FSD) textproto for the dual-T3K cluster
from the live PhysicalSystemDescriptor JSON (ttfm query-physical-topology).

Schema (tt.scaleout_tools.fsd.proto.FactorySystemDescriptor):
  hosts { hostname hall aisle rack shelf_u motherboard }         # host_id = 0-based index
  board_types { board_locations { host_id tray_id board_type } }
  eth_connections { connection { endpoint_a{host_id tray_id asic_location chan_id}
                                 endpoint_b{...} } }

The controller enumerates ASICs from the union of eth_connection endpoints and
synthesizes deterministic unique_ids; we only need correct
(host_id, tray_id, asic_location, chan_id) coordinates + a board_type per
(host_id, tray_id). tray_id 1..4 gives each n300 board its own slot so the UI
stops collapsing the 8 chips per host into 2.
"""
import json, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "psd_dualt3k.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "dual_t3k_fsd.textproto"

d = json.load(open(SRC))
pt = d["physical_topology"]

# host ordering follows host_to_rank (node-a=rank0, node-b=rank1) so FSD host_id == rank
host_order = [h["host_name"] for h in sorted(pt["host_to_rank"], key=lambda x: x["rank"])]
host_id = {h: i for i, h in enumerate(host_order)}

# asic_id -> (host, asic_location)
asic_host = {}
asic_loc = {}
for m in pt["asic_descriptors"]:
    a = m["asic_descriptor"]
    asic_host[m["asic_id"]] = a["host_name"]
    asic_loc[m["asic_id"]] = a["asic_location"]

# Collect all directed channel edges from the system graph.
# edge: (src_asic, src_chan, dst_asic, dst_chan, is_local)
edges = []
for hc in pt["system_graph"]["asic_connectivity_graph"]:
    for at in hc["asic_topologies"]:
        src = at["asic_id"]
        for c in at["topology"].get("asic_connections", []):
            dst = c["dst_asic_id"]
            for e in c["eth_connections"]:
                edges.append((src, e["src_chan"], dst, e["dst_chan"], e.get("is_local", False)))

# --- Determine board (tray) grouping: the n300 internal link pairs one loc0
#     chip with exactly one loc1 chip on the same host. Find those pairs. ---
loc0_to_loc1 = {}  # loc0 asic -> set of loc1 partners (local)
for s, sc, dt, dc, local in edges:
    if local and asic_host[s] == asic_host[dt] and asic_loc[s] == 0 and asic_loc[dt] == 1:
        loc0_to_loc1.setdefault(s, set()).add(dt)

for s, partners in loc0_to_loc1.items():
    assert len(partners) == 1, f"loc0 {s} has {len(partners)} loc1 partners: {partners}"

# Per host: sorted loc0 asics -> tray 1..4; each board = {loc0, its loc1 partner}
tray_of = {}  # asic_id -> tray_id
for h in host_order:
    loc0s = sorted([a for a in asic_host if asic_host[a] == h and asic_loc[a] == 0], key=int)
    assert len(loc0s) == 4, f"host {h} has {len(loc0s)} loc0 chips"
    for tray, a0 in enumerate(loc0s, start=1):
        a1 = next(iter(loc0_to_loc1[a0]))
        tray_of[a0] = tray
        tray_of[a1] = tray


def slot(asic):
    return (host_id[asic_host[asic]], tray_of[asic], asic_loc[asic])


# --- Dedupe undirected channel links. Canonical key over both endpoints. ---
seen = set()
conns = []  # (epA, epB) where ep = (host_id, tray, loc, chan)
for s, sc, dt, dc, local in edges:
    ha, ta, la = slot(s)
    hb, tb, lb = slot(dt)
    epA = (ha, ta, la, sc)
    epB = (hb, tb, lb, dc)
    key = tuple(sorted([epA, epB]))
    if key in seen:
        continue
    seen.add(key)
    conns.append((epA, epB))

conns.sort()

# sanity: cross-host connection count (is_local False) -> should be 8 channels (4 cables x2)
xhost = sum(1 for (a, b) in conns if a[0] != b[0])
print(
    f"[gen] hosts={host_order} asics={len(asic_host)} trays/host=4 "
    f"total_channel_links={len(conns)} cross_host_channel_links={xhost}",
    file=sys.stderr,
)

# --- Emit textproto ---
L = []
for i, h in enumerate(host_order):
    L.append("hosts {")
    L.append(f'  hostname: "{h}"')
    L.append(f"  rack: 1")
    L.append(f"  shelf_u: {32 - i * 4}")
    L.append("}")

L.append("board_types {")
for h in host_order:
    for tray in range(1, 5):
        L.append("  board_locations {")
        L.append(f"    host_id: {host_id[h]}")
        L.append(f"    tray_id: {tray}")
        L.append('    board_type: "N300"')
        L.append("  }")
L.append("}")

L.append("eth_connections {")
for epA, epB in conns:
    L.append("  connection {")
    for name, ep in (("endpoint_a", epA), ("endpoint_b", epB)):
        hh, tt, ll, ch = ep
        L.append(f"    {name} {{")
        L.append(f"      host_id: {hh}")
        L.append(f"      tray_id: {tt}")
        L.append(f"      asic_location: {ll}")
        L.append(f"      chan_id: {ch}")
        L.append("    }")
    L.append("  }")
L.append("}")

open(OUT, "w").write("\n".join(L) + "\n")
print(f"[gen] wrote {OUT}", file=sys.stderr)
