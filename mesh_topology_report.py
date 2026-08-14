#!/usr/bin/env python3
"""Report physical mesh layout and ring/torus completeness for a TT box.

Usage:  python3 mesh_topology_report.py <live_cluster_descriptor.yaml> [system_health.txt]

Reads the control-plane dumps in ./generated/fabric/ (written automatically by any
run that opens a mesh device), so run something on the device first.
"""
import sys, re, glob, collections, yaml

REF_GLOB = "tt_metal/third_party/tt-cluster-descriptors/*/*/*clus_desc.yaml"


def load_generated():
    (cmap,) = glob.glob("generated/fabric/physical_chip_mesh_coordinate_mapping_*.yaml")
    coord = {int(k): tuple(v) for k, v in yaml.safe_load(open(cmap))["chips"].items()}
    pos = {}
    for f in glob.glob("generated/fabric/asic_to_fabric_node_mapping_rank_*.yaml"):
        d = yaml.safe_load(open(f))["asic_to_fabric_node_mapping"]
        for h in d["hostnames"]:
            for e in h["mesh"]:
                for c in e.get("chips", []):
                    ap = c["asic_position"]
                    pos[c["umd_chip_id"]] = (ap["tray_id"], ap["asic_location"])
    return coord, pos


def link_pairs(cd):
    """Undirected chip-pair -> ethernet channel count, intra-box only."""
    L = collections.Counter()
    for a, b in cd["ethernet_connections"]:
        if a["chip"] != b["chip"]:
            L[tuple(sorted((a["chip"], b["chip"])))] += 1
    return L


def main():
    cd = yaml.safe_load(open(sys.argv[1]))
    coord, pos = load_generated()
    L = link_pairs(cd)
    c2 = {v: k for k, v in coord.items()}
    NR = max(r for r, _ in coord.values()) + 1
    NC = max(c for _, c in coord.values()) + 1
    n = lambda p, q: L.get(tuple(sorted((c2[p], c2[q]))), 0)

    print(f"physical mesh = {NR} x {NC}   (dim0 = Y = NS = rows, dim1 = X = EW = cols)")
    print(f"intra-box chip pairs = {len(L)}, ethernet channels = {sum(L.values())}\n")

    print("mesh(row,col) -> chip / tray.asic")
    for r in range(NR):
        row = []
        for c in range(NC):
            ch = c2[(r, c)]
            t, a = pos.get(ch, ("?", "?"))
            row.append(f"{ch:>2}={t}.{a}")
        print(f"  r{r}  " + "  ".join(row))

    print("\nring completeness")
    bad_x = [r for r in range(NR) if n((r, NC - 1), (r, 0)) == 0]
    bad_y = [c for c in range(NC) if n((NR - 1, c), (0, c)) == 0]
    for r in range(NR):
        if any(n((r, c), (r, c + 1)) == 0 for c in range(NC - 1)):
            print(f"  !! row {r} has a broken hop (not just the wrap)")
    for c in range(NC):
        if any(n((r, c), (r + 1, c)) == 0 for r in range(NR - 1)):
            print(f"  !! col {c} has a broken hop (not just the wrap)")
    print(
        f"  X rings (dim1, len {NC}, one per row): {NR - len(bad_x)}/{NR} closed"
        + (f"   OPEN rows: {bad_x}" if bad_x else "")
    )
    print(
        f"  Y rings (dim0, len {NR}, one per col): {NC - len(bad_y)}/{NC} closed"
        + (f"   OPEN cols: {bad_y}" if bad_y else "")
    )

    # cabling census in tray/asic terms
    def census(pairs, asic_of):
        same, cross = collections.Counter(), collections.Counter()
        for a, b in pairs:
            x, y = asic_of[a], asic_of[b]
            if x == y:
                same[x] += 1
            else:
                cross[(min(x, y), max(x, y))] += 1
        return same, cross

    asic_live = {ch: a for ch, (t, a) in pos.items()}
    same, cross = census(L, asic_live)
    print("\ninter-tray asicN<->asicN cables (live):", dict(sorted(same.items())))

    for ref in sorted(glob.glob(REF_GLOB)):
        rcd = yaml.safe_load(open(ref))
        if "asic_locations" not in rcd:
            continue
        RL = link_pairs(rcd)
        if len(rcd["asic_locations"]) != len(pos):
            continue
        rsame, rcross = census(RL, rcd["asic_locations"])
        diff = {k: (rsame[k], same[k]) for k in set(rsame) | set(same) if rsame[k] != same[k]}
        cdiff = {k: (rcross[k], cross[k]) for k in set(rcross) | set(cross) if rcross[k] != cross[k]}
        print(f"\nvs reference {ref.split('/')[-1]}: pairs {len(RL)} -> {len(L)}")
        print(f"  inter-tray asic cable diff (ref, live): {diff or 'identical'}")
        print(f"  intra-tray cable diff      (ref, live): {cdiff or 'identical'}")

    # optional: port-level detail
    if len(sys.argv) > 2:
        txt = open(sys.argv[2]).read()
        peers, cur = {}, None
        for line in txt.splitlines():
            m = re.match(r"Chip: (\d+) Unique ID", line)
            if m:
                cur = int(m.group(1))
                peers[cur] = {}
                continue
            c = re.match(r"\s*eth channel (\d+)", line)
            if not c or cur is None:
                continue
            p = re.search(r"connected to chip (\d+) ", line)
            if "link UP" in line:
                pid = int(p.group(1)) if p else None
                peers[cur][int(c.group(1))] = pid if (pid is not None and pid in coord) else "EXT"
            else:
                peers[cur][int(c.group(1))] = None
        dead = collections.Counter()
        for ch, m in peers.items():
            for k, v in m.items():
                if v is None:
                    dead[k] += 1
        never = {k for k, v in dead.items() if v == len(peers)}
        print(f"\nchannels dead on ALL chips (unused by design): {sorted(never)}")
        print("chips with a free port pair beyond those:")
        for ch in sorted(peers, key=lambda k: coord[k]):
            free = sorted(k for k, v in peers[ch].items() if v is None and k not in never)
            if free:
                t, a = pos[ch]
                print(f"  chip {ch:>2}  mesh{coord[ch]}  tray{t}.asic{a}  free channels {free}")


main()
