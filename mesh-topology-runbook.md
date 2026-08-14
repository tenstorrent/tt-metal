# Mesh topology runbook

How to find the physical layout of a TT box, which mesh axis is X vs Y, and whether
either axis can actually form a ring. Written 2026-08-14 against
`docker-bh-glx-110-a10u14` (Blackhole Galaxy, 32 chips).

## Conventions

| mesh dim | fabric name | direction | a.k.a. |
|---|---|---|---|
| dim 0 | NS | N / S | **Y**, rows |
| dim 1 | EW | E / W | **X**, cols |

Source: `mesh_graph_descriptor.cpp` (`mesh_ns_size = dims(0)`, `mesh_ew_size = dims(1)`), and the
torus descriptors — `*_torus_x_*` is `dim_types: [LINE, RING]` (RING on dim 1), `*_torus_y_*` is
`[RING, LINE]`.

## Findings on this box

Physical mesh is **8 x 4** (`single_bh_galaxy_mesh_graph_descriptor.textproto`,
`device_topology { dims: [8, 4] }`). Rows span tray pairs, cols sit within a tray pair:

```
             col0    col1  |  col2    col3
  r0..r3    tray 1         |        tray 2
  r4..r7    tray 3         |        tray 4
```

**Two inter-tray cables are not linking**, versus the repo's golden reference
`tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/`:

| | reference | live |
|---|---|---|
| intra-box chip pairs | 68 | 66 |
| degree histogram | `{4:24, 5:8}` | `{3:4, 4:20, 5:8}` |
| asic1 inter-tray cables | 4 | 3 |
| asic2 inter-tray cables | 2 | 1 |
| intra-tray wiring | — | identical |

- **tray3.asic2 <-> tray4.asic2** — ports `ch0/1`, chips 25 and 17 -> physical **row 6** X ring open
- **tray2.asic1 <-> tray4.asic1** — ports `ch2/3`, chips 8 and 16 -> physical **col 3** Y ring open

All four ports report `DOWN/unconnected` with 0 retrains, and each has that port pair free while its
counterparts elsewhere in the box use it for exactly this cable. Software cannot distinguish cable
absent / unseated / failed / port disabled — 0 retrains only means the link never trained.

Note asic2 and asic6 having just 2 inter-tray cables is **by design** (the reference shows the same);
they spend a port pair on off-box scale-out. Only the asic1 and asic2 deltas above are real.

Consequence: neither axis gives uniform rings. CP=4 -> 7 of 8 groups are rings; CP=8 -> 3 of 4.

## Procedure

Run on the target machine, from the repo root, with the devices free
(`fuser -v /dev/tenstorrent/*` should be empty).

**1. Populate the control-plane dumps.** Any run that opens a mesh device writes these
automatically (`control_plane.cpp`, no flags needed):

```bash
python3 -c "import ttnn; d=ttnn.open_mesh_device(); print(list(d.shape)); ttnn.close_mesh_device(d)"
ls generated/fabric/
#   physical_chip_mesh_coordinate_mapping_1_of_1.yaml   chip -> [row, col]
#   asic_to_fabric_node_mapping_rank_1_of_1.yaml        chip -> tray / asic_location
```

Opening with no shape gives the true system shape — that is the physical mesh.

**2. Dump live ethernet topology and per-port link state.**

```bash
./build_Release/tools/umd/topology       # prints path to a cluster_descriptor.yaml
./build_Release/tools/umd/system_health > health.txt
```

**3. Analyse.** Run the script at the end of this file:

```bash
python3 mesh_topology_report.py /tmp/umd_XXXX/cluster_descriptor.yaml health.txt
```

It prints the physical shape, the coord -> chip/tray.asic table, ring completeness per axis, the
inter-tray cable census, a diff against every matching golden reference descriptor, and which chips
have a free port pair.

**4. Optional — exercise the links that are up.**

```bash
./build_Release/tools/scaleout/run_cluster_validation \
    --print-connectivity --send-traffic --log-ethernet-metrics \
    --output-path generated/cluster_validation
```

Passing `--cabling-descriptor-path` or `--factory-descriptor-path` additionally enables spec
validation; without one it only does discovery plus traffic. No BH Galaxy cabling descriptor ships
in the repo, so the step-3 diff against the golden cluster descriptor covers that ground instead.

## What each signal confirms

| signal | meaning |
|---|---|
| system mesh shape from step 1 | the real physical grid; anything else you request is derived from it |
| ring completeness, step 3 | whether a wrap link physically exists per row / per col |
| cable diff vs golden reference | separates by-design gaps from genuinely missing cables |
| free port pair at **both** ends | a cable could land there — supports "missing" over "by design" |
| `retrain: 0` on a DOWN port | link never trained (vs. trained then dropped) |
| clean `--send-traffic` on up links | the rest of the fabric is healthy; problem is localised |

## Gotchas

- **Requesting a shape that isn't the system shape silently transposes it.** On an 8x4 system,
  `MeshShape(4, 8)` succeeds: `SystemMesh::get_mapped_devices` rotates until it fits, then maps
  logical `(i,j)` -> system `(j,i)`. Adjacency is preserved (clean transpose, not a snake) but the
  axes swap — logical dim 0 becomes physical X, logical dim 1 becomes physical Y. Open the mesh at
  the system shape and map CP/SP explicitly to avoid this.
- **Default auto-discovery builds LINE x LINE — no rings at all.** You need
  `TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/<..._torus_*>.textproto`.
- **A torus MGD opens successfully even when the wrap cables are missing.** The topology mapper does
  not validate wrap edges — the logical adjacency histogram is identical for LINE x LINE and
  RING x RING. A successful open is *not* evidence the ring works. Verify with step 3.
- Channels `10/11` are dead on all 32 chips here — unused by design, not a fault.

## Appendix: mesh_topology_report.py

Ships alongside this doc as `./mesh_topology_report.py` — run it from the repo root. Inlined here so
this file stands alone if you carry it to another machine.

```python
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
```

Expected output on this box:

```
physical mesh = 8 x 4   (dim0 = Y = NS = rows, dim1 = X = EW = cols)
intra-box chip pairs = 66, ethernet channels = 136
...
  X rings (dim1, len 4, one per row): 7/8 closed   OPEN rows: [6]
  Y rings (dim0, len 8, one per col): 3/4 closed   OPEN cols: [3]
inter-tray asicN<->asicN cables (live): {1: 3, 2: 1, 3: 4, 4: 4, 5: 4, 6: 2, 7: 4, 8: 4}
vs reference single_bh_galaxy_clus_desc.yaml: pairs 68 -> 66
  inter-tray asic cable diff (ref, live): {1: (4, 3), 2: (2, 1)}
  intra-tray cable diff      (ref, live): identical
```

On a healthy box both ring lines read fully closed and both diffs read `identical`.

Chips `3, 11, 19, 27` (the four `tray*.asic4`) also show a free `ch0/1` pair — that is normal, the
reference has the same asic4 cable count.
