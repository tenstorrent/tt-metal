#!/usr/bin/env python3
"""mk_flow_yamls.py: X1 of the slow-state checks (approved 2026-11-05; yaml only): one flow at a time. The test infra's
sequential_neighbor_exchange pattern runs one neighbor pair per iteration (one sender, one direction; every directed
neighbor pair of the mesh in turn), on the NeighborExchange topology (the NX rows' router images): unicast_write,
2048 and 4096 B, num_links 1 and 2, 100k packets, 8 repeats. (Explicit unicast senders on the NeighborExchange
topology crash in ControlPlane::get_chip_neighbors on the mock, so the pairs come from the test infra.)"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def test(name):
    return f"""- name: {name}
  benchmark_mode: true
  sync: true
  fabric_setup:
    topology: NeighborExchange
  parametrization_params:
    num_links:
    - 1
    - 2
    ntype:
    - unicast_write
    size:
    - 2048
    - 4096
  defaults:
    ftype: unicast
    num_packets: 100000
  patterns:
  - type: sequential_neighbor_exchange
"""


open(os.path.join(HERE, "pcs_nx_x1_x8.yaml"), "w").write(
    "# cisp slow-state check X1 (mk_flow_yamls.py): one neighbor pair per iteration, 8 repeats.\n"
    "allocation_policies:\n  receiver:\n    policy: RoundRobin\n    max_configs_per_core: 4\nTests:\n"
    + "".join(test(f"SeqNeighborExchangeR{r:02d}") for r in range(1, 9)))


# X3 (coordinator / user 2026-11-05, yaml only): separate outbound-only from inbound-only concurrency on one chip.
# Subject chips at mesh [0,0] (chip 1) and [0,2] (chip 2), both seen slowing in X1. Per subject, num_links 1:
#   Lone  the subject sends one flow (E); nothing else runs
#   SO    the subject sends two flows (E and S); nothing inbound to it
#   SR    the subject sends S while its E neighbour sends W into it (one out, one in)
#   RR    ([0,2] only) its W and E neighbours send E / W into it; it sends nothing
# Senders by direction (destination hops: explicit device destinations crash on the NeighborExchange topology).
# unicast_write, 2048 and 4096 B, 100k packets, 3 iterations (top_level_iterations), 8 repeats.
def x3_test(name, senders):
    s = f"""- name: {name}
  benchmark_mode: true
  sync: true
  top_level_iterations: 3
  fabric_setup:
    topology: NeighborExchange
  parametrization_params:
    num_links:
    - 1
    ntype:
    - unicast_write
    size:
    - 2048
    - 4096
  defaults:
    ftype: unicast
    num_packets: 100000
  senders:
"""
    for (r, c), d in senders:
        s += f"""  - device: [{r}, {c}]
    patterns:
    - destination:
        hops:
          {d}: 1
"""
    return s


def x3_arms(sub):
    # N hops from row 1 crash on the mock (MeshGraph::chip_to_coordinate map::at), so inbound flows come from row-0
    # neighbours (E / W). [0,0] has one row-0 neighbour: no RR arm there.
    r, c = sub
    arms = {"Lone": [(sub, "E")], "SO": [(sub, "E"), (sub, "S")], "SR": [(sub, "S"), ((r, c + 1), "W")]}
    if c > 0:
        arms["RR"] = [((r, c - 1), "E"), ((r, c + 1), "W")]
    return arms


body3 = ""
for rep in range(1, 9):
    for tag, sub in (("C00", (0, 0)), ("C02", (0, 2))):
        for arm, snd in x3_arms(sub).items():
            body3 += x3_test(f"X3{arm}{tag}R{rep:02d}", snd)
open(os.path.join(HERE, "pcs_nx_x3_x8.yaml"), "w").write(
    "# cisp slow-state check X3 (mk_flow_yamls.py): outbound-only vs inbound concurrency on one chip, 8 repeats.\n"
    "allocation_policies:\n  receiver:\n    policy: RoundRobin\n    max_configs_per_core: 4\nTests:\n" + body3)


# X4 (coordinator 2026-11-05, observation only): SO with placed sender cores, and SO on other chips. num_links 1, as X3.
#   SOdef   E from logical (11,8) = noc0 (15,10), S from (11,7) = (15,9) (the allocator's own choice in X3)
#   SOswap  E from (11,7), S from (11,8): does the slowdown follow the core or the direction / router?
#   SOfar   E from (11,8), S from logical (5,2) = noc0 (6,4): its NoC0 path (column 6 down to the eth row) neither
#           shares nor crosses (15,10)'s (row 10 east, then column 4 for e5)
#   subjects [0,0] = chip 1 and [0,2] = chip 2 (all three arms), [0,1] = chip 0 and [0,3] = chip 3 (SOdef; [0,3] has no E
#   neighbour: W + S)
def x4_test(name, senders):
    s = x3_test(name, [])
    for (r, c), d, core in senders:
        s += f"""  - device: [{r}, {c}]
    core: [{core[0]}, {core[1]}]
    patterns:
    - destination:
        hops:
          {d}: 1
"""
    return s


A, B, FAR = (11, 8), (11, 7), (5, 2)
body4 = ""
for rep in range(1, 9):
    for tag, sub in (("C00", (0, 0)), ("C02", (0, 2))):
        body4 += x4_test(f"X4SOdef{tag}R{rep:02d}", [(sub, "E", A), (sub, "S", B)])
        body4 += x4_test(f"X4SOswap{tag}R{rep:02d}", [(sub, "E", B), (sub, "S", A)])
        body4 += x4_test(f"X4SOfar{tag}R{rep:02d}", [(sub, "E", A), (sub, "S", FAR)])
    body4 += x4_test(f"X4SOdefC01R{rep:02d}", [((0, 1), "E", A), ((0, 1), "S", B)])
    body4 += x4_test(f"X4SOdefC03R{rep:02d}", [((0, 3), "W", A), ((0, 3), "S", B)])
open(os.path.join(HERE, "pcs_nx_x4_x8.yaml"), "w").write(
    "# cisp slow-state check X4 (mk_flow_yamls.py): two outbound flows with placed sender cores, 8 repeats.\n"
    "allocation_policies:\n  receiver:\n    policy: RoundRobin\n    max_configs_per_core: 4\nTests:\n" + body4)


# X5 (coordinator / user 2026-11-05, observation only): one sender core against several for the same flows. The test
# infra packs every pattern of one `senders` entry onto one core (add_senders_from_pairs builds one sender per source
# device for NX; the sender allocation policy only packs whole sender entries, it cannot split one entry's patterns),
# so the arrangement is set by how the yaml groups patterns into entries:
#   SO1   [0,0] / [0,2]: one entry with E and S (one core, two configs: the benchmark round-robin loop, as NX)
#   SO2   the same flows as two entries (two cores, single-config stateful loops: X3's SO)
#   NX1   every chip to every neighbour (NX's flows), one entry per chip: one core per chip (as the stock NX rows)
#   NXs   the same flows, one entry per (chip, direction): one core per flow
# A yaml `device: [a, b]` is [mesh_id, chip_id] (row-major chip ids: [0, 0..3] top row, [0, 4..7] bottom row), not
# [row, col]: the earlier crashes on [1, c] (N hops, X1's explicit devices) were a mesh id of 1.
def x5_entry(dev, dirs):  # dev = (row, col) of the 2x4 mesh
    s = f"""  - device: [0, {dev[0] * 4 + dev[1]}]
    patterns:
"""
    for d in dirs:
        s += f"""    - destination:
        hops:
          {d}: 1
"""
    return s


def nx_flows():
    out = {}
    for r in (0, 1):
        for c in range(4):
            out[(r, c)] = ([] if c == 3 else ["E"]) + ([] if c == 0 else ["W"]) + (["S"] if r == 0 else ["N"])
    return out


body5 = ""
for rep in range(1, 9):
    for tag, sub in (("C00", (0, 0)), ("C02", (0, 2))):
        body5 += x3_test(f"X5SO1{tag}R{rep:02d}", []) + x5_entry(sub, ["E", "S"])
        body5 += x3_test(f"X5SO2{tag}R{rep:02d}", []) + x5_entry(sub, ["E"]) + x5_entry(sub, ["S"])
    fl = nx_flows()
    body5 += x3_test(f"X5NX1R{rep:02d}", []) + "".join(x5_entry(d, ds) for d, ds in fl.items())
    body5 += x3_test(f"X5NXsR{rep:02d}", []) + "".join(x5_entry(d, [x]) for d, ds in fl.items() for x in ds)
open(os.path.join(HERE, "pcs_nx_x5_x8.yaml"), "w").write(
    "# cisp slow-state check X5 (mk_flow_yamls.py): one sender core against one per flow, 8 repeats.\n"
    "allocation_policies:\n  receiver:\n    policy: RoundRobin\n    max_configs_per_core: 4\nTests:\n" + body5)
