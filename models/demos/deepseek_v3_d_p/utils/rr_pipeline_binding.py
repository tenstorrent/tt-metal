# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Rank bindings for [RING,RING] 4x4 prefill pipelines: in-Galaxy tori alternating with two-host seam tori.

A Galaxy holds one torus of its own (ASICs {3,4,7,8} on all four trays, slices 1+2) and two halves of
seam tori (ASICs {1,2,5,6}, slice 3 on trays 2/4 or 3/4 and slice 0 on trays 1/3 or 1/2). Stage 2i is
host i's own torus; stage 2i+1 joins host i's slice 3 to host i+1's slice 0 over the inter-Galaxy cables.
So a pipeline of N stages on hosts h0, h1, ... in cabling ring order has one rank for each even stage and
two ranks (mesh_host_rank 0 on host i, 1 on host i+1) for each odd one. Ranks are ordered mesh-major,
host-rank-minor, as control_plane assigns them. Slice composition and the rank table live in
tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py (shared with the blitz decode pipelines).

    generate  on hardware: discover the 2x4 slice -> device map with one local rank (needs a build with
              test_physical_discovery) and fill the binding template. Every Galaxy of a pod is wired
              identically, so one host's map serves all.
    predict   from cabling descriptors (tt-cluster-descriptors submodule), no devices needed: fill the
              binding, and report per stage whether its 16 chips hold a C4xC4 torus at 2 links (or 1) per
              edge and how many links each stage boundary has. Picks the descriptor: the 2-link MGD when
              every torus has 2-link edges, else the 1-link variant.

Both write a rankfile (`rank i=host slot=<n>`, n counting the ranks on that host), the placement the binding needs: tt-run takes no host
field, so launch with `--mpi-args "--map-by rankfile:file=<rankfile>"`.
"""
import argparse
import collections
import importlib.util
import itertools
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, os.environ.get("TT_METAL_HOME", str(REPO)))
from models.demos.deepseek_v3_d_p.utils.assemble_pipeline_binding import SC4_MAPPING, Galaxy, load_mapping  # noqa: E402
from models.demos.deepseek_v3_d_p.utils.gen_pipeline_binding import TOPO, fill_rank_bindings  # noqa: E402


def _load_rank_table_module():
    path = REPO / "tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py"
    spec = importlib.util.spec_from_file_location("generate_rank_bindings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grb = _load_rank_table_module()

TEMPLATE_SC4 = TOPO / "pipeline_prefill_request_sc4_rr_12rank_4x4_sp1.yaml"
TEMPLATE_SC2 = TOPO / "pipeline_prefill_request_sc2_rr_4rank_4x4_sp1.yaml"
TORUS_SIDE = 4
# UBB tray id = index of the bus high nibble + 1 (control_plane.cpp ubb_bus_ids for Blackhole).
_BH_UBB_TRAY_BUS = (0x00, 0x40, 0xC0, 0x80)


# --- descriptor model -----------------------------------------------------------------------------


def chip_position(g, chip):
    """(tray_id, asic_location) of a chip, as the MGD's physical_asic_position names them."""
    return _BH_UBB_TRAY_BUS.index(g.bus[chip] & 0xF0) + 1, g.bus[chip] & 0x0F


def slice_chips(g, slice_id):
    """The 8 chips of a 2x4 slice of galaxy g, as descriptor chip ids."""
    by_position = {chip_position(g, c): c for c in g.bus}
    return [by_position[pos] for pos in grb.slice_positions(slice_id, g.rev_c)]


def slice_logical_ids(g):
    """slice id -> TT_VISIBLE_DEVICES ids (UMD logical ids, BDF-sorted indices) of g."""
    return {s: sorted(g.logical[c] for c in slice_chips(g, s)) for s in grb.SLICE_ASIC_LOCATIONS}


def link_counts(galaxies):
    """{frozenset of (host index, chip)} -> links, over intra-host and inter-host cables."""
    owner = {u: (gi, c) for gi, g in enumerate(galaxies) for c, u in g.unique.items()}
    counts = collections.Counter()
    for gi, g in enumerate(galaxies):
        for pair, n in g.links.items():
            counts[frozenset((gi, c) for c in pair)] += n
    # An inter-host cable is listed by both of its hosts; count it once, as the larger of the two lists.
    remote = {}
    for gi, g in enumerate(galaxies):
        for (chip, remote_unique), n in g.remote.items():
            if remote_unique not in owner:
                continue
            key = frozenset(((gi, chip), owner[remote_unique]))
            remote[key] = max(remote.get(key, 0), n)
    counts.update(remote)
    return counts


def stage_chips(galaxies, table, host_index):
    """mesh id -> set of (host index, chip), from the rank table's (mesh, host_rank, host, slices) rows.

    host_index maps a host name to the index link_counts() used for it."""
    by_name = {g.name: g for g in galaxies}
    stages = collections.defaultdict(set)
    for mesh_id, _, host, slices in table:
        stages[mesh_id].update((host_index[host], c) for s in slices for c in slice_chips(by_name[host], s))
    return stages


def find_torus(chips, counts, min_links, side=TORUS_SIDE):
    """A side x side torus on `chips` using only edges of >= min_links links: grid[r][c] chip, or None.

    Backtracking over grid cells in row-major order. A torus is vertex-transitive, so the first cell can
    be any chip. Each cell must be cabled to its already placed left and upper neighbours, and the last
    column and row close the wrap."""
    adjacent = collections.defaultdict(set)
    for pair, n in counts.items():
        if n >= min_links and pair <= chips:
            a, b = tuple(pair)
            adjacent[a].add(b)
            adjacent[b].add(a)
    cells = [(r, c) for r in range(side) for c in range(side)]
    grid = {}

    def place(i, used):
        if i == len(cells):
            return True
        r, c = cells[i]
        wanted = []
        if c:
            wanted.append(grid[(r, c - 1)])
        if r:
            wanted.append(grid[(r - 1, c)])
        for chip in sorted(set.intersection(*(adjacent[w] for w in wanted)) - used):
            grid[(r, c)] = chip
            closes = []
            if c == side - 1:
                closes.append(grid[(r, 0)])
            if r == side - 1:
                closes.append(grid[(0, c)])
            if all(w in adjacent[chip] for w in closes) and place(i + 1, used | {chip}):
                return True
            del grid[(r, c)]
        return False

    for first in sorted(chips):
        grid[(0, 0)] = first
        if place(1, {first}):
            return [[grid[(r, c)] for c in range(side)] for r in range(side)]
    return None


def torus_min_links(grid, counts, side=TORUS_SIDE):
    return min(
        counts[frozenset((grid[r][c], grid[(r + dr) % side][(c + dc) % side]))]
        for r in range(side)
        for c in range(side)
        for dr, dc in ((0, 1), (1, 0))
    )


def best_torus(chips, counts):
    """(grid, min links on its edges) for the largest k in 1..4 that has a torus, or (None, 0)."""
    for k in (4, 3, 2, 1):
        grid = find_torus(chips, counts, k)
        if grid:
            return grid, torus_min_links(grid, counts)
    return None, 0


def boundary_links(a, b, counts):
    return sum(n for pair, n in counts.items() if len(pair & a) == 1 and len(pair & b) == 1 and pair <= a | b)


def ring_orders(galaxies, num_stages):
    """Host orders whose stages all hold a torus: [(min torus edge links, order)], best first.

    The order sets which host pairs form seam stages, and the cabling is directional: host i's slice 3 is
    cabled to host i+1's slice 0. A full ring (2 stages per host) is the same under rotation, so its first
    host stays fixed; a chain tries every order."""
    cache = {}

    def torus_links(chips):
        key = frozenset(chips)
        if key not in cache:
            cache[key] = best_torus(key, counts)[1]
        return cache[key]

    counts = link_counts(galaxies)
    host_index = {g.name: i for i, g in enumerate(galaxies)}
    found = []
    full_ring = num_stages == 2 * len(galaxies)
    indices = range(len(galaxies))
    orders = (
        [(0, *rest) for rest in itertools.permutations(indices[1:])] if full_ring else itertools.permutations(indices)
    )
    for permutation in orders:
        order = [galaxies[i] for i in permutation]
        table = grb.build_split_ring_rank_table([g.name for g in order], num_stages)
        links = min(torus_links(chips) for chips in stage_chips(order, table, host_index).values())
        if links:
            found.append((links, order))
    return sorted(found, key=lambda f: -f[0])


def stage_report(galaxies, table):
    """(per-stage [(mesh id, kind, hosts, torus min links)], per-boundary [(from, to, links)])."""
    counts = link_counts(galaxies)
    stages = stage_chips(galaxies, table, {g.name: i for i, g in enumerate(galaxies)})
    hosts_of = collections.defaultdict(list)
    for mesh_id, _, host, _ in table:
        hosts_of[mesh_id].append(host)
    rows = []
    for mesh_id in sorted(stages):
        _, min_links = best_torus(stages[mesh_id], counts)
        kind = "in-Galaxy torus" if len(hosts_of[mesh_id]) == 1 else "seam torus"
        rows.append((mesh_id, kind, hosts_of[mesh_id], min_links))
    boundaries = [
        (m, m + 1, boundary_links(stages[m], stages[m + 1], counts)) for m in sorted(stages) if m + 1 in stages
    ]
    return rows, boundaries


# --- binding --------------------------------------------------------------------------------------


def default_template(num_stages):
    templates = {8: TEMPLATE_SC4, 3: TEMPLATE_SC2}
    if num_stages not in templates:
        raise SystemExit(f"no binding template for {num_stages} stages; pass --template")
    return templates[num_stages]


def single_link_mgd(path):
    return path.replace("_mgd.textproto", "_1link_mgd.textproto")


def render(template, rank_table, source, hosts, mgd=None):
    """Fill the template's TT_VISIBLE_DEVICES for rank_table (one dict per rank, in rank order)."""
    text = Path(template).read_text()
    if mgd:
        old = next(l for l in text.splitlines() if l.startswith("mesh_graph_desc_path: "))
        text = text.replace(old, f"mesh_graph_desc_path: {mgd}")
    mesh_ranks = [(r["mesh_id"], r["host_rank"]) for r in rank_table]
    text = fill_rank_bindings(text, [r["devices"].split(",") for r in rank_table], template, mesh_ranks=mesh_ranks)
    placement = ", ".join(f"{i}={r['host']}" for i, r in enumerate(rank_table))
    banner = (
        f"# GENERATED by rr_pipeline_binding.py ({source}) -- do not hand-edit the TT_VISIBLE_DEVICES lists.\n"
        f"# Host ring order: {', '.join(hosts)}\n"
        f"# Rank -> host: {placement}\n"
    )
    return banner + text


def rankfile(rank_table):
    """One line per rank; ranks sharing a host take its slots 0, 1, ... in rank order."""
    seen = collections.Counter()
    lines = []
    for i, r in enumerate(rank_table):
        lines.append(f"rank {i}={r['host']} slot={seen[r['host']]}\n")
        seen[r["host"]] += 1
    return "".join(lines)


def mock_mapping(rank_table, descriptors):
    """A per-rank mock cluster mapping: each rank points at its host's descriptor."""
    lines = ["# GENERATED by rr_pipeline_binding.py: one entry per rank, in rank order.\n"]
    lines.append("rank_to_cluster_mock_cluster_desc:\n")
    for i, r in enumerate(rank_table):
        desc = descriptors[r["host"]]
        lines.append(f'  {i}: "{desc}"\n')
    return "".join(lines)


def predicted_rank_table(galaxies, num_stages):
    table = grb.build_split_ring_rank_table([g.name for g in galaxies], num_stages)
    by_name = {g.name: slice_logical_ids(g) for g in galaxies}
    ranks = []
    for entry in table:
        ranks += grb.resolve_rank_table_devices([entry], by_name[entry[2]], f"descriptor of {entry[2]}")
    return table, ranks


# --- CLI ------------------------------------------------------------------------------------------


def write_outputs(args, text, rank_table):
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out} ({len(rank_table)} ranks)")
    if args.rankfile_out:
        Path(args.rankfile_out).write_text(rankfile(rank_table))
        print(f"wrote {args.rankfile_out}")


def cmd_generate(args):
    hosts = [h for h in args.hosts.split(",") if h]
    rank_table = grb.resolve_split_ring_rank_table(
        hosts, args.stages, mpi_if=args.mpi_if, run_discovery=not args.no_discovery, work_dir=args.work_dir
    )
    template = args.template or default_template(args.stages)
    write_outputs(args, render(template, rank_table, "slice discovery", hosts, args.mgd), rank_table)
    return 0


def cmd_predict(args):
    descriptors = load_mapping(args.mapping)
    galaxies = [Galaxy(d) for d in descriptors]
    descriptor_of = {g.name: d for g, d in zip(galaxies, descriptors)}
    if args.host:
        by_name = {g.name: g for g in galaxies}
        missing = [h for h in args.host if h not in by_name]
        if missing:
            raise SystemExit(f"hosts {missing} are not in {args.mapping} (has {sorted(by_name)})")
        galaxies = [by_name[h] for h in args.host]
    num_stages = args.stages or 2 * len(galaxies)
    if args.print_order:
        counts = link_counts(galaxies)
        print(f"RR_CROSS_LINKS {sum(n for pair, n in counts.items() if len({host for host, _ in pair}) > 1)}")
    if args.auto_order:
        orders = ring_orders(galaxies, num_stages)
        if args.print_order and not orders:
            print("RR_ORDER none")
            return 3
        if args.print_order:
            print(f"RR_ORDER {' '.join(g.name for g in orders[0][1])} MIN_LINKS {orders[0][0]}")
        print("host orders with a torus on every stage (min edge links, order):")
        for links, order in orders:
            print(f"  {links}  {', '.join(g.name for g in order)}")
        if not orders:
            raise SystemExit("no host order gives a torus on every stage")
        galaxies = orders[0][1]
    table, rank_table = predicted_rank_table(galaxies, num_stages)
    hosts = [g.name for g in galaxies]
    revs = {g.rev_c for g in galaxies}
    if len(revs) != 1:
        raise SystemExit("hosts mix Rev C and Rev A/B boards")

    print(f"hosts (ring order): {', '.join(hosts)}   board revision: {'C' if revs.pop() else 'A/B'}")
    for i, r in enumerate(rank_table):
        print(f"rank {i:2d}  mesh {r['mesh_id']} host_rank {r['host_rank']}  {r['host']:20} {r['devices']}")
    rows, boundaries = stage_report(galaxies, table)
    print("\nstages (torus = C4xC4 on the stage's 16 chips; edge links = fewest links on any torus edge):")
    for mesh_id, kind, stage_hosts, min_links in rows:
        found = f"edges >= {min_links} link(s)" if min_links else "NO TORUS"
        print(f"  stage {mesh_id}  {kind:16} {'+'.join(stage_hosts):40} {found}")
    print("\nboundaries (links between consecutive stages):")
    for a, b, n in boundaries:
        print(f"  {a} -> {b}  {n}")
    if any(n == 0 for _, _, n in boundaries) or any(m == 0 for *_, m in rows):
        raise SystemExit("a stage has no torus or a boundary has no links: wrong host order for this cabling?")
    template = args.template or default_template(num_stages)
    mgd = args.mgd
    if not mgd:
        mgd = next(l.split(": ", 1)[1] for l in Path(template).read_text().splitlines() if l.startswith("mesh_graph_desc_path"))
        if min(m for *_, m in rows) < 2:
            mgd = single_link_mgd(mgd)
    print(f"\ndescriptor: {mgd}")
    print(f"launch: --mpi-args \"--map-by rankfile:file=<rankfile>\"")
    write_outputs(args, render(template, rank_table, f"predicted from {args.mapping}", hosts, mgd), rank_table)
    if args.mock_mapping_out:
        Path(args.mock_mapping_out).write_text(mock_mapping(rank_table, descriptor_of))
        print(f"wrote {args.mock_mapping_out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="fill the binding from a live slice discovery")
    g.add_argument("--hosts", required=True, help="comma-separated hosts in cabling ring order")
    g.add_argument("--mpi-if", help="network interface for the discovery mpirun")
    g.add_argument("--no-discovery", action="store_true", help="use slice_to_pcie_device_mapping.yaml in --work-dir")
    g.add_argument("--work-dir", type=Path)
    p = sub.add_parser("predict", help="fill the binding from cabling descriptors and report tori and seams")
    p.add_argument("--mapping", default=str(SC4_MAPPING), help="mock mapping yaml listing one descriptor per host")
    p.add_argument("--host", action="append", help="host to use, repeated in cabling ring order (default: the mapping's)")
    p.add_argument("--auto-order", action="store_true", help="search the host orders for the ring order")
    p.add_argument("--print-order", action="store_true", help="print RR_CROSS_LINKS and RR_ORDER lines for scripts; exit 3 if no order")
    p.add_argument("--mock-mapping-out", help="also write a per-rank mock cluster mapping here")
    for s in (g, p):
        s.add_argument("--stages", type=int, default=None, help="pipeline stages (default 2 per host)")
        s.add_argument("--template")
        s.add_argument("--mgd", help="mesh graph descriptor to name instead of the template's")
        s.add_argument("--out", help="binding to write")
        s.add_argument("--rankfile-out", help="rankfile to write")
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        if not args.stages:
            args.stages = 2 * len([h for h in args.hosts.split(",") if h])
        return cmd_generate(args)
    return cmd_predict(args)


if __name__ == "__main__":
    sys.exit(main())
