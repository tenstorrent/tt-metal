# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data parallelism on a Blackhole Galaxy: one QuietBox 2 replica (the 1 x 4 tensor-parallel gpt-oss-20b) per 2 x 2
ring of chips, one process per replica.

A QuietBox 2 is 4 Blackhole chips in a closed ring with 2 Ethernet links per edge; the gpt-oss CCLs (and the fused
decode / DFlash all-reduces) run their ring on it. A Galaxy's 32 chips form a grid with 2 links per edge, and any 2 x 2
square of it is such a ring (a plain row of 4 is not: no link closes it). find_rings tiles the Galaxy with disjoint
squares (8 of them), read from the cluster descriptor, and replica_env opens one process on one of them as a QuietBox 2:
TT_VISIBLE_DEVICES = its chips, the P300 x 2 mesh graph descriptor.

Each replica process is pinned to its own share of the host's cores, and its torch / OpenMP thread pools are capped to
that share: unpinned, 8 processes x 64 threads oversubscribe the host and every replica's DFlash rounds slow down
(measured on a 64-core host, 8 replicas of tests/test_dflash_batched.py users_32: 199-243 tok/s/user unpinned, 242.3 -
243.2 pinned, as one replica alone).

Nothing here opens a device: find_rings reads the cluster descriptor through tt_umd before any replica starts.
"""

import collections
import itertools
import os
from pathlib import Path

MESH_GRAPH_DESCRIPTOR = "tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"
RING_LINKS = 2  # Ethernet links per ring edge (a QuietBox 2's)


def chip_links(cluster_descriptor=None):
    """{chip: {neighbour: links}} from the cluster descriptor (tt_umd topology discovery when None)."""
    if cluster_descriptor is None:
        import tt_umd

        cluster_descriptor = tt_umd.TopologyDiscovery.create_cluster_descriptor()
    links = collections.defaultdict(collections.Counter)
    for chip, channels in cluster_descriptor.get_ethernet_connections().items():
        for remote, _ in channels.values():
            if remote != chip:
                links[chip][remote] += 1
    for chip in cluster_descriptor.get_all_chips():
        links.setdefault(chip, collections.Counter())
    return {chip: dict(n) for chip, n in links.items()}


def find_rings(links, ring_links=RING_LINKS):
    """Disjoint 2 x 2 rings covering every chip of `links` ({chip: {neighbour: links}}), each a sorted chip list, the
    rings sorted by their first chip; [] if the chips do not tile into rings (an edge counts with >= ring_links links).
    """
    adj = {c: {n for n, k in nbrs.items() if k >= ring_links} for c, nbrs in links.items()}
    squares = set()
    for a in adj:  # 4-cycles a - b - c - d - a
        for b, d in itertools.combinations(sorted(adj[a]), 2):
            for c in (adj[b] & adj[d]) - {a}:
                squares.add(frozenset((a, b, c, d)))
    by_chip = collections.defaultdict(list)
    for s in squares:
        for c in s:
            by_chip[c].append(s)

    def cover(left):  # exact cover by backtracking from the chip in the fewest squares
        if not left:
            return []
        chip = min(left, key=lambda c: sum(s <= left for s in by_chip[c]))
        for s in sorted(by_chip[chip], key=sorted):
            if s <= left:
                rest = cover(left - s)
                if rest is not None:
                    return [s] + rest
        return None

    rings = cover(frozenset(adj))
    return sorted((sorted(r) for r in rings), key=lambda r: r[0]) if rings else []


def cpu_shares(n, cpus=None):
    """n disjoint, equal, contiguous shares of the host cores this process may run on."""
    cpus = sorted(os.sched_getaffinity(0) if cpus is None else cpus)
    k = len(cpus) // n
    assert k >= 1, f"{len(cpus)} cores for {n} replicas"
    return [cpus[i * k : (i + 1) * k] for i in range(n)]


def replica_env(ring, cpus, base_env=None, tt_metal_home=None):
    """The environment of the replica process on `ring` (its chips) pinned to `cpus`."""
    env = dict(os.environ if base_env is None else base_env)
    home = Path(tt_metal_home or env.get("TT_METAL_HOME") or Path(__file__).parents[4])
    threads = str(len(cpus))
    env.update(
        TT_VISIBLE_DEVICES=",".join(str(c) for c in ring),
        TT_MESH_GRAPH_DESC_PATH=str(home / MESH_GRAPH_DESCRIPTOR),
        OMP_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads,
        TORCH_NUM_THREADS=threads,
    )
    return env


def pin(cpus):
    """A subprocess preexec_fn pinning the child to `cpus` (before it imports torch / ttnn)."""
    cpus = set(cpus)
    return lambda: os.sched_setaffinity(0, cpus)
