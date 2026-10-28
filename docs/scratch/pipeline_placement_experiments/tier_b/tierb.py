# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Prototype of Tier B: slim checks vs full search + carve, on the repo's MGDs.

Not production code. Mirrors the plan in docs/scratch/mgd-pipeline-graph-design.md, 2.2.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from google.protobuf import text_format

import mesh_graph_descriptor_pb2 as pb

RING = pb.TorusTopology.RING


# ---------------------------------------------------------------- MGD flattening


@dataclass
class Mesh:
    idx: int
    desc: str
    dims: tuple
    ring: tuple
    host: tuple

    @property
    def slice(self):
        return tuple(d // h for d, h in zip(self.dims, self.host))

    @property
    def nslices(self):
        return math.prod(self.host)

    def key(self):
        return (self.desc, self.dims, self.ring, self.host)


@dataclass
class Fabric:
    meshes: list
    cap: dict  # (a, b) with a < b -> channels
    graph_level_conns: int = 0  # connections between sub-graphs, spread over every mesh pair

    def c(self, a, b):
        if a == b:
            return 0
        return self.cap.get((min(a, b), max(a, b)), 0)


def load_mgd(path) -> Fabric:
    m = pb.MeshGraphDescriptor()
    text_format.Parse(open(path).read(), m)
    mdesc = {d.name: d for d in m.mesh_descriptors}
    gdesc = {d.name: d for d in m.graph_descriptors}
    meshes: list[Mesh] = []
    cap: dict = {}
    fab = Fabric(meshes, cap)

    def meshes_under(node):
        if node[0] == "mesh":
            return [node[1]]
        return node[2]

    def add(a, b, count):
        ma, mb = meshes_under(a), meshes_under(b)
        if a[0] != "mesh" or b[0] != "mesh":
            fab.graph_level_conns += 1
        for x in ma:
            for y in mb:
                if x != y:
                    k = (min(x, y), max(x, y))
                    cap[k] = cap.get(k, 0) + count

    def inst(ref):
        kind = ref.WhichOneof("node_ref")
        if kind == "mesh":
            d = mdesc[ref.mesh.mesh_descriptor]
            dims = tuple(d.device_topology.dims)
            types = list(d.device_topology.dim_types) + [0] * len(dims)
            ring = tuple(t == RING for t in types[: len(dims)])
            host = tuple(d.host_topology.dims) or tuple(1 for _ in dims)
            mesh = Mesh(len(meshes), d.name, dims, ring, host)
            meshes.append(mesh)
            return ("mesh", mesh.idx)
        if kind == "graph":
            g = gdesc[ref.graph.graph_descriptor]
            members = []
            by_key = {}
            for sub in g.instances:
                node = inst(sub)
                k = sub.WhichOneof("node_ref")
                if k == "mesh":
                    by_key[("mesh", sub.mesh.mesh_descriptor, sub.mesh.mesh_id)] = node
                elif k == "graph":
                    by_key[("graph", sub.graph.graph_descriptor, sub.graph.graph_id)] = node
                members.append(node)
            allm = [x for n in members for x in meshes_under(n)]
            gnode = ("graph", ref.graph.graph_descriptor, allm, by_key)
            if g.HasField("graph_topology"):
                cnt = g.graph_topology.channels.count
                n = len(members)
                if g.graph_topology.layout_type == pb.GraphTopology.ALL_TO_ALL:
                    for i in range(n):
                        for j in range(i + 1, n):
                            add(members[i], members[j], cnt)
                else:
                    pairs = {tuple(sorted((i, (i + 1) % n))) for i in range(n) if n > 1}
                    for i, j in pairs:
                        add(members[i], members[j], cnt)
            for conn in g.connections:
                ends = [resolve(nr, by_key) for nr in conn.nodes]
                if len(ends) == 2 and all(e is not None for e in ends):
                    add(ends[0], ends[1], conn.channels.count)
            return gnode
        return ("switch", None, [])

    def resolve(nr, by_key):
        kind = nr.WhichOneof("node_ref")
        if kind == "mesh":
            return by_key.get(("mesh", nr.mesh.mesh_descriptor, nr.mesh.mesh_id))
        if kind == "graph":
            node = by_key.get(("graph", nr.graph.graph_descriptor, nr.graph.graph_id))
            if node is None:
                return None
            if nr.graph.HasField("sub_ref"):
                return resolve(nr.graph.sub_ref, node[3]) or node
            return node
        return None

    inst(m.top_level_instance)
    return fab


# ---------------------------------------------------------------- pipelines


@dataclass
class Pipeline:
    name: str
    shapes: list  # stage shape (tuple) per stage
    edges: list  # (src, dst) stage indices; one entry per copy


def need(shape, mesh: Mesh):
    """Slices a stage takes on mesh as a rect (per axis), or None."""
    if len(shape) != len(mesh.dims):
        return None
    rect = []
    for s, d, sl in zip(shape, mesh.dims, mesh.slice):
        if s > d or s % sl:
            return None
        rect.append(s // sl)
    return tuple(rect)


# ---------------------------------------------------------------- slim Tier B


def slim_check(fab: Fabric, pipes: list) -> tuple[bool, str]:
    stages = [(p, s, sh) for p in pipes for s, sh in enumerate(p.shapes)]
    bind = {}
    for p, s, sh in stages:
        b = {m.idx: math.prod(r) for m in fab.meshes if (r := need(sh, m)) is not None}
        if not b:
            return False, "binding"
        bind[(p.name, s)] = b
    total = sum(m.nslices for m in fab.meshes)
    if sum(min(b.values()) for b in bind.values()) > total:
        return False, "capacity"
    # Hall-style check per distinct binding set: stages confined to set B fit in B.
    for B in {frozenset(b) for b in bind.values()}:
        confined = [b for b in bind.values() if set(b) <= B]
        if sum(min(b.values()) for b in confined) > sum(fab.meshes[m].nslices for m in B):
            return False, "capacity-class"
    # Connectivity: a pipeline that fits on no single mesh needs a connected group of meshes.
    adj = {m.idx: set() for m in fab.meshes}
    for (a, b), c in fab.cap.items():
        if c > 0:
            adj[a].add(b)
            adj[b].add(a)
    for p in pipes:
        bs = [bind[(p.name, s)] for s in range(len(p.shapes))]
        common = set.intersection(*(set(b) for b in bs))
        if any(sum(b[m] for b in bs) <= fab.meshes[m].nslices for m in common):
            continue
        union = set().union(*bs)
        seen, ok = set(), False
        for start in union:
            if start in seen:
                continue
            comp, q = {start}, [start]
            while q:
                x = q.pop()
                for y in adj[x]:
                    if y in union and y not in comp:
                        comp.add(y)
                        q.append(y)
            seen |= comp
            if all(set(b) & comp for b in bs) and sum(min(v for m, v in b.items() if m in comp) for b in bs) <= sum(
                fab.meshes[m].nslices for m in comp
            ):
                ok = True
                break
        if not ok:
            return False, "connectivity"
    return True, ""


# ---------------------------------------------------------------- full Tier B


class Budget(Exception):
    pass


@dataclass
class Counter:
    limit: int
    nodes: int = 0

    def tick(self):
        self.nodes += 1
        if self.nodes > self.limit:
            raise Budget()


class Carver:
    def __init__(self, fab: Fabric):
        self.fab = fab
        self.memo = {}

    def grid(self, mesh: Mesh):
        H = mesh.host
        if len(H) == 1:
            H = (1, H[0])
            ring = (False, mesh.ring[0])
        else:
            ring = mesh.ring[:2]
        return H, ring

    def carve(self, mesh: Mesh, stages, rects, inner_edges, ctr: Counter) -> bool:
        key = (mesh.key(), tuple(sorted(stages)))
        if key in self.memo:
            return self.memo[key]
        H, ring = self.grid(mesh)
        R, C = H[0], H[-1] if len(H) > 1 else 1
        if len(H) > 2:
            self.memo[key] = True  # not modelled: pass
            return True

        def bit(r, c):
            return 1 << (r * C + c)

        def grow(mask):
            g = mask
            for r in range(R):
                for c in range(C):
                    if mask & bit(r, c):
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            rr, cc = r + dr, c + dc
                            if ring[0] and R > 2:
                                rr %= R
                            if ring[1] and C > 2:
                                cc %= C
                            if 0 <= rr < R and 0 <= cc < C:
                                g |= bit(rr, cc)
            return g

        pos = {}
        for s in stages:
            rr, cc = rects[s] if len(rects[s]) == 2 else (1, rects[s][0])
            opts = []
            for r0 in range(R - rr + 1):
                for c0 in range(C - cc + 1):
                    m = 0
                    for r in range(r0, r0 + rr):
                        for c in range(c0, c0 + cc):
                            m |= bit(r, c)
                    opts.append((m, grow(m)))
            pos[s] = opts
        nb = {s: [] for s in stages}
        for a, b in inner_edges:
            if a != b:
                nb[a].append(b)
                nb[b].append(a)
        # order: largest first, then BFS along inner edges
        order, seen = [], set()
        for start in sorted(stages, key=lambda s: -math.prod(rects[s])):
            if start in seen:
                continue
            q = [start]
            seen.add(start)
            while q:
                x = q.pop(0)
                order.append(x)
                for y in sorted(nb[x], key=lambda s: -math.prod(rects[s])):
                    if y not in seen:
                        seen.add(y)
                        q.append(y)
        area = {s: math.prod(rects[s]) for s in stages}
        placed = {}

        def dfs(i, occ, free):
            ctr.tick()
            if i == len(order):
                return True
            s = order[i]
            rest = sum(area[t] for t in order[i:])
            if rest > free:
                return False
            for m, g in pos[s]:
                if m & occ:
                    continue
                if any(t in placed and not (placed[t][1] & m) for t in nb[s]):
                    continue
                placed[s] = (m, g)
                if dfs(i + 1, occ | m, free - area[s]):
                    del placed[s]
                    return True
                del placed[s]
            return False

        ok = dfs(0, 0, R * C)
        self.memo[key] = ok
        return ok


def full_check(fab: Fabric, pipes: list, limit=10**6, carve=True) -> tuple[str, str, int]:
    """Returns (verdict in PASS/REJECT/UNDECIDED, reason, nodes)."""
    ok, why = slim_check(fab, pipes)  # steps 1 and 2 (and the cheap walk) first
    if not ok:
        return "REJECT", why, 0
    stages, rects_by_mesh, nbrs = [], {}, {}
    gid = 0
    edges = []
    for p in pipes:
        base = gid
        for sh in p.shapes:
            stages.append(sh)
            gid += 1
        for a, b in p.edges:
            edges.append((base + a, base + b))
    S, M = len(stages), len(fab.meshes)
    needm = [[None] * M for _ in range(S)]
    rect = [[None] * M for _ in range(S)]
    for s, sh in enumerate(stages):
        for m in fab.meshes:
            r = need(sh, m)
            if r is not None:
                rect[s][m.idx] = r
                needm[s][m.idx] = math.prod(r)
    nbrs = {s: {} for s in range(S)}
    for a, b in edges:
        if a != b:
            nbrs[a][b] = nbrs[a].get(b, 0) + 1
            nbrs[b][a] = nbrs[b].get(a, 0) + 1
    slices = [m.nslices for m in fab.meshes]
    # interchangeable classes: same key and same cap to every other mesh
    cls = list(range(M))
    for i in range(M):
        for j in range(i):
            if cls[j] != j:
                continue
            if fab.meshes[i].key() == fab.meshes[j].key() and all(
                fab.c(i, k) == fab.c(j, k) for k in range(M) if k not in (i, j)
            ):
                cls[i] = j
                break
    # stage order: most constrained first, BFS along edges
    dom = [sum(x is not None for x in needm[s]) for s in range(S)]
    biggest = [max((x for x in needm[s] if x is not None), default=0) for s in range(S)]
    order, seen = [], set()
    for start in sorted(range(S), key=lambda s: (dom[s], -biggest[s], s)):
        if start in seen:
            continue
        q = [start]
        seen.add(start)
        while q:
            x = q.pop(0)
            order.append(x)
            for y in sorted(nbrs[x], key=lambda s: (dom[s], -biggest[s], s)):
                if y not in seen:
                    seen.add(y)
                    q.append(y)
    assign = [None] * S
    used = [0] * M
    cross = {}
    carver = Carver(fab)
    ctr = Counter(limit)
    deepest = [0, ""]

    def leaf():
        if not carve:
            return True
        on = {}
        for s in range(S):
            on.setdefault(assign[s], []).append(s)
        for m, ss in on.items():
            sset = set(ss)
            inner = [(a, b) for a, b in edges if a in sset and b in sset]
            if not carver.carve(fab.meshes[m], ss, {s: rect[s][m] for s in ss}, inner, ctr):
                return False
        return True

    def lookahead(depth):
        free = sum(slices) - sum(used)
        rest = 0
        for s in order[depth:]:
            best = None
            for m in range(M):
                n = needm[s][m]
                if n is None or used[m] + n > slices[m]:
                    continue
                okc = True
                for t, k in nbrs[s].items():
                    if assign[t] is not None and assign[t] != m:
                        a, b = min(m, assign[t]), max(m, assign[t])
                        if cross.get((a, b), 0) + k > fab.c(a, b):
                            okc = False
                            break
                if okc:
                    best = n if best is None else min(best, n)
            if best is None:
                return False
            rest += best
        return rest <= free

    def dfs(depth):
        ctr.tick()
        if depth == S:
            return leaf()
        s = order[depth]
        holds = {assign[t] for t in nbrs[s] if assign[t] is not None}
        cand = [m for m in range(M) if needm[s][m] is not None]
        cand.sort(key=lambda m: (m not in holds, -(slices[m] - used[m]), m))
        for m in cand:
            if used[m] + needm[s][m] > slices[m]:
                continue
            if used[m] == 0 and any(cls[x] == cls[m] and x < m and used[x] == 0 for x in range(M)):
                continue
            added = []
            okc = True
            for t, k in nbrs[s].items():
                if assign[t] is not None and assign[t] != m:
                    a, b = min(m, assign[t]), max(m, assign[t])
                    if cross.get((a, b), 0) + k > fab.c(a, b):
                        okc = False
                        break
                    added.append(((a, b), k))
            if not okc:
                continue
            for p, k in added:
                cross[p] = cross.get(p, 0) + k
            assign[s] = m
            used[m] += needm[s][m]
            if lookahead(depth + 1) and dfs(depth + 1):
                return True
            used[m] -= needm[s][m]
            assign[s] = None
            for p, k in added:
                cross[p] -= k
        if depth > deepest[0]:
            deepest[0], deepest[1] = depth, f"stage {s}"
        return False

    try:
        ok = dfs(0)
    except Budget:
        return "UNDECIDED", "budget", ctr.nodes
    return ("PASS" if ok else "REJECT"), ("" if ok else "search: " + deepest[1]), ctr.nodes
