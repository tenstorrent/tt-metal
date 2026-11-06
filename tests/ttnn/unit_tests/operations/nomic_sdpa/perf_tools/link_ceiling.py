"""Link-load ceiling (perf-ceiling-dm Step 4b, L1 flavor) for nomic_sdpa's reads: every page read's response routed
dimension-ordered on a NoC torus over the logical 11x10 grid (approximation: harvested / non-worker columns ignored),
bytes per link counted, ceiling = busiest link / 64 B/cycle. Banks: page % 110 -> logical core row-major (approx).
usage: python link_ceiling.py B S [noc_split]   noc_split: 'v1' sends V reads on NOC1 (-x/-y)."""
import sys
from collections import defaultdict
import importlib

m = importlib.import_module("ttnn.operations.nomic_sdpa.nomic_sdpa")
from ttnn.operations.nomic_sdpa.nomic_sdpa_program_descriptor import kv_multicast_topology, ROLE_RECEIVE, ROLE_SEND

B, S = int(sys.argv[1]), int(sys.argv[2])
split = sys.argv[3] if len(sys.argv) > 3 else ""
p = m.plan(B, S, m.KV_DEPTH_L1, True, m.Q_DEPTH_L1)
GX, GY = 11, 10
NB = GX * GY


class FakeDev:
    def worker_core_from_logical_core(self, c):
        return c


roles, _ = kv_multicast_topology(p, FakeDev())
load = [defaultdict(int), defaultdict(int)]


def route(src, dst, nbytes, noc):
    (x, y), (dx, dy) = src, dst
    step = 1 if noc == 0 else -1
    while x != dx:
        nx = (x + step) % GX
        load[noc][((x, y), (nx, y))] += nbytes
        x = nx
    while y != dy:
        ny = (y + step) % GY
        load[noc][((x, y), (x, ny))] += nbytes
        y = ny


def bank(page):
    b = page % NB
    return (b % GX, b // GX)


T = 2048
per_core = []
for c in range(p.num_cores):
    me = (c % GX, c // GX)
    nbytes = 0
    begin, end = p.core_row_start[c], p.core_row_start[c + 1]
    for g in range(begin, end):
        for dt in range(2):
            route(bank(2 * g + dt), me, T, 0)
            nbytes += T
    for k, pair in enumerate(p.core_pairs(c)):
        role = roles[c][k] if k < p.kv_depth else 0
        if role == ROLE_RECEIVE:
            continue
        for st in range(p.seq_tiles):
            for dt in range(2):
                pg = (pair * p.seq_tiles + st) * 2 + dt
                route(bank(pg), me, T, 0)
                nbytes += T
                route(bank(pg), me, T, 1 if split == "v1" else 0)
                nbytes += T
    per_core.append(nbytes)
busiest = max(max(l.values(), default=0) for l in load)
tot = sum(per_core)
print(
    f"{B}x{S} split={split or 'none'}: total read {tot/2**20:.1f} MiB, per-core max {max(per_core)/1024:.0f} KiB; "
    f"busiest link {busiest/1024:.0f} KiB -> link ceiling {busiest/64/1350:.1f} us @1350MHz (64 B/cyc); "
    f"per-core NIU ingress bound {max(per_core)/64/1350:.1f} us"
)
