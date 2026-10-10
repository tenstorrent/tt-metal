"""Static NoC link-load model of flat_combine_overlap on a Blackhole p150 (LoudBox, harvest mask 192).

Bytes per layer on every directed link, per NoC and per flow class; routes on the 17 x 12 physical torus:
NoC 0 +x then +y, NoC 1 -y then -x (the y-then-x order seen in the unified study's NoC traces). A read's data
travels from the target back to the requester on the requester's NoC; a write's from source to destination.
Multicasts: the union of the unicast routes. DRAM: each bank exposes one port per NoC (soc descriptor
worker_endpoint). Interleaved pages spread evenly over the 8 banks.
"""
import json, sys
from collections import defaultdict

GX, GY = 17, 12
LX = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]  # logical x -> physical x (cols 7, 10 harvested, 8 L2CPU, 9 DRAM)


def phys(c):
    return (LX[c[0]], c[1] + 2)


# bank -> (NoC0 port, NoC1 port), from blackhole_140_arch.yaml dram + dram_views worker_endpoint
DRAM_PORTS = [[(0, 0), (0, 1), (0, 11)], [(0, 2), (0, 10), (0, 3)], [(0, 9), (0, 4), (0, 8)], [(0, 5), (0, 7), (0, 6)],
              [(9, 0), (9, 1), (9, 11)], [(9, 2), (9, 10), (9, 3)], [(9, 9), (9, 4), (9, 8)], [(9, 5), (9, 7), (9, 6)]]
WEP = [(2, 1), (0, 1), (0, 1), (0, 1), (2, 1), (2, 1), (2, 1), (2, 1)]
BANK = [(DRAM_PORTS[b][WEP[b][0]], DRAM_PORTS[b][WEP[b][1]]) for b in range(8)]


def route(src, dst, noc):
    """Directed links (x, y, dir) from src to dst."""
    (x, y), (tx, ty) = src, dst
    links = []
    if noc == 0:
        while x != tx:
            links.append((x, y, "E")); x = (x + 1) % GX
        while y != ty:
            links.append((x, y, "S")); y = (y + 1) % GY
    else:
        while y != ty:
            links.append((x, y, "N")); y = (y - 1) % GY
        while x != tx:
            links.append((x, y, "W")); x = (x - 1) % GX
    return links


class Load:
    def __init__(self):
        self.l = defaultdict(lambda: defaultdict(float))  # (noc, link) -> flow -> bytes

    def write(self, flow, src, dst, noc, b):
        for k in route(src, dst, noc):
            self.l[(noc, k)][flow] += b

    def read(self, flow, req, tgt, noc, b):  # data comes back from tgt to req on req's noc
        self.write(flow, tgt, req, noc, b)

    def mcast(self, flow, src, dsts, noc, b):
        u = set()
        for d in dsts:
            u.update(route(src, d, noc))
        for k in u:
            self.l[(noc, k)][flow] += b

    def to_banks(self, flow, src, noc, b, read=False):
        for bk in range(8):
            port = BANK[bk][noc]
            (self.read if read else self.write)(flow, src, port, noc, b / 8)


def flat_load(L, p, T, H, I, E, w_tile=576, h_tile=1088):
    It, Ht = I // 32, H // 32
    gu, rd, dn, rl = [phys(c) for c in p["gu"]], [phys(c) for c in p["readers"]], [phys(c) for c in p["down"]], \
        [phys(c) for c in p["relays"]]
    r_ = len(gu) // len(rd)
    gu_bytes = 2 * It * Ht * w_tile * E  # gate + up, all experts
    for r, c in enumerate(rd):
        b = gu_bytes / len(rd)
        L.read("W gu read", c, BANK[r % 8][0], 0, b)
        for j in range(r_):
            L.write("W gu fwd", c, gu[r * r_ + j], 1, b / r_)
    # down weights: down core d, its pcd columns, bank d % 8 (NoC0); reader tails pcd_r columns, bank i % 8
    pcds = p["pcds"]
    for d, c in enumerate(dn):
        L.read("W down read", c, BANK[d % 8][0], 0, It * pcds[d] * w_tile * E)
    tails = [rd[r] for r, _ in p["rdn"]]
    for i, c in enumerate(tails):
        L.read("W down read", c, BANK[i % 8][0], 0, It * p["pcd_r"] * w_tile * E)
    # x: each rectangle's primary relay (relays[k]) multicasts all of x to rectangle k; primaries and helpers read
    # half of x each from DRAM (NoC1), helpers write their half to the primary (NoC0)
    xb = T * H * 2
    nr = 2
    rects = [[], []]
    for c in p["gu"]:
        rects[0 if c[0] <= 5 else 1].append(phys(c))
    for k in range(nr):
        prim = rl[k]
        helpers = rl[nr + k::nr] if len(rl) > nr else []
        share = xb / (1 + len(helpers))
        L.to_banks("x read", prim, 1, share, read=True)
        for h in helpers:
            L.to_banks("x read", h, 1, share, read=True)
            L.write("x help", h, prim, 0, share)
        L.mcast("x mcast", prim, rects[k], 0, xb * 1088 / 2048)  # tilized bfp8 x
    # h: every gu core writes its slice to every chain head (NoC1); chains forward whole h (NoC1)
    hb = T * I * h_tile / 1024
    succ = {a: b for a, b in p["d_succ"]}
    heads = [d for d in range(len(dn)) if d not in set(succ.values())]
    for g in gu:
        for hd in heads:
            L.write("h gather", g, dn[hd], 1, hb / len(gu))
    for a, b in succ.items():
        L.write("h chain", dn[a], dn[b], 1, hb)
    for r, t in p["rdn"]:
        L.write("h chain", dn[t], rd[r], 1, hb)
    # y: down cores' columns on NoC1 (VC2), reader tails' on NoC0, rows spread over the banks
    for d, c in enumerate(dn):
        L.to_banks("y write", c, 1, T * pcds[d] * 64)
    for c in tails:
        L.to_banks("y write", c, 0, T * p["pcd_r"] * 64)


def combine_load(L, senders, T, H, hops_fwd=1.125):
    """senders: logical cores of combine's reader / sender streams (one eth core directly above each)."""
    tok = H * 2
    S = [phys(c) for c in senders]
    eth = [(x, 1) for x, _ in S]
    n = len(S)
    own = T * 7 / 8 * tok  # own tokens that leave the chip
    fwd = T * (hops_fwd) * (tok + 64)  # tokens forwarded through this chip (fwd buffer)
    fin = T * 7 / 8 * tok  # tokens arriving here as their final destination
    loc = T / 8 * tok
    for s, e in zip(S, eth):
        L.to_banks("C y read", s, 0, own / n, read=True)
        L.to_banks("C fwd read", s, 0, fwd / n, read=True)
        L.write("C send", s, e, 1, (own + fwd) / n)
        L.to_banks("C local", s, 0, loc / n, read=True)
        L.to_banks("C local", s, 0, loc / n)
        L.to_banks("C eth->fwd", e, 1, fwd / n)  # fabric local writes: NoC1, VC 2/3
        L.to_banks("C eth->out", e, 1, fin / n)


def report(L, us, top=14, title=""):
    cap = 64 * 1.35e9 * us * 1e-6  # bytes a link carries in `us`
    rows = []
    for (noc, k), f in L.l.items():
        rows.append((sum(f.values()), noc, k, f))
    rows.sort(key=lambda r: -r[0])
    print(f"--- {title}: top links, % of one link's capacity over {us} us ({cap / 1e6:.0f} MB)")
    for tot, noc, (x, y, d), f in rows[:top]:
        parts = ", ".join(f"{k} {v / 1e6:.1f}" for k, v in sorted(f.items(), key=lambda kv: -kv[1]) if v > 0.3e6)
        print(f"  NoC{noc} ({x:2d},{y:2d}){d}  {tot / 1e6:6.1f} MB {100 * tot / cap:5.0f}%   [{parts}]")
    return rows


if __name__ == "__main__":
    plan = json.load(open(sys.argv[1]))
    H, I, E, T, us = (int(a) for a in sys.argv[2:7])
    senders = [(2, 0), (6, 0), (7, 0), (8, 0)]
    Lf = Load(); flat_load(Lf, plan, T, H, I, E)
    report(Lf, us, title="flat alone")
    Lc = Load(); combine_load(Lc, senders, T, H)
    report(Lc, us, 8, title="combine alone")
    Lb = Load(); flat_load(Lb, plan, T, H, I, E); combine_load(Lb, senders, T, H)
    rows = report(Lb, us, 20, title="flat + combine")
    # where combine's bytes sit on links flat already loads heavily
    print("--- combine flows on links where flat carries > 25 MB")
    agg = defaultdict(float)
    for tot, noc, k, f in rows:
        flat_b = sum(v for kk, v in f.items() if not kk.startswith("C "))
        if flat_b > 25e6:
            for kk, v in f.items():
                if kk.startswith("C "):
                    agg[(noc, kk)] += v
    for (noc, kk), v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  NoC{noc} {kk}: {v / 1e6:.1f} MB summed over those links")
