"""NoC flow model of one Blackhole Galaxy chip running the EP-4 experts phase (ws_lane5, G 4 one-way lanes).

Every flow is routed on the physical 17 x 12 NoC torus: NOC0 east then south, NOC1 north then west (dimension order,
wrapping). A layout gives each flow its source, destination, NoC and rate (GB/s at the target phase time); the model
sums the rates on every directed link and reports the busiest links with what loads them, and per lane the busiest
link its weight traffic crosses.

Logical -> physical: the Tensix columns a chip has enabled (ENABLED_TENSIX_COL, bit i = NoC x HARVEST_X[i]),
logical column c = the c-th enabled one; logical row r = NoC y r + 2. Ethernet cores at y 1 (channel -> x BH_ETH_X).
DRAM: channel c's three endpoints (blackhole_140_arch.yaml), the NoC's endpoint by the legacy or relocated view.
"""

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

W, H = 17, 12
TENSIX_X = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]
HARVEST_X = [1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10]  # (harvest mask bit i -> NoC x: UMD HARVESTING_NOC_LOCATIONS)
DRAM = [
    [(0, 0), (0, 1), (0, 11)],
    [(0, 2), (0, 10), (0, 3)],
    [(0, 9), (0, 4), (0, 8)],
    [(0, 5), (0, 7), (0, 6)],
    [(9, 0), (9, 1), (9, 11)],
    [(9, 2), (9, 10), (9, 3)],
    [(9, 9), (9, 4), (9, 8)],
    [(9, 5), (9, 7), (9, 6)],
]
# (NOC0 endpoint index, NOC1 endpoint index) per channel
EP_LEGACY = [(2, 1), (0, 1), (0, 1), (0, 1), (2, 1), (2, 1), (2, 1), (2, 1)]
EP_RELOC = [(0, 2), (1, 2), (1, 2), (1, 2), (2, 0), (2, 0), (2, 0), (2, 0)]
LINK_GBS = 64 * 1.35  # (a NoC link: 64 B a cycle at 1350 MHz)


def route(noc, s, d):
    """Directed links (x, y, dir) from s to d: NOC0 'E' then 'S', NOC1 'N' then 'W'."""
    (x, y), (dx, dy) = s, d
    out = []
    if noc == 0:
        while x != dx:
            out.append((0, x, y, "E"))
            x = (x + 1) % W
        while y != dy:
            out.append((0, x, y, "S"))
            y = (y + 1) % H
    else:
        while y != dy:
            out.append((1, x, y, "N"))
            y = (y - 1) % H
        while x != dx:
            out.append((1, x, y, "W"))
            x = (x - 1) % W
    return out


@dataclass
class Chip:
    mask: int = 0x3FFF  # ENABLED_TENSIX_COL
    reloc: bool = False
    eth_out: tuple = (3, 2, 14, 15)  # lane -> its ethernet core's NoC x (lanes 0 / 1 link 0 fwd / rev, 2 / 3 link 1)

    def cols(self):
        return sorted(x for i, x in enumerate(HARVEST_X) if self.mask >> i & 1)

    def phys(self, lx, ly):
        return (self.cols()[lx], ly + 2)

    def dram_ep(self, ch, noc):
        return DRAM[ch][(EP_RELOC if self.reloc else EP_LEGACY)[ch][noc]]


@dataclass
class Layout:
    """The G 4 layout (logical): rectangles 5 x 5 at origins, one mux a group, collectors, lane -> rectangle."""

    name: str = "base"
    origins: tuple = ((1, 0), (1, 5), (7, 0), (7, 5))
    mux: tuple = ((0, 1), (0, 6), (6, 1), (6, 6))  # rectangle -> its mux core
    colls: dict = None  # rectangle -> collector cores (logical)
    coll_noc: tuple = (1, 1, 0, 0)  # rectangle -> the NoC its collectors write rows on
    banks: tuple = tuple(range(8))  # DRAM channels the result rows land in (equal shares)
    perm: tuple = (2, 3, 0, 1)  # lane -> rectangle
    sink: tuple = ()  # rectangles whose collectors write nothing (yc_sink_mask)
    land_noc: int = 1  # the routers' landing writes into the lane cores
    gather_noc: int = 1  # the hubs' x row reads (data DRAM -> hub)
    mux_in_noc: int = 1  # the lane cores' packets into their mux
    mux_noc: int = 1  # mux -> ethernet core


def default_colls(lay, yc=3):
    out = {}
    taken = {m for m in lay.mux}
    for r, (ox, oy) in enumerate(lay.origins):
        cx = lay.mux[r][0]
        free = [(cx, oy + i) for i in range(5) if (cx, oy + i) not in taken]
        out[r] = free[:yc]
    return out


@dataclass
class Rates:
    lane: float = 20.0  # weight stream a lane, each way (in from the ring, out to the ring), GB/s
    rows: float = 27.0  # result rows a rectangle (12 n rows x 7680 B over 4 rectangles at 8 ms)
    xg: float = 14.0  # x rows gathered a rectangle


def flows(chip: Chip, lay: Layout, rt: Rates):
    """[(kind, rectangle, noc, src, dst, GB/s)]."""
    colls = lay.colls or default_colls(lay)
    rect_cores = {
        r: [chip.phys(ox + c, oy + y) for c in range(5) for y in range(5)] for r, (ox, oy) in enumerate(lay.origins)
    }
    F = []
    for lane, r in enumerate(lay.perm):
        cores = rect_cores[r]
        mux = chip.phys(*lay.mux[r])
        e_out = (chip.eth_out[lane], 1)
        e_in = (chip.eth_out[lane ^ 1], 1)
        for c in cores:
            F.append(("w_in", r, lay.land_noc, e_in, c, rt.lane / 25))
            F.append(("w_mux", r, lay.mux_in_noc, c, mux, rt.lane / 25))
        F.append(("w_eth", r, lay.mux_noc, mux, e_out, rt.lane))
    for r, cores in rect_cores.items():
        hub = cores[0]
        for ch in range(8):
            F.append(("x_gather", r, lay.gather_noc, chip.dram_ep(ch, lay.gather_noc), hub, rt.xg / 8))
        cl = [chip.phys(*xy) for xy in colls[r]]
        for c in cores:
            for k in cl:
                F.append(("slices", r, 1, c, k, rt.rows / 25 / len(cl)))
        if r in lay.sink:
            continue
        noc = lay.coll_noc[r]
        for k in cl:
            for ch in lay.banks:
                F.append(("row_wr", r, noc, k, chip.dram_ep(ch, noc), rt.rows / len(cl) / len(lay.banks)))
    return F, rect_cores


def load(F):
    L = defaultdict(float)
    who = defaultdict(lambda: defaultdict(float))
    for kind, r, noc, s, d, gbs in F:
        for lk in route(noc, s, d):
            L[lk] += gbs
            who[lk][f"{kind}{r}"] += gbs
    return L, who


def rule_breaks(F, rect_cores):
    """NOC0 flows that enter a rectangle from outside (the multicast rule), GB/s by kind."""
    rs = {r: set(c) for r, c in rect_cores.items()}
    out = defaultdict(float)
    for kind, r, noc, s, d, gbs in F:
        if noc != 0:
            continue
        x, y = s
        for _, lx, ly, dr in route(0, s, d):
            nx, ny = ((lx + 1) % W, ly) if dr == "E" else (lx, (ly + 1) % H)
            for q, cs in rs.items():
                if (nx, ny) in cs and (lx, ly) not in cs:
                    out[(kind, r, q)] += gbs
    return dict(out)


def lane_bottleneck(F, L, lay):
    """Per lane: the busiest link its weight flows (w_in, w_mux, w_eth) cross, as a fraction of LINK_GBS."""
    out = {}
    for lane, r in enumerate(lay.perm):
        worst = 0.0
        for kind, rr, noc, s, d, gbs in F:
            if rr != r or not kind.startswith("w_"):
                continue
            for lk in route(noc, s, d):
                worst = max(worst, L[lk])
        out[lane] = worst / LINK_GBS
    return out


def out_bottleneck(F, L):
    """Per rectangle: the busiest link its result path (slices, row writes) crosses, as a fraction of LINK_GBS."""
    out = defaultdict(float)
    for kind, r, noc, s, d, gbs in F:
        if kind not in ("slices", "row_wr"):
            continue
        for lk in route(noc, s, d):
            out[r] = max(out[r], L[lk] / LINK_GBS)
    return dict(out)


def report(chip, lay, rt=Rates(), top=6, verbose=True):
    F, rc = flows(chip, lay, rt)
    L, who = load(F)
    rect_worst = {}
    for lane, r in enumerate(lay.perm):
        rect_worst[r] = lane_bottleneck(F, L, lay)[lane]
    ob = out_bottleneck(F, L)
    hot = sorted(L.items(), key=lambda kv: -kv[1])[:top]
    rb = rule_breaks(F, rc)
    if verbose:
        print(f"== {lay.name}  mask {chip.mask:#06x} reloc {int(chip.reloc)}  cols {chip.cols()[:12]}")
        print("   per rectangle, its lane's busiest link (x link rate): " + ", ".join(f"r{r} {v:.2f}" for r, v in sorted(rect_worst.items())))
        print("   per rectangle, its result path's busiest link: " + ", ".join(f"r{r} {v:.2f}" for r, v in sorted(ob.items())))
        for lk, v in hot:
            parts = sorted(who[lk].items(), key=lambda kv: -kv[1])[:5]
            print(f"   NOC{lk[0]} ({lk[1]},{lk[2]}) {lk[3]}: {v:5.1f} GB/s ({v / LINK_GBS:.2f})  " + ", ".join(f"{k} {g:.1f}" for k, g in parts))
        if rb:
            print("   multicast rule: NOC0 into a rectangle: " + ", ".join(f"{k}{r}->rect{q} {g:.1f}" for (k, r, q), g in rb.items()))
    return rect_worst, hot, rb, ob
