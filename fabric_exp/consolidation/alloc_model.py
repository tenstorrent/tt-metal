#!/usr/bin/env python3
"""alloc_model.py: a credit-window model of the benchmark rows, fitted to the slot allocation sweeps, and the slot
allocation it implies (an algorithm instead of a hand-picked split). Three router families on Blackhole:
  vc1  VC0 + VC1 routers on the VC1 split (two 2x2 meshes; v4gi -O3; RESULTS §98 jobs 109-112)
  2d   VC0-only mesh routers (2x4 P150x8; v4us; RESULTS §98 slot allocation sweep, jobs 070-075, 4352 B suite)
  1d   VC0-only Linear routers (v4us; same sweep, 1D suite)

Model. A flow's path crosses windows: sender channels (the worker, the forwarding channels) and the far rings they
fill (the receivers). A window of W slots carries one packet per
    RT0 / W + b   cycles
where RT0 is its credit loop's round trip with nothing else in flight and b what each packet in flight adds to it (the
loop's stages get busier as the flow speeds up; Little's law with RT = RT0 + b W). A row runs at the slowest of its
windows, or at its service bound P (E0, E1 and the link per packet) when no window binds:
    cycles/packet = max(P, max over its windows of RT0_k / W_k + b_k)
RT0 and b are fitted per window kind (one channel can be several kinds: its loop depends on the traffic, e.g. a
bidirectional or a scatter row), completion group T (v4: from the ring size) and packet size: a least-squares line
through rate x W = RT0 + b W where a kind has points at three or more window sizes, else b = 0 and RT0 the median.
A larger group adds a step per extra slot in the group (the oldest slot waits for the group to fill), fitted where a
kind was measured at two groups (VC1's VC0 forwarding: ~93 cycles), else that value.

Allocation (allocate()): 1. VC0 receivers at T = 4 (fewest credits: E0's pass binds the 1-hop rows, and T 4 -> 2 cost
them 9.5) and a power of 2 (a receiver ring of another size costs E1 ~8 cycles/packet where E1 binds: rings, RESULTS
§98 slot sweep), the smallest that no row's window on it binds. 2. The worker: the smallest that no 1-hop row's worker
window binds (from its fit, else from an input round trip). 3. The other slots one unit at a time (a unit: one slot per
forwarding channel of a VC, or one VC1 ring slot) to the window that binds the forwarded row with the worst weighted
cycles/packet (max-min water-filling; rows at their service bound are frozen); a unit that does not lower it is
skipped. 4. Leftovers to the worker.

  alloc_model.py fit [FAMILY]              the fitted parameters and every measured point against the model
  alloc_model.py alloc [FAMILY] [VC0:VC1]  the allocation (vc1: weights VC0:VC1, default 1:1) and its predicted rows
  alloc_model.py predict FAMILY ALLOC ...  the model's rows for given allocations (comma-separated, the family's fields)
  alloc_model.py check PREFIX INTER_PREFIX LABEL=W,F0,R0,F1,R1 ...   vc1 runs it was not fitted to, against it
"""
import csv, glob, math, os, re, statistics, sys
from collections import defaultdict

RES = "/home/aagarwal/tt-metal/fabric_exp/results"
DEFAULT_GROUP_STEP = 93  # cycles per extra slot in a completion group (VC1's VC0 forwarding, jobs 109-112)


def group(n):  # v4's completion group of a ring of n slots
    return 1 if n < 8 else (2 if n < 16 else 4)


def pow2(n):
    return n & (n - 1) == 0


def line(points):
    """Least squares rate x W = RT0 + b W over (W, rate x W); under three window sizes: b = 0, RT0 the median (two
    sizes, one of them near a bound, gave b = 134 for the VC1 ring)."""
    ws = sorted({w for w, _ in points})
    if len(ws) < 3:
        return statistics.median(x for _, x in points), 0.0
    n = len(points)
    mw, mx = sum(w for w, _ in points) / n, sum(x for _, x in points) / n
    b = sum((w - mw) * (x - mx) for w, x in points) / sum((w - mw) ** 2 for w, _ in points)
    return mx - b * mw, b


# ---- families ---------------------------------------------------------------------------------------------------------

class VC1:
    name = "vc1"
    fields = ("worker", "f0", "r0", "f1", "r1")
    capacity = 81  # 4448-byte slots
    far_ring = dict(worker="r0", f0="r0", r0="r0", f1="r1", r1="r1")
    vc0_rings = ("r0",)
    units = dict(f0=3, f1=3, r1=1)  # 3 forwarding channels per VC; the VC1 ring is a water-filled unit
    classes = {"VC0": ("2-hop",), "VC1": ("inter + VC1",)}
    one_hop = ("1-hop", "inter direct")
    worker_rt = 2730  # no worker point binds in these sweeps but f8r32 (15 slots: bursts of 8 up to 182)
    threshold = 0.08  # a point within this of its class's bound is taken as at the bound
    # Window kinds: the VC0 ring behind an intra-mesh link (r0) and behind the inter-mesh link (r0x: the same ring, a
    # longer round trip); VC0 and VC1 forwarding channels; the VC1 ring.
    rows = {
        "VC1Mesh1HopEW": ("1-hop", [("worker", "worker"), ("r0", "r0")]),
        "VC1Mesh2Hop": ("2-hop", [("worker", "worker"), ("r0", "r0"), ("f0", "f0")]),
        "VC1Inter_0_0_to_1_1": ("inter direct", [("worker", "worker"), ("r0", "r0x")]),
        "VC1Inter_0_0_to_1_0": ("inter + VC1", [("worker", "worker"), ("r0", "r0x"), ("f1", "f1"), ("r1", "r1")]),
    }
    ALLOCS = {
        "main": (51, 4, 8, 2, 4), "r16": (43, 4, 16, 2, 4), "f8": (39, 8, 8, 2, 4), "f8r16": (31, 8, 16, 2, 4),
        "f8r16v": (21, 8, 16, 4, 8), "f12r16": (19, 12, 16, 2, 4), "f8r32": (15, 8, 32, 2, 4),
        "f10r16v4": (19, 10, 16, 4, 4), "f8r16v6": (19, 8, 16, 6, 4),
        "a12": (19, 12, 16, 2, 4), "a10": (19, 10, 16, 4, 4), "a9": (19, 9, 16, 5, 4), "a8": (19, 8, 16, 6, 4),
        "a7": (19, 7, 16, 7, 4),
        "eq": (18, 8, 16, 6, 5), "w32": (18, 9, 16, 5, 5), "w12": (19, 5, 16, 8, 7), "pow2": (25, 8, 16, 4, 4),
    }
    RUNS = [("vca", None, ["main", "r16", "f8", "f8r16", "f8r16v", "f12r16", "f8r32"]),
            ("vcb_v4gi", "vcbi_v4gi", ["main", "f8", "f8r16", "f12r16", "f8r16v", "f10r16v4", "f8r16v6"]),
            ("vcc_v4gi", "vcci_v4gi", ["main", "a12", "a10", "a9", "a8", "a7"])]
    # Job 112 (the model's own allocations, predicted beforehand: vcd_predictions.txt; `check` compares them with the
    # fit of jobs 109-111). FIT_JOB112=0 leaves them out.
    fit_job112 = os.environ.get("FIT_JOB112", "1") != "0"
    if fit_job112:
        RUNS = RUNS + [("vcd_v4gi", "vcdi_v4gi", ["eq", "w32", "w12", "pow2", "a9"])]
    # Classes whose service bound the sweeps reached; the others' bound is the one of the path they extend (a lower
    # bound). inter + VC1 was reached in job 112 (19,5,16,8,7: ~290).
    saturated = ("1-hop", "inter direct", "inter + VC1") if fit_job112 else ("1-hop", "inter direct")
    extends = {"2-hop": "1-hop"} if fit_job112 else {"2-hop": "1-hop", "inter + VC1": "inter direct"}

    @staticmethod
    def exclude(a):
        return False

    @staticmethod
    def binding(cls, a):
        """The (field, kind) that binds a slow point, by the row's construction."""
        w, f0, r0, f1, r1 = a
        if cls in ("1-hop", "inter direct"):
            return None if r0 >= 16 else ("r0", "r0" if cls == "1-hop" else "r0x")
        if cls == "2-hop":
            return ("f0", "f0")
        return ("f1", "f1") if f1 <= r1 else ("r1", "r1")

    @classmethod
    def load(cls):
        acc = defaultdict(list)
        for main_prefix, inter_prefix, labels in cls.RUNS:
            for lab in labels:
                for prefix in filter(None, (main_prefix, inter_prefix)):
                    for p in (1, 2):
                        f = os.path.join(RES, f"{prefix}_{lab}_p{p}_o3_13129088_vc1_tests.csv")
                        if not os.path.exists(f):
                            continue
                        for r in csv.DictReader(open(f)):
                            name, size = r["test_name"].split("_size_")[0], r["packet_size"]
                            if name in cls.rows:
                                acc[(cls.rows[name][0], size, cls.ALLOCS[lab])].append(
                                    1.35e9 / float(r["packets_per_second"]))
        return {k: statistics.mean(v) for k, v in acc.items()}


class VC0Family:
    """VC0-only routers of the slot sweep (v4us, jobs 070-075): worker, forwarding (every forwarding channel), receiver.
    Runs: results/<prefix>_<tag>_<suite>.csv with the depths each run got in its log (TT_FABRIC_EXP_LOG_SLOTS)."""
    fields = ("worker", "f", "r")
    far_ring = dict(worker="r", f="r", r="r")
    vc0_rings = ("r",)
    classes = {"forwarded": ("3-hop", "multicast")}
    worker_rt = None  # fitted
    threshold = 0.02
    extends = {"3-hop": "1-hop"}

    @staticmethod
    def exclude(a):
        return not pow2(a[2])  # a receiver ring that is not a power of 2 costs E1 ~8 (not a window: not modelled)

    @classmethod
    def load(cls):
        acc = defaultdict(list)
        for prefix, suite in cls.RUNS:
            for f in glob.glob(f"{RES}/{prefix}_*_{suite}.csv"):
                tag = os.path.basename(f)[len(prefix) + 1: -len(suite) - 5]
                if "_val" in tag:
                    continue
                a = cls.depths(f[:-4] + ".log")
                if a is None:
                    continue
                for r in csv.DictReader(open(f)):
                    key = f'{r["test_name"]} {r["ntype"].replace("NOC_UNICAST_", "").lower()}'
                    if key in cls.rows:
                        acc[(cls.rows[key][0], r["packet_size"], a)].append(1.35e9 / float(r["avg_packets_per_s"]))
        return {k: statistics.mean(v) for k, v in acc.items()}

    @staticmethod
    def depths(log):
        if not os.path.exists(log):
            return None
        for line in open(log, errors="replace"):
            m = re.search(r"VC0 sender slots \{([^}]*)\} receiver slots \{([^}]*)\}", line)
            if m and not re.search(r"\b0\b", m.group(1)):
                s = [int(x) for x in m.group(1).split(",")]
                return (s[0], s[1], int(m.group(2).split(",")[0]))
        return None


def one_hop_rows(prefix):
    return {
        f"{prefix}1HopEW write": ("1-hop", [("worker", "w"), ("r", "r")]),
        f"{prefix}1HopNS write": ("1-hop", [("worker", "w"), ("r", "r")]),
        f"{prefix}1HopEWBidir write": ("1-hop bidir", [("worker", "wbi"), ("r", "rbi")]),
        f"{prefix}1HopEW scatter_write": ("1-hop scatter", [("worker", "wsc"), ("r", "rsc")]),
        f"{prefix}3HopEW write": ("3-hop", [("worker", "w"), ("r", "r3"), ("f", "f")]),
        f"{prefix}Multicast write": ("multicast", [("r", "rmc"), ("f", "fmc")]),
    }


class Mesh2D(VC0Family):
    name = "2d"
    capacity = 81
    units = dict(f=3)  # 3 forwarding channels
    one_hop = ("1-hop", "1-hop bidir", "1-hop scatter")
    saturated = ("1-hop", "1-hop bidir", "1-hop scatter", "multicast")
    rows = one_hop_rows("Mesh")
    RUNS = [("v4us", "4352"), ("v4usc", "4352"), ("v4usd", "4352")]

    @staticmethod
    def binding(cls, a):
        w, f, r = a
        if r < 16:
            return ("r", dict(zip(("1-hop", "1-hop bidir", "1-hop scatter", "3-hop", "multicast"),
                                  ("r", "rbi", "rsc", "r3", "rmc")))[cls])
        if cls in ("1-hop", "1-hop bidir", "1-hop scatter"):
            return ("worker", dict(zip(("1-hop", "1-hop bidir", "1-hop scatter"), ("w", "wbi", "wsc")))[cls])
        return ("f", "f" if cls == "3-hop" else "fmc")


class Linear1D(VC0Family):
    name = "1d"
    capacity = 83  # 4400-byte slots
    units = dict(f=1)  # 1 forwarding channel
    one_hop = ("1-hop", "1-hop bidir", "1-hop scatter")
    saturated = ("1-hop", "1-hop bidir", "1-hop scatter", "multicast")
    rows = one_hop_rows("Linear")
    RUNS = [("v4usa", "1d"), ("v4usd", "1d")]

    @staticmethod
    def binding(cls, a):
        w, f, r = a
        if cls in ("1-hop", "1-hop bidir"):
            return ("worker", "w" if cls == "1-hop" else "wbi")
        if cls == "1-hop scatter":  # its worker and ring loops are both longer; a 16-slot ring binds it at 51 workers
            return ("r", "rsc") if w >= 27 else ("worker", "wsc")
        return ("f", "f" if cls == "3-hop" else "fmc")


FAMILIES = {f.name: f for f in (VC1, Mesh2D, Linear1D)}


# ---- model --------------------------------------------------------------------------------------------------------------

class Model:
    def __init__(self, fam):
        self.fam = fam
        data = {k: v for k, v in fam.load().items() if not fam.exclude(k[2])}
        self.data = data
        self.P = defaultdict(float)
        for (cls, size, a), rate in data.items():
            if cls in fam.saturated:
                self.P[(cls, size)] = min(self.P.get((cls, size)) or 1e9, rate)
        for cls, base in fam.extends.items():
            for size in {s for (_, s, _) in data}:
                self.P[(cls, size)] = self.P[(base, size)]
        self.samples = defaultdict(list)
        for (cls, size, a), rate in data.items():
            b = fam.binding(cls, a)
            if b and cls in fam.saturated and rate < (1 + fam.threshold) * self.P[(cls, size)]:
                b = None  # at (or near) the service bound: no window binds
            if b:
                field, kind = b
                self.samples[(kind, self.t_of(field, a), size)].append((self.size_of(field, a), rate * self.size_of(field, a)))
        self.fit = {k: line(v) for k, v in self.samples.items()}
        self.tstep = {}
        for (kind, t, size), (rt0, b) in self.fit.items():
            for (k2, t2, s2), (rt2, b2) in self.fit.items():
                if k2 == kind and s2 == size and t2 > t:
                    W = sorted({w for w, _ in self.samples[(k2, t2, s2)]})[0]
                    self.tstep[(kind, size)] = ((rt2 + b2 * W) - (rt0 + b * W)) / (t2 - t)

    def size_of(self, field, a):
        return a[self.fam.fields.index(field)]

    def t_of(self, field, a):  # the group of the ring a window fills (its own for a ring)
        return group(self.size_of(self.fam.far_ring[field], a))

    def window(self, kind, t, size, W):
        """Cycles per packet that a window of W slots under group t allows (None: kind never measured)."""
        known = [(tt, v) for (k, tt, s), v in self.fit.items() if k == kind and s == size]
        if not known:
            return None
        tt, (rt0, b) = min(known, key=lambda x: abs(x[0] - t))
        if tt != t:
            rt0 += (t - tt) * self.tstep.get((kind, size), DEFAULT_GROUP_STEP)
        return rt0 / W + b

    def windows(self, cls):
        return next(w for c, w in self.fam.rows.values() if c == cls)

    def predict(self, cls, size, a, detail=False):
        worst, which = self.P[(cls, size)], None
        for field, kind in self.windows(cls):
            v = self.window(kind, self.t_of(field, a), size, self.size_of(field, a))
            if v is not None and v > worst:
                worst, which = v, field
        return (worst, which) if detail else worst

    def sizes(self):
        return sorted({s for (_, s, _) in self.data}, key=int)


# ---- allocation ---------------------------------------------------------------------------------------------------------

def allocate(m, weights=None, size=None):
    fam = m.fam
    size = size or m.sizes()[-1]  # the largest packets: the longest loops
    weights = weights or {c: 1.0 for c in fam.classes}
    a = {f: 1 for f in fam.fields}
    with_field = lambda field: [c for c in {c for c, _ in fam.rows.values()} if any(f == field for f, _ in m.windows(c))]
    as_tuple = lambda: tuple(a[f] for f in fam.fields)

    def binds(field, value, classes):
        a[field] = value
        return any(m.predict(c, size, as_tuple(), detail=True)[1] == field for c in classes)

    # 1. VC0 rings: T = 4, powers of 2, the smallest that no row's window on it binds
    for ring in fam.vc0_rings:
        a.update({f: 10 ** 6 for f in fam.fields if f != ring})  # nothing else binds while sizing it
        r = 16
        while binds(ring, r, with_field(ring)) and r < 128:
            r *= 2
        a[ring] = r
    # 2. the worker: the smallest that no 1-hop row's worker window binds
    if fam.worker_rt is None:
        w = 1
        while binds("worker", w, fam.one_hop) and w < 127:
            w += 1
    else:
        w = math.ceil(fam.worker_rt / min(m.P[(c, size)] for c in fam.one_hop))
    a["worker"] = w
    # 3. water-fill the forwarding side
    for f in fam.units:
        a[f] = 1
    used = lambda: sum(a[f] * (fam.units.get(f, 1) if f in fam.units else 1) for f in fam.fields)

    def cost(name):
        """The class's slowest row that a forwarding-side unit can still speed up (rows at their service bound, or
        bound by a window step 3 does not size, are frozen), weighted; (0, None) when all are frozen."""
        live = [(r, f) for r, f in (m.predict(c, size, as_tuple(), detail=True) for c in fam.classes[name])
                if f in fam.units]
        if not live:
            return 0.0, None
        worst, field = max(live)
        return worst * weights[name], field

    while True:
        progressed = False
        for name in sorted(fam.classes, key=lambda n: -cost(n)[0]):
            c, field = cost(name)
            if field is None or used() + fam.units[field] > fam.capacity:
                continue
            a[field] += 1
            if cost(name)[0] < c:
                progressed = True
                break
            a[field] -= 1
        if not progressed:
            break
    a["worker"] += fam.capacity - used()  # 4. leftovers
    return as_tuple()


# ---- commands -----------------------------------------------------------------------------------------------------------

def cmd_fit(family="vc1"):
    m = Model(FAMILIES[family])
    print(f"[{family}] service bounds P (cycles/packet; reached, or the path's they extend):")
    for (cls, size), p in sorted(m.P.items()):
        print(f"  {cls:13} {size:>5} B  {p:6.1f}")
    print("\nWindows: rate x W = RT0 + b W (cycles; the points: W -> rate x W):")
    for (kind, t, size), (rt0, b) in sorted(m.fit.items()):
        pts = ", ".join(f"{W}: {x:.0f}" for W, x in sorted(m.samples[(kind, t, size)]))
        print(f"  {kind:4} T={t} {size:>5} B  RT0 {rt0:6.0f}  b {b:5.1f}   ({pts})")
    for (kind, size), st in sorted(m.tstep.items()):
        print(f"  group step {kind} {size} B: {st:.0f} cycles per extra slot in the group")
    print("\nMeasured against the model (cycles/packet):")
    errs = []
    for (cls, size, a), rate in sorted(m.data.items()):
        pr = m.predict(cls, size, a)
        errs.append(abs(pr - rate) / rate)
        print(f"  {cls:13} {size:>5} B  {str(a):22} measured {rate:6.1f}  model {pr:6.1f}  {100 * (pr - rate) / rate:+5.1f}%")
    print(f"\n{len(errs)} points: mean |error| {100 * statistics.mean(errs):.1f}%, max {100 * max(errs):.1f}%")


def cmd_alloc(family="vc1", arg=None):
    fam = FAMILIES[family]
    m = Model(fam)
    weights = None
    if arg:
        weights = dict(zip(fam.classes, (float(x) for x in arg.split(":"))))
    alloc = allocate(m, weights)
    total = sum(v * fam.units.get(f, 1) for f, v in zip(fam.fields, alloc))
    print(f"[{family}] weights {arg or 'equal'}: allocation {dict(zip(fam.fields, alloc))}, {total} of {fam.capacity} slots")
    for size in m.sizes():
        rows = ", ".join(f"{c} {m.predict(c, size, alloc):.0f}" for c in sorted({c for c, _ in fam.rows.values()}))
        print(f"  model, {size:>4} B: {rows}")


def cmd_predict(family, *allocs):
    m = Model(FAMILIES[family])
    for spec in allocs:
        a = tuple(int(x) for x in spec.split(","))
        for size in m.sizes():
            rows = "  ".join(f"{c} {m.predict(c, size, a):6.1f}" for c in sorted({c for c, _ in m.fam.rows.values()}))
            print(f"{spec:14} {size:>4} B  {rows}")


def cmd_check(prefix="vcd_v4gi", inter_prefix="vcdi_v4gi", *specs):
    """vc1 runs the model was not fitted to (labels PREFIX_<label>_p1_o3, specs label=W,F0,R0,F1,R1), against it."""
    m = Model(VC1)
    errs = []
    for spec in specs:
        lab, alloc = spec.split("=")
        a = tuple(int(x) for x in alloc.split(","))
        acc = defaultdict(list)
        for pre in (prefix, inter_prefix):
            f = os.path.join(RES, f"{pre}_{lab}_p1_o3_13129088_vc1_tests.csv")
            if os.path.exists(f):
                for r in csv.DictReader(open(f)):
                    name = r["test_name"].split("_size_")[0]
                    if name in VC1.rows:
                        acc[(VC1.rows[name][0], r["packet_size"])].append(1.35e9 / float(r["packets_per_second"]))
        for (cls, size), v in sorted(acc.items()):
            rate, pr = statistics.mean(v), m.predict(cls, size, a)
            errs.append(abs(pr - rate) / rate)
            print(f"  {lab:5} {alloc:13} {cls:13} {size:>5} B  measured {rate:6.1f}  model {pr:6.1f}  {100 * (pr - rate) / rate:+5.1f}%")
    if errs:
        print(f"{len(errs)} points: mean |error| {100 * statistics.mean(errs):.1f}%, max {100 * max(errs):.1f}%")


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "fit"), sys.argv[2:]
    {"fit": cmd_fit, "alloc": cmd_alloc, "predict": cmd_predict, "check": cmd_check}[cmd](*args)
