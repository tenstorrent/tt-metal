#!/usr/bin/env python3
"""sls_model.py: the slot study's credit-window model (SLOT_STUDY.md, 2026-11-01). alloc_model.py's model and allocation
rule (imported unchanged), fitted to other data:
  v4gi   alloc_model.py's VC1 family as is (v4gi, jobs 109-112, pre-merge placement): the 2026-10-28 fit
  p150   the lead (v4rxv2i3 -O3) on the P150 VC1 split, post-merge placement ([0, 0] lands on [1, 0]: VC1Inter 1_0 is
         the direct row, 1_1 / 1_2 one VC1 hop, 1_3 two): results/sls*_<tag>_p1_o3 runs (slots from their logs), plus the
         lead's earlier rows at 19,9,16,5,4 (census sdb: censdbvc1[i]_v4rxv2i3, builder default) and job 360 (bndw_*)
  prov   the provisional lead model used for the predictions written before P1: p150's data where a window kind has
         lead points (v4gi's slope b kept, the intercept refitted), v4gi's fit elsewhere
  glx    the lead on the Galaxy 2x2 split (P2; results under GLX_RES, row mapping in GLX_ROWS)
  2d     VC0-only 2D mesh routers on the lead (P3): P150 sls2d_* runs, or a Galaxy job's 2d_* runs (SLS2D_RES=<dir>),
         at SLS2D_SIZE (default 1024)

Window kinds (as alloc_model.py): worker; r0 (VC0 ring behind an intra-mesh link), r0x (behind the inter-mesh link); f0,
f1 (VC0 / VC1 forwarding); r1 (VC1 ring); plus f1x / r1x for the two-VC1-hop row (its own loops) on p150 / glx.

  sls_model.py fit FAMILY                    parameters and every point against the model
  sls_model.py alloc FAMILY [VC0:VC1 ...]    the rule's allocation per weight and its rows
  sls_model.py predict FAMILY ALLOC ...      rows for given allocations (W,F0,R0,F1,R1)
"""
import csv, glob, gzip, os, re, statistics, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alloc_model as am  # noqa: E402

RES = am.RES
SLOT_RE0 = re.compile(r"VC0 sender slots \{([^}]*)\} receiver slots \{([^}]*)\}")
SLOT_RE1 = re.compile(r"VC1 sender slots \{([^}]*)\} receiver slots \{([^}]*)\}")
STUDY_DEFAULT = (19, 9, 16, 5, 4)  # builder default for kernel v2 VC1 routers since job 113 (no slot switch)


def log_lines(log):
    """The log's lines (LOG or LOG.gz, as the Galaxy results are kept), or [] when neither exists."""
    if os.path.exists(log):
        return open(log, errors="replace")
    if os.path.exists(log + ".gz"):
        return gzip.open(log + ".gz", "rt", errors="replace")
    return []


def log_slots(log):
    """(W, F0, R0, F1, R1) of the routers with a worker and VC1 forwarding (the first such VC0 line and the VC1 line
    after it), or None when the run did not log slots."""
    vc0 = None
    for line in log_lines(log):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = SLOT_RE0.search(line)
        if m:
            s = [int(x) for x in m.group(1).split(",")]
            vc0 = (s[0], s[1], int(m.group(2).split(",")[0])) if len(s) >= 2 and s[1] > 0 else None
            continue
        m = SLOT_RE1.search(line)
        if m and vc0:
            s = [int(x) for x in m.group(1).split(",")]
            if s[0] > 0:
                return vc0 + (s[0], int(m.group(2).split(",")[0]))
    return None


def cycles_per_packet(r):
    """Cycles/packet of a per-test CSV row (its own cycles over its packets; = 1.35e9 / pps on the P150)."""
    n = int(r["num_packets"]) * max(1, int(r.get("total_traffic_count") or 1))
    return int(r["cycles"]) / n if n and int(r["cycles"]) else None


ROW_WIN = {
    "1-hop": [("worker", "worker"), ("r0", "r0")],
    "2-hop": [("worker", "worker"), ("r0", "r0"), ("f0", "f0")],
    "inter direct": [("worker", "worker"), ("r0", "r0x")],
    "inter + VC1": [("worker", "worker"), ("r0", "r0x"), ("f1", "f1"), ("r1", "r1")],
    "inter + 2 VC1": [("worker", "worker"), ("r0", "r0x"), ("f1", "f1x"), ("r1", "r1x")],
    # Galaxy split only: the VC1 hop N/S in the destination mesh, and a VC0 hop into the exit chip before the crossing
    "inter + VC1 N/S": [("worker", "worker"), ("r0", "r0x"), ("f1", "f1n"), ("r1", "r1n")],
    "VC0 hop + inter": [("worker", "worker"), ("r0", "r0x"), ("f0", "f0x")],
}


class LeadVC1(am.VC1):
    """The lead on a VC1 split; rows by test name (placement-specific), runs found by label pattern."""
    name = "p150"
    ROWS = {"VC1Mesh1HopEW": "1-hop", "VC1Mesh1HopNS": "1-hop", "VC1Mesh2Hop": "2-hop",
            "VC1Inter_0_0_to_1_0": "inter direct", "VC1Inter_0_0_to_1_1": "inter + VC1",
            "VC1Inter_0_0_to_1_2": "inter + VC1", "VC1Inter_0_0_to_1_3": "inter + 2 VC1"}
    rows = {}  # filled per class below: {test: (class, windows)}
    classes = {"VC0": ("2-hop",), "VC1": ("inter + VC1",)}
    saturated = ("1-hop", "inter direct", "inter + VC1", "inter + 2 VC1")
    extends = {"2-hop": "1-hop"}
    worker_rt = 2730  # replaced by the fitted worker window where the lead has worker-bound points (fitted_worker_rt)
    # earlier lead runs without logged slots (builder default) or with them (job 360)
    EARLIER = [("censdbvc1_v4rxv2i3_default_p1_o3", STUDY_DEFAULT), ("censdbvc1i_v4rxv2i3_default_p1_o3", STUDY_DEFAULT),
               ("bndw_d_default_p1_o3", None), ("bndw_d_2_default_p1_o3", None), ("bndw_w_default_p1_o3", None)]
    PATTERN = "sls[a-z0-9]*_*_p1_o3"
    use_earlier = True

    @classmethod
    def runs(cls):
        out = []
        if cls.use_earlier:
            for lab, a in cls.EARLIER:
                out.append((os.path.join(RES, f"{lab}_13129088_vc1"), a))
        for f in sorted(glob.glob(os.path.join(RES, cls.PATTERN + "_13129088_vc1_tests.csv"))):
            out.append((f[: -len("_tests.csv")], None))
        return out

    @classmethod
    def load(cls):
        acc = defaultdict(list)
        for base, a in cls.runs():
            a = a or log_slots(base + ".log")
            if a is None or not os.path.exists(base + "_tests.csv"):
                continue
            for r in csv.DictReader(open(base + "_tests.csv")):
                name = r["test_name"].split("_size_")[0]
                if name in cls.rows:
                    v = cycles_per_packet(r)
                    if v:
                        acc[(cls.rows[name][0], r["packet_size"], a)].append(v)
        return {k: statistics.mean(v) for k, v in acc.items()}

    WORKER_CAL = 16  # a worker below this binds the 1-hop rows (P 142): the worker's calibration points
    WORKER_SLOW = 10  # ... and below this the slower rows too (cw8: 2-hop 277 / 293 against 239 / 258 at 12-19)
    # VC1 forwarding against the VC1 ring at the same path: the forwarding window binds while F1 < RATIO x R1, RATIO the
    # ratio of their round trips (P150 lead: forwarding ~1,960-2,050, ring ~1,460-1,515 at 4352 B, from cf3-cf5 at
    # ring 7 and cr3-cr4 at forwarding 8 -> 1.33). With RATIO 1 (alloc_model.py's rule), 18,8,16,6,5 and 19,9,16,5,4 were
    # taken as ring-bound; cf5 (5 at ring 7: 400.5) and cr4 (ring 4 at forwarding 8: 364.6) show both are forwarding-bound.
    RATIO_F1_R1 = 1.33

    @classmethod
    def binding(cls_, cls, a):
        """The window a slow point is taken to be bound by, by the run's design: the calibration runs change one window
        kind each (a small worker, an 8-slot VC0 ring, a small VC1 forwarding or ring window); a point whose row may be
        bound by a window the run also shrank is not used (None)."""
        w, f0, r0, f1, r1 = a
        if cls in ("1-hop", "inter direct"):
            if r0 < 16:
                return None if w < cls_.WORKER_CAL else ("r0", "r0" if cls == "1-hop" else "r0x")
            return ("worker", "worker") if w < cls_.WORKER_CAL and cls == "1-hop" else None
        if w < cls_.WORKER_SLOW:
            return None
        if cls == "2-hop":
            return ("f0", "f0")
        if cls == "VC0 hop + inter":
            return ("f0", "f0x")
        x = {"inter + 2 VC1": "x", "inter + VC1 N/S": "n"}.get(cls, "")
        return ("f1", "f1" + x) if f1 < cls_.RATIO_F1_R1 * r1 else ("r1", "r1" + x)


LeadVC1.rows = {t: (c, ROW_WIN[c]) for t, c in LeadVC1.ROWS.items()}


class GlxVC1(LeadVC1):
    """The lead on the Galaxy 2x2 split (P2). GLX_RES: the job's results directory (labels vc1_<pass>_<tag>); GLX_ROWS:
    test=class,... (the placement is read from the data, SLOT_STUDY.md); GLX_RATIO: the forwarding / ring loop ratio."""
    name = "glx"
    use_earlier = False
    RATIO_F1_R1 = float(os.environ.get("GLX_RATIO", "1.33"))
    ROWS = {"VC1Mesh1HopEW": "1-hop", "VC1Mesh1HopNS": "1-hop", "VC1Mesh2Hop": "2-hop",
            "GlxInter_0_1_to_1_0": "inter direct", "GlxInter_0_1_to_1_1": "inter + VC1",
            "GlxInter_0_1_to_1_3": "inter + 2 VC1", "GlxInter_0_1_to_1_2": "inter + VC1 N/S",
            "GlxInter_0_0_to_1_0": "VC0 hop + inter"}
    saturated = ("1-hop", "2-hop", "inter direct", "inter + VC1", "inter + 2 VC1", "inter + VC1 N/S", "VC0 hop + inter")
    extends = {}  # 2-hop is wire-bound at 4352 B from ~11 forwarding slots: its own best rate is its bound

    # GLX_KIND: the runs' label kind (vc1: 4352 B slots, P2; vc12k: 2 KiB slots); GLX_CAP: slots per ERISC (81; 169 at
    # 2 KiB slots)
    capacity = int(os.environ.get("GLX_CAP", "81"))

    @classmethod
    def runs(cls):
        kind = os.environ.get("GLX_KIND", "vc1")
        return [(f[: -len("_tests.csv")], None) for res in os.environ.get("GLX_RES", "").split(":")
                for f in sorted(glob.glob(os.path.join(res, f"{kind}_*_tests.csv")))]


GlxVC1.rows = {t: (c, ROW_WIN[c]) for t, c in GlxVC1.ROWS.items()}


class Lead2D(am.VC0Family):
    """VC0-only 2D mesh routers on the lead (P3): worker, forwarding (every forwarding channel), receiver. Runs: P150
    results/sls2d_<pass>_<tag> (SLS2D_RES unset) or a Galaxy job's 2d_<pass>_<tag> (SLS2D_RES=<results dir>); size
    SLS2D_SIZE (default 1024). Window kinds per row class: the worker (1-hop rows), the receiver, E/W forwarding (3-hop
    E/W, the turn), N/S forwarding (3-hop / 7-hop N/S, the 10-hop corner; Galaxy only)."""
    name = "2d"
    capacity = int(os.environ.get("SLS2D_CAP", "81"))  # 169 at 2 KiB slots
    units = dict(f=3)
    one_hop = ("1-hop",)
    # every class's bound is its best measured rate (the wire at 4352 B on the Galaxy; a non-window floor such as the
    # Galaxy's N/S rows' ~182 at 1 KiB): points within `threshold` of it are not window samples
    saturated = ("1-hop", "3-hop E/W", "turn 4", "3-hop N/S", "7-hop N/S", "corner 10")
    extends = {}
    worker_rt = None
    threshold = 0.03
    classes = {"forwarded": ("3-hop E/W", "turn 4", "3-hop N/S", "7-hop N/S", "corner 10")}
    rows = {
        "Sls1HopEW": ("1-hop", [("worker", "w"), ("r", "r")]),
        "Sls1HopNS": ("1-hop", [("worker", "w"), ("r", "r")]),
        "Sls3HopEW": ("3-hop E/W", [("worker", "wf"), ("r", "r3"), ("f", "few")]),
        "SlsTurn4": ("turn 4", [("worker", "wf"), ("r", "r3"), ("f", "fturn")]),
        "Sls3HopNS": ("3-hop N/S", [("worker", "wf"), ("r", "r3"), ("f", "fns3")]),
        "Sls7HopNS": ("7-hop N/S", [("worker", "wf"), ("r", "r3"), ("f", "fns7")]),
        "SlsCorner10": ("corner 10", [("worker", "wf"), ("r", "r3"), ("f", "fns7")]),
    }

    @staticmethod
    def exclude(a):
        return False

    @staticmethod
    def binding(cls, a):
        w, f, r = a
        if r < 16:
            return None if w < 17 else ("r", "r" if cls == "1-hop" else "r3")
        if cls == "1-hop":
            return ("worker", "w") if w < 20 else None
        if w < 12:
            return ("worker", "wf") if f >= 18 else None
        kind = {"3-hop E/W": "few", "turn 4": "fturn", "3-hop N/S": "fns3", "7-hop N/S": "fns7", "corner 10": "fns7"}[cls]
        return ("f", kind)

    @classmethod
    def load(cls):
        res = os.environ.get("SLS2D_RES")
        size = os.environ.get("SLS2D_SIZE", "1024")
        kind = os.environ.get("SLS2D_KIND", "2d")  # 2d (P3), 2dk / 2d2k (4352 B / 2 KiB slots)
        files = ([f for r in res.split(":") for f in glob.glob(os.path.join(r, f"{kind}_*_tests.csv"))] if res
                 else glob.glob(os.path.join(RES, "sls2d_*_tests.csv")))
        acc = defaultdict(list)
        for f in files:
            base = f[: -len("_tests.csv")]
            if re.search(r"_(tx|rx|sp|rp)\d+(_[a-z0-9]+)?$", base):
                continue  # a delay arm (N/S floor A/B), not a slot point
            a = cls.depths(base + ".log")
            if a is None:
                continue
            for r in csv.DictReader(open(f)):
                name = r["test_name"].split("_size_")[0].split("_ntype_")[0]
                if name in cls.rows and r["packet_size"] == size:
                    v = cycles_per_packet(r)
                    if v:
                        acc[(cls.rows[name][0], size, a)].append(v)
        return {k: statistics.mean(v) for k, v in acc.items()}

    @staticmethod
    def depths(log):
        for line in log_lines(log):
            m = re.search(r"VC0 sender slots \{([^}]*)\} receiver slots \{([^}]*)\}", re.sub(r"\x1b\[[0-9;]*m", "", line))
            if m and not re.search(r"\b0\b", m.group(1)):
                s = [int(x) for x in m.group(1).split(",")]
                return (s[0], s[1], int(m.group(2).split(",")[0]))
        return None


if os.environ.get("GLX_ROWS"):
    GlxVC1.ROWS = dict(kv.split("=") for kv in os.environ["GLX_ROWS"].split(","))
    GlxVC1.rows = {t: (c, ROW_WIN[c]) for t, c in GlxVC1.ROWS.items()}


class Provisional(am.Model):
    """v4gi's fit with each window kind's intercept moved to the lead's points where it has any (slope kept), and the
    lead's service bounds where measured."""

    def __init__(self):
        old = am.Model(am.VC1)
        lead = am.Model(LeadVC1)
        self.fam, self.data = LeadVC1, lead.data
        self.P = defaultdict(float, old.P)
        self.P.update({k: v for k, v in lead.P.items() if v})
        if ("inter + 2 VC1", "4352") not in lead.P:
            for s in ("16", "4352"):
                self.P[("inter + 2 VC1", s)] = self.P[("inter + VC1", s)]
        self.samples, self.fit, self.tstep = lead.samples, dict(old.fit), dict(old.tstep)
        for (kind, t, size), pts in lead.samples.items():
            base = old.fit.get((kind.rstrip("x"), t, size))
            if base is None:
                self.fit[(kind, t, size)] = am.line(pts)
                continue
            b = base[1]
            self.fit[(kind, t, size)] = (statistics.mean(x - b * w for w, x in pts), b)
        for (kind, t, size), v in list(self.fit.items()):  # the two-hop kinds default to the one-hop ones
            if kind in ("f1", "r1") and (kind + "x", t, size) not in self.fit:
                self.fit[(kind + "x", t, size)] = v


def model(family):
    if family == "v4gi":
        return am.Model(am.VC1)
    if family == "prov":
        return Provisional()
    fam = {"p150": LeadVC1, "glx": GlxVC1, "2d": Lead2D}[family]
    m = am.Model(fam)
    if any(k[0] == "worker" for k in m.fit):
        fam.worker_rt = None  # size the worker from its fitted window (allocate()'s step 2)
    for (kind, t, size), v in list(m.fit.items()):
        for x in ("x", "n"):
            if kind in ("f1", "r1") and (kind + x, t, size) not in m.fit:
                m.fit[(kind + x, t, size)] = v
    return m


def classes_of(m):
    return sorted({c for c, _ in m.fam.rows.values()})


def cmd_fit(family="p150"):
    m = model(family)
    print(f"[{family}] service bounds P (cycles/packet; reached, or the path's they extend):")
    for (cls, size), p in sorted(m.P.items()):
        print(f"  {cls:14} {size:>5} B  {p:6.1f}")
    print("\nWindows: rate x W = RT0 + b W (cycles; the points: W -> rate x W):")
    for (kind, t, size), (rt0, b) in sorted(m.fit.items()):
        pts = ", ".join(f"{W}: {x:.0f}" for W, x in sorted(m.samples.get((kind, t, size), [])))
        print(f"  {kind:5} T={t} {size:>5} B  RT0 {rt0:6.0f}  b {b:5.1f}   ({pts})")
    errs = []
    print("\nMeasured against the model (cycles/packet):")
    for (cls, size, a), rate in sorted(m.data.items()):
        pr, which = m.predict(cls, size, a, detail=True)
        errs.append(abs(pr - rate) / rate)
        print(f"  {cls:14} {size:>5} B  {str(a):22} measured {rate:6.1f}  model {pr:6.1f} ({which or 'P'})"
              f"  {100 * (pr - rate) / rate:+5.1f}%")
    if errs:
        print(f"\n{len(errs)} points: mean |error| {100 * statistics.mean(errs):.1f}%, max {100 * max(errs):.1f}%")


def cmd_alloc(family="p150", *weights):
    m = model(family)
    for arg in weights or ("1:1",):
        w = dict(zip(m.fam.classes, (float(x) for x in arg.split(":"))))
        a = am.allocate(m, w)
        rows = "  ".join(f"{c} {m.predict(c, '4352', a):6.1f}" for c in classes_of(m))
        print(f"[{family}] {arg:5} -> {','.join(map(str, a)):14}  4352 B: {rows}")


def cmd_predict(family, *allocs):
    m = model(family)
    for spec in allocs:
        a = tuple(int(x) for x in spec.split(","))
        for size in m.sizes():
            rows = "  ".join(f"{c} {m.predict(c, size, a):6.1f}" for c in classes_of(m))
            print(f"{spec:14} {size:>4} B  {rows}")


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "fit"), sys.argv[2:]
    {"fit": cmd_fit, "alloc": cmd_alloc, "predict": cmd_predict}[cmd](*args)
