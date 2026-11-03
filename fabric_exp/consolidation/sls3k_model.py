#!/usr/bin/env python3
"""sls3k_model.py SIZE [CAP]: the slot rule at an intermediate slot payload (3072 / 3584 B) on the Galaxy, before any run
at that size (SLOT_STUDY.md, "3-3.5 KiB slots"). sls_model.py's window model, with each window kind's round trip
interpolated linearly in payload between the Galaxy fits at 2048 B (2 KiB slots, jobs 167648 + 167673) and at 4352 B
(4352 B slots, job 167464), and each class's bound = max(its router bound, the 200G wire at SIZE): router bounds from the
2048 B fit (per packet), the VC1 path's interpolated (it grew from 276 at 2048 B to ~297 at 4352 B), the wire
12.80 + P / 18.50 cycles/packet (+3.0 above ~4 KiB; RESULTS §98, job 155281). Hypothesis-grade by construction.

  sls3k_model.py vc1 SIZE CAP     the VC1 split: windows, bounds, the rule's pick per weight and its rows
  sls3k_model.py 2d SIZE CAP      the 2D split (VC0 only): the rule by hand (receiver 16 / 32, the worker that binds no
                                  1-hop row at 1 KiB and at SIZE, forwarding the rest) and the rows' bounds
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(HERE), "results")
sys.path.insert(0, HERE)


def wire(p):
    return 12.80 + p / 18.50 + (3.0 if p + 96 > 4096 else 0.0)


def vc1_models():
    os.environ.update(GLX_RES=f"{R}/glx167648:{R}/glx167673", GLX_KIND="vc12k", GLX_CAP="169")
    import importlib
    import sls_model as s
    importlib.reload(s)
    m2 = s.model("glx")
    os.environ.update(GLX_RES=f"{R}/glx167464", GLX_KIND="vc1", GLX_CAP="81")
    s = importlib.reload(s)
    m4 = s.model("glx")
    return s, m2, m4


def cmd_vc1(size, cap):
    s, m2, m4 = vc1_models()
    am = s.am
    f = (size - 2048) / (4352 - 2048)
    S = str(size)

    class Fam(s.GlxVC1):
        capacity = cap
    m = am.Model.__new__(am.Model)
    m.fam, m.data, m.samples, m.tstep = Fam, {}, {}, {}
    m.fit, m.P = {}, {}
    print(f"[vc1 @ {size} B, {cap} slots] windows (rate x W = RT0 + b W), interpolated {1 - f:.2f} x 2048 B + {f:.2f} x 4352 B:")
    for (kind, t, sz), (rt0, b) in sorted(m2.fit.items()):
        if sz != "2048":
            continue
        o = m4.fit.get((kind, t, "4352"))
        if o is None:
            m.fit[(kind, t, S)] = (rt0, b)
            note = "2048 B only"
        else:
            m.fit[(kind, t, S)] = (rt0 + f * (o[0] - rt0), b + f * (o[1] - b))
            note = f"2048 B {rt0:.0f} + {b:.1f} W, 4352 B {o[0]:.0f} + {o[1]:.1f} W"
        print(f"  {kind:5} T={t}  {m.fit[(kind, t, S)][0]:6.0f} + {m.fit[(kind, t, S)][1]:5.1f} W   ({note})")
    print(f"bounds (max(router, wire {wire(size):.1f})):")
    for (cls, sz), p2 in sorted(m2.P.items()):
        if sz != "2048":
            continue
        p4 = m4.P.get((cls, "4352"))
        router = p2
        if cls.startswith("inter +") and p4:  # the VC1 path: router-bound at both sizes
            router = p2 + f * (p4 - p2)
        m.P[(cls, S)] = max(router, wire(size))
        print(f"  {cls:16} router {router:6.1f} -> {m.P[(cls, S)]:6.1f}")
    picks = {}
    for wt in ("1:2", "1:1", "3:2", "2:1", "3:1"):
        w = dict(zip(Fam.classes, (float(x) for x in wt.split(":"))))
        picks[wt] = am.allocate(m, w, size=S)
    print("rule's picks:", "; ".join(f"{k} {','.join(map(str, v))}" for k, v in picks.items()))
    for a in sorted(set(picks.values())):
        print(f"  {','.join(map(str, a)):16} " + "  ".join(
            f"{c} {m.predict(c, S, a):.1f}" for c in sorted({c for c, _ in Fam.rows.values()})))
    return m, Fam


def cmd_predict(size, cap, *allocs):
    m, Fam = cmd_vc1(size, cap)
    S = str(size)
    for spec in allocs:
        a = tuple(int(x) for x in spec.split(","))
        print(f"  {spec:16} " + "  ".join(f"{c} {m.predict(c, S, a, detail=True)[0]:.1f} [{m.predict(c, S, a, detail=True)[1] or 'P'}]"
                                         for c in sorted({c for c, _ in Fam.rows.values()})))


if __name__ == "__main__":
    what, size, cap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if what == "vc1":
        cmd_vc1(size, cap)
    elif what == "predict":
        cmd_predict(size, cap, *sys.argv[4:])
