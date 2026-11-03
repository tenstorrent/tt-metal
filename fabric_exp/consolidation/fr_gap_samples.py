#!/usr/bin/env python3
"""fr_gap_samples.py: the point-5 (Galaxy full-ring multicast 4352 B) PC-sample tables of FULLRING_MCAST_GAP.md, from
the phase-2 sample sets (report/breakdown/sets_phase2.json) through report/scripts/bd_map.py (read only; the shared
builder is not touched).

  fr_gap_samples.py timeline SET TARGET,...        work share per 2.5 s bin (finds the idle start of a window)
  fr_gap_samples.py classes SET ERISC [--tmin S]   per ring class, mean cycles/packet by category and rule
  fr_gap_samples.py split SET [--tmin S]           E0: send-path and TX-queue-spin samples by calling channel visit

Ring classes (the note's section 3): R8 = channels 2 and 4 (the 8-chip rings, 7 flows per link); R4sat = channels
0/6/8 whose E0 spends > 50% in work categories (TX-queue saturated 4-chip-ring routers); R4low = the other channels
0/6/8. A set's R4sat list is derived from its own E0 samples. Cycles/packet = sample share x the sampled run's rate.
--tmin drops samples before S seconds into the window (glx_169481_b_new_p5 starts traffic at ~17.5 s)."""
import argparse
import collections
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "report", "scripts"))
import bd_map as B  # noqa: E402

SETS = os.environ.get("FRG_SETS") or os.path.join(HERE, "..", "report", "breakdown", "sets_phase2.json")


def targets(s, erisc, tmin):
    """Yield (chip, chan, [elf addresses], chains) for each sampled target of one ERISC kind."""
    meta, per = B.load_samples(os.path.join(B.RES, s["prefix"]))
    hashes = B.log_hashes(s["logs"])
    t0 = min(t for v in per.values() for t, _ in v)
    for i, t in enumerate(meta["targets"]):
        if erisc is not None and t["erisc"] != erisc:
            continue
        key = f"{t['chip']}/{t['chan']}"
        lay = meta.get("layouts_end", {}).get(key) or meta.get("layouts", {}).get(key, {})
        bs = B.find_builds(s["cache"], t["erisc"], lay, hashes)
        if not bs:
            continue
        elf, _, _, _, base = bs[0]
        e = lay["erisc"][str(t["erisc"])]
        load, size = e["load"], e["text_size"]
        smp = [(tt - t0, pc - load + base) for tt, pc in per[i] if load <= pc < load + size and tt - t0 >= tmin]
        if len(smp) < 200:
            continue
        yield t, smp, B.a2l(elf, sorted({a for _, a in smp}))


def work_share(clf, mp, erisc, smp, ch):
    w = mp["work_categories"][f"E{erisc}"]
    return sum(clf.classify(ch[a])[0] in w for _, a in smp) / len(smp)


def saturated(s, mp, clf, tmin):
    return {f"{t['chip']}/{t['chan']}" for t, smp, ch in targets(s, 0, tmin)
            if t["chan"] in (0, 6, 8) and work_share(clf, mp, 0, smp, ch) > 0.5}


def ring_class(t, sat):
    key = f"{t['chip']}/{t['chan']}"
    if t["chan"] in (2, 4):
        return "R8"
    if t["chan"] in (0, 6, 8):
        return "R4sat" if key in sat else "R4low"
    return None


def cmd_timeline(a, s, mp, clf):
    want = set(a.targets.split(","))
    for t, smp, ch in targets(s, None, 0):
        name = f"{t['chip']}/{t['chan']}/E{t['erisc']}"
        if name not in want:
            continue
        w = mp["work_categories"][f"E{t['erisc']}"]
        bins = collections.defaultdict(lambda: [0, 0])
        for tt, adr in smp:
            b = bins[int(tt // 2.5)]
            b[1] += 1
            b[0] += clf.classify(ch[adr])[0] in w
        print(f"{name:10} " + " ".join(f"{100 * v[0] / v[1]:3.0f}" for _, v in sorted(bins.items())))


def cmd_classes(a, s, mp, clf):
    rate, _ = B.row_rate(s)
    sat = saturated(s, mp, clf, a.tmin)
    cls = collections.defaultdict(list)
    for t, smp, ch in targets(s, a.erisc, a.tmin):
        g = ring_class(t, sat)
        if g is None:
            continue
        cnt = collections.Counter()
        for _, adr in smp:
            cat, rid, _, _ = clf.classify(ch[adr])
            cnt["cat:" + cat] += 1
            cnt["rule:" + str(rid)] += 1
        cls[g].append({k: v / len(smp) * rate for k, v in cnt.items()})
    order = [g for g in ("R8", "R4sat", "R4low") if cls[g]]
    print(f"{a.set} E{a.erisc}: rate {rate:.1f}; targets " + ", ".join(f"{g} {len(cls[g])}" for g in order))
    keys = collections.Counter()
    for g in order:
        for d in cls[g]:
            for k, v in d.items():
                keys[k] += v
    for k, _ in sorted(keys.items(), key=lambda kv: (not kv[0].startswith("cat:"), -kv[1])):
        vals = [st.mean(d.get(k, 0) for d in cls[g]) for g in order]
        if max(vals) >= 1.0:
            print(f"  {k:36} " + " ".join(f"{g}={v:6.1f}" for g, v in zip(order, vals)))


def cmd_split(a, s, mp, clf):
    rate, _ = B.row_rate(s)
    sat = saturated(s, mp, clf, a.tmin)
    new = s["map"] == "new"
    cls = collections.defaultdict(list)
    for t, smp, ch in targets(s, 0, a.tmin):
        g = ring_class(t, sat)
        if g is None:
            continue
        c = collections.Counter()
        for _, adr in smp:
            fns = [(fn, ln) for fn, _, ln in ch[adr]]
            if new:  # v4: send<CH>() inlined per channel visit; the post-send spin is router_sender.hpp:313-314
                if not any(fn == "sender::send" for fn, _ in fns):
                    continue
                who = "worker" if any(fn == "sender::visit_worker" for fn, _ in fns) else "forwarding"
                spin = any(fn == "sender::send" and ln == 314 for fn, ln in fns)
            else:  # main: send_next_data:603 is the post-send spin; the per-channel lambda is operator:3122 (ch 0)
                if not any(fn == "send_next_data" for fn, _ in fns):
                    continue
                ops = [ln for fn, ln in fns if fn == "operator"]
                who = "worker" if ops and ops[0] == 3122 else "forwarding"
                spin = any(fn == "send_next_data" and ln == 603 for fn, ln in fns)
            c[("spin" if spin else "send", who)] += 1
        cls[g].append({k: v / len(smp) * rate for k, v in c.items()})
    print(f"{a.set} E0 send path by channel (cycles/packet at {rate:.1f})")
    for g in ("R8", "R4sat", "R4low"):
        L = cls[g]
        if not L:
            continue
        m = {k: st.mean(d.get(k, 0) for d in L) for k in {k for d in L for k in d}}
        w = m.get(("send", "worker"), 0) + m.get(("spin", "worker"), 0)
        f = m.get(("send", "forwarding"), 0) + m.get(("spin", "forwarding"), 0)
        print(f"  {g:6} n={len(L):3}  send w/f {m.get(('send', 'worker'), 0):5.1f} / {m.get(('send', 'forwarding'), 0):5.1f}"
              f"  spin w/f {m.get(('spin', 'worker'), 0):5.1f} / {m.get(('spin', 'forwarding'), 0):5.1f}"
              f"  worker share of send-path samples {w / max(w + f, 1e-9):.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("timeline"); p.add_argument("set"); p.add_argument("targets")
    p = sub.add_parser("classes"); p.add_argument("set"); p.add_argument("erisc", type=int)
    p.add_argument("--tmin", type=float, default=0.0)
    p = sub.add_parser("split"); p.add_argument("set"); p.add_argument("--tmin", type=float, default=0.0)
    a = ap.parse_args()
    s = B.load_sets(SETS)[a.set]
    mp = B.load_map(s["map"])
    clf = B.Classifier(mp, B.Sources(s["sources"]))
    {"timeline": cmd_timeline, "classes": cmd_classes, "split": cmd_split}[a.cmd](a, s, mp, clf)


if __name__ == "__main__":
    main()
