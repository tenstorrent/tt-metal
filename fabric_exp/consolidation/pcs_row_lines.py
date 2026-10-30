#!/usr/bin/env python3
"""pcs_row_lines.py CACHE GDIR PREFIX ROW... [--erisc 0|1] [--top N]: per source line, where the busiest sending router ERISC
of each sampled row spends its time (innermost inline frame), as a share of its kernel samples and that share times the
row's cycles/packet. Same sample mapping as pcs_profile.py (pcsv_runs.sh samples, -g twins from pcs_sections.py twins);
the busiest sender is the target with the most samples in the send (v4 `sender::send<`)."""
import argparse, collections, csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pcs_sections as ps  # noqa: E402
import pcs_profile as pp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cache"); ap.add_argument("gdir"); ap.add_argument("prefix"); ap.add_argument("rows", nargs="+")
ap.add_argument("--erisc", type=int, default=0); ap.add_argument("--top", type=int, default=25)
a = ap.parse_args()
for row in a.rows:
    vals = [float(r["avg_packets_per_s"]) for r in csv.DictReader(open(f"{ps.RES}/{a.prefix}_{row}_13129088_trial.csv"))]
    cpp = 1.35e9 / (sum(vals) / len(vals))
    best = None
    a2l = {}
    for i, (tgt, h, addrs) in pp.target_samples(a.cache, a.prefix, row).items():
        if tgt["erisc"] != a.erisc or not addrs:
            continue
        proc = "active_erisc" if tgt["erisc"] == 0 else "subordinate_active_erisc"
        twin = f"{a.gdir}/{h}/{proc}/{proc}.elf"
        if not os.path.exists(twin):
            continue
        cache = a2l.setdefault(twin, {})
        need = sorted(set(addrs) - set(cache))
        if need:
            cache.update(ps.a2l(twin, need))
        sends = sum(1 for x in addrs if any("sender::send<" in f for f, _, _ in cache.get(x, [])))
        if best is None or sends > best[0]:
            best = (sends, tgt, [cache.get(x, []) for x in addrs])
    if best is None:
        print(f"\n### {row}: no target"); continue
    sends, tgt, frames = best
    n = len(frames)
    lines = collections.Counter()
    for fr in frames:
        f, fl, ln = fr[0] if fr else ("?", "?", 0)
        lines[f"{fl}:{ln}"] += 1
    print(f"\n### {row}: {cpp:.1f} cycles/packet; E{tgt['erisc']} chip {tgt['chip']} chan {tgt['chan']}, {n} samples ({100*sends/n:.0f}% in send)")
    for k, c in lines.most_common(a.top):
        print(f"  {c / n * cpp:6.1f}  {100 * c / n:5.1f}%  {k}")
