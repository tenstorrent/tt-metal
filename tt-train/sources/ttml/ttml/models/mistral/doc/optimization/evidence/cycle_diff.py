"""Kernel cycle-time diff between two ranked per-op tables (tp_rank_ops.py output): per chip the totals, the
collectives and the kernels without them, then every row whose ms/step changed by more than `--min` (or appeared /
vanished). Usage: cycle_diff.py OLD_ranked_ops_full.txt NEW_ranked_ops_full.txt --label "42 -> 44" [--min 0.01]"""
import argparse, re

p = argparse.ArgumentParser()
p.add_argument("old")
p.add_argument("new")
p.add_argument("--label", default="")
p.add_argument("--min", type=float, default=0.01)
args = p.parse_args()
COLL = ("ReduceScatter", "AllGather", "AllBroadcast", "AllReduce")


def parse(path):
    chips, chip = {}, None
    for line in open(path):
        m = re.match(r"chip (\d+):", line)
        if m:
            chip = int(m.group(1))
            chips[chip] = {}
            continue
        m = re.match(r"\s+([\d.]+)\s+(\d+)\s+(.+?)\s*$", line)
        if m and chip is not None and "|" in m.group(3):
            key = m.group(3).strip()
            chips[chip][key] = chips[chip].get(key, 0.0) + float(m.group(1))
    return chips


old, new = parse(args.old), parse(args.new)
print(f"kernel ms per profiled 4-layer step, {args.label} ({args.old.split('/')[-1]} vs {args.new.split('/')[-1]})")
for chip in sorted(set(old) | set(new)):
    o, n = old.get(chip, {}), new.get(chip, {})

    def tot(d, coll):
        return sum(v for k, v in d.items() if k.startswith(COLL) == coll)

    to, tn = sum(o.values()), sum(n.values())
    co, cn = tot(o, True), tot(n, True)
    print(
        f"chip {chip}: total {to:.3f} -> {tn:.3f} ({tn - to:+.3f}); collectives {co:.3f} -> {cn:.3f} ({cn - co:+.3f}); "
        f"kernels without collectives {to - co:.3f} -> {tn - cn:.3f} ({(tn - cn) - (to - co):+.3f})"
    )
    rows = [(abs(n.get(k, 0.0) - o.get(k, 0.0)), k) for k in set(o) | set(n)]
    for d, k in sorted(rows, reverse=True):
        if d < args.min:
            break
        print(f"  {o.get(k, 0.0):7.3f} -> {n.get(k, 0.0):7.3f} ({n.get(k, 0.0) - o.get(k, 0.0):+.3f})  {k}")
