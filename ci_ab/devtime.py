#!/usr/bin/env python3
"""Scratch, not for merge. Per tracy run: the device kernel time of every op over 50 us, per device, in call order.
--summary: per test (the k-th long op) and arm, the per-device times over rounds; banked against onebank and banked2
against banked, as the change of the pooled median and of the mean of per-device, per-round paired ratios."""
import collections, csv, glob, json, os, re, statistics, sys

COL = "DEVICE KERNEL DURATION [ns]"
OP = re.compile(r'TT_DNN_DEVICE_OP: "([^"]+)", \d+, \d+, \w+, (\d+)')


def long_ops(out):
    logs = os.path.join(out, ".logs")
    p = os.path.join(logs, "cpp_device_perf_report.csv")
    if not os.path.exists(p):
        return []
    rows = list(csv.DictReader(open(p)))
    names = {}
    q = os.path.join(logs, "tracy_ops_data.csv")
    if os.path.exists(q):
        names = {int(n): op for op, n in OP.findall(open(q).read())}
    calls = collections.defaultdict(dict)
    for r in rows:
        try:
            d = int(r[COL])
        except (ValueError, KeyError):
            continue
        if d < int(os.environ.get("THRESH_NS", "50000")):
            continue
        g = int(r["GLOBAL CALL COUNT"])
        calls[g][int(r["DEVICE ID"])] = d
    return [(g, names.get(g, "?"), calls[g]) for g in sorted(calls)]


def one(out, arm, rnd):
    ops = long_ops(out)
    res = []
    for k, (g, name, dev) in enumerate(ops):
        v = sorted(dev.values())
        print(f"   {arm} r{rnd} op{k} call {g} {name}: {len(v)} devices, median {statistics.median(v)} ns, "
              f"min {v[0]} max {v[-1]}; " + " ".join(f"d{d}={t}" for d, t in sorted(dev.items())))
        res.append({"op": k, "name": name, "dev": dev})
    json.dump(res, open(out + ".json", "w"))


def summary(root):
    data = collections.defaultdict(lambda: collections.defaultdict(dict))  # [op][arm][(rnd, dev)] = ns
    for p in glob.glob(os.path.join(root, "*.json")):
        arm, rnd = re.match(r"(\w+?)_(\d+)\.json$", os.path.basename(p)).groups()
        for o in json.load(open(p)):
            for d, t in o["dev"].items():
                data[o["op"]][arm][(int(rnd), int(d))] = t
    for op in sorted(data):
        arms = data[op]
        print(f"== op{op}: " + ", ".join(f"{a} n={len(arms[a])} median {statistics.median(arms[a].values()):.0f} ns"
                                         for a in sorted(arms)))
        for a, b in (("onebank", "banked"), ("banked", "banked2")):
            if a not in arms or b not in arms:
                continue
            keys = sorted(set(arms[a]) & set(arms[b]))
            if not keys:
                continue
            rat = [arms[b][k] / arms[a][k] - 1 for k in keys]
            pm = statistics.median(arms[b].values()) / statistics.median(arms[a].values()) - 1
            print(f"   {b} vs {a}: pooled median {100 * pm:+.2f}%, paired mean {100 * statistics.mean(rat):+.2f}% "
                  f"(sd {100 * statistics.pstdev(rat):.2f}%, n {len(rat)}, min {100 * min(rat):+.2f}%, "
                  f"max {100 * max(rat):+.2f}%)")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2])
    else:
        one(*sys.argv[1:4])
