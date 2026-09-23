import collections
import csv
import sys

path, wall = sys.argv[1], float(sys.argv[2])
agg = collections.defaultdict(lambda: [0, 0.0, set()])
for r in csv.DictReader(open(path)):
    op = (r["OP Code"] or "").strip()
    try:
        t = float(str(r["Device Time"]).replace("us", "").strip())
    except ValueError:
        continue
    e = agg[op]
    e[0] += 1
    e[1] += t
    e[2].add((r["Math Fidelity"] or "").strip()[:26])
tot = sum(v[1] for v in agg.values())
print("  %-44s %4s %9s %7s %8s  %s" % ("OP", "x", "us", "kernel%", "us/call", "fidelity"))
for k, (n, t, f) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    if t < 10:
        continue
    print(
        "  %-44s %4d %9.1f %6.1f%% %8.1f  %s"
        % (k[:44], n, t, 100 * t / tot, t / n, "/".join(sorted(x for x in f if x))[:28])
    )
print(
    "  %-44s %4d %9.1f   (traced wall %.3f ms; Tracy is untraced)"
    % ("kernel total", sum(v[0] for v in agg.values()), tot, wall)
)
