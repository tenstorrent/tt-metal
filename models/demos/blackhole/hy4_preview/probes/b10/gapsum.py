import collections
import csv
import statistics as st
import sys

f = sys.argv[1]
rows = [r for r in csv.DictReader(open(f)) if r["METAL TRACE ID"] != ""]
sess = sorted({int(r["METAL TRACE REPLAY SESSION ID"]) for r in rows})
S = sess[-1]
per = collections.defaultdict(dict)
for r in rows:
    if int(r["METAL TRACE REPLAY SESSION ID"]) != S:
        continue
    g = int(r["GLOBAL CALL COUNT"])
    fl = lambda k: float(r[k]) if r[k] else 0.0
    per[g >> 10][g & 1023] = (fl("OP TO OP LATENCY [ns]"), fl("DEVICE KERNEL DURATION [ns]"))
ids = sorted(per)[1:]
gm = [st.mean(v[0] for v in per[i].values()) / 1e3 for i in ids]
km = [st.mean(v[1] for v in per[i].values()) / 1e3 for i in ids]
big = [g for g in gm if g > 3]
print(
    f,
    "sessions",
    sess,
    "ops",
    len(ids),
    "chips",
    len(per[ids[0]]),
    "sum kernel %.0f us sum gap %.0f us (%.1f%%)" % (sum(km), sum(gm), 100 * sum(gm) / (sum(km) + sum(gm))),
)
print(
    " gaps>3us: n=%d sum %.0f us; <=3us: n=%d sum %.0f us mean %.2f; hist >3:"
    % (len(big), sum(big), len(gm) - len(big), sum(g for g in gm if g <= 3), st.mean(g for g in gm if g <= 3)),
    sorted(round(g, 1) for g in big)[:60],
)
