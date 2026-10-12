"""One run's device times from its tracy report: per matmul signature (op, in0 Y x X, in1 Y x X, fidelity) the call
count and the summed device kernel duration, and the sum over every op. usage: sum_ops.py <item> <variant> <run> <out dir>"""
import csv, glob, sys
item, v, r, d = sys.argv[1:5]
fs = glob.glob(d + "/reports/*/ops_perf_results_*.csv")
if not fs:
    print("RAS NOREPORT", item, v, r); sys.exit(0)
agg, tot, n = {}, 0.0, 0
for row in csv.DictReader(open(fs[0])):
    try:
        t = float(row["DEVICE KERNEL DURATION [ns]"])
    except (KeyError, ValueError):
        continue
    tot += t; n += 1
    op = row.get("OP CODE", "?")
    if "atmul" not in op and "inear" not in op:
        continue
    sig = "%s|%sx%s|%sx%s|%s" % (op.replace(" ", ""), row.get("INPUT_0_Y_PAD[LOGICAL]", "?").split("[")[0], row.get("INPUT_0_X_PAD[LOGICAL]", "?").split("[")[0],
                                 row.get("INPUT_1_Y_PAD[LOGICAL]", "?").split("[")[0], row.get("INPUT_1_X_PAD[LOGICAL]", "?").split("[")[0], row.get("MATH FIDELITY", "?"))
    c = agg.setdefault(sig, [0, 0.0]); c[0] += 1; c[1] += t
for sig, (k, s) in sorted(agg.items()):
    print("RAS", item, v, r, sig, k, round(s))
print("RAS", item, v, r, "ALL", n, round(tot))
