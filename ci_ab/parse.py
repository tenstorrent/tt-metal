import csv, os, re, sys
out, arm, rnd = sys.argv[1:4]
d = os.path.join(out, ".logs")
rows = list(csv.DictReader(open(os.path.join(d, "cpp_device_perf_report.csv"))))
names = {}
q = os.path.join(d, "tracy_ops_data.csv")
if os.path.exists(q):
    names = {int(n): op for op, n in re.findall(r'TT_DNN_DEVICE_OP: "([^"]+)", \d+, \d+, \w+, (\d+)', open(q).read())}
i = 0
for r in rows:
    try:
        t = int(r["DEVICE KERNEL DURATION [ns]"])
    except Exception:
        continue
    if t >= 3000:
        print(f"RES {arm} r{rnd} op{i} {names.get(int(r['GLOBAL CALL COUNT']), '?')} {t} ns")
        i += 1
