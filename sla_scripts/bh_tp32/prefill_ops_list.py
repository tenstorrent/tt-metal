import re, sys, pandas as pd

d = sys.argv[1]
dev = pd.read_csv(f"{d}/cpp_device_perf_report.csv", low_memory=False)
names = {}
for line in open(f"{d}/tracy_ops_data.csv", errors="ignore"):
    m = re.search(r'TT_DNN_DEVICE_OP:\s*"([A-Za-z0-9_:]+)",\s*[0-9]+,\s*[0-9]+,\s*\w+,\s*([0-9]+)', line)
    if m:
        names[int(m.group(2))] = m.group(1)
dev["OP"] = dev["GLOBAL CALL COUNT"].map(names)
K = "DEVICE KERNEL DURATION [ns]"
print([c for c in dev.columns][:40])
for D in sorted(dev["DEVICE ID"].unique()):
    d0 = dev[dev["DEVICE ID"] == D].sort_values("GLOBAL CALL COUNT").reset_index(drop=True)
    emb = [i for i, n in enumerate(d0["OP"]) if isinstance(n, str) and "Embedding" in n]
    seg = d0.iloc[emb[-1] if emb else 0 :]
    print("DEVICE", D, "ops", len(seg), "kernel ms %.2f" % (seg[K].sum() / 1e6))
    if D == 0:
        for _, r in seg.iterrows():
            if r[K] > 15000:
                print(
                    "%9d %-40s %8.1f us cores=%s"
                    % (r["GLOBAL CALL COUNT"], str(r["OP"])[:40], r[K] / 1e3, r.get("CORE COUNT", ""))
                )
