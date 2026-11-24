import csv, collections, sys

f = "generated/profiler/.logs/profile_log_device.csv"
rows = list(csv.reader(open(f)))[2:]
# key: (dev, run_id)
by = collections.defaultdict(list)
for r in rows:
    dev, x, y, risc, tid, t, data, run = r[0], int(r[1]), int(r[2]), r[3], r[4], int(r[5]), r[6], r[7]
    name, typ = r[10], r[11]
    by[(dev, run)].append((x, y, risc, name, typ, t))
for dev in sorted({k[0] for k in by}):
    last = max((k[1] for k in by if k[0] == dev), key=int)
    ev = by[(dev, last)]
    t0 = min(t for *_, name, typ, t in ev if name.endswith("-KERNEL") and typ == "ZONE_START")
    t1 = max(t for *_, name, typ, t in ev if name.endswith("-KERNEL") and typ == "ZONE_END")
    print(f"dev {dev}: kernel span {(t1-t0)/1350:.1f} us")
    starts = {}
    z = collections.defaultdict(list)  # name -> [(core, s, e)]
    for x, y, risc, name, typ, t in ev:
        k = (x, y, risc, name)
        if typ == "ZONE_START":
            starts.setdefault(k, []).append(t)
        elif typ == "ZONE_END" and starts.get(k):
            s = starts[k].pop()
            z[name].append(((x, y, risc), (s - t0) / 1350, (t - t0) / 1350))
    for name in [
        "psd_hdr_setup",
        "psd_open",
        "psd_fence",
        "psd_wait_ready",
        "psd_send",
        "prd_all",
        "rd_w0",
        "rd_local_a",
        "rd_wait_remote",
        "cmp_head",
        "wr_head",
    ]:
        L = z.get(name, [])
        if not L:
            continue
        if name == "cmp_head":
            L = [l for l in L if l[0][2] == "TRISC_0"]
        ss = sorted(s for _, s, _ in L)
        es = sorted(e for _, _, e in L)
        ds = sorted(e - s for _, s, e in L)
        q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
        print(
            f"  {name:15s} n={len(L):3d} start[min/med/max] {ss[0]:6.1f} {q(ss,.5):6.1f} {ss[-1]:6.1f}  end {es[0]:6.1f} {q(es,.5):6.1f} {es[-1]:6.1f}  dur med {q(ds,.5):6.1f} max {ds[-1]:6.1f}"
        )
