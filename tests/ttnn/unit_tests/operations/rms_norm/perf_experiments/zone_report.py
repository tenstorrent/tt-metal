import sys, csv, collections


# NOTE: this file lives under ttnn/ttnn/operations/, and ttnn/ttnn/operations/__init__.py
# exec_module()s EVERY .py it walks at `import ttnn` time. Module-level work here therefore
# ran on every ttnn import in the repo (it parsed sys.argv[1] -> ValueError under pytest,
# breaking `import ttnn` for every test). Keep all work inside main().
def main():
    path = "generated/profiler/.logs/profile_log_device.csv"
    FREQ = 1350.0
    lines = open(path).readlines()
    rd = list(csv.reader(lines[2:]))
    runids = sorted({int(r[7]) for r in rd if len(r) > 7 and r[7].strip().isdigit()})
    print("run host ids:", runids)
    want = int(sys.argv[1]) if len(sys.argv) > 1 else runids[-1]
    open_stack = collections.defaultdict(list)
    dur = collections.defaultdict(list)
    for r in rd:
        if len(r) < 12 or not r[7].strip().isdigit() or int(r[7]) != want:
            continue
        core = (r[1].strip(), r[2].strip())
        risc = r[3].strip()
        t = int(r[5])
        zone = r[10].strip()
        typ = r[11].strip()
        k = (core, risc, zone)
        if typ == "ZONE_START":
            open_stack[k].append(t)
        elif typ == "ZONE_END" and open_stack[k]:
            dur[k].append(t - open_stack[k].pop())
    per = collections.defaultdict(lambda: collections.defaultdict(float))
    cnt = collections.defaultdict(lambda: collections.defaultdict(int))
    for (core, risc, zone), ds in dur.items():
        per[(risc, zone)][core] += sum(ds)
        cnt[(risc, zone)][core] += len(ds)
    print(f"run={want}  {'RISC':8s} {'zone':22s} {'ncores':>6s} {'inst':>6s} {'ns avg':>10s} {'ns max':>10s}")
    out = []
    for (risc, zone), m in per.items():
        n = len(m)
        tot = list(m.values())
        inst = sum(cnt[(risc, zone)].values()) / n
        out.append((risc, zone, n, inst, sum(tot) / n / FREQ * 1000, max(tot) / FREQ * 1000))
    for risc, zone, n, inst, a, mx in sorted(out, key=lambda z: (z[0], -z[5])):
        print(f"{'':13s}{risc:8s} {zone:22s} {n:6d} {inst:6.1f} {a:10.0f} {mx:10.0f}")


if __name__ == "__main__":
    main()
