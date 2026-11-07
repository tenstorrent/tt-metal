"""Zone timeline of one core from a device profiler log (perf probes with AGRMS_PROFILE_ZONES=1).
usage: python zone_timeline.py generated/profiler/.logs/profile_log_device.csv <chip> <noc_x> <noc_y>"""
import csv, sys, collections

rows = list(csv.reader(open(sys.argv[1])))[2:]
chip, x, y = sys.argv[2], sys.argv[3], sys.argv[4]
last = max(int(r[7]) for r in rows if r[0] == chip)
sel = sorted(
    (r for r in rows if r[0] == chip and int(r[7]) == last and r[1] == x and r[2] == y), key=lambda r: int(r[5])
)
t0 = min(int(r[5]) for r in sel)
st = collections.defaultdict(list)
for r in sel:
    name = r[10]
    risc = r[3]
    if not name.startswith("AG_"):
        continue
    if r[11] == "ZONE_START":
        st[(risc, name)].append(int(r[5]))
    elif st[(risc, name)]:
        s = st[(risc, name)].pop()
        print(f"{risc:6s} {name:16s} {(s-t0)/1350:7.2f} {(int(r[5])-s)/1350:6.2f}")
