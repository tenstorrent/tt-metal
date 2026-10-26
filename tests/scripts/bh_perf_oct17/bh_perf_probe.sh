#!/bin/bash
# Oct 17 Blackhole eltwise perf investigation: measure a few dashboard ops exactly as the dashboard does
# (ttperf, DEVICE KERNEL DURATION, bf16 1x1x32x32), record the machine, and break one run of each op down per core.
# Runs from /work inside the Custom test dispatch container. Usage: bh_perf_probe.sh [runs]
set -u
RUNS=${1:-5}
TESTS="test_identity test_relu test_exp test_add test_mul test_where test_lerp test_mean test_max test_assign"
cd /work
echo "=== PROBE commit $(git rev-parse HEAD 2>/dev/null) host $(hostname)"
echo "=== PROBE machine"
for f in /sys/class/tenstorrent/*/tt_card_type /sys/class/tenstorrent/*/tt_serial /sys/class/tenstorrent/*/tt_fw_bundle_ver \
         /sys/class/tenstorrent/*/tt_arc_fw_ver /sys/class/tenstorrent/*/tt_eth_fw_ver /sys/class/tenstorrent/*/tt_m3app_fw_ver \
         /sys/class/tenstorrent/*/tt_aiclk /sys/class/tenstorrent/*/tt_axiclk /sys/class/tenstorrent/*/tt_arcclk \
         /sys/module/tenstorrent/version; do
    [ -r "$f" ] && echo "MACHINE $f = $(cat "$f" 2>/dev/null)"
done
ls /sys/class/tenstorrent/*/ 2>/dev/null | head -40
command -v tt-smi >/dev/null && tt-smi -s 2>/dev/null | head -60

command -v ttperf >/dev/null || uv pip install ttperf
curl -fsSL https://raw.githubusercontent.com/Aswincloud/ttnn-performance-dashboard/main/test_eltwise_operations.py \
    -o /work/test_eltwise_operations.py || exit 1
export PERF_SHAPE=1,1,32,32 PERF_DTYPE=bfloat16

for r in $(seq 1 "$RUNS"); do
    for t in $TESTS; do
        out=$(ttperf "test_eltwise_operations.py::TestEltwiseOperations::$t" 2>&1)
        d=$(echo "$out" | grep -oE 'DEVICE KERNEL DURATION \[ns\] total:\s+[0-9.]+' | grep -oE '[0-9.]+$')
        echo "RESULT $t run$r ${d:-FAIL}"
        if [ "$r" = 1 ]; then
            [ -z "$d" ] && echo "$out" | tail -30
            csv=$(ls -t $(find /work/generated -name profile_log_device.csv 2>/dev/null) 2>/dev/null | head -1)
            [ -n "$csv" ] && python3 - "$csv" "$t" <<'EOF'
import collections, csv, sys
path, test = sys.argv[1], sys.argv[2]
with open(path) as f:
    head = f.readline().strip()
    rows = list(csv.reader(f))
hdr = [h.strip() for h in rows[0]]
ix = {h: i for i, h in enumerate(hdr)}
tcol = next(h for h in hdr if h.startswith("time["))
rcol = next(h for h in hdr if "RISC" in h)
runcol = next(h for h in hdr if "run host ID" in h)
pcol = next(h for h in hdr if h == "type" or h.startswith("zone phase"))
spans = collections.defaultdict(lambda: collections.defaultdict(lambda: [None, None]))
for r in rows[1:]:
    if not r[ix["zone name"]].strip().endswith("-KERNEL"):
        continue
    key = (int(r[ix["core_x"]]), int(r[ix["core_y"]]), r[ix[rcol]].strip())
    t = int(r[ix[tcol]])
    s = spans[int(r[ix[runcol]])][key]
    if "START" in r[ix[pcol]]:
        s[0] = t if s[0] is None else min(s[0], t)
    else:
        s[1] = t if s[1] is None else max(s[1], t)
print(f"PERCORE {test} header: {head}")
for run, sp in sorted(spans.items()):
    sp = {k: v for k, v in sp.items() if None not in v}
    if not sp:
        continue
    t0 = min(v[0] for v in sp.values())
    t1 = max(v[1] for v in sp.values())
    core = collections.defaultdict(lambda: [None, None])
    for (x, y, risc), (s, e) in sp.items():
        c = core[(x, y)]
        c[0] = s if c[0] is None else min(c[0], s)
        c[1] = e if c[1] is None else max(c[1], e)
    starts = sorted(v[0] - t0 for v in core.values())
    ends = sorted(v[1] - t0 for v in core.values())
    longest = max(core.items(), key=lambda kv: kv[1][1] - kv[1][0])
    first = min(core.items(), key=lambda kv: kv[1][0])
    last = max(core.items(), key=lambda kv: kv[1][1])
    q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
    print(
        f"PERCORE {test} run={run} cores={len(core)} total={t1 - t0}cyc "
        f"start_offsets[min/p50/p90/max]={starts[0]}/{q(starts, .5)}/{q(starts, .9)}/{starts[-1]} "
        f"end_offsets[min/p50/p90/max]={ends[0]}/{q(ends, .5)}/{q(ends, .9)}/{ends[-1]} "
        f"work_core={longest[0]} start={longest[1][0] - t0} end={longest[1][1] - t0} "
        f"first_start_core={first[0]} last_end_core={last[0]}"
    )
    if len(core) > 1 and test in ("test_identity", "test_add"):
        xs = sorted({x for x, _ in core}); ys = sorted({y for _, y in core})
        print(f"STARTMAP {test} run={run} (rows=y, cols=x {xs}) first kernel start per core, cycles after first core")
        for y in ys:
            print(f"STARTMAP {test} y={y:2d} " + " ".join(f"{core[(x, y)][0] - t0:5d}" if (x, y) in core else "    -" for x in xs))
    if len(core) > 1:
        for (x, y, risc), (s, e) in sorted(sp.items()):
            if (x, y) in (longest[0], first[0], last[0]):
                print(f"PERCORE {test} run={run}   core=({x},{y}) {risc:8s} start={s - t0} end={e - t0}")
EOF
        fi
    done
done
echo "=== PROBE done"
