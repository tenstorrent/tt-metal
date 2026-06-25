#!/bin/bash
# Helper functions for extracting timing from TT-Metal device profiler CSVs.

extract_profiler_compute_time() {
    local profiler_csv="$1"
    local work_dir="${2:-$WORK_DIR}"

    if [[ ! -f "$profiler_csv" ]]; then
        echo "0"
        return 1
    fi

    python3 - "$profiler_csv" "$work_dir" <<'PY' 2>/dev/null
import csv
import sys

profiler_csv, work_dir = sys.argv[1:3]

try:
    with open(profiler_csv) as f:
        first = f.readline()
        chip_freq = None
        for part in first.split(","):
            if "CHIP_FREQ[MHz]" in part:
                chip_freq = int(part.split(":")[1].strip())
                break
        if not chip_freq:
            raise ValueError(first)

        rows = csv.DictReader(f)
        starts = {}
        durations = []
        for row in rows:
            row = {k.strip(): v.strip() for k, v in row.items()}
            if row.get("RISC processor type") != "TRISC_1":
                continue
            if "KERNEL" not in row.get("zone name", ""):
                continue
            key = (row.get("core_x"), row.get("core_y"), row.get("zone name"))
            cyc = int(row["time[cycles since reset]"])
            if row.get("type") == "ZONE_START":
                starts[key] = cyc
            elif row.get("type") == "ZONE_END" and key in starts:
                durations.append((cyc - starts[key]) / chip_freq)
        print(f"{(sum(durations) / len(durations)):.2f}" if durations else "0")
except Exception:
    print("0")
    sys.exit(1)
PY
}

extract_batch_profiler_times() {
    local profiler_csv="$1"
    local num_shapes="$2"
    local work_dir="${3:-$WORK_DIR}"

    if [[ ! -f "$profiler_csv" ]]; then
        python3 -c "print(','.join(['0']*int('$num_shapes')))"
        return 1
    fi

    python3 - "$profiler_csv" "$num_shapes" "$work_dir" <<'PY' 2>/dev/null
import csv
import sys

profiler_csv, num_shapes, work_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]

try:
    with open(profiler_csv) as f:
        first = f.readline()
        chip_freq = None
        for part in first.split(","):
            if "CHIP_FREQ[MHz]" in part:
                chip_freq = int(part.split(":")[1].strip())
                break
        if not chip_freq:
            raise ValueError(first)

        events = []
        for row in csv.DictReader(f):
            row = {k.strip(): v.strip() for k, v in row.items()}
            if row.get("RISC processor type") != "TRISC_1":
                continue
            if "KERNEL" not in row.get("zone name", ""):
                continue
            if row.get("type") not in ("ZONE_START", "ZONE_END"):
                continue
            events.append((int(row["time[cycles since reset]"]), row))

    if not events:
        print(",".join(["0"] * num_shapes))
        sys.exit(0)

    events.sort(key=lambda item: item[0])
    if num_shapes <= 1:
        clusters = [events]
    else:
        gaps = [(events[i + 1][0] - events[i][0], i) for i in range(len(events) - 1)]
        cut_idxs = sorted(i for _, i in sorted(gaps, reverse=True)[: max(0, num_shapes - 1)])
        clusters = []
        start = 0
        for idx in cut_idxs:
            clusters.append(events[start : idx + 1])
            start = idx + 1
        clusters.append(events[start:])

    times = []
    for cluster in clusters[:num_shapes]:
        starts = {}
        durations = []
        for cyc, row in cluster:
            key = (row.get("core_x"), row.get("core_y"), row.get("zone name"))
            if row.get("type") == "ZONE_START":
                starts[key] = cyc
            elif row.get("type") == "ZONE_END" and key in starts:
                durations.append((cyc - starts[key]) / chip_freq)
        times.append(sum(durations) / len(durations) if durations else 0.0)
    times += [0.0] * (num_shapes - len(times))
    print(",".join(f"{t:.2f}" for t in times[:num_shapes]))
except Exception:
    print(",".join(["0"] * num_shapes))
    sys.exit(1)
PY
}
