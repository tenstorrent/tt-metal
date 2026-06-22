import os, re, csv
import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

CSV = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/profile_log_device.csv")
PREFIXES = ("CMP-", "RDR-", "WR-")


def clear_csv():
    if os.path.exists(CSV):
        os.remove(CSV)


def parse_zones():
    """Return {zone_name: [per-core wall-duration ns]}. Per core/zone wall = max(END)-min(START)
    across all RISCs that ran that zone (compute spans TRISC_0/1/2)."""
    if not os.path.exists(CSV):
        return {}, 1000.0
    with open(CSV) as f:
        first = f.readline()
        m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", first)
        freq = float(m.group(1)) if m else 1000.0
        f.readline()  # column header
        events = {}
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 11:
                continue
            cx, cy, risc, timer_id, cycles = parts[1], parts[2], parts[3], parts[4], parts[5]
            zone, typ = parts[9], parts[10]
            if not zone.startswith(PREFIXES):
                continue
            d = events.setdefault((cx, cy, zone), {"s": [], "e": []})
            if typ == "ZONE_START":
                d["s"].append(int(cycles))
            elif typ == "ZONE_END":
                d["e"].append(int(cycles))
    out = {}
    for (cx, cy, zone), d in events.items():
        if d["s"] and d["e"]:
            out.setdefault(zone, []).append((max(d["e"]) - min(d["s"])) * 1000.0 / freq)
    return out, freq


def total_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    tot = 0.0
    for programs in (per_chip or {}).values():
        for p in programs:
            r = getattr(p, "program_analyses_results", None) or {}
            e = r.get("DEVICE KERNEL DURATION [ns]")
            if e is not None:
                tot += float(e.duration)
    return tot


# (b, c, h, w, regime_label)
SHAPES = [
    (1, 1, 2048, 256, "A"),
    (1, 1, 2048, 512, "A"),
    (1, 1, 32, 4096, "B"),
    (1, 1, 32, 8192, "B"),
    (1, 1, 32, 16384, "B"),
]

device = ttnn.open_device(device_id=0)
try:
    ttnn.synchronize_device(device)
    print("ZONE_PROFILE_BEGIN")
    for b, c, h, w, lbl in SHAPES:
        shape = (b, c, h, w) if b > 1 or c > 1 else (h, w)
        x = torch.randn(*shape, dtype=torch.float32)
        ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        rms_norm(ti)  # warmup / JIT
        ttnn.ReadDeviceProfiler(device)  # flush warmup out of device buffer + CSV
        clear_csv()

        rms_norm(ti)  # measured
        tot = total_ns(device)  # ReadDeviceProfiler -> dumps measured to CSV
        zones, freq = parse_zones()

        print(f"\n=== shape {shape}  expect Regime {lbl}  total_device_kernel_ns={tot:.0f}  freq={freq:.0f}MHz ===")
        order = [
            "RDR-input",
            "CMP-p1-square",
            "CMP-p1-reduce",
            "RDR-ar-wait",
            "RDR-ar-xport",
            "CMP-combine",
            "CMP-finalize",
            "CMP-pass2",
            "WR-write",
        ]
        seen = set()
        for z in order + sorted(zones):
            if z in seen or z not in zones:
                continue
            seen.add(z)
            v = zones[z]
            v.sort()
            mean = sum(v) / len(v)
            print(f"  {z:16s} ncores={len(v):3d}  min={min(v):8.0f}  mean={mean:8.0f}  max={max(v):8.0f} ns")
    print("\nZONE_PROFILE_END")
finally:
    ttnn.close_device(device)
