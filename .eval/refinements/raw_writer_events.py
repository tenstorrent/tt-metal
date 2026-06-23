import os, re
import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

CSV = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/profile_log_device.csv")


def clear_csv():
    if os.path.exists(CSV):
        os.remove(CSV)


def wr_blocks(cx_t, cy_t):
    """Return per-block (WR-issue dur, WR-bar dur) ns for the given core, in order."""
    freq = 1000.0
    ev = []
    with open(CSV) as f:
        first = f.readline()
        m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", first)
        freq = float(m.group(1)) if m else 1000.0
        f.readline()
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 12:
                continue
            cx, cy, risc, cyc, zone, typ = p[1], p[2], p[3], int(p[5]), p[10], p[11]
            if cx == cx_t and cy == cy_t and risc == "BRISC" and zone in ("WR-issue", "WR-bar"):
                ev.append((cyc, zone, typ))
    ev.sort()
    # pair START/END per zone in order
    out = []
    stack = {}
    for cyc, zone, typ in ev:
        if typ == "ZONE_START":
            stack.setdefault(zone, []).append(cyc)
        else:
            s = stack[zone].pop(0)
            out.append((zone, (cyc - s) * 1000.0 / freq))
    return out, freq


def critical_core():
    best = {}
    with open(CSV) as f:
        f.readline()
        f.readline()
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 12 and p[10] == "WR-write" and p[11] == "ZONE_END":
                best[(p[1], p[2])] = max(best.get((p[1], p[2]), 0), int(p[5]))
    return max(best, key=best.get)


device = ttnn.open_device(device_id=0)
try:
    ttnn.synchronize_device(device)
    print("WRBLK_BEGIN")
    for shape in [(2048, 256), (2048, 256), (2048, 256), (8192, 256)]:
        x = torch.randn(*shape, dtype=torch.float32)
        ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        rms_norm(ti)
        ttnn.ReadDeviceProfiler(device)
        clear_csv()
        rms_norm(ti)
        ttnn.ReadDeviceProfiler(device)
        cx, cy = critical_core()
        blocks, _ = wr_blocks(cx, cy)
        issue = [d for z, d in blocks if z == "WR-issue"]
        bar = [d for z, d in blocks if z == "WR-bar"]
        print(f"\nshape {shape} core=({cx},{cy}) {len(issue)} write-blocks:")
        print(f"  WR-issue per block (ns): {[f'{d:.0f}' for d in issue]}")
        print(f"  WR-bar   per block (ns): {[f'{d:.0f}' for d in bar]}")
    print("\nWRBLK_END")
finally:
    ttnn.close_device(device)
