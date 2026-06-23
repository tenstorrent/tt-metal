import os, re
import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

CSV = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/profile_log_device.csv")


def clear_csv():
    if os.path.exists(CSV):
        os.remove(CSV)


device = ttnn.open_device(device_id=0)
try:
    ttnn.synchronize_device(device)
    shape = (2048, 256)
    x = torch.randn(*shape, dtype=torch.float32)
    ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rms_norm(ti)
    ttnn.ReadDeviceProfiler(device)
    clear_csv()
    rms_norm(ti)
    ttnn.ReadDeviceProfiler(device)

    freq = 1000.0
    raw = []  # (cx,cy,risc,cyc,zone,typ)
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
            if zone.startswith(("CMP-", "RDR-", "WR-")):
                raw.append((cx, cy, risc, cyc, zone, typ))
    # pick the core whose WR-write ends latest
    cores = {}
    for cx, cy, risc, cyc, zone, typ in raw:
        if zone == "WR-write" and typ == "ZONE_END":
            k = (cx, cy)
            cores[k] = max(cores.get(k, 0), cyc)
    cx, cy = max(cores, key=cores.get)
    ev = [(cyc, risc, zone, typ) for (a, b, risc, cyc, zone, typ) in raw if a == cx and b == cy]
    ev.sort()
    t0 = ev[0][0]
    print(f"RAW_BEGIN core=({cx},{cy}) freq={freq:.0f}")
    print(f"{'t_ns':>8s} {'risc':8s} {'zone':16s} {'type':10s}")
    for cyc, risc, zone, typ in ev:
        # focus on BRISC (writer) + TRISC_1/2 (compute math/pack) + NCRISC reader
        print(f"{(cyc-t0)*1000.0/freq:8.0f} {risc:8s} {zone:16s} {typ}")
    print("RAW_END")
finally:
    ttnn.close_device(device)
