import os, re
import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc

CSV = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/profile_log_device.csv")
PREFIXES = ("CMP-", "RDR-", "WR-")


# which RISC each zone family runs on (for the timeline lane label)
def clear_csv():
    if os.path.exists(CSV):
        os.remove(CSV)


def events_for_core(cx_t, cy_t):
    """All (risc, zone, start, end) for one core, abs cycles. Compute zones: MATH thread (TRISC_1)."""
    freq = 1000.0
    rows = {}
    with open(CSV) as f:
        first = f.readline()
        m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", first)
        freq = float(m.group(1)) if m else 1000.0
        f.readline()
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 12:
                continue
            cx, cy, risc, cyc, zone, typ = p[1], p[2], p[3], p[5], p[10], p[11]
            if cx != cx_t or cy != cy_t or not zone.startswith(PREFIXES):
                continue
            if zone.startswith("CMP-") and risc != "TRISC_1":
                continue
            d = rows.setdefault((risc, zone), {"s": [], "e": []})
            (d["s"] if typ == "ZONE_START" else d["e"]).append(int(cyc))
    out = []
    for (risc, zone), d in rows.items():
        if d["s"] and d["e"]:
            out.append((risc, zone, min(d["s"]), max(d["e"])))  # span (covers all fires for this core)
    return out, freq


def total_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per = ttnn.get_latest_programs_perf_data()
    t = 0.0
    for progs in (per or {}).values():
        for pr in progs:
            r = getattr(pr, "program_analyses_results", None) or {}
            e = r.get("DEVICE KERNEL DURATION [ns]")
            if e is not None:
                t += float(e.duration)
    return t


def timeline(name, shape, transport=None):
    x = torch.randn(*shape, dtype=torch.float32)
    ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    desc._FORCE_TRANSPORT = transport
    rms_norm(ti)
    ttnn.ReadDeviceProfiler(device)
    clear_csv()
    rms_norm(ti)
    tot = total_ns(device)
    desc._FORCE_TRANSPORT = None
    # pick the core with the LATEST end (the critical core)
    # scan CSV for all cores first
    cores = set()
    with open(CSV) as f:
        f.readline()
        f.readline()
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 12 and p[10].startswith(PREFIXES):
                cores.add((p[1], p[2]))
    best = None
    for cx, cy in cores:
        evs, freq = events_for_core(cx, cy)
        if not evs:
            continue
        end = max(e for _, _, _, e in evs)
        if best is None or end > best[0]:
            best = (end, cx, cy, evs, freq)
    _, cx, cy, evs, freq = best
    t0 = min(s for _, _, s, _ in evs)
    evs.sort(key=lambda r: r[2])
    print(f"\n========== {name}  shape={shape}  transport={transport}  total={tot:.0f}ns  core=({cx},{cy}) ==========")
    print(f"{'zone':16s} {'RISC':8s} {'start':>8s} {'end':>8s} {'dur':>7s}   timeline (each char ~ total/80 ns)")
    span = best[0] - t0
    scale = span / 80.0 if span else 1.0
    for risc, zone, s, e in evs:
        s_ns = (s - t0) * 1000.0 / freq
        e_ns = (e - t0) * 1000.0 / freq
        d_ns = e_ns - s_ns
        lpad = int(s_ns / (scale * 1000.0 / freq)) if scale else 0
        blen = max(1, int(d_ns / (scale * 1000.0 / freq)))
        bar = " " * lpad + "#" * blen
        print(f"{zone:16s} {risc:8s} {s_ns:8.0f} {e_ns:8.0f} {d_ns:7.0f}   {bar}")
    # per-RISC busy vs idle over the span
    print(f"  span={span*1000.0/freq:.0f}ns; per-RISC busy:")
    by_risc = {}
    for risc, zone, s, e in evs:
        by_risc.setdefault(risc, []).append((s, e))
    for risc, iv in sorted(by_risc.items()):
        iv.sort()
        # merge overlaps to get true busy
        busy = 0
        cur_s, cur_e = iv[0]
        for s, e in iv[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                busy += cur_e - cur_s
                cur_s, cur_e = s, e
        busy += cur_e - cur_s
        busy_ns = busy * 1000.0 / freq
        rspan = (max(e for _, e in iv) - min(s for s, _ in iv)) * 1000.0 / freq
        print(f"    {risc:8s} busy={busy_ns:8.0f}ns  idle_within_its_span={rspan-busy_ns:8.0f}ns")


device = ttnn.open_device(device_id=0)
try:
    ttnn.synchronize_device(device)
    print("TIMELINE_BEGIN")
    timeline("Regime A 1-row/core", (2048, 256))
    timeline("Regime A 4-rows/core", (8192, 256))
    timeline("Regime B wide-W (mode2)", (1, 1, 32, 8192), transport=2)
    print("\nTIMELINE_END")
finally:
    ttnn.close_device(device)
