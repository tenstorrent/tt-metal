# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import csv, os, threading, time, torch, ttnn


def main():
    records, lock = [], threading.Lock()

    def collect(r):
        with lock:
            records.append(
                {
                    "pid": r.program_id,
                    "start": r.start_timestamp,
                    "end": r.end_timestamp,
                    "freq": r.frequency,
                    "chip": r.chip_id,
                }
            )

    dev = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        l1_small_size=24576,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    h = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect)
    try:
        torch.manual_seed(0)
        a = ttnn.to_layout(
            ttnn.to_device(
                ttnn.from_torch(torch.randn(1024, 1024, dtype=torch.bfloat16)), dev, memory_config=ttnn.L1_MEMORY_CONFIG
            ),
            ttnn.TILE_LAYOUT,
        )
        b = ttnn.to_layout(
            ttnn.to_device(
                ttnn.from_torch(torch.randn(1024, 1024, dtype=torch.bfloat16)), dev, memory_config=ttnn.L1_MEMORY_CONFIG
            ),
            ttnn.TILE_LAYOUT,
        )
        ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=1, x=1))
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(50):
            ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=1, x=1))
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        for _ in range(10):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        ttnn.release_trace(dev, tid)
        ttnn.synchronize_device(dev)
    finally:
        time.sleep(2.0)
        ttnn.close_mesh_device(dev)
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(h)

    with lock:
        snap = list(records)

    csv_path = os.path.join("generated", "profiler", ".logs", "profile_log_device.csv")
    metal_home = os.environ.get("TT_METAL_HOME", "")
    if metal_home:
        csv_path = os.path.join(metal_home, csv_path)

    # Parse dispatch_s ZONE_START events from device profiler
    dp_zones = []  # (start_ts, cmd_type)
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            lines = f.readlines()
        for line in lines[2:]:
            parts = line.strip().split(",")
            if len(parts) < 12:
                continue
            risc = parts[3]
            if risc != "NCRISC":
                continue
            zone_name = parts[10]
            zone_type = parts[11]
            if "CQ-DISPATCH-SUBORDINATE:" not in zone_name:
                continue
            if zone_type != "ZONE_START":
                continue
            cmd_type = zone_name.split("CQ-DISPATCH-SUBORDINATE:")[-1]
            ts = int(parts[5])
            dp_zones.append((ts, cmd_type))
    else:
        print(f"CSV not found: {csv_path}")

    # Sort device profiler zones by start timestamp
    dp_zones.sort(key=lambda x: x[0])

    # Count command types
    cmd_counts = {}
    for _, ct in dp_zones:
        cmd_counts[ct] = cmd_counts.get(ct, 0) + 1
    print(f"=== Device Profiler: dispatch_s zones ===")
    print(f"Total zones: {len(dp_zones)}")
    for ct, count in sorted(cmd_counts.items(), key=lambda x: -x[1]):
        print(f"  {ct}: {count}")

    # RT profiler summary - show ALL records including pid=0
    print(f"\n=== RT Profiler: ALL {len(snap)} records ===")
    for i, r in enumerate(snap):
        dur_us = (r["end"] - r["start"]) / r["freq"] / 1000 if r["freq"] > 0 else 0
        short_tag = " *** SHORT ***" if (r["pid"] != 0 and r["freq"] > 0 and dur_us < 10) else ""
        print(f"  RT[{i:3d}] pid={r['pid']:5d} dur_us={dur_us:12.3f} start={r['start']} end={r['end']}{short_tag}")

    valid = [r for r in snap if r["pid"] != 0 and r["freq"] > 0]
    short_count = sum(
        1 for r in snap if r["pid"] != 0 and r["freq"] > 0 and (r["end"] - r["start"]) / r["freq"] / 1000 < 10
    )
    print(f"\nTotal: {len(snap)}, valid (pid!=0): {len(valid)}, SHORT (<10us): {short_count}")

    # Sequential cross-reference: match RT records to GO_SIGNAL zones only
    if dp_zones:
        go_zones = [(i, ts, ct) for i, (ts, ct) in enumerate(dp_zones) if ct == "CQ_DISPATCH_CMD_SEND_GO_SIGNAL"]
        non_go_zones = [(i, ts, ct) for i, (ts, ct) in enumerate(dp_zones) if ct != "CQ_DISPATCH_CMD_SEND_GO_SIGNAL"]

        print(f"\n=== Non-GO zones (should be filtered from RT, pid=0) ===")
        for dp_idx, ts, ct in non_go_zones:
            print(f"  DP[{dp_idx:3d}] {ct}")

        print(f"\n=== GO_SIGNAL zones vs RT records (1:1 match) ===")
        max_n = max(len(go_zones), len(snap))
        for i in range(max_n):
            if i < len(go_zones):
                dp_idx, _, _ = go_zones[i]
                dp_str = f"DP[{dp_idx:3d}] GO_SIGNAL"
            else:
                dp_str = "---"
            if i < len(snap):
                r = snap[i]
                dur_us = (r["end"] - r["start"]) / r["freq"] / 1000 if r["freq"] > 0 else 0
                rt_str = f"pid={r['pid']:5d} dur_us={dur_us:.3f}"
                short_tag = " *** SHORT ***" if (r["pid"] != 0 and r["freq"] > 0 and dur_us < 10) else ""
            else:
                rt_str = "---"
                short_tag = ""
            print(f"  GO[{i:3d}] {dp_str:30s} RT[{i:3d}]: {rt_str}{short_tag}")


if __name__ == "__main__":
    main()
