# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch, ttnn


def main():
    records = []

    def collect_records(batch):
        for r in batch.records:
            records.append(
                {
                    "runtime_id": r.runtime_id,
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
    h = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect_records)
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
        ttnn.close_mesh_device(dev)
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(h)

    snap = list(records)
    print(f"Total records from callback: {len(snap)}")
    short_list = []
    for i, r in enumerate(snap):
        if r["runtime_id"] == 0 or r["freq"] <= 0:
            continue
        dur_us = (r["end"] - r["start"]) / r["freq"] / 1000
        if dur_us < 10:
            short_list.append((i, r["runtime_id"], dur_us))
    total = len([r for r in snap if r["runtime_id"] != 0 and r["freq"] > 0])
    print(f"Total valid (runtime_id!=0): {total}, SHORT (<10us): {len(short_list)}")
    if short_list:
        print("SHORT records:")
        for idx, runtime_id, dur in short_list:
            r = snap[idx]
            print(
                f"  [{idx}] runtime_id={runtime_id} dur_us={dur:.3f} "
                f"start={r['start']} end={r['end']} delta={r['end']-r['start']}"
            )
            if idx > 0:
                prev = snap[idx - 1]
                prev_dur = (prev["end"] - prev["start"]) / prev["freq"] / 1000
                print(
                    f"    prev[{idx-1}] runtime_id={prev['runtime_id']} dur_us={prev_dur:.3f} "
                    f"start={prev['start']} end={prev['end']}"
                )


if __name__ == "__main__":
    main()
