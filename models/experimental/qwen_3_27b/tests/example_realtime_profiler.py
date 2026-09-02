# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Real-time profiler, end to end, on one TtQwen36GatedDeltaNet forward.

Two ways to use it: a raw callback when you want the records live, and the
repo helper when you just want a number. Not a pytest test -- run it directly:

    python models/experimental/qwen_3_27b/tests/example_realtime_profiler.py
"""

import threading
import time
from collections import defaultdict

import torch

import ttnn
from models.experimental.qwen_3_27b.tests.test_gated_deltanet import build_reference, to_device
from models.experimental.qwen_3_27b.tt.tt_gated_deltanet import D
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

SEQ_LEN = 128


# A record carries no op name, only kernel paths -- and one program's kernels span
# several op dirs (a tilize's reader lives under eltwise/unary, an
# untilize_with_unpadding borrows untilize's compute). So taking the first path
# mislabels 623 of 755 programs here. Order this most-specific-first instead, and
# validate the counts against Tracy before trusting them.
OP_PRIORITY = (
    "data_movement/tilize_with_val_padding",
    "data_movement/untilize_with_unpadding",
    "data_movement/untilize",
    "data_movement/tilize",
    "matmul",
    "normalization/layernorm",
    "eltwise/binary_ng",
    "data_movement/reshape_view",
    "data_movement/transpose",
    "data_movement/slice",
    "data_movement/concat",
    "copy/typecast",
    "eltwise/unary",
)


def op_name(kernel_sources):
    """Op name from kernel paths -- the only identity a record carries."""
    dirs = {s.split("/operations/")[-1].split("/device/")[0] for s in kernel_sources if "/operations/" in s}
    for op in OP_PRIORITY:
        if op in dirs:
            return op
    return next(iter(dirs), "unknown")


def duration_ns(record):
    """frequency is cycles per ns, so this converts raw cycles to real time."""
    return (int(record.end_timestamp) - int(record.start_timestamp)) / float(record.frequency)


def raw_callback(device, run_fn):
    """Full control: records arrive on a background thread while the work runs."""
    lock = threading.Lock()
    totals = defaultdict(lambda: [0, 0.0])  # op -> [count, ns]
    dropped = [0]
    first_arrival = [None]
    t0 = time.perf_counter()

    def on_batch(batch):
        # Runs on the RECEIVER thread, not main. Keep it cheap: appending is fine,
        # analysis is not -- slow callbacks hold the GIL and cause dropped records.
        with lock:
            if first_arrival[0] is None:
                first_arrival[0] = time.perf_counter() - t0
            dropped[0] += int(batch.dropped)
            for record in batch.records:
                entry = totals[op_name(record.kernel_sources)]
                entry[0] += 1
                entry[1] += duration_ns(record)

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(on_batch)
    try:
        run_fn()
        enqueue_returned = time.perf_counter() - t0
        ttnn.synchronize_device(device)  # REQUIRED, or trailing records are missed
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)

    with lock:
        assert dropped[0] == 0, f"{dropped[0]} records dropped -- callback too slow"
        print(f"\nfirst record at {first_arrival[0]*1000:.1f} ms, enqueue returned at {enqueue_returned*1000:.1f} ms")
        print(f"{'OP':<26}{'n':>7}{'us':>11}{'%':>8}")
        grand = sum(ns for _, ns in totals.values())
        for op, (count, ns) in sorted(totals.items(), key=lambda kv: -kv[1][1]):
            print(f"{op:<26}{count:>7}{ns/1e3:>11.1f}{100*ns/grand:>8.1f}")
        print(f"{'total':<26}{sum(c for c, _ in totals.values()):>7}{grand/1e3:>11.1f}")


def with_helper(device, run_fn):
    """When you only need the number: one call, records already merged per program."""
    _, records = profile_realtime_program_merged(device, run_fn)
    total_us = sum(r["duration_ns"] for r in records.values()) / 1e3
    print(f"\nhelper: {len(records)} programs, {total_us:.1f} us")


def main():
    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0)
    try:
        # Gate on this, and FAIL rather than skip -- an unmeasured perf run
        # reporting green is worse than no run. Needs the device already open.
        if not ttnn.device.IsProgramRealtimeProfilerActive():
            raise SystemExit("real-time profiler inactive on this setup")

        model = to_device(device, build_reference())
        x = ttnn.from_torch(
            torch.randn(1, SEQ_LEN, D, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        def run_fn():
            ttnn.deallocate(model(x))
            model.reset_state()

        run_fn()  # warmup: compiles kernels, would otherwise pollute the first measurement

        raw_callback(device, run_fn)
        with_helper(device, run_fn)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
