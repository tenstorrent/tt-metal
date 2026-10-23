# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Time the qkv prefetch section (BankStreamMatmul.section in the MoE reduce launch, qkv_prefetch_common.py) on
Galaxy row 0: CALLS launches (section + stub) captured in one trace; each replay is timed between device syncs.

One launch takes at least X_DELAY_US (the stub's wait) plus the x copy; the section's weight stream starts at the
launch start and its output follows x. Lower is better:

    QKV_PREFETCH_SAMPLE us=<per launch>
    QKV_PREFETCH_METRIC us=<median>
"""

import argparse
import statistics
import time

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import qkv_prefetch_common as qp

CALLS = 20
REPLAYS = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.parse_args()
    mesh, row = qp.open_row(trace_region_size=32 * 1024 * 1024)
    try:
        plan, stream, w, w_stream = qp.build(row)
        x = qp.x_tensor(row, qp.x_host(0))
        x_src = qp.x_tensor(row, qp.x_host(0))
        for _ in range(2):  # compile, warm the program cache
            qp.launch(row, stream, w_stream, x, x_src).deallocate(True)
        ttnn.synchronize_device(row)
        trace_id = ttnn.begin_trace_capture(row, cq_id=0)
        outs = [qp.launch(row, stream, w_stream, x, x_src) for _ in range(CALLS)]
        ttnn.end_trace_capture(row, trace_id, cq_id=0)
        ttnn.synchronize_device(row)
        samples = []
        for _ in range(REPLAYS):
            ttnn.synchronize_device(row)
            t0 = time.perf_counter()
            ttnn.execute_trace(row, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(row)
            samples.append((time.perf_counter() - t0) / CALLS * 1e6)
        ttnn.release_trace(row, trace_id)
        for o in outs:
            o.deallocate(True)
        for s in samples[1:]:  # the first replay warms the trace
            print(f"QKV_PREFETCH_SAMPLE us={s:.2f}", flush=True)
        print(f"QKV_PREFETCH_METRIC us={statistics.median(samples[1:]):.2f}", flush=True)
    finally:
        qp.close(mesh, row)


if __name__ == "__main__":
    main()
