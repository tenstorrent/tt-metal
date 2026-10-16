# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Time the MiMo decode projections (qkv + o + router of one layer, tt/proj_stream.py) on Galaxy row 0.

CALLS iterations of the three projections are captured in one trace; each replay is timed between device syncs:

    PROJ_DECODE_SAMPLE us=<per iteration (qkv + o + router)>
    PROJ_DECODE_METRIC us=<median>
"""

import argparse
import statistics
import time

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import proj_decode_common as proj_common

CALLS = 30
REPLAYS = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.parse_args()
    mesh, row = proj_common.open_row(trace_region_size=32 * 1024 * 1024)
    try:
        plan, proj, weights = proj_common.build(row)
        x_h, x_o = proj_common.inputs(row, 0)
        for _ in range(2):
            for t in proj_common.run(proj, weights, x_h, x_o):
                t.deallocate(True)
        ttnn.synchronize_device(row)
        trace_id = ttnn.begin_trace_capture(row, cq_id=0)
        outs = [proj_common.run(proj, weights, x_h, x_o) for _ in range(CALLS)]
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
            for t in o:
                t.deallocate(True)
        for s in samples[1:]:
            print(f"PROJ_DECODE_SAMPLE us={s:.2f}", flush=True)
        print(f"PROJ_DECODE_METRIC us={statistics.median(samples[1:]):.2f}", flush=True)
    finally:
        proj_common.close(mesh, row)


if __name__ == "__main__":
    main()
