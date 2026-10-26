# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Time the KV pair split's decode merge region (PairSplit.merge_heads, pair_merge_common.py) on Galaxy row 0, as a
decode step at c=16 runs it: 4 decode users, one call per global layer. CALLS calls captured in one trace; each
replay is timed between device syncs. Lower is better:

    PAIR_MERGE_SAMPLE us=<per call>
    PAIR_MERGE_METRIC us=<median>
"""

import argparse
import statistics
import time

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import pair_merge_common as pm

USERS = 4
CALLS = 4 * pm.LAYERS
REPLAYS = 9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.parse_args()
    mesh, row = pm.open_row(trace_region_size=32 * 1024 * 1024)
    try:
        plan, pair, concat = pm.build(row)
        o_tt, lse_tt = pm.device_inputs(plan, *pm.host_inputs(USERS, seed=1))
        for _ in range(2):  # compile, warm the program cache
            pm.run(pair, concat, o_tt, lse_tt).deallocate(True)
        ttnn.synchronize_device(row)
        trace_id = ttnn.begin_trace_capture(row, cq_id=0)
        outs = [pm.run(pair, concat, o_tt, lse_tt) for _ in range(CALLS)]
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
            print(f"PAIR_MERGE_SAMPLE us={s:.2f}", flush=True)
        print(f"PAIR_MERGE_METRIC us={statistics.median(samples[1:]):.2f}", flush=True)
    finally:
        pm.close(mesh, row)


if __name__ == "__main__":
    main()
