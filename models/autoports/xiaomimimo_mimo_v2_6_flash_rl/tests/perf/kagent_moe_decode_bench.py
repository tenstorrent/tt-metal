# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Time one MiMo MoE decode layer (router, top-8, experts, row all-reduce) on Galaxy row 0.

TP=8 over the row, 4 users, BFP4 experts from the tensor cache, as in the full model's traced decode.
CALLS forward_decode calls are captured in one trace; each replay is timed between device syncs, so the
time is device time with no per-op host dispatch. Prints one sample per replay and the median last:

    MOE_DECODE_SAMPLE us=<per call>
    MOE_DECODE_METRIC us=<median per call>
"""

import argparse
import statistics
import time

import torch

import ttnn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.perf import moe_decode_common as common

CALLS = 20
REPLAYS = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.parse_args()

    torch.manual_seed(0)
    mesh, row = common.open_row(trace_region_size=64 * 1024 * 1024)
    try:
        cfg = common.config()
        moe = common.build_moe(row, cfg, "model_bfp4experts", ttnn.bfloat4_b)
        # Normalized random tokens: routing spreads them over ~32 distinct experts, like real decode.
        x_rows = (
            torch.nn.functional.normalize(torch.randn(common.USERS, cfg.hidden_size), dim=-1) * cfg.hidden_size**0.5
        )
        x, mask = common.device_inputs(row, x_rows)

        for _ in range(2):  # compile + warm the program cache before capture
            moe.forward_decode(x, mask).deallocate(True)
        ttnn.synchronize_device(row)

        trace_id = ttnn.begin_trace_capture(row, cq_id=0)
        outs = [moe.forward_decode(x, mask) for _ in range(CALLS)]
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
            print(f"MOE_DECODE_SAMPLE us={s:.2f}", flush=True)
        print(f"MOE_DECODE_METRIC us={statistics.median(samples[1:]):.2f}", flush=True)
    finally:
        common.close(mesh, row)


if __name__ == "__main__":
    main()
