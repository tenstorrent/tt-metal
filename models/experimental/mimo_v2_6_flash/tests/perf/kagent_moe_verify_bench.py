# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Time one MiMo MoE decode layer (router, top-8, experts, row all-reduce) on Galaxy row 0 at speculative-verify
shapes: 8 decode slots (MIMO_SPEC_SLOTS=8), BFP4 experts from the tensor cache, real activations of the golden
prompts (moe_decode_common.verify_steps):

- one_request: 4 consecutive tokens of one request (its last token and 3 drafts), 4 slots idle;
- two_requests: 4 consecutive tokens of each of two requests.

These are the MoE calls of an AgentX verify step with one or two requests per engine (adaptive draft groups, K=3).
Each shape's CALLS forward_decode calls are captured in one trace; each replay is timed between device syncs. A
sample is the mean of the two shapes' per-call times in one replay round; the metric is the median sample:

    MOE_VERIFY_SAMPLE us=<mean per call of the two shapes>
    MOE_VERIFY_METRIC us=<median>
"""

import argparse
import statistics
import time

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common

CALLS = 20
REPLAYS = 7
USERS = 8
SHAPES = ("one_request", "two_requests")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.parse_args()

    torch.manual_seed(0)
    cfg = common.config()
    inputs = common.golden_moe_inputs(cfg, common.checkpoint(cfg))
    steps = {name: idx for name, idx in common.verify_steps(cfg, 4)[4:8] if name in SHAPES}
    mesh, row = common.open_row(trace_region_size=64 * 1024 * 1024)
    try:
        moe = common.build_moe(row, cfg, "model_bfp4experts", ttnn.bfloat4_b, users=USERS)
        traces = {}
        for name in SHAPES:
            x, mask = common.step_inputs(row, inputs, steps[name])
            for _ in range(2):  # compile + warm the program cache before capture
                moe.forward_decode(x, mask).deallocate(True)
            ttnn.synchronize_device(row)
            trace_id = ttnn.begin_trace_capture(row, cq_id=0)
            outs = [moe.forward_decode(x, mask) for _ in range(CALLS)]
            ttnn.end_trace_capture(row, trace_id, cq_id=0)
            ttnn.synchronize_device(row)
            traces[name] = (trace_id, outs, x, mask)

        times = {name: [] for name in SHAPES}
        for _ in range(REPLAYS):
            for name in SHAPES:
                ttnn.synchronize_device(row)
                t0 = time.perf_counter()
                ttnn.execute_trace(row, traces[name][0], cq_id=0, blocking=False)
                ttnn.synchronize_device(row)
                times[name].append((time.perf_counter() - t0) / CALLS * 1e6)
        for trace_id, outs, x, mask in traces.values():
            ttnn.release_trace(row, trace_id)
            for t in outs + [x, mask]:
                t.deallocate(True)
        for name in SHAPES:
            print(f"MOE_VERIFY_SHAPE {name} us={statistics.median(times[name][1:]):.2f}", flush=True)
        samples = [statistics.mean(times[n][r] for n in SHAPES) for r in range(1, REPLAYS)]  # replay 0 warms
        for s in samples:
            print(f"MOE_VERIFY_SAMPLE us={s:.2f}", flush=True)
        print(f"MOE_VERIFY_METRIC us={statistics.median(samples):.2f}", flush=True)
    finally:
        common.close(mesh, row)


if __name__ == "__main__":
    main()
