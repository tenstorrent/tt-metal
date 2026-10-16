# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Time the MiMo attention decode block (TtMiMoDecoderLayer.attention_decode) on Galaxy row 0.

TP=8 over the row, 4 users. Layer 1 (SWA) and layer 5 (global) are each captured as CALLS calls in one trace
and replayed; each replay is timed between device syncs. One sample is the per-layer time weighted by the
model's mix of layers (39 SWA : 9 global):

    ATTN_DECODE_SAMPLE us=<weighted per-layer time of one replay pair>
    ATTN_DECODE_METRIC us=<median>
"""

import argparse
import statistics
import time

import ttnn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.perf import attn_decode_common as attn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.perf import moe_decode_common as common

CALLS = 20
REPLAYS = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.parse_args()

    mesh, row = common.open_row(trace_region_size=64 * 1024 * 1024)
    try:
        cfg = common.config()
        share = attn.layer_weights(cfg)
        per_layer = {}
        for idx in attn.LAYERS:
            state = attn.build(row, cfg, idx)
            attn.prefill(state, cfg, seed=idx)
            x, rot, cur_pos, _ = attn.decode_inputs(state, cfg, step=0, seed=100 + idx)
            for _ in range(2):  # compile + warm the program cache before capture
                attn.decode(state, x, rot, cur_pos).deallocate(True)
            ttnn.synchronize_device(row)
            trace_id = ttnn.begin_trace_capture(row, cq_id=0)
            outs = [attn.decode(state, x, rot, cur_pos) for _ in range(CALLS)]
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
            per_layer[idx] = samples[1:]  # the first replay warms the trace
            print(f"layer {idx}: median {statistics.median(per_layer[idx]):.2f} us per call", flush=True)
        weighted = [sum(share[i] * per_layer[i][r] for i in attn.LAYERS) for r in range(REPLAYS - 1)]
        for s in weighted:
            print(f"ATTN_DECODE_SAMPLE us={s:.2f}", flush=True)
        print(f"ATTN_DECODE_METRIC us={statistics.median(weighted):.2f}", flush=True)
    finally:
        common.close(mesh, row)


if __name__ == "__main__":
    main()
