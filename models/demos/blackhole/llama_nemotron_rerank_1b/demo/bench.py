# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Traced latency of one forward (B sequences of S tokens) on one P150.

    python models/demos/blackhole/llama_nemotron_rerank_1b/demo/bench.py --shapes 1x128 1x512 8x512 32x512

Each shape: build inputs (B rows of S tokens), compile, capture a trace with the
attention mask applied, as for any batch whose rows hold packed or padded pairs
(``--no-mask``: the unmasked trace TracedReranker uses only when every row is one
pair of exactly S tokens), then replay it
``--iters`` times. Reported: the median replay latency over the second half (host-synchronized,
inputs already on device), sequences/s and tokens/s, and the trace output
checked against an untraced forward. ``--signpost`` wraps one replay in
tracy signposts for op-level profiling.
"""

import argparse
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model import NemotronReranker
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import MODEL_ID, config_from_overrides
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.weights import default_cache_dir, load_state_dict

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(*_a, **_k):
        pass


def bench_shape(model, device, B, S, iters, do_signpost, masked):
    g = torch.Generator().manual_seed(0)
    ids = [torch.randint(1000, 120000, (S,), generator=g).tolist() for _ in range(B)]
    host, lens = model.prepare_host_inputs(ids, seq_len=S)
    dev_in = model.to_device(host)
    model.rot_mats(S)
    ref = model.scores_from_output(model.forward(*dev_in, masked=masked), lens)  # compile + untraced result
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = model.forward(*dev_in, masked=masked)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    times = []
    for i in range(iters):
        t = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        times.append((time.perf_counter() - t) * 1e3)
    if do_signpost:
        signpost("start")
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        signpost("stop")
    got = model.scores_from_output(out, lens)
    maxdiff = max(abs(a - b) for a, b in zip(got, ref))
    ttnn.release_trace(device, tid)
    for t in dev_in:
        ttnn.deallocate(t)
    # Sustained: the board settles its clock under load, so the median skips the first half.
    med = statistics.median(times[len(times) // 2 :])
    print(
        f"B={B:<3} S={S:<5} {'masked  ' if masked else 'unmasked'} {med:8.2f} ms  "
        f"{B / med * 1e3:8.1f} seq/s  {B * S / med * 1e3:10.0f} tok/s  trace-vs-eager {maxdiff:.2e}"
    )
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["1x128", "1x512", "8x512", "32x512"])
    ap.add_argument("--iters", type=int, default=30, help="replays; the second half gives the sustained median")
    ap.add_argument("--signpost", action="store_true")
    ap.add_argument("--no-mask", action="store_true", help="run SDPA without the attention mask")
    ap.add_argument("--cfg", nargs="*", default=[], help="RerankConfig overrides, key=value")
    args = ap.parse_args()
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=400_000_000)
    try:
        model = NemotronReranker(
            dev, config_from_overrides(args.cfg), load_state_dict(MODEL_ID), default_cache_dir(MODEL_ID)
        )
        for s in args.shapes:
            B, S = map(int, s.split("x"))
            bench_shape(model, dev, B, S, args.iters, args.signpost, not args.no_mask)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
