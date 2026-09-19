# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Where a generated token's time goes, outside the traced decode step.

The tail is the output head, the device-to-host transfer of the distribution, RAS
sampling on the host, and the embedding lookup back onto the device. This script
times those four directly, which is the point of it: subtracting the traced-step
microbenchmark from the `generate()` per-token figure suggested ~2.7 ms of tail,
and the measurement came back at **0.352 ms**. The two benchmarks differed in
cache warmth, not only in scope. Subtracting two benchmarks is not a profile.

The question it answers is whether moving RAS onto `ttnn.sampling` is worth
building. That trade only pays if the transfer and the host sampling are a real
share of the tail; if the output head dominates, on-device sampling is
rearranging a rounding error. Measured, it is the latter: `ttnn.sampling` could
remove at most 0.217 ms, 1.7 % of a token, so RAS stays on the host.

    python models/demos/cosyvoice/scripts/profile_token_tail.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from models.demos.cosyvoice.tests.perf.test_llm_perf import LLM_WEIGHTS  # noqa: E402
from models.demos.cosyvoice.tt.llm.sampling import ras_sampling  # noqa: E402
from models.demos.cosyvoice.tt.weights import WeightBag  # noqa: E402


def main() -> int:
    import argparse

    from models.demos.cosyvoice.tt.llm.model import TtTransformerLM

    ap = argparse.ArgumentParser()
    # The traced decode step this tail sits alongside. PERF.md is where the current
    # figures live; pass the one for the part being measured.
    ap.add_argument("--step-ms", type=float, default=4.99, help="traced decode step, ms")
    args = ap.parse_args()

    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=67108864)
    try:
        bag = WeightBag.load(LLM_WEIGHTS)
        model = TtTransformerLM(device, bag, bag.meta)
        vocab = model.speech_token_size + 1
        n = 64

        # A plausible hidden state to drive the head with; the tail's cost does not
        # depend on the values, only on the shapes and the transfers.
        torch.manual_seed(0)
        ys = ttnn.from_torch(
            torch.randn(1, 1, bag.meta["ar_decoder"]["d_model"]) * 0.1,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        model.logits_for_last(ys)  # warm
        ttnn.synchronize_device(device)

        # 1. head matmul only, no transfer
        t0 = time.perf_counter()
        for _ in range(n):
            logits = ttnn.linear(ys, model.head_w, bias=model.head_b, compute_kernel_config=model.cc)
            ttnn.deallocate(logits)
        ttnn.synchronize_device(device)
        head_ms = (time.perf_counter() - t0) * 1e3 / n

        # 2. head matmul plus the device-to-host read
        t0 = time.perf_counter()
        for _ in range(n):
            model.logits_for_last(ys)
        full_ms = (time.perf_counter() - t0) * 1e3 / n
        d2h_ms = full_ms - head_ms

        # 3. RAS on the host, on a real-sized distribution
        logp = model.logits_for_last(ys)
        decoded = list(range(20))
        t0 = time.perf_counter()
        for _ in range(n):
            ras_sampling(logp.clone(), decoded)
        ras_ms = (time.perf_counter() - t0) * 1e3 / n

        # 4. token id -> embedding row -> device
        tok = 42
        t0 = time.perf_counter()
        for _ in range(n):
            row = model.speech_embedding_host[tok].reshape(1, 1, -1)
            e = ttnn.from_torch(row, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            ttnn.deallocate(e)
        ttnn.synchronize_device(device)
        embed_ms = (time.perf_counter() - t0) * 1e3 / n

        tail = head_ms + d2h_ms + ras_ms + embed_ms
        print(f"\n  per-token tail, vocab={vocab}, mean of {n}")
        print(f"    output head matmul        {head_ms:7.3f} ms  {100 * head_ms / tail:5.1f}%")
        print(f"    logits device -> host     {d2h_ms:7.3f} ms  {100 * d2h_ms / tail:5.1f}%")
        print(f"    RAS sampling on host      {ras_ms:7.3f} ms  {100 * ras_ms / tail:5.1f}%")
        print(f"    embedding row -> device   {embed_ms:7.3f} ms  {100 * embed_ms / tail:5.1f}%")
        print(f"    tail total                {tail:7.3f} ms")
        # The step this is a fraction *of* differs by architecture and by cache width, so
        # it is an argument rather than a constant. It was hardcoded to 12.52 ms, which
        # stopped being true the moment tracing and the in-place KV cache landed --
        # PERF.md now measures 4.99 ms on Blackhole and 8.20 on Wormhole, and the same
        # tail is twice the share of a Blackhole token that it is of a Wormhole one.
        print(
            f"    against a {args.step_ms:.2f} ms decode step, the tail is "
            f"{100 * tail / (args.step_ms + tail):4.1f}% of a token"
        )
        print(
            f"    on-device sampling could remove at most {d2h_ms + ras_ms:.3f} ms, i.e. "
            f"{100 * (d2h_ms + ras_ms) / (args.step_ms + tail):4.1f}%"
        )
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
