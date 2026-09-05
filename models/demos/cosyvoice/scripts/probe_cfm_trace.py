# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Isolate which half of a traced CFM step is wrong: the graph, or the input refresh.

A traced solver that returns a bad answer has exactly two failure modes, and they
need different fixes:

1. **the captured graph is wrong** -- replay does not reproduce what an untraced
   call computes from the same inputs;
2. **the refresh is wrong** -- the graph is fine, but writing new values into the
   persistent input buffers between replays does not reach the ops that read them.

Guessing between them costs a device round trip per guess. This asks all three
questions in one run:

    A. replay == untraced, at the capture-time inputs?
    B. does rewriting `x_buf` change the replayed output?
    C. does rewriting `t_buf` change it?

Run on device:  python models/demos/cosyvoice/scripts/probe_cfm_trace.py
"""
from __future__ import annotations

import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from models.demos.cosyvoice.tests.pcc.test_cfm import FLOW_WEIGHTS  # noqa: E402
from models.demos.cosyvoice.tt.flow.estimator import TtConditionalDecoder  # noqa: E402
from models.demos.cosyvoice.tt.weights import WeightBag  # noqa: E402


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main() -> int:
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000)
    try:
        bag = WeightBag.load(FLOW_WEIGHTS)
        est = TtConditionalDecoder(device, bag.sub("decoder").sub("estimator"), dtype=ttnn.bfloat16)

        t_len, ch = 608, 80
        torch.manual_seed(0)
        x0 = torch.randn(2, t_len, ch)
        x1 = torch.randn(2, t_len, ch)
        mu = torch.randn(2, t_len, ch)
        spks = torch.randn(2, 1, ch)
        cond = torch.randn(2, t_len, ch)

        def dev(t, mc=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)

        mu_d, spks_d, cond_d = dev(mu), dev(spks), dev(cond)
        x_buf = dev(x0)
        t_buf = dev(torch.full((2, 1, 1), 0.0))

        def body():
            return est(x_buf, mu_d, t_buf, spks=spks_d, cond=cond_d, batch=2)

        # untraced references
        ref_x0_t0 = ttnn.to_torch(body())
        ttnn.copy(dev(x1), x_buf)
        ref_x1_t0 = ttnn.to_torch(body())
        ttnn.copy(dev(torch.full((2, 1, 1), 0.5)), t_buf)
        ref_x1_t5 = ttnn.to_torch(body())

        # back to the capture-time state, then capture
        ttnn.copy(dev(x0), x_buf)
        ttnn.copy(dev(torch.full((2, 1, 1), 0.0)), t_buf)
        ttnn.deallocate(body())
        ttnn.synchronize_device(device)

        tid = ttnn.begin_trace_capture(device, cq_id=0)
        try:
            out = body()
        finally:
            ttnn.end_trace_capture(device, tid, cq_id=0)

        def replay():
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            return ttnn.to_torch(out)

        got_x0_t0 = replay()
        ttnn.copy(dev(x1), x_buf)
        got_x1_t0 = replay()
        ttnn.copy(dev(torch.full((2, 1, 1), 0.5)), t_buf)
        got_x1_t5 = replay()

        print(f"A. graph      replay vs untraced @ (x0,t0) : PCC {pcc(got_x0_t0, ref_x0_t0):.8f}")
        print(f"B. x refresh  replay vs untraced @ (x1,t0) : PCC {pcc(got_x1_t0, ref_x1_t0):.8f}")
        print(f"   (x ignored? replay(x1) vs ref(x0)      : PCC {pcc(got_x1_t0, ref_x0_t0):.8f})")
        print(f"C. t refresh  replay vs untraced @ (x1,t5) : PCC {pcc(got_x1_t5, ref_x1_t5):.8f}")
        print(f"   (t ignored? replay(t5) vs ref(t0)      : PCC {pcc(got_x1_t5, ref_x1_t0):.8f})")

        # D. the refresh source cfm.py actually uses: a `[1,T,80]` solver tensor
        # doubled with ttnn.concat, rather than a plain from_torch upload.
        half = dev(x0[:1])
        pair = ttnn.concat([half, half], dim=0)
        ttnn.copy(pair, x_buf)
        ttnn.copy(dev(torch.full((2, 1, 1), 0.0)), t_buf)
        got_cat = replay()
        ttnn.copy(pair, x_buf)
        ref_cat = ttnn.to_torch(body())
        print(f"D. concat src replay vs untraced           : PCC {pcc(got_cat, ref_cat):.8f}")
        print(f"   (refresh ignored? vs replay(x1,t5)      : PCC {pcc(got_cat, got_x1_t5):.8f})")

        ttnn.release_trace(device, tid)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
