# Ternary decode linears fed by the weight stream (tt/stream) vs the same linears reading DRAM: a chain of GDN-layer
# shapes, outputs compared, traced time per chain.
import time

import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt import core_ranges
from models.demos.blackhole.bonsai2.tt.stream.stream import Plan, Stream, drain
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import TernaryDecodeLinear3, pack_records_inline

SHAPES = [(5120, 16384), (6144, 5120), (5120, 34816), (17408, 5120)]


@pytest.mark.parametrize("device_params", [{"trace_region_size": 50_000_000}], indirect=True)
@pytest.mark.parametrize("layers", [1, 4])
def test_stream_linear(device, layers):
    print("DBG start", flush=True)
    st = Stream(device)
    print("DBG stream up", flush=True)
    core_ranges.set_layout(st.rx_order, st.defines())
    try:
        rng = np.random.default_rng(0)
        lins, acts, nbytes = [], [], 0
        for K, N in SHAPES:
            t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
            s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
            rec, ppr = pack_records_inline(t, s)
            lin = TernaryDecodeLinear3.from_packed(device, rec, ppr, K, N, cores=len(st.rx_order))
            lin.spread = True
            lins.append(lin)
            nbytes += rec.nbytes
            acts.append(
                ttnn.from_torch(
                    torch.randn(1, K).bfloat16().repeat(8, 1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                )
            )
        print("DBG weights up", flush=True)
        chain = list(zip(lins, acts)) * layers
        plan = Plan(st)
        takes = [plan.add(lin.stream_pages(device)) for lin, _ in chain]
        plan.build()
        print("PLAN slots", plan.slots, sum(plan.slots) / (len(plan.slots) * max(plan.slots)), flush=True)
        print("DBG plan built", flush=True)

        import os

        dbg = os.environ.get("STREAM_DBG") == "1"

        def run(stream):
            if stream:
                st.launch(plan)
            out = []
            for i, (lin, a) in enumerate(chain):
                out.append(lin(a, stream=st if stream else None))
                if dbg:
                    ttnn.synchronize_device(device)
                    print(f"DBG stream={stream} op {i} done", flush=True)
            return out

        ref = [ttnn.to_torch(y).float() for y in run(False)]
        got = [ttnn.to_torch(y).float() for y in run(True)]
        ttnn.synchronize_device(device)
        for i, (a, b) in enumerate(zip(ref, got)):
            assert torch.equal(a, b), (i, (a - b).abs().max().item())
        times = {}

        import os

        one = os.environ.get("DRAIN_ONE") == "1"
        tot = {}
        for t in takes:
            for k, v in t.items():
                tot[k] = tot.get(k, 0) + v

        def drains():
            st.launch(plan)
            if one:
                drain(st, tot)
                return
            for t in takes:
                drain(st, t)

        for mode in (False, True, "drain"):
            fn = drains if mode == "drain" else (lambda: run(mode))
            fn()
            ttnn.synchronize_device(device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            fn()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(5):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            times[mode] = (time.perf_counter() - t0) / 5
            ttnn.release_trace(device, tid)
        total = nbytes * layers
        print(
            f"STREAMLIN layers={layers} MB={total / 1e6:.1f} dram_us={times[False] * 1e6:.1f} "
            f"({total / times[False] / 1e9:.0f} GB/s) stream_us={times[True] * 1e6:.1f} ({total / times[True] / 1e9:.0f} GB/s) "
            f"drain_us={times['drain'] * 1e6:.1f} ({total / times['drain'] / 1e9:.0f} GB/s)"
        )
    finally:
        core_ranges.set_layout(None)
        st.close()
