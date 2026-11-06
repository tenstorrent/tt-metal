# The schedule-driven sender with a minimal consumer (stream.drain).
import numpy as np
import pytest

import ttnn
from models.demos.blackhole.bonsai2.tt.stream.stream import PPS, Plan, Stream, drain


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20_000_000}], indirect=True)
def test_stream_drain(device):
    st = Stream(device)
    try:
        plan = Plan(st)
        takes = []
        for k in range(3):
            pages = {i: np.zeros(((i % 5 + 1) * PPS, 1024), np.uint32) for i in range(len(st.rx_order) - 9 * k)}
            takes.append(plan.add(pages))
        plan.build()
        print("DRAIN built", plan.slots[:4], flush=True)
        b = [i for i, c in enumerate(st.senders) if (c.x, c.y) == (1, 0)][0]
        print("DRAIN who", b, plan.who[b][:45], plan.sched_stride, flush=True)
        print("DRAIN dev", ttnn.to_torch(plan.sched).reshape(-1)[b * 256 : b * 256 + 45].tolist(), flush=True)
        st.launch(plan)
        for t in takes:
            drain(st, t)
            ttnn.synchronize_device(device)
            print("DRAIN op done", flush=True)
    finally:
        st.close()


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20_000_000}], indirect=True)
@pytest.mark.parametrize("zero", [True, False])
def test_stream_uniform(device, zero):
    """The prototype's workload through Plan: every receiver 16 slots per op, 20 ops; zero or random pages."""
    import time

    st = Stream(device)
    try:
        plan = Plan(st)
        rng = np.random.default_rng(0)
        takes = []
        for k in range(20):
            if zero:
                pages = {i: np.zeros((16 * PPS, 1024), np.uint32) for i in range(len(st.rx_order))}
            else:
                pages = {
                    i: rng.integers(0, 2**32, (16 * PPS, 1024), dtype=np.uint32) for i in range(len(st.rx_order))
                }
            takes.append(plan.add(pages))
        plan.build()

        import os

        def step():
            st.launch(plan)
            if os.environ.get("NO_DRAIN") == "1":
                return
            for t in takes:
                drain(st, t)

        step()
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        step()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            if os.environ.get("NO_DRAIN") == "1":  # the senders' sub-device is not in the stall group
                ttnn.synchronize_device(device, sub_device_ids=[ttnn.SubDeviceId(0)])
        ttnn.synchronize_device(device)
        s = (time.perf_counter() - t0) / 5
        nbytes = sum(plan.slots) * st.slot
        print(f"UNIFORM zero={zero} MB={nbytes / 1e6:.1f} us={s * 1e6:.1f} GB/s={nbytes / s / 1e9:.1f}", flush=True)
    finally:
        st.close()
