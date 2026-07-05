# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace + 2CQ WALL-CLOCK timing for coqui/XTTS-v2 (ms, not PCC).

Opens the device with a trace region + 2 command queues, builds the Pipeline,
and for every stage:
  * captures ONE host-free step in begin/end_trace_capture (cq0),
  * times N eager forwards (no trace)              -> eager_ms,
  * times N traced execute_trace replays (cq0)     -> trace_ms,
  * for the write-staged stages, times execute_trace(cq0) overlapped with an
    input copy on cq1                              -> trace_2cq_ms.
The per-stage traced times are summed into one full-pipeline "step" in ms.
This is the wall-clock number trace+2CQ actually moves (device kernel time is
invariant to trace/2CQ; the win is host-dispatch overlap)."""

from __future__ import annotations

import importlib.util as ilu
import os
import time

import torch

import ttnn
from models.demos.xtts_v2.tt import pipeline as P

HF_MODEL_ID = "coqui/XTTS-v2"
N = 50          # timed replays per stage
WARMUP = 5


def _load_reference():
    here = os.path.dirname(os.path.abspath(__file__))
    rl = os.path.normpath(os.path.join(here, "..", "pcc", "_reference_loader.py"))
    spec = ilu.spec_from_file_location("_reference_loader", rl)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_reference_model(HF_MODEL_ID)


def _time_ms(fn, n, sync):
    for _ in range(WARMUP):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t0) / n * 1e3


def test_trace_2cq_timing():
    torch.manual_seed(0)
    device = ttnn.open_device(
        device_id=0, l1_small_size=24576, trace_region_size=200_000_000, num_command_queues=2
    )

    def sync():
        ttnn.synchronize_device(device)

    rows = []
    try:
        model = _load_reference()
        pipe = P.Pipeline(device, model, capacity=64)
        stages = list(pipe.PIPELINE_STAGES)

        for st in stages:
            pipe._trace_setup(st)

            # (1) eager: run the stage forward with no trace, host dispatch every call
            eager_ms = _time_ms(lambda: pipe._trace_step(st), N, sync)

            # (2) capture ONE host-free step, then time execute_trace replays (cq0)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = pipe._trace_step(st)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            trace_ms = _time_ms(
                lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=True), N, sync
            )

            # (3) 2CQ: overlap the input copy on cq1 with trace replay on cq0
            def _step_2cq():
                pipe._write_inputs(st)  # copy_host_to_device_tensor(..., cq_id=1)
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

            trace_2cq_ms = _time_ms(_step_2cq, N, sync)

            ttnn.release_trace(device, tid)
            ttnn.deallocate(out)
            rows.append((st, eager_ms, trace_ms, trace_2cq_ms))
            print(
                f"[timing] {st:<22} eager={eager_ms:8.3f} ms  "
                f"trace={trace_ms:8.3f} ms  trace+2cq={trace_2cq_ms:8.3f} ms"
            )

        eager_tot = sum(r[1] for r in rows)
        trace_tot = sum(r[2] for r in rows)
        trace2cq_tot = sum(r[3] for r in rows)
        print("=" * 78)
        print(f"[timing] FULL PIPELINE STEP  eager={eager_tot:.3f} ms  "
              f"trace={trace_tot:.3f} ms  trace+2cq={trace2cq_tot:.3f} ms")
        if trace2cq_tot > 0:
            print(f"[timing] trace+2cq speedup vs eager = {eager_tot/trace2cq_tot:.2f}x  "
                  f"({(eager_tot-trace2cq_tot)/eager_tot*100:+.1f}%)")
        # a real number must have been produced for every stage
        assert all(r[2] > 0 for r in rows)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_trace_2cq_timing()
