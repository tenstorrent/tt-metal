# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace-replay performance for `voxtral-tts-full` -- the measurement the trace gate reads.

The unit is ONE decode frame: `tt/pipeline.py::decode_step` advancing a single position against
the RESIDENT KV cache, at the fixed capacity `decode_trace_setup` pins.  That is what a real
generation loop repeats, so it is what a trace should record and replay.

The shape of the measurement is what makes it a trace measurement rather than a wall clock:

  * every input the step reads is staged BEFORE `ttnn.begin_trace_capture` -- the KV cache is
    seeded by `decode_trace_setup`, the step embedding is uploaded there too, and the position is
    pinned to a fixed integer.  A host->device write inside a capture is a fatal on a mesh and
    would hang the device here.
  * the capture wraps `decode_trace_step()` and NOTHING else.  Wrapping the driver (`run_tts`)
    would put the prompt upload and the stop check inside the trace.
  * the replay is checked against the eager result before it is timed: a trace that replays the
    wrong arithmetic is not a faster pipeline.

It reports `TRACE_PER_TOKEN_MS` (mean over the replays) and `TRACE_REPLAY_PATH=trace+1cq`, the
markers the perf harness parses into `test_tts_perf.py.trace_caps.json`.

    python -m pytest models/demos/voxtral_tts_full/tests/e2e/test_tts_perf.py -svv

Knobs, all read from the environment the harness sets: `TT_PERF_OSL_TOKENS` (replays to average
over), `TT_PERF_LAYERS` (cap the repeated stacks for a profiling build), `TT_PERF_TRACE_REGION`
(trace region bytes, when a capture needs more than the pipeline's default).
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn

from models.demos.voxtral_tts_full.tt import pipeline as P
from models.demos.voxtral_tts_full.tt import reference as ref

REPLAY_PCC = 0.99


@pytest.fixture(scope="module")
def perf_device():
    """This module's own device.  `trace_region_size` is the pipeline's default unless the harness
    grew it after a capture reported the exact bytes it needed."""
    region = int(os.environ.get("TT_PERF_TRACE_REGION") or P.DEFAULT_TRACE_REGION_SIZE)
    print(f"[perf] opening device with trace_region_size={region}", flush=True)
    dev = ttnn.open_device(device_id=0, trace_region_size=region)
    try:
        yield dev
    finally:
        ttnn.close_device(dev)


@pytest.fixture(scope="module")
def perf_pipe(perf_device):
    """The resident pipeline: weights staged once, then never touched by the timed path."""
    cap = os.environ.get("TT_PERF_LAYERS")
    t0 = time.time()
    print("[perf] loading the reference checkpoint", flush=True)
    model = ref.load_hf_model(dtype=torch.float32)
    print(f"[perf] loaded in {time.time() - t0:.0f}s; staging weights on device", flush=True)
    t0 = time.time()
    pipe = P.build_pipeline(perf_device, model=model, layers=int(cap) if cap else None)
    print(f"[perf] pipeline built in {time.time() - t0:.0f}s "
          f"(backbone {len(pipe.backbone_layers)}/{pipe.depths['backbone_total']} layers)", flush=True)
    return pipe


def test_tts_perf(perf_pipe, perf_device):
    """Capture one decode frame, replay it, and report the per-frame trace time."""
    pipe, device = perf_pipe, perf_device
    replays = int(os.environ.get("TT_PERF_OSL_TOKENS") or 32)

    # ---- everything the step reads is resident BEFORE the capture ------------------------------
    pinned = pipe.decode_trace_setup(pipe.decode_trace_inputs())
    print(f"[perf] decode pinned at C={pinned['C']}, step position {pinned['pos']}", flush=True)

    # First call compiles the kernels; the second is the steady state the capture will record.
    pipe.decode_trace_step()
    eager = ttnn.to_torch(pipe.decode_trace_step()).float()
    ttnn.synchronize_device(device)
    print("[perf] eager step warm, capturing", flush=True)

    # ---- the capture: one fixed-shape, host-op-free step ---------------------------------------
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = pipe.decode_trace_step()
    ttnn.end_trace_capture(device, tid, cq_id=0)

    try:
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        replayed = ttnn.to_torch(out).float()
        replay_pcc = ref.pcc(eager, replayed)
        print(f"trace replay PCC={replay_pcc}")
        assert replay_pcc >= REPLAY_PCC, (
            f"the replayed trace does not reproduce the eager decode step "
            f"(PCC {replay_pcc:.6f} < {REPLAY_PCC}) -- the measurement below would be timing the "
            f"wrong arithmetic")

        t0 = time.time()
        for _ in range(replays):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        elapsed = time.time() - t0
    finally:
        ttnn.release_trace(device, tid)

    per_token_ms = elapsed / replays * 1e3
    print(f"[perf] {replays} replays in {elapsed:.3f}s at C={pinned['C']} "
          f"over {pipe.depths['decode']} backbone layers")
    print("TRACE_REPLAY_PATH=trace+1cq")
    print(f"TRACE_PER_TOKEN_MS={per_token_ms:.4f}")
    assert per_token_ms > 0.0, "trace replay reported no elapsed time"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-vv"]))
