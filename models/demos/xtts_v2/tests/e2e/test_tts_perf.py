# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Performance (profiler) test for the 'tts' TTNN pipeline of coqui/XTTS-v2.

Builds the SAME chained TTNN pipeline as demo/demo_tts.py once
(`models.demos.xtts_v2.tt.pipeline.build_pipeline`) and times its forward:
text + speaker reference -> 24 kHz speech waveform, produced entirely by the
native TTNN modules.

This is a PERF-ONLY test: it runs the device forward IN-PROCESS (so Tracy can
see every op), bounds the work so the profiler's marker buffer never overflows,
and drains the device profiler periodically. No PCC / correctness assertions.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.demos.xtts_v2 import reference
from models.demos.xtts_v2.tt import pipeline as P

HF_MODEL_ID = "coqui/XTTS-v2"

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    # sized during bring-up to fit the traced decode step; raise via TT_PERF_TRACE_REGION if capture OOMs
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))


def _load_reference():
    return reference.load_reference_model(HF_MODEL_ID)


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
def test_tts_perf(device_params, device):
    torch.manual_seed(0)

    # 1) build the reference model EXACTLY as demo/demo_tts.py does
    model = _load_reference()

    # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
    #    operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter
    #    tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts (sdpa/eltwise/
    #    transpose/reduction slip through) and the 12000-marker buffer overflows on some device,
    #    dropping ops -> non-reproducible device_ms. Wrapping by TYPE never misses an op.
    counter = [0]
    _orig = []

    def _draining(fn):
        def inner(*a, **k):
            r = fn(*a, **k)
            counter[0] += 1
            if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            return r

        return inner

    _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
    for _mod in [_m for _m in _mods if _m is not None]:
        for _n in dir(_mod):
            _op = getattr(_mod, _n, None)
            if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                _orig.append((_mod, _n, _op))
                setattr(_mod, _n, _draining(_op))

    try:
        # build ONCE, outside the timed region: a served model uploads the 466.87 M
        # parameters a single time per process, so the forward wall below excludes it.
        # (The tracy profile still covers the build ops — device_ms stays comparable
        # to the historical build+single-forward profile of this test.)
        _b0 = time.monotonic()
        pipe = P.build_pipeline(device, model)
        ttnn.synchronize_device(device)
        print("BUILD_WALL_MS=%.4f" % ((time.monotonic() - _b0) * 1000.0), flush=True)
        # run the pipeline BOUNDED: cap the AR decode horizon via PERF_MAX_NEW_TOKENS and keep the
        # text prompt small (a representative dispatch-dense pass, not a max-shape stress run).
        # Sync on both sides of the timed region: ttnn calls are async enqueues, so an
        # unsynced wall under-measures.
        ttnn.synchronize_device(device)
        _fw0 = time.monotonic()
        out = pipe.forward(text="hello world.", language="en", N=PERF_MAX_NEW_TOKENS)
        ttnn.synchronize_device(device)
        _fwd_ms = (time.monotonic() - _fw0) * 1000.0
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception:
            pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)
    print("FORWARD_WALL_MS=%.4f" % _fwd_ms)
    assert out is not None  # perf only — NO PCC


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
def test_tts_perf_warm(device_params, device):
    # This test measures the FULL model wall: undo the module-level perf-only depth cap
    # (TT_PERF_LAYERS, set above for the tracy-profiled case) so the wall is the real 30-layer
    # forward, not the reduced-depth profile configuration.
    os.environ.pop("TT_PERF_LAYERS", None)
    """Cold-vs-warm end-to-end wall at the gated horizon — the served-model number.

    Builds once (`build_pipeline`), then times the SAME forward twice: the first
    utterance in the process (cold: first-touch program compiles not yet cached
    in-process) and the second (warm: what every later utterance pays). Runs WITHOUT
    tracy and is not the profiled case (`test_tts_perf` is), so its second forward
    never enters the device_ms profile. Horizon: XTTS_WARM_N (default 40, the gate N).
    """
    torch.manual_seed(0)
    model = _load_reference()
    pipe = P.build_pipeline(device, model)
    n = int(os.environ.get("XTTS_WARM_N", str(PERF_MAX_NEW_TOKENS)))
    ttnn.synchronize_device(device)
    _c0 = time.monotonic()
    pipe.forward(text="hello world.", language="en", N=n)
    ttnn.synchronize_device(device)
    print("FORWARD_WALL_COLD_MS=%.4f" % ((time.monotonic() - _c0) * 1000.0), flush=True)
    _w0 = time.monotonic()
    out = pipe.forward(text="hello world.", language="en", N=n)
    ttnn.synchronize_device(device)
    print("FORWARD_WALL_WARM_MS=%.4f" % ((time.monotonic() - _w0) * 1000.0), flush=True)
    assert out is not None  # perf only — NO PCC
