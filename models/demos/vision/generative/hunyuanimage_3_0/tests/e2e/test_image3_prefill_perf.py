# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PERFORMANCE test for the `image3_prefill` pipeline of HunyuanImage-3.0.

Builds and runs the SHARED TTNN pipeline (`tt/pipeline.py`) EXACTLY as
`demo/demo_image3_prefill.py` does — in-process so Tracy can profile every
device op — but bounded + profiler-safe. Perf only: no PCC / correctness
assertions.
"""

from __future__ import annotations

import os
import time

import pytest  # noqa: F401  (kept for harness discovery / marker parity)

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0.tt import pipeline as pl

# small, env-overridable seq len — a perf profile only needs a representative
# dispatch-dense pass, never the model's production / max shape (under tracy a
# max-seq forward stalls the host in synchronize_device for minutes).
PERF_SEQ_LEN = int(os.environ.get("TT_PERF_SEQ_LEN", "128"))
PERF_NUM_LAYERS = int(os.environ.get("TT_PERF_NUM_LAYERS", "1"))
PERF_PROMPT = os.environ.get("TT_PERF_PROMPT", "A serene mountain lake at sunrise, photorealistic.")

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"


def _open_perf_mesh():
    """Open the mesh EXACTLY as the demo does (self-opened sharded topology).

    Fabric must be enabled BEFORE opening the mesh (open_mesh_device takes no
    fabric_config kwarg). When TT_PERF_TRACE is set, pass trace_region_size /
    num_command_queues through the SAME open call so the trace replay runs the
    same sharded topology as the eager forward.
    """
    pl.enable_fabric_1d()
    open_kwargs = {"l1_small_size": 24576}
    if _PERF_TRACE:
        open_kwargs["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
        open_kwargs["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*pl._full_mesh_shape()), **open_kwargs)


def test_image3_prefill_perf():
    # DEVICE OPEN — match the source's topology EXACTLY: the demo self-opens a
    # full-mesh sharded device, so we lift that exact open here (no `device`
    # fixture, which would silently disable sharding) and close it in finally.
    device = _open_perf_mesh()
    try:
        # 1) build the pipeline EXACTLY as demo/demo_image3_prefill.py does
        model = pl.load_reference_model()
        pipe = pl.build_pipeline(device, model, num_layers=PERF_NUM_LAYERS, seq_len=PERF_SEQ_LEN)
        inputs = pipe.make_inputs(PERF_PROMPT)

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
                        ttnn.ReadDeviceProfiler(device)  # 'device' = mesh_device on multi-chip
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
        _fw0 = time.monotonic()
        try:
            # BOUNDED: a single prefill forward (prefill pipeline has no decode loop).
            hidden_tt, l_aux_tt = pipe.run_prefill(inputs)
            out = hidden_tt
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only — NO PCC

        if _PERF_TRACE:
            try:
                from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
                from models.experimental.perf_automation.agent.trace_replay import measure_adapter

                def _build_for_perf(dev):
                    from models.demos.vision.generative.hunyuanimage_3_0.tt.pipeline import build_pipeline

                    return build_pipeline(dev, model, num_layers=PERF_NUM_LAYERS, seq_len=PERF_SEQ_LEN)

                _prompt_ids = inputs["input_ids"][0, :16].tolist()
                # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
                # traced (+2CQ where the stage stages its inputs). Falls back to the single decode
                # contract for pipelines that expose only decode_step.
                _adapter = PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        pl._close_device(device)
