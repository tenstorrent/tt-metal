# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Performance profiling test for the `meituan-longcat/LongCat-Video` main T2V pipeline.

Runs the ONE shared pipeline in `tt/pipeline.py` on the SAME 1x4 mesh topology as
tests/e2e/test_pipeline_e2e.py's `_open_mesh` (with 1x1 fallback), driving the full
text_encode -> denoise -> vae-decode path via `run_t2v` with NO correctness checks.
Prints FORWARD_WALL_MS for the bounded eager forward and, when TT_PERF_TRACE=1, runs
the generic trace+2CQ stage adapter over the resident pipeline object.
"""

from __future__ import annotations

import os
import time

import pytest

import ttnn
from models.demos.hf_eager.longcat_video.tt.pipeline import build_pipeline

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))  # diffusion steps cap
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))


def _open_mesh():
    """Mirrors tests/e2e/test_pipeline_e2e.py::_open_mesh exactly (1x4 TP mesh, 1x1 fallback),
    with the perf device params (trace_region_size / num_command_queues) layered onto the open
    call so the trace+2CQ block below runs on the SAME sharded topology as the eager forward."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    try:
        return ttnn.open_mesh_device(ttnn.MeshShape(1, 4), **_DEV_PARAMS), True
    except Exception:
        return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **_DEV_PARAMS), False


def test_main_perf():
    device, is_mesh = _open_mesh()
    try:
        # 1) build the pipeline EXACTLY as the e2e test does
        pipe = build_pipeline(device)

        # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY
        #    ttnn operation (type 'FastOperation') across ttnn + its op submodules, so the flush
        #    counter tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts
        #    (sdpa/eltwise/transpose/reduction slip through) and the 12000-marker buffer overflows
        #    on some device, dropping ops -> non-reproducible device_ms. Wrapping by TYPE never
        #    misses an op.
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
            # bounded main-pipeline forward: small prompt, single frame, small spatial size,
            # diffusion steps capped via TT_PERF_MAX_NEW_TOKENS (mirrors the pcc test's shapes)
            out = pipe.run_t2v(
                "A cat playing piano in a sunny room",
                num_frames=1,
                height=32,
                width=32,
                steps=PERF_MAX_NEW_TOKENS,
            )
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only -- NO PCC

        if _PERF_TRACE:
            try:
                from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
                from models.experimental.perf_automation.agent.trace_replay import measure_adapter

                def _build_for_perf(dev):
                    from models.demos.hf_eager.longcat_video.tt.pipeline import build_pipeline as _bp

                    return _bp(dev)

                _prompt_ids = pipe.encode_prompt("A cat playing piano", max_length=32)
                # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry
                # gets traced (+2CQ where the stage stages its inputs). Falls back to the single
                # decode contract for pipelines that expose only decode_step.
                _adapter = PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s"]))
