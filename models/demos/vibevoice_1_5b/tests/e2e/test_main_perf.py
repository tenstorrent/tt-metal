# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Performance profiling test for the VibeVoice-1.5B end-to-end TTNN TTS pipeline.

Runs the SAME `tt/pipeline.py::run_tts` the demo/PCC gate uses, but bounded and
stripped of any HF golden / PCC comparison — this is a device-perf profile only.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn

# instrument BEFORE importing composites so child forwards are tracked
from models.demos.vibevoice_1_5b.tt import pipeline as P
from models.demos.vibevoice_1_5b.tt._golden import reference as R

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

# small, bounded diffusion sizes for a perf pass (the PCC gate uses N=6, S=5 for accuracy; here we
# just need a representative dispatch-dense forward, not a stress-size run)
_N = int(os.environ.get("TT_PERF_N", "2"))
_S = int(os.environ.get("TT_PERF_S", "2"))

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
def test_main_perf(device_params, device):
    torch.manual_seed(0)

    # build inputs exactly as the PCC gate does; the reference model is only used here to derive
    # config-driven shapes (acoustic_vae_dim) for the noise tensors — no HF golden is generated and
    # no torch forward / PCC comparison is performed.
    model = R.load_reference_model()
    processor = R.build_processor()
    inputs = R.make_inputs(processor, "Speaker 0: Hello there, this is a test.", R.default_voice_sample())
    inputs = dict(inputs)
    inputs["noises"] = R.make_noises(_N + 2, int(model.config.acoustic_vae_dim))

    # drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
    # operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter
    # tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts (sdpa/eltwise/
    # transpose/reduction slip through) and the 12000-marker buffer overflows on some device,
    # dropping ops -> non-reproducible device_ms. Wrapping by TYPE never misses an op.
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

    restore_stubs = P.instrument_stubs()
    _fw0 = time.monotonic()
    try:
        out = P.run_tts(
            device,
            model,
            processor,
            inputs=inputs,
            N=_N,
            S=_S,
            use_trace=False,
            two_cq=False,
        )
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception:
            pass
    finally:
        restore_stubs()
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)

    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
    assert out is not None  # perf only — NO PCC

    if _PERF_TRACE:
        try:
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter

            def _build_for_perf(dev):
                from models.demos.vibevoice_1_5b.tt.pipeline import build_pipeline  # lift the real import

                return build_pipeline(dev, model, processor)

            _prompt_ids = inputs.get("input_ids", None)
            # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
            # traced (+2CQ where the stage stages its inputs). Falls back to the single decode
            # contract for pipelines that expose only decode_step.
            _adapter = PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=1)
            measure_adapter(_adapter, device, mode="auto")
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
