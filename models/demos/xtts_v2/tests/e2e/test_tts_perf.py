import importlib.util as ilu
import os
import time

import pytest
import torch

import ttnn
from models.demos.xtts_v2.tt import pipeline as P

HF_MODEL_ID = "coqui/XTTS-v2"

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

# small representative text prompt — perf needs a dispatch-dense pass, NOT the demo's production shape
PERF_TEXT = os.environ.get("TT_PERF_TEXT", "hello world.")
PERF_LANGUAGE = os.environ.get("TT_PERF_LANGUAGE", "en")

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"


def _load_reference():
    # mirror demo/demo_tts.py::_load_reference exactly, resolving the loader relative to THIS file
    here = os.path.dirname(os.path.abspath(__file__))
    rl = os.path.normpath(os.path.join(here, "..", "pcc", "_reference_loader.py"))
    spec = ilu.spec_from_file_location("_reference_loader", rl)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_reference_model(HF_MODEL_ID)


def test_tts_perf():
    torch.manual_seed(0)

    # DEVICE OPEN — the demo SELF-OPENS a single device via ttnn.open_device(device_id=0, l1_small_size=24576).
    # Lift that exact open into the test; add trace_region_size / num_command_queues when TT_PERF_TRACE.
    _open_kwargs = {"device_id": 0, "l1_small_size": 24576}
    if _PERF_TRACE:
        _open_kwargs["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
        _open_kwargs["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))
    device = ttnn.open_device(**_open_kwargs)

    try:
        model = _load_reference()

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
                if type(_op).__name__ == "FastOperation":
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))

        _fw0 = time.monotonic()
        try:
            # run the pipeline EXACTLY as demo/demo_tts.py does, but BOUNDED: cap the AR decode
            # horizon N via PERF_MAX_NEW_TOKENS and use a small text prompt.
            out = P.run_tts(device, model, text=PERF_TEXT, language=PERF_LANGUAGE, N=PERF_MAX_NEW_TOKENS)
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
                from models.experimental.perf_automation.agent.trace_replay import measure_adapter
                from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter

                def _build_for_perf(dev):
                    # build the pipeline EXACTLY as this test builds it, on the passed-in device
                    _m = _load_reference()
                    return P.run_tts(dev, _m, text=PERF_TEXT, language=PERF_LANGUAGE, N=PERF_MAX_NEW_TOKENS)

                _prompt_ids = [1, 2, 3]
                _adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        ttnn.close_device(device)