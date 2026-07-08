import os
import time

import ttnn
from models.demos.hf_eager.longcat_video.tt.pipeline import build_pipeline

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_MAX_LENGTH = int(os.environ.get("TT_PERF_MAX_LENGTH", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"


def _try_open(shape, kwargs):
    try:
        return ttnn.open_mesh_device(shape, **kwargs)
    except TypeError:
        return ttnn.open_mesh_device(shape)


def _open_device():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    kwargs = {}
    if _PERF_TRACE:
        kwargs["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
        kwargs["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))
    try:
        return _try_open(ttnn.MeshShape(1, 4), kwargs)
    except Exception:
        print("[perf] single-chip fallback")
        return _try_open(ttnn.MeshShape(1, 1), kwargs)


def test_text_encode_perf():
    prompt = os.environ.get("TT_PERF_PROMPT", "A cat playing piano in a sunny room")

    dev = _open_device()
    try:
        pipe = build_pipeline(dev)

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
                        ttnn.ReadDeviceProfiler(dev)  # 'dev' = mesh_device on multi-chip
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
            ids = pipe.encode_prompt(prompt, max_length=PERF_MAX_LENGTH)
            out = pipe.run_text_encode(ids)
            try:
                ttnn.ReadDeviceProfiler(dev)
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

                def _build_for_perf(d):
                    from models.demos.hf_eager.longcat_video.tt.pipeline import build_pipeline as _bp

                    return _bp(d)

                _prompt_ids = "A cat playing piano"
                # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
                # traced (+2CQ where the stage stages its inputs). Falls back to the single decode
                # contract for pipelines that expose only decode_step.
                _adapter = PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, dev, mode="auto")
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        ttnn.close_mesh_device(dev)
