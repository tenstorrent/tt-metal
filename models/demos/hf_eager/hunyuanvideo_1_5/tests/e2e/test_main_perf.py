import os
import time

import pytest

import ttnn

# ---------------------------------------------------------------------------
# HuggingFace cache redirect (fixes: PermissionError [Errno 13] '/localdev/hub')
# ---------------------------------------------------------------------------
# The real-weight DiT build (build_real_transformer) resolves its checkpoint via
# huggingface_hub, whose default cache in this environment is /localdev/hub -- a
# path this process cannot write, so the build aborts with a PermissionError
# BEFORE any TTNN op runs. Redirect every HF cache env var to a writable location
# up front, BEFORE importing the model modules (transformers/huggingface_hub read
# these paths once at import). Override, do not setdefault: an inherited
# HF_HOME/HF_HUB_CACHE already pointing at /localdev is exactly what we must undo.
_HF_HOME = os.environ.get("TT_PERF_HF_HOME") or os.path.expanduser("~/.cache/huggingface")
try:
    os.makedirs(os.path.join(_HF_HOME, "hub"), exist_ok=True)
except Exception:
    pass
os.environ["HF_HOME"] = _HF_HOME
for _v in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
    _cur = os.environ.get(_v)
    if not _cur or _cur.startswith("/localdev"):
        # Point derived caches under the writable HF_HOME (or drop the bad override entirely).
        os.environ[_v] = os.path.join(_HF_HOME, "datasets" if "DATASETS" in _v else "hub")

from models.demos.hf_eager.hunyuanvideo_1_5.real_weights.weights import build_real_transformer
from models.demos.hf_eager.hunyuanvideo_1_5.tt import pipeline as P

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

# This is a diffusion VIDEO model: its device op/dispatch count is driven by DiT depth (num_layers),
# denoise TIMESTEPS, and latent FRAMES -- NOT by a token/sequence length. Cap ALL of them to small,
# env-overridable, representative sizes so a single dispatch-dense forward profiles fast under tracy
# (a full 50-step / full-frame / 54-layer forward blocks the host for many minutes).
PERF_N_LAYERS = int(os.environ.get("HY_N", "2"))  # DiT depth (source default is 2; do NOT use 54)
PERF_STEPS = int(os.environ.get("TT_PERF_STEPS", "2"))  # denoise timesteps (heavy diffusion axis)
PERF_FRAMES = int(os.environ.get("TT_PERF_FRAMES", "1"))  # latent frames (heavy video axis)


def coerce_bf16():
    """Run the DiT in bf16 (weights+activations) so full layers fit one chip:
    map every ttnn.float32 request -> bfloat16 (stubs keep fp32 dest-accum)."""
    import ttnn

    _from, _tc = ttnn.from_torch, ttnn.typecast
    ttnn.from_torch = lambda t, *a, **k: _from(
        t, *a, **{**k, "dtype": ttnn.bfloat16} if k.get("dtype") == ttnn.float32 else k
    )
    ttnn.typecast = lambda t, dt, *a, **k: _tc(t, ttnn.bfloat16 if dt == ttnn.float32 else dt, *a, **k)


def _trim_config(config):
    """Best-effort: shrink the diffusion timestep count and video frame count on the model config so
    the profiled forward stays small on the model-specific heavy axes. Only touches attributes that
    actually exist; a config that names them differently simply keeps its own (already-small) shape."""
    for _name in ("num_inference_steps", "num_steps", "num_train_timesteps", "sampling_steps", "steps"):
        if hasattr(config, _name):
            try:
                setattr(config, _name, PERF_STEPS)
            except Exception:
                pass
    for _name in ("num_frames", "video_length", "num_latent_frames", "frames", "n_frames"):
        if hasattr(config, _name):
            try:
                setattr(config, _name, PERF_FRAMES)
            except Exception:
                pass


_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    # Reserve the trace + 2-CQ budget at device-open, ONCE, for baseline and every candidate: the
    # second queue and the trace region exist before any candidate runs, so trace+2CQ is the fixed
    # measurement mode (never a per-candidate downgrade for lack of a queue). A device/config that
    # genuinely can't open 2 CQs still degrades gracefully in measure_adapter; override with TT_PERF_NUM_CQ.
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
def test_main_perf(device_params, device):
    # Build the TTNN pipeline EXACTLY as the PCC source does (genuine single-device 'device' fixture),
    # but keep ONLY the on-device forward: no torch/diffusers golden, no pcc/comp_pcc/assert.
    if os.environ.get("HY_BF16") == "1":
        coerce_bf16()
    print(f"\n[main perf] building real DiT at num_layers={PERF_N_LAYERS} ...", flush=True)
    model = build_real_transformer(num_layers=PERF_N_LAYERS)
    _trim_config(model.config)
    pipe = P.build_pipeline(device, model)
    inputs = P.build_inputs(model.config, task="t2v")

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

    _fw0 = time.monotonic()
    try:
        out = pipe.run(inputs, granularity="composite")  # single bounded forward (steps/frames/layers capped)
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
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

            def _build_for_perf(dev):
                from models.demos.hf_eager.hunyuanvideo_1_5.tt.pipeline import build_pipeline

                _m = build_real_transformer(num_layers=PERF_N_LAYERS)
                _trim_config(_m.config)
                return build_pipeline(dev, _m)

            _prompt_ids = [0]
            # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
            # traced (+2CQ where the stage stages its inputs). Falls back to the single decode
            # contract for pipelines that expose only decode_step.
            _adapter = PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=1)
            measure_adapter(_adapter, device, mode="auto")
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)