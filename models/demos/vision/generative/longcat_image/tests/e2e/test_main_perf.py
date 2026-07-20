import os
import time

import pytest
import torch

import ttnn
from models.demos.vision.generative.longcat_image.tt import pipeline as P

HF_MODEL_ID = "meituan-longcat/LongCat-Image"

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))  # denoise steps cap
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"


def test_main_perf():
    from diffusers import LongCatImagePipeline

    pipe = LongCatImagePipeline.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.set_progress_bar_config(disable=True)
    max_length = 128  # small fixed cap; demo/prod default (512) is a correctness stress size, not a perf shape
    pipe.tokenizer_max_length = max_length

    size = 256  # small fixed cap; the PCC test's 256 is already representative-dispatch, not max-shape
    num_cqs = 2 if _PERF_TRACE else 1

    open_kwargs = {"l1_small_size": 24576}
    if num_cqs == 2:
        open_kwargs["num_command_queues"] = 2
        open_kwargs["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    device = ttnn.open_device(device_id=0, **open_kwargs)
    try:
        vsf = pipe.vae_scale_factor
        lh = 2 * (size // (vsf * 2))
        gen = torch.Generator("cpu").manual_seed(0)
        raw = torch.randn(1, 16, lh, lh, generator=gen, dtype=torch.float32)
        latents_packed = P._pack_latents(raw, 1, 16, lh, lh)

        ttp = P.LongCatImagePipelineTT(device, pipe, num_cqs=num_cqs, profile=False)

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
            result = ttp.run_text_to_image(
                prompt="a photograph of a cat sitting on a red sofa",
                negative_prompt="",
                height=size,
                width=size,
                num_inference_steps=PERF_MAX_NEW_TOKENS,
                guidance_scale=4.5,
                seed=0,
                max_length=max_length,
                latents_packed=latents_packed,
            )
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert result is not None and result.get("image_denorm") is not None  # perf only — NO PCC

        if _PERF_TRACE:
            try:
                from models.experimental.perf_automation.agent.trace_replay import measure_adapter
                from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter

                def _build_for_perf(dev):
                    return P.LongCatImagePipelineTT(dev, pipe, num_cqs=num_cqs, profile=False)

                _prompt_ids = [0, 1, 2, 3]
                _adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        ttnn.close_device(device)