import os
import time

import pytest
import torch

import ttnn
from models.demos.vision.generative.longcat_image.demo._demo_common import save_png as _save_png
from models.demos.vision.generative.longcat_image.tt import pipeline as P
from models.demos.vision.generative.longcat_image.tt.pipeline import build_text_input_ids

HF_MODEL_ID = "meituan-longcat/LongCat-Image"
OUTPUT_DIR = "outputs"
PERF_PROMPT = "a cat sitting on a mat"

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_PERF_CQ = int(os.environ.get("TT_PERF_NUM_CQ", "2")) if _PERF_TRACE else 1
_PERF_L1_SMALL = 24576
_PERF_TRACE_REGION = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
# small, perf-only shapes -- the demo's defaults (512x512, max_length=512) are a correctness/
# production config, not a representative dispatch-dense perf pass; a full 512x512 + max_length=512
# request (Qwen text encoder + DiT + VAE at full res) drags the profiler for many minutes.
_PERF_SIZE = int(os.environ.get("TT_PERF_SIZE", "128"))
_PERF_MAX_LENGTH = int(os.environ.get("TT_PERF_MAX_LENGTH", "64"))
_PERF_DEVICE_ID = int(os.environ.get("TT_PERF_DEVICE_ID", "0"))
_PERF_TEXT_ENCODER_DEVICE_ID = int(os.environ.get("TT_PERF_TEXT_ENCODER_DEVICE_ID", "1"))


def test_server_perf():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from diffusers import LongCatImagePipeline

    pipe = LongCatImagePipeline.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.set_progress_bar_config(disable=True)
    pipe.tokenizer_max_length = _PERF_MAX_LENGTH

    # Self-opened exactly as demo_server.py does: the two chips (resident DiT+VAE / resident text
    # encoder) as ONE coherent 1x2 mesh, then split into two independently-addressable 1x1
    # submeshes -- NOT two independent ttnn.open_device() calls (see demo_server.py's module
    # docstring: opening individual chips out of a fabric-connected QB2 cluster one at a time is
    # flagged by tt-metal as slow/fragile and was observed hanging on real hardware).
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 2),
        l1_small_size=_PERF_L1_SMALL,
        num_command_queues=_PERF_CQ,
        trace_region_size=_PERF_TRACE_REGION if _PERF_TRACE else 0,
        physical_device_ids=[_PERF_DEVICE_ID, _PERF_TEXT_ENCODER_DEVICE_ID],
    )
    device = None
    text_encoder_device = None
    ttp = None
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

    try:
        device = mesh.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, 0))
        text_encoder_device = mesh.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, 1))

        ttp = P.LongCatImagePipelineTT(
            device, pipe, text_encoder_device=text_encoder_device, num_cqs=_PERF_CQ, profile=True
        )
        ttp.warmup(max_length=_PERF_MAX_LENGTH, height=_PERF_SIZE, width=_PERF_SIZE, guidance_scale=4.5)

        # MODEL-AGNOSTIC drain: wrap every dispatched ttnn op (type 'FastOperation') across ttnn and
        # its op submodules, so the flush counter tracks TOTAL device dispatch for any op mix.
        _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
        for _mod in [_m for _m in _mods if _m is not None]:
            for _n in dir(_mod):
                _op = getattr(_mod, _n, None)
                if type(_op).__name__ == "FastOperation":
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))

        _fw0 = time.monotonic()
        out = ttp.run_text_to_image(
            prompt=PERF_PROMPT,
            height=_PERF_SIZE,
            width=_PERF_SIZE,
            num_inference_steps=PERF_MAX_NEW_TOKENS,
            guidance_scale=4.5,
            seed=0,
            max_length=_PERF_MAX_LENGTH,
        )
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception:
            pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)

    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
    assert out is not None
    assert out.get("image_denorm") is not None
    _save_png(out["image_denorm"], os.path.join(OUTPUT_DIR, "perf_server.png"))

    if _PERF_TRACE:
        try:
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter
            from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter

            def _build_for_perf(dev):
                from diffusers import LongCatImagePipeline as _LCIP

                _pipe = _LCIP.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
                _pipe.set_progress_bar_config(disable=True)
                _pipe.tokenizer_max_length = _PERF_MAX_LENGTH
                return P.LongCatImagePipelineTT(dev, _pipe, profile=True)

            # run_text_to_image's denoise is eager (not decode_step-trace-capturable) -- the prompt
            # ids below are only for the adapter's interface; PipelineDecodeAdapter will raise if the
            # built pipeline has no trace-capturable decode_step, and the guard below falls back to
            # FORWARD_WALL_MS, which is expected for this pipeline.
            _prompt_ids, _, _, _ = build_text_input_ids(pipe, PERF_PROMPT, _PERF_MAX_LENGTH)
            _adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1)
            measure_adapter(_adapter, device, mode="auto")
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)

    if ttp is not None:
        ttp.close()
    if text_encoder_device is not None:
        ttnn.close_mesh_device(text_encoder_device)
    if device is not None:
        ttnn.close_mesh_device(device)
    ttnn.close_mesh_device(mesh)