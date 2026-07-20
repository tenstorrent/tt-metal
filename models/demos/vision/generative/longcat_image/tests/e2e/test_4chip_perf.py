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
_PERF_CQ = int(os.environ.get("TT_PERF_NUM_CQ", "2"))
_PERF_L1_SMALL = 24576
_PERF_TRACE_REGION = int(os.environ.get("TT_PERF_TRACE_REGION", "209715200"))
# small, perf-only text length -- the demo's max_length=512 is a correctness/max-shape config, not
# representative of a dispatch-dense perf pass, and a full 512-token encode pass drags the profiler.
_PERF_MAX_LENGTH = int(os.environ.get("TT_PERF_MAX_LENGTH", "128"))
_PERF_SIZE = int(os.environ.get("TT_PERF_SIZE", "512"))


def test_4chip_perf():
    if ttnn.get_num_devices() != 4:
        pytest.skip(f"test_4chip_perf needs exactly 4 chips; found {ttnn.get_num_devices()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from diffusers import LongCatImagePipeline

    pipe = LongCatImagePipeline.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.set_progress_bar_config(disable=True)
    pipe.tokenizer_max_length = _PERF_MAX_LENGTH

    # ONE coherent 4-chip ring mesh; encoder + DiT + VAE all tensor-parallel across it (tp=4) -- lifted
    # verbatim from demo_4chip.py's self-open (a plain `device` fixture would disable tp sharding).
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.RELAXED_INIT)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=_PERF_L1_SMALL,
        num_command_queues=_PERF_CQ if _PERF_TRACE else 1,
        trace_region_size=_PERF_TRACE_REGION if _PERF_TRACE else 0,
    )

    ttp = None
    counter = [0]
    _orig = []

    def _draining(fn):
        def inner(*a, **k):
            r = fn(*a, **k)
            counter[0] += 1
            if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                try:
                    ttnn.ReadDeviceProfiler(mesh)
                except Exception:
                    pass
            return r

        return inner

    try:
        ttp = P.LongCatImagePipelineTT(
            mesh, pipe, num_cqs=_PERF_CQ if _PERF_TRACE else 1, text_encoder_device=mesh,
            resident_text_encoder=True, profile=True,
        )

        # MODEL-AGNOSTIC drain: wrap every dispatched ttnn op (type 'FastOperation') across ttnn and
        # its op submodules, so the flush counter tracks TOTAL device dispatch for any op mix.
        _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
        for _mod in [_m for _m in _mods if _m is not None]:
            for _n in dir(_mod):
                _op = getattr(_mod, _n, None)
                if type(_op).__name__ == "FastOperation":
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))

        # Upload the RESIDENT tp=4 encoder FIRST (one throwaway forward) so warmup()'s DiT trace
        # capture reserves around the resident encoder weights rather than colliding with them.
        ids, mask, pre, suf = build_text_input_ids(pipe, PERF_PROMPT, _PERF_MAX_LENGTH)
        te, _owned = ttp._acquire_text_encoder()
        ttp._tt_text_encode(ids, mask, pre, suf, stub=te)
        ttp.warmup(max_length=_PERF_MAX_LENGTH, height=_PERF_SIZE, width=_PERF_SIZE, guidance_scale=4.5)

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
            ttnn.ReadDeviceProfiler(mesh)
        except Exception:
            pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)

    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
    assert out is not None
    assert out.get("image_denorm") is not None
    _save_png(out["image_denorm"], os.path.join(OUTPUT_DIR, "perf_4chip.png"))

    if _PERF_TRACE:
        try:
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter
            from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter

            def _build_for_perf(dev):
                from diffusers import LongCatImagePipeline as _LCIP

                _pipe = _LCIP.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
                _pipe.set_progress_bar_config(disable=True)
                _pipe.tokenizer_max_length = _PERF_MAX_LENGTH
                return P.LongCatImagePipelineTT(
                    dev, _pipe, num_cqs=_PERF_CQ, text_encoder_device=dev,
                    resident_text_encoder=True, profile=True,
                )

            _prompt_ids, _, _, _ = build_text_input_ids(pipe, PERF_PROMPT, _PERF_MAX_LENGTH)
            _adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1)
            measure_adapter(_adapter, mesh, mode="auto")
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)

    if ttp is not None:
        ttp.close()
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)