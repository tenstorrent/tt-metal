import os
import time
import sys

import torch
import pytest

import ttnn
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import pipeline as pl
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt._hf_compat import install_hf_compat

install_hf_compat()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

# perf-only input-size cap: use a SMALL fixed prompt length so every profiled device op stays in a
# representative dispatch-dense pass instead of the model's correctness/max shape (which would run
# orders of magnitude slower and stall ttnn.synchronize_device for minutes under tracy).
PERF_SEQ_LEN = int(os.environ.get("TT_PERF_SEQ_LEN", "128"))
PERF_PROMPT = "The capital of France is"


def test_text_generation_perf():
    # 1) build the pipeline EXACTLY as demo/demo_text_generation.py does
    tok = AutoTokenizer.from_pretrained(pl.HF_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pl.HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    eos = int(getattr(model.config, "eos_token_id", 2))

    input_ids = tok(PERF_PROMPT, return_tensors="pt")["input_ids"]
    # cap the input size small: never exceed a small fixed seq length under tracy
    if input_ids.shape[-1] > PERF_SEQ_LEN:
        input_ids = input_ids[:, :PERF_SEQ_LEN]

    device, is_mesh = pl.open_pipeline_mesh(l1_small_size=24576)

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

    out = None
    _fw0 = time.monotonic()
    try:
        pipe = pl.build_pipeline(device, model, compose=True)
        print(f"[perf] mesh={is_mesh} shard_active={pipe.shard_active}", flush=True)
        # run the pipeline BOUNDED: cap decode via PERF_MAX_NEW_TOKENS
        new_ids, _ = pipe.generate(input_ids, PERF_MAX_NEW_TOKENS, eos_token_id=eos)
        out = new_ids
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception:
            pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)
        pl.close_pipeline_mesh(device, is_mesh)
    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
    assert out is not None  # perf only — NO PCC