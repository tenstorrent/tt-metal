import json
import os
import time
from pathlib import Path

import pytest
import torch
import ttnn

from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import _invocation
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import pipeline as pl
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt._hf_compat import install_hf_compat

install_hf_compat()

from transformers import AutoModelForCausalLM  # noqa: E402

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# small representative prefill length — do NOT use the model's production/max seq under tracy.
PERF_SEQ_LEN = int(os.environ.get("TT_PERF_SEQ_LEN", "128"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

HF_MODEL_ID = pl.HF_MODEL_ID


def _compose():
    return os.environ.get("TT_E2E_COMPOSE", "1") == "1"


def _load_hf():
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    return model


def _small_input_ids():
    # small fixed prefill (valid low token ids); NOT the golden / max-seq input.
    return (torch.arange(PERF_SEQ_LEN, dtype=torch.long) % 4096).unsqueeze(0)


def test_main_perf():
    compose = _compose()

    # 1) build the TTNN pipeline EXACTLY as the e2e test does (mesh + HF-sourced weights),
    #    keeping ONLY the on-device forward — no golden load, no PCC.
    dev, is_mesh = pl.open_pipeline_mesh(l1_small_size=24576)
    hf = _load_hf()
    pipe = pl.build_pipeline(dev, hf, compose=compose)
    eos = int(getattr(hf.config, "eos_token_id", 2))
    print(
        f"[perf] mesh={is_mesh} shape={list(dev.shape) if is_mesh else [1, 1]} "
        f"shard_active={pipe.shard_active} compose={compose} seq_len={PERF_SEQ_LEN}",
        flush=True,
    )

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
                    ttnn.ReadDeviceProfiler(dev)
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

    _invocation.reset()
    ids = _small_input_ids()
    _fw0 = time.monotonic()
    try:
        # bounded greedy decode: small prefill + PERF_MAX_NEW_TOKENS steps.
        tt_new_ids, _tt_step_logits = pipe.generate(ids, PERF_MAX_NEW_TOKENS, eos_token_id=eos)
        out = tt_new_ids
        try:
            ttnn.ReadDeviceProfiler(dev)
        except Exception:
            pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)
        try:
            pl.close_pipeline_mesh(dev, is_mesh)
        except Exception:
            pass

    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
    assert out is not None  # perf only — NO PCC