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

HF_MODEL_ID = pl.HF_MODEL_ID

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

# Perf caps: a SMALL representative sequence, NOT the model's production/max shape. Under tracy every
# device op is instrumented, so a max-seq forward would stall the host in synchronize_device for minutes.
PERF_SEQ = int(os.environ.get("TT_PERF_SEQ", "128"))


def _compose():
    return os.environ.get("TT_E2E_COMPOSE", "1") == "1"


def _load_hf():
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    return model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_main_perf(device_params, device):
    compose = _compose()

    # Build the TTNN pipeline EXACTLY as the e2e test does (hf = weight source for the on-device
    # model, NOT a PCC reference). No torch/PCC comparison here — perf only.
    hf = _load_hf()
    pipe = pl.build_pipeline(device, hf, compose=compose)
    _invocation.reset()

    vocab = int(getattr(hf.config, "vocab_size", 32000))
    ids = torch.randint(0, vocab, (1, PERF_SEQ), dtype=torch.long)

    # Drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
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
        # Bounded forward: a single prefill pass over the small capped sequence.
        out = pipe.forward_logits(ids)
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception:
            pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)

    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
    assert out is not None  # perf only — NO PCC