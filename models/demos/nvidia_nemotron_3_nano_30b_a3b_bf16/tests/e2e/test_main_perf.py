import os
import time

import pytest
import torch

import ttnn

from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import pipeline as pl
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt._hf_compat import install_hf_compat

install_hf_compat()

from transformers import AutoModelForCausalLM  # noqa: E402

HF_MODEL_ID = pl.HF_MODEL_ID

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_SEQ_LEN = int(os.environ.get("TT_PERF_SEQ_LEN", "128"))


def _compose():
    return os.environ.get("TT_E2E_COMPOSE", "1") == "1"


def _load_hf():
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    return model


def test_main_perf():
    # Build + run the on-device TTNN pipeline EXACTLY as the e2e demo does, but perf-only:
    # this self-managed 4-chip TP=2 x DP=2 mesh is opened here (in-process) so tracy sees
    # every device op. No reference torch forward, no PCC.
    compose = _compose()

    device, is_mesh = pl.open_pipeline_mesh(l1_small_size=24576)
    hf = _load_hf()
    pipe = pl.build_pipeline(device, hf, compose=compose)
    print(
        f"[perf] mesh={is_mesh} shape={list(device.shape) if is_mesh else [1, 1]} "
        f"shard_active={pipe.shard_active} compose={compose}",
        flush=True,
    )

    # Small, fixed, representative input — NOT the model's production/max shape. Under tracy
    # every op is instrumented; a max-seq forward would stall in synchronize_device for minutes.
    vocab = int(getattr(hf.config, "vocab_size", 32000))
    seq = max(1, PERF_SEQ_LEN)
    ids = (torch.arange(1, seq + 1, dtype=torch.long) % vocab).unsqueeze(0)
    eos = int(getattr(hf.config, "eos_token_id", 2))

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
        # Bounded on-device forward: prefill + a capped greedy decode.
        out, _step_logits = pipe.generate(ids, PERF_MAX_NEW_TOKENS, eos_token_id=eos)
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception:
            pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)
        try:
            pl.close_pipeline_mesh(device, is_mesh)
        except Exception:
            pass

    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
    assert out is not None