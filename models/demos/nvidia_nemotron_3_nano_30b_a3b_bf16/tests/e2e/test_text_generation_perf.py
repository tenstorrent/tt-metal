# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Performance test for the 'text_generation' pipeline of NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.

Builds and runs the SHARED chained TTNN pipeline EXACTLY as demo/demo_text_generation.py does,
but BOUNDED and profiler-safe for a tracy capture: a small fixed input length, a short decode
loop, and a model-agnostic profiler drain that wraps every ttnn FastOperation by TYPE.

Perf only — NO PCC / correctness assertions.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import pipeline as pl
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt._hf_compat import install_hf_compat

install_hf_compat()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# Small fixed input length for the perf pass. Under tracy EVERY device op is instrumented, so a
# max-seq forward stalls the host for minutes in synchronize_device. A representative dispatch-dense
# pass only needs a small seq; override the model's production/max shapes with this.
PERF_SEQ_LEN = int(os.environ.get("TT_PERF_SEQ_LEN", "128"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

# The pipeline opens its OWN mesh via pl.open_pipeline_mesh (exactly as the demo does), so the
# compose flag mirrors the demo default (compose graduated child stubs).
PERF_COMPOSE = int(os.environ.get("TT_PERF_COMPOSE", "1"))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_text_generation_perf(device_params, device):
    # 1) build the pipeline EXACTLY as demo/demo_text_generation.py does
    tok = AutoTokenizer.from_pretrained(pl.HF_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pl.HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    eos = int(getattr(model.config, "eos_token_id", 2))

    # CAP THE INPUT SIZE SMALL: a small fixed sequence (do NOT use max_position_embeddings / max_seq).
    vocab = int(getattr(model.config, "vocab_size", 32000))
    hi = max(2, min(vocab, 1000))
    input_ids = torch.randint(0, hi, (1, PERF_SEQ_LEN), dtype=torch.long)

    dev, is_mesh = pl.open_pipeline_mesh(l1_small_size=24576)

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
    out = None
    try:
        pipe = pl.build_pipeline(dev, model, compose=bool(PERF_COMPOSE))
        print(f"[perf] mesh={is_mesh} shard_active={pipe.shard_active}", flush=True)
        new_ids, _ = pipe.generate(input_ids, PERF_MAX_NEW_TOKENS, eos_token_id=eos)
        out = new_ids
        try:
            ttnn.ReadDeviceProfiler(dev)
        except Exception:
            pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)
        pl.close_pipeline_mesh(dev, is_mesh)
    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
    assert out is not None  # perf only — NO PCC