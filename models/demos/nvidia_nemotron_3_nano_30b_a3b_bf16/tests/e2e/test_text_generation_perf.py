# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Performance test for the `text_generation` pipeline of the NemotronH-3-Nano-30B-A3B TTNN model.

Builds and runs the SHARED chained TTNN pipeline (tt/pipeline.py) EXACTLY as
demo/demo_text_generation.py does, but BOUNDED and profiler-safe: the device
forward runs IN-PROCESS (never shelled out) so tracy captures every op, the
decode loop is capped, and the profiler is drained periodically so tracy's
12000-marker buffer never overflows. Perf only — no PCC / correctness asserts.
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
PERF_PROMPT = os.environ.get("TT_PERF_PROMPT", "The capital of France is")
PERF_MAX_SEQ = int(os.environ.get("TT_PERF_MAX_SEQ", "128"))
PERF_COMPOSE = int(os.environ.get("TT_PERF_COMPOSE", "1"))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_text_generation_perf(device_params, device):
    # ---- 1) load tokenizer + reference model and build a SMALL bounded prompt (lifted from demo) ----
    tok = AutoTokenizer.from_pretrained(pl.HF_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pl.HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    eos = int(getattr(model.config, "eos_token_id", 2))

    input_ids = tok(PERF_PROMPT, return_tensors="pt")["input_ids"]
    # cap the input SMALL — a perf profile only needs a representative dispatch-dense pass
    if input_ids.shape[-1] > PERF_MAX_SEQ:
        input_ids = input_ids[:, :PERF_MAX_SEQ]

    # open the pipeline mesh EXACTLY as the demo does (this is the device tracy/profiler drains)
    mesh_device, is_mesh = pl.open_pipeline_mesh(l1_small_size=24576)
    device = mesh_device

    # ---- 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
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

    # ---- 3) build the pipeline + run the BOUNDED forward IN-PROCESS ----
    _fw0 = time.monotonic()
    out = None
    try:
        pipe = pl.build_pipeline(device, model, compose=bool(PERF_COMPOSE))
        print(f"[perf] mesh={is_mesh} shard_active={pipe.shard_active}", flush=True)
        new_ids, _ = pipe.generate(input_ids, PERF_MAX_NEW_TOKENS, eos_token_id=eos)
        out = new_ids
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception:
            pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)
        pl.close_pipeline_mesh(mesh_device, is_mesh)
    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
    assert out is not None  # perf only — NO PCC