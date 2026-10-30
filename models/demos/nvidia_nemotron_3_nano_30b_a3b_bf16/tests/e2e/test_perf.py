"""Bounded device-time perf workload for the Nemotron-3-Nano e2e pipeline.

Drives a prefill forward through the composed graduated-stub pipeline
(tt/pipeline.py) so the perf_automation tool can measure device_ms and the
tracy profiler can attribute per-op time.

Two profiler-overflow defenses (the seamless lesson: one full forward overruns
the 12000-marker/RISC device profiler budget, tracy then drops markers and
post-processing asserts on a missing op):

  1. Layer cap. The NemotronH layer pattern is MEMEM*... (M=mamba, E=MoE,
     *=attention); TT_PERF_LAYERS=6 (MEMEM*) keeps the full op mix and invokes
     every graduated child while running ~12% of the layers.

  2. Periodic drain. A single 128-expert MoE layer alone exceeds the budget on
     its busy cores (measured: block layer ~112 ops, +1 MoE layer ~+1368 ops),
     so layer-capping is not enough. We wrap ttnn.linear/ttnn.matmul to call
     ttnn.ReadDeviceProfiler every TT_PERF_FLUSH_EVERY ops, draining the
     on-device buffer mid-forward so no core ever overruns. This is exactly the
     device's own guidance ("run read device profiler more often").

pipeline.py is untouched: this only invokes its public forward_logits, caps the
public _N_LAYERS attribute, and temporarily wraps module-level ttnn entry points
for the duration of the call.
"""

from __future__ import annotations

import os

import pytest

import models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt.pipeline as pl
import ttnn
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tests.e2e.test_e2e_pipeline import (
    _compose,
    _load_golden,
    _load_hf,
    _reset_runtime_fallbacks,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_prefill_perf(device_params, device):
    compose = _compose()
    _reset_runtime_fallbacks()
    ids, _, _ = _load_golden()
    hf = _load_hf()
    pipe = pl.build_pipeline(device, hf, compose=compose)

    perf_layers = int(os.environ.get("TT_PERF_LAYERS", "6"))
    pipe.M._N_LAYERS = min(perf_layers, pipe.M._N_LAYERS)

    flush_every = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
    counter = [0]
    orig_linear, orig_matmul = ttnn.linear, ttnn.matmul

    def _draining(fn):
        def inner(*args, **kwargs):
            result = fn(*args, **kwargs)
            counter[0] += 1
            if flush_every and counter[0] % flush_every == 0:
                ttnn.ReadDeviceProfiler(device)
            return result

        return inner

    ttnn.linear = _draining(orig_linear)
    ttnn.matmul = _draining(orig_matmul)
    try:
        pipe.forward_logits(ids)
        ttnn.ReadDeviceProfiler(device)
    finally:
        ttnn.linear = orig_linear
        ttnn.matmul = orig_matmul
