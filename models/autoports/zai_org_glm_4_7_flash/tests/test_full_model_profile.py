# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-profiler variant of the full model: one real layer of each kind.

The all-layer stack is ~3200 device ops per decode step; collecting Tracy over
it produces multi-GB dumps and hits Tracy buffer limits, so profiling uses this
reduced build instead: HF layers 0 (dense) and 1 (moe), the real embedding,
final norm, LM head and on-device sampler, real paged-cache/page-table shapes,
real dtypes, and the same captured model + sampling traces. Everything the
full-model stage adds around the decoder stack appears here exactly once.

    python -m tracy -r -p -v -m pytest \\
        models/autoports/zai_org_glm_4_7_flash/tests/test_full_model_profile.py -q -s

Signpost windows:
    PERF_FM_DECODE_MODEL / _END        traced model decode (embed -> 2 layers -> norm -> LM head)
    PERF_FM_DECODE_TOKENOUT / _END     model trace + sampling trace (token-out)
    PERF_FM_PREFILL / _END             warmed prefill, 128-token prompt
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from tracy import signpost

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import GLM47FlashGenerator
from models.autoports.zai_org_glm_4_7_flash.tt.model import GLM47FlashModel

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC_DIR = MODEL_DIR / "doc" / "full_model"
PROBE_LAYERS = [0, 1]
PROBE_SEQ_LEN = 8192
PREFILL_S = 128
#: Small on purpose. The device profiler holds a bounded DRAM marker buffer, and
#: a 2-layer decode step is already ~80 ops; at 32 uncollected iterations the run
#: logs 1100 "Profiler DRAM buffers were full, markers were dropped!" lines and
#: the CSV ends up with roughly half the ops. Eight iterations with an explicit
#: ttnn.ReadDeviceProfiler after each one captures a complete op stream.
DECODE_ITERS = 8
TRACE_REGION_SIZE = 350_000_000


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=TRACE_REGION_SIZE
    )
    yield dev
    ttnn.close_mesh_device(dev)


@pytest.fixture(scope="module")
def reduced(device):
    model = GLM47FlashModel.from_pretrained(
        device, max_batch_size=1, max_seq_len=PROBE_SEQ_LEN, layer_indices=PROBE_LAYERS, progress=print
    )
    gen = GLM47FlashGenerator(model)
    gen._ensure_owned_state()
    gen.capture_decode_trace()
    gen.reset()
    yield gen
    gen.teardown()


def _write(name, payload):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / name).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {DOC_DIR / name}: {payload}")


def test_profile_decode(reduced):
    gen = reduced
    dev = gen.mesh_device
    ids = list(range(1000, 1000 + PREFILL_S))
    gen.reset()
    gen._prefill_and_sample_first(ids)
    gen.set_decode_positions([PREFILL_S])
    for _ in range(3):
        gen.decode_step_traced()
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)

    # Wall clock first, uninstrumented, so the profiler flushes below cannot
    # distort it. The signposted windows exist for device-op attribution only.
    t0 = time.perf_counter()
    for _ in range(DECODE_ITERS):
        ttnn.execute_trace(dev, gen._decode_trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    model_only = (time.perf_counter() - t0) / DECODE_ITERS
    t0 = time.perf_counter()
    for _ in range(DECODE_ITERS):
        gen.decode_step_traced()
    ttnn.synchronize_device(dev)
    token_out = (time.perf_counter() - t0) / DECODE_ITERS
    ttnn.ReadDeviceProfiler(dev)

    signpost("PERF_FM_DECODE_MODEL")
    for _ in range(DECODE_ITERS):
        ttnn.execute_trace(dev, gen._decode_trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)  # drain before the DRAM marker buffer fills
    signpost("PERF_FM_DECODE_MODEL_END")
    ttnn.ReadDeviceProfiler(dev)

    signpost("PERF_FM_DECODE_TOKENOUT")
    for _ in range(DECODE_ITERS):
        gen.decode_step_traced()
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
    signpost("PERF_FM_DECODE_TOKENOUT_END")
    ttnn.ReadDeviceProfiler(dev)

    _write(
        "perf_reduced_decode.json",
        {
            "variant": "reduced full model: HF layers 0 (dense) + 1 (moe), real embedding/norm/LM head/sampler",
            "layers": PROBE_LAYERS,
            "context_allocated": PROBE_SEQ_LEN,
            "decode_position": PREFILL_S,
            "iterations": DECODE_ITERS,
            "traced_model_only_ms": round(model_only * 1000, 3),
            "traced_token_out_ms": round(token_out * 1000, 3),
            "sampling_ms": round((token_out - model_only) * 1000, 3),
            "wall_clock_note": "measured before the signposted windows, with no profiler flushes in the loop",
            "signposts": [
                "PERF_FM_DECODE_MODEL",
                "PERF_FM_DECODE_MODEL_END",
                "PERF_FM_DECODE_TOKENOUT",
                "PERF_FM_DECODE_TOKENOUT_END",
            ],
        },
    )


def test_profile_prefill(reduced):
    gen = reduced
    dev = gen.mesh_device
    ids = list(range(2000, 2000 + PREFILL_S))
    gen.reset()
    gen._prefill_and_sample_first(ids)  # warm
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)

    signpost("PERF_FM_PREFILL")
    t0 = time.perf_counter()
    logits, _ = gen.model.prefill_forward_last_logits_device(
        ids, kv_cache=gen._kv_cache, page_table=gen._page_table_dev, seq_len=PREFILL_S
    )
    ttnn.synchronize_device(dev)
    wall = time.perf_counter() - t0
    signpost("PERF_FM_PREFILL_END")
    ttnn.ReadDeviceProfiler(dev)
    ttnn.deallocate(logits)

    _write(
        "perf_reduced_prefill.json",
        {
            "variant": "reduced full model: HF layers 0 (dense) + 1 (moe), real embedding/norm/LM head",
            "layers": PROBE_LAYERS,
            "prompt_len": PREFILL_S,
            "physical_prefill_len": gen.model.prefill_physical_len(PREFILL_S),
            "wall_ms": round(wall * 1000, 2),
            "signposts": ["PERF_FM_PREFILL", "PERF_FM_PREFILL_END"],
        },
    )
