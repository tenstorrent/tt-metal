# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model performance and capacity evidence (wall clock, all 47 layers).

Deliberately *not* run under Tracy: a 47-layer MoE decode step is ~3200 device
ops and the profiler dump is multi-GB. Device-level profiling uses the reduced
one-layer-of-each-kind variant in ``test_full_model_profile.py``; this module
measures what the host actually sees.

    pytest models/autoports/zai_org_glm_4_7_flash/tests/test_full_model_perf.py -q -s

Writes ``doc/full_model/perf.json`` and ``doc/full_model/capacity.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator
from models.autoports.zai_org_glm_4_7_flash.tt.model import source_manifest

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC_DIR = MODEL_DIR / "doc" / "full_model"
TRACE_REGION_SIZE = 350_000_000

#: Primary single-user profile, matching the vLLM benchmark shape.
PROMPT_LEN = 128
GEN_LEN = 128
DECODE_ITERS = 64

#: Optimized-decoder stage per-layer traced decode at ctx 1024, batch 1 (wall).
LAYER_LOWER_BOUND_MS = {"moe": 0.491, "dense": 0.447}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=TRACE_REGION_SIZE
    )
    yield dev
    ttnn.close_mesh_device(dev)


@pytest.fixture(scope="module")
def built(device):
    timings = {}
    t0 = time.perf_counter()
    gen = build_generator(MODEL_DIR, device, progress=lambda m: None)
    timings["build_total_s"] = round(time.perf_counter() - t0, 2)
    yield gen, timings
    gen.teardown()


def _prompt_ids(gen, seq):
    text = (
        "Tenstorrent builds AI accelerators. "
        "This paragraph exists so the tokenizer produces a long, ordinary, in-distribution prompt "
        "for the full-model performance measurement. "
    ) * 200
    ids = gen.tokenizer.encode(text, add_special_tokens=True)
    while len(ids) < seq:
        ids = ids + ids
    return ids[:seq]


def test_capacity_json(built):
    gen, _ = built
    model = gen.model
    weights = model.weight_bytes()
    cache = model.kv_cache_bytes()
    sampler = 0
    penalties = getattr(gen.sampling, "tt_penalties", None) if gen.sampling is not None else None
    if penalties is not None:
        seen = set()
        for name in ("prompt_mask", "output_mask", "output_counts", "output_counts_gathered", "zeros"):
            t = getattr(penalties, name, None)
            if t is None or id(t) in seen:
                continue
            seen.add(id(t))
            n = 1
            for d in tuple(t.padded_shape if hasattr(t, "padded_shape") else t.shape):
                n *= int(d)
            sampler += n * 4  # int32
    # Prefer the measured value from probe/dram_capacity_probe.py over the
    # constant, so the recorded figure has provenance rather than looking like
    # something this test measured.
    allocatable = int(31.5 * 2**30)
    allocatable_note = (
        "fallback constant; run probe/dram_capacity_probe.py to record a measured value in "
        "doc/full_model/dram_capacity.json"
    )
    probe = DOC_DIR / "dram_capacity.json"
    if probe.is_file():
        measured = json.loads(probe.read_text())
        allocatable = int(measured["allocatable_bytes"])
        allocatable_note = (
            f"measured by probe/dram_capacity_probe.py ({probe.name}): {measured['allocatable_mib']} MiB "
            f"in {measured['chunk_mib']} MiB chunks before the allocator refused"
        )
    payload = {
        "source_manifest": source_manifest([__file__]),
        "device": "Blackhole p150-class chip, 1x1 mesh, 11x10 compute grid, 8 DRAM banks",
        "measured_allocatable_dram_bytes": allocatable,
        "measured_allocatable_dram_note": allocatable_note,
        "weights_bytes": {k: int(v) for k, v in weights.items()},
        "kv_cache_bytes_batch1_full_context": int(cache),
        "kv_cache_context": model.max_seq_len,
        "kv_cache_dtype": str(model.cache_dtype),
        "kv_cache_bytes_per_token_per_layer": int(cache / model.max_seq_len / len(model.layers)),
        "sampler_penalty_buffers_bytes": int(sampler),
        "sampler_penalty_buffers_note": (
            "SamplingGenerator constructs TTPenalties unconditionally; these int32 [32, vocab] "
            "buffers are resident even in a greedy-only run"
        ),
        "total_resident_bytes": int(weights["total"] + cache + sampler),
        "trace_region_bytes": TRACE_REGION_SIZE,
        "headroom_bytes": int(allocatable - weights["total"] - cache - sampler - TRACE_REGION_SIZE),
        "gib": {
            "weights_total": round(weights["total"] / 2**30, 3),
            "kv_cache": round(cache / 2**30, 3),
            "sampler_penalty_buffers": round(sampler / 2**30, 3),
            "total_resident": round((weights["total"] + cache + sampler) / 2**30, 3),
        },
    }
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "capacity.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["gib"], indent=2))
    assert payload["headroom_bytes"] > 0


def test_full_model_perf(built):
    gen, timings = built
    model = gen.model
    ids = _prompt_ids(gen, PROMPT_LEN)

    # ---- warmed TTFT + end-to-end generate ----
    gen.reset()
    gen.generate(ids, 4, enable_trace=True, stop_on_eos=False)  # ensure this shape is warm
    gen.reset()
    gen.reset_counters()
    t0 = time.perf_counter()
    preds, timing = gen.generate(ids, GEN_LEN, enable_trace=True, stop_on_eos=False, return_timing=True)
    e2e_s = time.perf_counter() - t0
    counters = dict(gen.counters)
    assert len(preds) == GEN_LEN

    # ---- isolated windows over the same captured traces ----
    def bench(fn, iters=DECODE_ITERS):
        fn()
        ttnn.synchronize_device(model.mesh_device)
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        ttnn.synchronize_device(model.mesh_device)
        return (time.perf_counter() - t) / iters

    model_only_s = bench(gen.replay_decode_trace)
    with_sampling_s = bench(gen.decode_step_traced)
    token_out_s = bench(lambda: (gen.decode_step_traced(), gen.read_decode_tokens(1)))

    # ---- cold TTFT at an unseen prefill bucket (program compilation) ----
    cold_ids = _prompt_ids(gen, 3000)  # one 2048 chunk + a 1024 bucket tail, not warmed at build
    gen.reset()
    t0 = time.perf_counter()
    gen._prefill_and_sample_first(cold_ids)
    cold_ttft_s = time.perf_counter() - t0
    gen.reset()
    t0 = time.perf_counter()
    gen._prefill_and_sample_first(cold_ids)
    warm_ttft_3000_s = time.perf_counter() - t0

    n_moe = sum(1 for layer in model.layers if layer.layer_kind == "moe")
    n_dense = len(model.layers) - n_moe
    lower_bound_ms = n_moe * LAYER_LOWER_BOUND_MS["moe"] + n_dense * LAYER_LOWER_BOUND_MS["dense"]

    payload = {
        "source_manifest": source_manifest([__file__]),
        "workload": {
            "prompt_len": PROMPT_LEN,
            "generate_len": GEN_LEN,
            "batch": model.max_batch_size,
            "context_allocated": model.max_seq_len,
            "sampling": "greedy split sampling (k=1, p=0, temp=1) on device",
        },
        "setup_s": timings,
        "ttft_ms": {
            "prompt_128_warmed": round(timing["ttft_s"] * 1000, 1),
            "prompt_128_physical": model.prefill_physical_len(PROMPT_LEN),
            "prompt_3000_first_call": round(cold_ttft_s * 1000, 1),
            "prompt_3000_second_call": round(warm_ttft_3000_s * 1000, 1),
            "prompt_3000_physical": model.prefill_physical_len(3000),
        },
        "prefill_tokens_per_s": {
            "prompt_128": round(PROMPT_LEN / timing["ttft_s"], 1),
            "prompt_3000": round(3000 / warm_ttft_3000_s, 1),
        },
        "decode_ms_per_token": {
            "traced_model_only_no_sampling": round(model_only_s * 1000, 3),
            "traced_model_plus_sampling": round(with_sampling_s * 1000, 3),
            "token_out_incl_readback": round(token_out_s * 1000, 3),
            "generate_loop_measured": round(timing["decode_s"] / max(timing["decode_tokens"], 1) * 1000, 3),
        },
        "decode_tokens_per_s_per_user": {
            "traced_model_only_no_sampling": round(1 / model_only_s, 2),
            "token_out_incl_readback": round(1 / token_out_s, 2),
            "generate_loop_measured": round(timing["decode_tokens"] / timing["decode_s"], 2),
        },
        "end_to_end": {
            "prompt_128_generate_128_s": round(e2e_s, 3),
            "tokens_per_s": round(GEN_LEN / e2e_s, 2),
        },
        "layer_stack_lower_bound": {
            "source": "doc/optimized_decoder/README.md traced per-layer decode, ctx 1024, batch 1, wall",
            "moe_layers": n_moe,
            "dense_layers": n_dense,
            "ms_per_token": round(lower_bound_ms, 3),
            "full_model_only_costs_ms": round(model_only_s * 1000 - lower_bound_ms, 3),
            "sampling_ms": round((with_sampling_s - model_only_s) * 1000, 3),
            "token_readback_ms": round((token_out_s - with_sampling_s) * 1000, 3),
        },
        "host_work_counters_over_the_measured_generate": counters,
    }
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "perf.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))

    # Host refreshes must be per REQUEST, not per token: reset + the prefill
    # handoff account for all of them, and they do not scale with GEN_LEN
    # (test_split_sampling_trace_feedback asserts exactly 0 inside the loop).
    assert counters["token_input_refreshes"] <= 2, counters
    assert counters["position_refreshes"] <= 2, counters
    assert counters["page_table_refreshes"] == 0
    assert counters["full_logits_readbacks"] == 0
    assert counters["model_trace_replays"] == GEN_LEN - 1
    assert counters["sampling_trace_replays"] == GEN_LEN - 1
    # sampling must not dominate token-out decode
    sampling_fraction = (with_sampling_s - model_only_s) / token_out_s
    print(f"sampling is {sampling_fraction * 100:.1f}% of token-out decode")
    assert sampling_fraction < 0.25, sampling_fraction
