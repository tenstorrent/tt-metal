# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure the vLLM *adapter's* own traced decode cost, with no vLLM engine.

Answers one question the vLLM-integration stage left open: how much of the
45.0 ms/token served decode is vLLM engine/scheduler overhead, and how much is
the model doing genuinely different work than the batch-1 full-model harness
(which builds with ``max_batch_size=1`` and therefore takes the *indexed*
compact-top-k MoE decode path, while the adapter always builds 32 physical
rows and takes the *union* path).

Drives the real adapter (``GLM47FlashForCausalLM``) exactly the way
``vllm_tt_plugin/async_decode.py`` drives it:
``decode_forward(read_from_device=False)`` -> ``read_decode_output(async_read=True)``
-> ``ttnn.event_synchronize`` -> ``process_decode_output_host``.

    python models/autoports/zai_org_glm_4_7_flash/doc/optimized_vllm/probe_scripts/adapter_decode_floor.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

import ttnn

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))

from models.autoports.zai_org_glm_4_7_flash.tt import generator as _generator_module  # noqa: E402
from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator  # noqa: E402
from models.autoports.zai_org_glm_4_7_flash.tt.generator_vllm import (  # noqa: E402
    VLLM_PREFILL_BUCKETS,
    VLLM_PREFILL_CHUNK_SIZE,
    GLM47FlashForCausalLM,
)
from models.common.sampling import SamplingParams  # noqa: E402

MODEL_DIR = REPO / "models" / "autoports" / "zai_org_glm_4_7_flash"
OUT = (
    MODEL_DIR
    / "doc"
    / "optimized_vllm"
    / ("adapter_decode_floor_%s.json" % (sys.argv[1] if len(sys.argv) > 1 else "after"))
)
_ARM = sys.argv[1] if len(sys.argv) > 1 else "after"
COMPACT = _ARM != "before"
if _ARM == "kc64":
    # The rejected candidate, kept runnable: capture a full-width compact
    # bucket so row counts whose bound needs kc == n_experts take a COMPACT
    # trace instead of falling back to the union trace. The shipped
    # ``_kc_buckets`` filters ``kc == n_experts`` out on purpose, so this arm
    # replaces it rather than just widening COMPACT_KC_BUCKETS. This is the arm
    # the stage report cites when it says kc = n_experts is measured slower.
    def _kc_buckets_with_full_width(model):
        moe = next(l for l in model.layers if getattr(l, "layer_kind", None) == "moe")
        cap = min(moe.n_experts, _generator_module._next_pow2(model.max_batch_size * moe.top_k))
        return tuple(b for b in (4, 16, 32, 64) if b <= cap)

    _generator_module._kc_buckets = _kc_buckets_with_full_width
    _generator_module.COMPACT_KC_MIN_ROWS = {4: 1, 16: 2, 32: 7, 64: 1}
if _ARM == "kcexact":
    # The limit case: one bucket per reachable row count up to the point where
    # the bound reaches n_experts, i.e. kc = live_rows * top_k exactly. Eight
    # compact traces plus the union trace. Measures whether finer buckets keep
    # paying, and what they cost in trace region and capture time.
    _generator_module.COMPACT_KC_BUCKETS = (4, 8, 12, 16, 20, 24, 28, 32)
    _generator_module.COMPACT_KC_MIN_ROWS = {b: b // 4 for b in (4, 8, 12, 16, 20, 24, 28, 32)}

TRACE_REGION_SIZE = 350_000_000
L1_SMALL_SIZE = 32768
MAX_SEQ_LEN = 202752
BLOCK_SIZE = 64
MAX_BATCH_SIZE = 32
#: Exactly the pool vLLM chose in the vllm-integration run (server.log:
#: get_max_tokens_all_users -> 469104 tokens / block_size 64, capped by vLLM).
NUM_BLOCKS = 7362
BLOCKS_PER_USER = math.ceil(MAX_SEQ_LEN / BLOCK_SIZE)

ITERS = 30
WARM = 6


def _greedy_params(rows: int):
    return SamplingParams(
        temperature=[1.0] * rows,
        top_k=[1] * rows,
        top_p=[0.0] * rows,
    )


def build(device):
    generator = build_generator(
        MODEL_DIR,
        device,
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        defer_cache_and_traces=True,
        enable_sampling=True,
        host_sampling=False,
        moe_decode_compact=COMPACT,
        prefill_chunk_size=VLLM_PREFILL_CHUNK_SIZE,
        prefill_buckets=VLLM_PREFILL_BUCKETS,
        progress=lambda m: print(m, flush=True),
    )
    model = GLM47FlashForCausalLM(generator)
    kv_cache = model.allocate_kv_cache(
        kv_cache_shape=(NUM_BLOCKS, 1, BLOCK_SIZE, generator.model.layers[0].kvpe_dim),
        dtype=torch.bfloat16,
        num_layers=len(generator.model.layers),
    )
    model.warmup_model_decode(
        kv_cache=kv_cache,
        max_batch_size=MAX_BATCH_SIZE,
        num_blocks=NUM_BLOCKS,
        can_sample_on_device=True,
        enable_trace=True,
    )
    return model, kv_cache


def time_decode(model, kv_cache, active_rows: int, *, label: str):
    """Drive the adapter exactly like the plugin's async decode controller."""
    gen = model.generator
    # Page table: give each active row its own distinct blocks out of the pool.
    pt = torch.zeros((active_rows, BLOCKS_PER_USER), dtype=torch.int32)
    for r in range(active_rows):
        pt[r] = torch.arange(r * 16, r * 16 + BLOCKS_PER_USER, dtype=torch.int32) % NUM_BLOCKS
    # Distinct tokens and positions per row on purpose: identical rows all
    # route to the same 4 experts, which makes the union path look far cheaper
    # at batch than any real serving batch would (the 100/100/32 CI burst sends
    # 32 different random prompts). Row r gets a different vocabulary id and a
    # different decode position.
    toks = torch.tensor([(1000 + 977 * r) % 150000 for r in range(active_rows)], dtype=torch.int64)
    pos = torch.tensor([128 + 13 * r for r in range(active_rows)], dtype=torch.int64)

    params = _greedy_params(active_rows)
    # One reset_batch=True step to load host state, then steady state.
    model.decode_forward(
        tokens=toks,
        start_pos=pos,
        page_table=pt,
        kv_cache=kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=params,
        reset_batch=True,
    )

    def one_step():
        out = model.decode_forward(
            tokens=toks,
            start_pos=pos,
            page_table=pt,
            kv_cache=kv_cache,
            enable_trace=True,
            read_from_device=False,
            sampling_params=params,
            reset_batch=False,
        )
        host, events = model.read_decode_output(out, async_read=True)
        for e in events:
            ttnn.event_synchronize(e)
        return model.process_decode_output_host(host, is_tokens=True)

    for _ in range(WARM):
        one_step()
    ttnn.synchronize_device(model.mesh_device)

    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        one_step()
        samples.append((time.perf_counter() - t0) * 1e3)

    # Model-trace-only replay (no sampler, no readback) for the same rows.
    ttnn.synchronize_device(model.mesh_device)
    model_only = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        gen.replay_decode_trace()
        ttnn.synchronize_device(model.mesh_device)
        model_only.append((time.perf_counter() - t0) * 1e3)

    res = {
        "label": label,
        "active_rows": active_rows,
        "moe_kc_used": gen._active_kc,
        "decode_kc_buckets": list(gen._decode_kc_buckets),
        "physical_rows": MAX_BATCH_SIZE,
        "iters": ITERS,
        "adapter_async_token_out_ms": round(statistics.median(samples), 3),
        "adapter_async_token_out_ms_mean": round(statistics.fmean(samples), 3),
        "adapter_async_token_out_ms_min": round(min(samples), 3),
        "model_trace_only_ms": round(statistics.median(model_only), 3),
        "tps_per_user": round(1000.0 / statistics.median(samples), 3),
    }
    print(json.dumps(res, indent=2), flush=True)
    return res


def main():
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=L1_SMALL_SIZE, trace_region_size=TRACE_REGION_SIZE
    )
    try:
        t0 = time.perf_counter()
        model, kv_cache = build(device)
        build_s = time.perf_counter() - t0
        results = []
        # Every live-row count where the bucket choice changes, plus the ones
        # *inside* each bucket's range. Sweeping only the saturated points
        # (1/4/8/32) measures each bucket's best case and would miss a bucket
        # that is a net loss over part of the range it serves -- which is
        # exactly what rows 5-7 turned out to be.
        for rows in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32):
            results.append(time_decode(model, kv_cache, rows, label=f"active_rows={rows}"))
            model.generator.reset()
        trace_mem = {}
        try:
            view = ttnn.get_memory_view(device, ttnn.BufferType.TRACE)
            trace_mem["raw"] = str(view)
            for attr in (
                "total_bytes_allocated_per_bank",
                "total_bytes_per_bank",
                "total_bytes_free_per_bank",
                "largest_contiguous_bytes_free_per_bank",
                "num_banks",
            ):
                if hasattr(view, attr):
                    trace_mem[attr] = getattr(view, attr)
        except Exception as exc:  # noqa: BLE001
            trace_mem["error"] = str(exc)[:300]
        print("TRACE REGION:", json.dumps({k: v for k, v in trace_mem.items() if k != "raw"}), flush=True)
        print("TRACE REGION RAW:", trace_mem.get("raw", "")[:1500], flush=True)

        payload = {
            "trace_region": trace_mem,
            "trace_region_reserved_bytes": TRACE_REGION_SIZE,
            "decode_traces_captured": len(model.generator._decode_traces),
            "purpose": (
                "Adapter-only traced decode cost with no vLLM engine, driven exactly like "
                "vllm_tt_plugin/async_decode.py drives it. Isolates vLLM engine overhead from "
                "the model's own 32-physical-row decode cost."
            ),
            "config": {
                "max_batch_size": MAX_BATCH_SIZE,
                "max_seq_len": MAX_SEQ_LEN,
                "block_size": BLOCK_SIZE,
                "num_blocks": NUM_BLOCKS,
                "blocks_per_user": BLOCKS_PER_USER,
                "prefill_chunk_size": VLLM_PREFILL_CHUNK_SIZE,
                "sampling": "greedy split sampling (k=1, p=0, temp=1) on device",
            },
            "build_s": round(build_s, 1),
            "results": results,
            "counters": dict(model.generator.counters),
            "kc_replays": {str(k): v for k, v in model.generator.kc_replays.items()},
        }
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"WROTE {OUT}", flush=True)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
