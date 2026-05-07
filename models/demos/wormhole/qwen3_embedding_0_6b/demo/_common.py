# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for the standalone Qwen3-Embedding-0.6B perf entry points
(`demo_bs{1,8,32}_isl512.py` and `tests/perf/new_perf_bs{1,8,32}_isl512.py`).

Each entry-point file imports `run_perf_pytest` and parametrizes one workload.
This module bakes in every optimization env var validated for these shapes so
each entry point is a true "all-optimizations-on" run with no extra flags.

Recommended config, set before model build (see demo.py for the full impact
matrix). Each var is `setdefault`'d so the user can still override from the
shell:

    QWEN_QKV_BFP4=1
    QWEN_WO_BFP4=1
    QWEN_FF13_OUT_BFP8=1
    QWEN_FFNORM_IN_BFP8=1
    QWEN_RESIDUAL_BFP8=1      # post-FFN add -> BFP8 (closes BFP8 chain into QKV)
    QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1
                              # split nlp_create_qkv_heads by (seq tile, KV-head group);
                              # bs=1 ISL=512 goes 16 -> 128 work units / cores.
    QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1
                              # split nlp_concat_heads by (seq tile, head group);
                              # bs=1 ISL=512 goes 16 -> 128 work units / cores.
    QWEN_ROPE_PREFILL_L1=1
    QWEN_LN_BLOCK_SHARDED=1
    TT_SKIP_KV_CACHE_FILL=1
    TT_BATCHED_L1_PREFILL=1   # only for bs>1
"""

import math
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
    determine_device_name,
)

try:
    from tracy import signpost as _tracy_signpost
except ModuleNotFoundError:

    def _tracy_signpost(*_args, **_kwargs):  # noqa: D401 - matches tracy.signpost signature
        pass


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BLOCK_SIZE = 32


# ---------------------------------------------------------------------------
# Environment / optimizations
# ---------------------------------------------------------------------------


def apply_recommended_env(batched_l1: bool) -> None:
    """Set the recommended optimization env vars unless the user overrode them.

    `setdefault` semantics keep this idempotent and respect explicit shell
    overrides — useful for A/B comparisons of individual knobs.
    """
    os.environ.setdefault("HF_MODEL", MODEL_NAME)
    os.environ.setdefault("QWEN_QKV_BFP4", "1")
    os.environ.setdefault("QWEN_WO_BFP4", "1")
    os.environ.setdefault("QWEN_FF13_OUT_BFP8", "1")
    os.environ.setdefault("QWEN_FFNORM_IN_BFP8", "1")
    os.environ.setdefault("QWEN_RESIDUAL_BFP8", "1")
    os.environ.setdefault("QWEN_NLP_CREATE_HEADS_HEAD_SPLIT", "1")
    os.environ.setdefault("QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT", "1")
    os.environ.setdefault("QWEN_ROPE_PREFILL_L1", "1")
    os.environ.setdefault("QWEN_LN_BLOCK_SHARDED", "1")
    os.environ.setdefault("TT_SKIP_KV_CACHE_FILL", "1")
    if batched_l1:
        os.environ.setdefault("TT_BATCHED_L1_PREFILL", "1")


def qwen_embedding_optimizations(model_args):
    """`DecodersPrecision.performance` + opt-in BFP4 promotions for FF2/QKV/WO.

    Mirrors the helper in `demo.py` (kept in sync); see that file for the full
    cosine-similarity vs. wall-time impact matrix per knob.
    """
    base = DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

    promote_ff2 = os.getenv("QWEN_FF2_BFP4", "0") == "1"
    promote_qkv = os.getenv("QWEN_QKV_BFP4", "0") == "1"
    promote_wo = os.getenv("QWEN_WO_BFP4", "0") == "1"

    if not (promote_ff2 or promote_qkv or promote_wo):
        return base

    seen = set()
    for decoder_id in range(model_args.n_layers):
        opt = base.decoder_optimizations[decoder_id]
        if id(opt) in seen:
            continue
        seen.add(id(opt))
        tp = opt._opt_settings["TensorPrecision"]
        of = opt._opt_settings["OpFidelity"]
        if promote_ff2:
            tp[TensorGroup.FF2] = PrecisionSetting.BFP4
            of[OpGroup.LI_FF2] = MathFidelitySetting.LOFI
        if promote_qkv:
            tp[TensorGroup.WQKV] = PrecisionSetting.BFP4
            of[OpGroup.LI_QKV_PREFILL] = MathFidelitySetting.LOFI
            of[OpGroup.LI_QKV_DECODE] = MathFidelitySetting.LOFI
        if promote_wo:
            tp[TensorGroup.WO] = PrecisionSetting.BFP4
            of[OpGroup.LI_O_PREFILL] = MathFidelitySetting.LOFI
            of[OpGroup.LI_O_DECODE] = MathFidelitySetting.LOFI
    base._update_full_name()
    return base


# ---------------------------------------------------------------------------
# Model build / inputs
# ---------------------------------------------------------------------------


def _page_params_for(batch_size: int, seq_len: int) -> dict:
    """Page table sizing identical to the matching dp1-batch{1,8,32}-isl512
    entries in `demo.py` so the trace shapes are bit-identical."""
    if batch_size == 1:
        return {"page_block_size": BLOCK_SIZE, "page_max_num_blocks": 512}
    if batch_size == 8:
        return {"page_block_size": BLOCK_SIZE, "page_max_num_blocks": 1024}
    # bs >= 32: pad to a value that comfortably covers any batch*seq tile count
    page_max = max(512, math.ceil(seq_len / BLOCK_SIZE) * batch_size * 2)
    return {"page_block_size": BLOCK_SIZE, "page_max_num_blocks": page_max}


def build_single_device_model(mesh_device, batch_size: int, seq_len: int):
    """Build one model instance + Generator + page table on a single mesh.

    No data-parallelism, no submeshes — these standalone harnesses are
    explicitly single-device. For the multi-device DP=8/32 sweeps, use the
    parametrized `demo.py` instead.
    """
    page_params = _page_params_for(batch_size, seq_len)
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )

    model_args, model, kv_cache, _state_dict = create_tt_model(
        mesh_device,
        instruct=False,
        max_batch_size=batch_size,
        optimizations=qwen_embedding_optimizations,
        max_seq_len=seq_len,
        paged_attention_config=paged_attention_config,
        dtype=ttnn.bfloat8_b,
        state_dict=None,
    )
    # Generator expects kv_cache indexed first by DP instance, then by layer.
    # Single-device path -> exactly one DP instance.
    kv_caches = [[layer.attention.layer_past for layer in model.layers]]
    generator = Generator(
        [model],
        [model_args],
        mesh_device,
        tokenizer=model_args.tokenizer,
    )

    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(batch_size, paged_attention_config.max_num_blocks // batch_size)

    return generator, model_args, kv_caches, page_table


def generate_synthetic_inputs(tokenizer, batch_size: int, seq_len: int):
    """Random tokens of exactly `seq_len`. Matches `dp1-batch*-isl*` paths."""
    vocab_size = tokenizer.vocab_size
    high = min(vocab_size, 50000)
    input_ids = torch.randint(100, high, (batch_size, seq_len), dtype=torch.long)
    prompt_lens = [seq_len] * batch_size
    return input_ids, prompt_lens


def _run_prefill(generator, input_ids, page_table, kv_caches, prompt_lens, *, warmup: bool):
    """One forward through Generator.prefill_forward_text.

    `warmup=True` triggers the multi-seq warmup loop inside the generator (the
    one that compiles non-batched prefill traces for 128/256/512 seq lens, used
    by chunk-prefill paths). MUST only be set on the FIRST call — re-warming on
    every iteration leaves stale L1 allocations from the bs=1 traces that then
    collide with the bs>1 batched-prefill program's static circular buffers
    (observed as `TT_THROW: Statically allocated circular buffers ... clash
    with L1 buffers`). demo.py uses the same pattern.
    """
    return generator.prefill_forward_text(
        input_ids,
        page_table=page_table,
        kv_cache=kv_caches,
        prompt_lens=prompt_lens,
        enable_trace=True,
        return_hidden_states=True,
        warmup_prefill=warmup,
    )


# ---------------------------------------------------------------------------
# Top-level runner used by both demo and tracy entry points
# ---------------------------------------------------------------------------


def run_perf(
    mesh_device,
    batch_size: int,
    seq_len: int,
    num_iterations: int,
    *,
    emit_signposts: bool,
    is_ci_env: bool = False,
):
    """Build, compile, and benchmark Qwen3-Embedding-0.6B on a single mesh.

    Args:
        mesh_device: ttnn.MeshDevice (1x1 for the standalone path)
        batch_size: 1, 8, or 32 (must match `_page_params_for`)
        seq_len: input sequence length (typically 512)
        num_iterations: how many post-compile iterations to run. Set to 1 for
            tracy captures so the signposted zone covers exactly one
            trace-replay forward.
        emit_signposts: when True, the LAST iteration's
            `prefill_forward_text(...)` is wrapped in
            `tracy.signpost("start"/"stop")` markers. Use with `num_iterations=1`
            for clean device-time captures via the Tracy profiler.
        is_ci_env: forward to `create_benchmark_data` for CI dashboards.
    """
    profiler = BenchmarkProfiler()
    profiler.start("run")
    tt_device_name = determine_device_name(mesh_device)

    logger.info(f"Building Qwen3-Embedding-0.6B: bs={batch_size}, seq_len={seq_len}, device={tt_device_name}")
    profiler.start("build_model")
    generator, model_args, kv_caches, page_table = build_single_device_model(
        mesh_device, batch_size=batch_size, seq_len=seq_len
    )
    profiler.end("build_model")
    logger.info(f"Built in {profiler.get_duration('build_model'):.1f}s")

    input_ids, prompt_lens = generate_synthetic_inputs(model_args.tokenizer, batch_size, seq_len)
    total_input_tokens = sum(prompt_lens)

    logger.info("Compiling (first prefill captures hardware trace + runs warmup)...")
    profiler.start("compile_prefill")
    _ = _run_prefill(generator, input_ids, page_table, kv_caches, prompt_lens, warmup=True)
    profiler.end("compile_prefill")
    logger.info(f"Compile prefill: {profiler.get_duration('compile_prefill'):.2f}s")

    logger.info(f"Running {num_iterations} benchmark iteration(s)...")
    iteration_times = []
    embeddings = None
    last_iter_idx = num_iterations - 1
    for i in range(num_iterations):
        # paged_fill_cache overwrites unconditionally in attention.forward_prefill
        # and SDPA is causal, so no explicit KV-cache clear is needed for an
        # embedding workload. Skipping the clear saves ~31% of per-iter kernel
        # time on a 4 MB DRAM->DRAM write that gets immediately overwritten.
        generator.prev_page_table = None

        # Only signpost the LAST iteration: the first iteration sometimes shows
        # leftover compile-time L1 fragmentation effects, so for a 1-iter tracy
        # run use num_iterations=1 (last == first) and you still get a clean
        # zone. For multi-iter benchmarks, we don't signpost at all.
        sig = emit_signposts and i == last_iter_idx

        profiler.start(f"inference_prefill_{i}")
        if sig:
            _tracy_signpost("start")
        try:
            result = _run_prefill(generator, input_ids, page_table, kv_caches, prompt_lens, warmup=False)
            # Force device->host sync so the host-side stop signpost lands
            # AFTER the device finished, otherwise the captured zone clips.
            ttnn.synchronize_device(mesh_device)
        finally:
            if sig:
                _tracy_signpost("stop")
        profiler.end(f"inference_prefill_{i}")

        t = profiler.get_duration(f"inference_prefill_{i}")
        iteration_times.append(t)
        logger.info(f"  Iteration {i}: {t * 1000:.1f}ms")

        if embeddings is None:
            embeddings = result

    avg_t = sum(iteration_times) / len(iteration_times)
    best_t = min(iteration_times)
    measurements = {
        "compile_prefill": profiler.get_duration("compile_prefill"),
        "avg_prefill_time": avg_t,
        "best_prefill_time": best_t,
        "embeddings/s_avg": batch_size / avg_t,
        "embeddings/s_best": batch_size / best_t,
        "prefill_t/s_avg": total_input_tokens / avg_t,
        "prefill_t/s_best": total_input_tokens / best_t,
        "build_model_time": profiler.get_duration("build_model"),
        "batch_size": batch_size,
        "data_parallel": 1,
        "input_seq_len": seq_len,
        "max_seq_len": seq_len,
        "total_input_tokens": total_input_tokens,
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Qwen3-Embedding-0.6B Performance  ({tt_device_name})")
    logger.info("=" * 60)
    logger.info(f"  Batch size:           {batch_size}")
    logger.info(f"  Input seq length:     {seq_len}")
    logger.info(f"  Total input tokens:   {total_input_tokens}")
    logger.info(f"  Iterations:           {num_iterations}")
    logger.info("-" * 60)
    logger.info(f"  Model build time:     {measurements['build_model_time']:.1f}s")
    logger.info(f"  Compile (1st run):    {measurements['compile_prefill']:.2f}s")
    logger.info("-" * 60)
    logger.info(f"  Avg prefill time:     {avg_t * 1000:.1f}ms")
    logger.info(f"  Best prefill time:    {best_t * 1000:.1f}ms")
    logger.info(f"  Avg embeddings/s:     {measurements['embeddings/s_avg']:.1f}")
    logger.info(f"  Best embeddings/s:    {measurements['embeddings/s_best']:.1f}")
    logger.info(f"  Avg tokens/s:         {measurements['prefill_t/s_avg']:.0f}")
    logger.info(f"  Best tokens/s:        {measurements['prefill_t/s_best']:.0f}")
    logger.info("=" * 60)

    profiler.end("run")

    if is_ci_env:
        benchmark_data = create_benchmark_data(profiler, measurements, {}, {})
        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name=getattr(model_args, "base_model_name", "Qwen3-Embedding-0.6B"),
            ml_model_type="embedding",
            num_layers=model_args.n_layers,
            batch_size=batch_size,
            config_params={"data_parallel": 1, "tensor_parallel": 1},
            input_sequence_length=seq_len,
            output_sequence_length=0,
        )

    assert embeddings is not None
    return measurements


# ---------------------------------------------------------------------------
# Standalone (no-pytest) entry point
# ---------------------------------------------------------------------------


def standalone_main(batch_size: int, seq_len: int, iterations: int, device_id: int = 0) -> None:
    """`python <entry_file>` path — opens its own device, no pytest fixture."""
    apply_recommended_env(batched_l1=batch_size > 1)

    logger.info(f"Opening device {device_id}...")
    device = ttnn.open_device(
        device_id=device_id,
        l1_small_size=32768,
        trace_region_size=200_000_000,
        num_command_queues=1,
    )
    try:
        t0 = time.perf_counter()
        run_perf(
            device,
            batch_size=batch_size,
            seq_len=seq_len,
            num_iterations=iterations,
            emit_signposts=False,
        )
        logger.info(f"Total wall time: {time.perf_counter() - t0:.1f}s")
    finally:
        ttnn.close_device(device)
