# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for the standalone Qwen3-Embedding-4B perf entry points
(`demo_bs{1,32}_isl512.py` and `tests/perf/new_perf_bs{1,32}_isl512.py`).

Each entry-point file imports `run_perf` and parametrizes one workload.
This module bakes in every optimization env var validated for these shapes so
each entry point is a true "all-optimizations-on" run with no extra flags.

Qwen3-Embedding-4B architecture (vs 0.6B):
    hidden_size      = 2560   (0.6B: 1024)
    num_hidden_layers = 36    (0.6B: 28)
    num_attention_heads = 32  (0.6B: 16)
    num_key_value_heads = 8   (both)
    intermediate_size = 9728  (0.6B: 3072)
    head_dim          = 128   (both)
    GQA ratio         = 4:1   (0.6B: 2:1)

Memory budget for activations (bf16):
    bs1  ISL=512:  512 * 2560 * 2 =  2.5 MB  -> fits in L1 (single-user path)
    bs32 ISL=512:  32 * 512 * 2560 * 2 = 80 MB -> DRAM-resident

Optimization notes specific to 4B:
    - LN block sharding: dim=2560 on 8x8 grid gives block_h*block_w = 2*10 = 20
      which exceeds the 16-tile per-core cap, so QWEN_LN_BLOCK_SHARDED auto-
      disables. The env var is set but inert for this model. bs=1 LN runs on 16
      cores interleaved (~40 us/call); bs=32 LN runs on 130 cores.
    - Head-split NLP ops: n_kv_heads=8 (same as 0.6B), so head_groups=8 applies
      identically; bs=1 ISL=512 gets 16*8=128 work units.
    - RoPE L1: 512 * 128 * 2 = 128 KB, well within the 8 MB cap.
    - With 32 Q-heads (vs 16), SDPA has more batch-head parallelism.
    - 36 layers (vs 28) means ~29% more layer-level work.

Profiled device kernel time (P150, all optimizations enabled):

    bs=1  ISL=512:  31.8 ms total kernel time  (wall ~32.3 ms)
      Matmuls:              19.5 ms  (61.4%)  - all on 64-core (8x8) grid
        FF2 (down_proj):     8.1 ms  (25.4%)  - 224 us/layer x 36
        FF1+FF3 (gate/up):   7.4 ms  (23.2%)  - 102 us/layer x 36 x 2
        QKV:                 2.3 ms  ( 7.4%)  -  65 us/layer x 36
        WO:                  1.7 ms  ( 5.4%)  -  47 us/layer x 36
      Attention (SDPA+RoPE): 5.6 ms  (17.6%)
        SDPA:                3.6 ms  (11.2%)  -  99 us/layer (32 Q-heads on 64 cores)
        RoPE Q:              1.5 ms  ( 4.9%)  -  43 us/layer on 130 cores
      Norms:                 3.4 ms  (10.7%)  -  40 us/layer x 2 on 16 cores
      Element-wise:          2.5 ms  ( 7.7%)  - silu_mul 48 us/layer on 130 cores
      TM ops:                0.8 ms  ( 2.7%)  - head-split at 6-12 us/layer

    bs=32 ISL=512:  ~725 ms best wall time (130-core matmul grid)
      Previous (80-core, 8x10):  882 ms -> 130-core (13x10): 725 ms  (-18%)
      Matmuls (130 cores):  dominant cost on full (13,10) grid
      Other (130 cores):    SDPA/RoPE/LN/element-wise

Recommended config, set before model build. Each var is `setdefault`'d so the
user can still override from the shell:

    QWEN_QKV_BFP4=1
    QWEN_WO_BFP4=1
    QWEN_FF13_OUT_BFP8=1
    QWEN_FFNORM_IN_BFP8=1
    QWEN_RESIDUAL_BFP8=1
    QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1
    QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1
    QWEN_ROPE_PREFILL_L1=1
    QWEN_LN_BLOCK_SHARDED=1   # inert for 4B (auto-disables)
    TT_SKIP_KV_CACHE_FILL=1
    TT_BATCHED_L1_PREFILL=1   # only for bs>1 && fits in L1
    QWEN_MM_GRID=13,10        # 130-core matmul grid for DRAM-resident (bs>=11)
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


MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
BLOCK_SIZE = 32


# ---------------------------------------------------------------------------
# Environment / optimizations
# ---------------------------------------------------------------------------


def apply_recommended_env(batched_l1: bool) -> None:
    """Set the recommended optimization env vars unless the user overrode them.

    `setdefault` semantics keep this idempotent and respect explicit shell
    overrides -- useful for A/B comparisons of individual knobs.
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
    else:
        os.environ.setdefault("QWEN_MM_GRID", "13,10")


def qwen_embedding_optimizations(model_args):
    """`DecodersPrecision.performance` + opt-in BFP4 promotions for FF2/QKV/WO.

    Mirrors the helper in the 0.6B `demo.py`; see that file for the full
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
    """Page table sizing for the 4B model.

    KV cache per token per layer: 2 * n_kv_heads * head_dim * 2 bytes
      = 2 * 8 * 128 * 2 = 4 KB/token/layer  (×36 layers = 144 KB/token total)
    """
    if batch_size == 1:
        return {"page_block_size": BLOCK_SIZE, "page_max_num_blocks": 512}
    page_max = max(512, math.ceil(seq_len / BLOCK_SIZE) * batch_size * 2)
    return {"page_block_size": BLOCK_SIZE, "page_max_num_blocks": page_max}


def build_single_device_model(mesh_device, batch_size: int, seq_len: int):
    """Build one model instance + Generator + page table on a single mesh."""
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

    `warmup=True` triggers the multi-seq warmup loop inside the generator.
    MUST only be set on the FIRST call.
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
    """Build, compile, and benchmark Qwen3-Embedding-4B on a single mesh.

    Args:
        mesh_device: ttnn.MeshDevice (1x1 for the standalone path)
        batch_size: 1 or 32 (must match `_page_params_for`)
        seq_len: input sequence length (typically 512)
        num_iterations: how many post-compile iterations to run. Set to 1 for
            tracy captures so the signposted zone covers exactly one
            trace-replay forward.
        emit_signposts: when True, the LAST iteration's forward is wrapped in
            `tracy.signpost("start"/"stop")` markers.
        is_ci_env: forward to `create_benchmark_data` for CI dashboards.
    """
    profiler = BenchmarkProfiler()
    profiler.start("run")
    tt_device_name = determine_device_name(mesh_device)

    logger.info(f"Building Qwen3-Embedding-4B: bs={batch_size}, seq_len={seq_len}, device={tt_device_name}")
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
        generator.prev_page_table = None

        sig = emit_signposts and i == last_iter_idx

        profiler.start(f"inference_prefill_{i}")
        if sig:
            _tracy_signpost("start")
        try:
            result = _run_prefill(generator, input_ids, page_table, kv_caches, prompt_lens, warmup=False)
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
    logger.info(f"  Qwen3-Embedding-4B Performance  ({tt_device_name})")
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
            ml_model_name=getattr(model_args, "base_model_name", "Qwen3-Embedding-4B"),
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
    """`python <entry_file>` path -- opens its own device, no pytest fixture."""
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
