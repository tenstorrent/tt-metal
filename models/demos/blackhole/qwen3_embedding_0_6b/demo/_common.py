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
from models.demos.blackhole.qwen3_embedding_0_6b.tt.attention import Qwen3EmbeddingAttention
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    copy_host_to_device,
    get_block_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
)
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
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

    padded_seq_len = get_padded_prefill_len(seq_len)
    # Build the model directly (rather than via `create_tt_model`) so we can
    # plug in the qwen3-embedding-local Attention subclass that owns the
    # bs=1/ISL=512 prefill graph optimizations.
    model_args = ModelArgs(
        mesh_device,
        instruct=False,
        max_batch_size=batch_size,
        optimizations=qwen_embedding_optimizations,
        max_seq_len=padded_seq_len,
        prefetcher=None,
    )
    state_dict = model_args.load_state_dict()
    model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=paged_attention_config,
        attention_class=Qwen3EmbeddingAttention,
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


def build_dp_model(mesh_device, batch_size: int, seq_len: int, data_parallel: int):
    """Build DP model instances across submeshes of *mesh_device*.

    Each submesh gets its own model/Generator replica.  The returned
    ``Generator`` wraps all replicas and handles fan-out / fan-in
    automatically in ``prefill_forward_text``.

    When *data_parallel* is 1 on a multi-device mesh (e.g. Galaxy DP=1
    baseline), a single (1,1) submesh is carved out so the model runs on
    one chip while the full mesh stays initialised with fabric.
    """
    mesh_num_devices = mesh_device.shape[0] * mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    if data_parallel == 1 and mesh_num_devices > 1:
        submeshes = mesh_device.create_submeshes(ttnn.MeshShape(1, 1))[:1]
    else:
        submeshes = create_submeshes(mesh_device, data_parallel)
    assert len(submeshes) == data_parallel, f"Expected {data_parallel} submeshes, got {len(submeshes)}"

    page_params = _page_params_for(batch_size, seq_len)
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    padded_seq_len = get_padded_prefill_len(seq_len)

    all_model_args = []
    all_models = []
    all_kv_caches = []

    for submesh in submeshes:
        model_args_i = ModelArgs(
            submesh,
            instruct=False,
            max_batch_size=batch_size,
            optimizations=qwen_embedding_optimizations,
            max_seq_len=padded_seq_len,
            prefetcher=None,
        )
        state_dict = model_args_i.load_state_dict()
        model_i = Transformer(
            args=model_args_i,
            mesh_device=submesh,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            weight_cache_path=model_args_i.weight_cache_path(ttnn.bfloat8_b),
            paged_attention_config=paged_attention_config,
            attention_class=Qwen3EmbeddingAttention,
        )
        all_model_args.append(model_args_i)
        all_models.append(model_i)
        all_kv_caches.append([layer.attention.layer_past for layer in model_i.layers])

    generator = Generator(
        all_models,
        all_model_args,
        mesh_device,
        tokenizer=all_model_args[0].tokenizer,
    )

    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
    page_table = reverse_permutation.reshape(
        batch_size * data_parallel,
        paged_attention_config.max_num_blocks // batch_size,
    )

    return generator, all_model_args, all_kv_caches, page_table


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


def _prepare_dp_host_inputs(generator, input_ids, page_table, kv_caches, prompt_lens):
    """Pre-compute all host input tensors and trace keys for DP prefill.

    Calling ``prepare_prefill_inputs_trace`` involves ``ttnn.from_torch``
    (tensor creation) and torch slicing — pure Python/CPU work that doesn't
    need to be repeated when the inputs are identical across benchmark
    iterations.  Call this **once** before the timing loop.
    """
    dp = generator.data_parallel
    seq_len = int(prompt_lens[0])
    padded_seq_len = get_padded_prefill_len(seq_len)

    all_host_inputs = []
    all_trace_keys = []

    for i in range(dp):
        pt = page_table[i : i + 1]
        block_size = get_block_size(kv_caches[i])
        n_blocks = num_blocks_in_seq(padded_seq_len, block_size)
        if pt.shape[1] < n_blocks:
            pad = torch.ones(1, n_blocks - pt.shape[1], dtype=torch.int32) * -1
            pt = torch.cat([pt, pad], dim=1)
        pt = pt[:, :n_blocks]

        user_tokens = input_ids[i : i + 1, :seq_len]
        padded_tokens = torch.cat([user_tokens, torch.zeros(1, padded_seq_len - seq_len, dtype=torch.long)], dim=-1)

        host_inputs = generator.model[i].prepare_prefill_inputs_trace(
            padded_tokens, page_table=pt, batch_size=1, user_id=0
        )
        all_host_inputs.append((host_inputs[0], host_inputs[3], host_inputs[4]))
        all_trace_keys.append(f"{padded_seq_len}_{i}_1")

    return all_host_inputs, all_trace_keys


def _run_dp_prefill_phased(generator, prompt_lens, *, precomputed):
    """Phased DP prefill that maximizes device overlap.

    Splits execution into three phases so all device traces are launched
    before any post-processing begins.  H2D copy is performed every call
    to simulate real input variation.

        Phase 1  — H2D copy + fire all traces (tightest loop)
        Phase 2  — queue post-processing (slice+norm+to_layout) + D2H
        Phase 3  — synchronize + extract to host tensors
    """
    dp = generator.data_parallel
    hidden_size = generator.model_args[0].dim
    seq_len = int(prompt_lens[0])
    output_tensor = torch.zeros(dp, hidden_size)

    all_host_inputs, all_trace_keys = precomputed

    # ------------------------------------------------------------------
    # Phase 1: H2D copy + fire all traces
    # ------------------------------------------------------------------
    for i in range(dp):
        trace_key = all_trace_keys[i]
        copy_host_to_device(
            all_host_inputs[i],
            device_tensors=generator.trace_inputs_prefill[trace_key],
            mesh_device=generator.model_args[i].mesh_device,
        )
        ttnn.execute_trace(
            generator.model_args[i].mesh_device,
            generator.trace_id_prefill[trace_key],
            cq_id=0,
            blocking=False,
        )

    # ------------------------------------------------------------------
    # Phase 2: queue post-processing + D2H on each device
    # ------------------------------------------------------------------
    last_token_idx = seq_len - 1
    host_tensors = []
    for i in range(dp):
        hidden_states = generator.model[i].process_hidden_states_after_prefill_trace(
            generator.trace_output_prefill[all_trace_keys[i]], last_token_idx
        )
        host_tensors.append(hidden_states.cpu(blocking=False))

    # ------------------------------------------------------------------
    # Phase 3: synchronize all devices + extract to host
    # ------------------------------------------------------------------
    last_token_relative = last_token_idx % 32
    for i in range(dp):
        ttnn.synchronize_device(generator.model[i].mesh_device)
        ht = ttnn.to_torch(ttnn.get_device_tensors(host_tensors[i])[0])
        output_tensor[i] = ht[0, 0, last_token_relative, :hidden_size]

    return output_tensor


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
    data_parallel: int = 1,
):
    """Build, compile, and benchmark Qwen3-Embedding-0.6B.

    Args:
        mesh_device: ttnn.MeshDevice (1x1 for single-device, larger for DP)
        batch_size: per-device batch size (1, 8, or 32)
        seq_len: input sequence length (typically 512)
        num_iterations: how many post-compile iterations to run. Set to 1 for
            tracy captures so the signposted zone covers exactly one
            trace-replay forward.
        emit_signposts: when True, the LAST iteration's
            `prefill_forward_text(...)` is wrapped in
            `tracy.signpost("start"/"stop")` markers. Use with `num_iterations=1`
            for clean device-time captures via the Tracy profiler.
        is_ci_env: forward to `create_benchmark_data` for CI dashboards.
        data_parallel: number of DP groups (default 1 = single device).
    """
    profiler = BenchmarkProfiler()
    profiler.start("run")
    tt_device_name = determine_device_name(mesh_device)

    global_batch_size = batch_size * data_parallel

    logger.info(
        f"Building Qwen3-Embedding-0.6B: bs={batch_size}, seq_len={seq_len}, "
        f"DP={data_parallel}, device={tt_device_name}"
    )
    mesh_num_devices = mesh_device.shape[0] * mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    use_dp_build = data_parallel > 1 or mesh_num_devices > data_parallel

    profiler.start("build_model")
    if use_dp_build:
        generator, model_args_list, kv_caches, page_table = build_dp_model(
            mesh_device, batch_size=batch_size, seq_len=seq_len, data_parallel=data_parallel
        )
        model_args = model_args_list[0]
    else:
        generator, model_args, kv_caches, page_table = build_single_device_model(
            mesh_device, batch_size=batch_size, seq_len=seq_len
        )
    profiler.end("build_model")
    logger.info(f"Built in {profiler.get_duration('build_model'):.1f}s")

    input_ids, prompt_lens = generate_synthetic_inputs(model_args.tokenizer, global_batch_size, seq_len)
    total_input_tokens = sum(prompt_lens)

    logger.info("Compiling (first prefill captures hardware trace + runs warmup)...")
    profiler.start("compile_prefill")
    _ = _run_prefill(generator, input_ids, page_table, kv_caches, prompt_lens, warmup=True)
    profiler.end("compile_prefill")
    logger.info(f"Compile prefill: {profiler.get_duration('compile_prefill'):.2f}s")

    use_phased_dp = use_dp_build

    if use_phased_dp:
        dp_precomputed = _prepare_dp_host_inputs(generator, input_ids, page_table, kv_caches, prompt_lens)

    logger.info(f"Running {num_iterations} benchmark iteration(s)...")
    if use_phased_dp:
        logger.info("Using phased DP prefill (trace-first, post-process-second) with H2D every iteration")
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
            if use_phased_dp:
                result = _run_dp_prefill_phased(generator, prompt_lens, precomputed=dp_precomputed)
            else:
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
        "embeddings/s_avg": global_batch_size / avg_t,
        "embeddings/s_best": global_batch_size / best_t,
        "prefill_t/s_avg": total_input_tokens / avg_t,
        "prefill_t/s_best": total_input_tokens / best_t,
        "build_model_time": profiler.get_duration("build_model"),
        "batch_size": batch_size,
        "data_parallel": data_parallel,
        "input_seq_len": seq_len,
        "max_seq_len": seq_len,
        "total_input_tokens": total_input_tokens,
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Qwen3-Embedding-0.6B Performance  ({tt_device_name})")
    logger.info("=" * 60)
    logger.info(f"  Batch size (per DP):  {batch_size}")
    logger.info(f"  Data parallel:        {data_parallel}")
    logger.info(f"  Global batch size:    {global_batch_size}")
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
            config_params={"data_parallel": data_parallel, "tensor_parallel": 1},
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
