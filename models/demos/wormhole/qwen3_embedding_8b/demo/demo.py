# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Embedding-0.6B Performance Demo

Measures embedding throughput and latency on Tenstorrent hardware.
Supports data parallelism (DP) to run independent model instances on
separate devices — ideal for embedding workloads which are prefill-only.

Metrics reported:
  - Compile time: time to capture the first prefill trace
  - Prefill time: wall-clock time to embed the full batch
  - Embeddings/s: batch_size / prefill_time
  - Tokens/s:     total_tokens / prefill_time

Usage (standalone, single device):
    python models/demos/wormhole/qwen3_embedding_8b/demo/demo.py

Usage (pytest, picks device from MESH_DEVICE env):
    pytest models/demos/wormhole/qwen3_embedding_8b/demo/demo.py -sv -k "dp32"
    pytest .../demo.py -sv -k "dp1-batch1-seqlt512"   # batch=1, seq length < 512
"""

import json
import math
import os
import time

import pytest
import torch
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

import ttnn
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision, determine_device_name

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BLOCK_SIZE = 32

INSTRUCTION = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

SAMPLE_TEXTS = [
    "Artificial intelligence is transforming how we interact with technology.",
    "AI is changing the way humans use computers and machines.",
    "Machine learning algorithms are revolutionizing data analysis.",
    "The weather is sunny today with clear blue skies.",
    "Quantum computing promises to solve problems that are intractable for classical computers.",
    "Baking bread requires flour, water, yeast, and patience.",
    "Neural networks mimic the human brain's structure and function.",
    "Natural language processing enables computers to understand text.",
]

MESH_SHAPES = {
    1: (1, 1),
    2: (1, 2),
    8: (1, 8),
    32: (8, 4),
}


def load_input_texts(input_file, batch_size):
    """Load input texts from a JSON file or generate synthetic ones."""
    if input_file and os.path.exists(input_file):
        with open(input_file, "r") as f:
            data = json.load(f)
        texts = [item["text"] if isinstance(item, dict) else item for item in data]
    else:
        texts = [INSTRUCTION + t for t in SAMPLE_TEXTS]

    while len(texts) < batch_size:
        texts = texts * 2
    return texts[:batch_size]


def get_default_mesh_device_param():
    if ttnn.using_distributed_env():
        try:
            n = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
            return MESH_SHAPES.get(n, n)
        except Exception:
            pass
    n = len(ttnn.get_device_ids())
    return MESH_SHAPES.get(n, n)


def _submesh_has_local_devices(submesh):
    view = submesh.get_view()
    return any(
        view.is_local(ttnn.MeshCoordinate(row, col))
        for row in range(submesh.shape[0])
        for col in range(submesh.shape[1])
    )


def prepare_embedding_model(
    mesh_device,
    global_batch_size,
    max_seq_len,
    optimizations,
    page_params,
    data_parallel=1,
):
    """Build TT model(s), generator, and KV cache for embedding workloads.

    When data_parallel > 1, creates independent model instances on submeshes.
    """
    batch_per_dp = global_batch_size // data_parallel

    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )

    all_submeshes = create_submeshes(mesh_device, data_parallel)
    local_indices = (
        [i for i, s in enumerate(all_submeshes) if _submesh_has_local_devices(s)]
        if isinstance(mesh_device, ttnn.MeshDevice) and data_parallel > 1
        else list(range(len(all_submeshes)))
    )
    submeshes = [all_submeshes[i] for i in local_indices]

    if not submeshes:
        raise RuntimeError("No local submeshes available on this host rank")

    if len(submeshes) != len(all_submeshes):
        logger.info(f"Distributed mode: using {len(submeshes)}/{len(all_submeshes)} local submeshes")

    models = []
    model_args_list = []
    kv_caches = []
    state_dict = None

    for submesh in submeshes:
        model_args_i, model_i, kv_cache_i, state_dict = create_tt_model(
            submesh,
            instruct=False,
            max_batch_size=batch_per_dp,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
        )
        models.append(model_i)
        model_args_list.append(model_args_i)
        kv_caches.append([layer.attention.layer_past for layer in model_i.layers])

    tokenizer = model_args_list[0].tokenizer

    generator = Generator(
        models,
        model_args_list,
        mesh_device,
        tokenizer=tokenizer,
    )

    local_dp = len(submeshes)
    local_batch = batch_per_dp * local_dp

    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation).repeat(local_dp)
    page_table = reverse_permutation.reshape(local_batch, paged_attention_config.max_num_blocks // batch_per_dp)

    return generator, model_args_list[0], tokenizer, kv_caches, page_table


def tokenize_and_pad(tokenizer, texts, max_seq_len):
    """Tokenize texts, returning padded input_ids and original lengths."""
    encoded = tokenizer(texts, padding="max_length", max_length=max_seq_len, truncation=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    original_lens = attention_mask.sum(dim=1).tolist()
    return input_ids, [int(l) for l in original_lens]


def generate_synthetic_inputs(tokenizer, batch_size, seq_len):
    """Generate random token sequences of exactly seq_len for ISL benchmarking."""
    vocab_size = tokenizer.vocab_size
    low = max(100, 0)
    high = min(vocab_size, 50000)
    input_ids = torch.randint(low, high, (batch_size, seq_len), dtype=torch.long)
    prompt_lens = [seq_len] * batch_size
    return input_ids, prompt_lens


def run_embedding_prefill(
    generator, input_ids, page_table, kv_cache, prompt_lens, enable_trace, return_hidden_states, warmup_prefill=True
):
    """Run a single embedding prefill pass and return the result."""
    return generator.prefill_forward_text(
        input_ids,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        enable_trace=enable_trace,
        return_hidden_states=return_hidden_states,
        warmup_prefill=warmup_prefill,
    )


def clear_all_kv_caches(generator):
    """Zero out the KV cache for every DP model instance."""
    for model_instance in generator.model:
        for layer in model_instance.layers:
            k_cache, v_cache = layer.attention.layer_past
            k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
            v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size, max_seq_len, input_seq_len, page_params, num_iterations, enable_trace, data_parallel",
    [
        (  # dp1-batch1-short: single text, 1024 tokens, single device
            1,
            1024,
            None,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            5,
            True,
            1,
        ),
        (  # dp1-batch1-seqlt512: batch=1, max_seq_len and ISL both < 512 (synthetic tokens)
            1,
            256,
            256,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            3,
            True,
            1,
        ),
        (  # dp1-batch8-short: 8 texts, 1024 tokens, single device
            8,
            1024,
            None,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            3,
            True,
            1,
        ),
        (  # dp32-isl512
            32,
            8192,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            5,
            True,
            32,
        ),
        (  # dp32-isl1024
            32,
            8192,
            1024,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            5,
            True,
            32,
        ),
        (  # dp32-isl2048
            32,
            8192,
            2048,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            5,
            True,
            32,
        ),
        (  # dp32-isl4096
            32,
            8192,
            4096,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            5,
            True,
            32,
        ),
        (  # dp32-isl8192
            32,
            8192,
            8192,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            5,
            True,
            32,
        ),
    ],
    ids=[
        "dp1-batch1-short",
        "dp1-batch1-seqlt512",
        "dp1-batch8-short",
        "dp32-isl512",
        "dp32-isl1024",
        "dp32-isl2048",
        "dp32-isl4096",
        "dp32-isl8192",
    ],
)
@pytest.mark.parametrize(
    "optimizations",
    [lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)],
    ids=["performance"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "T3K": (1, 8),
            "TG": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), get_default_mesh_device_param())
    ],
    indirect=True,
)
def test_embedding_perf(
    mesh_device,
    batch_size,
    max_seq_len,
    input_seq_len,
    page_params,
    num_iterations,
    enable_trace,
    data_parallel,
    optimizations,
    is_ci_env,
    request,
):
    """
    Embedding performance demo: measures compile time, prefill latency, and throughput.

    max_seq_len:   model capacity (rotary embeddings, KV cache allocation)
    input_seq_len: actual tokens per input (None = use real sample texts)
    """
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1

    if data_parallel > num_devices:
        pytest.skip(f"data_parallel={data_parallel} requires {data_parallel} devices, only {num_devices} available")
    if batch_size % data_parallel != 0:
        pytest.skip(f"batch_size={batch_size} not evenly divisible by data_parallel={data_parallel}")

    skip_warmup = data_parallel > 1
    isl = input_seq_len or max_seq_len

    profiler = BenchmarkProfiler()
    profiler.start("run")

    test_id = request.node.callspec.id
    tt_device_name = determine_device_name(mesh_device)

    # ---- Build model ----
    batch_per_dp = batch_size // data_parallel
    logger.info(
        f"Building model: global_batch={batch_size}, batch_per_dp={batch_per_dp}, "
        f"dp={data_parallel}, max_seq_len={max_seq_len}, ISL={isl}, device={tt_device_name}"
    )
    profiler.start("build_model")
    generator, model_args, tokenizer, kv_caches, page_table = prepare_embedding_model(
        mesh_device,
        global_batch_size=batch_size,
        max_seq_len=max_seq_len,
        optimizations=optimizations,
        page_params=page_params,
        data_parallel=data_parallel,
    )
    profiler.end("build_model")
    logger.info(f"Model built in {profiler.get_duration('build_model'):.1f}s (dp={generator.data_parallel})")

    # ---- Prepare inputs ----
    profiler.start("loading_inputs")
    if input_seq_len is not None:
        input_ids, prompt_lens = generate_synthetic_inputs(tokenizer, batch_size, input_seq_len)
    else:
        texts = load_input_texts(None, batch_size)
        input_ids, prompt_lens = tokenize_and_pad(tokenizer, texts, max_seq_len)
    profiler.end("loading_inputs")

    total_input_tokens = sum(prompt_lens)
    logger.info(f"Prepared {batch_size} inputs, ISL={isl}, total tokens = {total_input_tokens}")

    # ---- Warmup / compile ----
    logger.info("Compiling (first prefill)...")
    profiler.start("compile_prefill")
    _ = run_embedding_prefill(
        generator,
        input_ids,
        page_table,
        kv_caches,
        prompt_lens,
        enable_trace,
        return_hidden_states=True,
        warmup_prefill=not skip_warmup,
    )
    profiler.end("compile_prefill")
    logger.info(f"Compile prefill: {profiler.get_duration('compile_prefill'):.2f}s")

    # ---- Benchmark iterations ----
    logger.info(f"Running {num_iterations} benchmark iterations...")
    iteration_times = []
    embeddings = None

    for i in range(num_iterations):
        clear_all_kv_caches(generator)
        generator.prev_page_table = None

        profiler.start(f"inference_prefill_{i}")
        result = run_embedding_prefill(
            generator,
            input_ids,
            page_table,
            kv_caches,
            prompt_lens,
            enable_trace,
            return_hidden_states=True,
            warmup_prefill=False,
        )
        profiler.end(f"inference_prefill_{i}")

        t = profiler.get_duration(f"inference_prefill_{i}")
        iteration_times.append(t)
        logger.info(f"  Iteration {i}: {t * 1000:.1f}ms")

        if embeddings is None:
            embeddings = result

    # ---- Compute metrics ----
    avg_prefill_time = sum(iteration_times) / len(iteration_times)
    best_prefill_time = min(iteration_times)

    embeddings_per_sec_avg = batch_size / avg_prefill_time
    embeddings_per_sec_best = batch_size / best_prefill_time
    tokens_per_sec_avg = total_input_tokens / avg_prefill_time
    tokens_per_sec_best = total_input_tokens / best_prefill_time

    measurements = {
        "compile_prefill": profiler.get_duration("compile_prefill"),
        "avg_prefill_time": avg_prefill_time,
        "best_prefill_time": best_prefill_time,
        "embeddings/s_avg": embeddings_per_sec_avg,
        "embeddings/s_best": embeddings_per_sec_best,
        "prefill_t/s_avg": tokens_per_sec_avg,
        "prefill_t/s_best": tokens_per_sec_best,
        "build_model_time": profiler.get_duration("build_model"),
        "batch_size": batch_size,
        "data_parallel": data_parallel,
        "input_seq_len": isl,
        "max_seq_len": max_seq_len,
        "total_input_tokens": total_input_tokens,
    }

    # ---- Print results ----
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Qwen3-Embedding-0.6B Performance  ({tt_device_name})")
    logger.info("=" * 60)
    logger.info(f"  Data parallel:        {data_parallel}")
    logger.info(f"  Global batch size:    {batch_size}")
    logger.info(f"  Batch per DP group:   {batch_per_dp}")
    logger.info(f"  Input seq length:     {isl}")
    logger.info(f"  Max seq length:       {max_seq_len}")
    logger.info(f"  Total input tokens:   {total_input_tokens}")
    logger.info(f"  Iterations:           {num_iterations}")
    logger.info("-" * 60)
    logger.info(f"  Model build time:     {measurements['build_model_time']:.1f}s")
    logger.info(f"  Compile (1st run):    {measurements['compile_prefill']:.2f}s")
    logger.info("-" * 60)
    logger.info(f"  Avg prefill time:     {avg_prefill_time * 1000:.1f}ms")
    logger.info(f"  Best prefill time:    {best_prefill_time * 1000:.1f}ms")
    logger.info(f"  Avg embeddings/s:     {embeddings_per_sec_avg:.1f}")
    logger.info(f"  Best embeddings/s:    {embeddings_per_sec_best:.1f}")
    logger.info(f"  Avg tokens/s:         {tokens_per_sec_avg:.0f}")
    logger.info(f"  Best tokens/s:        {tokens_per_sec_best:.0f}")
    logger.info("=" * 60)

    # ---- Cosine similarity sanity check (only for real text inputs) ----
    if data_parallel <= 1 and embeddings is not None and batch_size >= 2:
        emb_np = embeddings.float().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        if emb_np.ndim == 1:
            emb_np = emb_np.reshape(1, -1)
        elif emb_np.ndim > 2:
            emb_np = emb_np.reshape(batch_size, -1)

        sim = cosine_similarity(emb_np)
        logger.info(f"  Cosine similarity [0,1] = {sim[0, 1]:.4f} (should be high, both AI-related)")
        if batch_size >= 4:
            logger.info(f"  Cosine similarity [0,3] = {sim[0, 3]:.4f} (should be low, AI vs weather)")

    # ---- CI benchmark data ----
    profiler.end("run")

    if is_ci_env:
        model_name = model_args.base_model_name if hasattr(model_args, "base_model_name") else "Qwen3-Embedding-0.6B"
        benchmark_data = create_benchmark_data(profiler, measurements, {}, {})
        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name=model_name,
            ml_model_type="embedding",
            num_layers=model_args.n_layers,
            batch_size=batch_size,
            config_params={"data_parallel": data_parallel, "tensor_parallel": num_devices // data_parallel},
            input_sequence_length=isl,
            output_sequence_length=0,
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-Embedding-0.6B performance demo")
    parser.add_argument("--batch-size", type=int, default=1, help="Global batch size")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID (single-device mode)")
    args = parser.parse_args()

    page_max = max(512, math.ceil(args.max_seq_len / BLOCK_SIZE) * args.batch_size * 2)
    page_params = {"page_block_size": BLOCK_SIZE, "page_max_num_blocks": page_max}

    logger.info(f"Opening device {args.device_id}...")
    device = ttnn.open_device(
        device_id=args.device_id, l1_small_size=32768, trace_region_size=50000000, num_command_queues=1
    )

    try:
        profiler = BenchmarkProfiler()
        profiler.start("run")

        texts = load_input_texts(None, args.batch_size)
        optimizations = lambda ma: DecodersPrecision.performance(ma.n_layers, ma.model_name)

        generator, model_args, tokenizer, kv_caches, page_table = prepare_embedding_model(
            device,
            global_batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            optimizations=optimizations,
            page_params=page_params,
            data_parallel=1,
        )

        input_ids, prompt_lens = tokenize_and_pad(tokenizer, texts, args.max_seq_len)
        total_tokens = sum(prompt_lens)

        logger.info("Compile run...")
        _ = run_embedding_prefill(generator, input_ids, page_table, kv_caches, prompt_lens, True, True)

        logger.info(f"Benchmarking {args.iterations} iterations...")
        times = []
        for i in range(args.iterations):
            clear_all_kv_caches(generator)
            generator.prev_page_table = None
            t0 = time.perf_counter()
            _ = run_embedding_prefill(generator, input_ids, page_table, kv_caches, prompt_lens, True, True)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            logger.info(f"  Iter {i}: {(t1 - t0) * 1000:.1f}ms")

        avg_t = sum(times) / len(times)
        best_t = min(times)
        logger.info("")
        logger.info(
            f"Avg: {avg_t * 1000:.1f}ms | {args.batch_size / avg_t:.1f} emb/s | {total_tokens / avg_t:.0f} tok/s"
        )
        logger.info(
            f"Best: {best_t * 1000:.1f}ms | {args.batch_size / best_t:.1f} emb/s | {total_tokens / best_t:.0f} tok/s"
        )

        profiler.end("run")

    finally:
        ttnn.close_device(device)
