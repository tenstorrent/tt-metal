# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
BGE-M3 Performance Demo

Measures embedding throughput and latency on Tenstorrent hardware.

Metrics reported:
  - Compile time: time for the first forward pass
  - Forward time: wall-clock time to embed the full batch
  - Embeddings/s: batch_size / forward_time
  - Tokens/s:     total_tokens / forward_time

Usage (standalone, single device):
    python models/demos/wormhole/bge_m3/demo/perf_demo.py

Usage (pytest, picks device from MESH_DEVICE env):
    pytest models/demos/wormhole/bge_m3/demo/perf_demo.py -sv
"""

import json
import os

import pytest
import torch
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.demos.wormhole.bge_m3.tt.common import create_tt_model
from models.demos.wormhole.bge_m3.tt.model_config import determine_device_name
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.generator import create_submeshes

try:
    import tracy
except ImportError:
    tracy = None

MODEL_NAME = "BAAI/bge-m3"

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


def _tracy_signpost(message: str) -> None:
    """Emit Tracy signposts when Tracy is available."""
    if tracy is not None:
        tracy.signpost(message)


def load_input_texts(input_file, batch_size):
    """Load input texts from a JSON file or generate synthetic ones."""
    if input_file and os.path.exists(input_file):
        with open(input_file, "r") as f:
            data = json.load(f)
        texts = [item["text"] if isinstance(item, dict) else item for item in data]
    else:
        texts = SAMPLE_TEXTS[:]

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
    tt_data_parallel=1,
):
    """Build TT model(s) for embedding workloads.

    When tt_data_parallel > 1, creates independent model instances on submeshes.
    """
    if global_batch_size % tt_data_parallel != 0:
        raise ValueError(
            f"global_batch_size={global_batch_size} must be divisible by tt_data_parallel={tt_data_parallel}"
        )

    batch_per_dp = global_batch_size // tt_data_parallel

    all_submeshes = create_submeshes(mesh_device, tt_data_parallel)
    local_indices = (
        [i for i, s in enumerate(all_submeshes) if _submesh_has_local_devices(s)]
        if isinstance(mesh_device, ttnn.MeshDevice) and tt_data_parallel > 1
        else list(range(len(all_submeshes)))
    )
    submeshes = [all_submeshes[i] for i in local_indices]

    if not submeshes:
        raise RuntimeError("No local submeshes available on this host rank")

    if len(submeshes) != len(all_submeshes):
        logger.info(f"Distributed mode: using {len(submeshes)}/{len(all_submeshes)} local submeshes")

    models = []
    model_args_list = []
    state_dict = None

    for submesh in submeshes:
        model_args_i, model_i, state_dict = create_tt_model(
            mesh_device=submesh,
            max_batch_size=batch_per_dp,
            max_seq_len=max_seq_len,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            hf_model_name=MODEL_NAME,
        )
        models.append(model_i)
        model_args_list.append(model_args_i)

    if not model_args_list or model_args_list[0].tokenizer is None:
        raise RuntimeError("BGE-M3 model did not initialize model_args/tokenizer")

    runtime = {
        "models": models,
        "submeshes": submeshes,
        "batch_per_dp": batch_per_dp,
        "global_data_parallel": tt_data_parallel,
        "local_data_parallel": len(submeshes),
    }
    return runtime, model_args_list[0], model_args_list[0].tokenizer


def tokenize_and_pad(tokenizer, texts, max_seq_len):
    """Tokenize texts, returning padded input_ids and original lengths."""
    encoded = tokenizer(texts, padding="max_length", max_length=max_seq_len, truncation=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if input_ids.shape[1] != max_seq_len:
        raise RuntimeError(f"Tokenizer output length ({input_ids.shape[1]}) does not match max_seq_len ({max_seq_len})")
    token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids))
    original_lens = attention_mask.sum(dim=1).tolist()
    return input_ids, attention_mask, token_type_ids, [int(l) for l in original_lens]


def generate_synthetic_inputs(tokenizer, batch_size, seq_len):
    """Generate random token sequences of exactly seq_len for ISL benchmarking."""
    vocab_size = tokenizer.vocab_size
    low = max(100, 0)
    high = min(vocab_size, 50000)
    input_ids = torch.randint(low, high, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
    prompt_lens = [seq_len] * batch_size
    return input_ids, attention_mask, token_type_ids, prompt_lens


def _to_ttnn_ids(ids: torch.Tensor, mesh_device, dtype=ttnn.uint32) -> ttnn.Tensor:
    return ttnn.from_torch(
        ids.to(torch.int32),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def run_embedding_forward(runtime, input_ids, attention_mask, token_type_ids, profiler=None, step_name=None):
    """Run one BGE-M3 TT forward pass and return dense sentence embeddings."""
    models = runtime["models"]
    submeshes = runtime["submeshes"]
    batch_per_dp = runtime["batch_per_dp"]
    local_dp = runtime["local_data_parallel"]

    local_batch = batch_per_dp * local_dp
    if input_ids.shape[0] != local_batch:
        raise RuntimeError(
            f"Input batch ({input_ids.shape[0]}) must match local batch ({local_batch}) for local_data_parallel={local_dp}"
        )

    input_chunks = torch.chunk(input_ids, local_dp, dim=0)
    attention_chunks = torch.chunk(attention_mask, local_dp, dim=0)
    token_type_chunks = torch.chunk(token_type_ids, local_dp, dim=0)

    # Phase 1: Stage TT inputs for each submesh.
    staged_inputs = []
    for i in range(local_dp):
        staged_inputs.append(
            (
                _to_ttnn_ids(input_chunks[i], mesh_device=submeshes[i]),
                _to_ttnn_ids(attention_chunks[i], mesh_device=submeshes[i]),
                _to_ttnn_ids(token_type_chunks[i], mesh_device=submeshes[i]),
            )
        )

    # Phase 2: Dispatch forwards to all submeshes first.
    if profiler and step_name:
        profiler.start(step_name)

    tt_outputs = []
    for i in range(local_dp):
        tt_output = models[i](
            input_ids=staged_inputs[i][0],
            attention_mask=staged_inputs[i][1],
            token_type_ids=staged_inputs[i][2],
            position_ids=None,
        )
        tt_outputs.append(tt_output)

    # We need to explicitly sync the device so the host waits for the computation
    # to finish before we stop the timer, otherwise we're just measuring Python dispatch time.
    for i in range(local_dp):
        ttnn.synchronize_device(submeshes[i])

    if profiler and step_name:
        profiler.end(step_name)

    # Phase 3: Read back and pool outputs.
    pooled_chunks = []
    for i in range(local_dp):
        hidden_states = to_torch_auto_compose(tt_outputs[i], device=submeshes[i])
        if hidden_states.dim() == 4 and hidden_states.shape[1] == 1:
            hidden_states = hidden_states.squeeze(1)
        hidden_states = hidden_states[:, : input_chunks[i].shape[1], :].to(torch.float32)
        pooled_chunks.append(_mean_pool(hidden_states, attention_chunks[i][:, : hidden_states.shape[1]]))

    return torch.cat(pooled_chunks, dim=0)


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size, max_seq_len, input_seq_len, num_iterations, tt_data_parallel",
    [
        (1, 512, 512, 10, 1),
        # (1, 8192, None, 5, 1),
        # (8, 8192, None, 3, 1),
        # TG target: global batch 32, local batch 1 per chip (DP32).
        # (32, 8192, 512, 5, 32),
        # (32, 8192, 1024, 5, 32),
        # (32, 8192, 2048, 5, 32),
        # (32, 8192, 4096, 5, 32),
        # TG target: global batch 1024, local batch 32 per chip (DP32).
        # (1024, 8192, 512, 5, 32),
        # (1024, 8192, 1024, 5, 32),
        # (1024, 8192, 2048, 5, 32),
        # (1024, 8192, 4096, 5, 32),
        # Disabled: ISL 8192 is too large for local batch 32 per chip.
        # (1024, 8192, 8192, 5, 32),
    ],
    ids=[
        "batch1dp1-isl512",
        # "batch32dp32-isl512",
        # "batch32dp32-isl1024",
        # "batch32dp32-isl2048",
        # "batch32dp32-isl4096",
        # "batch1024-isl512",
        # "batch1024-isl1024",
        # "batch1024-isl2048",
        # "batch1024-isl4096",
        # "batch1024-isl8192",
    ],
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
    num_iterations,
    tt_data_parallel,
    is_ci_env,
):
    """
    Embedding performance demo: measures compile time, forward latency, and throughput.

    max_seq_len:   model capacity
    input_seq_len: actual tokens per input (None = use real sample texts)
    """
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1

    if tt_data_parallel > num_devices:
        pytest.skip(
            f"tt_data_parallel={tt_data_parallel} requires {tt_data_parallel} devices, only {num_devices} available"
        )
    if batch_size % max(tt_data_parallel, 1) != 0:
        pytest.skip(f"batch_size={batch_size} not evenly divisible by tt_data_parallel={tt_data_parallel}")
    if input_seq_len is not None and input_seq_len > max_seq_len:
        pytest.skip(f"input_seq_len={input_seq_len} exceeds max_seq_len={max_seq_len}")
    if batch_size < 1:
        pytest.skip("batch_size must be >= 1")

    profiler = BenchmarkProfiler()
    profiler.start("run")

    device_name_info = determine_device_name(mesh_device)
    tt_device_name = device_name_info[0] if isinstance(device_name_info, tuple) else str(device_name_info)

    # ---- Build model ----
    batch_per_dp = batch_size // max(tt_data_parallel, 1)
    logger.info(
        f"Building model: global_batch={batch_size}, batch_per_dp={batch_per_dp}, "
        f"tt_data_parallel={tt_data_parallel}, max_seq_len={max_seq_len}, input_seq_len={input_seq_len}, device={tt_device_name}"
    )
    profiler.start("build_model")
    runtime, model_args, tokenizer = prepare_embedding_model(
        mesh_device,
        global_batch_size=batch_size,
        max_seq_len=max_seq_len,
        tt_data_parallel=tt_data_parallel,
    )
    profiler.end("build_model")
    logger.info(
        f"Model built in {profiler.get_duration('build_model'):.1f}s "
        f"(global_dp={runtime['global_data_parallel']}, local_dp={runtime['local_data_parallel']})"
    )

    # ---- Prepare inputs ----
    profiler.start("loading_inputs")
    if input_seq_len is not None:
        input_ids, attention_mask, token_type_ids, prompt_lens = generate_synthetic_inputs(
            tokenizer, batch_size, input_seq_len
        )
    else:
        texts = load_input_texts(None, batch_size)
        input_ids, attention_mask, token_type_ids, prompt_lens = tokenize_and_pad(tokenizer, texts, max_seq_len)
    profiler.end("loading_inputs")

    isl = input_seq_len if input_seq_len is not None else int(sum(prompt_lens) / len(prompt_lens))
    total_input_tokens = sum(prompt_lens)
    logger.info(f"Prepared {batch_size} inputs, ISL={isl}, total tokens = {total_input_tokens}")

    # ---- Warmup / compile ----
    logger.info("Compiling (first forward)...")
    _tracy_signpost("Compilation pass")
    _ = run_embedding_forward(runtime, input_ids, attention_mask, token_type_ids, profiler, "compile_prefill")
    logger.info(f"Compile forward: {profiler.get_duration('compile_prefill'):.2f}s")

    # ---- Benchmark iterations ----
    logger.info(f"Running {num_iterations} benchmark iterations...")
    iteration_times = []
    embeddings = None

    for i in range(num_iterations):
        if i == 0:
            _tracy_signpost("Performance pass")
        result = run_embedding_forward(
            runtime, input_ids, attention_mask, token_type_ids, profiler, f"inference_prefill_{i}"
        )

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
        "data_parallel": tt_data_parallel,
        "input_seq_len": isl,
        "max_seq_len": max_seq_len,
        "total_input_tokens": total_input_tokens,
    }

    # ---- Print results ----
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  BGE-M3 Performance  ({tt_device_name})")
    logger.info("=" * 60)
    logger.info(f"  Data parallel:        {tt_data_parallel}")
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
    if input_seq_len is None and tt_data_parallel <= 1 and embeddings is not None and batch_size >= 2:
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
        model_name = model_args.hf_model_name if hasattr(model_args, "hf_model_name") else MODEL_NAME
        benchmark_data = create_benchmark_data(profiler, measurements, {}, {})
        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name=model_name,
            ml_model_type="embedding",
            num_layers=getattr(model_args, "n_layers", 0),
            batch_size=batch_size,
            config_params={
                "data_parallel": tt_data_parallel,
                "tensor_parallel": num_devices // max(tt_data_parallel, 1),
            },
            input_sequence_length=isl,
            output_sequence_length=0,
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BGE-M3 performance demo")
    parser.add_argument("--batch-size", type=int, default=1, help="Global batch size")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--input-seq-len", type=int, default=None, help="Synthetic input sequence length")
    parser.add_argument("--input-file", type=str, default=None, help="Optional JSON file with input texts")
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID (single-device mode)")
    parser.add_argument(
        "--tt-data-parallel",
        type=int,
        default=None,
        help="Data parallel groups to build (default: auto from opened device)",
    )
    args = parser.parse_args()

    logger.info(f"Opening device {args.device_id}...")
    device = ttnn.open_device(
        device_id=args.device_id, l1_small_size=32768, trace_region_size=50000000, num_command_queues=1
    )

    try:
        profiler = BenchmarkProfiler()
        profiler.start("run")

        num_devices = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
        tt_data_parallel = args.tt_data_parallel if args.tt_data_parallel is not None else num_devices
        if tt_data_parallel < 1:
            raise ValueError(f"--tt-data-parallel must be >= 1, got {tt_data_parallel}")
        if tt_data_parallel > num_devices:
            raise ValueError(
                f"--tt-data-parallel={tt_data_parallel} requires {tt_data_parallel} devices, but opened device exposes {num_devices}"
            )
        if args.batch_size % tt_data_parallel != 0:
            raise ValueError(
                f"--batch-size={args.batch_size} must be divisible by --tt-data-parallel={tt_data_parallel}"
            )
        logger.info(
            f"Standalone runtime config: global_batch={args.batch_size}, tt_data_parallel={tt_data_parallel}, num_devices={num_devices}"
        )

        runtime, model_args, tokenizer = prepare_embedding_model(
            device,
            global_batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            tt_data_parallel=tt_data_parallel,
        )

        if args.input_seq_len is not None:
            input_ids, attention_mask, token_type_ids, prompt_lens = generate_synthetic_inputs(
                tokenizer, args.batch_size, args.input_seq_len
            )
        else:
            texts = load_input_texts(args.input_file, args.batch_size)
            input_ids, attention_mask, token_type_ids, prompt_lens = tokenize_and_pad(
                tokenizer, texts, args.max_seq_len
            )

        total_tokens = sum(prompt_lens)

        logger.info("Compile run...")
        _tracy_signpost("Compilation pass")
        _ = run_embedding_forward(runtime, input_ids, attention_mask, token_type_ids, profiler, "compile_prefill")

        logger.info(f"Benchmarking {args.iterations} iterations...")
        times = []
        for i in range(args.iterations):
            if i == 0:
                _tracy_signpost("Performance pass")
            _ = run_embedding_forward(
                runtime, input_ids, attention_mask, token_type_ids, profiler, f"inference_prefill_{i}"
            )
            t = profiler.get_duration(f"inference_prefill_{i}")
            times.append(t)
            logger.info(f"  Iter {i}: {t * 1000:.1f}ms")

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
