# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Focused BGE-M3 performance test.

Runs only batch=1, data-parallel=1, input sequence length=512.
When run under Tracy, the measured inference forward emits `start`/`stop`
signposts so reports can exclude compile/warmup.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.demos.wormhole.bge_m3.tt.common import create_tt_model
from models.demos.wormhole.bge_m3.tt.model_config import determine_device_name
from models.perf.benchmarking_utils import BenchmarkProfiler

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(*_args, **_kwargs):
        pass


MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 1
SEQ_LEN = 512
NUM_ITERATIONS = 1


def generate_synthetic_inputs(tokenizer):
    vocab_size = tokenizer.vocab_size
    high = max(101, min(vocab_size, 50000))
    input_ids = torch.randint(100, high, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
    return input_ids, attention_mask, token_type_ids


def to_ttnn_ids(ids: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        ids.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def build_model(mesh_device):
    model_args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=BATCH_SIZE,
        max_seq_len=SEQ_LEN,
        dtype=ttnn.bfloat8_b,
        hf_model_name=MODEL_NAME,
    )
    return model_args, model, model_args.tokenizer


def run_forward(
    model,
    mesh_device,
    input_ids,
    attention_mask,
    token_type_ids,
    profiler=None,
    step_name=None,
    emit_tracy_signposts=False,
):
    tt_input_ids = to_ttnn_ids(input_ids, mesh_device)
    tt_attention_mask = to_ttnn_ids(attention_mask, mesh_device)
    tt_token_type_ids = to_ttnn_ids(token_type_ids, mesh_device)

    if profiler is not None and step_name is not None:
        profiler.start(step_name)

    if emit_tracy_signposts:
        signpost("start")
    try:
        tt_output = model(
            input_ids=tt_input_ids,
            attention_mask=tt_attention_mask,
            token_type_ids=tt_token_type_ids,
            position_ids=None,
        )
        ttnn.synchronize_device(mesh_device)
    finally:
        if emit_tracy_signposts:
            signpost("stop")

    if profiler is not None and step_name is not None:
        profiler.end(step_name)

    hidden_states = to_torch_auto_compose(tt_output, device=mesh_device)
    if hidden_states.dim() == 4 and hidden_states.shape[1] == 1:
        hidden_states = hidden_states.squeeze(1)
    hidden_states = hidden_states[:, :SEQ_LEN, :].to(torch.float32)
    return mean_pool(hidden_states, attention_mask[:, : hidden_states.shape[1]])


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "batch_size, seq_len, num_iterations, data_parallel",
    [(BATCH_SIZE, SEQ_LEN, NUM_ITERATIONS, 1)],
    ids=["batch1dp1-isl512"],
)
def test_embedding_perf(mesh_device, batch_size, seq_len, num_iterations, data_parallel, is_ci_env):
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    if num_devices < 1:
        pytest.skip("No devices available")

    profiler = BenchmarkProfiler()
    profiler.start("run")

    device_name_info = determine_device_name(mesh_device)
    tt_device_name = device_name_info[0] if isinstance(device_name_info, tuple) else str(device_name_info)

    logger.info(f"Building BGE-M3: batch={batch_size}, dp={data_parallel}, seq_len={seq_len}, device={tt_device_name}")
    profiler.start("build_model")
    model_args, model, tokenizer = build_model(mesh_device)
    profiler.end("build_model")
    logger.info(f"Model built in {profiler.get_duration('build_model'):.1f}s")

    input_ids, attention_mask, token_type_ids = generate_synthetic_inputs(tokenizer)
    total_input_tokens = batch_size * seq_len

    logger.info("Compiling (first forward)...")
    embeddings = run_forward(model, mesh_device, input_ids, attention_mask, token_type_ids, profiler, "compile_prefill")
    logger.info(f"Compile forward: {profiler.get_duration('compile_prefill'):.2f}s")

    logger.info(f"Running {num_iterations} benchmark iteration...")
    iteration_times = []
    for i in range(num_iterations):
        embeddings = run_forward(
            model,
            mesh_device,
            input_ids,
            attention_mask,
            token_type_ids,
            profiler,
            f"inference_prefill_{i}",
            emit_tracy_signposts=(i == 0),
        )
        iteration_time = profiler.get_duration(f"inference_prefill_{i}")
        iteration_times.append(iteration_time)
        logger.info(f"  Iteration {i}: {iteration_time * 1000:.1f}ms")

    avg_prefill_time = sum(iteration_times) / len(iteration_times)
    best_prefill_time = min(iteration_times)
    measurements = {
        "compile_prefill": profiler.get_duration("compile_prefill"),
        "avg_prefill_time": avg_prefill_time,
        "best_prefill_time": best_prefill_time,
        "embeddings/s_avg": batch_size / avg_prefill_time,
        "embeddings/s_best": batch_size / best_prefill_time,
        "prefill_t/s_avg": total_input_tokens / avg_prefill_time,
        "prefill_t/s_best": total_input_tokens / best_prefill_time,
        "build_model_time": profiler.get_duration("build_model"),
        "batch_size": batch_size,
        "data_parallel": data_parallel,
        "input_seq_len": seq_len,
        "max_seq_len": seq_len,
        "total_input_tokens": total_input_tokens,
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  BGE-M3 Performance  ({tt_device_name})")
    logger.info("=" * 60)
    logger.info(f"  Data parallel:        {data_parallel}")
    logger.info(f"  Global batch size:    {batch_size}")
    logger.info(f"  Input seq length:     {seq_len}")
    logger.info(f"  Max seq length:       {seq_len}")
    logger.info(f"  Total input tokens:   {total_input_tokens}")
    logger.info(f"  Iterations:           {num_iterations}")
    logger.info("-" * 60)
    logger.info(f"  Model build time:     {measurements['build_model_time']:.1f}s")
    logger.info(f"  Compile (1st run):    {measurements['compile_prefill']:.2f}s")
    logger.info("-" * 60)
    logger.info(f"  Avg prefill time:     {avg_prefill_time * 1000:.1f}ms")
    logger.info(f"  Best prefill time:    {best_prefill_time * 1000:.1f}ms")
    logger.info(f"  Avg embeddings/s:     {measurements['embeddings/s_avg']:.1f}")
    logger.info(f"  Best embeddings/s:    {measurements['embeddings/s_best']:.1f}")
    logger.info(f"  Avg tokens/s:         {measurements['prefill_t/s_avg']:.0f}")
    logger.info(f"  Best tokens/s:        {measurements['prefill_t/s_best']:.0f}")
    logger.info("=" * 60)

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
            config_params={"data_parallel": data_parallel, "tensor_parallel": num_devices // data_parallel},
            input_sequence_length=seq_len,
            output_sequence_length=0,
        )

    assert embeddings is not None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Focused BGE-M3 performance demo")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID")
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS, help="Benchmark iterations")
    args = parser.parse_args()

    logger.info(f"Opening device {args.device_id}...")
    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=32768,
        trace_region_size=50000000,
        num_command_queues=1,
    )

    try:
        profiler = BenchmarkProfiler()
        _model_args, model, tokenizer = build_model(device)
        input_ids, attention_mask, token_type_ids = generate_synthetic_inputs(tokenizer)
        total_input_tokens = BATCH_SIZE * SEQ_LEN

        logger.info("Compile run...")
        _ = run_forward(model, device, input_ids, attention_mask, token_type_ids, profiler, "compile_prefill")

        logger.info(f"Benchmarking {args.iterations} iteration...")
        times = []
        for i in range(args.iterations):
            _ = run_forward(
                model,
                device,
                input_ids,
                attention_mask,
                token_type_ids,
                profiler,
                f"inference_{i}",
                emit_tracy_signposts=(i == 0),
            )
            iteration_time = profiler.get_duration(f"inference_{i}")
            times.append(iteration_time)
            logger.info(f"  Iter {i}: {iteration_time * 1000:.1f}ms")

        avg_t = sum(times) / len(times)
        best_t = min(times)
        logger.info("")
        logger.info(
            f"Avg: {avg_t * 1000:.1f}ms | {BATCH_SIZE / avg_t:.1f} emb/s | {total_input_tokens / avg_t:.0f} tok/s"
        )
        logger.info(
            f"Best: {best_t * 1000:.1f}ms | {BATCH_SIZE / best_t:.1f} emb/s | {total_input_tokens / best_t:.0f} tok/s"
        )
    finally:
        ttnn.close_device(device)
