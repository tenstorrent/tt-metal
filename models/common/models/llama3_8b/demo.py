# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Llama 3.1-8B Demo — accuracy and performance measurement.

Uses executors directly — no vLLM adapter needed.

Usage:
    # Token accuracy test
    MESH_DEVICE=N150 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python_env/bin/pytest models/common/models/llama3_8b/demo.py -k "token-accuracy" -v

    # Batch-1 latency test
    MESH_DEVICE=N150 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python_env/bin/pytest models/common/models/llama3_8b/demo.py -k "batch-1" -v

    # Batch-32 throughput test
    MESH_DEVICE=T3K HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python_env/bin/pytest models/common/models/llama3_8b/demo.py -k "batch-32" -v
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.models.llama3_8b.executor import (
    LlamaExecutor,
    PerfBenchmarkExecutor,
    TeacherForceExecutor,
    TracedLlamaExecutor,
)
from models.common.models.llama3_8b.model import Llama3Transformer1D

# =============================================================================
# Expected metrics
# =============================================================================

EXPECTED_METRICS = {
    "performance": {
        "N150": {"top1": 90, "top5": 97, "tok_s_u": 28.3, "ttft_ms": 110},
        "N300": {"top1": 90, "top5": 97, "tok_s_u": 44.2, "ttft_ms": 70},
        "T3K": {"top1": 90, "top5": 98, "tok_s_u": 64.3, "ttft_ms": 55},
    },
    "accuracy": {
        "N150": {"top1": 96, "top5": 100, "tok_s_u": 25.2, "ttft_ms": 140},
        "N300": {"top1": 96, "top5": 100, "tok_s_u": 38.8, "ttft_ms": 80},
        "T3K": {"top1": 97, "top5": 100, "tok_s_u": 60.8, "ttft_ms": 81},
    },
}

PERF_TOLERANCE = 0.05


# =============================================================================
# Helpers
# =============================================================================


def get_device_name(mesh_device):
    """Detect device topology from mesh shape."""
    num_devices = mesh_device.get_num_devices()
    if num_devices == 1:
        return "N150"
    elif num_devices == 2:
        return "N300"
    elif num_devices == 8:
        return "T3K"
    return f"{num_devices}dev"


def load_reference_data(model_name: str):
    """Load reference tokens and top-5 predictions from .refpt file."""
    ref_path = Path("models/tt_transformers/tests/reference_outputs") / f"{model_name}.refpt"
    if not ref_path.exists():
        pytest.skip(f"Reference file not found: {ref_path}")

    ref_data = torch.load(ref_path, map_location="cpu")
    reference_tokens = ref_data["reference_tokens"]
    top5_tokens = ref_data["top5_tokens"]
    return reference_tokens, top5_tokens


def load_input_prompts(batch_size: int):
    """Load input prompts for performance testing."""
    prompts_path = Path("models/demos/utils/input_data_questions_prefill_128.json")
    if not prompts_path.exists():
        return ["What is the meaning of life?"] * batch_size

    with open(prompts_path) as f:
        data = json.load(f)

    prompts = data if isinstance(data, list) else data.get("prompts", [data.get("prompt", "")])
    while len(prompts) < batch_size:
        prompts = prompts * 2
    return prompts[:batch_size]


def create_model_and_args(mesh_device, optimizations="performance"):
    """Create Llama3Transformer1D and ModelArgs for testing."""
    from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    instruct = "Instruct" in hf_model

    model_args = ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=32,
        optimizations=(
            DecodersPrecision.from_string(optimizations) if isinstance(optimizations, str) else optimizations
        ),
    )

    state_dict = model_args.load_state_dict()
    dtype = ttnn.bfloat8_b

    model = Llama3Transformer1D.from_model_args(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
    )

    return model, model_args


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture(scope="module")
def mesh_device(request):
    """Mesh device fixture. Uses MESH_DEVICE env var or auto-detects."""
    device_name = os.environ.get("MESH_DEVICE", "N150")
    device_params = request.config.getoption("--device-params", default=None)

    num_devices = {"N150": 1, "N300": 2, "T3K": 8}.get(device_name, 1)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, num_devices))
    yield mesh
    ttnn.close_mesh_device(mesh)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param("token-accuracy", id="token-accuracy"),
        pytest.param("batch-1", id="batch-1"),
        pytest.param("batch-32", id="batch-32"),
    ],
)
@pytest.mark.parametrize("optimizations", ["performance", "accuracy"])
def test_llama3_8b(mesh_device, test_config, optimizations):
    """Main test function for TTTv2 Llama 3.1-8B."""
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})

    model, model_args = create_model_and_args(mesh_device, optimizations)

    if test_config == "token-accuracy":
        _run_token_accuracy(model, model_args, mesh_device, expected)
    elif test_config == "batch-1":
        _run_perf_benchmark(model, model_args, mesh_device, expected, batch_size=1)
    elif test_config == "batch-32":
        _run_perf_benchmark(model, model_args, mesh_device, expected, batch_size=32)


# =============================================================================
# Token accuracy
# =============================================================================


def _run_token_accuracy(model, model_args, mesh_device, expected):
    """Run teacher-forcing token accuracy test."""
    model_name = model_args.model_name
    reference_tokens, top5_tokens = load_reference_data(model_name)

    half = len(reference_tokens) // 2
    prompt_tokens = reference_tokens[:half].unsqueeze(0)  # [1, half]

    executor = LlamaExecutor(model, mesh_device, model_args=model_args)

    from models.tt_transformers.tt.common import PagedAttentionConfig

    paged_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
    kv_cache_shape = (
        paged_config.max_num_blocks,
        model_args.n_kv_heads // mesh_device.get_num_devices(),
        paged_config.block_size,
        model_args.head_dim,
    )
    kv_cache = executor.allocate_kv_cache(kv_cache_shape, torch.bfloat16, model_args.n_layers)

    teacher = TeacherForceExecutor(executor)
    result = teacher.run(
        prompt_tokens=prompt_tokens,
        reference_tokens=reference_tokens,
        top5_tokens=top5_tokens[half:],
        kv_cache=kv_cache,
        max_batch_size=model_args.max_batch_size,
    )

    top1 = result.top1_accuracy() * 100
    top5 = result.top5_accuracy() * 100

    logger.info(f"Token accuracy — top1: {top1:.1f}%, top5: {top5:.1f}%")

    if "top1" in expected:
        assert top1 >= expected["top1"] * (
            1 - PERF_TOLERANCE
        ), f"Top-1 accuracy {top1:.1f}% below threshold {expected['top1']}%"
    if "top5" in expected:
        assert top5 >= expected["top5"] * (
            1 - PERF_TOLERANCE
        ), f"Top-5 accuracy {top5:.1f}% below threshold {expected['top5']}%"


# =============================================================================
# Performance benchmark
# =============================================================================


def _run_perf_benchmark(model, model_args, mesh_device, expected, batch_size):
    """Run performance benchmark (TTFT + tok/s/u)."""
    traced_executor = TracedLlamaExecutor(model, mesh_device, model_args=model_args)

    from models.tt_transformers.tt.common import PagedAttentionConfig

    paged_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
    kv_cache_shape = (
        paged_config.max_num_blocks,
        model_args.n_kv_heads // mesh_device.get_num_devices(),
        paged_config.block_size,
        model_args.head_dim,
    )
    kv_cache = traced_executor.allocate_kv_cache(kv_cache_shape, torch.bfloat16, model_args.n_layers)

    prompts = load_input_prompts(batch_size)
    tokenizer = model_args.tokenizer
    encoded = [tokenizer.encode(p)[:128] for p in prompts]

    max_prompt_len = max(len(e) for e in encoded)
    input_tokens = torch.zeros(batch_size, max_prompt_len, dtype=torch.long)
    for i, enc in enumerate(encoded):
        input_tokens[i, : len(enc)] = torch.tensor(enc)

    bench = PerfBenchmarkExecutor(traced_executor)
    result = bench.run(
        tokens=input_tokens,
        kv_cache=kv_cache,
        num_decode_tokens=128,
        max_batch_size=model_args.max_batch_size,
        enable_trace=True,
    )

    logger.info(
        f"Performance — TTFT: {result.ttft_ms:.1f}ms, "
        f"tok/s/u: {result.tok_s_u:.1f}, "
        f"tok/s: {result.tok_s:.1f}, "
        f"decode latency: {result.decode_latency_mean_ms:.2f}ms"
    )

    if expected:
        targets = result.meets_target(expected, PERF_TOLERANCE)
        for metric, passed in targets.items():
            if not passed:
                logger.warning(
                    f"{metric} did not meet target: got {getattr(result, metric)}, expected {expected[metric]}"
                )
        if "tok_s_u" in expected:
            assert targets["tok_s_u"], f"tok/s/u {result.tok_s_u:.1f} below target {expected['tok_s_u']}"

    traced_executor.cleanup()
