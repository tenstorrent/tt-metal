# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Llama 3.1-8B Demo — accuracy and performance measurement.

Uses executors directly — no vLLM adapter needed.

Usage:
    # Token accuracy test
    MESH_DEVICE=N150 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python_env/bin/pytest models/common/tests/demos/llama3_8b/demo.py -k "token-accuracy" -v

    # Batch-1 latency test
    MESH_DEVICE=N150 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python_env/bin/pytest models/common/tests/demos/llama3_8b/demo.py -k "batch-1" -v

    # Batch-32 throughput test
    MESH_DEVICE=T3K HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python_env/bin/pytest models/common/tests/demos/llama3_8b/demo.py -k "batch-32" -v
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.models.executor import run_perf_benchmark, run_teacher_forcing
from models.common.models.llama3_8b.executor import EagerLlamaExecutor, TracedLlamaExecutor
from models.common.models.llama3_8b.hf_adaptor import from_pretrained
from models.common.models.llama3_8b.model import Llama31_8BPagedAttentionConfig
from models.common.sampling.sampling_params import SamplingParams
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.common.tests.demos.llama3_8b.demo_utils import load_input_prompts, preprocess_llama3_8b_chat_prompts

# =============================================================================
# Expected metrics
# =============================================================================

# Expected accuracy metrics from PERF.md for Llama-3.1-8B (top1, top5 only)
# Performance metrics (tok_s_u, ttft_ms) are collected dynamically by running
# simple_text_demo.py with the corresponding test case to get real baseline values.
EXPECTED_METRICS = {
    "performance": {
        "N150": {"top1": 90, "top5": 97, "tok_s_u": 28.3, "ttft_ms": 104},
        "N300": {"top1": 90, "top5": 97, "tok_s_u": 44.2, "ttft_ms": 67},
        "T3K": {"top1": 90, "top5": 98, "tok_s_u": 64.3, "ttft_ms": 53},
    },
    "accuracy": {
        "N150": {"top1": 96, "top5": 100, "tok_s_u": 25.2, "ttft_ms": 138},
        "N300": {"top1": 96, "top5": 100, "tok_s_u": 38.8, "ttft_ms": 79},
        "T3K": {"top1": 97, "top5": 100, "tok_s_u": 60.8, "ttft_ms": 81},
    },
}

PERF_TOLERANCE = 0.05
DEMO_DIR = Path(__file__).parent


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
    ref_path = DEMO_DIR / "reference_outputs" / f"{model_name}.refpt"
    if not ref_path.exists():
        pytest.skip(f"Reference file not found: {ref_path}")

    ref_data = torch.load(ref_path, map_location="cpu")
    reference_tokens = ref_data["reference_tokens"]
    top5_tokens = ref_data["top5_tokens"]
    return reference_tokens, top5_tokens


def log_generated_text(prompts, generated_token_ids, tokenizer):
    """Print the final generated continuation for each user."""
    logger.info("Finished decoding, printing the final outputs...\n")
    for user, output_ids in enumerate(generated_token_ids):
        prompt_text = prompts[user] if user < len(prompts) else ""
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        short_prompt = (
            prompt_text[:100] + "\n<long prompt not printed in full>\n" + prompt_text[-100:]
            if len(prompt_text) > 200
            else prompt_text
        )
        logger.info(f"\n==USER {user} - PROMPT\n{short_prompt}\n==USER {user} - OUTPUT\n{generated_text}\n")


def log_teacher_forcing_text(prompt_tokens, predicted_tokens_per_user, reference_tokens, tokenizer):
    """Print prompt, predicted continuation, and reference continuation for every teacher-forced user."""
    reference_text = tokenizer.decode(reference_tokens.tolist(), skip_special_tokens=True).strip()
    for user, user_prompt_tokens in enumerate(prompt_tokens):
        prompt_text = tokenizer.decode(user_prompt_tokens.tolist(), skip_special_tokens=True)
        predicted_text = tokenizer.decode(predicted_tokens_per_user[user], skip_special_tokens=True).strip()
        short_prompt = (
            prompt_text[:100] + "\n<long prompt not printed in full>\n" + prompt_text[-100:]
            if len(prompt_text) > 200
            else prompt_text
        )
        logger.info(
            f"\n==USER {user} - PROMPT\n{short_prompt}\n==USER {user} - OUTPUT\n{predicted_text}\n==USER {user} - REFERENCE\n{reference_text}\n"
        )


def create_llama3_for_causal_lm(mesh_device, optimizations="performance", max_batch_size=32):
    """Create product-level Llama3ForCausalLM for testing."""
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    instruct = "Instruct" in hf_model

    max_seq_len = 1024

    block_size = 32
    max_num_blocks = 1024
    paged_attention_config = Llama31_8BPagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    return from_pretrained(
        mesh_device=mesh_device,
        hf_model=hf_model,
        instruct=instruct,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        optimizations=optimizations,
        dtype=ttnn.bfloat8_b,
        paged_attention_config=paged_attention_config,
    )


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
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        {"mesh_shape": (1, 1), "trace_region_size": 50000000, "num_command_queues": 1},
        {"mesh_shape": (1, 2), "trace_region_size": 50000000, "num_command_queues": 1},
        {"mesh_shape": (1, 8), "trace_region_size": 50000000, "num_command_queues": 1},
    ],
    ids=[
        "1x1",
        "1x2",
        "1x8",
    ],
    indirect=True,
)
@pytest.mark.parametrize("optimizations", ["performance", "accuracy"])
def test_llama3_8b(test_config, ttnn_mesh_device, optimizations):
    """Main test function for TTTv2 Llama 3.1-8B."""
    device_name = get_device_name(ttnn_mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    llm = None

    try:
        batch_size = 32 if test_config == "batch-32" else 1
        llm = create_llama3_for_causal_lm(ttnn_mesh_device, optimizations, max_batch_size=batch_size)

        if test_config == "token-accuracy":
            _run_token_accuracy(llm, ttnn_mesh_device, expected)
        elif test_config == "batch-1":
            _run_perf_benchmark(
                llm,
                ttnn_mesh_device,
                expected,
                batch_size=1,
                case_name=f"{optimizations}/{test_config}",
            )
        elif test_config == "batch-32":
            _run_perf_benchmark(
                llm,
                ttnn_mesh_device,
                expected,
                batch_size=32,
                case_name=f"{optimizations}/{test_config}",
            )
    finally:
        cleanup_model_case(llm.model if llm is not None else None, ttnn_mesh_device)


# =============================================================================
# Token accuracy
# =============================================================================


def _attention_config(model):
    return model.config.block_configs[0].attention_config


def _run_token_accuracy(llm, mesh_device, expected):
    """Run teacher-forcing token accuracy test."""
    model = llm.model
    model_config = model.config
    runtime_config = llm.runtime_config
    attention_config = _attention_config(model)
    model_name = llm.model_name
    reference_tokens, top5_tokens = load_reference_data(model_name)

    # Ensure reference_tokens is 1D for slicing
    if reference_tokens.dim() > 1:
        reference_tokens = reference_tokens.squeeze()

    half = len(reference_tokens) // 2
    prompt_tokens = reference_tokens[:half].unsqueeze(0)  # [1, half]

    executor = EagerLlamaExecutor(model, mesh_device, model_args=runtime_config)

    max_batch_size = model_config.max_batch_size
    prompt_tokens = prompt_tokens.repeat(max_batch_size, 1)
    block_size = 32
    max_num_blocks = runtime_config.paged_attention_config.max_num_blocks
    max_num_blocks_per_user = max_num_blocks // max_batch_size

    kv_cache_shape = (
        max_num_blocks,
        attention_config.n_kv_heads // mesh_device.get_num_devices(),
        block_size,
        attention_config.head_dim,
    )
    kv_cache = executor.allocate_kv_cache(kv_cache_shape, torch.bfloat16, model_config.n_layers)
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(max_batch_size, max_num_blocks_per_user)

    target_top5 = top5_tokens[half - 1 :] if top5_tokens.shape[0] < len(reference_tokens) else top5_tokens[half:]
    result = run_teacher_forcing(
        executor,
        prompt_tokens=prompt_tokens,
        reference_tokens=reference_tokens,
        top5_tokens=target_top5,
        kv_cache=kv_cache,
        page_table=page_table,
        max_batch_size=max_batch_size,
    )

    top1 = result.top1_accuracy() * 100
    top5 = result.top5_accuracy() * 100

    logger.info(f"Token accuracy — top1: {top1:.1f}%, top5: {top5:.1f}%")
    log_teacher_forcing_text(prompt_tokens, result.predicted_tokens_per_user, reference_tokens[half:], llm.tokenizer)

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


def _run_perf_benchmark(llm, mesh_device, expected, batch_size, case_name):
    """Run performance benchmark (TTFT + tok/s/u)."""
    model = llm.model
    model_config = model.config
    runtime_config = llm.runtime_config
    attention_config = _attention_config(model)
    traced_executor = TracedLlamaExecutor(model, mesh_device, model_args=runtime_config)
    try:
        block_size = 32
        max_batch_size = model_config.max_batch_size
        max_num_blocks = runtime_config.paged_attention_config.max_num_blocks
        max_num_blocks_per_user = max_num_blocks // max_batch_size

        kv_cache_shape = (
            max_num_blocks,
            attention_config.n_kv_heads // mesh_device.get_num_devices(),
            block_size,
            attention_config.head_dim,
        )
        kv_cache = traced_executor.allocate_kv_cache(kv_cache_shape, torch.bfloat16, model_config.n_layers)
        page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(max_batch_size, max_num_blocks_per_user)

        prompts_path = DEMO_DIR / "sample_prompts" / "input_data_questions_prefill_128.json"
        prompts = load_input_prompts(prompts_path, batch_size)
        tokenizer = llm.tokenizer
        input_tokens, prompt_lens = preprocess_llama3_8b_chat_prompts(
            prompts,
            llm,
            reserve_decode_tokens=128,
        )

        sampling_mode = os.environ.get("SAMPLING_MODE", "on_device_topk").lower()
        on_device_params = {
            "on_device": SamplingParams(temperature=0.0, top_k=1, top_p=0.0),
            "on_device_topk": SamplingParams(temperature=0.0, top_k=32, top_p=0.08),
        }
        sampling_params = (
            on_device_params[sampling_mode]
            if sampling_mode in on_device_params and getattr(model, "supports_on_device_sampling", False)
            else None
        )
        logger.info(f"[{case_name}] SAMPLING_MODE={sampling_mode} -> sampling_params={sampling_params}")

        result = run_perf_benchmark(
            traced_executor,
            tokens=input_tokens,
            kv_cache=kv_cache,
            page_table=page_table,
            num_decode_tokens=128,
            max_batch_size=max_batch_size,
            prompt_lens=prompt_lens,
            sampling_params=sampling_params,
        )
        log_generated_text(prompts, result.generated_token_ids, tokenizer)

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
            failures = []
            if "tok_s_u" in expected and not targets["tok_s_u"]:
                failures.append(f"tok/s/u {result.tok_s_u:.1f} below target {expected['tok_s_u']}")
            if "ttft_ms" in expected and not targets["ttft_ms"]:
                failures.append(f"ttft_ms {result.ttft_ms:.1f} above target {expected['ttft_ms']}")
            assert not failures, f"{case_name}: " + "; ".join(failures)
    finally:
        traced_executor.cleanup()
