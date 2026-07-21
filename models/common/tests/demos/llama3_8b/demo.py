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
from models.common.device_utils import get_device_name
from models.common.llm_runtime.config import LLMExecutorConfig, PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.models.llama3_8b.executor import build_llama3_executor
from models.common.models.llama3_8b.hf_adaptor import from_pretrained
from models.common.models.llama3_8b.model import Llama31_8BPagedAttentionConfig
from models.common.sampling.sampling_params import SamplingParams
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.common.tests.demos.llama3_8b.demo_utils import load_input_prompts, preprocess_llama3_8b_chat_prompts
from models.common.tests.demos.run_helpers import run_perf_benchmark, run_teacher_forcing

# =============================================================================
# Expected metrics
# =============================================================================

# Expected accuracy metrics from measuring TTTv1 for Llama-3.1-8B (top1, top5 only).
# Decode-throughput targets are measured TTTv1 parity numbers from the old tt_transformers demo
# sweep recorded in consolidated_git_status_markdown.md. T3K batch-1 TTFT uses comparable
# simple_text_demo measurements; batch-32 TTFT uses the corresponding batch-1 guardrail until
# we have direct batch-32 wall-clock baselines.
EXPECTED_METRICS = {
    "performance": {
        "N150": {
            "top1": 90,
            "top5": 97,
            "batch-1": {"tok_s_u": 9.49, "ttft_ms": 177.1},
            "batch-32": {"tok_s_u": 8.81, "ttft_ms": 177.1},
        },
        "N300": {
            "top1": 90,
            "top5": 97,
            "batch-1": {"tok_s_u": 25.4, "ttft_ms": 90.4},
            "batch-32": {"tok_s_u": 22.2, "ttft_ms": 90.4},
        },
        "T3K": {
            "top1": 90,
            "top5": 98,
            "batch-1": {"tok_s_u": 70.3, "ttft_ms": 43.1},
            "batch-32": {"tok_s_u": 56.1, "ttft_ms": 39.9},
        },
    },
    "accuracy": {
        "N150": {
            "top1": 96,
            "top5": 100,
            "batch-1": {"tok_s_u": 9.11, "ttft_ms": 206.8},
            "batch-32": {"tok_s_u": 8.49, "ttft_ms": 206.8},
        },
        "N300": {
            "top1": 96,
            "top5": 100,
            "batch-1": {"tok_s_u": 23.4, "ttft_ms": 96.3},
            "batch-32": {"tok_s_u": 20.6, "ttft_ms": 96.3},
        },
        "T3K": {
            "top1": 97,
            "top5": 100,
            "batch-1": {"tok_s_u": 64.4, "ttft_ms": 46.04},
            "batch-32": {"tok_s_u": 52.2, "ttft_ms": 41.9},
        },
    },
}

PERF_TOLERANCE = 0.05
DEMO_DIR = Path(__file__).parent


# =============================================================================
# Helpers
# =============================================================================


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
    n_layers = int(os.environ.get("LLAMA3_8B_TTTV2_NUM_LAYERS", "32"))

    block_size = 32
    max_num_blocks = 1024
    paged_attention_config = Llama31_8BPagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    return from_pretrained(
        mesh_device=mesh_device,
        hf_model=hf_model,
        instruct=instruct,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        n_layers=n_layers,
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
                _expected_for_case(expected, test_config),
                batch_size=1,
                case_name=f"{optimizations}/{test_config}",
            )
        elif test_config == "batch-32":
            _run_perf_benchmark(
                llm,
                ttnn_mesh_device,
                _expected_for_case(expected, test_config),
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


def _build_demo_executor(llm, *, trace_mode, device_sampling_enabled, include_decode_top_k=False):
    attention_config = _attention_config(llm.model)
    paged_attention_config = attention_config.paged_attention_config
    config = LLMExecutorConfig(
        trace=TraceConfig(mode=trace_mode),
        warmup=WarmupConfig(include_decode_top_k=include_decode_top_k),
        paged_kv_cache=PagedKVCacheConfig(
            block_size=int(paged_attention_config.block_size),
            max_num_blocks=int(paged_attention_config.max_num_blocks),
            num_blocks=int(paged_attention_config.max_num_blocks),
            dtype=attention_config.kv_cache_dtype,
        ),
        device_sampling_enabled=device_sampling_enabled,
    )
    return build_llama3_executor(llm, config)


def _warmup_demo_executor(executor, *, kv_cache, page_table):
    can_sample_on_device = executor.config.device_sampling_enabled
    prefill_kwargs = {
        "kv_cache": kv_cache,
        "can_sample_on_device": can_sample_on_device,
    }
    decode_kwargs = {
        "kv_cache": kv_cache,
        "max_batch_size": int(executor.model.config.max_batch_size),
        "num_blocks": int(page_table.shape[-1]),
        "can_sample_on_device": can_sample_on_device,
    }

    # Compile both graph families before capturing either trace so trace plans
    # never depend on which warmup happens to run first.
    executor.warmup_model_prefill(enable_trace=False, **prefill_kwargs)
    executor.warmup_model_decode(enable_trace=False, **decode_kwargs)

    if executor.config.trace.prefill_enabled:
        executor.already_warmed_up_prefill = False
        executor.warmup_model_prefill(enable_trace=True, **prefill_kwargs)
    if executor.config.trace.decode_enabled:
        executor.warmup_model_decode(enable_trace=True, **decode_kwargs)


def _expected_for_case(expected, test_config):
    """Merge per-case performance targets into the device-level expectations."""
    case_expected = expected.get(test_config)
    if case_expected is None:
        return expected
    return {**expected, **case_expected}


def _run_token_accuracy(llm, mesh_device, expected):
    """Run teacher-forcing token accuracy test."""
    top1, top5 = _measure_teacher_forcing_accuracy(llm, mesh_device, log_text=True)

    if "top1" in expected:
        assert top1 >= expected["top1"] * (
            1 - PERF_TOLERANCE
        ), f"Top-1 accuracy {top1:.1f}% below threshold {expected['top1']}%"
    if "top5" in expected:
        assert top5 >= expected["top5"] * (
            1 - PERF_TOLERANCE
        ), f"Top-5 accuracy {top5:.1f}% below threshold {expected['top5']}%"


def _measure_teacher_forcing_accuracy(llm, mesh_device, *, log_text=False):
    """Run teacher forcing and return top-1/top-5 percentages."""
    model = llm.model
    model_config = model.config
    model_name = llm.model_name
    reference_tokens, top5_tokens = load_reference_data(model_name)

    # Ensure reference_tokens is 1D for slicing
    if reference_tokens.dim() > 1:
        reference_tokens = reference_tokens.squeeze()

    half = len(reference_tokens) // 2
    prompt_tokens = reference_tokens[:half].unsqueeze(0)  # [1, half]

    max_batch_size = model_config.max_batch_size
    prompt_tokens = prompt_tokens.repeat(max_batch_size, 1)
    executor = _build_demo_executor(
        llm,
        trace_mode="none",
        device_sampling_enabled=False,
        include_decode_top_k=False,
    )
    try:
        kv_cache = executor.allocate_kv_cache()
        max_num_blocks = executor.paged_kv_cache_config.num_blocks
        max_num_blocks_per_user = max_num_blocks // max_batch_size
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
    finally:
        executor.cleanup()

    top1 = result.top1_accuracy() * 100
    top5 = result.top5_accuracy() * 100

    logger.info(f"Token accuracy — top1: {top1:.1f}%, top5: {top5:.1f}%")
    if log_text:
        log_teacher_forcing_text(
            prompt_tokens, result.predicted_tokens_per_user, reference_tokens[half:], llm.tokenizer
        )

    return top1, top5


# =============================================================================
# Performance benchmark
# =============================================================================


def _run_perf_benchmark(llm, mesh_device, expected, batch_size, case_name):
    """Run performance benchmark (TTFT + tok/s/u)."""
    model = llm.model
    model_config = model.config
    prompts_path = DEMO_DIR / "sample_prompts" / "input_data_questions_prefill_128.json"
    prompts = load_input_prompts(prompts_path, batch_size)
    tokenizer = llm.tokenizer
    num_decode_tokens = int(os.environ.get("LLAMA3_8B_TTTV2_DECODE_TOKENS", "200"))
    input_tokens, prompt_lens = preprocess_llama3_8b_chat_prompts(
        prompts,
        llm,
        reserve_decode_tokens=num_decode_tokens,
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
    pipeline_readback = os.environ.get("PIPELINE_READBACK", "1").lower() not in ("0", "false", "no")
    logger.info(f"[{case_name}] PIPELINE_READBACK={pipeline_readback}")

    executor = None
    try:
        executor = _build_demo_executor(
            llm,
            trace_mode="all",
            device_sampling_enabled=sampling_params is not None,
            include_decode_top_k=sampling_params is not None and sampling_mode == "on_device_topk",
        )
        kv_cache = executor.allocate_kv_cache()
        max_batch_size = model_config.max_batch_size
        max_num_blocks = executor.paged_kv_cache_config.num_blocks
        max_num_blocks_per_user = max_num_blocks // max_batch_size
        page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(max_batch_size, max_num_blocks_per_user)
        _warmup_demo_executor(executor, kv_cache=kv_cache, page_table=page_table)

        result = run_perf_benchmark(
            executor,
            tokens=input_tokens,
            kv_cache=kv_cache,
            page_table=page_table,
            num_decode_tokens=num_decode_tokens,
            max_batch_size=max_batch_size,
            prompt_lens=prompt_lens,
            sampling_params=sampling_params,
            pipeline_readback=pipeline_readback,
        )
        logger.info(
            f"Performance — TTFT: {result.ttft_ms:.1f}ms, "
            f"tok/s/u: {result.tok_s_u:.1f}, "
            f"tok/s: {result.tok_s:.1f}, "
            f"decode latency: {result.decode_latency_mean_ms:.2f}ms"
        )
        log_generated_text(prompts, result.generated_token_ids, tokenizer)
    finally:
        if executor is not None:
            executor.cleanup()

    skip_perf_teacher_forcing = os.environ.get("LLAMA3_8B_TTTV2_SKIP_PERF_TEACHER_FORCING", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if skip_perf_teacher_forcing:
        logger.info("Skipping performance-side teacher forcing because profiling requested decode-only execution")
    else:
        top1, top5 = _measure_teacher_forcing_accuracy(llm, mesh_device, log_text=False)
        logger.info(f"Performance-side teacher forcing — top1: {top1:.1f}%, top5: {top5:.1f}%")

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
