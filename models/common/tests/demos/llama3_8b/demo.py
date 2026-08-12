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

import math
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.device_utils import get_device_name
from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.llm_runtime.lane_group import LaneGroupExecutor
from models.common.models.llama3_8b.executor import Llama3ExecutorConfig, build_llama3_executor
from models.common.models.llama3_8b.hf_adaptor import from_pretrained
from models.common.models.llama3_8b.model import Llama31_8BPagedAttentionConfig
from models.common.sampling.sampling_params import SamplingParams
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.common.tests.demos.llama3_8b.demo_utils import load_input_prompts, preprocess_llama3_8b_chat_prompts
from models.common.tests.demos.run_helpers import (
    PerfBenchmarkResult,
    assert_no_special_tokens,
    run_perf_benchmark,
    run_teacher_forcing,
)
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.demos.utils.model_targets import resolve_accuracy_targets
from models.demos.utils.trace_region_sizes import hf_model_name_candidates, resolve_trace_region_size
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.generator import create_submeshes

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


def _benchmark_model_identity(hf_model: str, fallback_model_name: str) -> tuple[str, str]:
    """Return TTTv1-compatible base identity plus a stable model variant."""
    canonical_model = next(
        (
            candidate
            for candidate in hf_model_name_candidates(hf_model)
            if "/" in candidate and not Path(candidate).is_absolute() and not Path(candidate).exists()
        ),
        fallback_model_name,
    )
    model_variant = Path(canonical_model).name
    instruct_suffix = "-Instruct"
    base_model = (
        model_variant[: -len(instruct_suffix)]
        if model_variant.lower().endswith(instruct_suffix.lower())
        else model_variant
    )
    return base_model, model_variant


@dataclass(frozen=True)
class DemoCase:
    name: str
    batch_size: int
    max_seq_len: int
    num_decode_tokens: int
    data_parallel: int = 1
    performance_case: str | None = None
    repeat_batches: int = 1
    use_prefetcher: bool = False
    report_perf: bool = False


DEMO_CASES = {
    "token-accuracy": DemoCase("token-accuracy", batch_size=1, max_seq_len=1024, num_decode_tokens=0),
    "batch-1": DemoCase(
        "batch-1",
        batch_size=1,
        max_seq_len=1024,
        num_decode_tokens=200,
        performance_case="batch-1",
    ),
    "batch-32": DemoCase(
        "batch-32",
        batch_size=32,
        max_seq_len=1024,
        num_decode_tokens=200,
        performance_case="batch-32",
    ),
    "batch-32-ci": DemoCase(
        "batch-32-ci",
        batch_size=32,
        max_seq_len=2048,
        num_decode_tokens=1024,
        performance_case="batch-32-ci",
    ),
    "eval-32-repeat-3": DemoCase(
        "eval-32",
        batch_size=32,
        max_seq_len=1024,
        num_decode_tokens=200,
        repeat_batches=3,
    ),
    "eval-32-repeat-1": DemoCase(
        "eval-32",
        batch_size=32,
        max_seq_len=1024,
        num_decode_tokens=200,
        performance_case="eval-32",
        repeat_batches=1,
        report_perf=True,
    ),
    "ci-b1-DP-2": DemoCase("ci-b1-DP-2", batch_size=2, max_seq_len=1024, num_decode_tokens=200, data_parallel=2),
    "ci-b1-DP-4": DemoCase("ci-b1-DP-4", batch_size=4, max_seq_len=4096, num_decode_tokens=2048, data_parallel=4),
    "ci-b1-DP-8": DemoCase("ci-b1-DP-8", batch_size=8, max_seq_len=4096, num_decode_tokens=2048, data_parallel=8),
    "ci-b1-DP-16": DemoCase("ci-b1-DP-16", batch_size=16, max_seq_len=1024, num_decode_tokens=200, data_parallel=16),
    "ci-b1-DP-32": DemoCase("ci-b1-DP-32", batch_size=32, max_seq_len=1024, num_decode_tokens=200, data_parallel=32),
}


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
    metadata = ref_data.get("metadata", {}) if isinstance(ref_data, dict) else {}
    prompt_len = ref_data.get("prompt_len") if isinstance(ref_data, dict) else None
    if prompt_len is None and isinstance(metadata, dict):
        prompt_len = metadata.get("prompt_len")
    return reference_tokens, top5_tokens, prompt_len, metadata


def _resolve_llama_head_counts(hf_model: str | None = None) -> tuple[int, int]:
    hf_model = hf_model or os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    try:
        config = AutoConfig.from_pretrained(hf_model, local_files_only=os.getenv("CI") == "true")
    except OSError:
        if hf_model.rstrip("/").split("/")[-1] == "Llama-3.1-8B-Instruct":
            return 32, 8
        raise
    text_config = getattr(config, "text_config", config)
    return int(text_config.num_attention_heads), int(text_config.num_key_value_heads)


def _validate_tp_topology(mesh_device, *, num_devices: int | None = None) -> None:
    num_devices = mesh_device.get_num_devices() if num_devices is None else int(num_devices)
    n_heads, n_kv_heads = _resolve_llama_head_counts()
    assert n_heads % num_devices == 0, f"n_heads={n_heads} must be divisible by num_devices={num_devices}"
    assert n_kv_heads % num_devices == 0, f"n_kv_heads={n_kv_heads} must be divisible by num_devices={num_devices}"


def _skip_unsupported_case(case: DemoCase, mesh_device) -> None:
    device_name = get_device_name(mesh_device)
    if case.use_prefetcher:
        pytest.skip("TTTv2 does not support the TTTv1 DRAM prefetcher")
    expected_repeat_batches = 1 if case.report_perf or case.name != "eval-32" else 3
    if case.repeat_batches != expected_repeat_batches:
        pytest.skip(f"{case.name} requires repeat_batches={expected_repeat_batches}; got {case.repeat_batches}")
    if case.name == "batch-32-ci" and device_name == "N150":
        pytest.skip("batch-32-ci max_seq_len=2048 capacity is not enabled for N150 until verified")
    if case.data_parallel > 1:
        num_devices = mesh_device.get_num_devices()
        if num_devices % case.data_parallel != 0:
            pytest.skip(f"{case.name} requires device count divisible by DP={case.data_parallel}; got {num_devices}")
        per_lane_devices = num_devices // case.data_parallel
        _validate_tp_topology(mesh_device, num_devices=per_lane_devices)


def _sampling_params_for_model(model, *, case_name: str):
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
    return sampling_mode, sampling_params


def _prefill_sampling_params(model, sampling_params):
    if sampling_params is not None and model.config.num_devices > 1:
        logger.info("Using host argmax for multi-device prefill; decode sampling remains on-device.")
        return None
    return sampling_params


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


def create_llama3_for_causal_lm(mesh_device, optimizations="performance", max_batch_size=32, max_seq_len=1024):
    """Create product-level Llama3ForCausalLM for testing."""
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    instruct = "Instruct" in hf_model

    n_layers = int(os.environ.get("LLAMA3_8B_TTTV2_NUM_LAYERS", "32"))

    block_size = 32
    max_num_blocks = max_batch_size * math.ceil(max_seq_len / block_size)
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


mesh_device_name = os.environ.get("MESH_DEVICE", "").strip().upper()
mesh_device_shape = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (4, 8)}.get(mesh_device_name)
if mesh_device_shape is None:
    pytest.skip(
        f"Unsupported MESH_DEVICE={mesh_device_name!r}; use N150, N300, T3K, or TG.",
        allow_module_level=True,
    )
ttnn_mesh_device_params = {
    "mesh_shape": mesh_device_shape,
    "trace_region_size": resolve_trace_region_size("llama3.1-8b", mesh_device_name),
    "num_command_queues": 1,
}
pytestmark = pytest.mark.parametrize(
    "ttnn_mesh_device",
    [ttnn_mesh_device_params],
    indirect=True,
    ids=[mesh_device_name],
)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param("token-accuracy", id="token-accuracy-repeat_batch-1-prefetcher-off"),
        "batch-1",
        pytest.param("batch-32", id="batch-32-repeat_batch-1-prefetcher-off"),
        "batch-32-ci",
        pytest.param(
            "eval-32-repeat-3",
            id="eval-32-repeat_batch-3-prefetcher-off-perf-report-off",
        ),
        pytest.param(
            "eval-32-repeat-1",
            id="eval-32-repeat_batch-1-prefetcher-off-perf-report-on",
        ),
        "ci-b1-DP-2",
        "ci-b1-DP-4",
        "ci-b1-DP-8",
        "ci-b1-DP-16",
        "ci-b1-DP-32",
    ],
)
@pytest.mark.parametrize("optimizations", ["performance", "accuracy"])
@pytest.mark.usefixtures("silicon_arch_name")
def test_llama3_8b(test_config, ttnn_mesh_device, optimizations):
    """Main test function for TTTv2 Llama 3.1-8B."""
    mesh_device = ttnn_mesh_device
    case = DEMO_CASES[test_config]
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    llm = None

    try:
        _skip_unsupported_case(case, mesh_device)

        if case.data_parallel > 1:
            _run_dp_smoke(mesh_device, optimizations, case)
            return

        _validate_tp_topology(mesh_device)
        llm = create_llama3_for_causal_lm(
            mesh_device,
            optimizations,
            max_batch_size=case.batch_size,
            max_seq_len=case.max_seq_len,
        )

        if case.name == "token-accuracy":
            _run_token_accuracy(llm, mesh_device, expected, optimizations)
        elif case.name in ("batch-1", "batch-32", "batch-32-ci"):
            _run_perf_benchmark(
                llm,
                mesh_device,
                _expected_for_case(expected, case.performance_case),
                batch_size=case.batch_size,
                case_name=f"{optimizations}/{case.name}",
                num_decode_tokens=case.num_decode_tokens,
            )
        elif case.name == "eval-32":
            profiler = BenchmarkProfiler() if case.report_perf else None
            reported_batch = _run_eval_repeat_batches(
                llm,
                batch_size=case.batch_size,
                repeat_batches=case.repeat_batches,
                num_decode_tokens=case.num_decode_tokens,
                profiler=profiler,
            )
            if case.report_perf:
                result, prompt_lens, sampling_mode, prompts = reported_batch
                _report_performance(
                    llm,
                    mesh_device,
                    _expected_for_case(expected, case.performance_case),
                    prompts=prompts,
                    case_name=f"{optimizations}/{case.name}",
                    profiler=profiler,
                    result=result,
                    prompt_lens=prompt_lens,
                    sampling_mode=sampling_mode,
                )
    finally:
        cleanup_model_case(llm.model if llm is not None else None, mesh_device)


# =============================================================================
# Token accuracy
# =============================================================================


def _attention_config(model):
    return model.config.block_configs[0].attention_config


def _build_demo_executor(llm, *, trace_mode, device_sampling_enabled, include_decode_top_k=False):
    attention_config = _attention_config(llm.model)
    paged_attention_config = attention_config.paged_attention_config
    config = Llama3ExecutorConfig(
        trace=TraceConfig(mode=trace_mode),
        warmup=WarmupConfig(include_decode_top_k=include_decode_top_k),
        paged_kv_cache=PagedKVCacheConfig(
            block_size=int(paged_attention_config.block_size),
            max_num_blocks=int(paged_attention_config.max_num_blocks),
            # Unlike vLLM, the direct demo has no later scheduler-selected
            # physical capacity. Resolve num_blocks to the configured maximum
            # now; PageTableLayout is final at executor construction and the
            # subsequent KV allocation intentionally materializes this maximum.
            num_blocks=int(paged_attention_config.max_num_blocks),
            dtype=attention_config.kv_cache_dtype,
        ),
        device_sampling_enabled=device_sampling_enabled,
    )
    return build_llama3_executor(llm, config)


def _force_decode_top_k(sampling_mode, sampling_params, num_devices):
    return sampling_params is not None and sampling_mode == "on_device_topk" and int(num_devices) == 8


def _warmup_demo_executor(executor, *, kv_cache, page_table, prefill_can_sample_on_device=None):
    config = getattr(executor, "config", None)
    if config is None:
        config = executor.lanes[0].config
    can_sample_on_device = config.device_sampling_enabled
    if prefill_can_sample_on_device is None:
        prefill_can_sample_on_device = can_sample_on_device
    max_batch_size = getattr(executor, "max_batch_size", None)
    if max_batch_size is None:
        max_batch_size = int(executor.model.config.max_batch_size)
    prefill_kwargs = {
        "kv_cache": kv_cache,
        "can_sample_on_device": bool(prefill_can_sample_on_device),
    }
    decode_kwargs = {
        "kv_cache": kv_cache,
        "max_batch_size": int(max_batch_size),
        "num_blocks": int(page_table.shape[-1]),
        "can_sample_on_device": can_sample_on_device,
    }

    # Compile both graph families before capturing either trace so trace plans
    # never depend on which warmup happens to run first.
    executor.warmup_model_prefill(enable_trace=False, **prefill_kwargs)
    executor.warmup_model_decode(enable_trace=False, **decode_kwargs)

    if config.trace.prefill_enabled:
        executor.warmup_model_prefill(enable_trace=True, **prefill_kwargs)
    if config.trace.decode_enabled:
        executor.warmup_model_decode(enable_trace=True, **decode_kwargs)


def _expected_for_case(expected, test_config):
    """Return a complete secondary in-test performance gate for one case."""
    if test_config is None:
        return None
    case_expected = expected.get(test_config)
    missing_metrics = {"tok_s_u", "ttft_ms"} - set(case_expected or {})
    if missing_metrics:
        logger.warning(
            f"No complete in-test performance gate for {test_config}; "
            f"missing {', '.join(sorted(missing_metrics))}. "
            "Centralized post-run validation remains authoritative."
        )
        return None
    return {metric: case_expected[metric] for metric in ("tok_s_u", "ttft_ms")}


def _run_token_accuracy(llm, mesh_device, expected, optimizations: str):
    """Run teacher-forcing token accuracy test."""
    top1, top5, prompt_len = _measure_teacher_forcing_accuracy(
        llm, mesh_device, optimizations=optimizations, log_text=True
    )

    if os.environ.get("CI") == "true":
        hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        model_target, _ = _benchmark_model_identity(hf_model, llm.model_name)
        central = resolve_accuracy_targets(
            model_target,
            get_device_name(mesh_device),
            batch_size=1,
            seq_len=prompt_len,
        )
        if not central or "top1" not in central or "top5" not in central:
            raise ValueError(
                f"No centralized accuracy target for {model_target} on {get_device_name(mesh_device)} "
                f"(batch_size=1, seq_len={prompt_len}); add an active entry to models/model_targets.yaml."
            )
        expected = {"top1": float(central["top1"]) - 0.5, "top5": float(central["top5"]) - 0.5}

    if "top1" in expected:
        measured_top1 = math.ceil(top1)
        assert (
            measured_top1 >= expected["top1"]
        ), f"Top-1 accuracy {top1:.1f}% (ceil {measured_top1}) below threshold {expected['top1']:.1f}%"
    if "top5" in expected:
        measured_top5 = math.ceil(top5)
        assert (
            measured_top5 >= expected["top5"]
        ), f"Top-5 accuracy {top5:.1f}% (ceil {measured_top5}) below threshold {expected['top5']:.1f}%"


def _measure_teacher_forcing_accuracy(llm, mesh_device, *, optimizations: str, log_text=False):
    """Run teacher forcing and return top-1/top-5 percentages."""
    model = llm.model
    model_config = model.config
    model_name = llm.model_name
    reference_tokens, top5_tokens, prompt_len, metadata = load_reference_data(model_name)

    # Ensure reference_tokens is 1D for slicing
    if reference_tokens.dim() > 1:
        reference_tokens = reference_tokens.squeeze()

    if prompt_len is None:
        prompt_len = len(reference_tokens) // 2
        logger.info(f"Reference missing prompt_len metadata; using legacy half-split={prompt_len}.")
    else:
        prompt_len = int(prompt_len)
        logger.info(f"Using reference prompt_len metadata={prompt_len}.")
    if metadata:
        logger.info(f"Reference metadata: {metadata}")

    prompt_tokens = reference_tokens[:prompt_len].unsqueeze(0)

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

        target_top5 = (
            top5_tokens[prompt_len - 1 :] if top5_tokens.shape[0] < len(reference_tokens) else top5_tokens[prompt_len:]
        )
        profiler = BenchmarkProfiler()
        profiler.start("run")
        result = run_teacher_forcing(
            executor,
            prompt_tokens=prompt_tokens,
            reference_tokens=reference_tokens,
            top5_tokens=target_top5,
            kv_cache=kv_cache,
            page_table=page_table,
            max_batch_size=max_batch_size,
            profiler=profiler,
        )
        profiler.end("run")
    finally:
        executor.cleanup()

    top1 = result.top1_accuracy() * 100
    top5 = result.top5_accuracy() * 100

    logger.info(f"Token accuracy — top1: {top1:.1f}%, top5: {top5:.1f}%")
    if log_text:
        log_teacher_forcing_text(
            prompt_tokens, result.predicted_tokens_per_user, reference_tokens[prompt_len:], llm.tokenizer
        )

    if os.environ.get("CI") == "true":
        hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        model_target, model_variant = _benchmark_model_identity(hf_model, llm.model_name)
        num_target = len(reference_tokens) - prompt_len
        measurements = {
            "prefill_t/s": result.prefill_tok_s,
            "prefill_time_to_token": result.prefill_time_to_token_s,
            "decode_t/s": result.decode_tok_s,
            "decode_t/s/u": result.decode_tok_s_u,
        }
        benchmark_data = create_benchmark_data(
            profiler, measurements, {"inference_prefill": 0, "inference_decode": 1}, targets={}
        )
        benchmark_data.add_measurement(profiler, 0, "inference_decode", "top1_token_accuracy", top1, target=None)
        benchmark_data.add_measurement(profiler, 0, "inference_decode", "top5_token_accuracy", top5, target=None)
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo_accuracy",
            ml_model_name=model_target,
            ml_model_type="llm",
            device_name=get_device_name(mesh_device),
            num_layers=len(model_config.block_configs),
            batch_size=1,
            config_params={
                "model_variant": model_variant,
                "optimization_profile": optimizations,
                "workload": "token-accuracy",
            },
            input_sequence_length=prompt_len,
            output_sequence_length=num_target,
        )

    return top1, top5, prompt_len


# =============================================================================
# Performance benchmark
# =============================================================================


def _run_batch_once(
    llm,
    prompts: list[str],
    *,
    case_name: str,
    num_decode_tokens: int,
    profiler=None,
) -> tuple[PerfBenchmarkResult, torch.Tensor, str]:
    """Run one warmed-up batch and return its result and reporting metadata."""
    model = llm.model
    model_config = model.config
    input_tokens, prompt_lens = preprocess_llama3_8b_chat_prompts(
        prompts,
        llm,
        reserve_decode_tokens=num_decode_tokens,
    )

    sampling_mode, sampling_params = _sampling_params_for_model(model, case_name=case_name)
    pipeline_readback = os.environ.get("PIPELINE_READBACK", "1").lower() not in ("0", "false", "no")
    logger.info(f"[{case_name}] PIPELINE_READBACK={pipeline_readback}")

    executor = None
    result = None
    try:
        executor = _build_demo_executor(
            llm,
            trace_mode="all",
            device_sampling_enabled=sampling_params is not None,
            include_decode_top_k=_force_decode_top_k(
                sampling_mode,
                sampling_params,
                model_config.num_devices,
            ),
        )
        kv_cache = executor.allocate_kv_cache()
        max_batch_size = model_config.max_batch_size
        max_num_blocks = executor.paged_kv_cache_config.num_blocks
        max_num_blocks_per_user = max_num_blocks // max_batch_size
        page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(max_batch_size, max_num_blocks_per_user)
        _warmup_demo_executor(executor, kv_cache=kv_cache, page_table=page_table)

        if profiler is not None:
            profiler.start("run")
        try:
            result = run_perf_benchmark(
                executor,
                tokens=input_tokens,
                kv_cache=kv_cache,
                page_table=page_table,
                num_decode_tokens=num_decode_tokens,
                max_batch_size=max_batch_size,
                prompt_lens=prompt_lens,
                sampling_params=sampling_params,
                prefill_sampling_params=_prefill_sampling_params(model, sampling_params),
                pipeline_readback=pipeline_readback,
                profiler=profiler,
            )
        finally:
            if profiler is not None:
                profiler.end("run")
        assert_no_special_tokens(result.generated_token_ids, llm.tokenizer, case_name=case_name)
        return result, prompt_lens, sampling_mode
    finally:
        if executor is not None:
            executor.cleanup()


def _report_performance(
    llm,
    mesh_device,
    expected,
    *,
    prompts,
    case_name,
    profiler,
    result,
    prompt_lens,
    sampling_mode,
    log_text=True,
    data_parallel=1,
) -> None:
    """Log and persist one run, applying gates only when ``expected`` is non-empty."""
    model_config = llm.model.config
    logger.info(
        f"Performance — TTFT: {result.ttft_ms:.1f}ms, "
        f"tok/s/u: {result.tok_s_u:.1f}, "
        f"tok/s: {result.tok_s:.1f}, "
        f"decode latency: {result.decode_latency_mean_ms:.2f}ms"
    )
    if log_text:
        log_generated_text(prompts, result.generated_token_ids, llm.tokenizer)

    if os.environ.get("CI") == "true":
        hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        model_target, model_variant = _benchmark_model_identity(hf_model, llm.model_name)
        prefill_seq_len = int(prompt_lens.max())
        measurements = {
            "prefill_t/s": (
                (result.batch_size * prefill_seq_len) / result.prefill_time_s if result.prefill_time_s > 0 else 0.0
            ),
            "prefill_time_to_token": result.prefill_time_s / result.batch_size,
            "decode_t/s": result.tok_s,
            "decode_t/s/u": result.tok_s_u,
        }
        benchmark_data = create_benchmark_data(
            profiler, measurements, {"inference_prefill": 0, "inference_decode": 1}, targets={}
        )
        decode_iteration_times = result.decode_iteration_times_s or result.decode_times_s
        for token_pos, decode_time_s in enumerate(decode_iteration_times, start=1):
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                f"time_to_token_{token_pos}",
                decode_time_s * 1000,
                step_warm_up_num_iterations=None,
                target=None,
            )
        for token_pos in (1, 128, 1024, 2048, 4096, 8192):
            if token_pos <= len(decode_iteration_times):
                benchmark_data.add_measurement(
                    profiler,
                    0,
                    "inference_decode",
                    f"decode_latency_ms_token_{token_pos}",
                    decode_iteration_times[token_pos - 1] * 1000,
                    step_warm_up_num_iterations=None,
                    target=None,
                )
        # Match TTTv1's historical first-128 window: compile iteration 0 is
        # excluded, leaving steady-state iterations 1 through 127.
        first_window = decode_iteration_times[:127]
        if first_window:
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                "avg_decode_time_first_128",
                sum(first_window) * 1000 / len(first_window),
                step_warm_up_num_iterations=None,
                target=None,
            )
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo_perf",
            ml_model_name=model_target,
            ml_model_type="llm",
            device_name=get_device_name(mesh_device),
            num_layers=len(model_config.block_configs),
            batch_size=result.batch_size,
            config_params={
                "model_variant": model_variant,
                "data_parallel": data_parallel,
                "tensor_parallel": model_config.num_devices,
                "sampling_mode": sampling_mode,
                "optimization_profile": case_name.split("/", 1)[0],
                "workload": case_name.split("/", 1)[1],
            },
            input_sequence_length=prefill_seq_len,
            output_sequence_length=result.num_decode_tokens,
        )

    if expected:
        targets = result.meets_target(expected, PERF_TOLERANCE)
        for metric, passed in targets.items():
            if not passed:
                logger.warning(
                    f"{metric} did not meet target: got {getattr(result, metric)}, expected {expected[metric]}"
                )


def _run_perf_benchmark(llm, mesh_device, expected, batch_size, case_name, num_decode_tokens=None):
    """Run performance benchmark (TTFT + tok/s/u)."""
    prompts_path = DEMO_DIR / "sample_prompts" / "input_data_questions_prefill_128.json"
    prompts = load_input_prompts(prompts_path, batch_size)
    default_decode_tokens = 200 if num_decode_tokens is None else int(num_decode_tokens)
    num_decode_tokens = int(os.environ.get("LLAMA3_8B_TTTV2_DECODE_TOKENS", str(default_decode_tokens)))
    profiler = BenchmarkProfiler()
    result, prompt_lens, sampling_mode = _run_batch_once(
        llm,
        prompts,
        case_name=case_name,
        num_decode_tokens=num_decode_tokens,
        profiler=profiler,
    )
    _report_performance(
        llm,
        mesh_device,
        expected,
        prompts=prompts,
        case_name=case_name,
        profiler=profiler,
        result=result,
        prompt_lens=prompt_lens,
        sampling_mode=sampling_mode,
    )


def _contiguous_page_table(max_batch_size: int, max_seq_len: int, *, repeat_per_lane: bool = False) -> torch.Tensor:
    max_num_blocks_per_user = math.ceil(max_seq_len / 32)
    if repeat_per_lane:
        return torch.arange(max_num_blocks_per_user, dtype=torch.int32).repeat(max_batch_size, 1)
    max_num_blocks = max_num_blocks_per_user * max_batch_size
    return torch.arange(max_num_blocks, dtype=torch.int32).reshape(max_batch_size, max_num_blocks_per_user)


def _eval_repeat_prompts(batch_size: int) -> list[str]:
    return load_input_prompts(
        Path("models/tt_transformers/demo/sample_prompts/eval_repeat_prompts_batch32.json"), batch_size
    )


def _rotate(items: list, amount: int) -> list:
    amount %= len(items)
    return items[amount:] + items[:amount]


def _truncate_at_stop(output_ids, tokenizer) -> list[int]:
    stop = set()
    if tokenizer.eos_token_id is not None:
        stop.add(tokenizer.eos_token_id)
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0:
        stop.add(eot)
    seq = list(output_ids)
    for index, token in enumerate(seq):
        if token in stop:
            return seq[:index]
    return seq


def _run_eval_repeat_batches(
    llm,
    *,
    batch_size: int,
    repeat_batches: int,
    num_decode_tokens: int,
    profiler=None,
) -> tuple[PerfBenchmarkResult, torch.Tensor, str, list[str]]:
    tokenizer = llm.tokenizer
    prompts = _eval_repeat_prompts(batch_size)

    per_repeat = []
    reported_batch = None
    for repeat in range(repeat_batches):
        rotated_prompts = _rotate(prompts, repeat)
        result, prompt_lens, sampling_mode = _run_batch_once(
            llm,
            rotated_prompts,
            case_name=f"eval-{batch_size}/repeat-{repeat}",
            num_decode_tokens=num_decode_tokens,
            profiler=profiler if repeat == 0 else None,
        )
        if repeat == 0:
            reported_batch = result, prompt_lens, sampling_mode, rotated_prompts
        unrotated = _rotate([_truncate_at_stop(ids, tokenizer) for ids in result.generated_token_ids], -repeat)
        per_repeat.append(unrotated)

    failures = []
    for left_repeat, right_repeat in zip(per_repeat, per_repeat[1:]):
        for user, (left, right) in enumerate(zip(left_repeat, right_repeat)):
            if left != right:
                failures.append(user)
    assert not failures, f"eval-{batch_size} generated token IDs differed for users {failures[:10]}"
    return reported_batch


def _run_dp_smoke(mesh_device, optimizations: str, case: DemoCase) -> None:
    """Run a functional DP smoke with telemetry, not a performance gate.

    ``optimizations`` names the model optimization profile; it does not make
    this a gated performance test. TTTv1 DP parity requires logging and CI
    artifacts while functional execution determines pass/fail.
    """
    data_parallel = case.data_parallel
    per_lane_batch_size = case.batch_size // data_parallel
    assert per_lane_batch_size == 1, f"{case.name} expects one active user per DP lane"
    submeshes = list(create_submeshes(mesh_device, data_parallel))
    assert len(submeshes) == data_parallel, f"Expected {data_parallel} submeshes, got {len(submeshes)}"

    llms = []
    lanes = []
    group = None
    try:
        for submesh in submeshes:
            _validate_tp_topology(submesh)
            llm = create_llama3_for_causal_lm(
                submesh,
                optimizations,
                max_batch_size=per_lane_batch_size,
                max_seq_len=case.max_seq_len,
            )
            llms.append(llm)

        sampling_mode, sampling_params = _sampling_params_for_model(llms[0].model, case_name=case.name)
        for llm in llms:
            lanes.append(
                _build_demo_executor(
                    llm,
                    trace_mode="all",
                    device_sampling_enabled=sampling_params is not None,
                    include_decode_top_k=_force_decode_top_k(
                        sampling_mode,
                        sampling_params,
                        llm.model.config.num_devices,
                    ),
                )
            )

        group = LaneGroupExecutor(lanes, mesh_device=mesh_device)
        kv_cache = group.allocate_kv_cache()
        page_table = _contiguous_page_table(case.batch_size, case.max_seq_len, repeat_per_lane=True)
        _warmup_demo_executor(
            group,
            kv_cache=kv_cache,
            page_table=page_table,
            prefill_can_sample_on_device=False,
        )

        prompts = load_input_prompts(
            DEMO_DIR / "sample_prompts" / "input_data_questions_prefill_128.json", case.batch_size
        )
        input_tokens, prompt_lens = preprocess_llama3_8b_chat_prompts(
            prompts,
            llms[0],
            reserve_decode_tokens=case.num_decode_tokens,
        )
        profiler = BenchmarkProfiler()
        profiler.start("run")
        try:
            result = run_perf_benchmark(
                group,
                tokens=input_tokens,
                kv_cache=kv_cache,
                page_table=page_table,
                num_decode_tokens=case.num_decode_tokens,
                max_batch_size=case.batch_size,
                prompt_lens=prompt_lens,
                sampling_params=sampling_params,
                prefill_sampling_params=None,
                pipeline_readback=os.environ.get("PIPELINE_READBACK", "1").lower() not in ("0", "false", "no"),
                profiler=profiler,
            )
        finally:
            profiler.end("run")
        # Match TTTv1's correctness-before-telemetry ordering: a failed DP run
        # must not leave a benchmark partial for post-failure artifact processing.
        assert len(result.generated_token_ids) == data_parallel
        assert all(result.generated_token_ids), f"{case.name}: every DP lane must return output"
        assert_no_special_tokens(result.generated_token_ids, llms[0].tokenizer, case_name=case.name)
        _report_performance(
            llms[0],
            mesh_device,
            {},
            prompts=prompts,
            case_name=f"{optimizations}/{case.name}",
            profiler=profiler,
            result=result,
            prompt_lens=prompt_lens,
            sampling_mode=sampling_mode,
            log_text=False,
            data_parallel=data_parallel,
        )
    finally:
        if group is not None:
            group.cleanup()
        else:
            for lane in lanes:
                lane.cleanup()
        for llm, submesh in zip(llms, submeshes):
            cleanup_model_case(llm.model, submesh)
        if data_parallel > 1:
            mesh_device.quiesce_devices()
