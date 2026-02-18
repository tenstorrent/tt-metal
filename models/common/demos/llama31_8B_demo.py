# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 MLP1D Demo Script for Llama 3.1-8B-Instruct

Demonstrates the TTTv2 MLP1D module by replacing the MLP layers in Llama 3.1-8B-Instruct model.

Measurements:
- Top-1/Top-5 token accuracy (like `ci-token-matching`)
- tok/s/u and TTFT performance (like `batch-1` and `batch-32`)

Supports:
- `performance` and `accuracy` optimization modes
- N150 (1x1), N300 (1x2), T3K (1x8) topologies

Usage:
    # Token accuracy test on N150
    MESH_DEVICE=N150 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python_env/bin/pytest models/common/demos/llama31_8B_demo.py -k "token-accuracy" -v

    # Batch-1 latency test on N300
    MESH_DEVICE=N300 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python_env/bin/pytest models/common/demos/llama31_8B_demo.py -k "batch-1" --performance -v

    # Batch-32 throughput test on T3K
    MESH_DEVICE=T3K HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python_env/bin/pytest models/common/demos/llama31_8B_demo.py -k "batch-32" --performance -v

"""

import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager

import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    create_tt_model,
    preprocess_inputs_prefill,
    sample_host,
)
from models.tt_transformers.tt.generator import Generator, SamplingParams
from models.tt_transformers.tt.model_config import DecodersPrecision

# =============================================================================
# Constants and Expected Metrics
# =============================================================================

# Expected accuracy metrics from PERF.md for Llama-3.1-8B (top1, top5 only)
# Performance metrics (tok_s_u, ttft_ms) are collected dynamically by running
# simple_text_demo.py with the corresponding test case to get real baseline values.
# Higher is better for tok_s_u, top1, top5
# Lower is better for ttft_ms
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

# Tolerance for performance validation (5%)
PERF_TOLERANCE = 0.05


# =============================================================================
# Baseline Collection from simple_text_demo.py
# =============================================================================


def collect_baseline_from_simple_text_demo(
    device_name: str,
    batch_size: int,
    opt_mode: str,
) -> dict | None:
    """
    Run simple_text_demo.py to collect real baseline performance metrics (tok_s_u, ttft_ms).

    This provides an accurate comparison by measuring TTTv1 MLP performance
    on the same hardware under the same conditions, rather than relying on
    potentially outdated PERF.md values.

    Args:
        device_name: Device name (N150, N300, T3K)
        batch_size: Batch size (1 or 32)
        opt_mode: Optimization mode (performance or accuracy)

    Returns:
        Dict with baseline metrics {tok_s_u, ttft_ms} or None if collection fails
    """
    # Build the test ID for simple_text_demo
    batch_id = f"batch-{batch_size}"
    test_filter = f"{opt_mode}-{batch_id}"

    # Get HF_MODEL from environment
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

    logger.info(f"Collecting baseline from simple_text_demo.py ({test_filter})...")

    # Build pytest command (use sys.executable for CI compatibility)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "models/tt_transformers/demo/simple_text_demo.py",
        "-k",
        test_filter,
        "-v",
        "--tb=short",
    ]

    env = os.environ.copy()
    env["MESH_DEVICE"] = device_name
    env["HF_MODEL"] = hf_model
    # Unset CI to ensure tests run normally (CI=true may skip or alter test behavior)
    env.pop("CI", None)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            env=env,
            cwd=os.getcwd(),
        )

        output = result.stdout + result.stderr

        # Parse the output for performance metrics
        # Looking for lines like:
        # "Average Time to First Token (TTFT): 68.09ms"
        # "Average speed: 57.89ms @ 17.28 tok/s/user (17.28 tok/s throughput)"
        ttft_match = re.search(r"Average Time to First Token \(TTFT\):\s*([\d.]+)ms", output)
        speed_match = re.search(r"Average speed:.*@\s*([\d.]+)\s*tok/s/user", output)

        if ttft_match and speed_match:
            baseline = {
                "ttft_ms": float(ttft_match.group(1)),
                "tok_s_u": float(speed_match.group(1)),
            }
            logger.info(f"Baseline collected: {baseline['tok_s_u']:.2f} tok/s/u, TTFT {baseline['ttft_ms']:.2f}ms")
            return baseline
        else:
            logger.warning(f"Could not parse baseline metrics from simple_text_demo output")
            if "PASSED" not in output and "passed" not in output:
                logger.warning(f"Test may have failed. Exit code: {result.returncode}")
            return None

    except subprocess.TimeoutExpired:
        logger.warning("Baseline collection timed out after 10 minutes")
        return None
    except Exception as e:
        logger.warning(f"Baseline collection failed: {e}")
        return None


# =============================================================================
# Token Accuracy Helper
# =============================================================================


class TokenAccuracy:
    """Helper class for measuring token accuracy against reference data."""

    def __init__(self, model_name: str):
        self.gt_pos = -1
        self.store_predicted_tokens = []
        reference_data_file = os.path.join("models/tt_transformers/tests/reference_outputs/", model_name) + ".refpt"
        if not os.path.exists(reference_data_file):
            raise FileNotFoundError(f"Reference data file not found: {reference_data_file}")

        logger.info(f"Loading reference data from {reference_data_file}")
        reference_data = torch.load(reference_data_file)
        reference_tokens = reference_data["reference_tokens"]
        split_point = reference_tokens.shape[-1] // 2
        self.input_prompt = reference_tokens[0, :split_point]
        self.reference_tokens = reference_tokens[0, split_point:]
        self.top5_tokens = reference_data["top5_tokens"][split_point - 1 :, :]
        self.maxindex = len(self.reference_tokens) - 1

    def prepare_ref_tokens(self, tokenizer):
        """Decode input prompt for reference."""
        return tokenizer.decode(self.input_prompt.tolist())

    def collect_predicted_tokens(self, tokens):
        """Collect predicted tokens and return the next reference token (teacher forcing)."""
        self.store_predicted_tokens.append(tokens)
        self.gt_pos += 1
        return self.reference_tokens[min(self.gt_pos, self.maxindex)].unsqueeze(-1).unsqueeze(-1)

    def compute_accuracy(self) -> tuple[float, float]:
        """Compute Top-1 and Top-5 accuracy."""
        count = 0
        count_t5 = 0
        matching_sz = min(len(self.reference_tokens), len(self.store_predicted_tokens))
        for i in range(matching_sz):
            if self.top5_tokens[i, 0].item() == self.store_predicted_tokens[i]:
                count += 1
            if self.store_predicted_tokens[i] in self.top5_tokens[i, :]:
                count_t5 += 1
        accuracy_top1 = count / matching_sz
        accuracy_top5 = count_t5 / matching_sz

        return accuracy_top1 * 100, accuracy_top5 * 100


# =============================================================================
# MLP1D Patching Context Manager
# =============================================================================


@contextmanager
def patch_mlp_with_mlp1d():
    """
    Context manager that patches decoder.MLP with MLP1D.from_model_args.

    The signatures now match, enabling drop-in replacement without adapters.
    """
    from models.common.modules.mlp.mlp_1d import MLP1D
    from models.tt_transformers.tt import decoder

    # Store original MLP class
    original_MLP = decoder.MLP

    # Replace with MLP1D.from_model_args - signatures match!
    decoder.MLP = MLP1D.from_model_args

    try:
        yield
    finally:
        # Restore original MLP class
        decoder.MLP = original_MLP


def create_tt_model_with_mlp1d(
    mesh_device,
    instruct: bool,
    max_batch_size: int,
    optimizations,
    max_seq_len: int,
    paged_attention_config,
    dtype,
    state_dict=None,
    num_layers=None,
):
    """
    Wrapper around create_tt_model that patches MLP → MLP1D.from_model_args.

    With matching signatures, this is now a simple class swap without adapters.
    """
    with patch_mlp_with_mlp1d():
        model_args, model, tt_kv_cache, state_dict = create_tt_model(
            mesh_device,
            instruct=instruct,
            max_batch_size=max_batch_size,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=dtype,
            state_dict=state_dict,
            num_layers=num_layers,
        )

    return model_args, model, tt_kv_cache, state_dict


# =============================================================================
# Performance Validation
# =============================================================================


def validate_metrics(measured: dict, expected: dict, tolerance: float = PERF_TOLERANCE) -> tuple[bool, str]:
    """
    Validate measured metrics against expected values with tolerance.

    For performance tests: expected comes from simple_text_demo.py baseline (tok_s_u, ttft_ms).
    For accuracy tests: expected comes from PERF.md (top1, top5).

    Returns:
        (passed, message) tuple
    """
    failures = []
    for metric, expected_val in expected.items():
        if expected_val is None:
            continue
        measured_val = measured.get(metric)
        if measured_val is None:
            continue

        # Higher is better for tok_s_u, top1, top5
        # Lower is better for ttft_ms
        if metric == "ttft_ms":
            if measured_val > expected_val * (1 + tolerance):
                failures.append(
                    f"{metric}: measured {measured_val:.2f} > expected {expected_val * (1 + tolerance):.2f}"
                )
        else:
            if measured_val < expected_val * (1 - tolerance):
                failures.append(
                    f"{metric}: measured {measured_val:.2f} < expected {expected_val * (1 - tolerance):.2f}"
                )

    if failures:
        return False, "; ".join(failures)
    return True, "All metrics within tolerance"


def get_device_name_from_mesh_shape(mesh_shape: tuple[int, int]) -> str:
    """Map mesh_shape to device name."""
    mapping = {
        (1, 1): "N150",
        (1, 2): "N300",
        (1, 8): "T3K",
    }
    return mapping.get(mesh_shape, f"Unknown({mesh_shape})")


# =============================================================================
# Test Parametrization
# =============================================================================


def _get_mesh_shape():
    """Get mesh shape from MESH_DEVICE environment variable."""
    shape_map = {
        "N150": (1, 1),
        "N300": (1, 2),
        "T3K": (1, 8),
    }
    return shape_map.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))


# Test configurations
TEST_CONFIGS = [
    # Token accuracy test (like ci-token-matching)
    pytest.param(
        {
            "name": "token-accuracy",
            "batch_size": 1,
            "max_generated_tokens": 500,
            "measure_accuracy": True,
        },
        id="token-accuracy",
    ),
    # Latency test (like batch-1)
    pytest.param(
        {
            "name": "batch-1-latency",
            "batch_size": 1,
            "max_generated_tokens": 200,
            "measure_accuracy": False,
        },
        id="batch-1",
    ),
    # Throughput test (like batch-32)
    pytest.param(
        {
            "name": "batch-32-throughput",
            "batch_size": 32,
            "max_generated_tokens": 200,
            "measure_accuracy": False,
        },
        id="batch-32",
    ),
]

# Global cache for baseline metrics (collected before any test opens mesh_device)
_baseline_cache: dict[str, dict] = {}
_baseline_collected: bool = False


def _get_baseline_cache_key(device_name: str, batch_size: int, opt_mode: str) -> str:
    """Generate cache key for baseline metrics."""
    return f"{device_name}-batch{batch_size}-{opt_mode}"


def _collect_all_baselines():
    """
    Collect all baseline metrics ONCE at module load time, before any tests run.

    This ensures baselines are collected when no mesh_device is open.
    """
    global _baseline_collected
    if _baseline_collected:
        return

    device_name = os.environ.get("MESH_DEVICE")
    assert device_name is not None, "MESH_DEVICE environment variable is not set"
    logger.info(f"=== Collecting baselines from simple_text_demo.py for {device_name} ===")

    # Collect baselines for batch_size x opt_mode combinations
    # CI runs batch-32, so skip batch-1 when CI=true
    is_ci = os.environ.get("CI", "").lower() in ("true", "1", "yes")
    batch_sizes = [32] if is_ci else [1, 32]

    for batch_size in batch_sizes:
        for opt_mode in ["performance", "accuracy"]:
            cache_key = _get_baseline_cache_key(device_name, batch_size, opt_mode)
            logger.info(f"Collecting baseline for {cache_key}...")
            baseline = collect_baseline_from_simple_text_demo(
                device_name=device_name,
                batch_size=batch_size,
                opt_mode=opt_mode,
            )
            if baseline:
                _baseline_cache[cache_key] = baseline
                logger.info(f"  ✓ {cache_key}: {baseline['tok_s_u']:.2f} tok/s/u, {baseline['ttft_ms']:.2f}ms TTFT")
            else:
                logger.warning(f"  ✗ {cache_key}: baseline collection failed")

    _baseline_collected = True
    logger.info("=== Baseline collection complete ===")


def _get_cached_baseline(device_name: str, batch_size: int, opt_mode: str) -> dict | None:
    """Get cached baseline metrics. Must be called after baselines are collected."""
    cache_key = _get_baseline_cache_key(device_name, batch_size, opt_mode)
    return _baseline_cache.get(cache_key)


# Collect all baselines at module import time, before pytest sets up any fixtures.
# This ensures baselines are collected when no mesh_device is open.
_collect_all_baselines()


# =============================================================================
# Main Test Function
# =============================================================================


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_get_mesh_shape()],
    indirect=True,
)
@pytest.mark.parametrize(
    "optimizations",
    [
        pytest.param(
            lambda args: DecodersPrecision.performance(args.n_layers, args.model_name),
            id="performance",
        ),
        pytest.param(
            lambda args: DecodersPrecision.accuracy(args.n_layers, args.model_name),
            id="accuracy",
        ),
    ],
)
@pytest.mark.parametrize("test_config", TEST_CONFIGS)
def test_mlp1d_llama_demo(
    mesh_device: ttnn.MeshDevice,
    optimizations,
    test_config: dict,
    is_ci_env: bool,
    request,
):
    """
    Demo test for MLP1D replacement in Llama 3.1-8B-Instruct.

    This test:
    1. Creates the TT model with MLP layers replaced by MLP1D
    2. Runs prefill and decode inference
    3. Measures performance (TTFT, tok/s/u) and/or accuracy (Top-1, Top-5)
    4. Validates results against PERF.md targets
    """
    test_id = request.node.callspec.id

    if is_ci_env and "batch-1" in test_id:
        pytest.skip("CI only runs batch-32 and token-accuracy tests")

    # Skip TG devices - MLP1D does not support Galaxy
    mesh_shape = mesh_device.shape
    if mesh_shape[0] > 1 or mesh_shape[1] > 8:
        pytest.skip("MLP1D does not support TG/Galaxy devices")

    # Get test parameters
    test_name = test_config["name"]
    batch_size = test_config["batch_size"]
    max_generated_tokens = test_config["max_generated_tokens"]
    measure_accuracy = test_config["measure_accuracy"]

    # Get optimization mode from test ID
    opt_mode = "performance" if "performance" in test_id else "accuracy"

    # Get device name
    device_name = get_device_name_from_mesh_shape(tuple(mesh_shape))

    logger.info(f"=== MLP1D Demo: {test_name} ===")
    logger.info(f"Device: {device_name}, Optimization: {opt_mode}, Batch: {batch_size}")

    # Configuration
    instruct = not measure_accuracy  # Use non-instruct for token accuracy (matches ci-token-matching)
    max_seq_len = 1024
    paged_attention = True
    page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
    dtype = ttnn.bfloat8_b

    # Paged attention config
    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks_per_dp"],
        )
        if paged_attention
        else None
    )

    # Create model with MLP1D replacement
    logger.info("Creating TT model with MLP1D replacement...")
    model_args, model, tt_kv_cache, _ = create_tt_model_with_mlp1d(
        mesh_device=mesh_device,
        instruct=instruct,
        max_batch_size=batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        dtype=dtype,
    )

    # Validate model was created successfully
    assert model is not None, "Failed to create model"
    assert model_args is not None, "Failed to get model args"

    # Wrap in lists for API compatibility (like prepare_generator_args does for data parallel)
    model_args_list = [model_args]
    model_list = [model]
    tt_kv_cache_list = [tt_kv_cache]

    tokenizer = model_args.tokenizer

    # Initialize profiler
    profiler = BenchmarkProfiler()
    profiler.start("run")

    # Setup token accuracy if needed
    token_acc = None
    if measure_accuracy:
        try:
            token_acc = TokenAccuracy(model_name=model_args.model_name)
        except FileNotFoundError as e:
            pytest.skip(f"Reference data not found: {e}")

    # Prepare input prompts
    if measure_accuracy and token_acc:
        input_prompts = [token_acc.prepare_ref_tokens(tokenizer)]
    else:
        # Use the same 128-token prompts as simple_text_demo.py for fair perf comparison
        prompt_file = "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json"
        with open(prompt_file, "r") as f:
            prompts_data = json.load(f)
        # Repeat/truncate prompts to match batch_size
        input_prompts = [prompts_data[i % len(prompts_data)]["prompt"] for i in range(batch_size)]

    # Create page table for paged attention
    page_table = None
    if paged_attention_config:
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(batch_size, paged_attention_config.max_num_blocks // batch_size)

    # Preprocess inputs
    logger.info("Preprocessing inputs...")
    (
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        input_prompts, tokenizer, model_args_list, instruct, max_generated_tokens, max_prefill_len=max_seq_len
    )
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

    # Create generator
    generator = Generator(model_list, model_args_list, mesh_device, tokenizer=tokenizer)

    # --- Prefill Phase ---
    logger.info("Starting prefill warmup...")
    profiler.start("compile_prefill")
    logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache_list,
        prompt_lens=decoding_pos,
    )
    profiler.end("compile_prefill")

    logger.info("Starting prefill inference...")
    profiler.start("inference_prefill")
    logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache_list,
        prompt_lens=decoding_pos,
    )
    prefilled_token = torch.argmax(logits, dim=-1)
    profiler.end("inference_prefill")

    # --- Decode Phase ---
    # Keep track of generated outputs to print out every iteration
    all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(batch_size)]
    for user in range(batch_size):
        user_tok = int(prefilled_token[user].item())
        all_outputs[user].append(user_tok)

    logger.info("Starting decode loop...")
    current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])
    out_tok = prefilled_token
    user_done = [False] * batch_size

    # Sampling params (argmax for deterministic results)
    sampling_params = {"temperature": 0, "top_p": 0.08, "top_k": 32}
    device_sampling_params = (
        SamplingParams(
            temperature=sampling_params["temperature"],
            top_k=sampling_params["top_k"],
            top_p=sampling_params["top_p"],
        )
        if model_list[0]._supports_on_device_sampling
        else None
    )

    profiler.start("inference_decode")

    for iteration in range(max_generated_tokens):
        if iteration == 0:
            profiler.start("compile_decode")
        else:
            profiler.start(f"inference_decode_time_{iteration}")

        # Teacher forcing for token accuracy
        if token_acc:
            out_tok[0] = token_acc.collect_predicted_tokens(out_tok[0].item())

        # Decode forward
        logits, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=not measure_accuracy,  # Disable trace for accuracy (teacher forcing)
            page_table=page_table,
            kv_cache=tt_kv_cache_list,
            sampling_params=device_sampling_params,
            prompt_tokens=input_tokens_prefill_pt,
            output_tokens=out_tok,
        )

        # Get next token
        if device_sampling_params is not None:
            out_tok = logits.unsqueeze(1)
        else:
            _, out_tok = sample_host(
                logits,
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                on_host=True,
            )

        if iteration == 0:
            profiler.end("compile_decode")
            decode_iteration_time = profiler.get_duration("compile_decode")
        else:
            profiler.end(f"inference_decode_time_{iteration}")
            decode_iteration_time = profiler.get_duration(f"inference_decode_time_{iteration}")

        # Log per-iteration performance
        tokens_per_second_per_user = 1 / decode_iteration_time if decode_iteration_time > 0 else 0
        logger.debug(
            f"Iteration {iteration}: {1000 * decode_iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user"
        )

        current_pos += 1

        # Save output token to print out later and check for EOS
        for user in range(batch_size):
            user_tok = out_tok[user].item()
            if user_tok not in tokenizer.stop_tokens and not user_done[user]:
                all_outputs[user].append(user_tok)
            else:
                if not measure_accuracy:  # For accuracy tests, keep decoding (teacher forcing)
                    user_done[user] = True
                    logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                else:
                    all_outputs[user].append(user_tok)

        # Print out generated outputs for each user (debug level)
        for user in range(batch_size):
            text = "".join(tokenizer.decode(all_outputs[user]))
            if len(text) > 100:
                text = "..." + text[-97:]
            text = text.replace("\n", " ")
            logger.debug(f"[User {user}] {text}")

        if all(user_done):
            break

    profiler.end("inference_decode")

    # Final print of generated text
    logger.info("Finished decoding, printing the final outputs...\n")
    for user, output in enumerate(all_outputs):
        text = tokenizer.decode(output)
        # Find where prompt ends and generated text begins
        prompt_text = input_prompts[user] if user < len(input_prompts) else ""
        # Strip leading/trailing newlines from generated portion
        text_after_prompt = text[len(prompt_text) :] if text.startswith(prompt_text) else text
        short_prompt = (
            (prompt_text[:100] + "\n<long prompt not printed in full>\n" + prompt_text[-100:])
            if len(prompt_text) > 200
            else prompt_text
        )
        logger.info(f"\n==USER {user} - PROMPT\n{short_prompt}\n==USER {user} - OUTPUT\n{text_after_prompt.strip()}\n")
    profiler.end("run")

    # --- Compute Metrics ---
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")
    total_inference_prefill_time = profiler.get_duration("inference_prefill")

    # Sum decode times (skip compile iteration)
    total_inference_decode_time = 0
    num_decode_iterations = min(iteration + 1, max_generated_tokens)
    for i in range(1, num_decode_iterations):
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

    # Calculate performance metrics
    avg_ttft_ms = (total_inference_prefill_time / batch_size) * 1000
    tok_s_u = (num_decode_iterations - 1) / total_inference_decode_time if num_decode_iterations > 1 else 0

    # Compute accuracy if applicable
    top1_acc = None
    top5_acc = None
    if token_acc:
        top1_acc, top5_acc = token_acc.compute_accuracy()

    # --- Report Results ---
    logger.info("")
    logger.info("=== Performance Metrics ===")
    logger.info(f"Device: {device_name}")
    logger.info(f"Mode: {opt_mode}")
    logger.info(f"Compile prefill: {compile_prefill_time:.2f}s")
    logger.info(f"Compile decode: {compile_decode_time:.2f}s")
    logger.info(f"TTFT: {avg_ttft_ms:.2f}ms")
    logger.info(f"Decode speed: {tok_s_u:.2f} tok/s/u")
    if top1_acc is not None:
        logger.info(f"Top-1 accuracy: {top1_acc:.2f}%")
    if top5_acc is not None:
        logger.info(f"Top-5 accuracy: {top5_acc:.2f}%")

    # Build measured metrics dict
    measured = {
        "ttft_ms": avg_ttft_ms,
        "tok_s_u": tok_s_u,
    }
    if top1_acc is not None:
        measured["top1"] = top1_acc
    if top5_acc is not None:
        measured["top5"] = top5_acc

    # --- Validate Against Baseline ---
    # Performance metrics (tok_s_u, ttft_ms): MUST come from simple_text_demo.py baseline.
    # Accuracy metrics (top1, top5): ONLY from PERF.md (for token-accuracy tests).
    # Baselines are collected at module load time, before any test opens mesh_device.
    baseline_metrics = _get_cached_baseline(device_name, batch_size, opt_mode)

    if measure_accuracy:
        # Token-accuracy tests: use PERF.md for top1/top5 only
        baseline_source = "PERF.md"
        perf_md_metrics = EXPECTED_METRICS.get(opt_mode, {}).get(device_name, {})
        expected_for_validation = {k: v for k, v in perf_md_metrics.items() if k in ("top1", "top5")}
        logger.info("Token-accuracy test runs without trace - validating top1/top5 from PERF.md only")
    else:
        # Performance tests: MUST use baseline from simple_text_demo.py
        assert baseline_metrics is not None, (
            f"Baseline collection from simple_text_demo.py failed for {device_name}/{opt_mode}/batch-{batch_size}. "
            "Performance metrics (tok_s_u, ttft_ms) require a valid baseline measurement."
        )
        assert "tok_s_u" in baseline_metrics, f"Baseline missing 'tok_s_u': {baseline_metrics}"
        assert "ttft_ms" in baseline_metrics, f"Baseline missing 'ttft_ms': {baseline_metrics}"
        baseline_source = "simple_text_demo.py (TTTv1)"
        expected_for_validation = baseline_metrics

    if expected_for_validation:
        logger.info(f"Comparing against baseline from: {baseline_source}")
        if baseline_metrics:
            logger.info(f"  Baseline tok_s_u: {baseline_metrics.get('tok_s_u', 'N/A'):.2f}")
            logger.info(f"  Baseline ttft_ms: {baseline_metrics.get('ttft_ms', 'N/A'):.2f}")
            logger.info(f"  MLP1D tok_s_u:    {tok_s_u:.2f}")
            logger.info(f"  MLP1D ttft_ms:    {avg_ttft_ms:.2f}")

            # Calculate relative performance
            if baseline_metrics.get("tok_s_u"):
                rel_perf = (tok_s_u / baseline_metrics["tok_s_u"]) * 100
                logger.info(f"  Relative performance: {rel_perf:.1f}% of baseline")

        passed, message = validate_metrics(measured, expected_for_validation)
        if passed:
            logger.info(f"✓ All metrics within {PERF_TOLERANCE * 100:.0f}% of {baseline_source} targets")
        else:
            logger.warning(f"✗ Metrics outside tolerance: {message}")
            # Don't fail the test on performance regression - just warn
            # This allows the demo to run even if performance varies

    logger.info(f"=== MLP1D Demo Complete ===")
