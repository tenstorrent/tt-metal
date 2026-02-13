# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Fused op unit test for lm_head (vocabulary projection).

lm_head is the linear projection from hidden_size (7168) to vocab_size (129280).
The weight is sharded across all 32 devices, with each device holding vocab/32 = 4040
output features.

Sequence of ops:
    Decode:  output = ttnn.linear(x, **cfg["linear"])
    Prefill: output = ttnn.linear(x, program_config=..., **cfg["linear"])

Key characteristics:
    - Weight dtype: bfloat4_b (quantized)
    - Weight shape per device: [7168, 4040]
    - Weight memory: WIDTH_SHARDED DRAM
    - Decode input: WIDTH_SHARDED L1
    - Prefill input: DRAM INTERLEAVED
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tests.fused_op_unit_tests.test_utils import (
    collect_device_perf,
    compare_with_reference,
    get_int_env,
    log_run_mode,
    maybe_skip_long_seq,
    measure_perf_us,
)
from models.demos.deepseek_v3.tt.lm_head import LMHead
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import get_model_config, get_test_weight_config, pad_or_trim_seq_len
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

LONG_SEQ_ENV_VAR = "DEEPSEEK_V3_LONG_SEQ_TESTS"
DEVICE_PERF_ENV_VAR = "DS_LM_HEAD_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US = {
    ("decode", 1): {"kernel": 99.682, "op_to_op": 903.166},  # Measured: kernel=90.62, op_to_op=821.06
    ("prefill", 128): {"kernel": 590.038, "op_to_op": 75402.335},  # Measured: kernel=536.40, op_to_op=68547.58
}


class DeepseekV3LMHead(nn.Module):
    """PyTorch reference model for LMHead."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


def ds_lm_head_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation for lm_head linear projection.

    Args:
        x: Input tensor of shape [1, 1, seq_len, hidden_size] or [1, num_chunks, chunk_size, hidden_size]
        weight: Weight tensor of shape [vocab_size, hidden_size]

    Returns:
        Output tensor of shape [1, 1, seq_len, vocab_size] or [1, num_chunks, chunk_size, vocab_size]
    """
    return torch.nn.functional.linear(x, weight)


def ds_lm_head_ttnn_decode(
    x: ttnn.Tensor,
    cfg: dict,
) -> ttnn.Tensor:
    """
    TTNN implementation for lm_head linear projection (decode mode).

    Uses DRAM sharded matmul with WIDTH_SHARDED L1 activations.

    Args:
        x: Input tensor (WIDTH_SHARDED L1)
        cfg: Configuration dictionary containing linear config

    Returns:
        Output tensor (WIDTH_SHARDED L1)
    """
    output = LMHead._fwd_linear(x, cfg)
    return output


def ds_lm_head_ttnn_prefill(
    x: ttnn.Tensor,
    cfg: dict,
    seq_len: int,
) -> ttnn.Tensor:
    """
    TTNN implementation for lm_head linear projection (prefill mode).

    Uses multicore multicast matmul with DRAM INTERLEAVED tensors.
    For long sequences, handles chunking.

    Args:
        x: Input tensor (DRAM INTERLEAVED)
        cfg: Configuration dictionary containing linear and linear_pc_gen configs
        seq_len: Original sequence length (before any chunking)

    Returns:
        Output tensor (DRAM INTERLEAVED)
    """
    # Use effective sequence length (chunk size) for program config to avoid L1 overflow
    effective_seq_len = min(seq_len, cfg.get("max_rows", seq_len))

    # Generate program config based on effective sequence length
    program_config = LMHead._get_prefill_pc(seq_len=effective_seq_len, **cfg["linear_pc_gen"])

    output = LMHead._fwd_linear(x, cfg, program_config=program_config)
    return output


def _run_ds_lm_head_test(
    mesh_device: ttnn.MeshDevice,
    run_config: dict,
    tt_input: ttnn.Tensor,
    ref_output: torch.Tensor,
    expected_pcc: float,
    expected_atol: float,
    expected_rtol: float,
    expected_perf_us: float,
    trace_mode: bool,
    program_cache_enabled: bool,
    mode: str,
    seq_len: int,
    batch_size: int,
    step_prefix: str,
):
    log_run_mode(mode, trace_mode, program_cache_enabled, seq_len)

    # Run lm_head
    if mode == "decode":
        tt_output = ds_lm_head_ttnn_decode(tt_input, run_config)
    else:
        tt_output = ds_lm_head_ttnn_prefill(tt_input, run_config, seq_len)

    # Convert output to torch - concatenate vocab dimension across all 32 devices
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))

    # Slice to actual vocab_size if needed (output may be padded)
    if tt_output_torch.shape[-1] > ref_output.shape[-1]:
        tt_output_torch = tt_output_torch[..., : ref_output.shape[-1]]

    pcc_value, max_abs_error = compare_with_reference(
        tt_output_torch,
        ref_output,
        expected_pcc,
        expected_atol,
        expected_rtol,
        convert_to_float=True,
        strict_assert=False,
    )

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        warmup_iters = get_int_env("DS_LM_HEAD_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = get_int_env("DS_LM_HEAD_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)
        logger.info(
            f"Starting e2e perf measurement: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}"
        )

        perf_profiler.start("run")
        perf_profiler.start(step_name)

        if mode == "decode":

            def op_fn():
                return ds_lm_head_ttnn_decode(tt_input, run_config)

        else:

            def op_fn():
                return ds_lm_head_ttnn_prefill(tt_input, run_config, seq_len)

        perf_us = measure_perf_us(
            mesh_device,
            op_fn,
            warmup_iters,
            measure_iters,
            trace_mode=trace_mode,
            profiler_name="ds_lm_head_perf",
        )
        logger.info(f"Perf avg: {perf_us:.3f} us over {measure_iters} iters (warmup {warmup_iters})")
        perf_profiler.end(step_name)
        perf_profiler.end("run")

        benchmark_data.add_measurement(
            perf_profiler,
            0,
            step_name,
            f"{step_name}-avg_us",
            perf_us,
            step_warm_up_num_iterations=PERF_WARMUP_ITERS,
            target=expected_perf_us if expected_perf_us > 0 and not trace_mode and program_cache_enabled else None,
        )
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-pcc", pcc_value)
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-max_abs_error", max_abs_error)
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-expected_atol", expected_atol)
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-expected_rtol", expected_rtol)
        benchmark_data.save_partial_run_json(
            perf_profiler,
            run_type="deepseek_v3_fused_ops",
            ml_model_name="deepseek-v3",
            batch_size=batch_size,
            input_sequence_length=seq_len,
            config_params={
                "mode": mode,
                "trace": trace_mode,
                "program_cache_enabled": program_cache_enabled,
                "module": "lm_head",
                "mesh_device": os.getenv("MESH_DEVICE", "TG"),
                "op_type": "lm_head",
            },
        )
        if expected_perf_us > 0 and not trace_mode and program_cache_enabled:
            perf_margin = 0.2
            assert perf_us <= expected_perf_us * (
                1 + perf_margin
            ), f"Perf regression: {perf_us:.3f}us exceeds expected {expected_perf_us:.3f}us"
        elif expected_perf_us == 0 and not trace_mode and program_cache_enabled:
            logger.warning("TODO: Set expected_perf_us using a measured baseline.")
    else:
        logger.info("Skipping e2e perf measurement during device-perf profiling.")
        from tracy import signpost

        if mode == "decode":

            def op_fn():
                return ds_lm_head_ttnn_decode(tt_input, run_config)

        else:

            def op_fn():
                return ds_lm_head_ttnn_prefill(tt_input, run_config, seq_len)

        for _ in range(PERF_WARMUP_ITERS):
            output = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            traced_output = op_fn()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                ttnn.execute_trace(mesh_device, trace_id, blocking=False)
                ttnn.synchronize_device(mesh_device)
            signpost("stop")
            ttnn.release_trace(mesh_device, trace_id)
            ttnn.deallocate(traced_output)
        else:
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                output = op_fn()
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(output)
            signpost("stop")

    # Clean up output
    ttnn.deallocate(tt_output)


def _build_lm_head_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config: Any,
    cache_path: Path,
    force_recalculate_weight_config: bool,
    use_real_weights: bool,
    mode: str,
    seq_len: int,
    state_dict: dict[str, torch.Tensor] | None,
):
    """Build inputs for lm_head test.

    Args:
        mesh_device: The mesh device
        hf_config: HuggingFace config
        cache_path: Path for weight caching
        force_recalculate_weight_config: Whether to force recalculate weights
        use_real_weights: Whether to use real model weights
        mode: "decode" or "prefill"
        seq_len: Sequence length
        state_dict: Model state dict (needed if use_real_weights=True)

    LMHead shape convention (from original test_lm_head.py):
    - Decode: height = 32 (USERS_PER_ROW users × 1 token each)
    - Prefill: height = seq_len (1 user × seq_len tokens)
    """
    hidden_size = hf_config.hidden_size

    # LMHead uses different batch conventions:
    # - Decode: batch_size=32 (USERS_PER_ROW), seq_len=1 → height=32
    # - Prefill: batch_size=1, seq_len=N → height=N
    if mode == "decode":
        batch_size = USERS_PER_ROW
        effective_height = batch_size * seq_len  # 32 × 1 = 32
    else:
        batch_size = 1
        effective_height = seq_len  # 1 × seq_len = seq_len

    # Create reference model
    reference_model = DeepseekV3LMHead(hf_config).eval()

    if use_real_weights and state_dict is not None:
        # Use provided real weights
        lm_head_state_dict = sub_state_dict(state_dict, "lm_head.")
        reference_model.load_state_dict(lm_head_state_dict, strict=False)
    else:
        # Use random weights (already initialized by nn.Linear)
        pass

    # Get state dict for TTNN weight conversion
    lm_head_state_dict = sub_state_dict(reference_model.state_dict(), "lm_head.")

    # Generate reference output
    # Shape: [1, 1, effective_height, hidden_size]
    torch_input = torch.randn(1, 1, effective_height, hidden_size)
    reference_output = reference_model(torch_input)

    # Pad input to SEQ_LEN_CHUNK_SIZE if necessary for TTNN
    torch_input_padded = pad_or_trim_seq_len(torch_input, mode, effective_height)

    # Generate TTNN configs
    # input_row_idx=3 matches the default in LMHead module test
    input_row_idx = 3
    weight_config = get_test_weight_config(
        LMHead, hf_config, (lm_head_state_dict,), cache_path, mesh_device, force_recalculate_weight_config
    )
    # Note: LMHead doesn't take seq_len in model_config - it computes program config dynamically
    model_config = get_model_config(LMHead, mode, hf_config, mesh_device, input_row_idx)
    model_state = LMHead.create_state(hf_config, mesh_device, None)  # CCL not needed for linear only
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert input to TTNN - replicate to all devices
    tt_input = ttnn.from_torch(
        torch_input_padded,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=run_config["input_memory_config"],
        layout=ttnn.TILE_LAYOUT,
    )

    return run_config, tt_input, reference_output, batch_size


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        ("decode", 1, 0.97, 1.0, 1.0, 0.0),  # batch=32, seq=1 → 32 tokens
        ("prefill", 128, 0.97, 1.0, 1.0, 0.0),  # batch=32, seq=128 → 4096 tokens
        pytest.param(
            "prefill",
            1024,
            0.97,
            1.0,
            1.0,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),  # batch=32, seq=1024 → 32768 tokens
        pytest.param(
            "prefill",
            8192,
            0.97,
            1.0,
            1.0,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            32768,
            0.97,
            1.0,
            1.0,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            131072,
            0.97,
            1.0,
            1.0,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),  # batch=32, seq=128k
    ],
)
@pytest.mark.parametrize(
    "use_real_weights",
    [True, pytest.param(False, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"))],
    ids=["real_weights", "random_weights"],
)
@pytest.mark.parametrize(
    "program_cache_enabled",
    [True, pytest.param(False, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"))],
    ids=["program_cache", "no_program_cache"],
)
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 23740416,  # Large trace region for vocab projection
        }
    ],
    indirect=True,
)
def test_ds_lm_head(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    use_real_weights,
    program_cache_enabled,
    trace_mode,
    hf_config,
    cache_path,
    mesh_device,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict: dict[str, torch.Tensor],
):
    """Test lm_head fused op (vocabulary projection).

    This tests the linear projection from hidden_size (7168) to vocab_size (129280).
    The weight is sharded across all 32 devices with bfloat4_b quantization.
    """
    # CI skip logic: only run specific combinations in CI
    in_ci = os.getenv("CI") == "true"
    if in_ci:
        # Only run these combinations in CI:
        # - decode + seq_len=1 + trace + program_cache + real_weights
        # - prefill + seq_len=128 + eager + program_cache + real_weights
        keep_in_ci = (
            mode == "decode" and seq_len == 1 and trace_mode and program_cache_enabled and use_real_weights
        ) or (mode == "prefill" and seq_len == 128 and not trace_mode and program_cache_enabled and use_real_weights)
        if not keep_in_ci:
            pytest.skip(
                "Skip in CI - only run decode/1/trace and prefill/128/eager with program_cache and real_weights"
            )

    # Trace capture requires program cache enabled
    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace mode requires program cache enabled (skip trace + no_program_cache).")

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    run_config, tt_input, ref_output, batch_size = _build_lm_head_inputs(
        mesh_device,
        hf_config,
        cache_path,
        force_recalculate_weight_config,
        use_real_weights,
        mode,
        seq_len,
        state_dict if use_real_weights else None,
    )

    _run_ds_lm_head_test(
        mesh_device,
        run_config,
        tt_input,
        ref_output,
        expected_pcc,
        expected_atol,
        expected_rtol,
        expected_perf_us,
        trace_mode,
        program_cache_enabled,
        mode,
        seq_len,
        batch_size,
        f"ds_lm_head_{mode}_seq{seq_len}",
    )

    # Cleanup
    ttnn.deallocate(tt_input)


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        ("decode", 1, 0.97, 1.0, 1.0, 0.0),
        ("prefill", 128, 0.97, 1.0, 1.0, 0.0),
        ("prefill", 1024, 0.97, 1.0, 1.0, 0.0),
        ("prefill", 131072, 0.97, 1.0, 1.0, 0.0),
    ],
)
@pytest.mark.parametrize("use_real_weights", [True, False], ids=["real_weights", "random_weights"])
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 23740416,
        }
    ],
    indirect=True,
)
def test_ds_lm_head_single_device(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    use_real_weights,
    program_cache_enabled,
    trace_mode,
    hf_config,
    cache_path,
    mesh_device,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict: dict[str, torch.Tensor],
):
    """Single device test for lm_head.

    The lm_head linear projection can run on a single device with per-device
    weight shard (vocab/32). However, the output would only be 1/32 of the full vocab.
    For simplicity, we skip this test as the multi-device test is the primary use case.
    """
    pytest.skip(
        "Single-device test skipped: lm_head outputs vocab_size/32 per device. "
        "Multi-device test with all_gather is the primary use case."
    )


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
        ("prefill", 128),
        pytest.param("prefill", 1024, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
        pytest.param("prefill", 8192, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
        pytest.param("prefill", 32768, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
        pytest.param("prefill", 131072, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
    ],
)
def test_ds_lm_head_device_perf(mode, seq_len):
    maybe_skip_long_seq(seq_len, LONG_SEQ_ENV_VAR)

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    # batch_size=32 (USERS_PER_ROW) for all modes
    batch_size = USERS_PER_ROW

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_lm_head_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_lm_head.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len} and real_weights"
    command = f'pytest {test_path}::test_ds_lm_head -k "{expr}"'

    perf_profiler.start("run")
    perf_profiler.start(step_name)
    os.environ[DEVICE_PERF_ENV_VAR] = "1"
    op_stats, total_kernel_ns, total_op_to_op_ns = collect_device_perf(
        command,
        subdir="deepseek_v3_fused_ops_device_perf",
        warmup_iters=0,
        use_signposts=True,
    )
    os.environ.pop(DEVICE_PERF_ENV_VAR, None)
    perf_profiler.end(step_name)
    perf_profiler.end("run")

    assert op_stats, "No device perf stats captured."
    total_kernel_us = total_kernel_ns / 1000.0
    total_op_to_op_us = total_op_to_op_ns / 1000.0
    logger.info(f"Device perf per-op averages (ns): {json.dumps(op_stats, indent=2)}")
    logger.info(f"Device perf totals: kernel={total_kernel_us:.3f} us, op_to_op={total_op_to_op_us:.3f} us")
    assert total_kernel_ns > 0, "Total kernel duration must be positive."
    assert total_op_to_op_ns >= 0, "Total op-to-op latency must be non-negative."
    targets = DEVICE_PERF_TARGETS_US.get((mode, seq_len))
    if targets is None or targets["kernel"] == 0.0:
        logger.warning("No device perf targets configured; skipping perf assertions.")
    else:
        kernel_target_us = targets["kernel"]
        op_to_op_target_us = targets["op_to_op"]
        kernel_limit_us = kernel_target_us * (1 + DEVICE_PERF_MARGIN)
        op_to_op_limit_us = op_to_op_target_us * (1 + DEVICE_PERF_MARGIN)
        assert (
            total_kernel_us <= kernel_limit_us
        ), f"Kernel perf regression: {total_kernel_us:.3f}us exceeds {kernel_target_us:.3f}us (+{DEVICE_PERF_MARGIN:.0%})"
        assert (
            total_op_to_op_us <= op_to_op_limit_us
        ), f"Op-to-op perf regression: {total_op_to_op_us:.3f}us exceeds {op_to_op_target_us:.3f}us (+{DEVICE_PERF_MARGIN:.0%})"

    benchmark_data.add_measurement(
        perf_profiler,
        0,
        step_name,
        "total_kernel_duration_us",
        total_kernel_us,
    )
    benchmark_data.add_measurement(
        perf_profiler,
        0,
        step_name,
        "total_op_to_op_latency_us",
        total_op_to_op_us,
    )
    benchmark_data.save_partial_run_json(
        perf_profiler,
        run_type="deepseek_v3_fused_ops_device_perf",
        ml_model_name="deepseek-v3",
        batch_size=batch_size,
        input_sequence_length=seq_len,
    )


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
        ("prefill", 128),
        ("prefill", 1024),
    ],
)
def test_ds_lm_head_single_device_device_perf(mode, seq_len):
    pytest.skip(
        "Single-device device perf test skipped: lm_head outputs vocab_size/32 per device. "
        "Multi-device test with all_gather is the primary use case."
    )


if __name__ == "__main__":
    pytest.main([__file__])
