# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3.tests.fused_op_unit_tests.test_utils import (
    collect_device_perf,
    compare_with_reference,
    get_int_env,
    log_run_mode,
    maybe_skip_long_seq,
    measure_perf_us,
)
from models.demos.deepseek_v3.tt.embedding.embedding1d import Embedding1D
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, even_int_div
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    get_model_config,
    get_test_weight_config,
    system_name_to_mesh_shape,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

LONG_SEQ_ENV_VAR = "DEEPSEEK_V3_LONG_SEQ_TESTS"
DEVICE_PERF_ENV_VAR = "DS_ALL_GATHER_EMBEDDING_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US = {
    ("decode", 1): {"kernel": 30.404, "op_to_op": 788.854},  # Measured: kernel=27.64, op_to_op=717.14
    ("prefill", 128): {"kernel": 46.585, "op_to_op": 754.083},  # Measured: kernel=42.35, op_to_op=685.53
}


def ds_all_gather_embedding_reference(
    x: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    """
    Reference implementation for AllGather in embedding module.

    The all_gather in embedding gathers across cluster_axis=0 (mesh rows) on dim=-1.
    Input per device: [1, 1, batch, per_device_hidden] where per_device_hidden = hidden_size/32
    Output per device: [1, 1, batch, per_row_hidden] where per_row_hidden = per_device_hidden * num_rows

    In the reference model (without tensor parallelism), this simulates gathering
    data from all rows by concatenating along the last dimension.

    Args:
        x: Input tensor of shape [1, 1, batch, per_device_hidden * num_rows]
           representing the full data that would be gathered from all rows
        num_rows: Number of mesh rows (4 for TG)

    Returns:
        Output tensor (same as input in reference model since we simulate full data)
    """
    # In reference, input already contains the full gathered data
    return x


def ds_all_gather_embedding_ttnn(
    x: ttnn.Tensor,
    cfg: dict,
    ccl,
    persistent_output_buffer: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """
    TTNN implementation for AllGather in embedding module.

    This performs an all-gather operation across mesh rows (cluster_axis=0)
    to collect embedding data from all rows after the embedding lookup.

    Input per device: [1, 1, batch, 224] (224 = hidden_size/32)
    Output per device: [1, 1, batch, 896] (896 = 224 * 4 rows)

    Args:
        x: Input tensor sharded across devices
        cfg: Configuration dictionary containing all_gather config
        ccl: CCL runtime object
        persistent_output_buffer: Optional persistent output for trace mode (ignored, kept for backward compatibility)

    Returns:
        Output tensor after all-gather
    """
    return Embedding1D._fwd_all_gather_embedding(x, cfg, ccl)


def _run_ds_all_gather_embedding_test(
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
    ccl,
    step_prefix: str,
):
    # Log run configuration for superset
    log_run_mode(mode, trace_mode, program_cache_enabled, seq_len)

    # Log config for verification
    logger.info(f"=== ALL_GATHER EMBEDDING OP CONFIG VERIFICATION ===")
    logger.info(f"Input shape: {tt_input.shape}")
    logger.info(f"Input memory_config: {tt_input.memory_config()}")
    logger.info(f"AllGather config: {run_config['all_gather']}")
    logger.info(f"=== END CONFIG VERIFICATION ===")

    tt_output = ds_all_gather_embedding_ttnn(tt_input, run_config, ccl)

    # After all_gather on cluster_axis=0, data is gathered across rows
    # Each device now has [1, 1, batch, 896] (896 = 224 * 4)
    # Take output from first device for comparison
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])

    logger.info(f"tt_output_torch shape: {tt_output_torch.shape}")
    logger.info(f"ref_output shape: {ref_output.shape}")

    pcc_value, max_abs_error = compare_with_reference(
        tt_output_torch, ref_output, expected_pcc, expected_atol, expected_rtol, convert_to_float=True
    )

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        warmup_iters = get_int_env("DS_ALLGATHER_EMBEDDING_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = get_int_env("DS_ALLGATHER_EMBEDDING_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)
        logger.info(
            f"Starting e2e perf measurement: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}"
        )

        perf_profiler.start("run")
        perf_profiler.start(step_name)

        def op_fn(*, persistent_output_buffer=None):
            return ds_all_gather_embedding_ttnn(
                tt_input, run_config, ccl, persistent_output_buffer=persistent_output_buffer
            )

        perf_us = measure_perf_us(
            mesh_device,
            op_fn,
            warmup_iters,
            measure_iters,
            trace_mode=trace_mode,
            profiler_name="ds_all_gather_embedding_perf",
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
        # Log PCC and ATOL metrics to superset
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-pcc", pcc_value)
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-max_abs_error", max_abs_error)
        benchmark_data.save_partial_run_json(
            perf_profiler,
            run_type="deepseek_v3_fused_ops",
            ml_model_name="deepseek-v3",
            batch_size=batch_size,
            input_sequence_length=seq_len,
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

        def op_fn(*, persistent_output_buffer=None):
            return ds_all_gather_embedding_ttnn(
                tt_input, run_config, ccl, persistent_output_buffer=persistent_output_buffer
            )

        for _ in range(PERF_WARMUP_ITERS):
            output = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            # Trace mode: use a persistent output buffer
            persistent_output = op_fn()
            ttnn.synchronize_device(mesh_device)
            _ = op_fn(persistent_output_buffer=persistent_output)
            ttnn.synchronize_device(mesh_device)

            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            op_fn(persistent_output_buffer=persistent_output)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                ttnn.execute_trace(mesh_device, trace_id, blocking=False)
                ttnn.synchronize_device(mesh_device)
            signpost("stop")
            ttnn.release_trace(mesh_device, trace_id)
            ttnn.deallocate(persistent_output)
        else:
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                output = op_fn()
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(output)
            signpost("stop")


def _build_all_gather_embedding_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config,
    cache_path: str,
    ccl,
    force_recalculate_weight_config: bool,
    mode: str,
    seq_len: int,
):
    from models.demos.deepseek_v3.tt.embedding.embedding1d import Embedding1D

    # Generate random embedding state dict for weight config
    embedding_state_dict = {"weight": torch.randn(hf_config.vocab_size, hf_config.hidden_size, dtype=torch.float32)}

    weight_config = get_test_weight_config(
        Embedding1D,
        hf_config,
        (embedding_state_dict,),
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    model_config = get_model_config(Embedding1D, mode, hf_config, mesh_device)
    model_state = Embedding1D.create_state(hf_config, mesh_device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state)

    batch_size = USERS_PER_ROW  # Always 32 for all modes
    num_rows, num_cols = mesh_device.shape
    num_devices = mesh_device.get_num_devices()

    # Per-device hidden dimension
    # hidden_size is split across all 32 devices
    per_device_hidden = even_int_div(hf_config.hidden_size, num_devices)  # 224 = 7168/32

    # After all_gather on cluster_axis=0, the hidden dimension is gathered across rows
    # So output per device has: per_device_hidden * num_rows = 224 * 4 = 896
    per_row_hidden = per_device_hidden * num_rows  # 896

    input_seq_len = seq_len

    torch_full_output = torch.randn(1, 1, input_seq_len, per_row_hidden, dtype=torch.bfloat16)

    # Reference output is the full gathered data
    ref_output = torch_full_output

    torch_input_per_row = torch.zeros(num_rows, 1, input_seq_len, per_device_hidden, dtype=torch.bfloat16)
    for r in range(num_rows):
        torch_input_per_row[r] = torch_full_output[:, :, :, r * per_device_hidden : (r + 1) * per_device_hidden]

    tt_input = ttnn.from_torch(
        torch_input_per_row,
        device=mesh_device,
        # Shard dim 0 across rows (4 rows), replicate across columns (8 cols)
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    return run_config, tt_input, ref_output, batch_size, input_seq_len


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # batch_size=32 for all modes
        ("decode", 1, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 128, 0.9999, 0.2, 0.2, 0.0),
        pytest.param(
            "prefill",
            1024,
            0.9999,
            0.2,
            0.2,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            8192,
            0.9999,
            0.2,
            0.2,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            32768,
            0.9999,
            0.2,
            0.2,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            131072,
            0.9999,
            0.2,
            0.2,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
    ],
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
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 2967552,
        }
    ],
    indirect=True,
)
def test_ds_all_gather_embedding(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    program_cache_enabled,
    trace_mode,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    # Trace capture replays pre-compiled binaries. When program cache is disabled, ops may
    # trigger compilation/program writes during capture, which is forbidden and can TT_FATAL.
    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace mode requires program cache enabled (skip trace + no_program_cache).")

    if mode == "decode":
        assert seq_len == 1, "Decode mode uses seq_len=1, batch=32"
    else:
        assert mode == "prefill", "Unsupported mode"
        # Skip removed: now handled by pytest.param marks for seq_len > 2048

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    run_config, tt_input, ref_output, batch_size, original_seq_len = _build_all_gather_embedding_inputs(
        mesh_device,
        hf_config,
        cache_path,
        ccl,
        force_recalculate_weight_config,
        mode,
        seq_len,
    )
    _run_ds_all_gather_embedding_test(
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
        ccl,
        f"ds_all_gather_embedding_{mode}_seq{seq_len}",
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
def test_ds_all_gather_embedding_device_perf(mode, seq_len):
    if mode == "decode":
        assert seq_len == 1, "Decode mode uses seq_len=1, batch=32"
    else:
        assert mode == "prefill", "Unsupported mode"
        maybe_skip_long_seq(seq_len, LONG_SEQ_ENV_VAR, threshold=2048)

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    batch_size = USERS_PER_ROW * mesh_shape[0]

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_all_gather_embedding_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_all_gather_embedding.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    # Use substring matching in-k filter to select the right test variant
    # This matches test IDs like: test_name[prefill-128-...-program_cache-eager]
    command = f'pytest {test_path}::test_ds_all_gather_embedding -k "{mode}-{seq_len}"'

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


if __name__ == "__main__":
    pytest.main([__file__])
