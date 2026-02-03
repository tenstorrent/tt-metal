# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
from collections import defaultdict
from copy import deepcopy

import pandas as pd
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, profiler
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    get_model_config,
    get_test_weight_config,
    system_name_to_mesh_shape,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

DEVICE_PERF_ENV_VAR = "DS_MOE_ALL_GATHER_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
TEST_CHECK_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US: dict[tuple[str, int], dict[str, float]] = {}
CI_ACTIVE = os.getenv("CI") == "true"

_TRACE_REQUIRES_CACHE_MARK = pytest.mark.skip(reason="Trace mode requires program cache to be enabled.")
_CI_SKIP_MARK = pytest.mark.skipif(
    CI_ACTIVE,
    reason="CI runs only decode/prefill-128 with program_cache+trace+real_weights coverage.",
)

# Single-device tests are not applicable for ops that include CCLs.

_CI_FOCUSED_SKIP_MARK = pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="CI runs focused coverage only.",
)


def ds_moe_all_gather_reference(torch_input: torch.Tensor, mesh_rows: int) -> torch.Tensor:
    assert torch_input.ndim == 3, f"Expected [1, tokens, hidden], got {torch_input.shape}"
    assert torch_input.shape[1] % mesh_rows == 0, "Token dimension must be divisible by mesh rows"
    tokens_per_row = torch_input.shape[1] // mesh_rows
    return torch_input[:, :tokens_per_row, :].unsqueeze(1)


def ds_moe_all_gather_ttnn(x: ttnn.Tensor, cfg: dict, ccl) -> ttnn.Tensor:
    # Reuse the module helper to mirror the in-model call sequence.
    from models.demos.deepseek_v3.tt.moe import MoE

    return MoE._fwd_all_gather(x, cfg)


def _compare_with_reference(
    tt_output: torch.Tensor, ref_output: torch.Tensor, expected_pcc: float, atol: float, rtol: float
):
    passing, pcc = comp_pcc(ref_output, tt_output, expected_pcc)
    logger.info(f"PCC: {pcc}")
    if not torch.isfinite(ref_output).all() or not torch.isfinite(tt_output).all():
        ref_nonfinite = (~torch.isfinite(ref_output)).sum().item()
        tt_nonfinite = (~torch.isfinite(tt_output)).sum().item()
        logger.warning(f"Non-finite values detected: ref={ref_nonfinite}, tt={tt_nonfinite}")
    abs_diff = torch.abs(tt_output - ref_output)
    max_abs_diff = abs_diff.max().item() if abs_diff.numel() > 0 else 0.0
    rel_diff = abs_diff / (torch.abs(ref_output) + 1e-12)
    max_rel_diff = rel_diff.max().item() if rel_diff.numel() > 0 else 0.0
    logger.info(f"Max abs diff: {max_abs_diff}, Max rel diff: {max_rel_diff}")
    assert passing, f"PCC {pcc} is below required {expected_pcc}"
    torch.testing.assert_close(tt_output, ref_output, rtol=rtol, atol=atol)


def _measure_perf_us(
    mesh_device: ttnn.MeshDevice, op_fn, warmup_iters: int, measure_iters: int, trace_mode: bool = False
) -> float:
    ttnn.synchronize_device(mesh_device)
    if trace_mode:
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

        if warmup_iters > 0:
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for _ in range(warmup_iters):
                output = op_fn()
                ttnn.deallocate(output)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(measure_iters):
            output = op_fn()
            ttnn.deallocate(output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        profiler.clear()
        profiler.start("ds_moe_all_gather_perf")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        profiler.end("ds_moe_all_gather_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get("ds_moe_all_gather_perf") * 1e6

    for _ in range(warmup_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

    profiler.clear()
    profiler.start("ds_moe_all_gather_perf")
    for _ in range(measure_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)
    profiler.end("ds_moe_all_gather_perf", PERF_CNT=measure_iters)
    return profiler.get("ds_moe_all_gather_perf") * 1e6


def _merge_device_rows_for_perf(df: pd.DataFrame) -> pd.DataFrame:
    block_by_device = defaultdict(list)

    for _, row in df.iterrows():
        op_name = row["OP CODE"]
        op_type = row["OP TYPE"]

        if op_type == "tt_dnn_device":
            device_id = int(row["DEVICE ID"])
            block_by_device[device_id].append((op_name, row.to_dict()))

    device_ids = sorted(block_by_device.keys())
    merged_blocks = []
    global_index = 0
    while max(len(block_by_device[device_id]) for device_id in device_ids) > 0:
        blocks = []
        op_name = None
        missing_devices = []
        for device_id in device_ids:
            if not len(block_by_device[device_id]):
                logger.warning(f"Warning: Device {device_id} is missing operation {op_name} at index {global_index}")
                continue
            if op_name is None:
                op_name = block_by_device[device_id][0][0]
            elif op_name != block_by_device[device_id][0][0]:
                missing_devices.append(device_id)
                continue

            blocks.append(block_by_device[device_id].pop(0))

        if missing_devices:
            logger.warning(
                f"Warning: {op_name} at index {global_index} not present in CSV for {len(missing_devices)} devices {missing_devices} - do not trust data for this op or directly subsequent ops with the same name"
            )

        if not blocks:
            break

        is_collective = any(tag in op_name for tag in ("AllGather", "ReduceScatter", "AllReduce", "AllToAll"))
        if is_collective:
            device_kernel_durations = [
                d["DEVICE KERNEL DURATION [ns]"]
                for _, d in blocks
                if "DEVICE KERNEL DURATION [ns]" in d and not math.isnan(d["DEVICE KERNEL DURATION [ns]"])
            ]
            average_duration = (
                sum(device_kernel_durations) / len(device_kernel_durations) if device_kernel_durations else float("nan")
            )
            base_block = blocks[0][1].copy()
            base_block["DEVICE KERNEL DURATION [ns]"] = average_duration
            merged_blocks.append(base_block)
        else:
            max_duration_block = max(blocks, key=lambda x: x[1]["DEVICE KERNEL DURATION [ns]"])
            merged_blocks.append(max_duration_block[1])

        global_index += 1

    return pd.DataFrame(merged_blocks)


def _collect_device_perf(
    command: str, subdir: str, warmup_iters: int, use_signposts: bool = False
) -> tuple[dict[str, dict[str, float]], float, float]:
    device_analysis_types = ["device_kernel_duration"]
    run_device_profiler(
        command,
        subdir,
        device_analysis_types=device_analysis_types,
        op_support_count=10000,
    )
    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)

    if use_signposts:
        markers = df[df["OP TYPE"] == "signpost"]["OP CODE"]
        assert not markers.empty, "No signposts found in device perf log."
        start_indices = markers[markers == "start"].index
        stop_indices = markers[markers == "stop"].index
        assert not start_indices.empty, "Missing signpost 'start' in device perf log."
        assert not stop_indices.empty, "Missing signpost 'stop' in device perf log."
        start_idx = start_indices[0]
        stop_idx = stop_indices[-1]
        assert start_idx < stop_idx, "Signpost 'stop' must come after 'start'."
        df = df.iloc[start_idx + 1 : stop_idx]

    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]
    df = _merge_device_rows_for_perf(df)

    required_cols = ["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    assert not missing_cols, f"Missing device perf columns: {missing_cols}"

    df["DEVICE KERNEL DURATION [ns]"] = pd.to_numeric(df["DEVICE KERNEL DURATION [ns]"], errors="coerce").fillna(0.0)
    df["OP TO OP LATENCY [ns]"] = pd.to_numeric(df["OP TO OP LATENCY [ns]"], errors="coerce").fillna(0.0)

    op_stats: dict[str, dict[str, float]] = {}
    for op_code, group in df.groupby("OP CODE"):
        kernel_vals = group["DEVICE KERNEL DURATION [ns]"].tolist()
        op_to_op_vals = group["OP TO OP LATENCY [ns]"].tolist()
        if warmup_iters > 0:
            kernel_vals = kernel_vals[warmup_iters:]
            op_to_op_vals = op_to_op_vals[warmup_iters:]
        assert kernel_vals, f"No kernel duration samples for op {op_code}"
        assert op_to_op_vals, f"No op-to-op latency samples for op {op_code}"
        op_stats[op_code] = {
            "avg_kernel_duration_ns": sum(kernel_vals) / len(kernel_vals),
            "avg_op_to_op_latency_ns": sum(op_to_op_vals) / len(op_to_op_vals),
        }

    total_kernel_ns = sum(entry["avg_kernel_duration_ns"] for entry in op_stats.values())
    total_op_to_op_ns = sum(entry["avg_op_to_op_latency_ns"] for entry in op_stats.values())
    return op_stats, total_kernel_ns, total_op_to_op_ns


def _build_moe_all_gather_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config_short,
    cache_path: str,
    ccl,
    force_recalculate_weight_config: bool,
    state_dict: dict,
    mode: str,
    seq_len: int,
    use_real_weights: bool,
    topk_fallback: bool,
):
    from models.demos.deepseek_v3.tt.moe import MoE

    hf_config_local = deepcopy(hf_config_short)
    hf_config_local.n_shared_experts = None

    if use_real_weights:
        moe_layer_idx = hf_config_local.first_k_dense_replace
        module_path = f"model.layers.{moe_layer_idx}.mlp"
        module_state_dict = sub_state_dict(state_dict, module_path + ".")
        weight_state_dict = module_state_dict
    else:
        reference_model = DeepseekV3MoE(hf_config_local).eval()
        weight_state_dict = add_inv_scale_to_state_dict(
            reference_model.state_dict(),
            block_shape=hf_config_local.quantization_config["weight_block_size"],
        )

    weight_config = get_test_weight_config(
        MoE,
        hf_config_local,
        (weight_state_dict,),
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    model_config = get_model_config(MoE, mode, hf_config_local, mesh_device, topk_fallback=topk_fallback)
    model_state = MoE.create_state(hf_config_local, mesh_device, ccl)
    model_shared_state = MoE.create_shared_state(hf_config_local, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    mesh_rows = mesh_device.shape[0]
    if mode == "decode":
        num_tokens = USERS_PER_ROW * mesh_rows
        batch_size = num_tokens
    else:
        num_tokens = seq_len
        batch_size = 1
        assert num_tokens % mesh_rows == 0, f"prefill seq_len {num_tokens} must be divisible by mesh rows {mesh_rows}"

    torch_input = torch.randn(1, num_tokens, hf_config_local.hidden_size, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    return run_config, tt_input, torch_input, batch_size


def _run_ds_moe_all_gather_test(
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
    tt_output = None
    for iter_idx in range(TEST_CHECK_ITERS):
        tt_output = ds_moe_all_gather_ttnn(tt_input, run_config, ccl)
        if iter_idx == TEST_CHECK_ITERS - 1:
            tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])
            _compare_with_reference(tt_output_torch, ref_output, expected_pcc, expected_atol, expected_rtol)
        ttnn.deallocate(tt_output)

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        perf_profiler.start("run")
        perf_profiler.start(step_name)
        perf_us = _measure_perf_us(
            mesh_device,
            lambda: ds_moe_all_gather_ttnn(tt_input, run_config, ccl),
            PERF_WARMUP_ITERS,
            PERF_MEASURE_ITERS,
            trace_mode=trace_mode,
        )
        logger.info(f"Perf avg: {perf_us:.3f} us over {PERF_MEASURE_ITERS} iters (warmup {PERF_WARMUP_ITERS})")
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
        from tracy import signpost

        def op_fn():
            return ds_moe_all_gather_ttnn(tt_input, run_config, ccl)

        for _ in range(PERF_WARMUP_ITERS):
            output = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            output = op_fn()
            ttnn.deallocate(output)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                ttnn.execute_trace(mesh_device, trace_id, blocking=False)
                ttnn.synchronize_device(mesh_device)
            signpost("stop")
            ttnn.release_trace(mesh_device, trace_id)
        else:
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                output = op_fn()
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(output)
            signpost("stop")


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        ("decode", 1, 1.0, 0.2, 0.2, 1160.944),
        ("prefill", 128, 1.0, 0.2, 0.2, 820.322),
        pytest.param("prefill", 1024, 1.0, 0.2, 0.2, 820.322, marks=_CI_SKIP_MARK, id="prefill-1024"),
        pytest.param("prefill", 8192, 1.0, 0.2, 0.2, 820.322, marks=_CI_SKIP_MARK, id="prefill-8192"),
        pytest.param("prefill", 32768, 1.0, 0.2, 0.2, 820.322, marks=_CI_SKIP_MARK, id="prefill-32768"),
        pytest.param("prefill", 131072, 1.0, 0.2, 0.2, 820.322, marks=_CI_SKIP_MARK, id="prefill-131072"),
    ],
)
@pytest.mark.parametrize(
    "program_cache_enabled, trace_mode",
    [
        pytest.param(False, False, marks=_CI_FOCUSED_SKIP_MARK),
        (True, True),
    ],
)
@pytest.mark.parametrize(
    "use_real_weights",
    [
        True,
        pytest.param(False, marks=_CI_FOCUSED_SKIP_MARK),
    ],
    ids=["real_weights", "random_weights"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 10485760,
        }
    ],
    indirect=True,
)
def test_ds_moe_all_gather(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    program_cache_enabled,
    trace_mode,
    use_real_weights,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict,
):
    if mode == "decode":
        assert seq_len == 1, "Decode only supports seq_len=1"
    else:
        assert mode == "prefill", "Unsupported mode"

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    run_config, tt_input, torch_input, batch_size = _build_moe_all_gather_inputs(
        mesh_device,
        hf_config_short,
        cache_path,
        ccl,
        force_recalculate_weight_config,
        state_dict,
        mode,
        seq_len,
        use_real_weights,
        topk_fallback=True,
    )

    ref_output = ds_moe_all_gather_reference(torch_input, mesh_device.shape[0])

    _run_ds_moe_all_gather_test(
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
        f"ds_moe_all_gather_{mode}_seq{seq_len}",
    )

    ttnn.deallocate(tt_input)


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
        ("prefill", 128),
        pytest.param(
            "prefill",
            131072,
            marks=[_CI_SKIP_MARK, _CI_FOCUSED_SKIP_MARK],
            id="prefill-131072",
        ),
    ],
)
def test_ds_moe_all_gather_device_perf(mode, seq_len):
    if mode == "decode":
        assert seq_len == 1, "Decode only supports seq_len=1"
    else:
        assert mode == "prefill", "Unsupported mode"

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    batch_size = USERS_PER_ROW * mesh_shape[0] if mode == "decode" else 1

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_moe_all_gather_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_all_gather.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    # Select a concrete test variant that exists in the parametrized ids.
    trace_cache_token = "True-True" if mode == "decode" else "False-False"
    mode_token = f"{mode}-{seq_len}"
    expr = f"{trace_cache_token} and {mode_token} and real_weights"
    command = f'pytest {test_path}::test_ds_moe_all_gather -k "{expr}"'

    profiler.start("run")
    profiler.start(step_name)
    os.environ[DEVICE_PERF_ENV_VAR] = "1"
    op_stats, total_kernel_ns, total_op_to_op_ns = _collect_device_perf(
        command,
        subdir="deepseek_v3_fused_ops_device_perf",
        warmup_iters=0,
        use_signposts=True,
    )
    os.environ.pop(DEVICE_PERF_ENV_VAR, None)
    profiler.end(step_name)
    profiler.end("run")

    assert op_stats, "No device perf stats captured."
    total_kernel_us = total_kernel_ns / 1000.0
    total_op_to_op_us = total_op_to_op_ns / 1000.0
    logger.info(f"Device perf per-op averages (ns): {json.dumps(op_stats, indent=2)}")
    logger.info(f"Device perf totals: kernel={total_kernel_us:.3f} us, op_to_op={total_op_to_op_us:.3f} us")
    assert total_kernel_ns > 0, "Total kernel duration must be positive."
    assert total_op_to_op_ns >= 0, "Total op-to-op latency must be non-negative."
    targets = DEVICE_PERF_TARGETS_US.get((mode, seq_len))
    if targets is None:
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
        profiler,
        0,
        step_name,
        "total_kernel_duration_us",
        total_kernel_us,
    )
    benchmark_data.add_measurement(
        profiler,
        0,
        step_name,
        "total_op_to_op_latency_us",
        total_op_to_op_us,
    )
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="deepseek_v3_fused_ops_device_perf",
        ml_model_name="deepseek-v3",
        batch_size=batch_size,
        input_sequence_length=seq_len,
    )
