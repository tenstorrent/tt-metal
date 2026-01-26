# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
from collections import defaultdict

import pandas as pd
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, profiler
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate
from models.demos.deepseek_v3.tt.moe_gate import MoEGate
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    dequantize_state_dict,
    get_model_config,
    get_test_weight_config,
    system_name_to_mesh_shape,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

DEVICE_PERF_ENV_VAR = "DS_MOE_EXPERT_SELECTION_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
TEST_CHECK_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US: dict[tuple[str, int], dict[str, float]] = {}

_CI_FOCUSED_SKIP_MARK = pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="CI runs focused coverage only.",
)

_TRACE_REQUIRES_CACHE_MARK = pytest.mark.skip(reason="Trace mode requires program cache to be enabled.")


def ds_moe_gate_projection_scores_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    logits = torch.nn.functional.linear(x.to(torch.float32), weight.to(torch.float32))
    scores = torch.sigmoid(logits)
    return scores.to(torch.bfloat16)


def ds_moe_expert_selection_reference(
    scores: torch.Tensor,
    scores_with_bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    topk_fn,
    epsilon: float = 1e-20,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim == 3:
        scores = scores.unsqueeze(1)
    if scores_with_bias.ndim == 3:
        scores_with_bias = scores_with_bias.unsqueeze(1)

    num_experts = scores.shape[-1]
    experts_per_group = num_experts // n_group

    grouped_scores = scores_with_bias.reshape(scores.shape[0], scores.shape[2], n_group, experts_per_group)
    topk_scores_within_groups, _ = topk_fn(grouped_scores, k=2, dim=-1, sorted=True)
    group_scores = topk_scores_within_groups.sum(dim=-1)

    _, topk_group_indices = topk_fn(group_scores, k=topk_group, dim=-1, sorted=True)
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(-1, topk_group_indices, True)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, -1, -1, experts_per_group)
        .reshape(scores.shape[0], scores.shape[2], num_experts)
    )

    scores_for_choice = scores_with_bias.squeeze(1).to(torch.float32)
    masked_scores = scores_for_choice.masked_fill(~score_mask, float("-inf")).unsqueeze(1)

    _, topk_indices = topk_fn(masked_scores, k=num_experts_per_tok, dim=-1, sorted=True)
    gathered_scores = torch.gather(scores.to(torch.float32), dim=-1, index=topk_indices)
    normalized_scores = gathered_scores / (gathered_scores.sum(dim=-1, keepdim=True) + epsilon)
    scaled_scores = normalized_scores * routed_scaling_factor

    return topk_indices, scaled_scores.to(torch.bfloat16)


def ds_moe_expert_selection_ttnn(
    scores: ttnn.Tensor,
    scores_with_bias: ttnn.Tensor,
    cfg: dict,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    return MoEGate._fwd_expert_selection(scores, scores_with_bias, cfg)


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

    def _deallocate_output(output) -> None:
        if isinstance(output, (tuple, list)):
            for item in output:
                ttnn.deallocate(item)
        else:
            ttnn.deallocate(output)

    if trace_mode:
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        _deallocate_output(output)

        if warmup_iters > 0:
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for _ in range(warmup_iters):
                output = op_fn()
                _deallocate_output(output)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(measure_iters):
            output = op_fn()
            _deallocate_output(output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        profiler.clear()
        profiler.start("ds_moe_expert_selection_perf")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        profiler.end("ds_moe_expert_selection_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get("ds_moe_expert_selection_perf") * 1e6

    for _ in range(warmup_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        _deallocate_output(output)

    profiler.clear()
    profiler.start("ds_moe_expert_selection_perf")
    for _ in range(measure_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        _deallocate_output(output)
    profiler.end("ds_moe_expert_selection_perf", PERF_CNT=measure_iters)
    return profiler.get("ds_moe_expert_selection_perf") * 1e6


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


def _build_moe_expert_selection_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config_short,
    cache_path: str,
    force_recalculate_weight_config: bool,
    state_dict: dict,
    mode: str,
    seq_len: int,
    use_real_weights: bool,
):
    if use_real_weights:
        moe_layer_idx = hf_config_short.first_k_dense_replace
        module_path = f"model.layers.{moe_layer_idx}.mlp.gate"
        module_state_dict = sub_state_dict(state_dict, module_path + ".")
        reference_model = ReferenceMoEGate(hf_config_short, use_bitonic_sort=True).eval()
        reference_model.load_state_dict(dequantize_state_dict(module_state_dict, hf_config_short))
        weight_state_dict = module_state_dict
    else:
        reference_model = ReferenceMoEGate(hf_config_short, use_bitonic_sort=True).eval()
        weight_state_dict = reference_model.state_dict()

    reference_model = reference_model.to(torch.bfloat16)

    weight_config = get_test_weight_config(
        MoEGate,
        hf_config_short,
        (weight_state_dict,),
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    model_config = get_model_config(
        MoEGate,
        mode,
        hf_config_short,
        mesh_device,
        topk_fallback=True,
        use_bitonic_sort=True,
    )
    model_state = MoEGate.create_state(hf_config_short, mesh_device=mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state)

    mesh_rows = mesh_device.shape[0]
    if mode == "decode":
        num_tokens = USERS_PER_ROW * mesh_rows
        batch_size = num_tokens
    else:
        num_tokens = seq_len
        batch_size = 1
        assert num_tokens % mesh_rows == 0, f"prefill seq_len {num_tokens} must be divisible by mesh rows {mesh_rows}"

    torch_input = torch.randn(1, num_tokens, hf_config_short.hidden_size, dtype=torch.bfloat16)
    torch_input_4d = torch_input.unsqueeze(1)

    tt_input = ttnn.from_torch(
        torch_input_4d,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_scores_seed = MoEGate._fwd_gate_projection_and_sigmoid(tt_input, run_config)
    ttnn.deallocate(tt_input)

    tt_scores_with_bias_seed = MoEGate._fwd_add_score_correction_bias(tt_scores_seed, run_config)

    ref_scores = ttnn.to_torch(
        tt_scores_seed,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0]
    ref_scores_with_bias = ttnn.to_torch(
        tt_scores_with_bias_seed,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0]
    ref_indices, ref_scores_norm = ds_moe_expert_selection_reference(
        ref_scores,
        ref_scores_with_bias,
        hf_config_short.n_group,
        hf_config_short.topk_group,
        hf_config_short.num_experts_per_tok,
        hf_config_short.routed_scaling_factor,
        reference_model.topk_fn,
    )

    return run_config, tt_scores_seed, tt_scores_with_bias_seed, (ref_indices, ref_scores_norm), batch_size


def _run_ds_moe_expert_selection_test(
    mesh_device: ttnn.MeshDevice,
    run_config: dict,
    tt_scores_seed: ttnn.Tensor,
    tt_scores_with_bias_seed: ttnn.Tensor,
    ref_output: tuple[torch.Tensor, torch.Tensor],
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
    ref_indices, ref_scores_norm = ref_output

    def _make_input_copy() -> tuple[ttnn.Tensor, ttnn.Tensor]:
        tt_scores = ttnn.zeros_like(tt_scores_seed)
        ttnn.copy(tt_scores_seed, tt_scores)
        tt_scores_with_bias = ttnn.zeros_like(tt_scores_with_bias_seed)
        ttnn.copy(tt_scores_with_bias_seed, tt_scores_with_bias)
        return tt_scores, tt_scores_with_bias

    last_scores_torch = None
    last_indices_torch = None
    for iter_idx in range(TEST_CHECK_ITERS):
        tt_scores, tt_scores_with_bias = _make_input_copy()
        tt_scores_norm, tt_indices = ds_moe_expert_selection_ttnn(tt_scores, tt_scores_with_bias, run_config)
        if iter_idx == TEST_CHECK_ITERS - 1:
            last_scores_torch = ttnn.to_torch(
                tt_scores_norm,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
            )[0]
            last_indices_torch = ttnn.to_torch(
                tt_indices,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
            )[0]
        ttnn.deallocate(tt_scores_norm)
        ttnn.deallocate(tt_indices)

    if last_scores_torch is None or last_indices_torch is None:
        raise AssertionError("Missing outputs for correctness check.")

    if last_scores_torch.ndim != ref_scores_norm.ndim:
        if last_scores_torch.ndim == 3 and ref_scores_norm.ndim == 4:
            last_scores_torch = last_scores_torch.unsqueeze(1)
        elif last_scores_torch.ndim == 4 and ref_scores_norm.ndim == 3:
            ref_scores_norm = ref_scores_norm.unsqueeze(1)
    _compare_with_reference(last_scores_torch, ref_scores_norm, expected_pcc, expected_atol, expected_rtol)

    if last_indices_torch.ndim != ref_indices.ndim:
        if last_indices_torch.ndim == 3 and ref_indices.ndim == 4:
            last_indices_torch = last_indices_torch.unsqueeze(1)
        elif last_indices_torch.ndim == 4 and ref_indices.ndim == 3:
            ref_indices = ref_indices.unsqueeze(1)
    sorted_ref_indices = torch.sort(ref_indices.to(torch.int64), dim=-1, stable=True)[0]
    sorted_tt_indices = torch.sort(last_indices_torch.to(torch.int64), dim=-1, stable=True)[0]
    indices_match_ratio = (sorted_ref_indices == sorted_tt_indices).float().mean().item()
    logger.info(f"Top-k indices match ratio: {indices_match_ratio:.6f}")
    assert indices_match_ratio >= 0.84, "Top-k expert indices mismatch rate exceeds 16%."

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_input_count = PERF_WARMUP_ITERS + PERF_MEASURE_ITERS + (1 if trace_mode else 0)
        perf_inputs = [_make_input_copy() for _ in range(perf_input_count)]

        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        perf_profiler.start("run")
        perf_profiler.start(step_name)
        perf_us = _measure_perf_us(
            mesh_device,
            lambda: ds_moe_expert_selection_ttnn(*perf_inputs.pop(0), run_config),
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
            tt_scores, tt_scores_with_bias = _make_input_copy()
            return ds_moe_expert_selection_ttnn(tt_scores, tt_scores_with_bias, run_config)

        for _ in range(PERF_WARMUP_ITERS):
            scores_norm, indices = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(scores_norm)
            ttnn.deallocate(indices)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            scores_norm, indices = op_fn()
            ttnn.deallocate(scores_norm)
            ttnn.deallocate(indices)
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
                scores_norm, indices = op_fn()
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(scores_norm)
                ttnn.deallocate(indices)
            signpost("stop")


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        ("decode", 1, 0.95, 0.2, 0.2, 0.0),
        ("prefill", 128, 0.966, 0.2, 0.2, 0.0),
        pytest.param(
            "prefill",
            32768,
            0.966,
            0.2,
            0.2,
            0.0,
            marks=_CI_FOCUSED_SKIP_MARK,
            id="prefill-32768",
        ),
        pytest.param(
            "prefill",
            131072,
            0.966,
            0.2,
            0.2,
            0.0,
            marks=_CI_FOCUSED_SKIP_MARK,
            id="prefill-131072",
        ),
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
def test_ds_moe_expert_selection(
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

    run_config, tt_scores_seed, tt_scores_with_bias_seed, ref_output, batch_size = _build_moe_expert_selection_inputs(
        mesh_device,
        hf_config_short,
        cache_path,
        force_recalculate_weight_config,
        state_dict,
        mode,
        seq_len,
        use_real_weights,
    )

    _run_ds_moe_expert_selection_test(
        mesh_device,
        run_config,
        tt_scores_seed,
        tt_scores_with_bias_seed,
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
        f"ds_moe_expert_selection_{mode}_seq{seq_len}",
    )

    ttnn.deallocate(tt_scores_seed)
    ttnn.deallocate(tt_scores_with_bias_seed)


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        ("decode", 1, 0.95, 0.2, 0.2, 0.0),
        ("prefill", 128, 0.966, 0.2, 0.2, 0.0),
        pytest.param(
            "prefill",
            32768,
            0.966,
            0.2,
            0.2,
            0.0,
            marks=_CI_FOCUSED_SKIP_MARK,
            id="prefill-32768",
        ),
        pytest.param(
            "prefill",
            131072,
            0.966,
            0.2,
            0.2,
            0.0,
            marks=_CI_FOCUSED_SKIP_MARK,
            id="prefill-131072",
        ),
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
def test_ds_moe_expert_selection_single_device(
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
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict,
):
    if mode == "decode":
        assert seq_len == 1, "Decode only supports seq_len=1"
        seq_len_per_device = seq_len
    else:
        assert mode == "prefill", "Unsupported mode"
        assert seq_len % mesh_device.shape[0] == 0, "Prefill seq_len must be divisible by mesh rows."
        seq_len_per_device = seq_len // mesh_device.shape[0]

    if mesh_device.get_num_devices() == 1:
        single_device_mesh = mesh_device
    else:
        single_device_mesh = mesh_device.create_submeshes(ttnn.MeshShape(1, 1))[0]

    if not program_cache_enabled:
        single_device_mesh.disable_and_clear_program_cache()

    run_config, tt_scores_seed, tt_scores_with_bias_seed, ref_output, batch_size = _build_moe_expert_selection_inputs(
        single_device_mesh,
        hf_config_short,
        cache_path,
        force_recalculate_weight_config,
        state_dict,
        mode,
        seq_len_per_device,
        use_real_weights,
    )

    _run_ds_moe_expert_selection_test(
        single_device_mesh,
        run_config,
        tt_scores_seed,
        tt_scores_with_bias_seed,
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
        f"ds_moe_expert_selection_single_device_{mode}_seq{seq_len}",
    )

    ttnn.deallocate(tt_scores_seed)
    ttnn.deallocate(tt_scores_with_bias_seed)


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
        ("prefill", 128),
        pytest.param("prefill", 131072, marks=_CI_FOCUSED_SKIP_MARK, id="prefill-131072"),
    ],
)
def test_ds_moe_expert_selection_device_perf(mode, seq_len):
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
    step_name = f"ds_moe_expert_selection_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_expert_selection.py"
    trace_filter = "eager"
    command = f'pytest {test_path}::test_ds_moe_expert_selection -k "{expr}"'

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


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
        ("prefill", 128),
        pytest.param("prefill", 131072, marks=_CI_FOCUSED_SKIP_MARK, id="prefill-131072"),
    ],
)
def test_ds_moe_expert_selection_single_device_device_perf(mode, seq_len):
    if mode == "decode":
        assert seq_len == 1, "Decode only supports seq_len=1"
    else:
        assert mode == "prefill", "Unsupported mode"

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    batch_size = USERS_PER_ROW if mode == "decode" else 1

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_moe_expert_selection_single_device_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_expert_selection.py"
    trace_filter = "eager"
    expr = (
        "single_device and program_cache and not no_program_cache and "
        f"{trace_filter} and {mode} and {seq_len} and real_weights"
    )
    command = f'pytest {test_path}::test_ds_moe_expert_selection_single_device -k "{expr}"'

    profiler.start("run")
    profiler.start(step_name)
    os.environ[DEVICE_PERF_ENV_VAR] = "1"
    op_stats, total_kernel_ns, total_op_to_op_ns = _collect_device_perf(
        command,
        subdir="deepseek_v3_fused_ops_single_device_perf",
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
        run_type="deepseek_v3_fused_ops_single_device_perf",
        ml_model_name="deepseek-v3",
        batch_size=batch_size,
        input_sequence_length=seq_len,
    )
