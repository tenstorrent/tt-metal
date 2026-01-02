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
from models.demos.deepseek_v3.reference.modeling_deepseek import apply_rotary_pos_emb
from models.demos.deepseek_v3.tests.fused_op_unit_tests.test_ds_fused_wqkva import ds_fused_wqkva_ttnn
from models.demos.deepseek_v3.tt.rms_norm.rms_norm import RMSNorm
from models.demos.deepseek_v3.tt.rope import get_cos_sin_matrix
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    dequantize_state_dict,
    get_model_config,
    get_rope_tensors,
    get_test_weight_config,
    system_name_to_mesh_shape,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

DEVICE_PERF_ENV_VAR = "DS_MLA_NORM_AND_ROPE_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US: dict[tuple[str, int], dict[str, float]] = {}


def _rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_dtype = x.dtype
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    weight = weight.view((1,) * (y.dim() - 1) + (-1,))
    return (y * weight).to(x_dtype)


def ds_mla_norm_and_rope_reference(
    q_input: torch.Tensor,
    kv_nope_input: torch.Tensor,
    kv_rope_input: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    position_ids: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_normed = _rms_norm_reference(q_input, q_norm_weight, rms_norm_eps)
    kv_nope_normed = _rms_norm_reference(kv_nope_input, kv_norm_weight, rms_norm_eps)

    kv_rope = kv_rope_input.permute(0, 2, 1, 3)
    kv_rope_for_rot = kv_rope.permute(1, 0, 2, 3)
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(1)
    kv_rope_for_rot, _ = apply_rotary_pos_emb(
        kv_rope_for_rot, kv_rope_for_rot, cos, sin, position_ids, unsqueeze_dim=1, meta_style=True
    )
    kv_rope = kv_rope_for_rot.permute(1, 0, 2, 3)
    kv_rope = kv_rope.permute(0, 2, 1, 3)

    kvpe = torch.cat([kv_nope_normed, kv_rope], dim=-1)
    pad_size = ttnn.TILE_SIZE - kvpe.shape[1]
    assert pad_size >= 0, "Expected kvpe dim 1 to be <= tile size."
    if pad_size:
        padded = kvpe.new_zeros((kvpe.shape[0], kvpe.shape[1] + pad_size, kvpe.shape[2], kvpe.shape[3]))
        padded[:, : kvpe.shape[1], :, :] = kvpe
        kvpe = padded
    kvpe = kvpe.permute(0, 2, 1, 3)
    return q_normed, kvpe


def ds_mla_norm_and_rope_ttnn(
    tt_q: ttnn.Tensor,
    tt_kv_nope: ttnn.Tensor,
    tt_kv_rope: ttnn.Tensor,
    cfg: dict,
    rope_tensors: dict,
    deallocate_inputs: bool = True,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    tt_q = RMSNorm.forward_decode(tt_q, cfg["q_norm"])

    tt_kv_nope = RMSNorm.forward_decode(tt_kv_nope, cfg["kv_norm"])
    tt_kv_nope = ttnn.to_memory_config(tt_kv_nope, memory_config=ttnn.L1_MEMORY_CONFIG)

    tt_kv_rope = ttnn.permute(tt_kv_rope, **cfg["kv_rope_permute"])
    tt_kv_rope = ttnn.to_memory_config(tt_kv_rope, **cfg["kv_rope_reshard"])
    tt_kv_rope = ttnn.experimental.rotary_embedding_llama(
        tt_kv_rope,
        rope_tensors["cos_matrix"],
        rope_tensors["sin_matrix"],
        rope_tensors["trans_matrix"],
        is_decode_mode=True,
    )
    tt_kv_rope = ttnn.to_memory_config(tt_kv_rope, **cfg["kv_rope_out_reshard"])
    tt_kv_rope = ttnn.permute(tt_kv_rope, **cfg["kv_rope_permute"])

    tt_kvpe = ttnn.concat([tt_kv_nope, tt_kv_rope], **cfg["kv_concat"])
    tt_kvpe = ttnn.pad(tt_kvpe, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)
    tt_kvpe = ttnn.permute(tt_kvpe, (0, 2, 1, 3))
    tt_kvpe = ttnn.mesh_partition(
        tt_kvpe,
        dim=1,
        cluster_axis=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_kvpe = ttnn.to_memory_config(tt_kvpe, **cfg["kvpe_reshard"])

    if deallocate_inputs:
        ttnn.deallocate(tt_kv_nope)
        ttnn.deallocate(tt_kv_rope)
    return tt_q, tt_kvpe


def _partition_reference(tensor: torch.Tensor, mesh_shape: ttnn.MeshShape, dim: int, cluster_axis: int):
    cluster_size = mesh_shape[cluster_axis]
    assert tensor.shape[dim] % cluster_size == 0
    dim_per_device = tensor.shape[dim] // cluster_size
    torch_reference_slices = []
    for x in range(mesh_shape[0]):
        for y in range(mesh_shape[1]):
            _, j = (x, y) if cluster_axis == 1 else (y, x)
            torch_reference_slice = tensor.split(dim_per_device, dim=dim)[j]
            torch_reference_slices.append(torch_reference_slice)
    return torch_reference_slices


def _compare_with_reference(
    tt_output: torch.Tensor, ref_output: torch.Tensor, expected_pcc: float, atol: float, rtol: float
):
    passing, pcc = comp_pcc(ref_output, tt_output, expected_pcc)
    logger.info(f"PCC: {pcc}")
    assert passing, f"PCC {pcc} is below required {expected_pcc}"
    torch.testing.assert_close(tt_output, ref_output, rtol=rtol, atol=atol)


def _measure_perf_us(
    mesh_device: ttnn.MeshDevice, op_fn, warmup_iters: int, measure_iters: int, trace_mode: bool = False
) -> float:
    ttnn.synchronize_device(mesh_device)
    if trace_mode:
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        for output in outputs:
            ttnn.deallocate(output)

        if warmup_iters > 0:
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for _ in range(warmup_iters):
                outputs = op_fn()
                for output in outputs:
                    ttnn.deallocate(output)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(measure_iters):
            outputs = op_fn()
            for output in outputs:
                ttnn.deallocate(output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        profiler.clear()
        profiler.start("ds_mla_norm_and_rope_perf")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        profiler.end("ds_mla_norm_and_rope_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get("ds_mla_norm_and_rope_perf") * 1e6

    for _ in range(warmup_iters):
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        for output in outputs:
            ttnn.deallocate(output)

    profiler.clear()
    profiler.start("ds_mla_norm_and_rope_perf")
    for _ in range(measure_iters):
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        for output in outputs:
            ttnn.deallocate(output)
    profiler.end("ds_mla_norm_and_rope_perf", PERF_CNT=measure_iters)
    return profiler.get("ds_mla_norm_and_rope_perf") * 1e6


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


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        ("decode", 1, 0.9997, 0.2, 0.2, 63090.436),
    ],
)
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 4194304,
        }
    ],
    indirect=True,
)
def test_ds_mla_norm_and_rope(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    program_cache_enabled,
    trace_mode,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict,
):
    assert mode == "decode", "ds_mla_norm_and_rope only supports decode."
    assert seq_len == 1, "Decode only supports seq_len=1"

    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace capture requires program cache to be enabled.")

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    module_path = "model.layers.0.self_attn"
    module_state_dict = sub_state_dict(state_dict, module_path + ".")
    dequant_state_dict = dequantize_state_dict(module_state_dict, hf_config_short)

    q_a_weight = dequant_state_dict["q_a_proj.weight"]
    kv_a_weight = dequant_state_dict["kv_a_proj_with_mqa.weight"]
    q_norm_weight = dequant_state_dict["q_a_layernorm.weight"]
    kv_norm_weight = dequant_state_dict["kv_a_layernorm.weight"]

    from models.demos.deepseek_v3.tt.mla.mla1d import MLA1D

    weight_config = get_test_weight_config(
        MLA1D,
        hf_config_short,
        (module_state_dict,) * mesh_device.shape[0],
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    model_config = get_model_config(MLA1D, mode, hf_config_short, mesh_device)
    model_state = {
        "mesh_device": mesh_device,
        "mesh_shape": mesh_device.shape,
        "ccl": ccl,
    }
    run_config = create_run_config(model_config, weight_config, model_state)

    batch_size = USERS_PER_ROW
    torch_input = torch.randn(batch_size, seq_len, hf_config_short.hidden_size, dtype=torch.bfloat16)
    torch_input = torch_input.permute(1, 0, 2)

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=run_config["input_memory_config"],
        layout=ttnn.TILE_LAYOUT,
    )

    q_lora_rank = hf_config_short.q_lora_rank
    kv_lora_rank = hf_config_short.kv_lora_rank
    qk_rope_head_dim = hf_config_short.qk_rope_head_dim

    q_kv = torch.nn.functional.linear(torch_input.unsqueeze(0), torch.cat([q_a_weight, kv_a_weight], dim=0))
    ref_q = q_kv[..., :q_lora_rank]
    ref_kv_nope = q_kv[..., q_lora_rank : q_lora_rank + kv_lora_rank]
    ref_kv_rope = q_kv[..., q_lora_rank + kv_lora_rank : q_lora_rank + kv_lora_rank + qk_rope_head_dim]

    tt_q, tt_kv_nope, tt_kv_rope = ds_fused_wqkva_ttnn(
        tt_input, run_config, ccl, q_lora_rank, kv_lora_rank, qk_rope_head_dim, mode
    )

    position_ids = torch.randint(0, hf_config_short.max_seq_len - 1, (batch_size,))
    rope_tensors = get_rope_tensors(hf_config_short, batch_size, seq_len, position_ids, mesh_device)
    cos, sin = get_cos_sin_matrix(hf_config_short)
    cos = cos.squeeze(0).squeeze(0).to(torch_input.dtype)
    sin = sin.squeeze(0).squeeze(0).to(torch_input.dtype)

    ref_q_normed, ref_kvpe = ds_mla_norm_and_rope_reference(
        ref_q,
        ref_kv_nope,
        ref_kv_rope,
        q_norm_weight,
        kv_norm_weight,
        hf_config_short.rms_norm_eps,
        position_ids,
        cos,
        sin,
    )

    tt_q, tt_kvpe = ds_mla_norm_and_rope_ttnn(tt_q, tt_kv_nope, tt_kv_rope, run_config, rope_tensors)

    tt_q_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_q)[0])
    _compare_with_reference(tt_q_torch, ref_q_normed, expected_pcc, expected_atol, expected_rtol)

    ref_kvpe_slices = _partition_reference(ref_kvpe, mesh_device.shape, dim=1, cluster_axis=1)
    for tt_kvpe_shard, ref_kvpe_slice in zip(ttnn.get_device_tensors(tt_kvpe), ref_kvpe_slices):
        tt_kvpe_torch = ttnn.to_torch(tt_kvpe_shard)
        _compare_with_reference(tt_kvpe_torch, ref_kvpe_slice, expected_pcc, expected_atol, expected_rtol)

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        tt_q_perf, tt_kv_nope_perf, tt_kv_rope_perf = ds_fused_wqkva_ttnn(
            tt_input, run_config, ccl, q_lora_rank, kv_lora_rank, qk_rope_head_dim, mode
        )
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"ds_mla_norm_and_rope_{mode}_seq{seq_len}_{trace_suffix}_{cache_suffix}"

        perf_profiler.start("run")
        perf_profiler.start(step_name)
        perf_us = _measure_perf_us(
            mesh_device,
            lambda: ds_mla_norm_and_rope_ttnn(
                tt_q_perf, tt_kv_nope_perf, tt_kv_rope_perf, run_config, rope_tensors, deallocate_inputs=False
            ),
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
        ttnn.deallocate(tt_kv_nope_perf)
        ttnn.deallocate(tt_kv_rope_perf)
    else:
        logger.info("Skipping e2e perf measurement during device-perf profiling.")
        tt_q_perf, tt_kv_nope_perf, tt_kv_rope_perf = ds_fused_wqkva_ttnn(
            tt_input, run_config, ccl, q_lora_rank, kv_lora_rank, qk_rope_head_dim, mode
        )
        from tracy import signpost

        def op_fn():
            return ds_mla_norm_and_rope_ttnn(
                tt_q_perf, tt_kv_nope_perf, tt_kv_rope_perf, run_config, rope_tensors, deallocate_inputs=False
            )

        for _ in range(PERF_WARMUP_ITERS):
            outputs = op_fn()
            ttnn.synchronize_device(mesh_device)
            for output in outputs:
                ttnn.deallocate(output)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            outputs = op_fn()
            for output in outputs:
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
                outputs = op_fn()
                ttnn.synchronize_device(mesh_device)
                for output in outputs:
                    ttnn.deallocate(output)
            signpost("stop")
        ttnn.deallocate(tt_kv_nope_perf)
        ttnn.deallocate(tt_kv_rope_perf)


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
    ],
)
def test_ds_mla_norm_and_rope_device_perf(mode, seq_len):
    assert mode == "decode", "ds_mla_norm_and_rope only supports decode."
    assert seq_len == 1, "Decode only supports seq_len=1"

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    batch_size = USERS_PER_ROW * mesh_shape[0]

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_mla_norm_and_rope_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/test_ds_mla_norm_and_rope.py"
    trace_filter = "trace"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_ds_mla_norm_and_rope -k "{expr}"'

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
