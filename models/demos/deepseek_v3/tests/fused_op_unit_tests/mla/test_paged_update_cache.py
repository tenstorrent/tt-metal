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
from models.demos.deepseek_v3.tt.mla.mla1d import MLA1D
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, get_mesh_coords
from models.demos.deepseek_v3.utils.test_utils import (
    get_model_config,
    paged_cache_from_torch,
    system_name_to_mesh_shape,
    torch_cache_from_paged,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

LONG_SEQ_ENV_VAR = "DEEPSEEK_V3_LONG_SEQ_TESTS"
DEVICE_PERF_ENV_VAR = "DS_PAGED_UPDATE_CACHE_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US: dict[tuple[str, int], dict[str, float]] = {}


def paged_update_cache_reference(cache: torch.Tensor, update: torch.Tensor, update_idxs: torch.Tensor) -> torch.Tensor:
    updated_cache = cache.clone()
    for user_idx, update_idx in enumerate(update_idxs.tolist()):
        if update_idx < 0:
            continue
        updated_cache[user_idx, 0, update_idx : update_idx + 1, :] = update[0, 0, user_idx : user_idx + 1, :]
    return updated_cache


def paged_update_cache_ttnn(
    kvpe_cache: ttnn.Tensor,
    tt_kvpe: ttnn.Tensor,
    update_idxs_tensor: ttnn.Tensor,
    page_table: ttnn.Tensor,
    mesh_shape: ttnn.MeshShape,
    row_idx: int,
) -> ttnn.Tensor:
    ttnn.experimental.paged_update_cache(
        kvpe_cache,
        tt_kvpe,
        update_idxs_tensor=update_idxs_tensor,
        page_table=page_table,
        mesh_coords=set(get_mesh_coords(mesh_shape, row_idx)),
    )
    return kvpe_cache


def _maybe_skip_long_seq(seq_len: int):
    if seq_len <= 8192:
        return
    if os.getenv(LONG_SEQ_ENV_VAR) is None:
        pytest.skip(f"Set {LONG_SEQ_ENV_VAR}=1 to enable seq_len={seq_len} coverage.")


def _get_cache_on_host(tt_cache: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    return ttnn.to_torch(
        tt_cache,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )


def _compare_cache_with_reference(
    tt_cache: torch.Tensor,
    ref_cache: torch.Tensor,
    update_idxs: torch.Tensor,
    row_idx: int,
    expected_pcc: float,
    atol: float,
    rtol: float,
):
    row_start = row_idx * USERS_PER_ROW
    tt_updated = torch.stack([tt_cache[row_start + i, 0, int(idx.item())] for i, idx in enumerate(update_idxs)], dim=0)
    ref_updated = torch.stack(
        [ref_cache[row_start + i, 0, int(idx.item())] for i, idx in enumerate(update_idxs)], dim=0
    )
    passing, pcc = comp_pcc(ref_updated, tt_updated, expected_pcc)
    logger.info(f"PCC (updated slices): {pcc}")
    assert passing, f"PCC {pcc} is below required {expected_pcc}"
    ref_cache = ref_cache.to(tt_cache.dtype)
    torch.testing.assert_close(tt_cache, ref_cache, rtol=rtol, atol=atol)


def _measure_perf_us(
    mesh_device: ttnn.MeshDevice, op_fn, warmup_iters: int, measure_iters: int, trace_mode: bool = False
) -> float:
    ttnn.synchronize_device(mesh_device)
    if trace_mode:
        op_fn()
        ttnn.synchronize_device(mesh_device)

        if warmup_iters > 0:
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for _ in range(warmup_iters):
                op_fn()
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(measure_iters):
            op_fn()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        profiler.clear()
        profiler.start("paged_update_cache_perf")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        profiler.end("paged_update_cache_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get("paged_update_cache_perf") * 1e6

    for _ in range(warmup_iters):
        op_fn()
        ttnn.synchronize_device(mesh_device)

    profiler.clear()
    profiler.start("paged_update_cache_perf")
    for _ in range(measure_iters):
        op_fn()
        ttnn.synchronize_device(mesh_device)
    profiler.end("paged_update_cache_perf", PERF_CNT=measure_iters)
    return profiler.get("paged_update_cache_perf") * 1e6


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
        ("decode", 1, 0.99997, 0.3, 0.3, 510.378),
        ("prefill", 128, 0.999, 0.3, 0.3, 0.0),
        ("prefill", 1024, 0.999, 0.3, 0.3, 0.0),
        ("prefill", 8192, 0.999, 0.3, 0.3, 0.0),
        ("prefill", 131072, 0.999, 0.3, 0.3, 0.0),
    ],
)
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 1048576,
        }
    ],
    indirect=True,
)
def test_paged_update_cache(
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
    if mode == "decode":
        assert seq_len == 1, "Decode only supports seq_len=1"
    else:
        assert mode == "prefill", "Unsupported mode"
        _maybe_skip_long_seq(seq_len)
        # MLA1D only uses paged_update_cache in decode; prefill is covered by paged_fill_cache.
        pytest.skip("paged_update_cache is only used in decode mode for MLA1D.")

    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace capture requires program cache to be enabled.")

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    kvpe_dim = hf_config_short.kv_lora_rank + hf_config_short.qk_rope_head_dim

    torch_cache_row = torch.zeros((USERS_PER_ROW, 1, hf_config_short.max_seq_len, kvpe_dim), dtype=torch.bfloat16)
    paged_cache_row, torch_page_table = paged_cache_from_torch(
        torch_cache_row, (1, mesh_device.shape[1]), paged_config, user_id=None
    )

    model_config = get_model_config(MLA1D, mode, hf_config_short, mesh_device)
    model_state = MLA1D.create_state(
        hf_config_short,
        paged_config,
        mesh_device,
        ccl,
        (paged_cache_row,) * mesh_device.shape[0],
    )
    kvpe_cache = model_state["kvpe_cache"]

    row_idx = 0
    position_idxs = torch.randint(0, hf_config_short.max_seq_len - 1, (USERS_PER_ROW,), dtype=torch.int32)
    kvpe_update = torch.randn((1, 1, USERS_PER_ROW, kvpe_dim), dtype=torch.bfloat16)

    update_idxs_tensor = ttnn.from_torch(
        position_idxs,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=mesh_device.shape),
        dtype=ttnn.int32,
    )
    page_table = MLA1D.create_page_table(
        page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device
    )

    tt_kvpe = ttnn.from_torch(
        kvpe_update,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_kvpe = ttnn.pad(tt_kvpe, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)
    tt_kvpe = ttnn.permute(tt_kvpe, (0, 2, 1, 3))
    tt_kvpe = ttnn.mesh_partition(
        tt_kvpe,
        dim=1,
        cluster_axis=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_kvpe = ttnn.to_memory_config(tt_kvpe, **model_config["kvpe_reshard"])

    ref_cache_row = paged_update_cache_reference(torch_cache_row, kvpe_update, position_idxs)
    ref_cache_full = (
        torch_cache_row.unsqueeze(0)
        .repeat(mesh_device.shape[0], 1, 1, 1, 1)
        .reshape(mesh_device.shape[0] * USERS_PER_ROW, 1, hf_config_short.max_seq_len, kvpe_dim)
    )
    row_start = row_idx * USERS_PER_ROW
    ref_cache_full[row_start : row_start + USERS_PER_ROW] = ref_cache_row

    paged_update_cache_ttnn(
        kvpe_cache,
        tt_kvpe,
        update_idxs_tensor,
        page_table,
        mesh_device.shape,
        row_idx,
    )

    tt_cache = torch_cache_from_paged(
        _get_cache_on_host(kvpe_cache, mesh_device),
        torch_page_table,
        mesh_device.get_num_devices(),
    )
    _compare_cache_with_reference(
        tt_cache, ref_cache_full, position_idxs, row_idx, expected_pcc, expected_atol, expected_rtol
    )

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"paged_update_cache_{mode}_seq{seq_len}_{trace_suffix}_{cache_suffix}"

        perf_profiler.start("run")
        perf_profiler.start(step_name)
        perf_us = _measure_perf_us(
            mesh_device,
            lambda: paged_update_cache_ttnn(
                kvpe_cache,
                tt_kvpe,
                update_idxs_tensor,
                page_table,
                mesh_device.shape,
                row_idx,
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
            batch_size=USERS_PER_ROW,
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
            return paged_update_cache_ttnn(
                kvpe_cache,
                tt_kvpe,
                update_idxs_tensor,
                page_table,
                mesh_device.shape,
                row_idx,
            )

        for _ in range(PERF_WARMUP_ITERS):
            op_fn()
            ttnn.synchronize_device(mesh_device)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            op_fn()
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
                op_fn()
                ttnn.synchronize_device(mesh_device)
            signpost("stop")


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
        ("prefill", 128),
        ("prefill", 1024),
        ("prefill", 8192),
        ("prefill", 131072),
    ],
)
def test_paged_update_cache_device_perf(mode, seq_len):
    if mode == "decode":
        assert seq_len == 1, "Decode only supports seq_len=1"
    else:
        assert mode == "prefill", "Unsupported mode"
        _maybe_skip_long_seq(seq_len)
        # MLA1D only uses paged_update_cache in decode; prefill is covered by paged_fill_cache.
        pytest.skip("paged_update_cache is only used in decode mode for MLA1D.")

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    batch_size = USERS_PER_ROW * mesh_shape[0]

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"paged_update_cache_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/test_paged_update_cache.py"
    if mode == "decode":
        logger.warning("paged_update_cache device perf uses eager mode because trace capture is unsupported.")
        trace_filter = "eager"
    else:
        trace_filter = "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_paged_update_cache -k "{expr}"'

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
        target=targets["kernel"] if targets is not None else None,
    )
    benchmark_data.add_measurement(
        profiler,
        0,
        step_name,
        "total_op_to_op_latency_us",
        total_op_to_op_us,
        target=targets["op_to_op"] if targets is not None else None,
    )
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="deepseek_v3_fused_ops",
        ml_model_name="deepseek-v3",
        batch_size=batch_size,
        input_sequence_length=seq_len,
    )
