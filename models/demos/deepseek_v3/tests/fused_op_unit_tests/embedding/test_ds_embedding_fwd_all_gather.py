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
from models.demos.deepseek_v3.tt.embedding.embedding1d import Embedding1D
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    get_model_config,
    get_test_weight_config,
    system_name_to_mesh_shape,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

LONG_SEQ_ENV_VAR = "DEEPSEEK_V3_LONG_SEQ_TESTS"
DEVICE_PERF_ENV_VAR = "DS_EMBEDDING_FWD_ALL_GATHER_DEVICE_PERF"
PCC_ITERS = 100
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US: dict[tuple[str, int], dict[str, float]] = {}
CI_ACTIVE = os.getenv("CI") == "true"

_LONG_SEQ_SKIP_MARK = pytest.mark.skipif(
    CI_ACTIVE and os.getenv(LONG_SEQ_ENV_VAR) is None,
    reason=f"Set {LONG_SEQ_ENV_VAR}=1 to enable long sequence coverage on CI.",
)

_TRACE_REQUIRES_CACHE_MARK = pytest.mark.skip(reason="Trace capture requires program cache.")

# Single-device tests are not applicable for ops that include CCLs.


def ds_embedding_fwd_all_gather_reference(
    input_ids: torch.Tensor, reference_weight: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, int]:
    assert input_ids.ndim == 3, f"Expected ids shape [1, 1, seq], got {input_ids.shape}"
    original_seq_len = input_ids.shape[-1]
    pad_len = (ttnn.TILE_SIZE - (original_seq_len % ttnn.TILE_SIZE)) % ttnn.TILE_SIZE
    if pad_len:
        input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)

    embeddings = torch.nn.functional.embedding(input_ids.long(), reference_weight)
    if embeddings.ndim == 3:
        embeddings = embeddings.unsqueeze(0)
    return embeddings.to(dtype), original_seq_len


def ds_embedding_fwd_all_gather_ttnn(embeddings_tc: ttnn.Tensor, cfg: dict, ccl) -> ttnn.Tensor:
    # Only exercise the fused op under test (all_gather).
    embeddings_ag = Embedding1D._fwd_all_gather(embeddings_tc, cfg, ccl)
    return embeddings_ag


def _build_embeddings_tc(tt_input: ttnn.Tensor, cfg: dict) -> ttnn.Tensor:
    embeddings, _ = Embedding1D._fwd_embedding(tt_input, cfg)
    embeddings_tc = Embedding1D._fwd_typecast(embeddings, cfg)
    return embeddings_tc


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
        output, _ = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

        if warmup_iters > 0:
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for _ in range(warmup_iters):
                output, _ = op_fn()
                ttnn.deallocate(output)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(measure_iters):
            output, _ = op_fn()
            ttnn.deallocate(output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        profiler.clear()
        profiler.start("ds_embedding_fwd_all_gather_perf")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        profiler.end("ds_embedding_fwd_all_gather_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get("ds_embedding_fwd_all_gather_perf") * 1e6

    for _ in range(warmup_iters):
        output, _ = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

    profiler.clear()
    profiler.start("ds_embedding_fwd_all_gather_perf")
    for _ in range(measure_iters):
        output, _ = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)
    profiler.end("ds_embedding_fwd_all_gather_perf", PERF_CNT=measure_iters)
    return profiler.get("ds_embedding_fwd_all_gather_perf") * 1e6


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

    df = _merge_device_rows_for_perf(df)
    op_stats = {}
    for _, row in df.iterrows():
        op_code = row["OP CODE"]
        kernel_vals = row["DEVICE KERNEL DURATION [ns]"]
        op_to_op_vals = row["OP TO OP LATENCY [ns]"]
        if math.isnan(kernel_vals) or math.isnan(op_to_op_vals):
            continue
        op_stats[op_code] = {
            "avg_kernel_duration_ns": kernel_vals,
            "avg_op_to_op_latency_ns": op_to_op_vals,
        }

    total_kernel_ns = sum(entry["avg_kernel_duration_ns"] for entry in op_stats.values())
    total_op_to_op_ns = sum(entry["avg_op_to_op_latency_ns"] for entry in op_stats.values())
    return op_stats, total_kernel_ns, total_op_to_op_ns


def _build_embedding_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config_short,
    cache_path: str,
    ccl,
    force_recalculate_weight_config: bool,
    state_dict: dict,
    mode: str,
    seq_len: int,
    use_real_weights: bool,
):
    force_recalculate = force_recalculate_weight_config or not use_real_weights
    weight_cache_path = cache_path / ("real_weights" if use_real_weights else "random_weights")
    if use_real_weights:
        module_state_dict = sub_state_dict(state_dict, "model.embed_tokens.")
        reference_weight = module_state_dict["weight"].to(torch.bfloat16)
        weight_state_dict = {"weight": reference_weight}
    else:
        reference_weight = torch.randn(
            hf_config_short.vocab_size,
            hf_config_short.hidden_size,
            dtype=torch.bfloat16,
        )
        weight_state_dict = {"weight": reference_weight}

    weight_config = get_test_weight_config(
        Embedding1D,
        hf_config_short,
        (weight_state_dict,),
        weight_cache_path,
        mesh_device,
        force_recalculate,
    )
    model_config = get_model_config(Embedding1D, mode, hf_config_short, mesh_device)
    model_state = Embedding1D.create_state(hf_config_short, mesh_device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state)

    torch_input = torch.randint(
        0,
        hf_config_short.vocab_size,
        (1, 1, seq_len),
        dtype=torch.int64,
    )
    if mode == "decode":
        ref_dtype = torch.float32
    else:
        ref_dtype = torch.bfloat16
    ref_output, original_seq_len = ds_embedding_fwd_all_gather_reference(torch_input, reference_weight, ref_dtype)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    return run_config, tt_input, ref_output, original_seq_len


def _compare_all_gather_output(
    mesh_device: ttnn.MeshDevice,
    tt_output: ttnn.Tensor,
    ref_output: torch.Tensor,
    expected_pcc: float,
    expected_atol: float,
    expected_rtol: float,
):
    mesh_rows, mesh_cols = mesh_device.shape
    hidden_size = ref_output.shape[-1]
    hidden_shard = hidden_size // mesh_cols
    assert hidden_size % mesh_cols == 0, "Hidden size must be divisible by mesh columns"

    device_tensors = ttnn.get_device_tensors(tt_output)
    assert len(device_tensors) == mesh_rows * mesh_cols, "Mismatch in device tensor count"

    for row in range(mesh_rows):
        for col in range(mesh_cols):
            device_index = row * mesh_cols + col
            start = col * hidden_shard
            end = start + hidden_shard
            ref_slice = ref_output[..., start:end]
            tt_slice = ttnn.to_torch(device_tensors[device_index])
            _compare_with_reference(tt_slice, ref_slice, expected_pcc, expected_atol, expected_rtol)


def _run_ds_embedding_fwd_all_gather_test(
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
    for iter_idx in range(PCC_ITERS):
        embeddings_tc = _build_embeddings_tc(tt_input, run_config)
        tt_output = ds_embedding_fwd_all_gather_ttnn(embeddings_tc, run_config, ccl)
        if iter_idx == PCC_ITERS - 1:
            _compare_all_gather_output(
                mesh_device,
                tt_output,
                ref_output,
                expected_pcc,
                expected_atol,
                expected_rtol,
            )
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
            lambda: ds_embedding_fwd_all_gather_ttnn(_build_embeddings_tc(tt_input, run_config), run_config, ccl),
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
            return ds_embedding_fwd_all_gather_ttnn(_build_embeddings_tc(tt_input, run_config), run_config, ccl)

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


if CI_ACTIVE:
    _MODE_SEQ_PARAMS = [
        ("decode", 1, 1.0, 0.2, 0.2, 1073.942),
        ("prefill", 128, 1.0, 0.2, 0.2, 1279.149),
    ]
    _PROGRAM_TRACE_PARAMS = [(True, True)]
    _PROGRAM_TRACE_IDS = ["program_cache-trace"]
    _USE_REAL_WEIGHTS_PARAMS = [True]
    _USE_REAL_WEIGHTS_IDS = ["real_weights"]
else:
    _MODE_SEQ_PARAMS = [
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        ("decode", 1, 1.0, 0.2, 0.2, 1073.942),
        ("prefill", 128, 1.0, 0.2, 0.2, 1279.149),
        ("prefill", 1024, 1.0, 0.2, 0.2, 1751.616),
        ("prefill", 8192, 1.0, 0.2, 0.2, 2239.408),
        pytest.param("prefill", 32768, 1.0, 0.2, 0.2, 5246.868, marks=_LONG_SEQ_SKIP_MARK, id="prefill-32768"),
        pytest.param("prefill", 131072, 1.0, 0.2, 0.2, 17448.759, marks=_LONG_SEQ_SKIP_MARK, id="prefill-131072"),
    ]
    _PROGRAM_TRACE_PARAMS = [
        (True, False),
        (False, False),
        (True, True),
        pytest.param(False, True, marks=_TRACE_REQUIRES_CACHE_MARK),
    ]
    _PROGRAM_TRACE_IDS = [
        "program_cache-eager",
        "no_program_cache-eager",
        "program_cache-trace",
        "no_program_cache-trace",
    ]
    _USE_REAL_WEIGHTS_PARAMS = [True, False]
    _USE_REAL_WEIGHTS_IDS = ["real_weights", "random_weights"]


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    _MODE_SEQ_PARAMS,
)
@pytest.mark.parametrize(
    "program_cache_enabled, trace_mode",
    _PROGRAM_TRACE_PARAMS,
    ids=_PROGRAM_TRACE_IDS,
)
@pytest.mark.parametrize(
    "use_real_weights",
    _USE_REAL_WEIGHTS_PARAMS,
    ids=_USE_REAL_WEIGHTS_IDS,
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
def test_ds_embedding_fwd_all_gather(
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

    run_config, tt_input, ref_output, original_seq_len = _build_embedding_inputs(
        mesh_device,
        hf_config_short,
        cache_path,
        ccl,
        force_recalculate_weight_config,
        state_dict,
        mode,
        seq_len,
        use_real_weights,
    )
    assert original_seq_len == seq_len, "Reference seq_len mismatch"

    _run_ds_embedding_fwd_all_gather_test(
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
        batch_size=1,
        ccl=ccl,
        step_prefix=f"ds_embedding_fwd_all_gather_{mode}_seq{seq_len}",
    )

    ttnn.deallocate(tt_input)


if CI_ACTIVE:
    _DEVICE_PERF_PARAMS = [("decode", 1), ("prefill", 128)]
else:
    _DEVICE_PERF_PARAMS = [
        ("decode", 1),
        ("prefill", 128),
        ("prefill", 1024),
        ("prefill", 8192),
        pytest.param("prefill", 32768, marks=_LONG_SEQ_SKIP_MARK, id="prefill-32768"),
        pytest.param("prefill", 131072, marks=_LONG_SEQ_SKIP_MARK, id="prefill-131072"),
    ]


@pytest.mark.parametrize(
    "mode, seq_len",
    _DEVICE_PERF_PARAMS,
)
def test_ds_embedding_fwd_all_gather_device_perf(mode, seq_len):
    if mode == "decode":
        assert seq_len == 1, "Decode only supports seq_len=1"
    else:
        assert mode == "prefill", "Unsupported mode"

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_embedding_fwd_all_gather_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding_fwd_all_gather.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len} and real_weights"
    command = f'pytest {test_path}::test_ds_embedding_fwd_all_gather -k "{expr}"'

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
        batch_size=1,
        input_sequence_length=seq_len,
    )
