# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, profiler
from models.demos.deepseek_v3.tt.mla.mla1d import MLA1D
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict, get_model_config, get_test_weight_config
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

LONG_SEQ_ENV_VAR = "DEEPSEEK_V3_LONG_SEQ_TESTS"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100


def ds_fused_wqkva_reference(
    x: torch.Tensor,
    q_a_weight: torch.Tensor,
    kv_a_weight: torch.Tensor,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_a = torch.nn.functional.linear(x, q_a_weight)
    kv_a = torch.nn.functional.linear(x, kv_a_weight)
    q_kv = torch.cat([q_a, kv_a], dim=-1)

    q = q_kv[..., :q_lora_rank]
    kv_nope = q_kv[..., q_lora_rank : q_lora_rank + kv_lora_rank]
    kv_rope = q_kv[..., q_lora_rank + kv_lora_rank : q_lora_rank + kv_lora_rank + qk_rope_head_dim]
    return q, kv_nope, kv_rope


def ds_fused_wqkva_ttnn(
    x: ttnn.Tensor,
    cfg: dict,
    ccl,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    mode: str,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    seq_or_bsz = x.shape[2]

    tt_q_kv = ttnn.linear(x, **cfg["wq_kv_a"])

    if mode == "decode":
        ag_cfg = cfg["wq_kv_a_ag_decode"]
        r_cfg = cfg["wq_kv_a_r_decode"]
        q_slice_cfg = cfg["q_slice_decode"]
        kv_nope_slice_cfg = cfg["kv_nope_slice_decode"]
        kv_rope_slice_cfg = cfg["kv_rope_slice_decode"]
    else:
        ag_cfg = cfg["wq_kv_a_ag_prefill"]
        r_cfg = cfg["wq_kv_a_r_prefill"]
        q_slice_cfg = None
        kv_nope_slice_cfg = None
        kv_rope_slice_cfg = None

    tt_q_kv = ttnn.experimental.all_gather_async(tt_q_kv, **ccl.populate_all_gather_runtime_args(ag_cfg))
    tt_q_kv = ttnn.experimental.fast_reduce_nc(tt_q_kv, **r_cfg)

    if q_slice_cfg is None:
        tt_q = ttnn.slice(tt_q_kv, [0, 0, 0, 0], [1, 1, seq_or_bsz, q_lora_rank])
        tt_kv_nope = ttnn.slice(tt_q_kv, [0, 0, 0, q_lora_rank], [1, 1, seq_or_bsz, q_lora_rank + kv_lora_rank])
        tt_kv_rope = ttnn.slice(
            tt_q_kv,
            [0, 0, 0, q_lora_rank + kv_lora_rank],
            [1, 1, seq_or_bsz, q_lora_rank + kv_lora_rank + qk_rope_head_dim],
        )
    else:
        tt_q = ttnn.slice(tt_q_kv, [0, 0, 0, 0], [1, 1, seq_or_bsz, q_lora_rank], **q_slice_cfg)
        tt_kv_nope = ttnn.slice(
            tt_q_kv,
            [0, 0, 0, q_lora_rank],
            [1, 1, seq_or_bsz, q_lora_rank + kv_lora_rank],
            **kv_nope_slice_cfg,
        )
        tt_kv_rope = ttnn.slice(
            tt_q_kv,
            [0, 0, 0, q_lora_rank + kv_lora_rank],
            [1, 1, seq_or_bsz, q_lora_rank + kv_lora_rank + qk_rope_head_dim],
            **kv_rope_slice_cfg,
        )

    ttnn.deallocate(tt_q_kv)
    return tt_q, tt_kv_nope, tt_kv_rope


def _maybe_skip_long_seq(seq_len: int):
    if seq_len <= 8192:
        return
    if os.getenv(LONG_SEQ_ENV_VAR) is None:
        pytest.skip(f"Set {LONG_SEQ_ENV_VAR}=1 to enable seq_len={seq_len} coverage.")


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
        profiler.start("ds_fused_wqkva_perf")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        profiler.end("ds_fused_wqkva_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get("ds_fused_wqkva_perf") * 1e6

    for _ in range(warmup_iters):
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        for output in outputs:
            ttnn.deallocate(output)

    profiler.clear()
    profiler.start("ds_fused_wqkva_perf")
    for _ in range(measure_iters):
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        for output in outputs:
            ttnn.deallocate(output)
    profiler.end("ds_fused_wqkva_perf", PERF_CNT=measure_iters)
    return profiler.get("ds_fused_wqkva_perf") * 1e6


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        ("decode", 1, 0.99993, 0.2, 0.2, 2003.264),
        ("prefill", 128, 0.99993, 0.2, 0.2, 1295.686),
        ("prefill", 1024, 0.99993, 0.2, 0.2, 4040.506),
        ("prefill", 8192, 0.99993, 0.2, 0.2, 24818.087),
        ("prefill", 131072, 0.99993, 0.2, 0.2, 0.0),
    ],
)
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["no_trace", "trace"])
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
def test_ds_fused_wqkva(
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

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    module_path = "model.layers.0.self_attn"
    module_state_dict = sub_state_dict(state_dict, module_path + ".")
    dequant_state_dict = dequantize_state_dict(module_state_dict, hf_config_short)

    q_a_weight = dequant_state_dict["q_a_proj.weight"]
    kv_a_weight = dequant_state_dict["kv_a_proj_with_mqa.weight"]

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

    batch_size = USERS_PER_ROW if mode == "decode" else 1
    torch_input = torch.randn(batch_size, seq_len, hf_config_short.hidden_size, dtype=torch.bfloat16)
    if mode == "decode":
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

    ref_q, ref_kv_nope, ref_kv_rope = ds_fused_wqkva_reference(
        torch_input.unsqueeze(0), q_a_weight, kv_a_weight, q_lora_rank, kv_lora_rank, qk_rope_head_dim
    )

    tt_q, tt_kv_nope, tt_kv_rope = ds_fused_wqkva_ttnn(
        tt_input, run_config, ccl, q_lora_rank, kv_lora_rank, qk_rope_head_dim, mode
    )

    # Output is replicated across devices after all-gather + reduce; avoid concatenating across mesh.
    tt_q_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_q)[0])
    tt_kv_nope_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_kv_nope)[0])
    tt_kv_rope_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_kv_rope)[0])

    _compare_with_reference(tt_q_torch, ref_q, expected_pcc, expected_atol, expected_rtol)
    _compare_with_reference(tt_kv_nope_torch, ref_kv_nope, expected_pcc, expected_atol, expected_rtol)
    _compare_with_reference(tt_kv_rope_torch, ref_kv_rope, expected_pcc, expected_atol, expected_rtol)

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    trace_suffix = "trace" if trace_mode else "no_trace"
    cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
    step_name = f"ds_fused_wqkva_{mode}_seq{seq_len}_{trace_suffix}_{cache_suffix}"

    perf_profiler.start("run")
    perf_profiler.start(step_name)
    perf_us = _measure_perf_us(
        mesh_device,
        lambda: ds_fused_wqkva_ttnn(tt_input, run_config, ccl, q_lora_rank, kv_lora_rank, qk_rope_head_dim, mode),
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
