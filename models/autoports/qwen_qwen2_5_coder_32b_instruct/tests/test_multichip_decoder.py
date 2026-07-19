# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import math
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen2_5_coder_32b_instruct.tests.test_optimized_decoder import (
    _assert_pcc,
    _config,
    _empty_caches,
    _hf_layer,
    _real_state,
    _recorded_activation,
    _reference_layer,
    _synthetic_state,
    _to_host,
    _tt_tensor,
)
from models.autoports.qwen_qwen2_5_coder_32b_instruct.tt.functional_decoder import EMITTED_BATCH, REPRESENTATIVE_LAYER
from models.autoports.qwen_qwen2_5_coder_32b_instruct.tt.multichip_decoder import (
    PAGE_BLOCK_SIZE,
    TARGET_MESH_SHAPE,
    TP_DEGREE,
    MultichipDecoder,
    _dram_sharded_memory_config,
    _same_memory_placement,
)
from models.autoports.qwen_qwen2_5_coder_32b_instruct.tt.optimized_decoder import OptimizationConfig, OptimizedDecoder
from models.common.utility_functions import comp_pcc

BASELINE_PATH_ENV = "QWEN2_5_CODER_32B_MULTICHIP_BASELINE_PATH"
PERF_ENV = "QWEN2_5_CODER_32B_MULTICHIP_RUN_PERF"
HF_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
HF_SNAPSHOT_REVISION = "381fc969f78efac66bc87ff7ddeadb7e73c218a7"
RESULTS_SUBDIR = "doc/optimized_multichip_decoder/results"


def _optional_env_int(name: str) -> int | None:
    return int(os.environ[name]) if name in os.environ else None


def _write_result(name: str, payload: dict) -> None:
    path = Path(__file__).resolve().parents[1] / RESULTS_SUBDIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _dram_snapshot(mesh_device) -> dict:
    return _memory_snapshot(mesh_device, ttnn.BufferType.DRAM)


def _memory_snapshot(mesh_device, buffer_type) -> dict:
    view = ttnn.get_memory_view(mesh_device, buffer_type)
    return {
        "num_banks": view.num_banks,
        "total_bytes": view.num_banks * view.total_bytes_per_bank,
        "allocated_bytes": view.num_banks * view.total_bytes_allocated_per_bank,
        "free_bytes": view.num_banks * view.total_bytes_free_per_bank,
        "largest_contiguous_bytes_free_per_bank": view.largest_contiguous_bytes_free_per_bank,
    }


def _tp_hidden(host: torch.Tensor, mesh_device):
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )


def _replicated_hidden(host: torch.Tensor, mesh_device):
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _compose_hidden(tensor, mesh_device) -> torch.Tensor:
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))


def _compose_cache(tensor, mesh_device) -> torch.Tensor:
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))


def _unpage_cache(physical: torch.Tensor, page_table: torch.Tensor) -> torch.Tensor:
    users = []
    for row in page_table:
        users.append(torch.cat([physical[int(block)] for block in row], dim=1))
    return torch.stack(users)


def _copy_replicated_page_table(host_table: torch.Tensor, device_table, mesh_device) -> None:
    """Refresh a trace-bound replicated page table without changing its address."""

    host_mesh = ttnn.from_torch(
        host_table.to(torch.int32).contiguous(),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.copy_host_to_device_tensor(host_mesh, device_table)


def test_multichip_contract_is_optimized_owned_and_host_free():
    assert issubclass(MultichipDecoder, OptimizedDecoder)
    assert TARGET_MESH_SHAPE == (1, 4)
    assert TP_DEGREE == 4
    for method in (
        MultichipDecoder._all_gather_hidden,
        MultichipDecoder._reduce_scatter_hidden,
        MultichipDecoder._prefill_linear,
        MultichipDecoder._prefill_row_parallel,
        MultichipDecoder._fill_prefill_cache,
        MultichipDecoder._prefill_mlp,
        MultichipDecoder.prefill_forward,
        MultichipDecoder._decode_mlp,
        MultichipDecoder.decode_forward,
        MultichipDecoder.forward,
    ):
        source = inspect.getsource(method)
        for token in ("torch", "from_torch", "to_torch", "super()"):
            assert token not in source, f"{method.__name__} contains runtime fallback token {token!r}"


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_capture_real_optimized_single_chip_baseline(mesh_device):
    """Capture an independent one-device optimized result for the TP4 gate."""

    output_path_text = os.getenv(BASELINE_PATH_ENV)
    if not output_path_text:
        pytest.skip(f"Set {BASELINE_PATH_ENV} to capture the independent optimized baseline")
    config = _config()
    state = _real_state(REPRESENTATIVE_LAYER)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        max_cache_len=128,
    )
    prefill_hidden = _recorded_activation(17, seed=250417).expand(1, EMITTED_BATCH, -1, -1).contiguous()
    decode_hidden = _recorded_activation(1, seed=250418).expand(1, EMITTED_BATCH, -1, -1).contiguous()
    key_cache, value_cache = _empty_caches(
        config,
        mesh_device,
        batch=EMITTED_BATCH,
        max_cache_len=128,
        dtype=decoder.optimization_config.kv_cache_dtype,
    )
    prefill = decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    prefill_key = _to_host(key_cache)[:, :, :17, :].clone()
    prefill_value = _to_host(value_cache)[:, :, :17, :].clone()
    decode = decoder.decode_forward(_tt_tensor(decode_hidden, mesh_device), key_cache, value_cache, current_pos=17)
    ttnn.synchronize_device(mesh_device)
    artifact = {
        "profile": decoder.optimization_config.name,
        "layer": REPRESENTATIVE_LAYER,
        "batch": EMITTED_BATCH,
        "prefill_hidden": prefill_hidden,
        "decode_hidden": decode_hidden,
        "prefill": _to_host(prefill).clone(),
        "prefill_key": prefill_key,
        "prefill_value": prefill_value,
        "decode": _to_host(decode).clone(),
        "decode_key": _to_host(key_cache)[:, :, 17:18, :].clone(),
        "decode_value": _to_host(value_cache)[:, :, 17:18, :].clone(),
    }
    output_path = Path(output_path_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    print(f"captured optimized single-chip baseline: {output_path}")


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_real_multichip_matches_optimized_single_chip_baseline(mesh_device):
    """Layer 32 represents the model's single homogeneous dense layer kind."""

    baseline_path_text = os.getenv(BASELINE_PATH_ENV)
    if not baseline_path_text:
        pytest.skip(f"Set {BASELINE_PATH_ENV} to the independent optimized baseline artifact")
    baseline = torch.load(baseline_path_text, map_location="cpu", weights_only=True)
    config = _config()
    state = _real_state(REPRESENTATIVE_LAYER)
    model = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
    )
    key_cache, value_cache = model.allocate_kv_cache()
    prefill = model.prefill_forward(_tp_hidden(baseline["prefill_hidden"], mesh_device), key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    pccs = {
        "prefill": _assert_pcc(
            baseline["prefill"], _compose_hidden(prefill, mesh_device), 0.99, "TP4 real prefill versus optimized"
        ),
        "prefill_key": _assert_pcc(
            baseline["prefill_key"],
            _compose_cache(key_cache, mesh_device)[:, :, :17, :],
            0.99,
            "TP4 real prefill key versus optimized",
        ),
        "prefill_value": _assert_pcc(
            baseline["prefill_value"],
            _compose_cache(value_cache, mesh_device)[:, :, :17, :],
            0.99,
            "TP4 real prefill value versus optimized",
        ),
    }
    decode = model.decode_forward(
        _tp_hidden(baseline["decode_hidden"], mesh_device),
        key_cache,
        value_cache,
        current_pos=17,
    )
    ttnn.synchronize_device(mesh_device)
    pccs.update(
        {
            "decode": _assert_pcc(
                baseline["decode"], _compose_hidden(decode, mesh_device), 0.99, "TP4 real decode versus optimized"
            ),
            "decode_key": _assert_pcc(
                baseline["decode_key"],
                _compose_cache(key_cache, mesh_device)[:, :, 17:18, :],
                0.99,
                "TP4 real decode key versus optimized",
            ),
            "decode_value": _assert_pcc(
                baseline["decode_value"],
                _compose_cache(value_cache, mesh_device)[:, :, 17:18, :],
                0.99,
                "TP4 real decode value versus optimized",
            ),
        }
    )
    _write_result(
        "real_baseline_pcc.json",
        {
            "hf_model": HF_MODEL,
            "hf_snapshot_revision": HF_SNAPSHOT_REVISION,
            "baseline_capture_sha256": hashlib.sha256(Path(baseline_path_text).read_bytes()).hexdigest(),
            "single_chip_baseline": baseline["profile"],
            "multichip_policy": model.precision_policy_name,
            "mesh": [1, 4],
            "layer": REPRESENTATIVE_LAYER,
            "layer_kind": "homogeneous dense Qwen2 decoder",
            "prefill_sequence_length": 17,
            "decode_position": 17,
            "pcc": pccs,
        },
    )
    if os.getenv("TT_METAL_WATCHER"):
        repo_root = Path(__file__).resolve().parents[4]
        watcher_path = repo_root / "generated/watcher/watcher.log"
        watcher_text = watcher_path.read_text(errors="replace")
        fault_patterns = ("error", "assert", "hang", "stuck", "timeout")
        matches = [pattern for pattern in fault_patterns if pattern in watcher_text.lower()]
        assert not matches, f"watcher fault signatures: {matches}"
        retained_path = Path(__file__).resolve().parents[1] / RESULTS_SUBDIR / "watcher_clean.log"
        retained_path.write_text(watcher_text)
        _write_result(
            "watcher_clean.json",
            {
                "enabled": os.environ["TT_METAL_WATCHER"],
                "test": "real layer-32 prefill/decode versus optimized TTNN baseline",
                "log_path": str(retained_path.relative_to(repo_root)),
                "disabled_features": ["ETH"] if os.getenv("TT_METAL_WATCHER_DISABLE_ETH") else [],
                "coverage_status": (
                    "partial_eth_gate_exception" if os.getenv("TT_METAL_WATCHER_DISABLE_ETH") else "full"
                ),
                "eth_disable_reason": (
                    "full Watcher instrumentation exceeds the active-Ethernet ring kernel-config buffer"
                    if os.getenv("TT_METAL_WATCHER_DISABLE_ETH")
                    else None
                ),
                "gate_exception": (
                    {
                        "accepted_scope": "worker and dispatch kernels",
                        "excluded_scope": "active Ethernet firmware instrumentation",
                        "runtime_limit_bytes": 25_600,
                        "active_ethernet_program_bytes": 27_920,
                        "reason": (
                            "the instrumented active-Ethernet Ring program cannot fit the runtime kernel-config "
                            "buffer; this is an instrumentation limit, not a model failure"
                        ),
                    }
                    if os.getenv("TT_METAL_WATCHER_DISABLE_ETH")
                    else None
                ),
                "fault_patterns": list(fault_patterns),
                "matches": matches,
                "log_sha256": hashlib.sha256(watcher_text.encode()).hexdigest(),
            },
        )


@pytest.mark.skipif(os.getenv(PERF_ENV) != "single", reason="manual serialized single-chip performance run")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 20_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_single_chip_optimized_perf_for_multichip(mesh_device):
    config = _config()
    state = _real_state(REPRESENTATIVE_LAYER)
    profile = OptimizationConfig.named("advisor_packed_bfp8_hifi2_1d")
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        max_cache_len=128,
        optimization_config=profile,
    )
    key_cache, value_cache = _empty_caches(
        config,
        mesh_device,
        batch=EMITTED_BATCH,
        max_cache_len=128,
        dtype=profile.kv_cache_dtype,
    )
    prefill_hidden = _recorded_activation(17, seed=250417).expand(1, EMITTED_BATCH, -1, -1).contiguous()
    stable_prefill = _tt_tensor(prefill_hidden, mesh_device)
    decoder.prefill_forward(stable_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    prefill_samples = []
    for _ in range(int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_PREFILL_TRIALS", "7"))):
        start = time.perf_counter()
        decoder.prefill_forward(stable_prefill, key_cache, value_cache)
        ttnn.synchronize_device(mesh_device)
        prefill_samples.append((time.perf_counter() - start) * 1000.0)

    decode_hidden = _recorded_activation(1, seed=250418).expand(1, EMITTED_BATCH, -1, -1).contiguous()
    stable_decode = _tt_tensor(decode_hidden, mesh_device)
    decoder.decode_forward(stable_decode, key_cache, value_cache, current_pos=17)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decoder.decode_forward(stable_decode, key_cache, value_cache, current_pos=17)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    replay_count = int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_DECODE_REPLAYS", "100"))
    trace_samples = []
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        for _ in range(int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_DECODE_TRIALS", "7"))):
            start = time.perf_counter()
            for _ in range(replay_count):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            trace_samples.append((time.perf_counter() - start) * 1000.0 / replay_count)
        first = _to_host(trace_output).clone()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        second = _to_host(trace_output).clone()
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    assert torch.equal(first, second)
    payload = {
        "decoder": "OptimizedDecoder",
        "profile": profile.name,
        "hardware": "one Blackhole p300c",
        "batch": EMITTED_BATCH,
        "sequence_length": 17,
        "prefill_samples_ms": prefill_samples,
        "prefill_median_ms": statistics.median(prefill_samples),
        "decode_trace_samples_ms": trace_samples,
        "decode_trace_median_ms": statistics.median(trace_samples),
        "decode_replays_per_trial": replay_count,
        "trace_bitwise_deterministic": True,
    }
    _write_result("single_chip_perf.json", payload)
    print("SINGLE_CHIP_PERF", json.dumps(payload, sort_keys=True))


@pytest.mark.skipif(os.getenv(PERF_ENV) != "multichip", reason="manual serialized TP4 performance run")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(1800)
def test_real_multichip_warmed_prefill_and_traced_decode_perf(mesh_device):
    baseline_path_text = os.getenv(BASELINE_PATH_ENV)
    if not baseline_path_text:
        pytest.skip(f"Set {BASELINE_PATH_ENV} to the independent optimized baseline artifact")
    baseline = torch.load(baseline_path_text, map_location="cpu", weights_only=True)
    config = _config()
    state = _real_state(REPRESENTATIVE_LAYER)
    ccl_dtype = {
        "bf16": ttnn.bfloat16,
        "bfp8": ttnn.bfloat8_b,
    }[os.getenv("QWEN2_5_CODER_32B_MULTICHIP_CCL_DTYPE", "bf16")]
    decode_dtype_by_name = {
        "bf16": ttnn.bfloat16,
        "bfp8": ttnn.bfloat8_b,
    }
    decode_matmul_output_dtype_name = os.getenv("QWEN2_5_CODER_32B_MULTICHIP_ACTIVATION_DTYPE", "bf16")
    decode_matmul_output_dtype = decode_dtype_by_name[decode_matmul_output_dtype_name]
    decode_attention_output_dtype = decode_dtype_by_name[
        os.getenv(
            "QWEN2_5_CODER_32B_MULTICHIP_ATTENTION_ACTIVATION_DTYPE",
            decode_matmul_output_dtype_name,
        )
    ]
    decode_mlp_output_dtype = decode_dtype_by_name[
        os.getenv(
            "QWEN2_5_CODER_32B_MULTICHIP_MLP_ACTIVATION_DTYPE",
            decode_matmul_output_dtype_name,
        )
    ]
    model = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
        precision_policy=os.getenv(
            "QWEN2_5_CODER_32B_MULTICHIP_PRECISION",
            "attention_bfp8_lofi_mlp_bfp4_lofi",
        ),
        decode_target_cores=int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_CORES", "16")),
        decode_down_target_cores=int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_DOWN_CORES", "16")),
        decode_qkv_target_cores=_optional_env_int("QWEN2_5_CODER_32B_MULTICHIP_QKV_CORES"),
        decode_o_target_cores=(
            _optional_env_int("QWEN2_5_CODER_32B_MULTICHIP_O_CORES")
            if "QWEN2_5_CODER_32B_MULTICHIP_O_CORES" in os.environ
            else 8
        ),
        decode_gate_target_cores=(
            _optional_env_int("QWEN2_5_CODER_32B_MULTICHIP_GATE_CORES")
            if "QWEN2_5_CODER_32B_MULTICHIP_GATE_CORES" in os.environ
            else 32
        ),
        decode_qkv_in0_block_w_limit=_optional_env_int("QWEN2_5_CODER_32B_MULTICHIP_QKV_BLOCK_W"),
        decode_o_in0_block_w_limit=_optional_env_int("QWEN2_5_CODER_32B_MULTICHIP_O_BLOCK_W"),
        decode_gate_in0_block_w_limit=_optional_env_int("QWEN2_5_CODER_32B_MULTICHIP_GATE_BLOCK_W"),
        decode_down_in0_block_w_limit=_optional_env_int("QWEN2_5_CODER_32B_MULTICHIP_DOWN_BLOCK_W"),
        decode_sdpa_grid_x=int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_SDPA_GRID_X", "8")),
        decode_sdpa_grid_y=int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_SDPA_GRID_Y", "4")),
        decode_sdpa_group_width=int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_SDPA_GROUP_WIDTH", "16")),
        prefill_grid_x=int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_PREFILL_GRID_X", "10")),
        prefill_grid_y=int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_PREFILL_GRID_Y", "10")),
        prefill_in0_block_w=int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_PREFILL_IN0_BLOCK_W", "10")),
        ccl_payload_dtype=ccl_dtype,
        decode_matmul_output_dtype=decode_matmul_output_dtype,
        decode_attention_output_dtype=decode_attention_output_dtype,
        decode_mlp_output_dtype=decode_mlp_output_dtype,
        use_persistent_decode_collectives=os.getenv("QWEN2_5_CODER_32B_MULTICHIP_PERSISTENT_CCL", "1") == "1",
        use_fused_decode_reduce_scatter=os.getenv("QWEN2_5_CODER_32B_MULTICHIP_FUSED_RS", "0") == "1",
        residual_contract=os.getenv("QWEN2_5_CODER_32B_MULTICHIP_RESIDUAL", "sharded"),
        keep_decode_residual_l1=os.getenv("QWEN2_5_CODER_32B_MULTICHIP_KEEP_RESIDUAL_L1", "1") == "1",
        use_packed_decode_gate_up=os.getenv("QWEN2_5_CODER_32B_MULTICHIP_PACKED_GATE_UP", "1") == "1",
        use_distributed_decode_norm=(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_DISTRIBUTED_NORM", "0") == "1"),
        use_fused_decode_all_gather_matmul=(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_FUSED_AG_MATMUL", "0") == "1"),
    )
    key_cache, value_cache = model.allocate_kv_cache()
    stable_prefill = _tp_hidden(baseline["prefill_hidden"], mesh_device)
    warm_prefill = model.prefill_forward(stable_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_prefill, True)
    prefill_samples = []
    for _ in range(int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_PREFILL_TRIALS", "7"))):
        start = time.perf_counter()
        output = model.prefill_forward(stable_prefill, key_cache, value_cache)
        ttnn.synchronize_device(mesh_device)
        prefill_samples.append((time.perf_counter() - start) * 1000.0)
        ttnn.deallocate(output, True)

    stable_decode = _tp_hidden(baseline["decode_hidden"], mesh_device)
    position_buffers = model.allocate_decode_position_buffers(17)
    warm_decode = model.decode_forward(
        stable_decode,
        key_cache,
        value_cache,
        current_pos=17,
        position_buffers=position_buffers,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_decode, True)
    model.prepare_decode_position_buffers(position_buffers, 18)
    model.prepare_decode_position_buffers(position_buffers, 17)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.decode_forward(
        stable_decode,
        key_cache,
        value_cache,
        current_pos=17,
        position_buffers=position_buffers,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    replay_count = int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_DECODE_REPLAYS", "100"))
    trace_samples = []
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        for _ in range(int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_DECODE_TRIALS", "7"))):
            start = time.perf_counter()
            for _ in range(replay_count):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            trace_samples.append((time.perf_counter() - start) * 1000.0 / replay_count)
        decode_host = _compose_hidden(trace_output, mesh_device).clone()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        decode_repeat = _compose_hidden(trace_output, mesh_device).clone()
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    assert torch.equal(decode_host, decode_repeat)

    final_prefill = model.prefill_forward(stable_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    pcc_threshold = float(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_PCC_THRESHOLD", "0.99"))
    pccs = {
        "prefill": _assert_pcc(
            baseline["prefill"],
            _compose_hidden(final_prefill, mesh_device),
            pcc_threshold,
            "timed TP4 prefill",
        ),
        "decode": _assert_pcc(baseline["decode"], decode_host, pcc_threshold, "timed TP4 traced decode"),
    }
    payload = {
        "batch": EMITTED_BATCH,
        "sequence_length": 17,
        "decode_position": 17,
        "pcc_threshold_for_this_candidate_run": pcc_threshold,
        "pcc": pccs,
        "prefill_samples_ms": prefill_samples,
        "prefill_median_ms": statistics.median(prefill_samples),
        "decode_trace_samples_ms": trace_samples,
        "decode_trace_median_ms": statistics.median(trace_samples),
        "decode_replays_per_trial": replay_count,
        "trace_bitwise_deterministic": True,
        "mesh_plan": model.mesh_plan_summary(),
    }
    result_name = os.getenv(
        "QWEN2_5_CODER_32B_MULTICHIP_RESULT_NAME",
        f"candidate_{model.precision_policy_name}_{model.decode_target_cores}c.json",
    )
    _write_result(result_name, payload)
    print(
        f"MULTICHIP_PERF prefill_ms={payload['prefill_median_ms']:.6f} "
        f"traced_decode_ms={payload['decode_trace_median_ms']:.6f} result={result_name}"
    )


@pytest.mark.skipif(
    os.getenv("QWEN2_5_CODER_32B_MULTICHIP_RUN_TOPOLOGY") != "1",
    reason="manual compiler-provenance topology benchmark",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(1800)
def test_multichip_compiler_provenance_topology(mesh_device):
    """Compare the selected fractured boundary to the compiler's replicated all-reduces."""

    config = _config()
    state = _real_state(REPRESENTATIVE_LAYER)
    selected = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
    )
    provenance = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
        use_persistent_decode_collectives=True,
        residual_contract="replicated_provenance",
    )
    host_hidden = _recorded_activation(1, seed=250419).expand(1, EMITTED_BATCH, -1, -1).contiguous()
    selected_hidden = _tp_hidden(host_hidden, mesh_device)
    provenance_hidden = _replicated_hidden(host_hidden, mesh_device)
    selected_key, selected_value = selected.allocate_kv_cache()
    provenance_key, provenance_value = provenance.allocate_kv_cache()
    selected_position = selected.allocate_decode_position_buffers(0)
    provenance_position = provenance.allocate_decode_position_buffers(0)

    warm = selected.decode_forward(
        selected_hidden,
        selected_key,
        selected_value,
        current_pos=0,
        position_buffers=selected_position,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm, True)
    selected_trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    selected_output = selected.decode_forward(
        selected_hidden,
        selected_key,
        selected_value,
        current_pos=0,
        position_buffers=selected_position,
    )
    ttnn.end_trace_capture(mesh_device, selected_trace, cq_id=0)
    replays = int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_DECODE_REPLAYS", "100"))
    trials = int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_DECODE_TRIALS", "7"))
    selected_samples = []
    try:
        ttnn.execute_trace(mesh_device, selected_trace, cq_id=0, blocking=True)
        selected_first_trace_host = _compose_hidden(selected_output, mesh_device)
        for _ in range(trials):
            start = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(mesh_device, selected_trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            selected_samples.append((time.perf_counter() - start) * 1000.0 / replays)
        ttnn.execute_trace(mesh_device, selected_trace, cq_id=0, blocking=True)
        selected_host = _compose_hidden(selected_output, mesh_device)
    finally:
        ttnn.release_trace(mesh_device, selected_trace)
    assert torch.equal(selected_first_trace_host, selected_host)

    selected_eager_samples = []
    selected_eager_output = None
    for _ in range(trials):
        start = time.perf_counter()
        selected_eager_output = selected.decode_forward(
            selected_hidden,
            selected_key,
            selected_value,
            current_pos=0,
            position_buffers=selected_position,
        )
        ttnn.synchronize_device(mesh_device)
        selected_eager_samples.append((time.perf_counter() - start) * 1000.0)
    selected_eager_host = _compose_hidden(selected_eager_output, mesh_device)

    warm = provenance.decode_forward(
        provenance_hidden,
        provenance_key,
        provenance_value,
        current_pos=0,
        position_buffers=provenance_position,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm, True)
    provenance_samples = []
    provenance_output = None
    for _ in range(trials):
        start = time.perf_counter()
        provenance_output = provenance.decode_forward(
            provenance_hidden,
            provenance_key,
            provenance_value,
            current_pos=0,
            position_buffers=provenance_position,
        )
        ttnn.synchronize_device(mesh_device)
        provenance_samples.append((time.perf_counter() - start) * 1000.0)
    provenance_host = ttnn.to_torch(ttnn.get_device_tensors(provenance_output)[0])

    provenance_trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    provenance_trace_output = provenance.decode_forward(
        provenance_hidden,
        provenance_key,
        provenance_value,
        current_pos=0,
        position_buffers=provenance_position,
    )
    ttnn.end_trace_capture(mesh_device, provenance_trace, cq_id=0)
    provenance_trace_samples = []
    try:
        ttnn.execute_trace(mesh_device, provenance_trace, cq_id=0, blocking=True)
        provenance_first_trace_host = ttnn.to_torch(ttnn.get_device_tensors(provenance_trace_output)[0])
        for _ in range(trials):
            start = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(mesh_device, provenance_trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            provenance_trace_samples.append((time.perf_counter() - start) * 1000.0 / replays)
        ttnn.execute_trace(mesh_device, provenance_trace, cq_id=0, blocking=True)
        provenance_trace_host = ttnn.to_torch(ttnn.get_device_tensors(provenance_trace_output)[0])
    finally:
        ttnn.release_trace(mesh_device, provenance_trace)
    assert torch.equal(provenance_first_trace_host, provenance_trace_host)
    selected_trace_eager_pcc = _assert_pcc(
        selected_eager_host,
        selected_host,
        0.999,
        "selected topology traced versus eager",
    )
    output_pcc = _assert_pcc(
        selected_eager_host,
        provenance_host,
        0.99,
        "selected sharded topology versus compiler-provenance all-reduce topology",
    )
    provenance_trace_eager_pcc = _assert_pcc(
        provenance_host,
        provenance_trace_host,
        0.999,
        "compiler-provenance topology traced versus eager",
    )
    key_pcc = _assert_pcc(
        _compose_cache(selected_key, mesh_device)[:, :, :1, :],
        _compose_cache(provenance_key, mesh_device)[:, :, :1, :],
        0.99,
        "topology-family key cache",
    )
    value_pcc = _assert_pcc(
        _compose_cache(selected_value, mesh_device)[:, :, :1, :],
        _compose_cache(provenance_value, mesh_device)[:, :, :1, :],
        0.99,
        "topology-family value cache",
    )
    selected_trace_median = statistics.median(selected_samples)
    selected_eager_median = statistics.median(selected_eager_samples)
    provenance_median = statistics.median(provenance_samples)
    provenance_trace_median = statistics.median(provenance_trace_samples)
    _write_result(
        "topology_family_benchmark.json",
        {
            "activation_kind": "recorded layer-32 activation distribution",
            "precision_policy": selected.precision_policy_name,
            "decode_position": 0,
            "selected": {
                "family": "sharded layer boundary; 2 all-gathers + 2 reduce-scatters",
                "eager_samples_ms": selected_eager_samples,
                "eager_median_ms": selected_eager_median,
                "trace_samples_ms": selected_samples,
                "trace_median_ms": selected_trace_median,
                "mesh_plan": selected.mesh_plan_summary(),
            },
            "compiler_provenance": {
                "family": "replicated layer boundary; 2 Ring all-reduces",
                "eager_samples_ms": provenance_samples,
                "eager_median_ms": provenance_median,
                "trace_samples_ms": provenance_trace_samples,
                "trace_median_ms": provenance_trace_median,
                "trace_bitwise_deterministic": True,
                "mesh_plan": provenance.mesh_plan_summary(),
            },
            "fair_eager_speedup_over_provenance": provenance_median / selected_eager_median,
            "fair_trace_speedup_over_provenance": provenance_trace_median / selected_trace_median,
            "output_pcc": output_pcc,
            "selected_trace_eager_pcc": selected_trace_eager_pcc,
            "provenance_trace_eager_pcc": provenance_trace_eager_pcc,
            "selected_trace_bitwise_deterministic": True,
            "key_cache_pcc": key_pcc,
            "value_cache_pcc": value_pcc,
        },
    )


@pytest.mark.skipif(
    os.getenv("QWEN2_5_CODER_32B_MULTICHIP_RUN_FUSED_AG") != "1",
    reason="manual fused all-gather+O-matmul compatibility and latency probe",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(900)
def test_multichip_fused_all_gather_o_projection(mesh_device):
    """Probe the alternate column-sharded O topology on the selected real layer."""

    config = _config()
    state = _real_state(REPRESENTATIVE_LAYER)
    model = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
    )
    prefill_hidden = _recorded_activation(17, seed=250417).expand(1, EMITTED_BATCH, -1, -1).contiguous()
    decode_hidden = _recorded_activation(1, seed=250418).expand(1, EMITTED_BATCH, -1, -1).contiguous()
    key_cache, value_cache = model.allocate_kv_cache()
    model.prefill_forward(_tp_hidden(prefill_hidden, mesh_device), key_cache, value_cache)

    captured = {}
    original_matmul = ttnn.matmul
    original_reduce_scatter = model._reduce_scatter_hidden
    reduce_scatter_calls = 0

    def capture_o_input(*args, **kwargs):
        if args[1] is model.output_weight:
            captured["o_input"] = args[0]
        return original_matmul(*args, **kwargs)

    def capture_attention_shard(tensor, *, memory_config, decode):
        nonlocal reduce_scatter_calls
        result = original_reduce_scatter(tensor, memory_config=memory_config, decode=decode)
        if decode and reduce_scatter_calls == 0:
            captured["attention_shard"] = _compose_hidden(result, mesh_device)
        reduce_scatter_calls += 1
        return result

    ttnn.matmul = capture_o_input
    model._reduce_scatter_hidden = capture_attention_shard
    try:
        model.decode_forward(
            _tp_hidden(decode_hidden, mesh_device),
            key_cache,
            value_cache,
            current_pos=17,
        )
    finally:
        ttnn.matmul = original_matmul
        model._reduce_scatter_hidden = original_reduce_scatter
    ttnn.synchronize_device(mesh_device)
    assert "o_input" in captured and "attention_shard" in captured

    o_weight = state[f"model.layers.{REPRESENTATIVE_LAYER}.self_attn.o_proj.weight"].to(torch.bfloat16)
    fused_weight = ttnn.from_torch(
        o_weight.T.contiguous().reshape(1, 1, model.attention_width, model.hidden_size),
        dtype=model.precision_policy["attention"],
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )
    fused_input_memory_config = ttnn.create_sharded_memory_config(
        # The fused primitive uses one memory config for both the local input
        # and the gathered output, so size its per-core shard for full K.
        shape=(model.batch, model.attention_width // 8),
        core_grid=ttnn.CoreGrid(x=8, y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    fused_output_memory_config = ttnn.create_sharded_memory_config(
        shape=(model.batch, model.local_hidden_size // 8),
        core_grid=ttnn.CoreGrid(x=8, y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    fused_input = ttnn.to_memory_config(captured["o_input"], fused_input_memory_config)
    fused_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 1),
        in0_block_w=model.attention_width // 32 // 8,
        out_subblock_h=1,
        out_subblock_w=5,
        per_core_M=1,
        per_core_N=model.local_hidden_size // 32 // 8,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    def fused_call(weight=fused_weight):
        return ttnn.experimental.all_gather_matmul_async(
            fused_input,
            weight,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=model.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            all_gather_core_grid_offset=(0, 4),
            barrier_semaphore=model.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=1,
            memory_config_ag=fused_input_memory_config,
            topology=model.topology,
            memory_config_mm=fused_output_memory_config,
            dtype=ttnn.bfloat16,
            program_config=fused_program_config,
            compute_kernel_config=model.attention_compute_config,
        )[1]

    warm = fused_call()
    ttnn.synchronize_device(mesh_device)
    passed, pcc = comp_pcc(
        captured["attention_shard"].float(),
        _compose_hidden(warm, mesh_device).float(),
        pcc=0.99,
    )
    permutation_results = [{"k_rank_order": [0, 1, 2, 3], "pcc": pcc}]
    if not passed:
        ttnn.deallocate(warm, True)
        input_rank_chunks = list(o_weight.T.contiguous().chunk(TP_DEGREE, dim=0))
        best_pcc = pcc
        best_permutation = (0, 1, 2, 3)
        for permutation in itertools.permutations(range(TP_DEGREE)):
            if permutation == (0, 1, 2, 3):
                continue
            permuted_weight = ttnn.from_torch(
                torch.cat([input_rank_chunks[rank] for rank in permutation], dim=0).reshape(
                    1, 1, model.attention_width, model.hidden_size
                ),
                dtype=model.precision_policy["attention"],
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            )
            candidate_output = fused_call(permuted_weight)
            ttnn.synchronize_device(mesh_device)
            candidate_passed, candidate_pcc = comp_pcc(
                captured["attention_shard"].float(),
                _compose_hidden(candidate_output, mesh_device).float(),
                pcc=0.99,
            )
            permutation_results.append({"k_rank_order": list(permutation), "pcc": candidate_pcc})
            if candidate_pcc > best_pcc:
                best_pcc = candidate_pcc
                best_permutation = permutation
            ttnn.deallocate(candidate_output, True)
            ttnn.deallocate(permuted_weight, True)
            if candidate_passed:
                pytest.fail(
                    "A correct fused K-rank permutation now exists; integrate and time it "
                    f"before accepting the selected topology: {permutation}"
                )
        _write_result(
            "fused_all_gather_o_probe.json",
            {
                "status": "rejected_rank_relative_gather_contract",
                "logical_transform": "AG local attention [32,1280] to [32,5120], then column-sharded O [5120,1280]",
                "pcc_vs_selected_row_parallel_o_reduce_scatter": pcc,
                "best_global_k_rank_permutation": list(best_permutation),
                "best_global_k_rank_permutation_pcc": best_pcc,
                "all_global_k_rank_permutations": permutation_results,
                "required_pcc": 0.99,
                "shape_probe_history": [
                    "local-width shard shape rejected: gathered width needs 32 shards on 8 cores",
                    "full-gather shard shape executed but did not preserve shared rank/head ordering",
                    "all 24 shared K-rank weight packings were executed; none can compensate a rank-relative gather order across every output device",
                ],
                "op_contract_blocker": "all_gather_matmul_async exposes a rank-relative gathered K order for this Ring shape, while one ShardTensorToMesh column-sharded weight has one shared K-row packing across ranks; the required per-output-rank K permutations are not expressible by that shared mesh tensor contract",
                "selection": "rejected before latency measurement because no globally packed TP weight is numerically equivalent",
            },
        )
        return
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = fused_call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    replays = int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_DECODE_REPLAYS", "100"))
    trials = int(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_DECODE_TRIALS", "7"))
    samples = []
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        for _ in range(trials):
            start = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter() - start) * 1000.0 / replays)
        traced_pcc = _assert_pcc(
            captured["attention_shard"],
            _compose_hidden(traced_output, mesh_device),
            0.99,
            "traced fused O all-gather+matmul",
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    _write_result(
        "fused_all_gather_o_probe.json",
        {
            "status": "pass",
            "logical_transform": "AG local attention [32,1280] to [32,5120], then column-sharded O [5120,1280]",
            "pcc_vs_selected_row_parallel_o_reduce_scatter": pcc,
            "trace_pcc": traced_pcc,
            "trace_samples_ms": samples,
            "trace_median_ms": statistics.median(samples),
            "selected_o_plus_reduce_scatter_device_time_ms": 0.051,
            "selected_device_time_source": "decode tt-perf-report: O matmul 27 us + first reduce-scatter 24 us",
            "selection_rule": "accept only if full-graph integration can improve the selected 0.792 ms trace",
        },
    )


@pytest.mark.skipif(os.getenv("QWEN2_5_CODER_32B_MULTICHIP_RUN_PROFILE") != "1", reason="manual Tracy profile")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(1800)
def test_profile_selected_multichip_decoder(mesh_device):
    from tracy import signpost

    config = _config()
    state = _real_state(REPRESENTATIVE_LAYER)
    model = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
    )
    key_cache, value_cache = model.allocate_kv_cache()
    prefill_hidden = _recorded_activation(17, seed=250417).expand(1, EMITTED_BATCH, -1, -1).contiguous()
    stable_prefill = _tp_hidden(prefill_hidden, mesh_device)
    warm = model.prefill_forward(stable_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm, True)
    signpost(header="PERF_PREFILL")
    profile_prefill = model.prefill_forward(stable_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    signpost(header="PERF_PREFILL_END")

    decode_hidden = _recorded_activation(1, seed=250418).expand(1, EMITTED_BATCH, -1, -1).contiguous()
    stable_decode = _tp_hidden(decode_hidden, mesh_device)
    position_buffers = model.allocate_decode_position_buffers(17)
    warm = model.decode_forward(
        stable_decode,
        key_cache,
        value_cache,
        current_pos=17,
        position_buffers=position_buffers,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm, True)
    model.prepare_decode_position_buffers(position_buffers, 18)
    model.prepare_decode_position_buffers(position_buffers, 17)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.decode_forward(
        stable_decode,
        key_cache,
        value_cache,
        current_pos=17,
        position_buffers=position_buffers,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        signpost(header="PERF_DECODE")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        signpost(header="PERF_DECODE_END")
        assert torch.isfinite(_compose_hidden(trace_output, mesh_device)).all()
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(profile_prefill, True)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_multichip_synthetic_non_aligned_prefill_decode_and_paged_cache(mesh_device, expect_error):
    config = _config()
    state = _synthetic_state(config)
    model = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
    )
    plan = model.mesh_plan_summary()
    assert plan["mesh_shape"] == [1, 4]
    assert plan["local_shapes"]["residual"] == 1280
    assert plan["local_shapes"]["q_heads"] == 10
    assert plan["local_shapes"]["kv_heads"] == 2
    assert plan["local_shapes"]["qkv"] == 1792
    assert plan["local_shapes"]["intermediate"] == 6912
    with expect_error(ValueError, "construct the decoder with max_cache_len"):
        model.allocate_kv_cache(max_cache_len=64)
    with expect_error(ValueError, "construct the decoder with max_cache_len"):
        model.allocate_page_table(max_cache_len=64)
    with expect_error(ValueError, "page_block_size must be a multiple of 32"):
        model.allocate_page_table(page_block_size=48)

    seq_len = 31
    generator = torch.Generator().manual_seed(250431)
    hidden = torch.randn(
        (1, EMITTED_BATCH, seq_len, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference_layer = _hf_layer(state, config, REPRESENTATIVE_LAYER)
    reference, reference_key, reference_value, reference_cache = _reference_layer(
        reference_layer, hidden, config, REPRESENTATIVE_LAYER
    )
    key_cache, value_cache = model.allocate_kv_cache()
    output = model.prefill_forward(_tp_hidden(hidden, mesh_device), key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    pccs = {
        "prefill": _assert_pcc(reference, _compose_hidden(output, mesh_device), 0.99, "TP4 non-aligned prefill"),
        "prefill_key": _assert_pcc(
            reference_key,
            _compose_cache(key_cache, mesh_device)[:, :, :seq_len, :],
            0.99,
            "TP4 local key cache",
        ),
        "prefill_value": _assert_pcc(
            reference_value,
            _compose_cache(value_cache, mesh_device)[:, :, :seq_len, :],
            0.99,
            "TP4 local value cache",
        ),
    }

    stacked_reference, _, _, stacked_reference_cache = _reference_layer(
        reference_layer, reference, config, REPRESENTATIVE_LAYER
    )
    stacked_model = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
        shared_rotary_tables=model.shared_rotary_tables,
        shared_decode_collective_buffers=model.shared_decode_collective_buffers,
    )
    assert stacked_model.rotary_cos is model.rotary_cos
    assert stacked_model.rotary_sin_row_major is model.rotary_sin_row_major
    assert stacked_model._decode_ag_persistent_buffers[0] is model._decode_ag_persistent_buffers[0]
    assert stacked_model._decode_rs_persistent_buffers[0][1] is model._decode_rs_persistent_buffers[0][1]
    stacked_key, stacked_value = stacked_model.allocate_kv_cache()
    stacked_output = stacked_model.prefill_forward(output, stacked_key, stacked_value)
    ttnn.synchronize_device(mesh_device)
    assert tuple(stacked_output.shape) == (1, EMITTED_BATCH, seq_len, 1280)
    pccs["stacked_handoff"] = _assert_pcc(
        stacked_reference,
        _compose_hidden(stacked_output, mesh_device),
        0.98,
        "direct TP decoder-to-decoder boundary",
    )

    decode_hidden = torch.randn(
        (1, EMITTED_BATCH, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    expected_decode, expected_key, expected_value, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        REPRESENTATIVE_LAYER,
        start_pos=seq_len,
        cache=reference_cache,
    )
    shared_stack_positions = model.allocate_decode_position_buffers(seq_len)
    decode_output = model.decode_forward(
        _tp_hidden(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=seq_len,
        position_buffers=shared_stack_positions,
    )
    ttnn.synchronize_device(mesh_device)
    assert decode_output.is_sharded()
    # The public [1, batch, 1, local_hidden] view carries the reshape-adjusted
    # shard height.  The next decoder's initial metadata-only reshape must
    # recover the canonical [1, 1, batch, local_hidden] L1 contract without a
    # layout conversion.
    decode_handoff = ttnn.reshape(decode_output, [1, 1, EMITTED_BATCH, model.local_hidden_size])
    assert _same_memory_placement(decode_handoff.memory_config(), model.local_residual_memory_config)
    pccs["decode"] = _assert_pcc(expected_decode, _compose_hidden(decode_output, mesh_device), 0.98, "TP4 decode")
    pccs["decode_key"] = _assert_pcc(
        expected_key,
        _compose_cache(key_cache, mesh_device)[:, :, seq_len : seq_len + 1, :],
        0.99,
        "TP4 decode key update",
    )
    pccs["decode_value"] = _assert_pcc(
        expected_value,
        _compose_cache(value_cache, mesh_device)[:, :, seq_len : seq_len + 1, :],
        0.99,
        "TP4 decode value update",
    )
    expected_stacked_decode, expected_stacked_key, expected_stacked_value, _ = _reference_layer(
        reference_layer,
        expected_decode,
        config,
        REPRESENTATIVE_LAYER,
        start_pos=seq_len,
        cache=stacked_reference_cache,
    )
    stacked_decode_output = stacked_model.decode_forward(
        decode_output,
        stacked_key,
        stacked_value,
        current_pos=seq_len,
        position_buffers=shared_stack_positions,
    )
    ttnn.synchronize_device(mesh_device)
    assert stacked_decode_output.is_sharded()
    stacked_decode_handoff = ttnn.reshape(stacked_decode_output, [1, 1, EMITTED_BATCH, stacked_model.local_hidden_size])
    assert _same_memory_placement(stacked_decode_handoff.memory_config(), stacked_model.local_residual_memory_config)
    pccs["stacked_decode_handoff"] = _assert_pcc(
        expected_stacked_decode,
        _compose_hidden(stacked_decode_output, mesh_device),
        0.98,
        "direct TP decoder-to-decoder decode boundary with shared CCL workspace",
    )
    pccs["stacked_decode_key"] = _assert_pcc(
        expected_stacked_key,
        _compose_cache(stacked_key, mesh_device)[:, :, seq_len : seq_len + 1, :],
        0.99,
        "stacked decoder key update",
    )
    pccs["stacked_decode_value"] = _assert_pcc(
        expected_stacked_value,
        _compose_cache(stacked_value, mesh_device)[:, :, seq_len : seq_len + 1, :],
        0.99,
        "stacked decoder value update",
    )

    permutation = torch.arange(2 * EMITTED_BATCH - 1, -1, -1, dtype=torch.int32)
    host_page_table = permutation.reshape(EMITTED_BATCH, 2)
    page_table = model.allocate_page_table(permutation=permutation)
    paged_key, paged_value = model.allocate_kv_cache(paged=True)
    paged_prefill = model.prefill_forward(
        _tp_hidden(hidden, mesh_device), paged_key, paged_value, page_table=page_table
    )
    paged_decode = model.decode_forward(
        _tp_hidden(decode_hidden, mesh_device),
        paged_key,
        paged_value,
        current_pos=seq_len,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    pccs["paged_vs_contiguous_prefill"] = _assert_pcc(
        _compose_hidden(output, mesh_device),
        _compose_hidden(paged_prefill, mesh_device),
        0.999,
        "paged prefill",
    )
    pccs["paged_vs_contiguous_decode"] = _assert_pcc(
        _compose_hidden(decode_output, mesh_device),
        _compose_hidden(paged_decode, mesh_device),
        0.999,
        "paged decode",
    )
    logical_key = _unpage_cache(_compose_cache(paged_key, mesh_device), host_page_table)
    logical_value = _unpage_cache(_compose_cache(paged_value, mesh_device), host_page_table)
    pccs["paged_logical_key"] = _assert_pcc(
        _compose_cache(key_cache, mesh_device)[:, :, : seq_len + 1, :],
        logical_key[:, :, : seq_len + 1, :],
        0.999,
        "paged logical key layout",
    )
    pccs["paged_logical_value"] = _assert_pcc(
        _compose_cache(value_cache, mesh_device)[:, :, : seq_len + 1, :],
        logical_value[:, :, : seq_len + 1, :],
        0.999,
        "paged logical value layout",
    )

    trace_hidden_host = torch.randn(
        (1, EMITTED_BATCH, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    expected_trace_32, expected_trace_key_32, expected_trace_value_32, reference_cache = _reference_layer(
        reference_layer,
        trace_hidden_host,
        config,
        REPRESENTATIVE_LAYER,
        start_pos=seq_len + 1,
        cache=reference_cache,
    )
    trace_hidden = _tp_hidden(trace_hidden_host, mesh_device)
    position_buffers = model.allocate_decode_position_buffers(seq_len + 1)
    warm = model.forward(
        trace_hidden,
        key_cache,
        value_cache,
        mode="decode",
        current_pos=seq_len + 1,
        position_buffers=position_buffers,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm, True)
    # Exercise in-place position refresh before capture so capture/replay is
    # not accidentally relying on construction-time values.
    model.prepare_decode_position_buffers(position_buffers, seq_len + 2)
    model.prepare_decode_position_buffers(position_buffers, seq_len + 1)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.forward(
        trace_hidden,
        key_cache,
        value_cache,
        mode="decode",
        current_pos=seq_len + 1,
        position_buffers=position_buffers,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        replay_outputs = []
        for _ in range(10):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            replay_outputs.append(_compose_hidden(trace_output, mesh_device))
        first_trace = replay_outputs[0]
        final_same_position_trace = replay_outputs[-1]
        pccs["trace_position_32"] = _assert_pcc(
            expected_trace_32, first_trace, 0.98, "TP4 traced decode at position 32"
        )
        pccs["trace_key_position_32"] = _assert_pcc(
            expected_trace_key_32,
            _compose_cache(key_cache, mesh_device)[:, :, seq_len + 1 : seq_len + 2, :],
            0.99,
            "TP4 traced key cache at position 32",
        )
        pccs["trace_value_position_32"] = _assert_pcc(
            expected_trace_value_32,
            _compose_cache(value_cache, mesh_device)[:, :, seq_len + 1 : seq_len + 2, :],
            0.99,
            "TP4 traced value cache at position 32",
        )

        model.prepare_decode_position_buffers(position_buffers, seq_len + 2)
        expected_trace_33, expected_trace_key_33, expected_trace_value_33, _ = _reference_layer(
            reference_layer,
            trace_hidden_host,
            config,
            REPRESENTATIVE_LAYER,
            start_pos=seq_len + 2,
            cache=reference_cache,
        )
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        advanced_trace = _compose_hidden(trace_output, mesh_device)
        pccs["trace_position_33"] = _assert_pcc(
            expected_trace_33, advanced_trace, 0.98, "TP4 traced decode at advanced position 33"
        )
        pccs["trace_key_position_33"] = _assert_pcc(
            expected_trace_key_33,
            _compose_cache(key_cache, mesh_device)[:, :, seq_len + 2 : seq_len + 3, :],
            0.99,
            "TP4 traced key cache at advanced position 33",
        )
        pccs["trace_value_position_33"] = _assert_pcc(
            expected_trace_value_33,
            _compose_cache(value_cache, mesh_device)[:, :, seq_len + 2 : seq_len + 3, :],
            0.99,
            "TP4 traced value cache at advanced position 33",
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    assert torch.isfinite(first_trace).all()
    pccs["trace_repeat"] = _assert_pcc(
        first_trace, final_same_position_trace, 0.99999, "TP4 10-replay trace determinism"
    )
    assert torch.equal(first_trace, final_same_position_trace)
    _write_result(
        "synthetic_correctness.json",
        {
            "mesh": [1, 4],
            "policy": model.precision_policy_name,
            "logical_sequence_length": seq_len,
            "public_sequence_alignment": "none",
            "page_block_size": PAGE_BLOCK_SIZE,
            "trace_replays_same_position": 10,
            "trace_bitwise_deterministic": True,
            "pcc": pccs,
        },
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_multichip_paged_trace_refresh_matches_eager(mesh_device):
    """A captured decode must observe in-place page-table and position updates."""

    config = _config()
    state = _synthetic_state(config)
    model = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
    )
    generator = torch.Generator().manual_seed(250464)
    prefill_hidden = torch.randn((1, EMITTED_BATCH, 64, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    decode_hidden = torch.randn((1, EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    stable_decode = _tp_hidden(decode_hidden, mesh_device)

    table_a = torch.arange(2 * EMITTED_BATCH, dtype=torch.int32).reshape(EMITTED_BATCH, 2)
    table_b = table_a.clone()
    table_b[:, 1] = table_a.flip(0)[:, 1]
    assert sorted(table_b.flatten().tolist()) == list(range(2 * EMITTED_BATCH))
    page_table = model.allocate_page_table(permutation=table_a.flatten())
    page_table_a_ref = model.allocate_page_table(permutation=table_a.flatten())
    page_table_b_ref = model.allocate_page_table(permutation=table_b.flatten())

    trace_key, trace_value = model.allocate_kv_cache(paged=True)
    eager_a_key, eager_a_value = model.allocate_kv_cache(paged=True)
    eager_b_key, eager_b_value = model.allocate_kv_cache(paged=True)
    for key_cache, value_cache, table in (
        (trace_key, trace_value, page_table),
        (eager_a_key, eager_a_value, page_table_a_ref),
        (eager_b_key, eager_b_value, page_table_b_ref),
    ):
        prefill_output = model.prefill_forward(
            _tp_hidden(prefill_hidden, mesh_device), key_cache, value_cache, page_table=table
        )
        ttnn.deallocate(prefill_output, True)

    eager_a_64 = model.decode_forward(
        stable_decode, eager_a_key, eager_a_value, current_pos=64, page_table=page_table_a_ref
    )
    ttnn.synchronize_device(mesh_device)
    eager_a_64_host = _compose_hidden(eager_a_64, mesh_device)
    eager_b_64 = model.decode_forward(
        stable_decode, eager_b_key, eager_b_value, current_pos=64, page_table=page_table_b_ref
    )
    ttnn.synchronize_device(mesh_device)
    eager_b_64_host = _compose_hidden(eager_b_64, mesh_device)
    eager_b_65 = model.decode_forward(
        stable_decode, eager_b_key, eager_b_value, current_pos=65, page_table=page_table_b_ref
    )
    ttnn.synchronize_device(mesh_device)
    eager_b_65_host = _compose_hidden(eager_b_65, mesh_device)

    position_buffers = model.allocate_decode_position_buffers(64)
    warm = model.decode_forward(
        stable_decode,
        trace_key,
        trace_value,
        current_pos=64,
        page_table=page_table,
        position_buffers=position_buffers,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm, True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.decode_forward(
        stable_decode,
        trace_key,
        trace_value,
        current_pos=64,
        page_table=page_table,
        position_buffers=position_buffers,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        pccs = {
            "original_table_output": _assert_pcc(
                eager_a_64_host, _compose_hidden(trace_output, mesh_device), 0.999, "paged trace table A"
            )
        }

        _copy_replicated_page_table(table_b, page_table, mesh_device)
        model.prepare_decode_position_buffers(position_buffers, 64)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        pccs["refreshed_table_output"] = _assert_pcc(
            eager_b_64_host, _compose_hidden(trace_output, mesh_device), 0.999, "paged trace table B"
        )
        logical_trace_key = _unpage_cache(_compose_cache(trace_key, mesh_device), table_b)
        logical_eager_key = _unpage_cache(_compose_cache(eager_b_key, mesh_device), table_b)
        logical_trace_value = _unpage_cache(_compose_cache(trace_value, mesh_device), table_b)
        logical_eager_value = _unpage_cache(_compose_cache(eager_b_value, mesh_device), table_b)
        pccs["refreshed_table_key"] = _assert_pcc(
            logical_eager_key[:, :, 64:65, :],
            logical_trace_key[:, :, 64:65, :],
            0.999,
            "paged traced key after table refresh",
        )
        pccs["refreshed_table_value"] = _assert_pcc(
            logical_eager_value[:, :, 64:65, :],
            logical_trace_value[:, :, 64:65, :],
            0.999,
            "paged traced value after table refresh",
        )

        model.prepare_decode_position_buffers(position_buffers, 65)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        pccs["advanced_position_output"] = _assert_pcc(
            eager_b_65_host,
            _compose_hidden(trace_output, mesh_device),
            0.999,
            "paged trace advanced position",
        )
        logical_trace_key = _unpage_cache(_compose_cache(trace_key, mesh_device), table_b)
        logical_eager_key = _unpage_cache(_compose_cache(eager_b_key, mesh_device), table_b)
        logical_trace_value = _unpage_cache(_compose_cache(trace_value, mesh_device), table_b)
        logical_eager_value = _unpage_cache(_compose_cache(eager_b_value, mesh_device), table_b)
        pccs["advanced_position_key"] = _assert_pcc(
            logical_eager_key[:, :, 65:66, :],
            logical_trace_key[:, :, 65:66, :],
            0.999,
            "paged traced key at advanced position",
        )
        pccs["advanced_position_value"] = _assert_pcc(
            logical_eager_value[:, :, 65:66, :],
            logical_trace_value[:, :, 65:66, :],
            0.999,
            "paged traced value at advanced position",
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    _write_result(
        "paged_trace_refresh.json",
        {
            "mesh": [1, 4],
            "policy": model.precision_policy_name,
            "prefill_sequence_length": 64,
            "page_block_size": PAGE_BLOCK_SIZE,
            "original_position": 64,
            "advanced_position": 65,
            "page_table_change": "reverse the unused second physical page across all 32 users",
            "pcc": pccs,
        },
    )


@pytest.mark.skipif(
    os.getenv("QWEN2_5_CODER_32B_MULTICHIP_RUN_CAPACITY") != "1",
    reason="manual full-stack DRAM capacity probe",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(600)
def test_multichip_full_stack_capacity(mesh_device):
    """Reserve model-resident state needed by a future full-model stack.

    This does not implement embedding/logits, which belong to the next stage;
    it does reserve their untied TP4 BFP8 tables so this decoder stage does not
    advertise a context that a full-model handoff cannot physically inhabit.
    """

    sequence_length = int(os.environ["QWEN2_5_CODER_32B_MULTICHIP_CAPACITY_SEQUENCE"])
    physical_cache_length = PAGE_BLOCK_SIZE * math.ceil(sequence_length / PAGE_BLOCK_SIZE)
    expected = os.getenv("QWEN2_5_CODER_32B_MULTICHIP_CAPACITY_EXPECT", "pass")
    if expected not in ("pass", "fail"):
        raise ValueError("QWEN2_5_CODER_32B_MULTICHIP_CAPACITY_EXPECT must be pass or fail")
    allocations = []
    snapshots = {"mesh_open": _dram_snapshot(mesh_device)}
    failed_stage = None
    error_text = None

    def reserve(
        shape,
        dtype,
        *,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        tensor = ttnn.empty(
            shape,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=memory_config,
        )
        allocations.append(tensor)
        return tensor

    try:
        failed_stage = "duplicated_prefill_decode_weights"
        # Decode owns five independent width-sharded projection tensors.
        decode_weight_roles = (
            ((5120, 2048), ttnn.bfloat8_b),
            ((1280, 5120), ttnn.bfloat8_b),
            ((5120, 7168), ttnn.bfloat4_b),
            ((5120, 7168), ttnn.bfloat4_b),
            ((7168, 5120), ttnn.bfloat4_b),
        )
        for (k, n), dtype in decode_weight_roles:
            for _ in range(64):
                reserve(
                    (1, 1, k, n),
                    dtype,
                    memory_config=_dram_sharded_memory_config(mesh_device, k, n),
                )
        # Prefill owns four interleaved tensors: QKV, O, packed gate/up, and
        # down. Do not split packed gate/up or reuse the decode sharding here;
        # buffer granularity and bank placement must match the implementation.
        prefill_weight_roles = (
            ((5120, 2048), ttnn.bfloat8_b),
            ((1280, 5120), ttnn.bfloat8_b),
            ((5120, 14_336), ttnn.bfloat4_b),
            ((7168, 5120), ttnn.bfloat4_b),
        )
        for (k, n), dtype in prefill_weight_roles:
            for _ in range(64):
                reserve((1, 1, k, n), dtype)

        failed_stage = "future_full_model_embedding_lm_head_and_final_norm"
        # Qwen2.5-Coder has untied [152064,5120] embedding and LM-head tables.
        # Reserve a vocab-row TP embedding and vocab-column TP LM head. The LM
        # head's local vocabulary is padded 38016->38144 for eight-bank tiling.
        local_vocab = 152_064 // TP_DEGREE
        padded_local_vocab = 38_144
        reserve(
            (1, 1, local_vocab, 5120),
            ttnn.bfloat8_b,
            memory_config=_dram_sharded_memory_config(mesh_device, local_vocab, 5120),
        )
        reserve(
            (1, 1, 5120, padded_local_vocab),
            ttnn.bfloat8_b,
            memory_config=_dram_sharded_memory_config(mesh_device, 5120, padded_local_vocab),
        )
        reserve((1, 1, 1, 5120), ttnn.bfloat16)

        failed_stage = "all_layer_kv_cache"
        # Runtime ownership is one [batch,local_heads,S,head_dim] K and V
        # buffer per layer. Keep all 128 allocations separate so DRAM page
        # alignment and fragmentation match an actual layer stack.
        for _ in range(64 * 2):
            reserve((32, 2, physical_cache_length, 128), ttnn.bfloat8_b)

        failed_stage = "rope_norm_bias_and_persistent_ccl"
        # RoPE is layer-independent. The stack construction contract shares one
        # cos+sin set in TILE and ROW_MAJOR layouts across all 64 layers.
        for _ in range(2):
            reserve((1, 1, sequence_length, 128), ttnn.bfloat16)
            reserve((1, 1, sequence_length, 128), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        # Two rank-1 norm vectors/layer. Each logical [5120] TILE tensor pads
        # independently to [32,5120], which aggregation would undercount.
        for _ in range(64 * 2):
            reserve((5120,), ttnn.bfloat16)
        # QKV bias has separate decode and prefill tensors per layer.
        for _ in range(64 * 2):
            reserve((1, 1, 1, 2048), ttnn.bfloat16)
        # Sequential layers share one persistent CCL workspace. It owns two
        # ping-pong AG full-hidden buffers and two RS intermediates in DRAM;
        # the two RS outputs use the runtime's 20-core L1 width-sharded layout.
        local_residual_l1 = ttnn.create_sharded_memory_config(
            shape=(32, 64),
            core_grid=ttnn.CoreGrid(x=10, y=2),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        for _ in range(4):
            reserve((1, 1, 32, 5120), ttnn.bfloat16)
        for _ in range(2):
            reserve((1, 1, 32, 1280), ttnn.bfloat16, memory_config=local_residual_l1)
        snapshots["persistent_ccl_l1"] = _memory_snapshot(mesh_device, ttnn.BufferType.L1)

        failed_stage = "decode_trace_control_state"
        # Position/RoPE outputs and the replicated page table are identical at
        # every sequential layer and are passed through the public API as one
        # fixed-address stack workspace.
        reserve((1, 1, 32, 128), ttnn.bfloat16)
        reserve((1, 1, 32, 128), ttnn.bfloat16)
        reserve((32,), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        reserve((32,), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        reserve((32, math.ceil(sequence_length / PAGE_BLOCK_SIZE)), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        snapshots["full_stack_static"] = _dram_snapshot(mesh_device)

        failed_stage = "prefill_live_set"
        # Ownership-aware peak: sharded residual, full norm input, retained
        # packed-QKV chunks, and QKV concatenation. MLP is 640-row chunked and
        # has a smaller sequence-proportional live set.
        reserve((1, 32, sequence_length, 1280), ttnn.bfloat16)
        reserve((1, 32, sequence_length, 5120), ttnn.bfloat16)
        reserve((1, 32, sequence_length, 2048), ttnn.bfloat16)
        reserve((1, 32, sequence_length, 2048), ttnn.bfloat16)
        snapshots["prefill_peak"] = _dram_snapshot(mesh_device)
        failed_stage = None
    except RuntimeError as error:
        error_text = str(error).split("backtrace:", maxsplit=1)[0].strip()
        snapshots["at_failure"] = _dram_snapshot(mesh_device)
    finally:
        payload = {
            "sequence_length": sequence_length,
            "physical_cache_length": physical_cache_length,
            "page_block_size": PAGE_BLOCK_SIZE,
            "batch": EMITTED_BATCH,
            "selected_precision": "attention BFP8/LoFi; MLP BFP4/LoFi; KV BFP8",
            "result": "pass" if failed_stage is None else "expected_out_of_memory",
            "failed_stage": failed_stage,
            "error": error_text,
            "snapshots": snapshots,
            "accounting": {
                "duplicated_projection_weight_bytes_per_device": 10_244_587_520,
                "future_tp4_embedding_lm_head_final_norm_bytes_per_device": 438_691_840,
                "embedding_local_vocab": 38_016,
                "lm_head_padded_local_vocab": 38_144,
                "duplicated_qkv_bias_bytes_per_device": 16_777_216,
                "bfp8_kv_bytes_per_device_per_physical_token": 1_114_112,
                "shared_bf16_rope_bytes_per_device_per_logical_token": 1_024,
                "shared_persistent_ccl_dram_bytes_per_device": 1_310_720,
                "shared_persistent_ccl_l1_bytes_per_device": 163_840,
                "norm_weight_bytes_per_device": 41_943_040,
                "shared_decode_position_payload_bytes_per_device": 16_640,
                "prefill_peak_live_bytes_per_device_per_logical_token": 671_744,
                "trace_region_size_parameter_bytes_per_bank": 200_000_000,
                "trace_reserved_bytes_per_device": 1_600_000_000,
            },
        }
        _write_result(f"capacity_seq{sequence_length}.json", payload)
        for tensor in reversed(allocations):
            ttnn.deallocate(tensor, True)

    if expected == "pass":
        assert failed_stage is None, f"capacity probe failed at {failed_stage}: {error_text}"
    else:
        assert failed_stage is not None, "capacity probe unexpectedly fit"
