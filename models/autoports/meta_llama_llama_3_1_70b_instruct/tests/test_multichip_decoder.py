# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import inspect
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.meta_llama_llama_3_1_70b_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _assert_pcc,
    _config,
    _real_state,
    _to_host,
)
from models.autoports.meta_llama_llama_3_1_70b_instruct.tt.multichip_decoder import (
    HEAD_AXIS,
    HIDDEN_AXIS,
    PAGED_BLOCK_SIZE,
    TARGET_MESH_SHAPE,
    TARGET_TP_DEGREE,
    MultiChipConfig,
    MultiChipDecoder,
    _logical_chunk_ranges,
    _pack_axis1_gate_up,
    _pack_axis1_qkv,
)
from models.autoports.meta_llama_llama_3_1_70b_instruct.tt.optimized_decoder import OptimizationConfig, OptimizedDecoder
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.utility_functions import comp_pcc

MAX_CACHE_LEN = 128
SINGLE_DEVICE_PARAMS = {"trace_region_size": 100_000_000}
MULTICHIP_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "trace_region_size": 100_000_000,
    "require_exact_physical_num_devices": True,
}
_REFERENCE = None

GEOMETRY_CANDIDATES = {
    "Q1": {"qkv_input_cores": 8, "qkv_in0_block_w": 32, "qkv_per_core_n": 5},
    "Q2": {"qkv_input_cores": 32, "qkv_in0_block_w": 8, "qkv_per_core_n": 5},
    "Q3": {"qkv_input_cores": 8, "qkv_in0_block_w": 32, "qkv_per_core_n": 2},
    "O1": {"output_input_cores": 16, "output_in0_block_w": 4, "output_per_core_n": 8},
    "O2": {"output_input_cores": 8, "output_in0_block_w": 8, "output_per_core_n": 8},
    "O4": {"output_input_cores": 8, "output_in0_block_w": 8, "output_per_core_n": 4},
    "G1": {"gate_up_input_cores": 32, "gate_up_in0_block_w": 8, "gate_up_per_core_n": 14},
    "G3": {"gate_up_input_cores": 64, "gate_up_in0_block_w": 4, "gate_up_per_core_n": 7},
    "G4": {"gate_up_input_cores": 16, "gate_up_in0_block_w": 16, "gate_up_per_core_n": 14},
    "G4A": {"gate_up_input_cores": 16, "gate_up_in0_block_w": 16, "gate_up_per_core_n": 7},
    "G4B": {"gate_up_input_cores": 16, "gate_up_in0_block_w": 8, "gate_up_per_core_n": 14},
    "G5": {"gate_up_input_cores": 8, "gate_up_in0_block_w": 32, "gate_up_per_core_n": 14},
    "G5A": {"gate_up_input_cores": 8, "gate_up_in0_block_w": 32, "gate_up_per_core_n": 7},
    "G5B": {"gate_up_input_cores": 8, "gate_up_in0_block_w": 16, "gate_up_per_core_n": 14},
    "G5C": {"gate_up_input_cores": 8, "gate_up_in0_block_w": 8, "gate_up_per_core_n": 14},
    "D1": {"down_input_cores": 16, "down_in0_block_w": 14, "down_per_core_n": 8},
    "D3": {"down_input_cores": 16, "down_in0_block_w": 14, "down_per_core_n": 4},
    "D4": {"down_input_cores": 8, "down_in0_block_w": 28, "down_per_core_n": 8},
    "D4A": {"down_input_cores": 8, "down_in0_block_w": 28, "down_per_core_n": 4},
    "D4B": {"down_input_cores": 8, "down_in0_block_w": 14, "down_per_core_n": 8},
    "D5": {"down_input_cores": 4, "down_in0_block_w": 56, "down_per_core_n": 8},
    "D5A": {"down_input_cores": 4, "down_in0_block_w": 56, "down_per_core_n": 4},
    "D5B": {"down_input_cores": 4, "down_in0_block_w": 28, "down_per_core_n": 8},
    "D5C": {"down_input_cores": 4, "down_in0_block_w": 14, "down_per_core_n": 8},
}

TOPOLOGY_CANDIDATES = {
    "persistent": {"collective_implementation": "decomposed_persistent"},
    "bf8": {"collective_dtype": ttnn.bfloat8_b},
    "links1": {"num_links": (2, 1)},
    "linear": {"topology": ttnn.Topology.Linear},
    "persistent_bf8": {
        "collective_implementation": "decomposed_persistent",
        "collective_dtype": ttnn.bfloat8_b,
    },
}


def _single_test(func):
    return pytest.mark.timeout(1800)(
        pytest.mark.parametrize(
            "mesh_device, device_params",
            [(1, SINGLE_DEVICE_PARAMS)],
            indirect=True,
            ids=["single-chip-optimized-reference"],
        )(func)
    )


def _mesh_test(func):
    return pytest.mark.timeout(1800)(
        pytest.mark.parametrize(
            "mesh_device, device_params",
            [(TARGET_MESH_SHAPE, MULTICHIP_DEVICE_PARAMS)],
            indirect=True,
            ids=["blackhole-1x4-ring"],
        )(func)
    )


def _device_tensor(tensor: torch.Tensor, mesh_device, *, mesh_mapper=None, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _mesh_input(tensor: torch.Tensor, mesh_device):
    return _device_tensor(tensor, mesh_device, mesh_mapper=MultiChipDecoder.mesh_mapper_for_input(mesh_device))


def _single_caches(config, mesh_device, *, batch: int):
    shape = (batch, config.num_key_value_heads, MAX_CACHE_LEN, 128)
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    return _device_tensor(zeros, mesh_device), _device_tensor(zeros, mesh_device)


def _mesh_contiguous_caches(config, mesh_device, *, batch: int):
    shape = (batch, config.num_key_value_heads, MAX_CACHE_LEN, 128)
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    mapper = MultiChipDecoder.mesh_mapper_for_cache(mesh_device)
    return _device_tensor(zeros, mesh_device, mesh_mapper=mapper), _device_tensor(
        zeros, mesh_device, mesh_mapper=mapper
    )


def _mesh_paged_caches(config, mesh_device, *, blocks: int):
    shape = (blocks, config.num_key_value_heads, PAGED_BLOCK_SIZE, 128)
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    mapper = MultiChipDecoder.mesh_mapper_for_cache(mesh_device)
    return _device_tensor(zeros, mesh_device, mesh_mapper=mapper), _device_tensor(
        zeros, mesh_device, mesh_mapper=mapper
    )


def _page_table(table: torch.Tensor, mesh_device):
    return ttnn.from_torch(
        table.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=MultiChipDecoder.mesh_mapper_for_page_table(mesh_device),
    )


def _position_tensor(position: int, mesh_device, *, rope: bool):
    host = torch.tensor([[position]] if rope else [position], dtype=torch.int32)
    return ttnn.from_torch(
        host,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32 if rope else ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _copy_mesh_host(tensor: torch.Tensor, destination, mesh_device, *, dtype, layout=ttnn.TILE_LAYOUT):
    host = ttnn.from_torch(
        tensor,
        device=None,
        layout=layout,
        dtype=dtype,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.copy_host_to_device_tensor(host, destination)


def _local_hosts(tensor):
    return [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]


def _compose_residual(tensor):
    local = _local_hosts(tensor)
    assert len(local) == TARGET_TP_DEGREE
    for rank in range(1, TARGET_TP_DEGREE):
        assert torch.equal(local[0], local[rank]), f"flat TP4 residual replica {rank} diverged"
    return local[0]


def _compose_cache(cache):
    local = _local_hosts(cache)
    assert len(local) == TARGET_TP_DEGREE
    return torch.cat(local, dim=1)


def _timed_ms(call, mesh_device, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        call()
    ttnn.synchronize_device(mesh_device)
    started = time.perf_counter()
    for _ in range(iterations):
        call()
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - started) * 1000.0 / iterations


def _measured_pcc(reference, actual, threshold: float, label: str) -> float:
    passed, value = comp_pcc(reference.float(), actual.float(), pcc=threshold)
    value = float(value)
    print(f"{label}: {value}")
    assert passed, f"{label}: PCC {value} is below {threshold}"
    return value


def test_two_axis_weight_packing_contract(expect_error):
    q = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    k = 100 + torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4)
    v = 200 + torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4)
    packed = _pack_axis1_qkv(q, k, v)
    expected0 = torch.cat((q.T[:, :4], k.T[:, :2], v.T[:, :2]), dim=-1)
    expected1 = torch.cat((q.T[:, 4:], k.T[:, 2:], v.T[:, 2:]), dim=-1)
    assert torch.equal(packed, torch.cat((expected0, expected1), dim=-1))

    gate = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    up = 100 + gate
    packed_gate_up = _pack_axis1_gate_up(gate, up)
    assert torch.equal(
        packed_gate_up,
        torch.cat((gate[:, :4], up[:, :4], gate[:, 4:], up[:, 4:]), dim=-1),
    )
    assert _logical_chunk_ranges(131072, 4096)[-1] == (126976, 131072)
    assert _logical_chunk_ranges(131071, 4096)[-1] == (126976, 131071)
    assert _logical_chunk_ranges(39, 32) == ((0, 32), (32, 39))
    for invalid in (0, 31, 33):
        with expect_error(ValueError, "prefill_mlp_chunk_size"):
            MultiChipConfig(prefill_mlp_chunk_size=invalid)


def test_runtime_path_is_real_multichip_and_host_fallback_free():
    assert issubclass(MultiChipDecoder, OptimizedDecoder)
    assert MultiChipDecoder.single_chip_baseline is OptimizedDecoder
    assert TARGET_MESH_SHAPE == (1, 4)
    assert TARGET_TP_DEGREE == 4
    assert (HIDDEN_AXIS, HEAD_AXIS) == (0, 1)
    assert MultiChipConfig().optimized.explicit_sdpa_program_config
    assert MultiChipConfig().optimized.explicit_sdpa_compute_kernel
    assert "self.prefill_sdpa_program_config = None" in inspect.getsource(MultiChipDecoder.__init__)
    for method in (
        MultiChipDecoder._all_reduce_flat,
        MultiChipDecoder._mlp_prefill_flat,
        MultiChipDecoder._mlp_decode_flat,
        MultiChipDecoder.prefill_forward,
        MultiChipDecoder._decode_forward_device_position,
        MultiChipDecoder.decode_forward_from_position_tensor,
        MultiChipDecoder.decode_forward,
    ):
        source = inspect.getsource(method)
        for token in ("from_torch", "to_torch", "torch.", "super().prefill", "super().decode"):
            assert token not in source, f"{method.__name__} contains forbidden runtime token {token!r}"
    assert "all_reduce_async" in inspect.getsource(MultiChipDecoder._all_reduce_flat)
    device_position_source = inspect.getsource(MultiChipDecoder._decode_forward_device_position)
    assert "paged_scaled_dot_product_attention_decode" in device_position_source
    assert "ttnn.embedding" in device_position_source
    assert "ttnn.plus_one" in device_position_source


def test_multichip_context_capacity_contract():
    path = Path(__file__).resolve().parents[1] / "doc" / "context_contract.json"
    contract = json.loads(path.read_text())
    evidence = contract["capacity_evidence"]
    assert contract["current_supported_context"] == contract["hf_advertised_context"] == 131072
    assert contract["limiting_reason"] is None
    assert evidence["target_mesh"] == "logical 1x4 ring over the four-chip physical cycle"
    assert evidence["per_device_kv_heads"] == 2
    assert evidence["page_pool_blocks_per_batch1_sequence"] * evidence["page_block_size"] == 131072
    assert evidence["per_device_weight_plus_kv_bytes"] < evidence["device_allocator_dram_bytes"]
    assert evidence["headroom_before_activations_trace_and_allocator_overhead_bytes"] > 12_000_000_000
    assert (
        evidence["unchunked_unbounded_peak_live_activation_bytes"]
        > evidence["headroom_before_activations_trace_and_allocator_overhead_bytes"]
    )
    reserved = (
        evidence["per_device_weight_plus_kv_bytes"]
        + evidence["peak_live_explicit_activation_bytes"]
        + evidence["shared_constants_and_metadata_reserve_bytes"]
        + evidence["trace_region_reserve_bytes"]
        + evidence["ccl_reduce_scatter_reserve_bytes"]
        + evidence["allocator_and_fragmentation_reserve_bytes"]
    )
    assert reserved == evidence["total_full_stack_peak_reserved_bytes"]
    assert evidence["device_allocator_dram_bytes"] - reserved == evidence["remaining_peak_margin_bytes"]
    assert reserved < evidence["device_allocator_dram_bytes"]


@_single_test
def test_00_single_chip_optimized_reference(mesh_device, record_property):
    config = _config()
    state = _real_state()
    batch = 1
    seq_len = 39
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=MAX_CACHE_LEN,
        optimization_config=OptimizationConfig(),
    )
    generator = torch.Generator().manual_seed(20260717)
    prefill_host = torch.randn((1, batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    decode_host = torch.randn((1, batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    prefill_input = _device_tensor(prefill_host, mesh_device)
    decode_input = _device_tensor(decode_host, mesh_device)
    key_cache, value_cache = _single_caches(config, mesh_device, batch=batch)
    prefill_output = decoder.prefill_forward(prefill_input, key_cache, value_cache)
    decode_output = decoder.decode_forward(decode_input, key_cache, value_cache, current_pos=seq_len)
    ttnn.synchronize_device(mesh_device)

    def prefill_call():
        return decoder.prefill_forward(prefill_input, key_cache, value_cache)

    def decode_call():
        return decoder.decode_forward(decode_input, key_cache, value_cache, current_pos=seq_len)

    prefill_ms = _timed_ms(prefill_call, mesh_device, warmup=2, iterations=5)
    decode_ms = _timed_ms(decode_call, mesh_device, warmup=5, iterations=30)
    # Like-for-like trace latency uses the same fixed-position layer graph on
    # one and four chips.  Advancing-position correctness is validated below
    # on the TP4 trace-safe entry point, because the completed single-chip
    # baseline intentionally remains unchanged by this stage.
    ttnn.synchronize_device(mesh_device)
    single_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    single_traced_output = decoder.decode_forward(
        decode_input,
        key_cache,
        value_cache,
        current_pos=seq_len,
    )
    ttnn.end_trace_capture(mesh_device, single_trace_id, cq_id=0)
    try:
        started = time.perf_counter()
        for _ in range(30):
            ttnn.execute_trace(mesh_device, single_trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        single_trace_ms = (time.perf_counter() - started) * 1000.0 / 30
        assert torch.isfinite(_to_host(single_traced_output)).all()
    finally:
        ttnn.release_trace(mesh_device, single_trace_id)
    signpost("SINGLE_PREFILL")
    prefill_call()
    ttnn.synchronize_device(mesh_device)
    signpost("SINGLE_PREFILL_END")
    signpost("SINGLE_DECODE")
    for _ in range(5):
        decode_call()
    ttnn.synchronize_device(mesh_device)
    signpost("SINGLE_DECODE_END")
    record_property("single_prefill_ms", f"{prefill_ms:.6f}")
    record_property("single_decode_ms", f"{decode_ms:.6f}")
    record_property("single_trace_replay_ms", f"{single_trace_ms:.6f}")

    dynamic_hosts = [
        torch.randn(
            (1, batch, 1, config.hidden_size),
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.bfloat16,
        )
        for seed in (7107, 7108)
    ]
    dynamic_key, dynamic_value = _single_caches(config, mesh_device, batch=batch)
    decoder.prefill_forward(prefill_input, dynamic_key, dynamic_value)
    dynamic_outputs = []
    dynamic_keys = []
    dynamic_values = []
    for offset, dynamic_host in enumerate(dynamic_hosts):
        dynamic_output = decoder.decode_forward(
            _device_tensor(dynamic_host, mesh_device),
            dynamic_key,
            dynamic_value,
            current_pos=seq_len + offset,
        )
        ttnn.synchronize_device(mesh_device)
        dynamic_outputs.append(_to_host(dynamic_output))
        dynamic_keys.append(_to_host(dynamic_key)[:, :, seq_len + offset : seq_len + offset + 1, :])
        dynamic_values.append(_to_host(dynamic_value)[:, :, seq_len + offset : seq_len + offset + 1, :])

    boundary_host = torch.randn(
        (1, batch, 1, config.hidden_size),
        generator=torch.Generator().manual_seed(6400),
        dtype=torch.bfloat16,
    )
    boundary_input = _device_tensor(boundary_host, mesh_device)
    boundary_key, boundary_value = _single_caches(config, mesh_device, batch=batch)
    boundary_output = decoder.decode_forward(
        boundary_input,
        boundary_key,
        boundary_value,
        current_pos=PAGED_BLOCK_SIZE,
    )
    ttnn.synchronize_device(mesh_device)

    # Repeat the real layer twice with independent layer-local caches.  This
    # is a direct stacked-decoder layout contract: layer zero's output must be
    # consumable by layer one without a host reshape, gather, or conversion.
    stack_host = torch.randn(
        (1, batch, 1, config.hidden_size),
        generator=torch.Generator().manual_seed(6500),
        dtype=torch.bfloat16,
    )
    stack_key_0, stack_value_0 = _single_caches(config, mesh_device, batch=batch)
    stack_key_1, stack_value_1 = _single_caches(config, mesh_device, batch=batch)
    stack_layer_0 = decoder.decode_forward(
        _device_tensor(stack_host, mesh_device),
        stack_key_0,
        stack_value_0,
        current_pos=5,
    )
    stack_layer_1 = decoder.decode_forward(
        stack_layer_0,
        stack_key_1,
        stack_value_1,
        current_pos=5,
    )
    ttnn.synchronize_device(mesh_device)

    global _REFERENCE
    _REFERENCE = {
        "prefill_input": prefill_host,
        "decode_input": decode_host,
        "prefill_output": _to_host(prefill_output),
        "decode_output": _to_host(decode_output),
        "key": _to_host(key_cache)[:, :, : seq_len + 1, :],
        "value": _to_host(value_cache)[:, :, : seq_len + 1, :],
        "single_prefill_ms": prefill_ms,
        "single_decode_ms": decode_ms,
        "single_trace_replay_ms": single_trace_ms,
        "dynamic_hosts": dynamic_hosts,
        "dynamic_outputs": dynamic_outputs,
        "dynamic_keys": dynamic_keys,
        "dynamic_values": dynamic_values,
        "boundary_input": boundary_host,
        "boundary_output": _to_host(boundary_output),
        "boundary_key": _to_host(boundary_key)[:, :, PAGED_BLOCK_SIZE : PAGED_BLOCK_SIZE + 1, :],
        "boundary_value": _to_host(boundary_value)[:, :, PAGED_BLOCK_SIZE : PAGED_BLOCK_SIZE + 1, :],
        "stack_input": stack_host,
        "stack_layer_0": _to_host(stack_layer_0),
        "stack_layer_1": _to_host(stack_layer_1),
    }
    del decoder, state
    gc.collect()


@_mesh_test
def test_01_multichip_correctness_paged_trace_and_perf(mesh_device, record_property):
    assert _REFERENCE is not None, "run the ordered single-chip reference item first"
    reference = _REFERENCE
    config = _config()
    state = _real_state()
    batch = 1
    seq_len = 39
    test_policy = MultiChipConfig()
    decoder = MultiChipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=MAX_CACHE_LEN,
        multichip_config=test_policy,
        tt_ccl=get_tt_ccl(mesh_device),
    )
    assert decoder.tt_ccl is get_tt_ccl(mesh_device)
    assert decoder.hidden_size == 8192
    assert decoder.num_heads == 16
    assert decoder.num_kv_heads == 2
    assert decoder.intermediate_size == 7168
    assert decoder.decode_sdpa_program_config is not None
    assert decoder.decode_sdpa_compute_kernel is not None
    assert decoder.prefill_sdpa_program_config is None
    assert decoder.multichip_config.prefill_mlp_chunk_size == 4096
    record_property("decode_sdpa_policy", "explicit_program_and_compute")
    record_property("prefill_sdpa_policy", "implicit_program")
    record_property("default_prefill_mlp_chunk_size", "4096")

    prefill_input = _mesh_input(reference["prefill_input"], mesh_device)
    decode_input = _mesh_input(reference["decode_input"], mesh_device)
    key_cache, value_cache = _mesh_contiguous_caches(config, mesh_device, batch=batch)
    prefill_output = decoder.prefill_forward(prefill_input, key_cache, value_cache)
    decode_output = decoder.decode_forward(decode_input, key_cache, value_cache, current_pos=seq_len)
    ttnn.synchronize_device(mesh_device)
    prefill_pcc = _measured_pcc(
        reference["prefill_output"], _compose_residual(prefill_output), 0.99, "flat TP4 prefill"
    )
    decode_pcc = _measured_pcc(reference["decode_output"], _compose_residual(decode_output), 0.99, "flat TP4 decode")
    key_pcc = _measured_pcc(reference["key"], _compose_cache(key_cache)[:, :, : seq_len + 1, :], 0.99, "flat TP4 key")
    value_pcc = _measured_pcc(
        reference["value"], _compose_cache(value_cache)[:, :, : seq_len + 1, :], 0.99, "flat TP4 value"
    )
    record_property("prefill_pcc", f"{prefill_pcc:.10f}")
    record_property("decode_pcc", f"{decode_pcc:.10f}")
    record_property("key_cache_pcc", f"{key_pcc:.10f}")
    record_property("value_cache_pcc", f"{value_pcc:.10f}")
    assert tuple(prefill_output.shape) == (1, batch, seq_len, 8192)
    assert tuple(decode_output.shape) == (1, batch, 1, 8192)
    assert tuple(key_cache.shape) == (batch, 2, MAX_CACHE_LEN, 128)

    # Keep direct coverage for a nonaligned internal chunk tail without using
    # the test-only chunk policy for any published default-path timing.
    tail_key, tail_value = _mesh_contiguous_caches(config, mesh_device, batch=batch)
    default_policy = decoder.multichip_config
    decoder.multichip_config = replace(default_policy, prefill_mlp_chunk_size=32)
    try:
        tail_output = decoder.prefill_forward(prefill_input, tail_key, tail_value)
        ttnn.synchronize_device(mesh_device)
    finally:
        decoder.multichip_config = default_policy
    tail_pcc = _measured_pcc(
        reference["prefill_output"],
        _compose_residual(tail_output),
        0.99,
        "flat TP4 prefill 32+7 internal tail",
    )
    record_property("nonaligned_internal_tail_pcc", f"{tail_pcc:.10f}")
    ttnn.deallocate(tail_output)
    ttnn.deallocate(tail_key)
    ttnn.deallocate(tail_value)

    stack_key_0, stack_value_0 = _mesh_contiguous_caches(config, mesh_device, batch=batch)
    stack_key_1, stack_value_1 = _mesh_contiguous_caches(config, mesh_device, batch=batch)
    stack_layer_0 = decoder.decode_forward(
        _mesh_input(reference["stack_input"], mesh_device),
        stack_key_0,
        stack_value_0,
        current_pos=5,
    )
    stack_layer_1 = decoder.decode_forward(
        stack_layer_0,
        stack_key_1,
        stack_value_1,
        current_pos=5,
    )
    ttnn.synchronize_device(mesh_device)
    _assert_pcc(reference["stack_layer_0"], _compose_residual(stack_layer_0), 0.99, "stack layer zero")
    stack_pcc = _measured_pcc(
        reference["stack_layer_1"],
        _compose_residual(stack_layer_1),
        0.99,
        "stack layer one",
    )
    record_property("stacked_two_layer_pcc", f"{stack_pcc:.10f}")
    assert tuple(stack_layer_1.shape) == tuple(decode_output.shape)

    # Non-identity page placement must reproduce the same logical cache and
    # output.  Logical page zero is deliberately stored in physical block one.
    table_host = torch.tensor([[1, 0]], dtype=torch.int32)
    table = _page_table(table_host, mesh_device)
    paged_key, paged_value = _mesh_paged_caches(config, mesh_device, blocks=2)
    paged_prefill = decoder.prefill_forward(prefill_input, paged_key, paged_value, page_table=table)
    paged_decode = decoder.decode_forward(
        decode_input,
        paged_key,
        paged_value,
        current_pos=seq_len,
        page_table=table,
    )
    ttnn.synchronize_device(mesh_device)
    _assert_pcc(reference["prefill_output"], _compose_residual(paged_prefill), 0.99, "paged flat TP4 prefill")
    _assert_pcc(reference["decode_output"], _compose_residual(paged_decode), 0.99, "paged flat TP4 decode")
    key_blocks = _compose_cache(paged_key)
    value_blocks = _compose_cache(paged_value)
    _assert_pcc(reference["key"], key_blocks[1:2, :, : seq_len + 1, :], 0.99, "paged physical key")
    _assert_pcc(reference["value"], value_blocks[1:2, :, : seq_len + 1, :], 0.99, "paged physical value")

    # Position 64 is the first token of logical page one, which this table
    # maps to physical block zero.  This catches off-by-one and aligned-public-
    # contract mistakes independently of the nonaligned length-39 path.
    boundary_input = _mesh_input(reference["boundary_input"], mesh_device)
    boundary_key, boundary_value = _mesh_paged_caches(config, mesh_device, blocks=2)
    boundary_output = decoder.decode_forward(
        boundary_input,
        boundary_key,
        boundary_value,
        current_pos=PAGED_BLOCK_SIZE,
        page_table=table,
    )
    ttnn.synchronize_device(mesh_device)
    _assert_pcc(reference["boundary_output"], _compose_residual(boundary_output), 0.99, "page-one decode")
    _assert_pcc(
        reference["boundary_key"],
        _compose_cache(boundary_key)[0:1, :, 0:1, :],
        0.99,
        "page-one physical key",
    )
    _assert_pcc(
        reference["boundary_value"],
        _compose_cache(boundary_value)[0:1, :, 0:1, :],
        0.99,
        "page-one physical value",
    )

    # Compile and capture the actual advancing graph.  Hidden input, current
    # position, RoPE index, and page-table contents all retain stable device
    # addresses and are refreshed without recapture.
    trace_key, trace_value = _mesh_paged_caches(config, mesh_device, blocks=2)
    decoder.prefill_forward(prefill_input, trace_key, trace_value, page_table=table)
    trace_input = _mesh_input(reference["dynamic_hosts"][0], mesh_device)
    current_pos_tensor = _position_tensor(seq_len, mesh_device, rope=False)
    rope_idx_tensor = _position_tensor(seq_len, mesh_device, rope=True)
    decoder.decode_forward_from_position_tensor(
        trace_input,
        trace_key,
        trace_value,
        current_pos_tensor=current_pos_tensor,
        rope_idx_tensor=rope_idx_tensor,
        page_table=table,
        advance_position=True,
    )
    _copy_mesh_host(
        torch.tensor([seq_len], dtype=torch.int32),
        current_pos_tensor,
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    _copy_mesh_host(
        torch.tensor([[seq_len]], dtype=torch.int32),
        rope_idx_tensor,
        mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward_from_position_tensor(
        trace_input,
        trace_key,
        trace_value,
        current_pos_tensor=current_pos_tensor,
        rope_idx_tensor=rope_idx_tensor,
        page_table=table,
        advance_position=True,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    replay_hosts = []
    try:
        for replay_idx, dynamic_host in enumerate(reference["dynamic_hosts"]):
            _copy_mesh_host(dynamic_host, trace_input, mesh_device, dtype=ttnn.bfloat16)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            replay = _compose_residual(traced_output)
            replay_hosts.append(replay)
            _assert_pcc(reference["dynamic_outputs"][replay_idx], replay, 0.99, f"advancing trace {replay_idx}")
            physical_offset = seq_len + replay_idx
            _assert_pcc(
                reference["dynamic_keys"][replay_idx],
                _compose_cache(trace_key)[1:2, :, physical_offset : physical_offset + 1, :],
                0.99,
                f"advancing trace key {replay_idx}",
            )
            _assert_pcc(
                reference["dynamic_values"][replay_idx],
                _compose_cache(trace_value)[1:2, :, physical_offset : physical_offset + 1, :],
                0.99,
                f"advancing trace value {replay_idx}",
            )
        assert not torch.equal(replay_hosts[0], replay_hosts[1])
        current_hosts = _local_hosts(current_pos_tensor)
        rope_hosts = _local_hosts(rope_idx_tensor)
        assert all(int(host.reshape(-1)[0]) == seq_len + 2 for host in current_hosts)
        assert all(int(host.reshape(-1)[0]) == seq_len + 2 for host in rope_hosts)

        # Reuse the same captured page-table tensor with different contents,
        # and cross the 63/64 page boundary.  Logical page one now maps to
        # physical block one.
        _copy_mesh_host(
            torch.tensor([[0, 1]], dtype=torch.int32),
            table,
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        _copy_mesh_host(
            torch.tensor([PAGED_BLOCK_SIZE], dtype=torch.int32),
            current_pos_tensor,
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        _copy_mesh_host(
            torch.tensor([[PAGED_BLOCK_SIZE]], dtype=torch.int32),
            rope_idx_tensor,
            mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        _copy_mesh_host(reference["boundary_input"], trace_input, mesh_device, dtype=ttnn.bfloat16)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc(reference["boundary_output"], _compose_residual(traced_output), 0.99, "dynamic page-one trace")
        _assert_pcc(
            reference["boundary_key"],
            _compose_cache(trace_key)[1:2, :, 0:1, :],
            0.99,
            "dynamic page-one physical key",
        )
        _assert_pcc(
            reference["boundary_value"],
            _compose_cache(trace_value)[1:2, :, 0:1, :],
            0.99,
            "dynamic page-one physical value",
        )

        # Replay-only wall time: refresh the positions once, enqueue 30
        # executions, and synchronize once.  Positions stay inside page zero.
        _copy_mesh_host(
            torch.tensor([[1, 0]], dtype=torch.int32),
            table,
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        _copy_mesh_host(
            torch.tensor([18], dtype=torch.int32),
            current_pos_tensor,
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        _copy_mesh_host(
            torch.tensor([[18]], dtype=torch.int32),
            rope_idx_tensor,
            mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.synchronize_device(mesh_device)
        started = time.perf_counter()
        for _ in range(30):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        multichip_trace_ms = (time.perf_counter() - started) * 1000.0 / 30
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    trace_speedup = reference["single_trace_replay_ms"] / multichip_trace_ms
    record_property("multichip_trace_replay_ms", f"{multichip_trace_ms:.6f}")
    record_property("trace_replay_speedup", f"{trace_speedup:.6f}")
    record_property("trace_replay_efficiency", f"{trace_speedup / TARGET_TP_DEGREE:.6f}")

    def prefill_call():
        return decoder.prefill_forward(prefill_input, key_cache, value_cache)

    def decode_call():
        return decoder.decode_forward(decode_input, key_cache, value_cache, current_pos=seq_len)

    prefill_ms = _timed_ms(prefill_call, mesh_device, warmup=2, iterations=5)
    decode_ms = _timed_ms(decode_call, mesh_device, warmup=5, iterations=30)
    signpost("MULTICHIP_PREFILL")
    prefill_call()
    ttnn.synchronize_device(mesh_device)
    signpost("MULTICHIP_PREFILL_END")
    signpost("MULTICHIP_DECODE")
    for _ in range(5):
        decode_call()
    ttnn.synchronize_device(mesh_device)
    signpost("MULTICHIP_DECODE_END")
    prefill_speedup = reference["single_prefill_ms"] / prefill_ms
    decode_speedup = reference["single_decode_ms"] / decode_ms
    record_property("multichip_prefill_ms", f"{prefill_ms:.6f}")
    record_property("multichip_decode_ms", f"{decode_ms:.6f}")
    record_property("prefill_speedup", f"{prefill_speedup:.6f}")
    record_property("decode_speedup", f"{decode_speedup:.6f}")
    record_property("prefill_efficiency", f"{prefill_speedup / TARGET_TP_DEGREE:.6f}")
    record_property("decode_efficiency", f"{decode_speedup / TARGET_TP_DEGREE:.6f}")
    print(
        f"PERF single_prefill_ms={reference['single_prefill_ms']:.6f} multi_prefill_ms={prefill_ms:.6f} "
        f"speedup={prefill_speedup:.4f} efficiency={prefill_speedup / TARGET_TP_DEGREE:.4f}"
    )
    print(
        f"PERF single_decode_ms={reference['single_decode_ms']:.6f} multi_decode_ms={decode_ms:.6f} "
        f"speedup={decode_speedup:.4f} efficiency={decode_speedup / TARGET_TP_DEGREE:.4f}"
    )
    print(
        f"TRACE_PERF single_ms={reference['single_trace_replay_ms']:.6f} "
        f"multi_ms={multichip_trace_ms:.6f} speedup={trace_speedup:.4f} "
        f"efficiency={trace_speedup / TARGET_TP_DEGREE:.4f}"
    )

    # Allocate the exact per-layer local K/V shape for the full advertised
    # context without constructing any full-model component.
    full_cache_shape = ttnn.Shape([2048, 2, PAGED_BLOCK_SIZE, 128])
    full_key = ttnn.allocate_tensor_on_device(full_cache_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device)
    full_value = ttnn.allocate_tensor_on_device(full_cache_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device)
    assert tuple(full_key.shape) == (2048, 2, PAGED_BLOCK_SIZE, 128)
    assert tuple(full_value.shape) == (2048, 2, PAGED_BLOCK_SIZE, 128)
    assert len(ttnn.get_device_tensors(full_key)) == TARGET_TP_DEGREE
    record_property("full_context_local_cache_shape", "[2048,2,64,128] K and V")
    ttnn.deallocate(full_key)
    ttnn.deallocate(full_value)
    del decoder, state
    gc.collect()


def _run_decode_policy_candidate(mesh_device, record_property, *, candidate_id: str, changes: dict):
    """Run one real-weight policy variable against the selected control."""

    config = _config()
    state = _real_state()
    policy = replace(MultiChipConfig(), **changes)
    ccl = get_tt_ccl(mesh_device)
    control = MultiChipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=MAX_CACHE_LEN,
        tt_ccl=ccl,
    )
    candidate = MultiChipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=MAX_CACHE_LEN,
        multichip_config=policy,
        tt_ccl=ccl,
    )
    host = torch.randn(
        (1, 1, 1, config.hidden_size),
        generator=torch.Generator().manual_seed(7201),
        dtype=torch.bfloat16,
    )
    hidden = _mesh_input(host, mesh_device)
    control_key, control_value = _mesh_contiguous_caches(config, mesh_device, batch=1)
    candidate_key, candidate_value = _mesh_contiguous_caches(config, mesh_device, batch=1)
    control_output = control.decode_forward(hidden, control_key, control_value, current_pos=7)
    candidate_output = candidate.decode_forward(hidden, candidate_key, candidate_value, current_pos=7)
    ttnn.synchronize_device(mesh_device)
    output_pcc = _measured_pcc(
        _compose_residual(control_output),
        _compose_residual(candidate_output),
        0.999,
        f"geometry {candidate_id} output",
    )
    key_pcc = _measured_pcc(
        _compose_cache(control_key)[:, :, 7:8, :],
        _compose_cache(candidate_key)[:, :, 7:8, :],
        0.999,
        f"geometry {candidate_id} key",
    )
    value_pcc = _measured_pcc(
        _compose_cache(control_value)[:, :, 7:8, :],
        _compose_cache(candidate_value)[:, :, 7:8, :],
        0.999,
        f"geometry {candidate_id} value",
    )

    def control_call():
        return control.decode_forward(hidden, control_key, control_value, current_pos=7)

    def candidate_call():
        return candidate.decode_forward(hidden, candidate_key, candidate_value, current_pos=7)

    control_ms = _timed_ms(control_call, mesh_device, warmup=5, iterations=30)
    candidate_ms = _timed_ms(candidate_call, mesh_device, warmup=5, iterations=30)

    def trace_ms(call):
        call()
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = call()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        try:
            started = time.perf_counter()
            for _ in range(100):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            elapsed = (time.perf_counter() - started) * 1000.0 / 100
            assert torch.isfinite(_compose_residual(traced_output)).all()
            return elapsed
        finally:
            ttnn.release_trace(mesh_device, trace_id)

    control_trace_ms = trace_ms(control_call)
    candidate_trace_ms = trace_ms(candidate_call)
    record_property("candidate_id", candidate_id)
    record_property("candidate_policy", repr(changes))
    record_property("output_pcc", f"{output_pcc:.10f}")
    record_property("key_pcc", f"{key_pcc:.10f}")
    record_property("value_pcc", f"{value_pcc:.10f}")
    record_property("control_eager_ms", f"{control_ms:.6f}")
    record_property("candidate_eager_ms", f"{candidate_ms:.6f}")
    record_property("control_trace_ms", f"{control_trace_ms:.6f}")
    record_property("candidate_trace_ms", f"{candidate_trace_ms:.6f}")
    print(
        f"POLICY {candidate_id} control_eager_ms={control_ms:.6f} candidate_eager_ms={candidate_ms:.6f} "
        f"control_trace_ms={control_trace_ms:.6f} candidate_trace_ms={candidate_trace_ms:.6f}"
    )


@_mesh_test
def test_multichip_decode_geometry_candidate(mesh_device, record_property):
    """One-real-weight-variable-at-a-time DRAM-sharded geometry sweep."""

    candidate_id = os.environ.get("MULTICHIP_GEOMETRY_CANDIDATE")
    if candidate_id not in GEOMETRY_CANDIDATES:
        pytest.skip(f"set MULTICHIP_GEOMETRY_CANDIDATE to one of {sorted(GEOMETRY_CANDIDATES)}")
    _run_decode_policy_candidate(
        mesh_device,
        record_property,
        candidate_id=candidate_id,
        changes=GEOMETRY_CANDIDATES[candidate_id],
    )


@_mesh_test
def test_multichip_topology_candidate(mesh_device, record_property):
    """Isolate collective decomposition, precision, link, and topology choices."""

    candidate_id = os.environ.get("MULTICHIP_TOPOLOGY_CANDIDATE")
    if candidate_id not in TOPOLOGY_CANDIDATES:
        pytest.skip(f"set MULTICHIP_TOPOLOGY_CANDIDATE to one of {sorted(TOPOLOGY_CANDIDATES)}")
    _run_decode_policy_candidate(
        mesh_device,
        record_property,
        candidate_id=candidate_id,
        changes=TOPOLOGY_CANDIDATES[candidate_id],
    )


class _PrefillL1InputsDecoder(MultiChipDecoder):
    """Whole-prefill candidate that applies every profiler L1-input hint."""

    def _prefill_linear(self, x, weight, *, role: str, seq_len: int, k: int, n: int, compute_kernel):
        l1_input = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        output = super()._prefill_linear(
            l1_input,
            weight,
            role=role,
            seq_len=seq_len,
            k=k,
            n=n,
            compute_kernel=compute_kernel,
        )
        ttnn.deallocate(l1_input)
        return output


@_mesh_test
def test_multichip_prefill_l1_inputs_candidate(mesh_device, record_property):
    """Measure all four advised L1 matmul inputs as one prefill family."""

    if os.environ.get("RUN_MULTICHIP_PREFILL_L1_CANDIDATE") != "1":
        pytest.skip("set RUN_MULTICHIP_PREFILL_L1_CANDIDATE=1 for the prefill L1-input family")
    config = _config()
    state = _real_state()
    ccl = get_tt_ccl(mesh_device)
    control = MultiChipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=MAX_CACHE_LEN,
        tt_ccl=ccl,
    )
    candidate = _PrefillL1InputsDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=MAX_CACHE_LEN,
        tt_ccl=ccl,
    )
    host = torch.randn(
        (1, 1, 7, config.hidden_size),
        generator=torch.Generator().manual_seed(7411),
        dtype=torch.bfloat16,
    )
    hidden = _mesh_input(host, mesh_device)
    control_key, control_value = _mesh_contiguous_caches(config, mesh_device, batch=1)
    candidate_key, candidate_value = _mesh_contiguous_caches(config, mesh_device, batch=1)
    control_output = control.prefill_forward(hidden, control_key, control_value)
    candidate_output = candidate.prefill_forward(hidden, candidate_key, candidate_value)
    ttnn.synchronize_device(mesh_device)
    output_pcc = _measured_pcc(
        _compose_residual(control_output),
        _compose_residual(candidate_output),
        0.999,
        "prefill L1-input output",
    )
    key_pcc = _measured_pcc(
        _compose_cache(control_key)[:, :, :7, :],
        _compose_cache(candidate_key)[:, :, :7, :],
        0.999,
        "prefill L1-input key",
    )
    value_pcc = _measured_pcc(
        _compose_cache(control_value)[:, :, :7, :],
        _compose_cache(candidate_value)[:, :, :7, :],
        0.999,
        "prefill L1-input value",
    )
    control_ms = _timed_ms(
        lambda: control.prefill_forward(hidden, control_key, control_value),
        mesh_device,
        warmup=5,
        iterations=30,
    )
    candidate_ms = _timed_ms(
        lambda: candidate.prefill_forward(hidden, candidate_key, candidate_value),
        mesh_device,
        warmup=5,
        iterations=30,
    )
    record_property("candidate_id", "prefill_l1_inputs")
    record_property("logical_seq_len", "7")
    record_property("output_pcc", f"{output_pcc:.10f}")
    record_property("key_pcc", f"{key_pcc:.10f}")
    record_property("value_pcc", f"{value_pcc:.10f}")
    record_property("control_prefill_ms", f"{control_ms:.6f}")
    record_property("candidate_prefill_ms", f"{candidate_ms:.6f}")
    print(f"PREFILL_L1 control_ms={control_ms:.6f} candidate_ms={candidate_ms:.6f}")
    del control, candidate, state
    gc.collect()


@_mesh_test
def test_multichip_stage_before_after_candidate(mesh_device, record_property):
    """Compare the inherited and selected policies with production defaults."""

    if os.environ.get("RUN_MULTICHIP_STAGE_AB") != "1":
        pytest.skip("set RUN_MULTICHIP_STAGE_AB=1 for the production-default stage A/B")
    config = _config()
    state = _real_state()
    final_policy = MultiChipConfig()
    starting_policy = replace(
        final_policy,
        optimized=replace(
            final_policy.optimized,
            explicit_sdpa_program_config=False,
            explicit_sdpa_compute_kernel=False,
        ),
    )
    ccl = get_tt_ccl(mesh_device)
    control = MultiChipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=MAX_CACHE_LEN,
        multichip_config=starting_policy,
        tt_ccl=ccl,
    )
    candidate = MultiChipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=MAX_CACHE_LEN,
        multichip_config=final_policy,
        tt_ccl=ccl,
    )
    seq_len = 39
    generator = torch.Generator().manual_seed(7421)
    prefill_hidden = _mesh_input(
        torch.randn((1, 1, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16),
        mesh_device,
    )
    decode_hidden = _mesh_input(
        torch.randn((1, 1, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16),
        mesh_device,
    )
    control_key, control_value = _mesh_contiguous_caches(config, mesh_device, batch=1)
    candidate_key, candidate_value = _mesh_contiguous_caches(config, mesh_device, batch=1)
    control_prefill = control.prefill_forward(prefill_hidden, control_key, control_value)
    candidate_prefill = candidate.prefill_forward(prefill_hidden, candidate_key, candidate_value)
    control_decode = control.decode_forward(decode_hidden, control_key, control_value, current_pos=seq_len)
    candidate_decode = candidate.decode_forward(decode_hidden, candidate_key, candidate_value, current_pos=seq_len)
    ttnn.synchronize_device(mesh_device)
    prefill_pcc = _measured_pcc(
        _compose_residual(control_prefill),
        _compose_residual(candidate_prefill),
        0.999,
        "stage A/B prefill",
    )
    decode_pcc = _measured_pcc(
        _compose_residual(control_decode),
        _compose_residual(candidate_decode),
        0.999,
        "stage A/B decode",
    )
    key_pcc = _measured_pcc(
        _compose_cache(control_key)[:, :, : seq_len + 1, :],
        _compose_cache(candidate_key)[:, :, : seq_len + 1, :],
        0.999,
        "stage A/B key",
    )
    value_pcc = _measured_pcc(
        _compose_cache(control_value)[:, :, : seq_len + 1, :],
        _compose_cache(candidate_value)[:, :, : seq_len + 1, :],
        0.999,
        "stage A/B value",
    )

    control_prefill_ms = _timed_ms(
        lambda: control.prefill_forward(prefill_hidden, control_key, control_value),
        mesh_device,
        warmup=5,
        iterations=30,
    )
    candidate_prefill_ms = _timed_ms(
        lambda: candidate.prefill_forward(prefill_hidden, candidate_key, candidate_value),
        mesh_device,
        warmup=5,
        iterations=30,
    )
    control_eager_ms = _timed_ms(
        lambda: control.decode_forward(decode_hidden, control_key, control_value, current_pos=seq_len),
        mesh_device,
        warmup=5,
        iterations=30,
    )
    candidate_eager_ms = _timed_ms(
        lambda: candidate.decode_forward(decode_hidden, candidate_key, candidate_value, current_pos=seq_len),
        mesh_device,
        warmup=5,
        iterations=30,
    )

    def trace_ms(decoder, key_cache, value_cache):
        call = lambda: decoder.decode_forward(
            decode_hidden,
            key_cache,
            value_cache,
            current_pos=seq_len,
        )
        call()
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = call()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        try:
            started = time.perf_counter()
            for _ in range(100):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            elapsed = (time.perf_counter() - started) * 1000.0 / 100
            assert torch.isfinite(_compose_residual(traced_output)).all()
            return elapsed
        finally:
            ttnn.release_trace(mesh_device, trace_id)

    control_trace_ms = trace_ms(control, control_key, control_value)
    candidate_trace_ms = trace_ms(candidate, candidate_key, candidate_value)
    for name, value in (
        ("prefill_pcc", prefill_pcc),
        ("decode_pcc", decode_pcc),
        ("key_pcc", key_pcc),
        ("value_pcc", value_pcc),
        ("control_prefill_ms", control_prefill_ms),
        ("candidate_prefill_ms", candidate_prefill_ms),
        ("control_eager_ms", control_eager_ms),
        ("candidate_eager_ms", candidate_eager_ms),
        ("control_trace_ms", control_trace_ms),
        ("candidate_trace_ms", candidate_trace_ms),
    ):
        record_property(name, f"{value:.10f}" if "pcc" in name else f"{value:.6f}")
    print(
        f"STAGE_AB prefill_ms={control_prefill_ms:.6f}/{candidate_prefill_ms:.6f} "
        f"eager_ms={control_eager_ms:.6f}/{candidate_eager_ms:.6f} "
        f"trace_ms={control_trace_ms:.6f}/{candidate_trace_ms:.6f}"
    )
    del control, candidate, state
    gc.collect()


@_mesh_test
def test_multichip_profiler_smoke(mesh_device):
    """Reduced-layer profiler target, intentionally separate from watcher."""

    if os.environ.get("RUN_MULTICHIP_DECODER_PROFILER") != "1":
        pytest.skip("set RUN_MULTICHIP_DECODER_PROFILER=1 for reduced profiler collection")
    config = _config()
    state = _real_state()
    decoder = MultiChipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=MAX_CACHE_LEN,
        tt_ccl=get_tt_ccl(mesh_device),
    )
    generator = torch.Generator().manual_seed(7001)
    prefill_input = _mesh_input(
        torch.randn((1, 1, 7, config.hidden_size), generator=generator, dtype=torch.bfloat16), mesh_device
    )
    decode_input = _mesh_input(
        torch.randn((1, 1, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16), mesh_device
    )
    key_cache, value_cache = _mesh_contiguous_caches(config, mesh_device, batch=1)
    signpost("MULTICHIP_PREFILL")
    decoder.prefill_forward(prefill_input, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    signpost("MULTICHIP_PREFILL_END")
    current_pos_tensor = _position_tensor(7, mesh_device, rope=False)
    rope_idx_tensor = _position_tensor(7, mesh_device, rope=True)
    decoder.decode_forward_from_position_tensor(
        decode_input,
        key_cache,
        value_cache,
        current_pos_tensor=current_pos_tensor,
        rope_idx_tensor=rope_idx_tensor,
        advance_position=True,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward_from_position_tensor(
        decode_input,
        key_cache,
        value_cache,
        current_pos_tensor=current_pos_tensor,
        rope_idx_tensor=rope_idx_tensor,
        advance_position=True,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    try:
        signpost("MULTICHIP_TRACE_DECODE")
        for _ in range(5):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        assert torch.isfinite(_compose_residual(traced_output)).all()
        signpost("MULTICHIP_TRACE_DECODE_END")
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@_mesh_test
def test_multichip_watcher_stress(mesh_device):
    """Mesh-only watcher target for CCL, paging, cache writes, and trace replay."""

    if os.environ.get("RUN_MULTICHIP_DECODER_WATCHER") != "1":
        pytest.skip("set RUN_MULTICHIP_DECODER_WATCHER=1 for the watcher stress run")
    config = _config()
    state = _real_state()
    decoder = MultiChipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=MAX_CACHE_LEN,
        tt_ccl=get_tt_ccl(mesh_device),
    )
    generator = torch.Generator().manual_seed(7017)
    prefill_input = _mesh_input(
        torch.randn((1, 1, 7, config.hidden_size), generator=generator, dtype=torch.bfloat16), mesh_device
    )
    decode_input = _mesh_input(
        torch.randn((1, 1, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16), mesh_device
    )
    page_table = _page_table(torch.tensor([[1, 0]], dtype=torch.int32), mesh_device)
    key_cache, value_cache = _mesh_paged_caches(config, mesh_device, blocks=2)

    prefill_output = decoder.prefill_forward(
        prefill_input,
        key_cache,
        value_cache,
        page_table=page_table,
    )
    assert torch.isfinite(_compose_residual(prefill_output)).all()
    assert tuple(key_cache.shape) == (2, 2, PAGED_BLOCK_SIZE, 128)
    assert len(ttnn.get_device_tensors(key_cache)) == TARGET_TP_DEGREE

    # Repeated nonaligned positions stress CCL semaphore reuse and paged cache
    # writes.  Position 64 also crosses to the page-table entry mapped to
    # physical block zero.
    for current_pos in (*range(7, 23), PAGED_BLOCK_SIZE):
        output = decoder.decode_forward(
            decode_input,
            key_cache,
            value_cache,
            current_pos=current_pos,
            page_table=page_table,
        )
        assert torch.isfinite(_compose_residual(output)).all()
    current_pos_tensor = _position_tensor(23, mesh_device, rope=False)
    rope_idx_tensor = _position_tensor(23, mesh_device, rope=True)
    decoder.decode_forward_from_position_tensor(
        decode_input,
        key_cache,
        value_cache,
        current_pos_tensor=current_pos_tensor,
        rope_idx_tensor=rope_idx_tensor,
        page_table=page_table,
        advance_position=True,
    )
    _copy_mesh_host(
        torch.tensor([23], dtype=torch.int32),
        current_pos_tensor,
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    _copy_mesh_host(
        torch.tensor([[23]], dtype=torch.int32),
        rope_idx_tensor,
        mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward_from_position_tensor(
        decode_input,
        key_cache,
        value_cache,
        current_pos_tensor=current_pos_tensor,
        rope_idx_tensor=rope_idx_tensor,
        page_table=page_table,
        advance_position=True,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    replay_hosts = []
    try:
        for _ in range(100):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            replay_hosts.append(_compose_residual(traced_output))
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    for replay in replay_hosts:
        assert torch.isfinite(replay).all()
    assert not torch.equal(replay_hosts[0], replay_hosts[-1])
    assert all(int(host.reshape(-1)[0]) == 123 for host in _local_hosts(current_pos_tensor))
    assert all(int(host.reshape(-1)[0]) == 123 for host in _local_hosts(rope_idx_tensor))
