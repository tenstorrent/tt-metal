# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import inspect
import json
import os
import time
from pathlib import Path

import pytest
import torch
from tracy import signpost

import models.autoports.meta_llama_llama_3_1_8b_instruct.tt.tp2_multichip_decoder as tp2_multichip_decoder_module
import models.common.modules.tt_ccl as tt_ccl_module
import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _assert_pcc,
    _config,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.tp2_multichip_decoder import (
    PAGED_BLOCK_SIZE,
    TARGET_MESH_SHAPE,
    TARGET_TP_DEGREE,
    MultiChipConfig,
    MultiChipDecoder,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import OptimizationConfig, OptimizedDecoder
from models.common.modules.tt_ccl import get_tt_ccl

MAX_CACHE_LEN = 128
MULTICHIP_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "trace_region_size": 100_000_000,
    "require_exact_physical_num_devices": True,
}


def _mesh_test(func):
    parametrized = pytest.mark.parametrize(
        "mesh_device, device_params",
        [(TARGET_TP_DEGREE, MULTICHIP_DEVICE_PARAMS)],
        indirect=True,
        ids=["p300-1x2-ring"],
    )(func)
    # First-use full-shape BFP packing and JIT compilation can exceed the
    # repository-wide 300 s alarm.  Interrupting a live TTNN frame makes
    # pytest render Tensor reprs, which performs a device read during failure
    # formatting and masks the timeout as a CQ hang.
    return pytest.mark.timeout(1800)(parametrized)


def _single_test(func):
    parametrized = pytest.mark.parametrize(
        "mesh_device, device_params",
        [(1, {"trace_region_size": 100_000_000})],
        indirect=True,
        ids=["blackhole-single-chip-reference"],
    )(func)
    return pytest.mark.timeout(1800)(parametrized)


def _mesh_input(tensor: torch.Tensor, mesh_device):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=MultiChipDecoder.mesh_mapper_for_input(mesh_device),
    )


def _single_input(tensor: torch.Tensor, mesh_device):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _single_caches(config, mesh_device, *, batch: int, dtype=ttnn.bfloat16):
    head_dim = config.hidden_size // config.num_attention_heads
    shape = (batch, config.num_key_value_heads, MAX_CACHE_LEN, head_dim)
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    kwargs = dict(
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.from_torch(zeros, **kwargs), ttnn.from_torch(zeros, **kwargs)


def _mesh_contiguous_caches(config, mesh_device, *, batch: int, dtype=ttnn.bfloat16):
    head_dim = config.hidden_size // config.num_attention_heads
    shape = (batch, config.num_key_value_heads, MAX_CACHE_LEN, head_dim)
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    kwargs = dict(
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=MultiChipDecoder.mesh_mapper_for_cache(mesh_device),
    )
    return ttnn.from_torch(zeros, **kwargs), ttnn.from_torch(zeros, **kwargs)


def _mesh_paged_caches(config, mesh_device, *, num_blocks: int, dtype=ttnn.bfloat16):
    head_dim = config.hidden_size // config.num_attention_heads
    shape = (num_blocks, config.num_key_value_heads, PAGED_BLOCK_SIZE, head_dim)
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    kwargs = dict(
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=MultiChipDecoder.mesh_mapper_for_cache(mesh_device),
    )
    return ttnn.from_torch(zeros, **kwargs), ttnn.from_torch(zeros, **kwargs)


def _page_table(table: torch.Tensor, mesh_device):
    return ttnn.from_torch(
        table.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _single_host(tensor):
    value = ttnn.to_torch(tensor)
    if isinstance(value, list):
        assert len(value) == 1
        value = value[0]
    return value


def _mesh_hosts(tensor):
    return [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]


def _replicated_host(tensor):
    values = _mesh_hosts(tensor)
    for other in values[1:]:
        assert torch.equal(values[0], other), "replicated decoder outputs diverged across TP ranks"
    return values[0]


def _compose_cache_heads(cache):
    return torch.cat(_mesh_hosts(cache), dim=1)


def _patterned_state(config):
    """Create exact production shapes without an expensive 218M-value RNG pass.

    Each projection uses a distinct rank-one row/column pattern.  TP ranks,
    Q/K/V regions, and MLP feature shards therefore remain nonidentical, so a
    bad split, pack order, or reduction is still visible to the baseline PCC.
    """

    prefix = f"model.layers.{LAYER_IDX}."
    head_dim = config.hidden_size // config.num_attention_heads
    kv_width = config.num_key_value_heads * head_dim

    def pattern(shape, phase):
        rows = ((torch.arange(shape[0], dtype=torch.float32) + phase) % 29 - 14) / 29
        columns = ((torch.arange(shape[1], dtype=torch.float32) + 3 * phase) % 31 - 15) / 31
        return torch.outer(rows, columns).mul_(0.02).to(torch.bfloat16)

    return {
        prefix + "input_layernorm.weight": torch.linspace(0.99, 1.01, config.hidden_size, dtype=torch.bfloat16),
        prefix
        + "post_attention_layernorm.weight": torch.linspace(1.01, 0.99, config.hidden_size, dtype=torch.bfloat16),
        prefix + "self_attn.q_proj.weight": pattern((config.hidden_size, config.hidden_size), 1),
        prefix + "self_attn.k_proj.weight": pattern((kv_width, config.hidden_size), 2),
        prefix + "self_attn.v_proj.weight": pattern((kv_width, config.hidden_size), 3),
        prefix + "self_attn.o_proj.weight": pattern((config.hidden_size, config.hidden_size), 4),
        prefix + "mlp.gate_proj.weight": pattern((config.intermediate_size, config.hidden_size), 5),
        prefix + "mlp.up_proj.weight": pattern((config.intermediate_size, config.hidden_size), 6),
        prefix + "mlp.down_proj.weight": pattern((config.hidden_size, config.intermediate_size), 7),
    }


def _selected_policy(multichip_config: MultiChipConfig | None = None):
    return multichip_config or MultiChipConfig(
        optimized=OptimizationConfig(
            attention_weight_dtype=ttnn.bfloat4_b,
            gate_up_weight_dtype=ttnn.bfloat4_b,
            down_weight_dtype=ttnn.bfloat4_b,
            decode_matmul_strategy="dram_sharded",
            output_cores=8,
        )
    )


def _build_baseline(mesh_device, *, batch: int, multichip_config: MultiChipConfig | None = None):
    config = _config()
    state = _patterned_state(config)
    policy = _selected_policy(multichip_config)
    baseline = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=MAX_CACHE_LEN,
        optimization_config=OptimizationConfig(
            attention_weight_dtype=policy.optimized.attention_weight_dtype,
            gate_up_weight_dtype=policy.optimized.gate_up_weight_dtype,
            down_weight_dtype=policy.optimized.down_weight_dtype,
        ),
    )
    ttnn.synchronize_device(mesh_device)
    return config, state, baseline


def _build_multichip(mesh_device, *, batch: int, multichip_config: MultiChipConfig | None = None):
    config = _config()
    state = _patterned_state(config)
    policy = _selected_policy(multichip_config)
    multichip = MultiChipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=MAX_CACHE_LEN,
        multichip_config=policy,
        tt_ccl=get_tt_ccl(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)
    return config, state, multichip


def test_runtime_path_is_real_multichip_and_host_fallback_free():
    assert issubclass(MultiChipDecoder, OptimizedDecoder)
    assert MultiChipDecoder.single_chip_baseline is OptimizedDecoder
    assert TARGET_MESH_SHAPE == (1, 2)
    for method in (
        MultiChipDecoder.prefill_forward,
        MultiChipDecoder.decode_forward,
        MultiChipDecoder._all_reduce_partial,
        MultiChipDecoder._mlp_prefill_tp,
        MultiChipDecoder._mlp_decode_tp,
    ):
        source = inspect.getsource(method)
        for token in ("from_torch", "to_torch", "torch.", "super().prefill", "super().decode"):
            assert token not in source, f"{method.__name__} contains forbidden runtime token {token!r}"
    assert "all_reduce_async" in inspect.getsource(MultiChipDecoder._all_reduce_partial)
    assert "tt_ccl" in inspect.signature(MultiChipDecoder.from_state_dict).parameters
    assert "get_tt_ccl" in inspect.getsource(MultiChipDecoder.__init__)


def test_multichip_context_capacity_contract():
    contract_path = Path(__file__).resolve().parents[1] / "doc" / "context_contract.json"
    contract = json.loads(contract_path.read_text())
    evidence = contract["capacity_evidence"]
    assert contract["current_supported_context"] == contract["hf_advertised_context"] == 131072
    assert contract["limiting_reason"] is None
    assert evidence["target_mesh"] == "1x2 Blackhole P300c, TP=2"
    assert evidence["page_pool_blocks"] * evidence["page_block_size"] == 131072
    assert evidence["per_device_kv_heads"] == 4
    kv_elements = 32 * 4 * 131072 * 128 * 2
    assert evidence["per_device_bf16_kv_cache_bytes"] == kv_elements * 2
    assert evidence["per_device_bfp8_kv_cache_bytes"] == kv_elements * 1088 // 1024
    assert evidence["bf16_plan_total_bytes"] < evidence["device_allocator_dram_bytes"]


def test_tp2_multichip_decoder_stack_shares_one_ccl_owner(monkeypatch):
    """A 32-layer stack must share one persistent semaphore owner per mesh."""

    class Grid:
        x = 11
        y = 10

    class FakeMesh:
        def __init__(self, mesh_id):
            self.mesh_id = mesh_id
            self.shape = TARGET_MESH_SHAPE

        def id(self):
            return self.mesh_id

        def compute_with_storage_grid_size(self):
            return Grid()

        def get_num_devices(self):
            return TARGET_TP_DEGREE

    semaphore_calls = []

    def create_global_semaphore(mesh, core_ranges, initial_value):
        handle = object()
        semaphore_calls.append((mesh, core_ranges, initial_value, handle))
        return handle

    class FakeTTNN:
        CoreCoord = staticmethod(lambda x, y: (x, y))
        CoreRange = staticmethod(lambda start, end: (start, end))
        CoreRangeSet = staticmethod(frozenset)

    FakeTTNN.create_global_semaphore = staticmethod(create_global_semaphore)

    mesh = FakeMesh(1701)
    foreign_mesh = FakeMesh(1702)
    monkeypatch.setattr(tt_ccl_module, "ttnn", FakeTTNN)
    tt_ccl_module.clear_tt_ccl_cache()
    try:
        owners = [tt_ccl_module.get_tt_ccl(mesh) for _ in range(32)]
        assert len({id(owner) for owner in owners}) == 1
        shared = owners[0]
        assert len(semaphore_calls) == 36
        assert sum(len(handles) for handles in shared.barrier_semaphore_handles) == 6
        assert sum(len(handles) for axis in shared.ag_semaphore_handles for handles in axis) == 12
        assert sum(len(handles) for axis in shared.rs_semaphore_handles for handles in axis) == 18

        def fake_optimized_init(instance, *, optimization_config, **kwargs):
            instance.optimization_config = optimization_config
            instance.mesh_device = kwargs["mesh_device"]
            instance.hidden_size = kwargs["hidden_size"]
            instance.num_heads = kwargs["num_heads"]
            instance.num_kv_heads = kwargs["num_kv_heads"]
            instance.intermediate_size = kwargs["intermediate_size"]
            instance.batch = kwargs["batch"]

        resolver_calls = []

        def resolve_tt_ccl(candidate_mesh):
            resolver_calls.append(candidate_mesh)
            return tt_ccl_module.get_tt_ccl(candidate_mesh)

        monkeypatch.setattr(OptimizedDecoder, "__init__", fake_optimized_init)
        monkeypatch.setattr(tp2_multichip_decoder_module, "_width_sharded_l1", lambda **kwargs: object())
        monkeypatch.setattr(tp2_multichip_decoder_module, "_dram_matmul_program_config", lambda **kwargs: object())
        monkeypatch.setattr(tp2_multichip_decoder_module, "get_tt_ccl", resolve_tt_ccl)

        constructor_kwargs = dict(
            multichip_config=_selected_policy(),
            global_num_heads=32,
            global_num_kv_heads=8,
            global_intermediate_size=14336,
            mesh_device=mesh,
            hidden_size=4096,
            num_heads=16,
            num_kv_heads=4,
            intermediate_size=7168,
            batch=32,
        )
        layers = [MultiChipDecoder(**constructor_kwargs) for _ in range(32)]
        assert len({id(layer.tt_ccl) for layer in layers}) == 1
        assert all(layer.tt_ccl is shared for layer in layers)
        assert all(layer._barrier_semaphores is shared.barrier_semaphore_handles[2] for layer in layers)
        assert resolver_calls == [mesh] * 32

        injected = MultiChipDecoder(**constructor_kwargs, tt_ccl=shared)
        assert injected.tt_ccl is shared
        assert resolver_calls == [mesh] * 32

        foreign_owner = tt_ccl_module.get_tt_ccl(foreign_mesh)
        with pytest.raises(ValueError, match="decoder's mesh_device"):  # allow-pytest.raises: fake mesh contract
            MultiChipDecoder(**constructor_kwargs, tt_ccl=foreign_owner)
    finally:
        tt_ccl_module.clear_tt_ccl_cache()


_CORRECTNESS_REFERENCE = None


@_single_test
def test_multichip_correctness_single_chip_reference(mesh_device):
    """Run the optimized baseline before the fabric mesh claims its FD queues."""

    batch = 32
    seq_len = 7
    config, state, baseline = _build_baseline(mesh_device, batch=batch)

    single_key, single_value = _single_caches(config, mesh_device, batch=batch)
    generator = torch.Generator().manual_seed(101)
    prefill = torch.randn((1, batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    decode = torch.randn((1, batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    prefill_input = _single_input(prefill, mesh_device)
    decode_input = _single_input(decode, mesh_device)
    baseline_prefill = baseline.prefill_forward(prefill_input, single_key, single_value)
    ttnn.synchronize_device(mesh_device)
    baseline_prefill_host = _single_host(baseline_prefill)
    baseline_decode = baseline.decode_forward(decode_input, single_key, single_value, current_pos=seq_len)
    ttnn.synchronize_device(mesh_device)
    baseline_decode_host = _single_host(baseline_decode)
    single_key_host = _single_host(single_key)[:, :, : seq_len + 1, :]
    single_value_host = _single_host(single_value)[:, :, : seq_len + 1, :]
    for tensor in (prefill_input, decode_input, baseline_prefill, baseline_decode, single_key, single_value):
        ttnn.deallocate(tensor)
    gc.collect()

    generator = torch.Generator().manual_seed(713)
    prefill = torch.randn((1, batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    decode = torch.randn((1, batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    paged_key, paged_value = _single_caches(config, mesh_device, batch=batch)
    paged_prefill_input = _single_input(prefill, mesh_device)
    paged_decode_input = _single_input(decode, mesh_device)
    paged_prefill_output = baseline.prefill_forward(paged_prefill_input, paged_key, paged_value)
    paged_decode_output = baseline.decode_forward(paged_decode_input, paged_key, paged_value, current_pos=seq_len)
    ttnn.synchronize_device(mesh_device)
    paged_decode_host = _single_host(paged_decode_output)
    paged_key_host = _single_host(paged_key)[:, :, : seq_len + 1, :]
    paged_value_host = _single_host(paged_value)[:, :, : seq_len + 1, :]
    for tensor in (
        paged_prefill_input,
        paged_decode_input,
        paged_prefill_output,
        paged_decode_output,
        paged_key,
        paged_value,
    ):
        ttnn.deallocate(tensor)
    gc.collect()

    # Cross virtual page 0 -> 1 with BFP8 cache storage.  The 31-token
    # non-aligned prefill stays within the baseline kernel's L1 limit; ordered
    # decode writes then advance the cache through positions 31..65.
    cross_prefill_len = 31
    generator = torch.Generator().manual_seed(6465)
    cross_prefill = torch.randn(
        (1, batch, cross_prefill_len, config.hidden_size), generator=generator, dtype=torch.bfloat16
    )
    cross_decode = torch.randn((1, batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    cross_key, cross_value = _single_caches(config, mesh_device, batch=batch, dtype=ttnn.bfloat8_b)
    cross_prefill_input = _single_input(cross_prefill, mesh_device)
    cross_decode_input = _single_input(cross_decode, mesh_device)
    baseline_cross_prefill = baseline.prefill_forward(cross_prefill_input, cross_key, cross_value)
    ttnn.synchronize_device(mesh_device)
    cross_prefill_host = _single_host(baseline_cross_prefill)
    ttnn.deallocate(baseline_cross_prefill)
    cross_decode_hosts = []
    for current_pos in range(cross_prefill_len, PAGED_BLOCK_SIZE + 2):
        output = baseline.decode_forward(cross_decode_input, cross_key, cross_value, current_pos=current_pos)
        if current_pos >= PAGED_BLOCK_SIZE - 1:
            cross_decode_hosts.append(_single_host(output))
        ttnn.deallocate(output)
    ttnn.synchronize_device(mesh_device)
    cross_key_host = _single_host(cross_key)[:, :, : PAGED_BLOCK_SIZE + 2, :]
    cross_value_host = _single_host(cross_value)[:, :, : PAGED_BLOCK_SIZE + 2, :]

    global _CORRECTNESS_REFERENCE
    _CORRECTNESS_REFERENCE = {
        "contiguous_prefill": baseline_prefill_host,
        "contiguous_decode": baseline_decode_host,
        "contiguous_key": single_key_host,
        "contiguous_value": single_value_host,
        "paged_decode": paged_decode_host,
        "paged_key": paged_key_host,
        "paged_value": paged_value_host,
        "cross_page_prefill": cross_prefill_host,
        "cross_page_decodes": cross_decode_hosts,
        "cross_page_key": cross_key_host,
        "cross_page_value": cross_value_host,
    }
    for tensor in (
        cross_prefill_input,
        cross_decode_input,
        cross_key,
        cross_value,
    ):
        ttnn.deallocate(tensor)
    ttnn.synchronize_device(mesh_device)
    del baseline, state
    gc.collect()


@_mesh_test
def test_multichip_correctness_cache_determinism_and_trace_contract(mesh_device):
    assert _CORRECTNESS_REFERENCE is not None, "run the single-chip reference item with this target item"
    reference = _CORRECTNESS_REFERENCE
    batch = 32
    seq_len = 7
    config, state, multichip = _build_multichip(mesh_device, batch=batch)
    assert multichip.tt_ccl is get_tt_ccl(mesh_device)
    assert get_tt_ccl(mesh_device) is get_tt_ccl(mesh_device)

    # Non-aligned contiguous prefill/decode and local KV-head ownership.
    mesh_key, mesh_value = _mesh_contiguous_caches(config, mesh_device, batch=batch)
    generator = torch.Generator().manual_seed(101)
    prefill = torch.randn((1, batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    decode = torch.randn((1, batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    prefill_input = _mesh_input(prefill, mesh_device)
    decode_input = _mesh_input(decode, mesh_device)
    actual_prefill = multichip.prefill_forward(prefill_input, mesh_key, mesh_value)
    actual_decode = multichip.decode_forward(decode_input, mesh_key, mesh_value, current_pos=seq_len)
    ttnn.synchronize_device(mesh_device)
    _assert_pcc(reference["contiguous_prefill"], _replicated_host(actual_prefill), 0.99, "TP2 prefill vs optimized")
    _assert_pcc(reference["contiguous_decode"], _replicated_host(actual_decode), 0.99, "TP2 decode vs optimized")
    cache_slice = slice(0, seq_len + 1)
    _assert_pcc(
        reference["contiguous_key"],
        _compose_cache_heads(mesh_key)[:, :, cache_slice, :],
        0.99,
        "TP2 local key-cache heads",
    )
    _assert_pcc(
        reference["contiguous_value"],
        _compose_cache_heads(mesh_value)[:, :, cache_slice, :],
        0.99,
        "TP2 local value-cache heads",
    )
    assert tuple(actual_prefill.shape) == (1, batch, seq_len, config.hidden_size)
    assert tuple(actual_decode.shape) == (1, batch, 1, config.hidden_size)
    assert tuple(mesh_key.shape) == (batch, config.num_key_value_heads // 2, MAX_CACHE_LEN, 128)
    for tensor in (prefill_input, decode_input, actual_prefill, actual_decode, mesh_key, mesh_value):
        ttnn.deallocate(tensor)
    gc.collect()

    # Nontrivial physical page placement, repeated deterministic execution,
    # and exact local-head cache reconstruction against the same baseline.
    generator = torch.Generator().manual_seed(713)
    prefill = torch.randn((1, batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    decode = torch.randn((1, batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    table_host = torch.arange(2 * batch, dtype=torch.int32).reshape(batch, 2).flip(1)
    table = _page_table(table_host, mesh_device)
    paged_prefill_input = _mesh_input(prefill, mesh_device)
    paged_decode_input = _mesh_input(decode, mesh_device)

    outputs = []
    final_key = final_value = None
    for run in range(3):
        mesh_key, mesh_value = _mesh_paged_caches(config, mesh_device, num_blocks=2 * batch)
        actual_prefill = multichip.prefill_forward(paged_prefill_input, mesh_key, mesh_value, page_table=table)
        actual_decode = multichip.decode_forward(
            paged_decode_input,
            mesh_key,
            mesh_value,
            current_pos=seq_len,
            page_table=table,
        )
        actual_prefill_host = _replicated_host(actual_prefill)
        actual_decode_host = _replicated_host(actual_decode)
        _assert_pcc(reference["paged_decode"], actual_decode_host, 0.99, f"paged TP2 decode run={run}")
        outputs.append((actual_prefill_host, actual_decode_host))
        ttnn.deallocate(actual_prefill)
        ttnn.deallocate(actual_decode)
        if run == 2:
            final_key, final_value = mesh_key, mesh_value
        else:
            ttnn.deallocate(mesh_key)
            ttnn.deallocate(mesh_value)
            gc.collect()

    for output in outputs[1:]:
        assert torch.equal(outputs[0][0], output[0])
        assert torch.equal(outputs[0][1], output[1])

    key_blocks = _compose_cache_heads(final_key)
    value_blocks = _compose_cache_heads(final_value)
    key_physical = torch.stack([key_blocks[int(table_host[user, 0]), :, : seq_len + 1, :] for user in range(batch)])
    value_physical = torch.stack([value_blocks[int(table_host[user, 0]), :, : seq_len + 1, :] for user in range(batch)])
    _assert_pcc(reference["paged_key"], key_physical, 0.99, "paged physical key block")
    _assert_pcc(reference["paged_value"], value_physical, 0.99, "paged physical value block")
    for tensor in (paged_prefill_input, paged_decode_input, final_key, final_value, table):
        ttnn.deallocate(tensor)
    gc.collect()

    # Cross from virtual page 0 to page 1.  Ordered writes through positions
    # 31..65 exercise both page-table columns and the exact 63 -> 64 boundary.
    cross_prefill_len = 31
    generator = torch.Generator().manual_seed(6465)
    prefill = torch.randn((1, batch, cross_prefill_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    decode = torch.randn((1, batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    page0 = torch.arange(batch - 1, -1, -1, dtype=torch.int32)
    page1 = torch.arange(2 * batch - 1, batch - 1, -1, dtype=torch.int32)
    trace_table = torch.stack((page0, page1), dim=1)
    assert torch.unique(trace_table).numel() == 2 * batch
    table = _page_table(trace_table, mesh_device)
    key_cache, value_cache = _mesh_paged_caches(config, mesh_device, num_blocks=2 * batch, dtype=ttnn.bfloat8_b)
    cross_prefill_input = _mesh_input(prefill, mesh_device)
    cross_decode_input = _mesh_input(decode, mesh_device)
    cross_prefill = multichip.prefill_forward(cross_prefill_input, key_cache, value_cache, page_table=table)
    ttnn.synchronize_device(mesh_device)
    _assert_pcc(
        reference["cross_page_prefill"],
        _replicated_host(cross_prefill),
        0.99,
        "boundary-walk seed TP2 prefill",
    )
    ttnn.deallocate(cross_prefill)
    eager_hosts = []
    for current_pos in range(cross_prefill_len, PAGED_BLOCK_SIZE + 2):
        output = multichip.decode_forward(
            cross_decode_input,
            key_cache,
            value_cache,
            current_pos=current_pos,
            page_table=table,
        )
        if current_pos >= PAGED_BLOCK_SIZE - 1:
            eager_hosts.append(_replicated_host(output))
        ttnn.deallocate(output)
    ttnn.synchronize_device(mesh_device)
    for current_pos, (expected, actual) in enumerate(
        zip(reference["cross_page_decodes"], eager_hosts), start=PAGED_BLOCK_SIZE - 1
    ):
        _assert_pcc(expected, actual, 0.99, f"page-boundary TP2 decode position={current_pos}")

    for cache_name, cache, expected in (
        ("key", key_cache, reference["cross_page_key"]),
        ("value", value_cache, reference["cross_page_value"]),
    ):
        for rank, local_cache in enumerate(_mesh_hosts(cache)):
            head_slice = slice(4 * rank, 4 * (rank + 1))
            actual_page0 = torch.stack([local_cache[int(page0[user]), :, :, :] for user in range(batch)])
            actual_page1 = torch.stack([local_cache[int(page1[user]), :, :2, :] for user in range(batch)])
            _assert_pcc(
                expected[:, head_slice, :PAGED_BLOCK_SIZE, :],
                actual_page0,
                0.99,
                f"rank={rank} physical {cache_name} page=0",
            )
            _assert_pcc(
                expected[:, head_slice, PAGED_BLOCK_SIZE : PAGED_BLOCK_SIZE + 2, :],
                actual_page1,
                0.99,
                f"rank={rank} physical {cache_name} page=1",
            )
            assert torch.count_nonzero(local_cache[page1.to(torch.long), :, 2:, :]) == 0
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = multichip.decode_forward(
        cross_decode_input,
        key_cache,
        value_cache,
        current_pos=PAGED_BLOCK_SIZE + 1,
        page_table=table,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    outputs = []
    try:
        for _ in range(5):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            outputs.append(_replicated_host(traced_output))
        for idx, output in enumerate(outputs):
            _assert_pcc(eager_hosts[-1], output, 0.999, f"second-page trace replay={idx}")
        for output in outputs[1:]:
            assert torch.equal(outputs[0], output)
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    for tensor in (
        cross_prefill_input,
        cross_decode_input,
        traced_output,
        key_cache,
        value_cache,
        table,
    ):
        ttnn.deallocate(tensor)
    ttnn.synchronize_device(mesh_device)
    del multichip, state
    gc.collect()


@_mesh_test
def test_multichip_watcher_stress(mesh_device):
    """Exercise only the final ring path so watcher shutdown need not reconfigure fabric."""

    if os.environ.get("RUN_MULTICHIP_DECODER_WATCHER") != "1":
        pytest.skip("Set RUN_MULTICHIP_DECODER_WATCHER=1 under TT_METAL_WATCHER")

    batch = 32
    seq_len = 7
    config, state, multichip = _build_multichip(mesh_device, batch=batch)
    table_host = torch.arange(2 * batch, dtype=torch.int32).reshape(batch, 2).flip(1)
    table = _page_table(table_host, mesh_device)
    key_cache, value_cache = _mesh_paged_caches(config, mesh_device, num_blocks=2 * batch, dtype=ttnn.bfloat8_b)
    generator = torch.Generator().manual_seed(911)
    prefill = _mesh_input(
        torch.randn((1, batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16),
        mesh_device,
    )
    hidden = _mesh_input(
        torch.randn((1, batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16),
        mesh_device,
    )
    prefill_output = multichip.prefill_forward(prefill, key_cache, value_cache, page_table=table)
    eager = multichip.decode_forward(hidden, key_cache, value_cache, current_pos=seq_len, page_table=table)
    ttnn.synchronize_device(mesh_device)
    eager_host = _replicated_host(eager)
    assert tuple(eager.shape) == (1, batch, 1, config.hidden_size)
    assert tuple(key_cache.shape) == (2 * batch, config.num_key_value_heads // 2, PAGED_BLOCK_SIZE, 128)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = multichip.decode_forward(hidden, key_cache, value_cache, current_pos=seq_len, page_table=table)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        first = None
        for replay in range(10):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            output = _replicated_host(traced)
            _assert_pcc(eager_host, output, 0.999, f"watcher trace replay={replay}")
            if first is None:
                first = output
            else:
                assert torch.equal(first, output)
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    for tensor in (prefill_output, eager, traced, key_cache, value_cache):
        ttnn.deallocate(tensor)
    ttnn.synchronize_device(mesh_device)
    del multichip, state
    gc.collect()


@_mesh_test
def test_multichip_fractured_residual_topology_probe(mesh_device):
    """Measure an exact add/norm/projection boundary with a fractured residual."""

    if os.environ.get("RUN_MULTICHIP_DECODER_TOPOLOGY_PROBE") != "1":
        pytest.skip("Set RUN_MULTICHIP_DECODER_TOPOLOGY_PROBE=1 for the topology probe")

    batch = 32
    config, state, multichip = _build_multichip(mesh_device, batch=batch)
    device_grid = mesh_device.compute_with_storage_grid_size()

    def width_l1(width: int):
        return ttnn.create_sharded_memory_config(
            (batch, width // 32),
            ttnn.CoreGrid(x=8, y=4),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    assert (device_grid.x, device_grid.y) == (11, 10)
    full_residual_mem = multichip.residual_mem_config
    local_residual_mem = width_l1(config.hidden_size // TARGET_TP_DEGREE)
    ag_output_mem = ttnn.create_sharded_memory_config(
        (batch, config.hidden_size // TARGET_TP_DEGREE),
        ttnn.CoreGrid(x=2, y=1),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    qkv_width = (config.num_attention_heads // 2 + config.num_key_value_heads) * 128
    qkv_output_mem = width_l1(qkv_width)
    stats_mem = ttnn.create_sharded_memory_config(
        (batch, 2 * ttnn.TILE_SIZE),
        ttnn.CoreGrid(x=1, y=1),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    distributed_norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        subblock_w=2,
        block_h=1,
        block_w=2,
        inplace=False,
    )
    qkv_agmm_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=1,
        per_core_N=3,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    generator = torch.Generator().manual_seed(20260717)
    partial_host = torch.randn((1, 1, batch, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    residual_host = torch.randn((1, 1, batch, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    partial = ttnn.to_memory_config(_mesh_input(partial_host, mesh_device), full_residual_mem)
    partial_interleaved = ttnn.sharded_to_interleaved(partial, ttnn.L1_MEMORY_CONFIG)
    replicated_residual = ttnn.to_memory_config(_mesh_input(residual_host, mesh_device), full_residual_mem)
    local_residual = ttnn.from_torch(
        residual_host,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    local_residual = ttnn.to_memory_config(local_residual, local_residual_mem)
    persistent_ag = ttnn.to_memory_config(_mesh_input(torch.zeros_like(partial_host), mesh_device), ag_output_mem)
    norm_key = f"model.layers.{LAYER_IDX}.input_layernorm.weight"
    local_gamma = ttnn.from_torch(
        state[norm_key].reshape(1, 1, config.hidden_size // ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    # Fused AGMM requires DRAM-interleaved BFP4 weights; this is a setup-time
    # alternate placement of the exact local QKV shards used by the decoder.
    qkv_interleaved = ttnn.to_memory_config(
        ttnn.reshape(multichip.qkv_weight, [1, 1, config.hidden_size, qkv_width]),
        ttnn.DRAM_MEMORY_CONFIG,
    )

    def current_boundary():
        gathered = multichip._all_reduce_partial(partial, memory_config=full_residual_mem)
        added = ttnn.add(
            replicated_residual,
            gathered,
            dtype=ttnn.bfloat16,
            memory_config=full_residual_mem,
        )
        normed = ttnn.rms_norm(
            added,
            epsilon=multichip.rms_norm_eps,
            weight=multichip.input_norm,
            program_config=multichip.norm_program_config,
            memory_config=full_residual_mem,
            compute_kernel_config=multichip.norm_compute_kernel,
        )
        normed = ttnn.to_memory_config(normed, multichip.qkv_input_mem_config)
        return ttnn.linear(
            normed,
            multichip.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=multichip.qkv_decode_program_config,
            compute_kernel_config=multichip.attention_compute_kernel,
        )

    def fractured_boundary():
        local_partial = ttnn.experimental.reduce_scatter_minimal_async(
            partial_interleaved,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=multichip.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=multichip.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=2,
            memory_config=local_residual_mem,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        local_added = ttnn.add(
            local_residual,
            local_partial,
            dtype=ttnn.bfloat16,
            memory_config=local_residual_mem,
        )
        local_stats = ttnn.rms_norm_pre_all_gather(
            local_added,
            compute_kernel_config=multichip.norm_compute_kernel,
            program_config=distributed_norm_program,
            dtype=ttnn.bfloat16,
        )
        gathered_stats = ttnn.experimental.all_gather_async(
            local_stats,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=multichip.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=multichip.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=2,
            memory_config=stats_mem,
            topology=ttnn.Topology.Ring,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        local_normed = ttnn.rms_norm_post_all_gather(
            local_added,
            gathered_stats,
            epsilon=multichip.rms_norm_eps,
            weight=local_gamma,
            compute_kernel_config=multichip.norm_compute_kernel,
            program_config=distributed_norm_program,
            dtype=ttnn.bfloat16,
            memory_config=local_residual_mem,
        )
        gathered = ttnn.experimental.all_gather_async(
            local_normed,
            persistent_output_buffer=persistent_ag,
            dim=3,
            multi_device_global_semaphore=multichip.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=multichip.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=2,
            memory_config=ag_output_mem,
            topology=ttnn.Topology.Ring,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        return ttnn.linear(
            gathered,
            qkv_interleaved,
            dtype=ttnn.bfloat16,
            memory_config=qkv_output_mem,
            program_config=qkv_agmm_program,
            compute_kernel_config=multichip.attention_compute_kernel,
        )

    current_ms, current_output = _trace_latency(mesh_device, current_boundary, 1000)
    fractured_ms, fractured_output = _trace_latency(mesh_device, fractured_boundary, 1000)
    for rank, (expected, actual) in enumerate(zip(_mesh_hosts(current_output), _mesh_hosts(fractured_output))):
        _assert_pcc(expected, actual, 0.99, f"fractured residual QKV rank={rank}")
    print(
        "MULTICHIP_TOPOLOGY_PROBE "
        f"current_allreduce_add_norm_matmul_ms={current_ms:.6f} "
        f"fractured_rs_add_distributed_norm_ag_matmul_ms={fractured_ms:.6f} "
        f"candidate_over_current={fractured_ms / current_ms:.6f} "
        "current_bytes=262144 candidate_bytes=262144 weight_dtype=bfloat4_b "
        "fused_agmm=unsupported_linear_p300_topology"
    )

    del multichip, state
    gc.collect()


def _variant_config() -> MultiChipConfig:
    variant = os.environ.get("MULTICHIP_DECODER_VARIANT", "default")
    optimized = OptimizationConfig(decode_matmul_strategy="dram_sharded", output_cores=8)
    collective_dtype = ttnn.bfloat16
    num_links = 2
    if variant == "geometry16":
        optimized = optimized.with_changes(
            qkv_cores=16, output_cores=16, gate_up_cores=16, down_cores=16, residual_cores=16
        )
    elif variant == "output16":
        optimized = optimized.with_changes(output_cores=16)
    elif variant == "output32":
        optimized = optimized.with_changes(output_cores=32)
    elif variant == "output8":
        optimized = optimized.with_changes(output_cores=8)
    elif variant == "ccl_bfp8":
        collective_dtype = ttnn.bfloat8_b
    elif variant == "packed_gate_up":
        optimized = optimized.with_changes(packed_gate_up=True)
    elif variant == "link1":
        num_links = 1
    elif variant == "output8_link1":
        optimized = optimized.with_changes(output_cores=8)
        num_links = 1
    elif variant == "output8_ccl_bfp8":
        optimized = optimized.with_changes(output_cores=8)
        collective_dtype = ttnn.bfloat8_b
    elif variant != "default":
        raise ValueError(f"Unknown MULTICHIP_DECODER_VARIANT={variant!r}")
    return MultiChipConfig(optimized=optimized, collective_dtype=collective_dtype, num_links=num_links)


def _trace_latency(mesh_device, forward, replay_count: int):
    forward()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = forward()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        start = time.perf_counter()
        for _ in range(replay_count):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - start) * 1000.0 / replay_count, output
    finally:
        ttnn.release_trace(mesh_device, trace_id)


_PERF_REFERENCE = None


@_single_test
def test_single_chip_warmed_perf_reference(mesh_device):
    if os.environ.get("RUN_MULTICHIP_DECODER_PERF") != "1":
        pytest.skip("Set RUN_MULTICHIP_DECODER_PERF=1 for warmed baseline/multichip timing")
    batch = int(os.environ.get("MULTICHIP_DECODER_BATCH", "1"))
    seq_len = int(os.environ.get("MULTICHIP_DECODER_SEQ_LEN", "18"))
    prefill_repeats = int(os.environ.get("MULTICHIP_DECODER_PREFILL_REPEATS", "10"))
    trace_replays = int(os.environ.get("MULTICHIP_DECODER_TRACE_REPLAYS", "100"))
    variant = os.environ.get("MULTICHIP_DECODER_VARIANT", "default")
    config, state, baseline = _build_baseline(mesh_device, batch=batch, multichip_config=_variant_config())
    single_key, single_value = _single_caches(config, mesh_device, batch=batch, dtype=ttnn.bfloat8_b)
    generator = torch.Generator().manual_seed(2026)
    prefill_host = torch.randn((1, batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    decode_host = torch.randn((1, batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    single_prefill = _single_input(prefill_host, mesh_device)
    single_decode = _single_input(decode_host, mesh_device)

    baseline.prefill_forward(single_prefill, single_key, single_value)
    ttnn.synchronize_device(mesh_device)
    signpost(header="PERF_SINGLE_PREFILL")
    start = time.perf_counter()
    for _ in range(prefill_repeats):
        baseline.prefill_forward(single_prefill, single_key, single_value)
    ttnn.synchronize_device(mesh_device)
    single_prefill_ms = (time.perf_counter() - start) * 1000.0 / prefill_repeats
    signpost(header="PERF_SINGLE_PREFILL_END")

    signpost(header="PERF_SINGLE_DECODE")
    single_decode_ms, single_output = _trace_latency(
        mesh_device,
        lambda: baseline.decode_forward(single_decode, single_key, single_value, current_pos=seq_len),
        trace_replays,
    )
    signpost(header="PERF_SINGLE_DECODE_END")

    global _PERF_REFERENCE
    _PERF_REFERENCE = {
        "variant": variant,
        "batch": batch,
        "seq_len": seq_len,
        "prefill_ms": single_prefill_ms,
        "decode_ms": single_decode_ms,
        "output": _single_host(single_output),
    }
    print(
        "SINGLE_CHIP_PERF_REFERENCE "
        f"variant={variant} batch={batch} seq={seq_len} "
        f"single_prefill_ms={single_prefill_ms:.6f} single_decode_ms={single_decode_ms:.6f} "
        f"prefill_repeats={prefill_repeats} trace_replays={trace_replays}"
    )
    del baseline, state
    gc.collect()


@_mesh_test
def test_single_and_multichip_warmed_perf(mesh_device):
    if os.environ.get("RUN_MULTICHIP_DECODER_PERF") != "1":
        pytest.skip("Set RUN_MULTICHIP_DECODER_PERF=1 for warmed baseline/multichip timing")
    assert _PERF_REFERENCE is not None, "run the single-chip perf reference item with this target item"
    reference = _PERF_REFERENCE
    batch = int(os.environ.get("MULTICHIP_DECODER_BATCH", "1"))
    seq_len = int(os.environ.get("MULTICHIP_DECODER_SEQ_LEN", "18"))
    prefill_repeats = int(os.environ.get("MULTICHIP_DECODER_PREFILL_REPEATS", "10"))
    trace_replays = int(os.environ.get("MULTICHIP_DECODER_TRACE_REPLAYS", "100"))
    variant = os.environ.get("MULTICHIP_DECODER_VARIANT", "default")
    assert (reference["variant"], reference["batch"], reference["seq_len"]) == (variant, batch, seq_len)
    config, state, multichip = _build_multichip(mesh_device, batch=batch, multichip_config=_variant_config())
    mesh_key, mesh_value = _mesh_contiguous_caches(config, mesh_device, batch=batch, dtype=ttnn.bfloat8_b)
    generator = torch.Generator().manual_seed(2026)
    prefill_host = torch.randn((1, batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    decode_host = torch.randn((1, batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    mesh_prefill = _mesh_input(prefill_host, mesh_device)
    mesh_decode = _mesh_input(decode_host, mesh_device)

    multichip.prefill_forward(mesh_prefill, mesh_key, mesh_value)
    ttnn.synchronize_device(mesh_device)
    signpost(header="PERF_MULTI_PREFILL")
    start = time.perf_counter()
    for _ in range(prefill_repeats):
        multichip.prefill_forward(mesh_prefill, mesh_key, mesh_value)
    ttnn.synchronize_device(mesh_device)
    multi_prefill_ms = (time.perf_counter() - start) * 1000.0 / prefill_repeats
    signpost(header="PERF_MULTI_PREFILL_END")

    signpost(header="PERF_MULTI_DECODE")
    multi_decode_ms, multi_output = _trace_latency(
        mesh_device,
        lambda: multichip.decode_forward(mesh_decode, mesh_key, mesh_value, current_pos=seq_len),
        trace_replays,
    )
    signpost(header="PERF_MULTI_DECODE_END")
    _assert_pcc(reference["output"], _replicated_host(multi_output), 0.99, f"perf {variant} output")

    single_prefill_ms = reference["prefill_ms"]
    single_decode_ms = reference["decode_ms"]
    prefill_speedup = single_prefill_ms / multi_prefill_ms
    decode_speedup = single_decode_ms / multi_decode_ms
    print(
        "MULTICHIP_PERF_RESULT "
        f"variant={variant} batch={batch} seq={seq_len} "
        f"single_prefill_ms={single_prefill_ms:.6f} multi_prefill_ms={multi_prefill_ms:.6f} "
        f"prefill_speedup={prefill_speedup:.6f} prefill_efficiency={prefill_speedup / 2:.6f} "
        f"single_decode_ms={single_decode_ms:.6f} multi_decode_ms={multi_decode_ms:.6f} "
        f"decode_speedup={decode_speedup:.6f} decode_efficiency={decode_speedup / 2:.6f} "
        f"prefill_repeats={prefill_repeats} trace_replays={trace_replays}"
    )

    del multichip, state
    gc.collect()
