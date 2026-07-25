# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.openai_gpt_oss_20b.tests.test_functional_decoder import (
    EMITTED_PREFILL_SEQUENCE,
    LAYER_IDX,
    _assert_pcc,
    _config,
    _position_tensor,
    _real_state_dict,
    _synthetic_state_dict,
    _to_torch,
    _to_tt,
)
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import _require_tensor
from models.autoports.openai_gpt_oss_20b.tt.multichip_decoder import (
    DECODE_COLLECTIVE_ALL_REDUCE,
    DECODE_COLLECTIVE_RS_AG_PAD64,
    EP_DEGREE,
    EXPERT_STRATEGY_EP,
    EXPERT_STRATEGY_TP,
    PAGE_BLOCK_SIZE,
    SUPPORTED_CONTEXT,
    TARGET_MESH_SHAPE,
    TP_DEGREE,
    MultichipConfig,
    MultichipDecoder,
    _validate_ep_prefill_geometry,
    _validate_qkv_geometry,
)
from models.autoports.openai_gpt_oss_20b.tt.optimized_decoder import OptimizedDecoder
from models.common.lightweightmodule import LightweightModule

FABRIC_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "trace_region_size": 64 * 1024 * 1024,
}
SINGLE_CHIP_DEVICE_PARAMS = {"trace_region_size": 64 * 1024 * 1024}
EVIDENCE_DIR = Path(__file__).parents[1] / "doc" / "multichip_decoder" / "logs"
SYNTHETIC_REFERENCE_PATH = EVIDENCE_DIR / "optimized_reference_synthetic.pt"
REAL_PCC_SEEDS = {"sliding_attention": 10010, "full_attention": 20260717}
NEAR_TIED_ROUTER_SEED = 20260718


def _perf_multichip_config_from_env():
    """Build measured candidates without changing the production default."""

    strategy = os.environ.get("MULTICHIP_EXPERT_STRATEGY", MultichipConfig().expert_strategy)
    qkv_ab = os.environ.get("MULTICHIP_QKV_AB")
    kwargs = {
        "expert_strategy": strategy,
        "decode_collective": os.environ.get("MULTICHIP_DECODE_COLLECTIVE_AB", MultichipConfig().decode_collective),
    }
    qkv_candidate = None
    if qkv_ab is not None:
        parts = qkv_ab.split(",")
        if len(parts) != 4:
            raise ValueError("MULTICHIP_QKV_AB must be input_cores,in0_block_w,output_tiles_per_core,out_subblock_w")
        try:
            qkv_candidate = tuple(int(part) for part in parts)
        except ValueError as error:
            raise ValueError("MULTICHIP_QKV_AB fields must be integers") from error
        (
            kwargs["qkv_input_cores"],
            kwargs["qkv_in0_block_w"],
            kwargs["qkv_output_tiles_per_core"],
            kwargs["qkv_out_subblock_w"],
        ) = qkv_candidate
    ep_prefill_ab = os.environ.get("MULTICHIP_EP_PREFILL_AB")
    if ep_prefill_ab is not None:
        parts = ep_prefill_ab.split(",")
        if len(parts) != 7:
            raise ValueError(
                "MULTICHIP_EP_PREFILL_AB must be gate_x,gate_y,gate_subblock," "down_x,down_y,down_subblock,chunk"
            )
        try:
            gate_x, gate_y, gate_subblock, down_x, down_y, down_subblock, chunk = (int(part) for part in parts)
        except ValueError as error:
            raise ValueError("MULTICHIP_EP_PREFILL_AB fields must be integers") from error
        kwargs.update(
            ep_prefill_gate_up_cores=(gate_x, gate_y),
            ep_prefill_gate_up_subblock_w=gate_subblock,
            ep_prefill_down_cores=(down_x, down_y),
            ep_prefill_down_subblock_w=down_subblock,
            active_prefill_chunk_size=chunk,
        )
    default_rewrite = "post_sparse_bf16" if MultichipConfig().ep_prefill_post_sparse_bf16 else "baseline"
    rewrite = os.environ.get("MULTICHIP_EP_PREFILL_REWRITE", default_rewrite)
    if rewrite not in ("baseline", "post_sparse_bf16"):
        raise ValueError("MULTICHIP_EP_PREFILL_REWRITE must be baseline or post_sparse_bf16")
    kwargs["ep_prefill_post_sparse_bf16"] = rewrite == "post_sparse_bf16"
    multichip_config = MultichipConfig(**kwargs)
    if qkv_candidate is not None:
        _validate_qkv_geometry(multichip_config, k_tiles=90, n_tiles=40, grid_x=11, grid_y=10)
    _validate_ep_prefill_geometry(multichip_config)
    return multichip_config, qkv_candidate


def _mesh_test(function):
    function = pytest.mark.parametrize("mesh_device", [TARGET_MESH_SHAPE], indirect=True)(function)
    return pytest.mark.parametrize("device_params", [FABRIC_DEVICE_PARAMS], indirect=True)(function)


def _single_chip_test(function):
    function = pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)(function)
    return pytest.mark.parametrize("device_params", [SINGLE_CHIP_DEVICE_PARAMS], indirect=True)(function)


def _decoder(
    state,
    config,
    mesh_device,
    *,
    max_cache_len=128,
    expert_strategy=None,
    multichip_config=None,
):
    if os.environ.get("MULTICHIP_TEST_CONFIG_FROM_ENV") == "1":
        env_config, _ = _perf_multichip_config_from_env()
        multichip_config = replace(
            env_config,
            expert_strategy=expert_strategy or env_config.expert_strategy,
        )
    return MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=max_cache_len,
        multichip_config=multichip_config
        or (MultichipConfig() if expert_strategy is None else MultichipConfig(expert_strategy=expert_strategy)),
    )


def _all_device_torch(tensor):
    return [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]


def _assert_replicated(name, tensor):
    locals_ = _all_device_torch(tensor)
    assert len(locals_) == TP_DEGREE
    for rank, local in enumerate(locals_[1:], start=1):
        assert torch.equal(locals_[0], local), f"{name} differs on rank {rank}"


def _host_hidden(hidden_states, mesh_device):
    return ttnn.from_torch(
        hidden_states.unsqueeze(0),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _host_position(position, mesh_device):
    return ttnn.from_torch(
        torch.tensor([position], dtype=torch.int32),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _host_page_table(physical_block_ids, mesh_device):
    return ttnn.from_torch(
        torch.tensor(physical_block_ids, dtype=torch.int32).reshape(1, -1),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def test_multichip_runtime_contract_and_fallback_audit():
    assert issubclass(MultichipDecoder, (OptimizedDecoder, LightweightModule))
    assert TARGET_MESH_SHAPE == (1, 4)
    assert TP_DEGREE == EP_DEGREE == 4
    assert PAGE_BLOCK_SIZE == 64
    assert SUPPORTED_CONTEXT == 131_072

    selected = MultichipConfig()
    assert selected.expert_strategy == EXPERT_STRATEGY_EP
    assert _validate_qkv_geometry(selected, k_tiles=90, n_tiles=40, grid_x=11, grid_y=10) == (9, 20, 2)
    assert selected.ep_prefill_gate_up_cores == selected.ep_prefill_down_cores == (9, 10)
    assert selected.ep_prefill_gate_up_subblock_w == selected.ep_prefill_down_subblock_w == 1
    assert selected.ep_prefill_post_sparse_bf16
    assert _validate_ep_prefill_geometry(selected) == (1, 1)
    assert selected.decode_collective == DECODE_COLLECTIVE_ALL_REDUCE

    runtime_methods = (
        MultichipDecoder._prefill_attention,
        MultichipDecoder._decode_attention,
        MultichipDecoder._decode_sliding_attention_mask,
        MultichipDecoder._decode_post_attention_norm,
        MultichipDecoder._active_prefill_expert_chunk,
        MultichipDecoder._active_prefill_sparse_moe,
        MultichipDecoder._ep_active_expert_chunk,
        MultichipDecoder._moe_forward,
        MultichipDecoder._all_reduce,
        MultichipDecoder.prefill_forward,
        MultichipDecoder.decode_forward,
        MultichipDecoder.forward,
    )
    forbidden = ("torch", "from_torch", "to_torch", "get_device_tensors", "cpu")
    for method in runtime_methods:
        source = inspect.getsource(method)
        assert all(token not in source for token in forbidden), method.__name__

    ep_source = inspect.getsource(MultichipDecoder._ep_active_expert_chunk)
    assert ep_source.count("ttnn.sparse_matmul(") == 3
    assert ep_source.count("nnz=None") == 3
    assert "ttnn.mesh_partition(" in ep_source
    assert "is_input_a_sparse=True" in ep_source
    assert "is_input_b_sparse=False" in ep_source


def test_multichip_perf_candidate_parsing(monkeypatch, expect_error):
    monkeypatch.setenv("MULTICHIP_EXPERT_STRATEGY", EXPERT_STRATEGY_EP)
    monkeypatch.setenv("MULTICHIP_QKV_AB", "30,3,1,1")
    config, candidate = _perf_multichip_config_from_env()
    assert config.expert_strategy == EXPERT_STRATEGY_EP
    assert candidate == (30, 3, 1, 1)
    assert config.ep_prefill_post_sparse_bf16
    assert _validate_qkv_geometry(config, k_tiles=90, n_tiles=40, grid_x=11, grid_y=10) == (3, 40, 4)

    monkeypatch.setenv("MULTICHIP_DECODE_COLLECTIVE_AB", DECODE_COLLECTIVE_RS_AG_PAD64)
    monkeypatch.setenv("MULTICHIP_EP_PREFILL_AB", "5,6,3,9,10,1,64")
    monkeypatch.setenv("MULTICHIP_EP_PREFILL_REWRITE", "baseline")
    config, _ = _perf_multichip_config_from_env()
    assert config.decode_collective == DECODE_COLLECTIVE_RS_AG_PAD64
    assert config.ep_prefill_gate_up_cores == (5, 6)
    assert config.ep_prefill_down_cores == (9, 10)
    assert config.active_prefill_chunk_size == 64
    assert not config.ep_prefill_post_sparse_bf16
    assert _validate_ep_prefill_geometry(config) == (3, 1)

    monkeypatch.setenv("MULTICHIP_QKV_AB", "30,3,3")
    with expect_error(ValueError, "must be input_cores"):
        _perf_multichip_config_from_env()

    monkeypatch.setenv("MULTICHIP_QKV_AB", "30,3,2,2")
    monkeypatch.setenv("MULTICHIP_EP_PREFILL_AB", "5,6,2,5,6,3,128")
    with expect_error(ValueError, "must divide per_core_N"):
        _perf_multichip_config_from_env()


@_mesh_test
@pytest.mark.parametrize("expert_strategy", [EXPERT_STRATEGY_TP, EXPERT_STRATEGY_EP])
def test_active_prefill_uses_exactly_four_routes_per_token(mesh_device, expert_strategy, monkeypatch):
    """Record all three sparse products for controlled, non-aligned top-4 routes."""

    config = _config()
    decoder = _decoder(
        _synthetic_state_dict(config),
        config,
        mesh_device,
        max_cache_len=64,
        expert_strategy=expert_strategy,
    )
    seq_len = 17
    hidden = torch.randn(1, seq_len, config.hidden_size, generator=torch.Generator().manual_seed(1704)).to(
        torch.bfloat16
    )
    routing = torch.zeros(seq_len, config.num_local_experts, dtype=torch.bfloat16)
    scales = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.bfloat16)
    for token in range(seq_len):
        selected = torch.tensor(
            [token % 32, (token + 7) % 32, (token + 15) % 32, (token + 23) % 32],
            dtype=torch.long,
        )
        routing[token, selected] = scales

    tt_hidden = _to_tt(hidden, mesh_device)
    tt_routing = ttnn.from_torch(
        routing,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    calls = []
    original_sparse_matmul = ttnn.sparse_matmul

    def recording_sparse_matmul(input_a, input_b, **kwargs):
        per_rank_active = [int(torch.count_nonzero(local)) for local in _all_device_torch(kwargs["sparsity"])]
        calls.append(
            {
                "per_rank_active": per_rank_active,
                "nnz": kwargs.get("nnz"),
                "input_a_sparse": kwargs.get("is_input_a_sparse", False),
                "input_b_sparse": kwargs.get("is_input_b_sparse", True),
            }
        )
        return original_sparse_matmul(input_a, input_b, **kwargs)

    monkeypatch.setattr(ttnn, "sparse_matmul", recording_sparse_matmul)
    normalized = tt_hidden
    if expert_strategy == EXPERT_STRATEGY_TP:
        output = decoder._active_prefill_sparse_moe(tt_hidden, normalized, tt_routing, seq_len)
    else:
        output = decoder._active_prefill_sparse_moe(tt_hidden, normalized, tt_routing, seq_len)
    ttnn.synchronize_device(mesh_device)

    assert tuple(output.shape) == (1, 1, seq_len, config.hidden_size)
    _assert_replicated(f"{expert_strategy}-controlled-top4", output)
    assert len(calls) == 3
    if expert_strategy == EXPERT_STRATEGY_TP:
        assert [call["per_rank_active"] for call in calls] == [[4 * seq_len] * TP_DEGREE] * 3
    else:
        assert [sum(call["per_rank_active"]) for call in calls] == [4 * seq_len] * 3
    assert [call["nnz"] for call in calls] == [None, None, None]
    assert [call["input_a_sparse"] for call in calls] == [False, False, True]
    assert [call["input_b_sparse"] for call in calls] == [True, True, False]
    assert 4 * seq_len != math.ceil(seq_len / ttnn.TILE_SIZE) * config.num_local_experts


@_single_chip_test
def test_capture_synthetic_single_chip_optimized_reference(mesh_device):
    """Capture the optimized baseline without overlapping a parent mesh."""

    config = _config()
    state = _synthetic_state_dict(config)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=128,
    )
    key_cache, value_cache = decoder.create_kv_cache()
    generator = torch.Generator().manual_seed(20260718)
    reference = {}
    for seq_len in (EMITTED_PREFILL_SEQUENCE, 33):
        hidden = torch.randn(1, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16)
        output = decoder.prefill_forward(_to_tt(hidden, mesh_device), key_cache=key_cache, value_cache=value_cache)
        ttnn.synchronize_device(mesh_device)
        reference[f"prefill_{seq_len}"] = _to_torch(output).cpu()
    reference["key_cache"] = ttnn.to_torch(ttnn.get_device_tensors(key_cache)[0]).cpu()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(reference, SYNTHETIC_REFERENCE_PATH)
    print(f"OPTIMIZED_REFERENCE_ARTIFACT {SYNTHETIC_REFERENCE_PATH}")


@_mesh_test
@pytest.mark.parametrize("expert_strategy", [EXPERT_STRATEGY_TP, EXPERT_STRATEGY_EP])
def test_synthetic_non_aligned_prefill_matches_optimized_and_cache_is_head_local(mesh_device, expert_strategy):
    assert SYNTHETIC_REFERENCE_PATH.exists(), "run the isolated optimized baseline capture first"
    reference = torch.load(SYNTHETIC_REFERENCE_PATH, map_location="cpu", weights_only=True)
    config = _config()
    decoder = _decoder(
        _synthetic_state_dict(config),
        config,
        mesh_device,
        max_cache_len=128,
        expert_strategy=expert_strategy,
    )
    assert decoder.local_num_heads == 16
    assert decoder.local_num_kv_heads == 2
    assert decoder.local_intermediate_size == 720
    assert decoder.local_num_experts == 8
    qkv_grid = decoder.tp_qkv_program_config.compute_with_storage_grid_size
    assert (qkv_grid.x, qkv_grid.y) == (11, 2)
    assert tuple(decoder.tp_qkv_input_config.shard_spec.shape) == (32, 288)
    assert tuple(decoder.tp_qkv_output_config.shard_spec.shape) == (32, 64)
    local_qkv = _all_device_torch(decoder.weights["qkv_weight"])
    assert [tuple(weight.shape) for weight in local_qkv] == [(2880, 1280)] * TP_DEGREE

    reverse_blocks = list(reversed(range(decoder.num_cache_blocks)))
    page_table = decoder.create_page_table(reverse_blocks)
    key_cache, value_cache = decoder.create_kv_cache()
    generator = torch.Generator().manual_seed(20260718)
    for seq_len in (EMITTED_PREFILL_SEQUENCE, 33):
        hidden = torch.randn(1, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16)
        output = decoder.prefill_forward(
            _to_tt(hidden, mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
        )
        _assert_pcc(
            f"{expert_strategy}-synthetic-prefill-{seq_len}",
            reference[f"prefill_{seq_len}"],
            _to_torch(output),
            0.99,
        )
        assert tuple(output.shape) == (1, 1, seq_len, config.hidden_size)
        _assert_replicated(f"{expert_strategy}-synthetic-prefill-{seq_len}", output)

    baseline_cache = reference["key_cache"]
    physical_page = reverse_blocks[0]
    for rank, local_cache in enumerate(_all_device_torch(key_cache)):
        reference_slice = baseline_cache[0, rank * 2 : (rank + 1) * 2, :PAGE_BLOCK_SIZE]
        _assert_pcc(
            f"{expert_strategy}-rank-{rank}-logical-page-0-local-kv-heads",
            reference_slice,
            local_cache[physical_page],
            0.999,
        )


def _real_reference_path(layer_type):
    return EVIDENCE_DIR / f"optimized_reference_{layer_type}.pt"


def _near_tied_reference_path():
    return EVIDENCE_DIR / f"optimized_reference_full_attention_seed{NEAR_TIED_ROUTER_SEED}.pt"


@_single_chip_test
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_capture_real_weight_single_chip_optimized_reference(mesh_device, layer_type):
    config = _config()
    config.layer_types = list(config.layer_types)
    config.layer_types[LAYER_IDX] = layer_type
    decoder = OptimizedDecoder.from_state_dict(
        _real_state_dict(),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=256,
    )
    key_cache, value_cache = decoder.create_kv_cache()
    generator = torch.Generator().manual_seed(REAL_PCC_SEEDS[layer_type])
    prefill_len = config.sliding_window
    prefill_hidden = torch.randn(1, prefill_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    prefill = decoder.prefill_forward(_to_tt(prefill_hidden, mesh_device), key_cache=key_cache, value_cache=value_cache)
    ttnn.synchronize_device(mesh_device)
    reference = {"prefill_hidden": prefill_hidden.cpu(), "prefill": _to_torch(prefill).cpu()}
    for position in range(prefill_len, prefill_len + 3):
        hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
        attention = decoder._decode_attention(
            _to_tt(hidden, mesh_device), key_cache, value_cache, position, _position_tensor(position, mesh_device)
        )
        normalized = ttnn.rms_norm(
            attention,
            epsilon=decoder.eps,
            weight=decoder.weights["post_attention_norm"],
            compute_kernel_config=decoder.compute_kernel_config,
        )
        routing = decoder._route(normalized, 1)
        output = decoder._sparse_moe_forward(attention, normalized, routing, 1)
        ttnn.synchronize_device(mesh_device)
        reference[f"hidden_{position}"] = hidden.cpu()
        reference[f"attention_{position}"] = _to_torch(attention).cpu()
        reference[f"routing_{position}"] = _to_torch(routing).cpu()
        reference[f"decode_{position}"] = _to_torch(output).cpu()
    path = _real_reference_path(layer_type)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(reference, path)
    print(f"OPTIMIZED_REFERENCE_ARTIFACT {path}")


@_single_chip_test
def test_capture_near_tied_single_chip_optimized_reference(mesh_device):
    """Capture the seed that deliberately straddles the fourth/fifth-expert boundary."""

    config = _config()
    config.layer_types = list(config.layer_types)
    config.layer_types[LAYER_IDX] = "full_attention"
    decoder = OptimizedDecoder.from_state_dict(
        _real_state_dict(),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=256,
    )
    key_cache, value_cache = decoder.create_kv_cache()
    generator = torch.Generator().manual_seed(NEAR_TIED_ROUTER_SEED)
    prefill_len = config.sliding_window
    prefill_hidden = torch.randn(1, prefill_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    decoder.prefill_forward(_to_tt(prefill_hidden, mesh_device), key_cache=key_cache, value_cache=value_cache)
    reference = {"prefill_hidden": prefill_hidden.cpu()}
    for position in range(prefill_len, prefill_len + 2):
        hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
        attention = decoder._decode_attention(
            _to_tt(hidden, mesh_device), key_cache, value_cache, position, _position_tensor(position, mesh_device)
        )
        normalized = ttnn.rms_norm(
            attention,
            epsilon=decoder.eps,
            weight=decoder.weights["post_attention_norm"],
            compute_kernel_config=decoder.compute_kernel_config,
        )
        routing = decoder._route(normalized, 1)
        output = decoder._sparse_moe_forward(attention, normalized, routing, 1)
        ttnn.synchronize_device(mesh_device)
        reference[f"hidden_{position}"] = hidden.cpu()
        reference[f"attention_{position}"] = _to_torch(attention).cpu()
        reference[f"routing_{position}"] = _to_torch(routing).cpu()
        reference[f"decode_{position}"] = _to_torch(output).cpu()
    path = _near_tied_reference_path()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(reference, path)
    print(f"NEAR_TIED_OPTIMIZED_REFERENCE_ARTIFACT {path}")


@_mesh_test
@pytest.mark.parametrize("expert_strategy", [EXPERT_STRATEGY_TP, EXPERT_STRATEGY_EP])
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_real_weight_prefill_decode_matches_single_chip_optimized(mesh_device, expert_strategy, layer_type):
    reference_path = _real_reference_path(layer_type)
    assert reference_path.exists(), "run the isolated real-weight optimized baseline capture first"
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    config = _config()
    config.layer_types = list(config.layer_types)
    config.layer_types[LAYER_IDX] = layer_type
    decoder = _decoder(_real_state_dict(), config, mesh_device, max_cache_len=256, expert_strategy=expert_strategy)
    page_table = decoder.create_page_table(list(reversed(range(decoder.num_cache_blocks))))
    key_cache, value_cache = decoder.create_kv_cache()
    prefill_len = config.sliding_window
    prefill_hidden = reference["prefill_hidden"]
    output = decoder.prefill_forward(
        _to_tt(prefill_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    _assert_pcc(f"{expert_strategy}-{layer_type}-prefill", reference["prefill"], _to_torch(output), 0.99)
    _assert_replicated(f"{expert_strategy}-{layer_type}-prefill", output)

    last_output = None
    for position in range(prefill_len, prefill_len + 3):
        hidden = reference[f"hidden_{position}"]
        attention = decoder._decode_attention(
            _to_tt(hidden, mesh_device),
            key_cache,
            value_cache,
            page_table,
            position,
            _position_tensor(position, mesh_device),
        )
        _assert_pcc(
            f"{expert_strategy}-{layer_type}-attention-{position}",
            reference[f"attention_{position}"],
            _to_torch(attention),
            0.99,
        )
        normalized = decoder._decode_post_attention_norm(attention)
        routing = decoder._route(normalized, 1)
        if expert_strategy == EXPERT_STRATEGY_TP:
            last_output = decoder._sparse_moe_forward(attention, normalized, routing, 1)
        else:
            partial = decoder._ep_active_expert_chunk(normalized, routing, is_decode=True)
            last_output = ttnn.add(attention, decoder._all_reduce(partial, memory_config=ttnn.L1_MEMORY_CONFIG))
        _assert_pcc(
            f"{expert_strategy}-{layer_type}-routing-{position}",
            reference[f"routing_{position}"],
            _to_torch(routing),
            0.99,
        )
        _assert_pcc(
            f"{expert_strategy}-{layer_type}-decode-{position}",
            reference[f"decode_{position}"],
            _to_torch(last_output),
            0.99,
        )
        _assert_replicated(f"{expert_strategy}-{layer_type}-decode-{position}", last_output)

    second_key, second_value = decoder.create_kv_cache()
    stacked = decoder.decode_forward(
        last_output,
        key_cache=second_key,
        value_cache=second_value,
        page_table=page_table,
        cache_position=0,
        cache_position_tensor=_position_tensor(0, mesh_device),
    )
    assert tuple(stacked.shape) == (1, 1, 1, config.hidden_size)
    _assert_replicated(f"{expert_strategy}-{layer_type}-stacked", stacked)


@_mesh_test
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_decode_collective_candidate_matches_all_reduce(mesh_device, layer_type):
    """Compare the semaphore-backed padded RS+AG candidate at both decode reductions."""

    if os.environ.get("RUN_MULTICHIP_CCL_AB") != "1":
        pytest.skip("set RUN_MULTICHIP_CCL_AB=1 for the decode collective A/B")
    config = _config()
    config.layer_types = list(config.layer_types)
    config.layer_types[LAYER_IDX] = layer_type
    state = _real_state_dict()
    baseline = _decoder(
        state,
        config,
        mesh_device,
        max_cache_len=256,
        multichip_config=MultichipConfig(decode_collective=DECODE_COLLECTIVE_ALL_REDUCE),
    )
    candidate = _decoder(
        state,
        config,
        mesh_device,
        max_cache_len=256,
        multichip_config=MultichipConfig(decode_collective=DECODE_COLLECTIVE_RS_AG_PAD64),
    )
    baseline_page_table = baseline.create_page_table(list(reversed(range(baseline.num_cache_blocks))))
    candidate_page_table = candidate.create_page_table(list(reversed(range(candidate.num_cache_blocks))))
    baseline_key, baseline_value = baseline.create_kv_cache()
    candidate_key, candidate_value = candidate.create_kv_cache()
    generator = torch.Generator().manual_seed(6114)
    prefill_hidden = torch.randn(1, config.sliding_window, config.hidden_size, generator=generator).to(torch.bfloat16)
    baseline.prefill_forward(
        _to_tt(prefill_hidden, mesh_device),
        key_cache=baseline_key,
        value_cache=baseline_value,
        page_table=baseline_page_table,
    )
    candidate.prefill_forward(
        _to_tt(prefill_hidden, mesh_device),
        key_cache=candidate_key,
        value_cache=candidate_value,
        page_table=candidate_page_table,
    )
    decode_hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    position = config.sliding_window
    baseline_output = baseline.decode_forward(
        _to_tt(decode_hidden, mesh_device),
        key_cache=baseline_key,
        value_cache=baseline_value,
        page_table=baseline_page_table,
        cache_position=position,
        cache_position_tensor=_position_tensor(position, mesh_device),
    )
    candidate_output = candidate.decode_forward(
        _to_tt(decode_hidden, mesh_device),
        key_cache=candidate_key,
        value_cache=candidate_value,
        page_table=candidate_page_table,
        cache_position=position,
        cache_position_tensor=_position_tensor(position, mesh_device),
    )
    _assert_pcc(
        f"collective-rs-ag-pad64-{layer_type}",
        _to_torch(baseline_output),
        _to_torch(candidate_output),
        0.999,
    )
    _assert_replicated(f"collective-rs-ag-pad64-{layer_type}", candidate_output)


@_mesh_test
def test_near_tied_router_isolated_to_tp_attention_rounding(mesh_device):
    """Control a deterministic top-k discontinuity at the component boundary."""

    path = _near_tied_reference_path()
    assert path.exists(), "capture the isolated near-tied optimized reference first"
    reference = torch.load(path, map_location="cpu", weights_only=True)
    config = _config()
    config.layer_types = list(config.layer_types)
    config.layer_types[LAYER_IDX] = "full_attention"
    decoder = _decoder(
        _real_state_dict(),
        config,
        mesh_device,
        max_cache_len=256,
        expert_strategy=EXPERT_STRATEGY_TP,
    )
    page_table = decoder.create_page_table(list(reversed(range(decoder.num_cache_blocks))))
    key_cache, value_cache = decoder.create_kv_cache()
    prefill_len = config.sliding_window
    decoder.prefill_forward(
        _to_tt(reference["prefill_hidden"], mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    target_position = prefill_len + 1
    for position in range(prefill_len, target_position + 1):
        attention = decoder._decode_attention(
            _to_tt(reference[f"hidden_{position}"], mesh_device),
            key_cache,
            value_cache,
            page_table,
            position,
            _position_tensor(position, mesh_device),
        )
    _assert_pcc(
        "near-tied-tp4-attention",
        reference[f"attention_{target_position}"],
        _to_torch(attention),
        0.9998,
    )

    normalized = decoder._decode_post_attention_norm(attention)
    actual_routing = decoder._route(normalized, 1)
    repeated_routing = decoder._route(normalized, 1)
    actual_host = _to_torch(actual_routing)
    assert torch.equal(actual_host, _to_torch(repeated_routing))
    assert not torch.equal(actual_host, reference[f"routing_{target_position}"])
    assert int(torch.count_nonzero(actual_host)) == int(config.num_experts_per_tok) == 4

    exact_attention = _to_tt(reference[f"attention_{target_position}"], mesh_device)
    exact_normalized = decoder._decode_post_attention_norm(exact_attention)
    exact_routing = decoder._route(exact_normalized, 1)
    _assert_pcc(
        "near-tied-exact-attention-routing",
        reference[f"routing_{target_position}"],
        _to_torch(exact_routing),
        0.999,
    )
    exact_output = decoder._sparse_moe_forward(exact_attention, exact_normalized, exact_routing, 1)
    _assert_pcc(
        "near-tied-exact-attention-active-experts",
        reference[f"decode_{target_position}"],
        _to_torch(exact_output),
        0.99,
    )
    print(
        "NEAR_TIED_ROUTER_DIAGNOSTIC "
        f"seed={NEAR_TIED_ROUTER_SEED} position={target_position} "
        "status=deterministic_fourth_expert_swap component_paths=pass"
    )


@_mesh_test
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_warmed_trace_replay_updates_hidden_position_and_paged_cache(mesh_device, layer_type):
    config = _config()
    config.layer_types = list(config.layer_types)
    config.layer_types[LAYER_IDX] = layer_type
    # Five pages leave two not-yet-populated physical allocations that can be
    # swapped after capture without illegally remapping live KV history.
    decoder = _decoder(_real_state_dict(), config, mesh_device, max_cache_len=320)
    reverse_blocks = list(reversed(range(decoder.num_cache_blocks)))
    eager_page_table = decoder.create_page_table(reverse_blocks)
    trace_page_table = decoder.create_page_table(reverse_blocks)
    eager_key, eager_value = decoder.create_kv_cache()
    trace_key, trace_value = decoder.create_kv_cache()
    generator = torch.Generator().manual_seed(4040)
    prefill_len = config.sliding_window
    prefill_hidden = torch.randn(1, prefill_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    tt_prefill = _to_tt(prefill_hidden, mesh_device)
    decoder.prefill_forward(
        tt_prefill,
        key_cache=eager_key,
        value_cache=eager_value,
        page_table=eager_page_table,
    )
    decoder.prefill_forward(
        tt_prefill,
        key_cache=trace_key,
        value_cache=trace_value,
        page_table=trace_page_table,
    )

    cases = []
    for position in range(prefill_len, prefill_len + 3):
        hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
        cases.append(
            {
                "position": position,
                "hidden": hidden,
                "eager_hidden": _to_tt(hidden, mesh_device),
                "eager_position": _position_tensor(position, mesh_device),
                "host_hidden": _host_hidden(hidden, mesh_device),
                "host_position": _host_position(position, mesh_device),
            }
        )

    trace_hidden = _to_tt(cases[0]["hidden"], mesh_device)
    trace_position = _position_tensor(prefill_len, mesh_device)
    decoder.decode_forward(
        trace_hidden,
        key_cache=trace_key,
        value_cache=trace_value,
        page_table=trace_page_table,
        cache_position=prefill_len,
        cache_position_tensor=trace_position,
    )
    ttnn.synchronize_device(mesh_device)

    eager_references = []
    for case in cases:
        eager = decoder.decode_forward(
            case["eager_hidden"],
            key_cache=eager_key,
            value_cache=eager_value,
            page_table=eager_page_table,
            cache_position=case["position"],
            cache_position_tensor=case["eager_position"],
        )
        ttnn.synchronize_device(mesh_device)
        eager_references.append(_to_torch(eager).clone())

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        trace_hidden,
        key_cache=trace_key,
        value_cache=trace_value,
        page_table=trace_page_table,
        cache_position=prefill_len,
        cache_position_tensor=trace_position,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc(f"trace-{layer_type}-capture", eager_references[0], _to_torch(traced_output), 0.999)
        for case, eager_reference in zip(cases[1:], eager_references[1:]):
            ttnn.copy_host_to_device_tensor(case["host_hidden"], trace_hidden)
            ttnn.copy_host_to_device_tensor(case["host_position"], trace_position)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            _assert_pcc(
                f"trace-{layer_type}-mutable-{case['position']}", eager_reference, _to_torch(traced_output), 0.999
            )
            _assert_replicated(f"trace-{layer_type}-{case['position']}", traced_output)

        # Paged attention bakes the page-table contents into this target's
        # captured programs.  Future block allocation can update the stable
        # tensor, but the caller must release and recapture before replay.
        alternate_blocks = reverse_blocks.copy()
        alternate_blocks[-2], alternate_blocks[-1] = alternate_blocks[-1], alternate_blocks[-2]
        host_page_table = _host_page_table(alternate_blocks, mesh_device)
        ttnn.copy_host_to_device_tensor(host_page_table, eager_page_table)
        ttnn.copy_host_to_device_tensor(host_page_table, trace_page_table)
        alternate_position = 3 * PAGE_BLOCK_SIZE
        alternate_hidden = torch.randn(
            1,
            1,
            config.hidden_size,
            generator=generator,
        ).to(torch.bfloat16)
        eager_alternate = decoder.decode_forward(
            _to_tt(alternate_hidden, mesh_device),
            key_cache=eager_key,
            value_cache=eager_value,
            page_table=eager_page_table,
            cache_position=alternate_position,
            cache_position_tensor=_position_tensor(alternate_position, mesh_device),
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.copy_host_to_device_tensor(_host_hidden(alternate_hidden, mesh_device), trace_hidden)
        ttnn.copy_host_to_device_tensor(_host_position(alternate_position, mesh_device), trace_position)
        ttnn.release_trace(mesh_device, trace_id)
        trace_id = None
        decoder.decode_forward(
            trace_hidden,
            key_cache=trace_key,
            value_cache=trace_value,
            page_table=trace_page_table,
            cache_position=alternate_position,
            cache_position_tensor=trace_position,
        )
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = decoder.decode_forward(
            trace_hidden,
            key_cache=trace_key,
            value_cache=trace_value,
            page_table=trace_page_table,
            cache_position=alternate_position,
            cache_position_tensor=trace_position,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc(
            f"trace-{layer_type}-recaptured-page-table-{alternate_position}",
            _to_torch(eager_alternate),
            _to_torch(traced_output),
            0.999,
        )
        _assert_replicated(f"trace-{layer_type}-mutable-page-table", traced_output)
        deterministic = _to_torch(traced_output).clone()
        for replay in range(5):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            assert torch.equal(deterministic, _to_torch(traced_output)), f"{layer_type} replay {replay} changed"
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)

    logical_block = alternate_position // PAGE_BLOCK_SIZE
    physical_block = alternate_blocks[logical_block]
    for cache_name, eager_cache, trace_cache in (
        ("key", eager_key, trace_key),
        ("value", eager_value, trace_value),
    ):
        for rank, (eager_local, trace_local) in enumerate(
            zip(_all_device_torch(eager_cache), _all_device_torch(trace_cache))
        ):
            assert torch.equal(
                eager_local[physical_block], trace_local[physical_block]
            ), f"{layer_type} rank {rank} {cache_name} trace page differs from eager"


@_mesh_test
def test_full_context_cache_allocation_and_last_page_update(mesh_device):
    if os.environ.get("RUN_MULTICHIP_CONTEXT") != "1":
        pytest.skip("set RUN_MULTICHIP_CONTEXT=1 for the 131072-token cache gate")
    config = _config()
    decoder = _decoder(_real_state_dict(), config, mesh_device, max_cache_len=SUPPORTED_CONTEXT)
    page_table = decoder.create_page_table(list(reversed(range(decoder.num_cache_blocks))))
    key_cache, value_cache = decoder.create_kv_cache()
    expected = (
        SUPPORTED_CONTEXT // PAGE_BLOCK_SIZE,
        config.num_key_value_heads // TP_DEGREE,
        PAGE_BLOCK_SIZE,
        config.head_dim,
    )
    assert tuple(key_cache.shape) == tuple(value_cache.shape) == expected
    hidden = torch.randn(1, 1, config.hidden_size, generator=torch.Generator().manual_seed(8080)).to(torch.bfloat16)
    output = decoder.decode_forward(
        _to_tt(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        cache_position=SUPPORTED_CONTEXT - 1,
        cache_position_tensor=_position_tensor(SUPPORTED_CONTEXT - 1, mesh_device),
    )
    assert tuple(output.shape) == (1, 1, 1, config.hidden_size)
    _assert_replicated("full-context-last-page", output)
    physical_page = 0
    last_offset = PAGE_BLOCK_SIZE - 1
    for rank, (local_key, local_value) in enumerate(zip(_all_device_torch(key_cache), _all_device_torch(value_cache))):
        assert torch.count_nonzero(local_key[physical_page, :, last_offset]).item() > 0, f"rank {rank} K not written"
        assert torch.count_nonzero(local_value[physical_page, :, last_offset]).item() > 0, f"rank {rank} V not written"


@_mesh_test
def test_sharded_residual_topology_candidate(mesh_device):
    """Compare the selected replicated boundary with a real distributed-norm stack boundary."""

    if os.environ.get("RUN_MULTICHIP_TOPOLOGY_PROBE") != "1":
        pytest.skip("set RUN_MULTICHIP_TOPOLOGY_PROBE=1 for the residual-layout comparison")
    config = _config()
    state = _real_state_dict()
    decoder = _decoder(state, config, mesh_device, max_cache_len=128)
    repeats = int(os.environ.get("MULTICHIP_TOPOLOGY_REPEATS", "20"))
    generator = torch.Generator().manual_seed(7070)
    hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    local_attention = torch.randn(1, 1, config.num_attention_heads * config.head_dim, generator=generator).to(
        torch.bfloat16
    )

    replicated_hidden = _to_tt(hidden, mesh_device)
    sharded_hidden = ttnn.from_torch(
        hidden.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    sharded_hidden = ttnn.to_memory_config(sharded_hidden, ttnn.L1_MEMORY_CONFIG)
    local_attention = ttnn.from_torch(
        local_attention.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    local_attention = ttnn.to_memory_config(local_attention, ttnn.L1_MEMORY_CONFIG)

    norm_weight = _require_tensor(state, LAYER_IDX, "post_attention_layernorm.weight")
    distributed_norm_weight = ttnn.from_torch(
        norm_weight.reshape(1, 1, 1, -1).to(torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    router_weight = _require_tensor(state, LAYER_IDX, "mlp.router.weight")
    distributed_router_weight = ttnn.from_torch(
        router_weight.transpose(-2, -1).to(torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    router_bias = _require_tensor(state, LAYER_IDX, "mlp.router.bias").float()
    rank_selective_bias = torch.stack([router_bias] + [torch.zeros_like(router_bias) for _ in range(TP_DEGREE - 1)])
    distributed_router_bias = ttnn.from_torch(
        rank_selective_bias,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def o_partial():
        partial = ttnn.linear(
            local_attention,
            decoder.weights["o_weight"],
            bias=decoder.weights["o_bias"],
            dtype=ttnn.bfloat16,
            memory_config=decoder.tp_o_output_config,
            program_config=decoder.tp_o_program_config,
            compute_kernel_config=decoder.decode_compute_kernel_config,
        )
        return ttnn.to_memory_config(partial, ttnn.L1_MEMORY_CONFIG)

    def replicated_contract():
        projected = decoder._all_reduce(o_partial(), memory_config=ttnn.L1_MEMORY_CONFIG)
        residual = ttnn.add(replicated_hidden, projected)
        normalized = ttnn.rms_norm(
            residual,
            epsilon=decoder.eps,
            weight=decoder.weights["post_attention_norm"],
            compute_kernel_config=decoder.decode_compute_kernel_config,
        )
        return normalized, decoder._route(normalized, 1)

    def sharded_contract():
        reduced = ttnn.reduce_scatter(
            o_partial(),
            dim=3,
            num_links=decoder.multichip_config.num_links,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        residual = ttnn.add(sharded_hidden, reduced)
        stats = ttnn.rms_norm_pre_all_gather(
            residual,
            compute_kernel_config=decoder.decode_compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
        stats = ttnn.all_gather(
            stats,
            dim=3,
            num_links=decoder.multichip_config.num_links,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        normalized_local = ttnn.rms_norm_post_all_gather(
            residual,
            stats,
            epsilon=decoder.eps,
            weight=distributed_norm_weight,
            compute_kernel_config=decoder.decode_compute_kernel_config,
        )
        router_input = ttnn.typecast(ttnn.reshape(normalized_local, [1, config.hidden_size // TP_DEGREE]), ttnn.float32)
        router_logits = ttnn.linear(
            router_input,
            distributed_router_weight,
            bias=distributed_router_bias,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=decoder.compute_kernel_config,
        )
        router_logits = decoder._all_reduce(router_logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        router_logits = ttnn.typecast(router_logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(router_logits, k=decoder.top_k, dim=-1, sorted=True)
        top_values = ttnn.softmax(top_values, dim=-1, numeric_stable=True)
        routing = ttnn.scatter(ttnn.zeros_like(router_logits), dim=1, index=top_indices, src=top_values)
        normalized_for_experts = ttnn.all_gather(
            normalized_local,
            dim=3,
            num_links=decoder.multichip_config.num_links,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return normalized_for_experts, routing

    baseline_normalized, baseline_routing = replicated_contract()
    candidate_normalized, candidate_routing = sharded_contract()
    ttnn.synchronize_device(mesh_device)
    _assert_pcc(
        "sharded-residual-distributed-norm", _to_torch(baseline_normalized), _to_torch(candidate_normalized), 0.999
    )
    _assert_pcc("sharded-residual-router", _to_torch(baseline_routing), _to_torch(candidate_routing), 0.999)
    _assert_replicated("sharded-residual-gathered-for-experts", candidate_normalized)

    start = time.perf_counter()
    for _ in range(repeats):
        replicated_contract()
    ttnn.synchronize_device(mesh_device)
    replicated_ms = (time.perf_counter() - start) * 1000.0 / repeats
    start = time.perf_counter()
    for _ in range(repeats):
        sharded_contract()
    ttnn.synchronize_device(mesh_device)
    sharded_ms = (time.perf_counter() - start) * 1000.0 / repeats

    result = {
        "boundary": "O projection -> residual -> post-attention RMSNorm -> router -> sparse-expert input",
        "mesh": list(TARGET_MESH_SHAPE),
        "repeats": repeats,
        "replicated_all_reduce_ms": replicated_ms,
        "sharded_reduce_scatter_distributed_norm_gather_ms": sharded_ms,
        "replicated_over_sharded_speedup": sharded_ms / replicated_ms,
        "candidate_stack_contract": "width-sharded residual [1,1,1,720] per device",
        "candidate_next_consumer": "distributed RMSNorm, row-sharded router and 32-logit all-reduce, then full-hidden gather for TP experts",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "residual_topology_candidate.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "RESIDUAL_TOPOLOGY_RESULT "
        f"replicated_ms={replicated_ms:.6f} sharded_ms={sharded_ms:.6f} "
        f"replicated_over_sharded_speedup={sharded_ms / replicated_ms:.6f} artifact={path}"
    )


def _perf_reference_path(seq_len):
    return EVIDENCE_DIR / f"single_chip_perf_reference_seq{seq_len}.json"


def _perf_result_path(seq_len):
    override = os.environ.get("MULTICHIP_PERF_RESULT_PATH")
    return Path(override) if override else EVIDENCE_DIR / f"multichip_perf_result_seq{seq_len}.json"


def _time_prefill(decoder, device, hidden, *, repeats, label, page_table=None):
    key_cache, value_cache = decoder.create_kv_cache()
    kwargs = {"key_cache": key_cache, "value_cache": value_cache}
    if page_table is not None:
        kwargs["page_table"] = page_table
    tt_hidden = _to_tt(hidden, device)
    decoder.prefill_forward(tt_hidden, **kwargs)
    ttnn.synchronize_device(device)
    signpost(header=f"PERF_PREFILL_{label}")
    start = time.perf_counter()
    for _ in range(repeats):
        decoder.prefill_forward(tt_hidden, **kwargs)
    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    signpost(header=f"PERF_PREFILL_{label}_END")
    return elapsed_ms, key_cache, value_cache


def _time_traced_decode(
    decoder,
    device,
    hidden,
    key_cache,
    value_cache,
    *,
    position,
    trace_replays,
    label,
    page_table=None,
):
    tt_hidden = _to_tt(hidden, device)
    position_tensor = _position_tensor(position, device)
    kwargs = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cache_position": position,
        "cache_position_tensor": position_tensor,
    }
    if page_table is not None:
        kwargs["page_table"] = page_table
    decoder.decode_forward(tt_hidden, **kwargs)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    decoder.decode_forward(tt_hidden, **kwargs)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        signpost(header=f"PERF_DECODE_{label}")
        start = time.perf_counter()
        for _ in range(trace_replays):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / trace_replays
        signpost(header=f"PERF_DECODE_{label}_END")
        return elapsed_ms
    finally:
        ttnn.release_trace(device, trace_id)


@_single_chip_test
def test_capture_single_chip_optimized_perf_reference(mesh_device):
    if os.environ.get("RUN_MULTICHIP_DECODER_PERF") != "1":
        pytest.skip("set RUN_MULTICHIP_DECODER_PERF=1 to capture warmed optimized timing")
    config = _config()
    seq_len = int(os.environ.get("MULTICHIP_DECODER_PERF_SEQ", config.sliding_window))
    repeats = int(os.environ.get("MULTICHIP_DECODER_PREFILL_REPEATS", "10"))
    trace_replays = int(os.environ.get("MULTICHIP_DECODER_TRACE_REPLAYS", "100"))
    decoder = OptimizedDecoder.from_state_dict(
        _real_state_dict(),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=max(128, seq_len + 1),
    )
    generator = torch.Generator().manual_seed(9191)
    prefill_hidden = torch.randn(1, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    decode_hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    prefill_ms, key_cache, value_cache = _time_prefill(
        decoder, mesh_device, prefill_hidden, repeats=repeats, label="SINGLE_CHIP"
    )
    decode_ms = _time_traced_decode(
        decoder,
        mesh_device,
        decode_hidden,
        key_cache,
        value_cache,
        position=seq_len,
        trace_replays=trace_replays,
        label="SINGLE_CHIP",
    )
    result = {
        "mesh": [1, 1],
        "seq_len": seq_len,
        "prefill_ms": prefill_ms,
        "traced_decode_ms": decode_ms,
        "prefill_repeats": repeats,
        "trace_replays": trace_replays,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = _perf_reference_path(seq_len)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"SINGLE_CHIP_PERF_REFERENCE {json.dumps(result, sort_keys=True)} artifact={path}")


@_mesh_test
def test_multichip_decoder_perf(mesh_device):
    if os.environ.get("RUN_MULTICHIP_DECODER_PERF") != "1":
        pytest.skip("set RUN_MULTICHIP_DECODER_PERF=1 to run warmed multichip timing")
    config = _config()
    seq_len = int(os.environ.get("MULTICHIP_DECODER_PERF_SEQ", config.sliding_window))
    repeats = int(os.environ.get("MULTICHIP_DECODER_PREFILL_REPEATS", "10"))
    trace_replays = int(os.environ.get("MULTICHIP_DECODER_TRACE_REPLAYS", "100"))
    reference_path = _perf_reference_path(seq_len)
    assert reference_path.exists(), "capture the isolated single-chip timing first"
    baseline = json.loads(reference_path.read_text())
    multichip_config, qkv_candidate = _perf_multichip_config_from_env()
    decoder = _decoder(
        _real_state_dict(),
        config,
        mesh_device,
        max_cache_len=max(128, seq_len + 1),
        multichip_config=multichip_config,
    )
    page_table = decoder.create_page_table()
    generator = torch.Generator().manual_seed(9191)
    prefill_hidden = torch.randn(1, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    decode_hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    prefill_ms, key_cache, value_cache = _time_prefill(
        decoder,
        mesh_device,
        prefill_hidden,
        repeats=repeats,
        label=f"MULTICHIP_{multichip_config.expert_strategy.upper()}",
        page_table=page_table,
    )
    decode_ms = _time_traced_decode(
        decoder,
        mesh_device,
        decode_hidden,
        key_cache,
        value_cache,
        position=seq_len,
        trace_replays=trace_replays,
        label=f"MULTICHIP_{multichip_config.expert_strategy.upper()}",
        page_table=page_table,
    )
    result = {
        "mesh": list(TARGET_MESH_SHAPE),
        "seq_len": seq_len,
        "expert_strategy": multichip_config.expert_strategy,
        "decode_collective": multichip_config.decode_collective,
        "ep_prefill_geometry": {
            "gate_up_cores": list(multichip_config.ep_prefill_gate_up_cores),
            "gate_up_subblock_w": multichip_config.ep_prefill_gate_up_subblock_w,
            "down_cores": list(multichip_config.ep_prefill_down_cores),
            "down_subblock_w": multichip_config.ep_prefill_down_subblock_w,
            "chunk_size": multichip_config.active_prefill_chunk_size,
            "post_sparse_bf16": multichip_config.ep_prefill_post_sparse_bf16,
        },
        "qkv_candidate": qkv_candidate,
        "qkv_geometry": [
            multichip_config.qkv_input_cores,
            multichip_config.qkv_in0_block_w,
            multichip_config.qkv_output_tiles_per_core,
            multichip_config.qkv_out_subblock_w,
        ],
        "single_chip": baseline,
        "multichip_prefill_ms": prefill_ms,
        "multichip_traced_decode_ms": decode_ms,
        "prefill_speedup": baseline["prefill_ms"] / prefill_ms,
        "prefill_efficiency": baseline["prefill_ms"] / prefill_ms / TP_DEGREE,
        "decode_speedup": baseline["traced_decode_ms"] / decode_ms,
        "decode_efficiency": baseline["traced_decode_ms"] / decode_ms / TP_DEGREE,
        "prefill_repeats": repeats,
        "trace_replays": trace_replays,
    }
    path = _perf_result_path(seq_len)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"MULTICHIP_PERF_RESULT {json.dumps(result, sort_keys=True)} artifact={path}")
