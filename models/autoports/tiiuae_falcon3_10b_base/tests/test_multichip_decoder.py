# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

import ttnn
from models.autoports.tiiuae_falcon3_10b_base.tests.test_functional_decoder import (
    REAL_WEIGHT_PCC,
    _assert_pcc,
    _config,
    _hf_decode,
    _hf_layer,
    _hf_prefill,
    _real_layer_state_dict,
    _synthetic_state_dict,
)
from models.autoports.tiiuae_falcon3_10b_base.tests.test_optimized_decoder import (
    _recorded_layer20_inputs,
    _recorded_layer20_seq31_inputs,
)
from models.autoports.tiiuae_falcon3_10b_base.tt.functional_decoder import IR_REPRESENTATIVE_LAYER
from models.autoports.tiiuae_falcon3_10b_base.tt.multichip_decoder import (
    TARGET_MESH_SHAPE,
    TENSOR_PARALLEL_SIZE,
    DecodeAllReduceResources,
    MultichipDecoder,
)
from models.autoports.tiiuae_falcon3_10b_base.tt.optimized_decoder import OptimizedDecoder, _compute_config
from models.common.modules.tt_ccl import TT_CCL
from models.common.utility_functions import comp_pcc

REPO_ROOT = Path(__file__).parents[4]
IMPLEMENTATION_PATH = Path(__file__).parents[1] / "tt" / "multichip_decoder.py"
RESULTS_DIR = Path(__file__).parents[1] / "doc" / "multichip_decoder" / "results"

MESH_PARAMS = [
    pytest.param(
        TARGET_MESH_SHAPE,
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000},
        id="1x4-ring",
    )
]


def _mesh_input(hidden_states: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        hidden_states.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _mesh_int32(host: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        host,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _first_rank(tensor: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def _write_result_artifact(filename: str, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("generated_at_utc", datetime.now(timezone.utc).isoformat())
    payload.setdefault(
        "repo_head",
        subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, text=True).strip(),
    )
    payload.setdefault("implementation_sha256", hashlib.sha256(IMPLEMENTATION_PATH.read_bytes()).hexdigest())
    payload.setdefault("test_sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    payload.setdefault("hardware", "4x Blackhole p300c, mesh 1x4, FABRIC_1D_RING")
    output_dir = Path(os.getenv("FALCON3_MULTICHIP_RESULTS_DIR", RESULTS_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _assert_replicated(tensor: ttnn.Tensor) -> None:
    ranks = [ttnn.to_torch(rank) for rank in ttnn.get_device_tensors(tensor)]
    assert len(ranks) == TENSOR_PARALLEL_SIZE
    for rank in ranks[1:]:
        assert torch.equal(ranks[0], rank)


def _paged_cache_to_torch(cache: ttnn.Tensor, page_table: torch.Tensor, logical_length: int) -> torch.Tensor:
    rank_caches = [ttnn.to_torch(rank).float() for rank in ttnn.get_device_tensors(cache)]
    users = []
    for user_id in range(page_table.shape[0]):
        rank_users = []
        for rank_cache in rank_caches:
            logical_pages = [rank_cache[int(page_id)] for page_id in page_table[user_id]]
            rank_users.append(torch.cat(logical_pages, dim=1)[:, :logical_length, :])
        users.append(torch.cat(rank_users, dim=0))
    return torch.stack(users)


def _contiguous_cache_to_torch(cache: ttnn.Tensor, logical_length: int) -> torch.Tensor:
    ranks = [ttnn.to_torch(rank).float()[:, :, :logical_length, :] for rank in ttnn.get_device_tensors(cache)]
    return torch.cat(ranks, dim=1)


def _single_device_input(hidden_states: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        hidden_states.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _single_device_position(batch: int, position: int, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.full((batch,), position, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _hf_key_value_samples(layer, config, hidden_states: torch.Tensor, positions: list[int]):
    """Reference token-local K/V projections and RoPE without full-sequence HF attention."""
    samples = hidden_states[:, positions, :]
    normed = layer.input_layernorm(samples)
    batch, sample_count, _ = samples.shape
    key = (
        layer.self_attn.k_proj(normed)
        .view(batch, sample_count, config.num_key_value_heads, config.head_dim)
        .transpose(1, 2)
    )
    value = (
        layer.self_attn.v_proj(normed)
        .view(batch, sample_count, config.num_key_value_heads, config.head_dim)
        .transpose(1, 2)
    )
    position_ids = torch.tensor(positions, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    cos, sin = LlamaRotaryEmbedding(config)(samples, position_ids)
    key, _ = apply_rotary_pos_emb(key, key, cos, sin)
    return key, value


def _release_model(model) -> None:
    names = (
        "qkv_weight",
        "qkv_decode_weight",
        "o_weight",
        "o_decode_weight",
        "gate_weight",
        "up_weight",
        "gate_decode_weight",
        "up_decode_weight",
        "gate_up_weight",
        "gate_up_decode_weight",
        "down_weight",
        "down_decode_weight",
        "input_norm_weight",
        "post_attention_norm_weight",
    )
    for name in names:
        tensor = getattr(model, name, None)
        if tensor is not None:
            tensor.deallocate(True)
    if getattr(model, "owns_rope_cache", False):
        model.cos_cache.deallocate(True)
        model.sin_cache.deallocate(True)
    if getattr(model, "owns_decode_all_reduce_resources", False):
        model.decode_all_reduce_resources.close()
    del model
    gc.collect()


def _trace_mesh_callable(mesh_device, function, *, samples: int = 5, iterations: int = 20):
    warm_output = function()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = function()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(iterations):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        timings.append((time.perf_counter() - start) * 1000 / iterations)
    return warm_output, trace_output, trace_id, statistics.median(timings), timings


def test_multichip_runtime_is_owned_and_host_fallback_free():
    assert MultichipDecoder.single_chip_baseline_cls is OptimizedDecoder
    hot_methods = (
        MultichipDecoder.prefill_forward,
        MultichipDecoder.decode_forward,
        MultichipDecoder.decode_forward_to_residual,
        MultichipDecoder.decode_forward_from_residual,
        MultichipDecoder.materialize_decode_output,
        MultichipDecoder._prefill_attention,
        MultichipDecoder._prefill_linear_chunked,
        MultichipDecoder._decode_attention,
        MultichipDecoder._decode_qkv,
        MultichipDecoder._apply_decode_rope_dedicated,
        MultichipDecoder._prefill_mlp,
        MultichipDecoder._prefill_mlp_chunk,
        MultichipDecoder._decode_mlp,
        MultichipDecoder._all_reduce_partial,
        DecodeAllReduceResources.all_reduce,
    )
    forbidden = ("from_torch", "to_torch", "OptimizedDecoder.", "FunctionalDecoder.")
    for method in hot_methods:
        assert method.__qualname__.startswith(("MultichipDecoder.", "DecodeAllReduceResources."))
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} contains runtime fallback token {token!r}"


@pytest.mark.parametrize("mesh_device,device_params", MESH_PARAMS, indirect=True)
@pytest.mark.timeout(1800)
def test_synthetic_tp4_prefill_decode_smoke(mesh_device):
    config = _config()
    batch, seq_len, max_cache_len = 1, 17, 64
    state_dict = _synthetic_state_dict(config, IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=max_cache_len,
        precision_policy="bf16_hifi4",
    )
    assert tuple(int(v) for v in mesh_device.shape) == TARGET_MESH_SHAPE
    assert model.logical_local_intermediate_size == 5760
    assert model.local_intermediate_size == 6144
    assert model.local_num_heads == 3
    assert model.local_num_kv_heads == 1

    generator = torch.Generator().manual_seed(20260717)
    hidden = torch.randn((batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    hf_cache = DynamicCache(config=config)
    expected = _hf_prefill(hf_layer, config, hidden, cache=hf_cache)
    key_cache, value_cache = model.allocate_kv_cache()
    tt_hidden = _mesh_input(hidden, mesh_device)
    tt_output = model.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache)
    _assert_replicated(tt_output)
    _assert_pcc("synthetic TP4 prefill", expected, _first_rank(tt_output).squeeze(0), REAL_WEIGHT_PCC)

    # The replicated public output is directly consumable by the next decoder
    # in a layer stack; validate the contract with a second block invocation.
    stacked_hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    stacked_hf_cache = DynamicCache(config=config)
    expected_stacked = _hf_prefill(stacked_hf_layer, config, expected, cache=stacked_hf_cache)
    stacked_key, stacked_value = model.allocate_kv_cache()
    stacked_output = model.prefill_forward(
        tt_output,
        key_cache=stacked_key,
        value_cache=stacked_value,
    )
    _assert_replicated(stacked_output)
    _assert_pcc(
        "synthetic TP4 stacked prefill",
        expected_stacked,
        _first_rank(stacked_output).squeeze(0),
        REAL_WEIGHT_PCC,
    )

    decode_hidden = torch.randn((batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    expected_decode = _hf_decode(hf_layer, config, decode_hidden, hf_cache, seq_len)
    tt_decode_hidden = _mesh_input(decode_hidden, mesh_device)
    cache_position = _mesh_int32(torch.full((batch,), seq_len, dtype=torch.int32), mesh_device)
    tt_decode = model.decode_forward(
        tt_decode_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        position_index=seq_len,
    )
    _assert_replicated(tt_decode)
    _assert_pcc("synthetic TP4 decode", expected_decode, _first_rank(tt_decode).squeeze(0), REAL_WEIGHT_PCC)

    for tensor in (
        tt_hidden,
        tt_output,
        stacked_output,
        stacked_key,
        stacked_value,
        tt_decode_hidden,
        cache_position,
        tt_decode,
        key_cache,
        value_cache,
    ):
        tensor.deallocate(True)
    _release_model(model)


@pytest.mark.parametrize("mesh_device,device_params", MESH_PARAMS, indirect=True)
@pytest.mark.timeout(1800)
def test_real_layer_paged_non_aligned_prefill_decode_cache_and_trace(mesh_device):
    config = _config()
    batch, seq_len, max_cache_len = 32, 31, 64
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=max_cache_len,
    )
    prefill, decode_31, decode_32 = _recorded_layer20_seq31_inputs(batch)
    hf_cache = DynamicCache(config=config)
    expected_prefill = _hf_prefill(hf_layer, config, prefill, cache=hf_cache)

    pages_per_user = math.ceil(max_cache_len / model.page_block_size)
    page_table_host = torch.arange(batch * pages_per_user, dtype=torch.int32).reshape(batch, pages_per_user).flip(1)
    page_table = _mesh_int32(page_table_host, mesh_device)
    key_cache, value_cache = model.allocate_kv_cache(paged=True)
    tt_prefill = _mesh_input(prefill, mesh_device)
    tt_prefill_output = model.prefill_forward(
        tt_prefill,
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    _assert_replicated(tt_prefill_output)
    _assert_pcc(
        "multichip real prefill seq=31", expected_prefill, _first_rank(tt_prefill_output).squeeze(0), REAL_WEIGHT_PCC
    )
    hf_layer_cache = hf_cache.layers[IR_REPRESENTATIVE_LAYER]
    _assert_pcc(
        "multichip paged key cache seq=31",
        hf_layer_cache.keys,
        _paged_cache_to_torch(key_cache, page_table_host, seq_len),
        REAL_WEIGHT_PCC,
    )
    _assert_pcc(
        "multichip paged value cache seq=31",
        hf_layer_cache.values,
        _paged_cache_to_torch(value_cache, page_table_host, seq_len),
        REAL_WEIGHT_PCC,
    )

    expected_31 = _hf_decode(hf_layer, config, decode_31, hf_cache, 31)
    tt_decode_31 = _mesh_input(decode_31, mesh_device)
    position_31 = _mesh_int32(torch.full((batch,), 31, dtype=torch.int32), mesh_device)
    out_31 = model.decode_forward(
        tt_decode_31,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=position_31,
        position_index=31,
        page_table=page_table,
    )
    _assert_replicated(out_31)
    _assert_pcc("multichip real decode position=31", expected_31, _first_rank(out_31).squeeze(0), REAL_WEIGHT_PCC)

    expected_32 = _hf_decode(hf_layer, config, decode_32, hf_cache, 32)
    tt_decode_32 = _mesh_input(decode_32, mesh_device)
    position_32 = _mesh_int32(torch.full((batch,), 32, dtype=torch.int32), mesh_device)
    out_32 = model.decode_forward(
        tt_decode_32,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=position_32,
        position_index=32,
        page_table=page_table,
    )
    _assert_replicated(out_32)
    _assert_pcc("multichip real decode position=32", expected_32, _first_rank(out_32).squeeze(0), REAL_WEIGHT_PCC)
    _assert_pcc(
        "multichip paged key cache position=32",
        hf_layer_cache.keys,
        _paged_cache_to_torch(key_cache, page_table_host, 33),
        REAL_WEIGHT_PCC,
    )
    _assert_pcc(
        "multichip paged value cache position=32",
        hf_layer_cache.values,
        _paged_cache_to_torch(value_cache, page_table_host, 33),
        REAL_WEIGHT_PCC,
    )

    position_33 = _mesh_int32(torch.full((batch,), 33, dtype=torch.int32), mesh_device)

    def traced_decode():
        return model.decode_forward(
            tt_decode_32,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=position_33,
            position_index=33,
            page_table=page_table,
        )

    warm, traced, trace_id, _, _ = _trace_mesh_callable(mesh_device, traced_decode, samples=2, iterations=4)
    warm_host = _first_rank(warm)
    traced_host = _first_rank(traced)
    _assert_replicated(traced)
    assert torch.equal(warm_host, traced_host)
    for _ in range(4):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    assert torch.equal(traced_host, _first_rank(traced))
    ttnn.release_trace(mesh_device, trace_id)

    for tensor in (
        tt_prefill,
        tt_prefill_output,
        tt_decode_31,
        position_31,
        out_31,
        tt_decode_32,
        position_32,
        out_32,
        position_33,
        warm,
        traced,
        page_table,
        key_cache,
        value_cache,
    ):
        tensor.deallocate(True)
    _release_model(model)


@pytest.mark.parametrize("mesh_device,device_params", MESH_PARAMS, indirect=True)
@pytest.mark.timeout(1800)
def test_decode_uses_heterogeneous_device_positions_per_user(mesh_device):
    """A batched decode must match independent batch-one calls at distinct positions."""
    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    _, decode_31, decode_32 = _recorded_layer20_seq31_inputs(1)
    positions = (17, 31)
    batch_hidden = torch.cat((decode_31, decode_32), dim=0)
    batch_model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        mesh_device=mesh_device,
        batch=2,
        max_cache_len=64,
    )
    single_model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=64,
    )
    batch_key, batch_value = batch_model.allocate_kv_cache()
    batch_input = _mesh_input(batch_hidden, mesh_device)
    batch_position = _mesh_int32(torch.tensor(positions, dtype=torch.int32), mesh_device)
    batch_output = batch_model.decode_forward(
        batch_input,
        key_cache=batch_key,
        value_cache=batch_value,
        cache_position=batch_position,
        position_index=max(positions),
    )
    _assert_replicated(batch_output)
    actual = _first_rank(batch_output).squeeze(0)

    single_tensors = []
    user_pcc = {}
    key_pcc = {}
    value_pcc = {}
    for user, (position, hidden) in enumerate(zip(positions, (decode_31, decode_32))):
        single_key, single_value = single_model.allocate_kv_cache()
        single_input = _mesh_input(hidden, mesh_device)
        single_position = _mesh_int32(torch.tensor([position], dtype=torch.int32), mesh_device)
        single_output = single_model.decode_forward(
            single_input,
            key_cache=single_key,
            value_cache=single_value,
            cache_position=single_position,
            position_index=position,
        )
        expected = _first_rank(single_output).squeeze(0)[0]
        passed, output_pcc = comp_pcc(expected.float(), actual[user].float(), pcc=0.999)
        assert passed, f"heterogeneous user {user} position {position}: PCC={output_pcc}"
        user_pcc[str(position)] = output_pcc

        per_rank_key = []
        per_rank_value = []
        for batch_rank, batch_value_rank, single_rank, single_value_rank in zip(
            ttnn.get_device_tensors(batch_key),
            ttnn.get_device_tensors(batch_value),
            ttnn.get_device_tensors(single_key),
            ttnn.get_device_tensors(single_value),
        ):
            batch_key_host = ttnn.to_torch(batch_rank)[user, :, position]
            batch_value_host = ttnn.to_torch(batch_value_rank)[user, :, position]
            single_key_host = ttnn.to_torch(single_rank)[0, :, position]
            single_value_host = ttnn.to_torch(single_value_rank)[0, :, position]
            per_rank_key.append(comp_pcc(single_key_host.float(), batch_key_host.float())[1])
            per_rank_value.append(comp_pcc(single_value_host.float(), batch_value_host.float())[1])
        assert min(per_rank_key) >= 0.999
        assert min(per_rank_value) >= 0.999
        key_pcc[str(position)] = min(per_rank_key)
        value_pcc[str(position)] = min(per_rank_value)
        single_tensors.extend((single_key, single_value, single_input, single_position, single_output))

    _write_result_artifact(
        "heterogeneous_positions.json",
        {
            "positions": list(positions),
            "output_pcc_vs_independent_batch1": user_pcc,
            "key_cache_pcc_vs_independent_batch1": key_pcc,
            "value_cache_pcc_vs_independent_batch1": value_pcc,
        },
    )
    for tensor in (batch_key, batch_value, batch_input, batch_position, batch_output, *single_tensors):
        tensor.deallocate(True)
    _release_model(batch_model)
    _release_model(single_model)


@pytest.mark.parametrize("mesh_device,device_params", MESH_PARAMS, indirect=True)
@pytest.mark.timeout(1800)
def test_real_layer_paged_prefill_1025_matches_hf(mesh_device):
    """Exercise the internal 1,024-row chunk boundary with a non-aligned prompt."""
    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    base, _, _ = _recorded_layer20_seq31_inputs(1)
    seq_len = 1025
    hidden = base.repeat(1, math.ceil(seq_len / base.shape[1]), 1)[:, :seq_len, :]
    hf_cache = DynamicCache(config=config)
    expected = _hf_prefill(hf_layer, config, hidden, cache=hf_cache)
    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=1056,
    )
    key_cache, value_cache = model.allocate_kv_cache(paged=True)
    pages = math.ceil(model.max_cache_len / model.page_block_size)
    page_table_host = torch.roll(torch.arange(pages, dtype=torch.int32), shifts=1).unsqueeze(0)
    page_table = _mesh_int32(page_table_host, mesh_device)
    tt_hidden = _mesh_input(hidden, mesh_device)
    output = model.prefill_forward(
        tt_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    _assert_replicated(output)
    output_pcc = _assert_pcc(
        "multichip paged prefill seq=1025",
        expected,
        _first_rank(output).squeeze(0),
        REAL_WEIGHT_PCC,
    )
    hf_cache_layer = hf_cache.layers[IR_REPRESENTATIVE_LAYER]
    key_pcc = _assert_pcc(
        "multichip paged key cache seq=1025",
        hf_cache_layer.keys,
        _paged_cache_to_torch(key_cache, page_table_host, seq_len),
        REAL_WEIGHT_PCC,
    )
    value_pcc = _assert_pcc(
        "multichip paged value cache seq=1025",
        hf_cache_layer.values,
        _paged_cache_to_torch(value_cache, page_table_host, seq_len),
        REAL_WEIGHT_PCC,
    )
    _write_result_artifact(
        "prefill_1025.json",
        {
            "logical_sequence_length": seq_len,
            "internal_chunk_rows": 1024,
            "page_table": "cyclic_permutation",
            "prefill_pcc": output_pcc,
            "key_cache_pcc": key_pcc,
            "value_cache_pcc": value_pcc,
        },
    )
    for tensor in (key_cache, value_cache, page_table, tt_hidden, output):
        tensor.deallocate(True)
    _release_model(model)


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_MAX_CONTEXT") != "1",
    reason="manual full HF-context capacity and correctness gate",
)
@pytest.mark.parametrize("mesh_device,device_params", MESH_PARAMS, indirect=True)
@pytest.mark.timeout(3600)
def test_batch1_advertised_context_paged_cache_and_last_position(mesh_device):
    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    base, _, _ = _recorded_layer20_seq31_inputs(1)
    _, decode_hidden = _recorded_layer20_inputs(1)
    max_cache_len = config.max_position_embeddings
    prefill = base.repeat(1, math.ceil(max_cache_len / base.shape[1]), 1)[:, :max_cache_len, :]
    sample_positions = [0, 31, 1023, 1024, 16383, max_cache_len - 1]
    expected_key, expected_value = _hf_key_value_samples(hf_layer, config, prefill, sample_positions)
    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        mesh_device=mesh_device,
        batch=1,
    )
    assert model.max_cache_len == max_cache_len == 32768
    key_cache, value_cache = model.allocate_kv_cache(paged=True)
    pages = max_cache_len // model.page_block_size
    page_table_host = torch.roll(torch.arange(pages, dtype=torch.int32), shifts=1).unsqueeze(0)
    page_table = _mesh_int32(page_table_host, mesh_device)
    tt_prefill = _mesh_input(prefill, mesh_device)
    prefill_start = time.perf_counter()
    prefill_output = model.prefill_forward(
        tt_prefill,
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    prefill_seconds = time.perf_counter() - prefill_start
    assert tuple(prefill_output.shape) == (1, 1, max_cache_len, config.hidden_size)
    first_output = ttnn.slice(prefill_output, [0, 0, 0, 0], [1, 1, 1, config.hidden_size])
    last_output = ttnn.slice(
        prefill_output,
        [0, 0, max_cache_len - 1, 0],
        [1, 1, max_cache_len, config.hidden_size],
    )
    for sample in (first_output, last_output):
        _assert_replicated(sample)
        assert torch.isfinite(_first_rank(sample)).all()
    actual_key = _paged_cache_to_torch(key_cache, page_table_host, max_cache_len)[:, :, sample_positions, :]
    actual_value = _paged_cache_to_torch(value_cache, page_table_host, max_cache_len)[:, :, sample_positions, :]
    sampled_key_pcc = _assert_pcc(
        "multichip max-context sampled key cache",
        expected_key,
        actual_key,
        REAL_WEIGHT_PCC,
    )
    sampled_value_pcc = _assert_pcc(
        "multichip max-context sampled value cache",
        expected_value,
        actual_value,
        REAL_WEIGHT_PCC,
    )
    tt_hidden = _mesh_input(decode_hidden, mesh_device)
    cache_position = _mesh_int32(torch.tensor([max_cache_len - 1], dtype=torch.int32), mesh_device)
    output = model.decode_forward(
        tt_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        position_index=max_cache_len - 1,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    _assert_replicated(output)
    assert tuple(_first_rank(output).shape) == (1, 1, 1, config.hidden_size)
    assert torch.isfinite(_first_rank(output)).all()
    local_cache_shape = list(ttnn.get_device_tensors(key_cache)[0].shape)
    assert local_cache_shape == [pages, 1, model.page_block_size, model.head_dim]
    _write_result_artifact(
        "max_context_batch1.json",
        {
            "max_cache_len": max_cache_len,
            "batch": 1,
            "page_block_size": model.page_block_size,
            "physical_pages_per_device": pages,
            "local_kv_heads_per_device": model.local_num_kv_heads,
            "local_cache_shape_each_k_or_v": local_cache_shape,
            "full_prefill_executed": True,
            "full_prefill_logical_sequence_length": max_cache_len,
            "full_prefill_internal_mlp_chunks": max_cache_len // 1024,
            "full_prefill_seconds": prefill_seconds,
            "sampled_reference_positions": sample_positions,
            "sampled_key_cache_pcc_vs_hf": sampled_key_pcc,
            "sampled_value_cache_pcc_vs_hf": sampled_value_pcc,
            "last_position_executed": max_cache_len - 1,
            "page_table": "cyclic_permutation",
            "finite_replicated_output": True,
        },
    )
    for tensor in (
        key_cache,
        value_cache,
        page_table,
        tt_prefill,
        prefill_output,
        first_output,
        last_output,
        tt_hidden,
        cache_position,
        output,
    ):
        tensor.deallocate(True)
    _release_model(model)


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_BASELINE") != "1",
    reason="manual serialized one-chip then four-chip optimized-baseline comparison",
)
@pytest.mark.timeout(1800)
def test_multichip_directly_matches_single_chip_optimized_baseline():
    """Run identical real layer-20 inputs first on one chip, then on the 1x4 ring."""
    if ttnn.GetNumAvailableDevices() < TENSOR_PARALLEL_SIZE:
        pytest.skip("four Blackhole devices are required")
    config = _config()
    batch, seq_len = 32, 17
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    prefill, decode_hidden = _recorded_layer20_inputs(batch)
    precision_policy = os.getenv("FALCON3_MULTICHIP_PRECISION_POLICY", "all_bfp4_lofi")
    decode_matmul_mode = os.getenv("FALCON3_MULTICHIP_DECODE_MATMUL_MODE", "dram_sharded")
    use_packed_mlp = os.getenv("FALCON3_MULTICHIP_PACKED_MLP", "0") == "1"
    packed_mlp_unpack_mode = os.getenv("FALCON3_MULTICHIP_PACKED_UNPACK", "dram")
    decode_rope_mode = os.getenv("FALCON3_MULTICHIP_DECODE_ROPE_MODE", "dedicated")
    decode_output_mode = os.getenv("FALCON3_MULTICHIP_DECODE_OUTPUT_MODE", "direct_dram")
    use_persistent_decode_all_reduce = os.getenv("FALCON3_MULTICHIP_PERSISTENT_DECODE_AR", "1") == "1"

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    baseline_device = ttnn.open_device(device_id=0, trace_region_size=100_000_000)
    try:
        baseline = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=baseline_device,
            batch=batch,
            max_cache_len=64,
            precision_policy=precision_policy,
        )
        baseline_input = _single_device_input(prefill, baseline_device)
        baseline_key, baseline_value = baseline.allocate_kv_cache()
        baseline_prefill_tt = baseline.prefill_forward(
            baseline_input,
            key_cache=baseline_key,
            value_cache=baseline_value,
        )
        baseline_prefill = ttnn.to_torch(baseline_prefill_tt).squeeze(0)
        baseline_decode_input = _single_device_input(decode_hidden, baseline_device)
        baseline_position = _single_device_position(batch, seq_len, baseline_device)
        baseline_decode_tt = baseline.decode_forward(
            baseline_decode_input,
            key_cache=baseline_key,
            value_cache=baseline_value,
            cache_position=baseline_position,
            position_index=seq_len,
        )
        baseline_decode = ttnn.to_torch(baseline_decode_tt).squeeze(0)
        baseline_key_host = ttnn.to_torch(baseline_key)[:, :, : seq_len + 1, :]
        baseline_value_host = ttnn.to_torch(baseline_value)[:, :, : seq_len + 1, :]
        for tensor in (
            baseline_input,
            baseline_key,
            baseline_value,
            baseline_prefill_tt,
            baseline_decode_input,
            baseline_position,
            baseline_decode_tt,
        ):
            tensor.deallocate(True)
        _release_model(baseline)
    finally:
        ttnn.close_device(baseline_device)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = None
    try:
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*TARGET_MESH_SHAPE),
            trace_region_size=100_000_000,
        )
        model = MultichipDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=64,
            decode_matmul_mode=decode_matmul_mode,
            use_packed_mlp=use_packed_mlp,
            packed_mlp_unpack_mode=packed_mlp_unpack_mode,
            decode_rope_mode=decode_rope_mode,
            decode_output_mode=decode_output_mode,
            use_persistent_decode_all_reduce=use_persistent_decode_all_reduce,
        )
        key_cache, value_cache = model.allocate_kv_cache()
        tt_prefill = _mesh_input(prefill, mesh_device)
        tt_prefill_output = model.prefill_forward(
            tt_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        actual_prefill = _first_rank(tt_prefill_output).squeeze(0)
        tt_decode = _mesh_input(decode_hidden, mesh_device)
        tt_position = _mesh_int32(torch.full((batch,), seq_len, dtype=torch.int32), mesh_device)
        tt_decode_output = model.decode_forward(
            tt_decode,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=tt_position,
            position_index=seq_len,
        )
        actual_decode = _first_rank(tt_decode_output).squeeze(0)
        prefill_pcc = comp_pcc(baseline_prefill.float(), actual_prefill.float())[1]
        decode_pcc = comp_pcc(baseline_decode.float(), actual_decode.float())[1]
        key_pcc = comp_pcc(
            baseline_key_host.float(),
            _contiguous_cache_to_torch(key_cache, seq_len + 1).float(),
        )[1]
        value_pcc = comp_pcc(
            baseline_value_host.float(),
            _contiguous_cache_to_torch(value_cache, seq_len + 1).float(),
        )[1]
        assert min(prefill_pcc, decode_pcc, key_pcc, value_pcc) >= REAL_WEIGHT_PCC
        _write_result_artifact(
            os.getenv("FALCON3_MULTICHIP_PCC_FILENAME", "direct_optimized_baseline_pcc.json"),
            {
                "batch": batch,
                "sequence_length": seq_len,
                "weights": "real_layer_20",
                "precision_policy": precision_policy,
                "single_chip_baseline": "OptimizedDecoder defaults",
                "multichip": "MultichipDecoder TP=4",
                "decode_matmul_mode": decode_matmul_mode,
                "use_packed_mlp": use_packed_mlp,
                "packed_mlp_unpack_mode": packed_mlp_unpack_mode,
                "decode_rope_mode": decode_rope_mode,
                "decode_output_mode": decode_output_mode,
                "persistent_decode_all_reduce": use_persistent_decode_all_reduce,
                "prefill_pcc": prefill_pcc,
                "decode_pcc": decode_pcc,
                "key_cache_pcc": key_pcc,
                "value_cache_pcc": value_pcc,
            },
        )
        for tensor in (
            key_cache,
            value_cache,
            tt_prefill,
            tt_prefill_output,
            tt_decode,
            tt_position,
            tt_decode_output,
        ):
            tensor.deallocate(True)
        _release_model(model)
    finally:
        if mesh_device is not None:
            ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        gc.collect()


@pytest.mark.skipif(os.getenv("FALCON3_RUN_MULTICHIP_PERF") != "1", reason="manual warmed TP4 performance gate")
@pytest.mark.parametrize("mesh_device,device_params", MESH_PARAMS, indirect=True)
@pytest.mark.timeout(1800)
def test_warmed_multichip_trace_performance(mesh_device):
    config = _config()
    batch = int(os.getenv("FALCON3_MULTICHIP_PERF_BATCH", "32"))
    max_cache_len = 64
    qkv_target_cores = int(os.getenv("FALCON3_MULTICHIP_QKV_CORES", "4"))
    o_target_cores = int(os.getenv("FALCON3_MULTICHIP_O_CORES", "2"))
    gate_up_target_cores = int(os.getenv("FALCON3_MULTICHIP_GATE_CORES", "24"))
    down_target_cores = int(os.getenv("FALCON3_MULTICHIP_DOWN_CORES", "8"))
    prefill_grid_x = int(os.getenv("FALCON3_MULTICHIP_PREFILL_GRID_X", "11"))
    prefill_in0_block_w = int(os.getenv("FALCON3_MULTICHIP_PREFILL_IN0_BLOCK_W", "8"))
    num_links = int(os.getenv("FALCON3_MULTICHIP_NUM_LINKS", "2"))
    topology_name = os.getenv("FALCON3_MULTICHIP_TOPOLOGY", "ring").lower()
    topology = ttnn.Topology.Ring if topology_name == "ring" else ttnn.Topology.Linear
    ccl_dtype_name = os.getenv("FALCON3_MULTICHIP_CCL_DTYPE", "bf16").lower()
    ccl_dtype = ttnn.bfloat16 if ccl_dtype_name == "bf16" else ttnn.bfloat8_b
    precision_policy = os.getenv("FALCON3_MULTICHIP_PRECISION_POLICY", "all_bfp4_lofi")
    decode_matmul_mode = os.getenv("FALCON3_MULTICHIP_DECODE_MATMUL_MODE", "dram_sharded")
    use_packed_mlp = os.getenv("FALCON3_MULTICHIP_PACKED_MLP", "0") == "1"
    packed_mlp_unpack_mode = os.getenv("FALCON3_MULTICHIP_PACKED_UNPACK", "dram")
    decode_rope_mode = os.getenv("FALCON3_MULTICHIP_DECODE_ROPE_MODE", "dedicated")
    decode_output_mode = os.getenv("FALCON3_MULTICHIP_DECODE_OUTPUT_MODE", "direct_dram")
    use_persistent_decode_all_reduce = os.getenv("FALCON3_MULTICHIP_PERSISTENT_DECODE_AR", "1") == "1"
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=max_cache_len,
        precision_policy=precision_policy,
        qkv_target_cores=qkv_target_cores,
        o_target_cores=o_target_cores,
        gate_up_target_cores=gate_up_target_cores,
        down_target_cores=down_target_cores,
        prefill_grid_x=prefill_grid_x,
        prefill_in0_block_w=prefill_in0_block_w,
        ccl_dtype=ccl_dtype,
        num_links=num_links,
        topology=topology,
        decode_matmul_mode=decode_matmul_mode,
        use_packed_mlp=use_packed_mlp,
        packed_mlp_unpack_mode=packed_mlp_unpack_mode,
        decode_rope_mode=decode_rope_mode,
        decode_output_mode=decode_output_mode,
        use_persistent_decode_all_reduce=use_persistent_decode_all_reduce,
    )
    prefill, decode = _recorded_layer20_inputs(batch)
    decode_position = int(prefill.shape[1])
    key_cache, value_cache = model.allocate_kv_cache()
    tt_prefill = _mesh_input(prefill, mesh_device)
    prefill_warm = model.prefill_forward(
        tt_prefill,
        key_cache=key_cache,
        value_cache=value_cache,
    )
    ttnn.synchronize_device(mesh_device)
    prefill_warm.deallocate(True)
    prefill_samples = []
    for _ in range(5):
        start = time.perf_counter()
        prefill_output = model.prefill_forward(
            tt_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        ttnn.synchronize_device(mesh_device)
        prefill_samples.append((time.perf_counter() - start) * 1000)
        prefill_output.deallocate(True)
    prefill_ms = statistics.median(prefill_samples)
    tt_decode = _mesh_input(decode, mesh_device)
    position = _mesh_int32(torch.full((batch,), decode_position, dtype=torch.int32), mesh_device)

    def function():
        return model.decode_forward(
            tt_decode,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=position,
            position_index=decode_position,
        )

    warm, traced, trace_id, latency_ms, decode_samples = _trace_mesh_callable(
        mesh_device,
        function,
        samples=5,
        iterations=100,
    )
    baseline_path = (
        Path(__file__).parents[1] / "doc" / "optimized_decoder" / "results" / "final" / f"final_batch{batch}.json"
    )
    baseline_results = json.loads(baseline_path.read_text())["results"]
    baseline_key = "optimized_selected_dram_all_bfp4_auto"
    baseline = baseline_results[baseline_key]
    single_decode_ms = baseline["decode_ms"]
    single_prefill_ms = baseline.get("prefill_ms")
    speedup = single_decode_ms / latency_ms
    efficiency = speedup / TENSOR_PARALLEL_SIZE
    prefill_speedup = None if single_prefill_ms is None else single_prefill_ms / prefill_ms
    prefill_efficiency = None if prefill_speedup is None else prefill_speedup / TENSOR_PARALLEL_SIZE
    print(f"MULTICHIP_PREFILL_MS={prefill_ms:.9f}")
    print(f"MULTICHIP_TRACED_DECODE_MS={latency_ms:.9f}")
    print(f"MULTICHIP_DECODE_SPEEDUP={speedup:.9f}")
    _write_result_artifact(
        os.getenv("FALCON3_MULTICHIP_PERF_FILENAME", "final_batch32.json"),
        {
            "batch": batch,
            "sequence_length": decode_position,
            "weights": "real_layer_20",
            "qkv_target_cores": qkv_target_cores,
            "o_target_cores": o_target_cores,
            "gate_up_target_cores": gate_up_target_cores,
            "down_target_cores": down_target_cores,
            "prefill_grid_x": prefill_grid_x,
            "prefill_in0_block_w": prefill_in0_block_w,
            "ccl_dtype": ccl_dtype_name,
            "precision_policy": precision_policy,
            "num_links": num_links,
            "api_topology": topology_name,
            "decode_matmul_mode": decode_matmul_mode,
            "use_packed_mlp": use_packed_mlp,
            "packed_mlp_unpack_mode": packed_mlp_unpack_mode,
            "decode_rope_mode": decode_rope_mode,
            "decode_output_mode": decode_output_mode,
            "persistent_decode_all_reduce": use_persistent_decode_all_reduce,
            "prefill_samples_ms": prefill_samples,
            "multichip_prefill_ms": prefill_ms,
            "single_chip_prefill_ms": single_prefill_ms,
            "prefill_speedup": prefill_speedup,
            "prefill_parallel_efficiency": prefill_efficiency,
            "iterations_per_decode_sample": 100,
            "decode_samples_ms": decode_samples,
            "single_chip_baseline_ms": single_decode_ms,
            "single_chip_baseline_key": baseline_key,
            "single_chip_provenance": str(baseline_path.relative_to(REPO_ROOT)),
            "multichip_trace_ms": latency_ms,
            "speedup": speedup,
            "parallel_efficiency": efficiency,
        },
    )
    ttnn.release_trace(mesh_device, trace_id)
    for tensor in (warm, traced, tt_prefill, tt_decode, position, key_cache, value_cache):
        tensor.deallocate(True)
    _release_model(model)


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_STACK_CONTRACT") != "1",
    reason="manual two-layer residual-layout contract A/B",
)
@pytest.mark.parametrize("mesh_device,device_params", MESH_PARAMS, indirect=True)
@pytest.mark.timeout(1800)
def test_warmed_two_layer_residual_contract(mesh_device):
    """Compare the public DRAM boundary with a stack-native L1 residual boundary."""
    config = _config()
    batch = 32
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    decode_matmul_mode = os.getenv("FALCON3_MULTICHIP_DECODE_MATMUL_MODE", "dram_sharded")
    first_model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=64,
        decode_matmul_mode=decode_matmul_mode,
    )
    second_model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=64,
        decode_matmul_mode=decode_matmul_mode,
    )
    first_model._ensure_decode_all_reduce_resources()
    second_model.decode_all_reduce_resources = first_model.decode_all_reduce_resources
    second_model._ensure_decode_all_reduce_resources()
    prefill, decode = _recorded_layer20_inputs(batch)
    decode_position = int(prefill.shape[1])
    model_cache_pairs = [
        (first_model, first_model.allocate_kv_cache()),
        (second_model, second_model.allocate_kv_cache()),
        (first_model, first_model.allocate_kv_cache()),
        (second_model, second_model.allocate_kv_cache()),
    ]
    tt_prefill = _mesh_input(prefill, mesh_device)
    for model, (key_cache, value_cache) in model_cache_pairs:
        prefill_output = model.prefill_forward(
            tt_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        ttnn.synchronize_device(mesh_device)
        prefill_output.deallocate(True)
    tt_decode = _mesh_input(decode, mesh_device)
    position = _mesh_int32(torch.full((batch,), decode_position, dtype=torch.int32), mesh_device)

    def dram_boundary():
        first = first_model.decode_forward(
            tt_decode,
            key_cache=model_cache_pairs[0][1][0],
            value_cache=model_cache_pairs[0][1][1],
            cache_position=position,
            position_index=decode_position,
        )
        second = second_model.decode_forward(
            first,
            key_cache=model_cache_pairs[1][1][0],
            value_cache=model_cache_pairs[1][1][1],
            cache_position=position,
            position_index=decode_position,
        )
        first.deallocate(True)
        return second

    def sharded_boundary():
        first = first_model.decode_forward_to_residual(
            tt_decode,
            key_cache=model_cache_pairs[2][1][0],
            value_cache=model_cache_pairs[2][1][1],
            cache_position=position,
            position_index=decode_position,
        )
        second = second_model.decode_forward_from_residual(
            first,
            key_cache=model_cache_pairs[3][1][0],
            value_cache=model_cache_pairs[3][1][1],
            cache_position=position,
            position_index=decode_position,
        )
        return second_model.materialize_decode_output(second)

    baseline_warm, baseline_trace_output, baseline_trace_id, baseline_ms, baseline_samples = _trace_mesh_callable(
        mesh_device, dram_boundary, samples=5, iterations=100
    )
    baseline_host = _first_rank(baseline_warm)
    ttnn.release_trace(mesh_device, baseline_trace_id)
    baseline_warm.deallocate(True)
    baseline_trace_output.deallocate(True)

    sharded_warm, sharded_trace_output, sharded_trace_id, sharded_ms, sharded_samples = _trace_mesh_callable(
        mesh_device, sharded_boundary, samples=5, iterations=100
    )
    sharded_host = _first_rank(sharded_warm)
    pcc = comp_pcc(baseline_host.float(), sharded_host.float())[1]
    assert pcc >= REAL_WEIGHT_PCC
    _write_result_artifact(
        os.getenv("FALCON3_MULTICHIP_STACK_FILENAME", "two_layer_residual_contract.json"),
        {
            "batch": batch,
            "sequence_length": decode_position,
            "weights": "real_layer_20_independently_materialized_for_two_decoder_instances",
            "decode_matmul_mode": decode_matmul_mode,
            "layers_per_replay": 2,
            "iterations_per_sample": 100,
            "dram_boundary_samples_ms": baseline_samples,
            "dram_boundary_ms": baseline_ms,
            "sharded_boundary_samples_ms": sharded_samples,
            "sharded_boundary_ms": sharded_ms,
            "speedup": baseline_ms / sharded_ms,
            "pcc": pcc,
            "inter_layer_collectives": 0,
            "residual_contract": (
                "replicated values, per-device L1 width-sharded [1,1,32,3072] "
                f"on {first_model.residual_num_cores} cores"
            ),
            "persistent_all_reduce_pool": "one shared owner-managed pool across both decoder instances",
        },
    )
    ttnn.release_trace(mesh_device, sharded_trace_id)
    sharded_warm.deallocate(True)
    sharded_trace_output.deallocate(True)
    for tensor in (tt_prefill, tt_decode, position):
        tensor.deallocate(True)
    for _, (key_cache, value_cache) in model_cache_pairs:
        key_cache.deallocate(True)
        value_cache.deallocate(True)
    _release_model(second_model)
    _release_model(first_model)


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_PERSISTENT_AR") != "1",
    reason="manual persistent-buffer all-reduce A/B",
)
@pytest.mark.parametrize("mesh_device,device_params", MESH_PARAMS, indirect=True)
@pytest.mark.timeout(1800)
def test_persistent_all_reduce_residual_probe(mesh_device):
    """A/B the exact decode residual all-reduce shape and layout."""
    torch.manual_seed(20260718)
    rows, hidden = 32, 3072
    residual_grid = ttnn.CoreGrid(x=8, y=4)
    residual_memory_config = ttnn.create_sharded_memory_config(
        shape=(rows, hidden // residual_grid.num_cores),
        core_grid=residual_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    host = torch.randn((1, 1, rows, TENSOR_PARALLEL_SIZE * hidden), dtype=torch.bfloat16)
    residual = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=residual_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    all_cores = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(
                    mesh_device.compute_with_storage_grid_size().x - 1,
                    mesh_device.compute_with_storage_grid_size().y - 1,
                ),
            )
        }
    )
    sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([all_cores])], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([sub_device_id])
    semaphore = ttnn.create_global_semaphore(mesh_device, all_cores, 0)
    intermediate_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            residual_memory_config.shard_spec.grid,
            (rows, TENSOR_PARALLEL_SIZE * hidden // residual_grid.num_cores),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    persistent_buffer = ttnn.zeros(
        (1, 1, rows, TENSOR_PARALLEL_SIZE * hidden),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=intermediate_memory_config,
    )

    def baseline():
        return ttnn.all_reduce(
            residual,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            num_links=2,
            memory_config=residual_memory_config,
            subdevice_id=sub_device_id,
        )

    def persistent():
        return ttnn.experimental.all_reduce_async(
            residual,
            persistent_buffer,
            cluster_axis=1,
            mesh_device=mesh_device,
            multi_device_global_semaphore=semaphore,
            memory_config=residual_memory_config,
            dtype=ttnn.bfloat16,
            topology=ttnn.Topology.Ring,
            num_links=2,
            subdevice_id=sub_device_id,
        )

    try:
        baseline_warm, baseline_trace_output, baseline_trace_id, baseline_ms, baseline_samples = _trace_mesh_callable(
            mesh_device, baseline, samples=5, iterations=100
        )
        baseline_host = _first_rank(baseline_warm)
        ttnn.release_trace(mesh_device, baseline_trace_id)
        baseline_warm.deallocate(True)
        baseline_trace_output.deallocate(True)
        (
            persistent_warm,
            persistent_trace_output,
            persistent_trace_id,
            persistent_ms,
            persistent_samples,
        ) = _trace_mesh_callable(mesh_device, persistent, samples=5, iterations=100)
        persistent_host = _first_rank(persistent_warm)
        pcc = comp_pcc(baseline_host.float(), persistent_host.float())[1]
        assert pcc >= 0.9999
        _write_result_artifact(
            "persistent_all_reduce_probe.json",
            {
                "shape_per_rank": [1, 1, rows, hidden],
                "residual_layout": "L1 width-sharded, 8x4 cores, shard [32,96]",
                "topology": "ring",
                "num_links": 2,
                "dtype": "bf16",
                "iterations_per_sample": 100,
                "default_samples_ms": baseline_samples,
                "default_ms": baseline_ms,
                "persistent_samples_ms": persistent_samples,
                "persistent_ms": persistent_ms,
                "speedup": baseline_ms / persistent_ms,
                "pcc": pcc,
            },
        )
        ttnn.release_trace(mesh_device, persistent_trace_id)
        persistent_warm.deallocate(True)
        persistent_trace_output.deallocate(True)
    finally:
        residual.deallocate(True)
        persistent_buffer.deallocate(True)
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
        gc.collect()


@pytest.mark.skipif(os.getenv("FALCON3_RUN_MULTICHIP_PROFILE") != "1", reason="manual Tracy profile")
@pytest.mark.parametrize("mesh_device,device_params", MESH_PARAMS, indirect=True)
@pytest.mark.timeout(1800)
def test_profile_selected_multichip_decoder(mesh_device):
    """Profile the selected TP4 layer with separate prefill/decode regions."""
    from tracy import signpost

    config = _config()
    batch, seq_len = 32, 17
    precision_policy = os.getenv("FALCON3_MULTICHIP_PRECISION_POLICY", "all_bfp4_lofi")
    decode_matmul_mode = os.getenv("FALCON3_MULTICHIP_DECODE_MATMUL_MODE", "dram_sharded")
    use_packed_mlp = os.getenv("FALCON3_MULTICHIP_PACKED_MLP", "0") == "1"
    packed_mlp_unpack_mode = os.getenv("FALCON3_MULTICHIP_PACKED_UNPACK", "dram")
    decode_rope_mode = os.getenv("FALCON3_MULTICHIP_DECODE_ROPE_MODE", "dedicated")
    decode_output_mode = os.getenv("FALCON3_MULTICHIP_DECODE_OUTPUT_MODE", "direct_dram")
    use_persistent_decode_all_reduce = os.getenv("FALCON3_MULTICHIP_PERSISTENT_DECODE_AR", "1") == "1"
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=64,
        precision_policy=precision_policy,
        decode_matmul_mode=decode_matmul_mode,
        use_packed_mlp=use_packed_mlp,
        packed_mlp_unpack_mode=packed_mlp_unpack_mode,
        decode_rope_mode=decode_rope_mode,
        decode_output_mode=decode_output_mode,
        use_persistent_decode_all_reduce=use_persistent_decode_all_reduce,
    )
    hidden, decode_hidden = _recorded_layer20_inputs(batch)
    key_cache, value_cache = model.allocate_kv_cache()
    tt_prefill = _mesh_input(hidden, mesh_device)
    warm_prefill = model.prefill_forward(
        tt_prefill,
        key_cache=key_cache,
        value_cache=value_cache,
    )
    ttnn.synchronize_device(mesh_device)
    warm_prefill.deallocate(True)
    ttnn.ReadDeviceProfiler(mesh_device)
    prefill_iterations = 2
    signpost(header="MULTICHIP_PREFILL")
    prefill_start = time.perf_counter()
    for _ in range(prefill_iterations):
        profile_prefill = model.prefill_forward(
            tt_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        ttnn.synchronize_device(mesh_device)
        profile_prefill.deallocate(True)
    prefill_wall_total_ms = (time.perf_counter() - prefill_start) * 1000.0
    signpost(header="MULTICHIP_PREFILL_END")
    ttnn.ReadDeviceProfiler(mesh_device)

    tt_decode = _mesh_input(decode_hidden, mesh_device)
    position = _mesh_int32(torch.full((batch,), seq_len, dtype=torch.int32), mesh_device)

    def decode():
        return model.decode_forward(
            tt_decode,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=position,
            position_index=seq_len,
        )

    warm_decode = decode()
    ttnn.synchronize_device(mesh_device)
    warm_decode.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decode()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    decode_iterations = 3
    try:
        signpost(header="MULTICHIP_DECODE")
        decode_start = time.perf_counter()
        for _ in range(decode_iterations):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        decode_wall_total_ms = (time.perf_counter() - decode_start) * 1000.0
        signpost(header="MULTICHIP_DECODE_END")
        ttnn.ReadDeviceProfiler(mesh_device)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    _write_result_artifact(
        os.getenv("FALCON3_MULTICHIP_PROFILE_FILENAME", "profile_wall.json"),
        {
            "batch": batch,
            "sequence_length": seq_len,
            "weights": "real_layer_20",
            "precision_policy": precision_policy,
            "decode_matmul_mode": decode_matmul_mode,
            "use_packed_mlp": use_packed_mlp,
            "packed_mlp_unpack_mode": packed_mlp_unpack_mode,
            "decode_rope_mode": decode_rope_mode,
            "decode_output_mode": decode_output_mode,
            "persistent_decode_all_reduce": use_persistent_decode_all_reduce,
            "prefill_iterations": prefill_iterations,
            "prefill_wall_total_ms": prefill_wall_total_ms,
            "prefill_wall_ms": prefill_wall_total_ms / prefill_iterations,
            "decode_iterations": decode_iterations,
            "decode_wall_total_ms": decode_wall_total_ms,
            "decode_wall_ms": decode_wall_total_ms / decode_iterations,
        },
    )
    for tensor in (trace_output, tt_prefill, tt_decode, position, key_cache, value_cache):
        tensor.deallocate(True)
    _release_model(model)


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_RS_PROBE") != "1",
    reason="manual AutoFix isolation of the fused output-projection reduce-scatter",
)
@pytest.mark.parametrize("mesh_device,device_params", MESH_PARAMS, indirect=True)
@pytest.mark.timeout(1800)
def test_fused_rs_workspace_probe(mesh_device):
    """Isolate fused RS from the distributed norm and following AG-matmul."""
    torch.manual_seed(20260717)
    rows, hidden = 32, 3072
    local_hidden = hidden // TENSOR_PARALLEL_SIZE
    num_links = int(os.getenv("FALCON3_RS_PROBE_LINKS", "2"))
    workspace_batch = int(os.getenv("FALCON3_RS_PROBE_WORKSPACE_BATCH", "1"))
    assert num_links in (1, 2)
    assert workspace_batch in (1, TENSOR_PARALLEL_SIZE)
    topology = ttnn.Topology.Ring
    compute_config = _compute_config(mesh_device, ttnn.MathFidelity.LoFi)
    full_input_host = torch.randn((1, 1, rows, hidden), dtype=torch.bfloat16)
    o_weight_host = torch.randn((1, 1, hidden, hidden), dtype=torch.bfloat16)
    local_input = ttnn.from_torch(
        full_input_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    o_weight = ttnn.from_torch(
        o_weight_host,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    o_program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 2),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=12,
        out_block_w=12,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([all_cores])], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([sub_device_id])
    ccl = TT_CCL(mesh_device)

    def persistent(shape):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    standalone_intermediate = persistent((1, 1, rows, hidden))
    standalone_output = persistent((1, 1, rows, local_hidden))
    fused_intermediate = persistent((workspace_batch, 1, rows, hidden))
    fused_output = persistent((1, 1, rows, local_hidden))
    standalone_partial = fused_partial = None
    standalone_reduced = fused_reduced = None
    try:
        standalone_partial = ttnn.matmul(
            local_input,
            o_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=o_program,
            compute_kernel_config=compute_config,
        )
        standalone_reduced = ttnn.experimental.reduce_scatter_minimal_async(
            standalone_partial,
            persistent_output_buffers=[standalone_intermediate, standalone_output],
            dim=3,
            multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(),
            num_links=num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            subdevice_id=sub_device_id,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_device_id])
        print("STANDALONE_RS_PASS", flush=True)

        fused_partial, fused_reduced = ttnn.experimental.matmul_reduce_scatter_async(
            local_input,
            o_weight,
            persistent_intermediate_buffer=fused_intermediate,
            persistent_output_buffer=fused_output,
            dim=3,
            multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(),
            reduce_scatter_core_grid_offset=(0, 6),
            num_links=num_links,
            memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            subdevice_id=sub_device_id,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            program_config=o_program,
            compute_kernel_config=compute_config,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_device_id])
        print("FUSED_RS_PASS", flush=True)

        for rank in ttnn.get_device_tensors(fused_partial):
            assert tuple(rank.shape) == (1, 1, rows, hidden)
        for rank in ttnn.get_device_tensors(fused_reduced):
            assert tuple(rank.shape) == (1, 1, rows, local_hidden)
        standalone_partial_host = ttnn.to_torch(
            standalone_partial,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
        )
        fused_partial_host = ttnn.to_torch(
            fused_partial,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
        )
        standalone_reduced_host = ttnn.to_torch(
            standalone_reduced,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
        )
        fused_reduced_host = ttnn.to_torch(
            fused_reduced,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
        )
        partial_passed, partial_pcc = comp_pcc(
            standalone_partial_host.float(),
            fused_partial_host.float(),
            pcc=0.99,
        )
        reduced_passed, reduced_pcc = comp_pcc(
            standalone_reduced_host.float(),
            fused_reduced_host.float(),
            pcc=0.99,
        )
        assert partial_passed, f"fused matmul partial PCC={partial_pcc}"
        assert reduced_passed, f"fused reduce-scatter PCC={reduced_pcc}"
        _write_result_artifact(
            f"autofix_rs_probe_links{num_links}_workspace_batch{workspace_batch}.json",
            {
                "num_links": num_links,
                "topology": "ring",
                "workspace_shape": [workspace_batch, 1, rows, hidden],
                "expected_workspace_shape": [1, 1, rows, hidden],
                "local_output_shape": [1, 1, rows, local_hidden],
                "standalone_rs_completed": True,
                "fused_rs_completed": True,
                "matmul_partial_pcc_vs_standalone": partial_pcc,
                "reduce_scatter_pcc_vs_standalone": reduced_pcc,
            },
        )
    finally:
        # The returned reduce-scatter tensors alias their persistent outputs.
        for tensor in (
            standalone_partial,
            fused_partial,
            local_input,
            o_weight,
            standalone_intermediate,
            standalone_output,
            fused_intermediate,
            fused_output,
        ):
            if tensor is not None:
                tensor.deallocate(True)
        del standalone_reduced, fused_reduced
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
        gc.collect()


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_SHARDED_BOUNDARY") != "1",
    reason="manual exact lower-data-movement graph-rewrite candidate",
)
@pytest.mark.parametrize("mesh_device,device_params", MESH_PARAMS, indirect=True)
@pytest.mark.timeout(1800)
def test_reduce_scatter_distributed_norm_all_gather_matmul_boundary(mesh_device):
    """Compare the exact O/residual/RMSNorm/QKV boundary before changing the model graph.

    The production graph all-reduces the row-parallel O result and keeps the
    residual replicated.  The candidate reduce-scatters O, performs the add and
    distributed RMSNorm on hidden/TP shards, then fuses the restoring all-gather
    with the following column-parallel QKV matmul.
    """
    torch.manual_seed(20260717)
    rows, hidden, qkv_width = 32, 3072, 5120
    local_hidden = hidden // TENSOR_PARALLEL_SIZE
    local_qkv_width = qkv_width // TENSOR_PARALLEL_SIZE
    topology = ttnn.Topology.Ring
    rs_mode = os.getenv("FALCON3_MULTICHIP_BOUNDARY_RS_MODE", "standalone").lower()
    ag_mode = os.getenv("FALCON3_MULTICHIP_BOUNDARY_AG_MODE", "standalone").lower()
    assert rs_mode in ("standalone", "fused")
    assert ag_mode in ("standalone", "fused")
    compute_config = _compute_config(mesh_device, ttnn.MathFidelity.LoFi)

    full_input_host = torch.randn((1, 1, rows, hidden), dtype=torch.bfloat16)
    o_weight_host = torch.randn((1, 1, hidden, hidden), dtype=torch.bfloat16)
    qkv_weight_host = torch.randn((1, 1, hidden, qkv_width), dtype=torch.bfloat16)
    gamma_flat_host = torch.ones((1, 1, 1, hidden), dtype=torch.bfloat16)
    local_input = ttnn.from_torch(
        full_input_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    replicated_input = ttnn.from_torch(
        full_input_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    o_weight = ttnn.from_torch(
        o_weight_host,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    qkv_weight = ttnn.from_torch(
        qkv_weight_host,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    gamma_replicated = ttnn.from_torch(
        gamma_flat_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    gamma_local = ttnn.from_torch(
        gamma_flat_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    o_program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 2),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=12,
        out_block_w=12,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    qkv_program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=5,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([all_cores])], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([sub_device_id])
    ccl = TT_CCL(mesh_device)
    rs_intermediate = ttnn.from_torch(
        torch.zeros((1, 1, rows, hidden), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rs_output = ttnn.from_torch(
        torch.zeros((1, 1, rows, local_hidden), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ag_output = ttnn.from_torch(
        torch.zeros_like(full_input_host),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def baseline_boundary():
        partial = ttnn.matmul(
            local_input,
            o_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=o_program,
            compute_kernel_config=compute_config,
        )
        reduced = ttnn.all_reduce(
            partial,
            cluster_axis=1,
            num_links=2,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        added = ttnn.add(replicated_input, reduced, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        normed = ttnn.rms_norm(
            added,
            epsilon=1e-6,
            weight=gamma_replicated,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )
        output = ttnn.matmul(
            normed,
            qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=qkv_program,
            compute_kernel_config=compute_config,
        )
        for tensor in (partial, reduced, added, normed):
            tensor.deallocate()
        return output

    def candidate_boundary():
        if rs_mode == "fused":
            partial, local_projected = ttnn.experimental.matmul_reduce_scatter_async(
                local_input,
                o_weight,
                persistent_intermediate_buffer=rs_intermediate,
                persistent_output_buffer=rs_output,
                dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(),
                reduce_scatter_core_grid_offset=(0, 6),
                num_links=2,
                memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                subdevice_id=sub_device_id,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                program_config=o_program,
                compute_kernel_config=compute_config,
            )
        else:
            partial = ttnn.matmul(
                local_input,
                o_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=o_program,
                compute_kernel_config=compute_config,
            )
            local_projected = ttnn.experimental.reduce_scatter_minimal_async(
                partial,
                persistent_output_buffers=[rs_intermediate, rs_output],
                dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(),
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                subdevice_id=sub_device_id,
            )
        local_added = ttnn.add(local_input, local_projected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        stats = ttnn.rms_norm_pre_all_gather(
            local_added,
            compute_kernel_config=compute_config,
            dtype=ttnn.bfloat16,
        )
        gathered_stats = ttnn.experimental.all_gather_async(
            stats,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=2,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
            subdevice_id=sub_device_id,
        )
        local_normed = ttnn.rms_norm_post_all_gather(
            local_added,
            gathered_stats,
            epsilon=1e-6,
            weight=gamma_local,
            compute_kernel_config=compute_config,
        )
        if ag_mode == "fused":
            gathered_norm, output = ttnn.experimental.all_gather_matmul_async(
                local_normed,
                qkv_weight,
                persistent_output_buffer=ag_output,
                dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
                all_gather_core_grid_offset=(0, 6),
                barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=2,
                memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                subdevice_id=sub_device_id,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                program_config=qkv_program,
                compute_kernel_config=compute_config,
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
        else:
            gathered_norm = ttnn.experimental.all_gather_async(
                local_normed,
                persistent_output_buffer=ag_output,
                dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=2,
                topology=topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
                subdevice_id=sub_device_id,
            )
            output = ttnn.matmul(
                gathered_norm,
                qkv_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=qkv_program,
                compute_kernel_config=compute_config,
            )
        for tensor in (partial, local_added, stats, gathered_stats, local_normed):
            tensor.deallocate()
        del local_projected, gathered_norm
        return output

    baseline_warm = baseline_trace = fused_warm = fused_trace = None
    baseline_trace_id = fused_trace_id = None
    try:
        baseline_warm, baseline_trace, baseline_trace_id, baseline_ms, baseline_samples = _trace_mesh_callable(
            mesh_device,
            baseline_boundary,
            samples=20,
            iterations=1,
        )
        ttnn.release_trace(mesh_device, baseline_trace_id)
        baseline_trace_id = None
        fused_warm, fused_trace, fused_trace_id, fused_ms, fused_samples = _trace_mesh_callable(
            mesh_device,
            candidate_boundary,
            samples=20,
            iterations=1,
        )
        ttnn.release_trace(mesh_device, fused_trace_id)
        fused_trace_id = None
        baseline_host = ttnn.to_torch(
            baseline_trace,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
        )
        fused_host = ttnn.to_torch(
            fused_trace,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
        )
        passed, pcc = comp_pcc(baseline_host.float(), fused_host.float(), pcc=0.99)
        assert passed, f"fused boundary PCC={pcc}"
        speedup = baseline_ms / fused_ms
        selected = fused_ms < baseline_ms
        _write_result_artifact(
            f"graph_rewrite_{rs_mode}_rs_{ag_mode}_ag_boundary.json",
            {
                "boundary": "O -> residual add -> RMSNorm -> QKV",
                "logical_input_shape": [1, 1, rows, hidden],
                "local_hidden_shape": [1, 1, rows, local_hidden],
                "local_qkv_output_shape": [1, 1, rows, local_qkv_width],
                "baseline": "matmul + ring all-reduce + replicated add/RMSNorm + matmul",
                "reduce_scatter_producer": rs_mode,
                "all_gather_consumer": ag_mode,
                "candidate": f"{rs_mode} matmul/reduce-scatter + sharded add/distributed RMSNorm + {ag_mode} all-gather/matmul",
                "baseline_trace_samples_ms": baseline_samples,
                "candidate_trace_samples_ms": fused_samples,
                "baseline_trace_ms": baseline_ms,
                "candidate_trace_ms": fused_ms,
                "candidate_speedup": speedup,
                "pcc_vs_baseline": pcc,
                "candidate_selected": selected,
                "decision": "integrate candidate" if selected else "retain replicated all-reduce graph",
            },
        )
        print(f"FUSED_BOUNDARY_BASELINE_MS={baseline_ms:.9f}")
        print(f"FUSED_BOUNDARY_CANDIDATE_MS={fused_ms:.9f}")
        print(f"FUSED_BOUNDARY_SPEEDUP={speedup:.9f}")
        print(f"FUSED_BOUNDARY_PCC={pcc}")
    finally:
        if baseline_trace_id is not None:
            ttnn.release_trace(mesh_device, baseline_trace_id)
        if fused_trace_id is not None:
            ttnn.release_trace(mesh_device, fused_trace_id)
        for tensor in (
            baseline_warm,
            baseline_trace,
            fused_warm,
            fused_trace,
            local_input,
            replicated_input,
            o_weight,
            qkv_weight,
            gamma_replicated,
            gamma_local,
            rs_intermediate,
            rs_output,
            ag_output,
        ):
            if tensor is not None:
                tensor.deallocate(True)
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
        gc.collect()
