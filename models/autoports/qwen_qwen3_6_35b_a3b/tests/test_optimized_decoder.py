# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
import math
import os
import textwrap
import time

import pytest
import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_functional_decoder import (
    BLOCK_SIZE,
    _assert_pcc,
    _causal_mask,
    _load_real_layer_state,
    _page_table,
    _randn,
    _rotary,
    _signpost,
    _state_for_perf,
    _synthetic_layer_state,
    _target_text_config,
    _to_torch,
    _torch_layer,
    _tt_bf16,
    _tt_int,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tt import optimized_decoder as optimized_decoder_module
from models.autoports.qwen_qwen3_6_35b_a3b.tt.functional_decoder import TILE_SIZE, _slice
from models.autoports.qwen_qwen3_6_35b_a3b.tt.optimized_decoder import (
    DEFAULT_OPTIMIZED_POLICY,
    OptimizedDecoder,
    OptimizedDecoderPolicy,
)


def _optimized_policy_from_name(name: str | None) -> OptimizedDecoderPolicy:
    policy_name = name or "default"
    if policy_name == "default":
        return DEFAULT_OPTIMIZED_POLICY
    if policy_name == "routed_bfp4":
        return OptimizedDecoderPolicy(routed_moe_weight_dtype=ttnn.bfloat4_b)
    if policy_name == "routed_bfp4_exact_nnz":
        return OptimizedDecoderPolicy(routed_moe_weight_dtype=ttnn.bfloat4_b, use_decode_exact_nnz=True)
    if policy_name == "moe_all_bfp4":
        return OptimizedDecoderPolicy(shared_moe_weight_dtype=ttnn.bfloat4_b, routed_moe_weight_dtype=ttnn.bfloat4_b)
    if policy_name == "decode_l1_sparse_inputs":
        return OptimizedDecoderPolicy(use_decode_l1_sparse_inputs=True)
    if policy_name == "decode_l1_sparse_inputs_exact_nnz":
        return OptimizedDecoderPolicy(use_decode_l1_sparse_inputs=True, use_decode_exact_nnz=True)
    if policy_name == "prefill_l1_sparse_inputs":
        return OptimizedDecoderPolicy(use_prefill_l1_sparse_inputs=True)
    if policy_name == "prefill_l1_sparse_inputs_exact_nnz":
        return OptimizedDecoderPolicy(use_prefill_l1_sparse_inputs=True, use_decode_exact_nnz=True)
    if policy_name == "decode_sdpa_k64":
        return OptimizedDecoderPolicy(use_decode_sdpa_program_config=True, decode_sdpa_k_chunk_size=64)
    if policy_name == "sparse_in0_block_w2":
        return OptimizedDecoderPolicy(sparse_in0_block_w=2)
    if policy_name == "sparse_in0_block_w4":
        return OptimizedDecoderPolicy(sparse_in0_block_w=4)
    if policy_name == "sparse_in0_block_w4_exact_nnz":
        return OptimizedDecoderPolicy(sparse_in0_block_w=4, use_decode_exact_nnz=True)
    if policy_name == "sparse_cores16_out2":
        return OptimizedDecoderPolicy(sparse_core_count_cap=16, sparse_out_subblock_w=2)
    if policy_name == "sparse_cores16_out2_exact_nnz":
        return OptimizedDecoderPolicy(
            sparse_core_count_cap=16,
            sparse_out_subblock_w=2,
            use_decode_exact_nnz=True,
        )
    if policy_name == "decode_exact_nnz":
        return OptimizedDecoderPolicy(use_decode_exact_nnz=True)
    raise ValueError(f"unknown Qwen3.6 optimized decoder policy: {policy_name}")


def _optimized_policy_from_env() -> OptimizedDecoderPolicy:
    return _optimized_policy_from_name(os.environ.get("QWEN36_OPTIMIZED_POLICY", "default"))


def _make_optimized_decoder(state, *, cfg, layer_idx: int, device, policy: OptimizedDecoderPolicy | None = None):
    return OptimizedDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=device,
        policy=policy or _optimized_policy_from_env(),
    )


def _prepare_optimized_decode_after_prefill(device, cfg, tt_layer: OptimizedDecoder, layer_idx: int, seq_len: int):
    batch = 1
    decode_hidden = _randn((batch, 1, cfg.hidden_size), seed=2800 + layer_idx + seq_len, scale=0.01)
    decode_input = _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device)
    current_pos = _tt_int(torch.tensor([seq_len], dtype=torch.int32), device)

    if cfg.layer_types[layer_idx] == "full_attention":
        max_seq_len = max(96, ((seq_len + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE)
        hidden = _randn((batch, seq_len, cfg.hidden_size), seed=2600 + seq_len, scale=0.01)
        position_ids = torch.arange(seq_len, dtype=torch.long).reshape(batch, seq_len)
        position_embeddings = _rotary(cfg, hidden, position_ids)
        page_table = _page_table(batch, max_seq_len)
        kv_cache = OptimizedDecoder.allocate_full_attention_cache(
            hf_config=cfg,
            mesh_device=device,
            max_batch_size=batch,
            max_seq_len=max_seq_len,
            block_size=BLOCK_SIZE,
        )
        tt_layer.prefill_forward(
            _tt_bf16(hidden.unsqueeze(0), device),
            position_embeddings=(
                _tt_bf16(position_embeddings[0].unsqueeze(1), device),
                _tt_bf16(position_embeddings[1].unsqueeze(1), device),
            ),
            page_table=_tt_int(page_table, device),
            kv_cache=kv_cache,
        )
        decode_position_embeddings = _rotary(cfg, decode_hidden, torch.tensor([[seq_len]], dtype=torch.long))
        kwargs = {
            "current_pos": current_pos,
            "position_embeddings": (
                _tt_bf16(decode_position_embeddings[0].unsqueeze(0), device),
                _tt_bf16(decode_position_embeddings[1].unsqueeze(0), device),
            ),
            "page_table": _tt_int(page_table, device),
            "kv_cache": kv_cache,
        }
    else:
        state = OptimizedDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=batch)
        hidden = _randn((batch, seq_len, cfg.hidden_size), seed=2700 + seq_len, scale=0.01)
        tt_prefill = tt_layer.prefill_forward(_tt_bf16(hidden.unsqueeze(0), device), linear_state=state)
        kwargs = {"current_pos": current_pos, "linear_state": tt_prefill.linear_state}
    return decode_input, kwargs


def _run_optimized_traced_decode(device, tt_layer: OptimizedDecoder, decode_input: ttnn.Tensor, decode_kwargs: dict):
    tt_layer.decode_forward(decode_input, **decode_kwargs)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.release_trace(device, trace_id)
    ttnn.synchronize_device(device)
    return traced


def _run_optimized_prefill_decode_parity(
    device,
    *,
    cfg,
    layer_idx: int,
    state: dict[str, torch.Tensor],
    seq_len: int,
    batch: int = 1,
    trace_decode: bool = False,
    policy: OptimizedDecoderPolicy | None = None,
):
    max_seq_len = max(96, ((seq_len + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE)
    hidden = _randn((batch, seq_len, cfg.hidden_size), seed=2200 + batch + layer_idx + seq_len, scale=0.01)
    decode_hidden = _randn((batch, 1, cfg.hidden_size), seed=2300 + batch + layer_idx + seq_len, scale=0.01)

    layer = _torch_layer(cfg, layer_idx, state)
    tt_layer = _make_optimized_decoder(state, cfg=cfg, layer_idx=layer_idx, device=device, policy=policy)
    assert type(tt_layer).__name__ == "OptimizedDecoder"

    if cfg.layer_types[layer_idx] == "full_attention":
        cache = DynamicCache()
        pos = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len).expand(batch, seq_len)
        position_embeddings = _rotary(cfg, hidden, pos)
        with torch.no_grad():
            ref_prefill = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=_causal_mask(batch, seq_len),
                past_key_values=cache,
            )
        page_table = _page_table(batch, max_seq_len)
        kv_cache = OptimizedDecoder.allocate_full_attention_cache(
            hf_config=cfg,
            mesh_device=device,
            max_batch_size=batch,
            max_seq_len=max_seq_len,
            block_size=BLOCK_SIZE,
        )
        tt_prefill = tt_layer.prefill_forward(
            _tt_bf16(hidden.unsqueeze(0), device),
            position_embeddings=(
                _tt_bf16(position_embeddings[0].unsqueeze(1), device),
                _tt_bf16(position_embeddings[1].unsqueeze(1), device),
            ),
            page_table=_tt_int(page_table, device),
            kv_cache=kv_cache,
        ).hidden_states
        prefill_msg = _assert_pcc("optimized full_attention prefill", ref_prefill, _to_torch(tt_prefill).squeeze(0))

        current_pos = torch.full((batch,), seq_len, dtype=torch.int32)
        decode_pos = current_pos.to(torch.long).reshape(batch, 1)
        decode_position_embeddings = _rotary(cfg, decode_hidden, decode_pos)
        with torch.no_grad():
            ref_decode = layer(
                decode_hidden,
                position_embeddings=decode_position_embeddings,
                attention_mask=None,
                past_key_values=cache,
            )
        decode_input = _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device)
        decode_kwargs = {
            "position_embeddings": (
                _tt_bf16(decode_position_embeddings[0].unsqueeze(0), device),
                _tt_bf16(decode_position_embeddings[1].unsqueeze(0), device),
            ),
            "page_table": _tt_int(page_table, device),
            "kv_cache": kv_cache,
            "current_pos": _tt_int(current_pos, device),
        }
        tt_decode = (
            _run_optimized_traced_decode(device, tt_layer, decode_input, decode_kwargs)
            if trace_decode
            else tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
        )
        decode_msg = _assert_pcc(
            "optimized full_attention decode",
            ref_decode,
            _to_torch(tt_decode).squeeze(0).transpose(0, 1),
        )
        return prefill_msg, decode_msg

    from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_functional_decoder import _LinearCache

    linear_cache = _LinearCache(layer_idx)
    dummy_pos = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len).expand(batch, seq_len)
    dummy_position_embeddings = _rotary(cfg, hidden, dummy_pos)
    with torch.no_grad():
        ref_prefill = layer(
            hidden,
            position_embeddings=dummy_position_embeddings,
            attention_mask=None,
            past_key_values=linear_cache,
        )
    linear_state = OptimizedDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=batch)
    tt_prefill_result = tt_layer.prefill_forward(_tt_bf16(hidden.unsqueeze(0), device), linear_state=linear_state)
    prefill_msg = _assert_pcc(
        "optimized linear_attention prefill", ref_prefill, _to_torch(tt_prefill_result.hidden_states).squeeze(0)
    )

    decode_position_embeddings = _rotary(cfg, decode_hidden, torch.full((batch, 1), seq_len, dtype=torch.long))
    with torch.no_grad():
        ref_decode = layer(
            decode_hidden,
            position_embeddings=decode_position_embeddings,
            attention_mask=None,
            past_key_values=linear_cache,
        )
    decode_input = _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device)
    decode_kwargs = {
        "current_pos": _tt_int(torch.full((batch,), seq_len, dtype=torch.int32), device),
        "linear_state": tt_prefill_result.linear_state,
    }
    tt_decode = (
        _run_optimized_traced_decode(device, tt_layer, decode_input, decode_kwargs)
        if trace_decode
        else tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    )
    decode_msg = _assert_pcc(
        "optimized linear_attention decode",
        ref_decode,
        _to_torch(tt_decode).squeeze(0).transpose(0, 1),
    )
    return prefill_msg, decode_msg


def test_optimized_decoder_graph_summary():
    summary = OptimizedDecoder.graph_summary
    assert summary.fused_graph.packed_attention_qkgv
    assert summary.fused_graph.packed_linear_attention_inputs
    assert summary.fused_graph.packed_shared_expert_gate_up
    assert summary.fused_graph.packed_routed_expert_gate_up
    assert summary.bf16_norms_and_residuals
    assert summary.reduced_attention_weights
    assert summary.reduced_linear_attention_weights
    assert summary.reduced_moe_weights


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(("layer_idx", "seq_len"), [(0, 5), (3, 33)], ids=["linear_seq5", "full_seq33"])
def test_synthetic_optimized_decoder_prefill_decode_against_hf(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    prefill_msg, decode_msg = _run_optimized_prefill_decode_parity(
        device,
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        trace_decode=True,
    )
    print(f"optimized prefill {prefill_msg}")
    print(f"optimized traced decode {decode_msg}")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(("layer_idx", "seq_len"), [(0, 5), (3, 33)], ids=["linear_batch2", "full_batch2"])
def test_synthetic_optimized_decoder_batch2_prefill_decode_against_hf(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    prefill_msg, decode_msg = _run_optimized_prefill_decode_parity(
        device,
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        batch=2,
        trace_decode=True,
    )
    print(f"optimized batch2 prefill {prefill_msg}")
    print(f"optimized batch2 traced decode {decode_msg}")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "seq_len"), [(0, 65), (3, 33)], ids=["linear_non_aligned65", "full_non_aligned33"]
)
def test_synthetic_optimized_decoder_non_aligned_lengths(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    prefill_msg, decode_msg = _run_optimized_prefill_decode_parity(
        device,
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        trace_decode=True,
    )
    print(f"optimized non-aligned prefill {prefill_msg}")
    print(f"optimized non-aligned traced decode {decode_msg}")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(("layer_idx", "seq_len"), [(0, 5), (3, 33)], ids=["linear_repeat", "full_repeat"])
def test_synthetic_optimized_decoder_repeated_input_determinism(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    tt_layer = _make_optimized_decoder(state, cfg=cfg, layer_idx=layer_idx, device=device)
    decode_input, decode_kwargs = _prepare_optimized_decode_after_prefill(device, cfg, tt_layer, layer_idx, seq_len)

    first = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.synchronize_device(device)
    second = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.synchronize_device(device)
    msg = _assert_pcc(
        f"optimized {cfg.layer_types[layer_idx]} repeated decode", _to_torch(first), _to_torch(second), pcc=0.9999
    )
    print(f"optimized repeated decode {msg}")


@pytest.mark.real_weights
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_REAL_WEIGHTS") != "1", reason="set RUN_QWEN36_REAL_WEIGHTS=1 to load checkpoint weights"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "seq_len"),
    [(0, 1), (3, 1), (0, 5), (3, 5)],
    ids=["real_linear_layer0_seq1", "real_full_layer3_seq1", "real_linear_layer0_seq5", "real_full_layer3_seq5"],
)
def test_real_weight_optimized_decoder_prefill_decode_against_hf(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _load_real_layer_state(layer_idx)
    prefill_msg, decode_msg = _run_optimized_prefill_decode_parity(
        device,
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        trace_decode=True,
    )
    print(f"optimized real prefill {prefill_msg}")
    print(f"optimized real traced decode {decode_msg}")


@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_OPTIMIZED_CANDIDATES") != "1",
    reason="set RUN_QWEN36_OPTIMIZED_CANDIDATES=1 to run optimization candidate evidence",
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(
    ("policy_name", "layer_idx", "seq_len"),
    [
        ("routed_bfp4", 0, 5),
        ("routed_bfp4", 3, 5),
        ("routed_bfp4_exact_nnz", 0, 5),
        ("routed_bfp4_exact_nnz", 3, 5),
        ("moe_all_bfp4", 0, 5),
        ("decode_l1_sparse_inputs", 0, 5),
        ("decode_l1_sparse_inputs", 3, 5),
        ("decode_l1_sparse_inputs_exact_nnz", 0, 5),
        ("decode_l1_sparse_inputs_exact_nnz", 3, 5),
        ("prefill_l1_sparse_inputs", 0, 5),
        ("prefill_l1_sparse_inputs", 3, 5),
        ("prefill_l1_sparse_inputs_exact_nnz", 0, 5),
        ("prefill_l1_sparse_inputs_exact_nnz", 3, 5),
        ("decode_sdpa_k64", 3, 5),
        ("sparse_in0_block_w2", 0, 5),
        ("sparse_in0_block_w2", 3, 5),
        ("sparse_in0_block_w4", 0, 5),
        ("sparse_in0_block_w4", 3, 5),
        ("sparse_in0_block_w4_exact_nnz", 0, 5),
        ("sparse_in0_block_w4_exact_nnz", 3, 5),
        ("sparse_cores16_out2", 0, 5),
        ("sparse_cores16_out2", 3, 5),
        ("sparse_cores16_out2_exact_nnz", 0, 5),
        ("sparse_cores16_out2_exact_nnz", 3, 5),
        ("decode_exact_nnz", 0, 5),
        ("decode_exact_nnz", 3, 5),
    ],
    ids=[
        "routed_bfp4_linear",
        "routed_bfp4_full",
        "routed_bfp4_exact_nnz_linear",
        "routed_bfp4_exact_nnz_full",
        "moe_all_bfp4_linear",
        "decode_l1_sparse_linear",
        "decode_l1_sparse_full",
        "decode_l1_sparse_exact_nnz_linear",
        "decode_l1_sparse_exact_nnz_full",
        "prefill_l1_sparse_linear",
        "prefill_l1_sparse_full",
        "prefill_l1_sparse_exact_nnz_linear",
        "prefill_l1_sparse_exact_nnz_full",
        "decode_sdpa_k64_full",
        "sparse_in0_block_w2_linear",
        "sparse_in0_block_w2_full",
        "sparse_in0_block_w4_linear",
        "sparse_in0_block_w4_full",
        "sparse_in0_block_w4_exact_nnz_linear",
        "sparse_in0_block_w4_exact_nnz_full",
        "sparse_cores16_out2_linear",
        "sparse_cores16_out2_full",
        "sparse_cores16_out2_exact_nnz_linear",
        "sparse_cores16_out2_exact_nnz_full",
        "decode_exact_nnz_linear",
        "decode_exact_nnz_full",
    ],
)
def test_candidate_qwen36_optimized_policy_real_weight_pcc(device, policy_name, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _load_real_layer_state(layer_idx)
    policy = _optimized_policy_from_name(policy_name)
    prefill_msg, decode_msg = _run_optimized_prefill_decode_parity(
        device,
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        trace_decode=True,
        policy=policy,
    )
    print(f"candidate policy={policy_name} prefill {prefill_msg}")
    print(f"candidate policy={policy_name} traced decode {decode_msg}")


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _find_largest_divisor(value: int, max_divisor: int = 8) -> int:
    for candidate in range(max_divisor, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _decode_dram_shard_core_grid(k: int) -> ttnn.CoreGrid:
    k_tiles = k // TILE_SIZE
    possible_cores = [cores for cores in range(1, min(64, k_tiles) + 1) if k_tiles % cores == 0]
    possible_cores.sort(key=lambda cores: abs(cores - 32))
    for cores in possible_cores:
        for rows in range(1, 9):
            if cores % rows != 0:
                continue
            cols = cores // rows
            if cols <= 8:
                return ttnn.CoreGrid(x=cols, y=rows)
    raise RuntimeError(f"could not choose DRAM-sharded decode grid for k={k}")


def _dram_width_sharded_weight(
    device,
    packed_weight: torch.Tensor,
    *,
    dtype,
    physical_width: int,
) -> ttnn.Tensor:
    num_banks = int(device.dram_grid_size().x * device.dram_grid_size().y)
    padded = packed_weight
    if physical_width != packed_weight.shape[0]:
        padded = torch.nn.functional.pad(packed_weight, (0, 0, 0, physical_width - packed_weight.shape[0]))
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
            )
        }
    )
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        [padded.shape[1], physical_width // num_banks],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    return ttnn.from_torch(
        padded.transpose(-1, -2).contiguous().unsqueeze(0).unsqueeze(0),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )


def _dense_projection_weight(device, packed_weight: torch.Tensor, *, dtype) -> ttnn.Tensor:
    return ttnn.from_torch(
        packed_weight.transpose(-1, -2).contiguous().unsqueeze(0).unsqueeze(0),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _dram_sharded_projection(device, hidden: ttnn.Tensor, weight: ttnn.Tensor, logical_width: int, physical_width: int):
    _, _, logical_m, k = tuple(int(dim) for dim in hidden.shape)
    m_padded = _ceil_to_multiple(logical_m, TILE_SIZE)
    grid = _decode_dram_shard_core_grid(k)
    num_cores = grid.x * grid.y
    in0_tiles_per_core = k // (TILE_SIZE * num_cores)
    in0_sharded = ttnn.interleaved_to_sharded(
        hidden,
        ttnn.CoreCoord(grid.x, grid.y),
        [m_padded, in0_tiles_per_core * TILE_SIZE],
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_find_largest_divisor(in0_tiles_per_core),
        per_core_M=m_padded // TILE_SIZE,
        per_core_N=math.ceil(physical_width / (TILE_SIZE * num_cores)),
        fused_activation=None,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    projected = ttnn.linear(
        in0_sharded,
        weight,
        program_config=program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        output_tile=ttnn.Tile([TILE_SIZE, TILE_SIZE]),
    )
    projected = ttnn.to_memory_config(projected, ttnn.DRAM_MEMORY_CONFIG)
    if physical_width != logical_width:
        projected = _slice(projected, (0, 0, 0, 0), (1, 1, logical_m, logical_width))
    return projected, grid, program_config


def _time_signposted(device, signpost_name: str, fn):
    fn()
    ttnn.synchronize_device(device)
    _signpost(signpost_name)
    start = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _signpost(f"{signpost_name}_END")
    return out, elapsed_ms


@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_OPTIMIZED_CANDIDATES") != "1",
    reason="set RUN_QWEN36_OPTIMIZED_CANDIDATES=1 to run optimization candidate evidence",
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(
    ("name", "layer_idx", "keys"),
    [
        (
            "linear_qkvzba",
            0,
            (
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_a.weight",
            ),
        ),
        (
            "full_qkgv",
            3,
            ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
        ),
    ],
    ids=["linear_qkvzba", "full_qkgv"],
)
def test_candidate_decode_dense_projection_dram_sharded(device, name, layer_idx, keys):
    cfg = _target_text_config()
    state = _load_real_layer_state(layer_idx)
    hidden = _randn((1, 1, cfg.hidden_size), seed=4400 + layer_idx, scale=0.01)
    hidden_tt = _tt_bf16(hidden.unsqueeze(0), device)
    packed_weight = torch.cat([state[key] for key in keys], dim=0)
    logical_width = packed_weight.shape[0]
    num_banks = int(device.dram_grid_size().x * device.dram_grid_size().y)
    physical_width = _ceil_to_multiple(logical_width, num_banks * TILE_SIZE)

    baseline_weight = _dense_projection_weight(device, packed_weight, dtype=ttnn.bfloat8_b)
    dram_weight = _dram_width_sharded_weight(
        device,
        packed_weight,
        dtype=ttnn.bfloat8_b,
        physical_width=physical_width,
    )

    baseline, baseline_ms = _time_signposted(
        device,
        f"OPT_CAND_{name.upper()}_BASE",
        lambda: ttnn.linear(hidden_tt, baseline_weight, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG),
    )
    candidate_result = {}

    def run_candidate():
        projected, grid, program_config = _dram_sharded_projection(
            device, hidden_tt, dram_weight, logical_width, physical_width
        )
        candidate_result["grid"] = grid
        candidate_result["program_config"] = program_config
        return projected

    try:
        candidate, candidate_ms = _time_signposted(device, f"OPT_CAND_{name.upper()}_DRAM_SHARDED", run_candidate)
    except Exception as exc:
        pytest.skip(f"{name} adapted DRAM-sharded projection rejected by TTNN: {exc!r}")

    pcc_msg = _assert_pcc(
        f"{name} adapted DRAM-sharded projection",
        _to_torch(baseline),
        _to_torch(candidate),
    )
    print(
        "candidate dense_dram_sharded "
        f"name={name} padded_width={physical_width} baseline_ms={baseline_ms:.3f} "
        f"candidate_ms={candidate_ms:.3f} grid={candidate_result['grid']} "
        f"program_config={candidate_result['program_config']} {pcc_msg}"
    )


def test_optimized_runtime_fallback_audit_source():
    runtime_functions = {
        "_OptimizedQwenMoe._routed_decode": optimized_decoder_module._OptimizedQwenMoe._routed_decode,
        "_OptimizedQwenMoe._routed_prefill_chunk": optimized_decoder_module._OptimizedQwenMoe._routed_prefill_chunk,
        "_OptimizedFullAttention.decode_forward": optimized_decoder_module._OptimizedFullAttention.decode_forward,
        "OptimizedDecoder.prefill_forward": optimized_decoder_module.OptimizedDecoder.prefill_forward,
        "OptimizedDecoder.decode_forward": optimized_decoder_module.OptimizedDecoder.decode_forward,
    }
    forbidden_names = {"torch"}
    forbidden_attrs = {"from_torch", "to_torch", "get_fallback_function"}
    violations = []
    for name, func in runtime_functions.items():
        source = textwrap.dedent(inspect.getsource(func))
        if "FunctionalDecoder.from_state_dict" in source or "FunctionalDecoder.prefill_forward" in source:
            violations.append(f"{name}: functional decoder fallback call")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(f"{name}: name {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
                violations.append(f"{name}: attribute {node.attr}")
    assert not violations


def _run_optimized_signposted_prefill(device, *, layer_idx: int, seq_len: int, signpost_name: str):
    cfg = _target_text_config()
    state = _state_for_perf(cfg, layer_idx)
    tt_layer = _make_optimized_decoder(state, cfg=cfg, layer_idx=layer_idx, device=device)

    if cfg.layer_types[layer_idx] == "full_attention":
        batch = 1
        max_seq_len = max(96, ((seq_len + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE)
        hidden = _randn((batch, seq_len, cfg.hidden_size), seed=3600 + seq_len, scale=0.01)
        position_ids = torch.arange(seq_len, dtype=torch.long).reshape(batch, seq_len)
        position_embeddings = _rotary(cfg, hidden, position_ids)
        page_table = _page_table(batch, max_seq_len)
        warm_cache = OptimizedDecoder.allocate_full_attention_cache(
            hf_config=cfg, mesh_device=device, max_batch_size=batch, max_seq_len=max_seq_len, block_size=BLOCK_SIZE
        )
        tt_layer.prefill_forward(
            _tt_bf16(hidden.unsqueeze(0), device),
            position_embeddings=(
                _tt_bf16(position_embeddings[0].unsqueeze(1), device),
                _tt_bf16(position_embeddings[1].unsqueeze(1), device),
            ),
            page_table=_tt_int(page_table, device),
            kv_cache=warm_cache,
        )
        measure_cache = OptimizedDecoder.allocate_full_attention_cache(
            hf_config=cfg, mesh_device=device, max_batch_size=batch, max_seq_len=max_seq_len, block_size=BLOCK_SIZE
        )
        ttnn.synchronize_device(device)
        _signpost(signpost_name)
        start = time.perf_counter()
        out = tt_layer.prefill_forward(
            _tt_bf16(hidden.unsqueeze(0), device),
            position_embeddings=(
                _tt_bf16(position_embeddings[0].unsqueeze(1), device),
                _tt_bf16(position_embeddings[1].unsqueeze(1), device),
            ),
            page_table=_tt_int(page_table, device),
            kv_cache=measure_cache,
        ).hidden_states
    else:
        hidden = _randn((1, seq_len, cfg.hidden_size), seed=3700 + seq_len, scale=0.01)
        hidden_tt = _tt_bf16(hidden.unsqueeze(0), device)
        warm_state = OptimizedDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=1)
        tt_layer.prefill_forward(hidden_tt, linear_state=warm_state)
        measure_state = OptimizedDecoder.allocate_linear_attention_state(
            hf_config=cfg, mesh_device=device, batch_size=1
        )
        ttnn.synchronize_device(device)
        _signpost(signpost_name)
        start = time.perf_counter()
        out = tt_layer.prefill_forward(hidden_tt, linear_state=measure_state).hidden_states

    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _signpost(f"{signpost_name}_END")
    print(f"{signpost_name} wall_ms={elapsed_ms:.3f} output_shape={tuple(out.shape)}")


def _run_optimized_signposted_traced_decode(device, *, layer_idx: int, seq_len: int, signpost_name: str):
    cfg = _target_text_config()
    state = _state_for_perf(cfg, layer_idx)
    tt_layer = _make_optimized_decoder(state, cfg=cfg, layer_idx=layer_idx, device=device)
    decode_input, decode_kwargs = _prepare_optimized_decode_after_prefill(device, cfg, tt_layer, layer_idx, seq_len)

    tt_layer.decode_forward(decode_input, **decode_kwargs)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)

    _signpost(signpost_name)
    start = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _signpost(f"{signpost_name}_END")
    print(f"{signpost_name} traced_wall_ms={elapsed_ms:.3f} output_shape={tuple(traced.shape)}")
    ttnn.release_trace(device, trace_id)


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_OPTIMIZED_PERF") != "1",
    reason="set RUN_QWEN36_OPTIMIZED_PERF=1 for Tracy performance evidence",
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_perf_qwen36_optimized_linear_prefill(device):
    _run_optimized_signposted_prefill(device, layer_idx=0, seq_len=5, signpost_name="OPT_LINEAR_PREFILL")


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_OPTIMIZED_PERF") != "1",
    reason="set RUN_QWEN36_OPTIMIZED_PERF=1 for Tracy performance evidence",
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_perf_qwen36_optimized_full_prefill(device):
    _run_optimized_signposted_prefill(device, layer_idx=3, seq_len=33, signpost_name="OPT_FULL_PREFILL")


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_OPTIMIZED_PERF") != "1",
    reason="set RUN_QWEN36_OPTIMIZED_PERF=1 for Tracy performance evidence",
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_perf_qwen36_optimized_linear_decode(device):
    _run_optimized_signposted_traced_decode(device, layer_idx=0, seq_len=5, signpost_name="OPT_LINEAR_DECODE")


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_OPTIMIZED_PERF") != "1",
    reason="set RUN_QWEN36_OPTIMIZED_PERF=1 for Tracy performance evidence",
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_perf_qwen36_optimized_full_decode(device):
    _run_optimized_signposted_traced_decode(device, layer_idx=3, seq_len=33, signpost_name="OPT_FULL_DECODE")
