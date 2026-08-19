# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
import math
import os
import textwrap
import time
from dataclasses import dataclass

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_functional_decoder import (
    BLOCK_SIZE,
    _assert_pcc,
    _load_real_layer_state,
    _page_table,
    _randn,
    _rotary,
    _signpost,
    _state_for_perf,
    _synthetic_layer_state,
    _target_text_config,
    _to_torch,
    _tt_bf16,
    _tt_int,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_optimized_decoder import (
    _run_optimized_signposted_prefill,
    _run_optimized_signposted_traced_decode,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tt import multichip_decoder as multichip_decoder_module
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_6_35b_a3b.tt.optimized_decoder import OptimizedDecoder


@dataclass(frozen=True)
class _BaselineRun:
    prefill: torch.Tensor
    decode: torch.Tensor
    cache: tuple[torch.Tensor, torch.Tensor] | None = None
    linear_state: tuple[tuple[torch.Tensor, ...], torch.Tensor] | None = None


def _tt_mesh_bf16(tensor: torch.Tensor, mesh_device, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _tt_mesh_int(tensor: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.to(torch.int32).contiguous(),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _local_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    return tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().float()


def _mesh_first_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    return _local_to_torch(ttnn.get_device_tensors(tensor)[0])


def _mesh_all_to_torch(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [_local_to_torch(local) for local in ttnn.get_device_tensors(tensor)]


def _run_multichip_traced_decode(
    mesh_device, tt_layer: MultichipDecoder, decode_input: ttnn.Tensor, decode_kwargs: dict
):
    tt_layer.decode_forward(decode_input, **decode_kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)
    return traced


def _run_with_single_device(func, *, trace_region_size: int = 16_000_000):
    original_default_device = ttnn.GetDefaultDevice()
    device = None
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    try:
        device = ttnn.CreateDevice(
            device_id=int(os.environ.get("TT_DEVICE_ID", "0")), trace_region_size=trace_region_size
        )
        ttnn.SetDefaultDevice(device)
        return func(device)
    finally:
        ttnn.SetDefaultDevice(original_default_device)
        if device is not None:
            ttnn.close_device(device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _run_with_target_mesh(func, *, trace_region_size: int = 32_000_000):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), trace_region_size=trace_region_size)
    try:
        return func(mesh_device)
    finally:
        ttnn.synchronize_device(mesh_device)
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _linear_state_to_torch(linear_state) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    return tuple(_to_torch(tap) for tap in linear_state.conv_state), _to_torch(linear_state.recurrent_state)


def _collect_optimized_baseline(
    device,
    *,
    cfg,
    layer_idx: int,
    state: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    decode_hidden: torch.Tensor,
    seq_len: int,
    batch: int,
) -> _BaselineRun:
    baseline = OptimizedDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=device,
    )

    if cfg.layer_types[layer_idx] == "full_attention":
        max_seq_len = max(96, math.ceil((seq_len + 1) / BLOCK_SIZE) * BLOCK_SIZE)
        position_ids = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len).expand(batch, seq_len)
        position_embeddings = _rotary(cfg, hidden, position_ids)
        page_table = _page_table(batch, max_seq_len)
        kv_cache = OptimizedDecoder.allocate_full_attention_cache(
            hf_config=cfg,
            mesh_device=device,
            max_batch_size=batch,
            max_seq_len=max_seq_len,
            block_size=BLOCK_SIZE,
        )
        prefill = baseline.prefill_forward(
            _tt_bf16(hidden.unsqueeze(0), device),
            position_embeddings=(
                _tt_bf16(position_embeddings[0].unsqueeze(1), device),
                _tt_bf16(position_embeddings[1].unsqueeze(1), device),
            ),
            page_table=_tt_int(page_table, device),
            kv_cache=kv_cache,
        ).hidden_states

        current_pos = torch.full((batch,), seq_len, dtype=torch.int32)
        decode_position_embeddings = _rotary(cfg, decode_hidden, current_pos.to(torch.long).reshape(batch, 1))
        decode = baseline.decode_forward(
            _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device),
            position_embeddings=(
                _tt_bf16(decode_position_embeddings[0].unsqueeze(0), device),
                _tt_bf16(decode_position_embeddings[1].unsqueeze(0), device),
            ),
            page_table=_tt_int(page_table, device),
            kv_cache=kv_cache,
            current_pos=_tt_int(current_pos, device),
        ).hidden_states
        ttnn.synchronize_device(device)
        return _BaselineRun(
            prefill=_to_torch(prefill),
            decode=_to_torch(decode),
            cache=(_to_torch(kv_cache.keys), _to_torch(kv_cache.values)),
        )

    linear_state = OptimizedDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=batch)
    prefill_result = baseline.prefill_forward(_tt_bf16(hidden.unsqueeze(0), device), linear_state=linear_state)
    current_pos = torch.full((batch,), seq_len, dtype=torch.int32)
    decode = baseline.decode_forward(
        _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device),
        current_pos=_tt_int(current_pos, device),
        linear_state=prefill_result.linear_state,
    ).hidden_states
    ttnn.synchronize_device(device)
    return _BaselineRun(
        prefill=_to_torch(prefill_result.hidden_states),
        decode=_to_torch(decode),
        linear_state=_linear_state_to_torch(prefill_result.linear_state),
    )


def _run_optimized_baseline(
    *,
    cfg,
    layer_idx: int,
    state: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    decode_hidden: torch.Tensor,
    seq_len: int,
    batch: int,
) -> _BaselineRun:
    def run(device):
        return _collect_optimized_baseline(
            device,
            cfg=cfg,
            layer_idx=layer_idx,
            state=state,
            hidden=hidden,
            decode_hidden=decode_hidden,
            seq_len=seq_len,
            batch=batch,
        )

    return _run_with_single_device(run, trace_region_size=16_000_000)


def _linear_conv_local_slice(full: torch.Tensor, cfg, col: int, tp: int = 2) -> torch.Tensor:
    key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
    value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
    local_key = key_dim // tp
    local_value = value_dim // tp
    return torch.cat(
        [
            full[..., col * local_key : (col + 1) * local_key],
            full[..., key_dim + col * local_key : key_dim + (col + 1) * local_key],
            full[..., 2 * key_dim + col * local_value : 2 * key_dim + (col + 1) * local_value],
        ],
        dim=-1,
    )


def _device_col(device_index: int) -> int:
    return device_index % 2


def _gather_logical_cache_tokens(
    cache: torch.Tensor, page_table: torch.Tensor, seq_len: int, block_size: int
) -> torch.Tensor:
    per_batch = []
    for batch_idx in range(page_table.shape[0]):
        tokens = []
        for pos in range(seq_len):
            block = int(page_table[batch_idx, pos // block_size].item())
            tokens.append(cache[block, :, pos % block_size, :])
        per_batch.append(torch.stack(tokens, dim=1))
    return torch.stack(per_batch, dim=0)


def _assert_full_cache_layout(cfg, baseline_cache, multichip_cache, page_table: torch.Tensor, seq_len: int) -> None:
    baseline_keys, baseline_values = baseline_cache
    baseline_keys = _gather_logical_cache_tokens(baseline_keys, page_table, seq_len, multichip_cache.block_size)
    baseline_values = _gather_logical_cache_tokens(baseline_values, page_table, seq_len, multichip_cache.block_size)
    for idx, (local_keys, local_values) in enumerate(
        zip(_mesh_all_to_torch(multichip_cache.keys), _mesh_all_to_torch(multichip_cache.values), strict=True)
    ):
        local_keys = _gather_logical_cache_tokens(local_keys, page_table, seq_len, multichip_cache.block_size)
        local_values = _gather_logical_cache_tokens(local_values, page_table, seq_len, multichip_cache.block_size)
        col = _device_col(idx)
        start = col * (cfg.num_key_value_heads // 2)
        end = start + (cfg.num_key_value_heads // 2)
        _assert_pcc(f"multichip local key cache device{idx}", baseline_keys[:, start:end], local_keys, pcc=0.999)
        _assert_pcc(f"multichip local value cache device{idx}", baseline_values[:, start:end], local_values, pcc=0.999)


def _assert_linear_state_layout(cfg, baseline_state, multichip_state) -> None:
    baseline_conv_state, baseline_recurrent = baseline_state
    local_value_heads = cfg.linear_num_value_heads // 2
    batch = baseline_recurrent.shape[1] // cfg.linear_num_value_heads
    baseline_recurrent_by_batch = baseline_recurrent.reshape(
        1,
        batch,
        cfg.linear_num_value_heads,
        cfg.linear_key_head_dim,
        cfg.linear_value_head_dim,
    )
    for idx, local_recurrent in enumerate(_mesh_all_to_torch(multichip_state.recurrent_state)):
        col = _device_col(idx)
        start = col * local_value_heads
        end = start + local_value_heads
        expected = baseline_recurrent_by_batch[:, :, start:end]
        expected = expected.reshape(1, batch * local_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim)
        _assert_pcc(
            f"multichip recurrent state device{idx}",
            expected,
            local_recurrent,
            pcc=0.999,
        )

    for tap_idx, (baseline_tap, multichip_tap) in enumerate(
        zip(baseline_conv_state, multichip_state.conv_state, strict=True)
    ):
        baseline_full = baseline_tap
        for idx, local_tap in enumerate(_mesh_all_to_torch(multichip_tap)):
            expected = _linear_conv_local_slice(baseline_full, cfg, _device_col(idx))
            _assert_pcc(f"multichip conv state tap{tap_idx} device{idx}", expected, local_tap, pcc=0.999)


def _collect_multichip_against_baseline(
    mesh_device,
    baseline_run: _BaselineRun,
    *,
    cfg,
    layer_idx: int,
    state: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    decode_hidden: torch.Tensor,
    seq_len: int,
    batch: int = 1,
    trace_decode: bool = False,
):
    multichip = MultichipDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )

    if cfg.layer_types[layer_idx] == "full_attention":
        max_seq_len = max(96, math.ceil((seq_len + 1) / BLOCK_SIZE) * BLOCK_SIZE)
        position_ids = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len).expand(batch, seq_len)
        position_embeddings = _rotary(cfg, hidden, position_ids)
        page_table = _page_table(batch, max_seq_len)

        multichip_cache = MultichipDecoder.allocate_full_attention_cache(
            hf_config=cfg,
            mesh_device=mesh_device,
            max_batch_size=batch,
            max_seq_len=max_seq_len,
            block_size=BLOCK_SIZE,
        )
        multichip_prefill = multichip.prefill_forward(
            _tt_mesh_bf16(hidden.unsqueeze(0), mesh_device),
            position_embeddings=(
                _tt_mesh_bf16(position_embeddings[0].unsqueeze(1), mesh_device),
                _tt_mesh_bf16(position_embeddings[1].unsqueeze(1), mesh_device),
            ),
            page_table=_tt_mesh_int(page_table, mesh_device),
            kv_cache=multichip_cache,
        ).hidden_states
        prefill_msg = _assert_pcc(
            "multichip full_attention prefill vs optimized",
            baseline_run.prefill,
            _mesh_first_to_torch(multichip_prefill),
        )
        _assert_full_cache_layout(cfg, baseline_run.cache, multichip_cache, page_table, seq_len)

        current_pos = torch.full((batch,), seq_len, dtype=torch.int32)
        decode_position_embeddings = _rotary(cfg, decode_hidden, current_pos.to(torch.long).reshape(batch, 1))
        decode_kwargs = {
            "position_embeddings": (
                _tt_mesh_bf16(decode_position_embeddings[0].unsqueeze(0), mesh_device),
                _tt_mesh_bf16(decode_position_embeddings[1].unsqueeze(0), mesh_device),
            ),
            "page_table": _tt_mesh_int(page_table, mesh_device),
            "kv_cache": multichip_cache,
            "current_pos": _tt_mesh_int(current_pos, mesh_device),
        }
        decode_input = _tt_mesh_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), mesh_device)
        multichip_decode = (
            _run_multichip_traced_decode(mesh_device, multichip, decode_input, decode_kwargs)
            if trace_decode
            else multichip.decode_forward(decode_input, **decode_kwargs).hidden_states
        )
        decode_msg = _assert_pcc(
            "multichip full_attention decode vs optimized",
            baseline_run.decode,
            _mesh_first_to_torch(multichip_decode),
        )
        return prefill_msg, decode_msg

    multichip_state = MultichipDecoder.allocate_linear_attention_state(
        hf_config=cfg, mesh_device=mesh_device, batch_size=batch
    )
    multichip_prefill_result = multichip.prefill_forward(
        _tt_mesh_bf16(hidden.unsqueeze(0), mesh_device),
        linear_state=multichip_state,
    )
    prefill_msg = _assert_pcc(
        "multichip linear_attention prefill vs optimized",
        baseline_run.prefill,
        _mesh_first_to_torch(multichip_prefill_result.hidden_states),
    )
    _assert_linear_state_layout(cfg, baseline_run.linear_state, multichip_prefill_result.linear_state)

    current_pos = torch.full((batch,), seq_len, dtype=torch.int32)
    decode_input = _tt_mesh_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), mesh_device)
    decode_kwargs = {
        "current_pos": _tt_mesh_int(current_pos, mesh_device),
        "linear_state": multichip_prefill_result.linear_state,
    }
    multichip_decode = (
        _run_multichip_traced_decode(mesh_device, multichip, decode_input, decode_kwargs)
        if trace_decode
        else multichip.decode_forward(decode_input, **decode_kwargs).hidden_states
    )
    decode_msg = _assert_pcc(
        "multichip linear_attention decode vs optimized",
        baseline_run.decode,
        _mesh_first_to_torch(multichip_decode),
    )
    return prefill_msg, decode_msg


def _run_multichip_against_optimized(
    *,
    cfg,
    layer_idx: int,
    state: dict[str, torch.Tensor],
    seq_len: int,
    batch: int = 1,
    trace_decode: bool = False,
):
    hidden = _randn((batch, seq_len, cfg.hidden_size), seed=4200 + batch + layer_idx + seq_len, scale=0.01)
    decode_hidden = _randn((batch, 1, cfg.hidden_size), seed=4300 + batch + layer_idx + seq_len, scale=0.01)
    baseline_run = _run_optimized_baseline(
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        hidden=hidden,
        decode_hidden=decode_hidden,
        seq_len=seq_len,
        batch=batch,
    )

    def run(mesh_device):
        return _collect_multichip_against_baseline(
            mesh_device,
            baseline_run,
            cfg=cfg,
            layer_idx=layer_idx,
            state=state,
            hidden=hidden,
            decode_hidden=decode_hidden,
            seq_len=seq_len,
            batch=batch,
            trace_decode=trace_decode,
        )

    return _run_with_target_mesh(run, trace_region_size=32_000_000)


def _run_multichip_signposted_prefill(mesh_device, *, layer_idx: int, seq_len: int, signpost_name: str):
    cfg = _target_text_config()
    state = _state_for_perf(cfg, layer_idx)
    tt_layer = MultichipDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)

    if cfg.layer_types[layer_idx] == "full_attention":
        batch = 1
        max_seq_len = max(96, math.ceil((seq_len + 1) / BLOCK_SIZE) * BLOCK_SIZE)
        hidden = _randn((batch, seq_len, cfg.hidden_size), seed=3600 + seq_len, scale=0.01)
        position_ids = torch.arange(seq_len, dtype=torch.long).reshape(batch, seq_len)
        position_embeddings = _rotary(cfg, hidden, position_ids)
        page_table = _page_table(batch, max_seq_len)
        warm_cache = MultichipDecoder.allocate_full_attention_cache(
            hf_config=cfg, mesh_device=mesh_device, max_batch_size=batch, max_seq_len=max_seq_len, block_size=BLOCK_SIZE
        )
        tt_layer.prefill_forward(
            _tt_mesh_bf16(hidden.unsqueeze(0), mesh_device),
            position_embeddings=(
                _tt_mesh_bf16(position_embeddings[0].unsqueeze(1), mesh_device),
                _tt_mesh_bf16(position_embeddings[1].unsqueeze(1), mesh_device),
            ),
            page_table=_tt_mesh_int(page_table, mesh_device),
            kv_cache=warm_cache,
        )
        measure_cache = MultichipDecoder.allocate_full_attention_cache(
            hf_config=cfg, mesh_device=mesh_device, max_batch_size=batch, max_seq_len=max_seq_len, block_size=BLOCK_SIZE
        )
        ttnn.synchronize_device(mesh_device)
        _signpost(signpost_name)
        start = time.perf_counter()
        out = tt_layer.prefill_forward(
            _tt_mesh_bf16(hidden.unsqueeze(0), mesh_device),
            position_embeddings=(
                _tt_mesh_bf16(position_embeddings[0].unsqueeze(1), mesh_device),
                _tt_mesh_bf16(position_embeddings[1].unsqueeze(1), mesh_device),
            ),
            page_table=_tt_mesh_int(page_table, mesh_device),
            kv_cache=measure_cache,
        ).hidden_states
    else:
        hidden = _randn((1, seq_len, cfg.hidden_size), seed=3700 + seq_len, scale=0.01)
        hidden_tt = _tt_mesh_bf16(hidden.unsqueeze(0), mesh_device)
        warm_state = MultichipDecoder.allocate_linear_attention_state(
            hf_config=cfg, mesh_device=mesh_device, batch_size=1
        )
        tt_layer.prefill_forward(hidden_tt, linear_state=warm_state)
        measure_state = MultichipDecoder.allocate_linear_attention_state(
            hf_config=cfg, mesh_device=mesh_device, batch_size=1
        )
        ttnn.synchronize_device(mesh_device)
        _signpost(signpost_name)
        start = time.perf_counter()
        out = tt_layer.prefill_forward(hidden_tt, linear_state=measure_state).hidden_states

    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _signpost(f"{signpost_name}_END")
    print(f"{signpost_name} wall_ms={elapsed_ms:.3f} output_shape={tuple(out.shape)}")


def _prepare_multichip_decode_after_prefill(mesh_device, cfg, tt_layer: MultichipDecoder, layer_idx: int, seq_len: int):
    batch = 1
    decode_hidden = _randn((batch, 1, cfg.hidden_size), seed=2800 + layer_idx + seq_len, scale=0.01)
    decode_input = _tt_mesh_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), mesh_device)
    current_pos = _tt_mesh_int(torch.tensor([seq_len], dtype=torch.int32), mesh_device)

    if cfg.layer_types[layer_idx] == "full_attention":
        max_seq_len = max(96, math.ceil((seq_len + 1) / BLOCK_SIZE) * BLOCK_SIZE)
        hidden = _randn((batch, seq_len, cfg.hidden_size), seed=2600 + seq_len, scale=0.01)
        position_ids = torch.arange(seq_len, dtype=torch.long).reshape(batch, seq_len)
        position_embeddings = _rotary(cfg, hidden, position_ids)
        page_table = _page_table(batch, max_seq_len)
        kv_cache = MultichipDecoder.allocate_full_attention_cache(
            hf_config=cfg,
            mesh_device=mesh_device,
            max_batch_size=batch,
            max_seq_len=max_seq_len,
            block_size=BLOCK_SIZE,
        )
        tt_layer.prefill_forward(
            _tt_mesh_bf16(hidden.unsqueeze(0), mesh_device),
            position_embeddings=(
                _tt_mesh_bf16(position_embeddings[0].unsqueeze(1), mesh_device),
                _tt_mesh_bf16(position_embeddings[1].unsqueeze(1), mesh_device),
            ),
            page_table=_tt_mesh_int(page_table, mesh_device),
            kv_cache=kv_cache,
        )
        decode_position_embeddings = _rotary(cfg, decode_hidden, torch.tensor([[seq_len]], dtype=torch.long))
        kwargs = {
            "current_pos": current_pos,
            "position_embeddings": (
                _tt_mesh_bf16(decode_position_embeddings[0].unsqueeze(0), mesh_device),
                _tt_mesh_bf16(decode_position_embeddings[1].unsqueeze(0), mesh_device),
            ),
            "page_table": _tt_mesh_int(page_table, mesh_device),
            "kv_cache": kv_cache,
        }
    else:
        state = MultichipDecoder.allocate_linear_attention_state(
            hf_config=cfg, mesh_device=mesh_device, batch_size=batch
        )
        hidden = _randn((batch, seq_len, cfg.hidden_size), seed=2700 + seq_len, scale=0.01)
        prefill = tt_layer.prefill_forward(_tt_mesh_bf16(hidden.unsqueeze(0), mesh_device), linear_state=state)
        kwargs = {"current_pos": current_pos, "linear_state": prefill.linear_state}
    return decode_input, kwargs


def _run_multichip_signposted_traced_decode(mesh_device, *, layer_idx: int, seq_len: int, signpost_name: str):
    cfg = _target_text_config()
    state = _state_for_perf(cfg, layer_idx)
    tt_layer = MultichipDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    decode_input, decode_kwargs = _prepare_multichip_decode_after_prefill(
        mesh_device, cfg, tt_layer, layer_idx, seq_len
    )

    tt_layer.decode_forward(decode_input, **decode_kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)

    _signpost(signpost_name)
    start = time.perf_counter()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _signpost(f"{signpost_name}_END")
    print(f"{signpost_name} traced_wall_ms={elapsed_ms:.3f} output_shape={tuple(traced.shape)}")
    ttnn.release_trace(mesh_device, trace_id)


def test_multichip_decoder_graph_summary():
    summary = MultichipDecoder.graph_summary
    assert summary.optimized_baseline.reduced_moe_weights
    assert summary.target_mesh_shape == (2, 2)
    assert summary.tensor_parallel_size == 2
    assert summary.expert_parallel_size == 2
    assert summary.replicated_residual_contract
    assert summary.full_attention_q_heads_per_device == 8
    assert summary.full_attention_kv_heads_per_device == 1
    assert summary.linear_attention_value_heads_per_device == 16
    assert summary.moe_active_decode_uses_routing_remap
    assert summary.moe_active_prefill_uses_token_sparse_path
    assert summary.moe_prefill_experts_per_ep_device == 4


def test_multichip_runtime_fallback_audit_source():
    runtime_functions = {
        "_MultichipQwenMoe._routed_decode": multichip_decoder_module._MultichipQwenMoe._routed_decode,
        "_MultichipQwenMoe._routed_prefill_chunk": multichip_decoder_module._MultichipQwenMoe._routed_prefill_chunk,
        "_MultichipFullAttention.prefill_forward": multichip_decoder_module._MultichipFullAttention.prefill_forward,
        "_MultichipFullAttention.decode_forward": multichip_decoder_module._MultichipFullAttention.decode_forward,
        "_MultichipLinearAttention.prefill_forward": multichip_decoder_module._MultichipLinearAttention.prefill_forward,
        "_MultichipLinearAttention.decode_forward": multichip_decoder_module._MultichipLinearAttention.decode_forward,
        "MultichipDecoder.prefill_forward": multichip_decoder_module.MultichipDecoder.prefill_forward,
        "MultichipDecoder.decode_forward": multichip_decoder_module.MultichipDecoder.decode_forward,
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


@pytest.mark.parametrize(("layer_idx", "seq_len"), [(0, 5), (3, 33)], ids=["linear_seq5", "full_seq33"])
def test_synthetic_multichip_decoder_prefill_decode_against_optimized(layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    prefill_msg, decode_msg = _run_multichip_against_optimized(
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        trace_decode=True,
    )
    print(f"multichip prefill {prefill_msg}")
    print(f"multichip traced decode {decode_msg}")


@pytest.mark.parametrize(
    ("layer_idx", "seq_len"), [(0, 65), (3, 33)], ids=["linear_non_aligned65", "full_non_aligned33"]
)
def test_synthetic_multichip_decoder_non_aligned_lengths(layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    prefill_msg, decode_msg = _run_multichip_against_optimized(
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        trace_decode=True,
    )
    print(f"multichip non-aligned prefill {prefill_msg}")
    print(f"multichip non-aligned traced decode {decode_msg}")


@pytest.mark.parametrize(("layer_idx", "seq_len"), [(0, 5), (3, 33)], ids=["linear_batch2", "full_batch2"])
def test_synthetic_multichip_decoder_batch2_against_optimized(layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    prefill_msg, decode_msg = _run_multichip_against_optimized(
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        batch=2,
    )
    print(f"multichip batch2 prefill {prefill_msg}")
    print(f"multichip batch2 decode {decode_msg}")


@pytest.mark.real_weights
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_REAL_WEIGHTS") != "1", reason="set RUN_QWEN36_REAL_WEIGHTS=1 to load checkpoint weights"
)
@pytest.mark.parametrize(("layer_idx", "seq_len"), [(0, 1), (3, 1)], ids=["real_linear_layer0", "real_full_layer3"])
def test_real_weight_multichip_decoder_prefill_decode_against_optimized(layer_idx, seq_len):
    cfg = _target_text_config()
    state = _load_real_layer_state(layer_idx)
    prefill_msg, decode_msg = _run_multichip_against_optimized(
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        trace_decode=True,
    )
    print(f"multichip real prefill {prefill_msg}")
    print(f"multichip real traced decode {decode_msg}")


def _run_perf_pair(*, layer_idx: int, seq_len: int, kind: str, base_signpost: str, multichip_signpost: str):
    if kind == "prefill":
        _run_with_single_device(
            lambda device: _run_optimized_signposted_prefill(
                device,
                layer_idx=layer_idx,
                seq_len=seq_len,
                signpost_name=base_signpost,
            ),
            trace_region_size=0,
        )
        _run_with_target_mesh(
            lambda mesh: _run_multichip_signposted_prefill(
                mesh,
                layer_idx=layer_idx,
                seq_len=seq_len,
                signpost_name=multichip_signpost,
            ),
            trace_region_size=0,
        )
    elif kind == "decode":
        _run_with_single_device(
            lambda device: _run_optimized_signposted_traced_decode(
                device,
                layer_idx=layer_idx,
                seq_len=seq_len,
                signpost_name=base_signpost,
            ),
            trace_region_size=32_000_000,
        )
        _run_with_target_mesh(
            lambda mesh: _run_multichip_signposted_traced_decode(
                mesh,
                layer_idx=layer_idx,
                seq_len=seq_len,
                signpost_name=multichip_signpost,
            ),
            trace_region_size=32_000_000,
        )
    else:
        raise ValueError(f"unknown perf kind: {kind}")


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_MULTICHIP_PERF") != "1",
    reason="set RUN_QWEN36_MULTICHIP_PERF=1 for Tracy performance evidence",
)
def test_perf_qwen36_multichip_linear_prefill_vs_optimized():
    _run_perf_pair(
        layer_idx=0,
        seq_len=5,
        kind="prefill",
        base_signpost="BASE_LINEAR_PREFILL",
        multichip_signpost="MC_LINEAR_PREFILL",
    )


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_MULTICHIP_PERF") != "1",
    reason="set RUN_QWEN36_MULTICHIP_PERF=1 for Tracy performance evidence",
)
def test_perf_qwen36_multichip_full_prefill_vs_optimized():
    _run_perf_pair(
        layer_idx=3,
        seq_len=33,
        kind="prefill",
        base_signpost="BASE_FULL_PREFILL",
        multichip_signpost="MC_FULL_PREFILL",
    )


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_MULTICHIP_PERF") != "1",
    reason="set RUN_QWEN36_MULTICHIP_PERF=1 for Tracy performance evidence",
)
def test_perf_qwen36_multichip_linear_decode_vs_optimized():
    _run_perf_pair(
        layer_idx=0,
        seq_len=5,
        kind="decode",
        base_signpost="BASE_LINEAR_DECODE",
        multichip_signpost="MC_LINEAR_DECODE",
    )


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_MULTICHIP_PERF") != "1",
    reason="set RUN_QWEN36_MULTICHIP_PERF=1 for Tracy performance evidence",
)
def test_perf_qwen36_multichip_full_decode_vs_optimized():
    _run_perf_pair(
        layer_idx=3,
        seq_len=33,
        kind="decode",
        base_signpost="BASE_FULL_DECODE",
        multichip_signpost="MC_FULL_DECODE",
    )
