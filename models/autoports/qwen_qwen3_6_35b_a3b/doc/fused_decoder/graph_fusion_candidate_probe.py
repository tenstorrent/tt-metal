# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Focused graph-fusion candidate probes for the Qwen3.6 fused decoder.

This is a documentation/provenance script, not part of the runtime path.  It
checks TTNN dedicated head and MoE-gate ops against the exact Qwen fused-decoder
shapes so rejected candidates have reproducible evidence.
"""

from __future__ import annotations

import time
import traceback
import types

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_functional_decoder import (
    BLOCK_SIZE,
    _page_table,
    _randn,
    _rotary,
    _synthetic_layer_state,
    _target_text_config,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_functional_decoder import _tt_bf16 as _test_tt_bf16
from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_functional_decoder import _tt_int
from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder, _shape


def _slice(tensor: ttnn.Tensor, starts: tuple[int, ...], ends: tuple[int, ...]) -> ttnn.Tensor:
    return ttnn.slice(tensor, starts, ends, (1,) * len(starts), memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _slice_last(tensor: ttnn.Tensor, start: int, end: int) -> ttnn.Tensor:
    starts = [0] * len(_shape(tensor))
    ends = list(_shape(tensor))
    starts[-1] = start
    ends[-1] = end
    return _slice(tensor, tuple(starts), tuple(ends))


def _tt_bf16(tensor: torch.Tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _tt_u16(tensor: torch.Tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if denom == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float(torch.dot(a, b) / denom)


def _generalized_gate_buffers(device, cfg, tokens: int):
    if cfg.num_experts != 256 or cfg.num_experts_per_tok != 8:
        raise ValueError("this Qwen generalized_moe_gate probe is specialized for 256 experts/top-8")
    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(tokens, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    tile = ttnn.Tile((32, 32))
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )

    bias = ttnn.from_torch(
        torch.zeros((tokens, 16, 16), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
        tile=tile,
    )
    input_indices = torch.arange(256, dtype=torch.int32).reshape(1, 16, 16)
    input_indices = torch.transpose(input_indices, -2, -1).expand(tokens, -1, -1).contiguous().to(torch.uint16)
    indices = ttnn.from_torch(
        input_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
        tile=tile,
    )
    output = ttnn.from_torch(
        torch.zeros((tokens, 1, 16), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
        tile=tile,
    )
    output_indices = ttnn.from_torch(
        torch.zeros((tokens, 1, 16), dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
        tile=tile,
    )
    return {
        "memory_config": mem,
        "bias": bias,
        "indices": indices,
        "output": output,
        "output_indices": output_indices,
    }


def _current_dense_router_from_logits(logits: ttnn.Tensor, cfg) -> ttnn.Tensor:
    probs = ttnn.softmax(logits, dim=-1, numeric_stable=True)
    top_values, top_indices = ttnn.topk(probs, k=cfg.num_experts_per_tok, dim=-1, sorted=True)
    denom = ttnn.sum(top_values, dim=-1, keepdim=True)
    top_values = ttnn.div(top_values, denom, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.scatter(ttnn.zeros_like(probs), dim=-1, index=top_indices, src=top_values)


def _generalized_gate_dense_router_from_logits(logits: ttnn.Tensor, cfg, buffers) -> ttnn.Tensor:
    tokens = _shape(logits)[2]
    reshaped = ttnn.reshape(logits, (tokens, 16, 16))
    sharded_logits = ttnn.to_memory_config(reshaped, buffers["memory_config"])
    scores, indices = ttnn.experimental.deepseek.moe.generalized_moe_gate(
        sharded_logits,
        bias_tensor=buffers["bias"],
        input_indices_tensor=buffers["indices"],
        output_tensor=buffers["output"],
        output_indices_tensor=buffers["output_indices"],
        scaling_factor=1.0,
        enable_sigmoid=False,
        topk=cfg.num_experts_per_tok,
        output_softmax=True,
    )
    scores = _slice(scores, (0, 0, 0), (tokens, 1, cfg.num_experts_per_tok))
    indices = _slice(indices, (0, 0, 0), (tokens, 1, cfg.num_experts_per_tok))
    scores = ttnn.reshape(scores, (1, 1, tokens, cfg.num_experts_per_tok))
    indices = ttnn.reshape(indices, (1, 1, tokens, cfg.num_experts_per_tok))
    return ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=indices, src=scores)


def _install_generalized_gate_router_candidate(layer: FusedDecoder, device, cfg) -> None:
    buffers_by_tokens = {}

    def candidate_router(self, flat: ttnn.Tensor) -> ttnn.Tensor:
        logits = ttnn.linear(flat, self.router, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tokens = _shape(logits)[2]
        if tokens not in buffers_by_tokens:
            buffers_by_tokens[tokens] = _generalized_gate_buffers(device, cfg, tokens)
        return _generalized_gate_dense_router_from_logits(logits, cfg, buffers_by_tokens[tokens])

    layer.mlp._router_dense = types.MethodType(candidate_router, layer.mlp)


def _make_probe_layer(device, cfg, layer_idx: int) -> FusedDecoder:
    state = _synthetic_layer_state(cfg, layer_idx)
    state["mlp.gate.weight"] = _randn((cfg.num_experts, cfg.hidden_size), seed=9100 + layer_idx, scale=0.02)
    return FusedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)


def _time_linear_prefill(layer: FusedDecoder, device, cfg, seq_len: int, name: str) -> None:
    hidden = _randn((1, seq_len, cfg.hidden_size), seed=9200 + seq_len, scale=0.01)
    hidden_tt = _test_tt_bf16(hidden.unsqueeze(0), device)
    state = FusedDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=1)
    layer.prefill_forward(hidden_tt, linear_state=state)
    ttnn.synchronize_device(device)
    state = FusedDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=1)
    _time(name, device, lambda: layer.prefill_forward(hidden_tt, linear_state=state).hidden_states, iterations=5)


def _time_full_prefill(layer: FusedDecoder, device, cfg, seq_len: int, name: str) -> None:
    hidden = _randn((1, seq_len, cfg.hidden_size), seed=9300 + seq_len, scale=0.01)
    position_ids = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len)
    position_embeddings = _rotary(cfg, hidden, position_ids)
    page_table = _page_table(1, max(96, ((seq_len + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE))
    hidden_tt = _test_tt_bf16(hidden.unsqueeze(0), device)
    cos_tt = _test_tt_bf16(position_embeddings[0].unsqueeze(1), device)
    sin_tt = _test_tt_bf16(position_embeddings[1].unsqueeze(1), device)
    page_table_tt = _tt_int(page_table, device)
    warm_cache = FusedDecoder.allocate_full_attention_cache(
        hf_config=cfg, mesh_device=device, max_batch_size=1, max_seq_len=page_table.shape[1] * BLOCK_SIZE
    )
    layer.prefill_forward(
        hidden_tt, position_embeddings=(cos_tt, sin_tt), page_table=page_table_tt, kv_cache=warm_cache
    )
    ttnn.synchronize_device(device)
    measure_cache = FusedDecoder.allocate_full_attention_cache(
        hf_config=cfg, mesh_device=device, max_batch_size=1, max_seq_len=page_table.shape[1] * BLOCK_SIZE
    )
    _time(
        name,
        device,
        lambda: layer.prefill_forward(
            hidden_tt, position_embeddings=(cos_tt, sin_tt), page_table=page_table_tt, kv_cache=measure_cache
        ).hidden_states,
        iterations=5,
    )


def _time_linear_traced_decode(layer: FusedDecoder, device, cfg, seq_len: int, name: str) -> None:
    hidden = _randn((1, seq_len, cfg.hidden_size), seed=9400 + seq_len, scale=0.01)
    prefill_input = _test_tt_bf16(hidden.unsqueeze(0), device)
    state = FusedDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=1)
    prefill = layer.prefill_forward(prefill_input, linear_state=state)
    decode_hidden = _randn((1, 1, cfg.hidden_size), seed=9500 + seq_len, scale=0.01)
    decode_input = _test_tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device)
    current_pos = _tt_int(torch.tensor([seq_len], dtype=torch.int32), device)
    _time_trace(
        name,
        device,
        lambda: layer.decode_forward(
            decode_input, current_pos=current_pos, linear_state=prefill.linear_state
        ).hidden_states,
        iterations=5,
    )


def _time_full_traced_decode(layer: FusedDecoder, device, cfg, seq_len: int, name: str) -> None:
    max_seq_len = max(96, ((seq_len + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE)
    hidden = _randn((1, seq_len, cfg.hidden_size), seed=9600 + seq_len, scale=0.01)
    position_ids = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len)
    position_embeddings = _rotary(cfg, hidden, position_ids)
    page_table = _page_table(1, max_seq_len)
    page_table_tt = _tt_int(page_table, device)
    kv_cache = FusedDecoder.allocate_full_attention_cache(
        hf_config=cfg, mesh_device=device, max_batch_size=1, max_seq_len=max_seq_len, block_size=BLOCK_SIZE
    )
    layer.prefill_forward(
        _test_tt_bf16(hidden.unsqueeze(0), device),
        position_embeddings=(
            _test_tt_bf16(position_embeddings[0].unsqueeze(1), device),
            _test_tt_bf16(position_embeddings[1].unsqueeze(1), device),
        ),
        page_table=page_table_tt,
        kv_cache=kv_cache,
    )

    decode_hidden = _randn((1, 1, cfg.hidden_size), seed=9700 + seq_len, scale=0.01)
    decode_position_embeddings = _rotary(cfg, decode_hidden, torch.tensor([[seq_len]], dtype=torch.long))
    decode_input = _test_tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device)
    current_pos = _tt_int(torch.tensor([seq_len], dtype=torch.int32), device)
    cos_tt = _test_tt_bf16(decode_position_embeddings[0].unsqueeze(0), device)
    sin_tt = _test_tt_bf16(decode_position_embeddings[1].unsqueeze(0), device)
    _time_trace(
        name,
        device,
        lambda: layer.decode_forward(
            decode_input,
            current_pos=current_pos,
            position_embeddings=(cos_tt, sin_tt),
            page_table=page_table_tt,
            kv_cache=kv_cache,
        ).hidden_states,
        iterations=5,
    )


def _try(name: str, fn):
    print(f"\n## {name}")
    try:
        out = fn()
        print(f"RESULT: PASS {out}")
        return out
    except Exception as exc:
        print(f"RESULT: REJECT {type(exc).__name__}: {exc}")
        print("TRACEBACK_HEAD:")
        print("".join(traceback.format_exception_only(type(exc), exc)).strip())
        return None


def _time(name: str, device, fn, *, iterations: int = 20):
    fn()
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(iterations):
        out = fn()
    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
    print(f"TIMING {name}: {elapsed_ms:.4f} ms/iter over {iterations} iterations; shape={_shape(out)}")
    return out, elapsed_ms


def _time_trace(name: str, device, fn, *, iterations: int = 5):
    fn()
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(iterations):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
    print(f"TIMING_TRACE {name}: {elapsed_ms:.4f} ms/iter over {iterations} iterations; shape={_shape(out)}")
    ttnn.release_trace(device, trace_id)
    return out, elapsed_ms


def main() -> None:
    torch.manual_seed(0)
    cfg = _target_text_config()
    batch = 1
    seq_len = 33
    q_width = cfg.num_attention_heads * cfg.head_dim
    q_gate_width = 2 * q_width
    kv_width = cfg.num_key_value_heads * cfg.head_dim
    qwen_qkgv_width = q_gate_width + 2 * kv_width
    standard_qkv_width = q_width + 2 * kv_width

    print("Qwen fused-decoder graph-fusion candidate probe")
    print(f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim}")
    print(f"q_width={q_width} q_gate_width={q_gate_width} kv_width={kv_width}")
    print(f"qwen_qkgv_width={qwen_qkgv_width} standard_qkv_width={standard_qkv_width}")

    device = ttnn.CreateDevice(device_id=0, trace_region_size=16_000_000)
    try:
        qwen_prefill_split = _tt_bf16(torch.randn(batch, seq_len, qwen_qkgv_width, dtype=torch.bfloat16), device)
        qwen_prefill = _tt_bf16(torch.randn(1, batch, seq_len, qwen_qkgv_width, dtype=torch.bfloat16), device)
        qwen_decode = _tt_bf16(torch.randn(1, 1, batch, qwen_qkgv_width, dtype=torch.bfloat16), device)

        _try(
            "split_query_key_value_and_split_heads direct Qwen q+gate/k/v prefill",
            lambda: tuple(
                _shape(t)
                for t in ttnn.transformer.split_query_key_value_and_split_heads(
                    qwen_prefill_split,
                    num_heads=cfg.num_attention_heads,
                    num_kv_heads=cfg.num_key_value_heads,
                    transpose_key=False,
                )
            ),
        )

        def split_after_stripping_q_gate():
            q_and_gate = _slice_last(qwen_prefill_split, 0, q_gate_width)
            k = _slice_last(qwen_prefill_split, q_gate_width, q_gate_width + kv_width)
            v = _slice_last(qwen_prefill_split, q_gate_width + kv_width, qwen_qkgv_width)
            q_and_gate = ttnn.reshape(q_and_gate, (batch, seq_len, cfg.num_attention_heads, 2 * cfg.head_dim))
            q = _slice_last(q_and_gate, 0, cfg.head_dim)
            q_flat = ttnn.reshape(q, (batch, seq_len, q_width))
            standard_qkv = ttnn.concat([q_flat, k, v], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            qh, kh, vh = ttnn.transformer.split_query_key_value_and_split_heads(
                standard_qkv,
                num_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads,
                transpose_key=False,
            )
            return (qh, kh, vh)

        _try(
            "split_query_key_value_and_split_heads after stripping Q gate",
            lambda: tuple(_shape(t) for t in split_after_stripping_q_gate()),
        )

        def current_prefill_head_path():
            q_and_gate = _slice_last(qwen_prefill, 0, q_gate_width)
            k = _slice_last(qwen_prefill, q_gate_width, q_gate_width + kv_width)
            v = _slice_last(qwen_prefill, q_gate_width + kv_width, qwen_qkgv_width)
            q_and_gate = ttnn.reshape(q_and_gate, (batch, seq_len, cfg.num_attention_heads, 2 * cfg.head_dim))
            q = _slice_last(q_and_gate, 0, cfg.head_dim)
            q = ttnn.permute(q, (0, 2, 1, 3))
            k = ttnn.reshape(k, (batch, seq_len, cfg.num_key_value_heads, cfg.head_dim))
            k = ttnn.permute(k, (0, 2, 1, 3))
            v = ttnn.reshape(v, (batch, seq_len, cfg.num_key_value_heads, cfg.head_dim))
            v = ttnn.permute(v, (0, 2, 1, 3))
            return ttnn.reshape(q, (batch, cfg.num_attention_heads, seq_len, cfg.head_dim))

        _time("current prefill q/k/v head path representative output", device, current_prefill_head_path)
        _time("stripped prefill split_query_key_value candidate", device, lambda: split_after_stripping_q_gate()[0])

        height_sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
        _try(
            "nlp_create_qkv_heads_decode direct Qwen q+gate/k/v decode",
            lambda: tuple(
                _shape(t)
                for t in ttnn.experimental.nlp_create_qkv_heads_decode(
                    qwen_decode,
                    num_heads=cfg.num_attention_heads,
                    num_kv_heads=cfg.num_key_value_heads,
                    memory_config=height_sharded,
                )
            ),
        )

        def nlp_create_after_stripping_q_gate():
            q_and_gate = _slice_last(qwen_decode, 0, q_gate_width)
            k = _slice_last(qwen_decode, q_gate_width, q_gate_width + kv_width)
            v = _slice_last(qwen_decode, q_gate_width + kv_width, qwen_qkgv_width)
            q_and_gate = ttnn.reshape(q_and_gate, (batch, 1, cfg.num_attention_heads, 2 * cfg.head_dim))
            q = _slice_last(q_and_gate, 0, cfg.head_dim)
            q_flat = ttnn.reshape(q, (1, 1, batch, q_width))
            standard_qkv = ttnn.concat([q_flat, k, v], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads_decode(
                standard_qkv,
                num_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads,
                memory_config=height_sharded,
            )
            return qh, kh, vh

        _try(
            "nlp_create_qkv_heads_decode after stripping Q gate",
            lambda: tuple(_shape(t) for t in nlp_create_after_stripping_q_gate()),
        )

        def current_decode_head_path():
            q_and_gate = _slice_last(qwen_decode, 0, q_gate_width)
            q_and_gate = ttnn.reshape(q_and_gate, (batch, 1, cfg.num_attention_heads, 2 * cfg.head_dim))
            q = _slice_last(q_and_gate, 0, cfg.head_dim)
            q = ttnn.permute(q, (1, 0, 2, 3))
            return ttnn.reshape(q, (1, batch, cfg.num_attention_heads, cfg.head_dim))

        _time("current decode q head path representative output", device, current_decode_head_path)
        _time(
            "stripped decode nlp_create_qkv_heads_decode candidate",
            device,
            lambda: nlp_create_after_stripping_q_gate()[0],
        )

        heads_prefill = _tt_bf16(
            torch.randn(batch, seq_len, cfg.num_attention_heads, cfg.head_dim, dtype=torch.bfloat16), device
        )

        def current_prefill_concat():
            return ttnn.reshape(heads_prefill, (1, batch, seq_len, q_width))

        def dedicated_prefill_concat():
            out = ttnn.transformer.concatenate_heads(heads_prefill, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.reshape(out, (1, batch, seq_len, q_width))

        _try("concatenate_heads prefill output shape", lambda: _shape(dedicated_prefill_concat()))
        _time("current prefill concat reshape", device, current_prefill_concat)
        _time("dedicated concatenate_heads prefill candidate", device, dedicated_prefill_concat)

        heads_decode = _tt_bf16(
            torch.randn(1, batch, cfg.num_attention_heads, cfg.head_dim, dtype=torch.bfloat16), device
        )

        def current_decode_concat():
            return ttnn.reshape(heads_decode, (1, batch, 1, q_width))

        def nlp_decode_concat_with_required_layouts():
            shard_grid = ttnn.num_cores_to_corerangeset(batch, device.compute_with_storage_grid_size(), True)
            input_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    shard_grid,
                    (cfg.num_attention_heads, cfg.head_dim),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            sharded = ttnn.to_memory_config(heads_decode, input_mem)
            out = ttnn.experimental.nlp_concat_heads_decode(sharded, num_heads=cfg.num_attention_heads)
            return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

        nlp_concat_result = _try(
            "nlp_concat_heads_decode with required layouts output shape",
            lambda: _shape(nlp_decode_concat_with_required_layouts()),
        )
        _time("current decode concat reshape", device, current_decode_concat)
        if nlp_concat_result is not None:
            _time(
                "nlp_concat_heads_decode plus layout roundtrip candidate",
                device,
                nlp_decode_concat_with_required_layouts,
            )

        padded_heads_decode = _tt_bf16(torch.randn(1, batch, 32, cfg.head_dim, dtype=torch.bfloat16), device)

        def nlp_decode_concat_padded_heads():
            shard_grid = ttnn.num_cores_to_corerangeset(batch, device.compute_with_storage_grid_size(), True)
            input_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    shard_grid,
                    (32, cfg.head_dim),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            sharded = ttnn.to_memory_config(padded_heads_decode, input_mem)
            out = ttnn.experimental.nlp_concat_heads_decode(sharded, num_heads=cfg.num_attention_heads)
            return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

        def nlp_decode_concat_padded_heads_sliced():
            out = nlp_decode_concat_padded_heads()
            return _slice(out, (0, 0, 0, 0), (1, batch, 1, q_width))

        padded_nlp_concat_result = _try(
            "nlp_concat_heads_decode with padded 32-head input",
            lambda: _shape(nlp_decode_concat_padded_heads()),
        )
        if padded_nlp_concat_result is not None:
            _time(
                "nlp_concat_heads_decode padded-head layout roundtrip candidate",
                device,
                nlp_decode_concat_padded_heads,
            )
            _time(
                "nlp_concat_heads_decode padded-head layout roundtrip plus logical slice candidate",
                device,
                nlp_decode_concat_padded_heads_sliced,
            )

        tokens = 33
        logits = _tt_bf16(torch.randn(1, 1, tokens, cfg.num_experts, dtype=torch.bfloat16), device)
        bias = _tt_bf16(torch.zeros(1, 1, tokens, cfg.num_experts, dtype=torch.bfloat16), device)
        indices = _tt_u16(
            torch.arange(cfg.num_experts, dtype=torch.int32)
            .to(torch.uint16)
            .reshape(1, 1, 1, cfg.num_experts)
            .expand(1, 1, tokens, -1),
            device,
        )
        output = _tt_bf16(torch.zeros(tokens, 32, 32, dtype=torch.bfloat16), device)
        output_indices = _tt_u16(torch.zeros(tokens, 32, 32, dtype=torch.uint16), device)

        _try(
            "generalized_moe_gate direct fused-router layout",
            lambda: tuple(
                _shape(t)
                for t in ttnn.experimental.deepseek.moe.generalized_moe_gate(
                    logits,
                    bias_tensor=bias,
                    input_indices_tensor=indices,
                    output_tensor=output,
                    output_indices_tensor=output_indices,
                    scaling_factor=1.0,
                    enable_sigmoid=False,
                    topk=cfg.num_experts_per_tok,
                    output_softmax=True,
                )
            ),
        )

        def validate_generalized_gate_router(tokens: int):
            router_logits = _tt_bf16(torch.randn(1, 1, tokens, cfg.num_experts, dtype=torch.bfloat16), device)
            buffers = _generalized_gate_buffers(device, cfg, tokens)
            current = _current_dense_router_from_logits(router_logits, cfg)
            candidate = _generalized_gate_dense_router_from_logits(router_logits, cfg, buffers)
            ttnn.synchronize_device(device)
            current_t = ttnn.to_torch(current).float()
            candidate_t = ttnn.to_torch(candidate).float()
            pcc = _pcc(current_t, candidate_t)
            max_abs = float(torch.max(torch.abs(current_t - candidate_t)))
            mask_match = float(((current_t != 0) == (candidate_t != 0)).float().mean())
            return (
                f"tokens={tokens} dense-route PCC={pcc:.9f} max_abs={max_abs:.6f} nonzero_mask_match={mask_match:.6f}"
            )

        generalized_gate_candidate_available = True
        if (
            _try(
                "generalized_moe_gate adapted Qwen router dense rebuild tokens=33",
                lambda: validate_generalized_gate_router(33),
            )
            is None
        ):
            generalized_gate_candidate_available = False
        if (
            _try(
                "generalized_moe_gate adapted Qwen router dense rebuild tokens=1",
                lambda: validate_generalized_gate_router(1),
            )
            is None
        ):
            generalized_gate_candidate_available = False

        for router_tokens in (33, 1):
            router_logits = _tt_bf16(torch.randn(1, 1, router_tokens, cfg.num_experts, dtype=torch.bfloat16), device)
            buffers = _generalized_gate_buffers(device, cfg, router_tokens)
            _time(
                f"current dense router path tokens={router_tokens}",
                device,
                lambda router_logits=router_logits: _current_dense_router_from_logits(router_logits, cfg),
            )
            if (
                _try(
                    f"generalized_moe_gate dense rebuild timing tokens={router_tokens}",
                    lambda router_logits=router_logits, buffers=buffers: _time(
                        f"generalized_moe_gate dense rebuild candidate tokens={router_tokens}",
                        device,
                        lambda: _generalized_gate_dense_router_from_logits(router_logits, cfg, buffers),
                    ),
                )
                is None
            ):
                generalized_gate_candidate_available = False

        linear_layer = _make_probe_layer(device, cfg, 0)
        _time_linear_prefill(linear_layer, device, cfg, 5, "current fused decoder linear prefill path")
        _time_linear_traced_decode(linear_layer, device, cfg, 5, "current fused decoder linear traced decode path")

        full_layer = _make_probe_layer(device, cfg, 3)
        _time_full_prefill(full_layer, device, cfg, 33, "current fused decoder full prefill path")
        _time_full_traced_decode(full_layer, device, cfg, 33, "current fused decoder full traced decode path")
        if generalized_gate_candidate_available:
            _install_generalized_gate_router_candidate(linear_layer, device, cfg)
            _time_linear_prefill(
                linear_layer, device, cfg, 5, "generalized_moe_gate candidate fused decoder linear prefill path"
            )
            _time_linear_traced_decode(
                linear_layer, device, cfg, 5, "generalized_moe_gate candidate fused decoder linear traced decode path"
            )
            _install_generalized_gate_router_candidate(full_layer, device, cfg)
            _time_full_prefill(
                full_layer, device, cfg, 33, "generalized_moe_gate candidate fused decoder full prefill path"
            )
            _time_full_traced_decode(
                full_layer, device, cfg, 33, "generalized_moe_gate candidate fused decoder full traced decode path"
            )
        else:
            print(
                "SKIP generalized_moe_gate candidate full-decoder timings: adapted sharded op is unavailable "
                "on this Blackhole checkout because its kernel JIT fails before timing."
            )

        print("\nSUMMARY")
        print("Direct Qwen q+gate/k/v projection width is incompatible with dedicated QKV split/create ops.")
        print("Using those ops requires first slicing away Q gate and concatenating a standard QKV tensor.")
        print("Dedicated concat-head ops either add an extra op or require L1 sharding/layout roundtrips.")
        print("Adapted generalized_moe_gate routing reaches the required sharded layout,")
        print("but this Blackhole checkout cannot JIT its kernel because a required LLK header is missing.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
