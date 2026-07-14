# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Measure the shard-advisor L1 width-sharded norm/eltwise candidate.

This is not production code. It is a reproducible rejection probe for the
advisor recommendation to keep the residual/norm/eltwise chain in L1
width-sharded layout.
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_functional_decoder import (
    DECODE_CACHE_POSITION,
    EMITTED_BATCH_SIZE,
    EMITTED_DECODE_CACHE_LEN,
    _assert_pcc,
    _load_real_model_or_skip,
    _tt_tensor,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_optimized_decoder import (
    _run_reference_decode_for_layer,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    build_decode_mask,
    build_rope_tables,
    build_update_indices,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import OptimizedDecoder


def _l1_width_config(batch: int, width: int) -> ttnn.MemoryConfig:
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    physical_height = batch * ttnn.TILE_SIZE
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [physical_height, width // 64], ttnn.ShardOrientation.ROW_MAJOR),
    )


class L1ChainCandidateDecoder(OptimizedDecoder):
    def _to_l1_width(self, tensor: ttnn.Tensor, width: int) -> ttnn.Tensor:
        return ttnn.interleaved_to_sharded(
            tensor,
            (8, 8),
            [self.batch * ttnn.TILE_SIZE, width // 64],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    def _decode_attention_mlp(self, hidden_states: ttnn.Tensor, attn: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.cfg
        hidden_l1 = self._to_l1_width(hidden_states, cfg.hidden_size)
        attn = ttnn.reshape(
            attn,
            [1, self.batch, 1, cfg.num_attention_heads * cfg.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_out = self._decode_dram_sharded_matmul(attn, self.o_decode_weight, k=cfg.hidden_size, n=cfg.hidden_size)
        hidden_cfg = _l1_width_config(self.batch, cfg.hidden_size)
        attn_residual = ttnn.add(
            self._to_l1_width(attn_out, cfg.hidden_size),
            hidden_l1,
            dtype=self.policy.activation_dtype,
            memory_config=hidden_cfg,
        )
        post_norm = ttnn.rms_norm(
            attn_residual,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.post_attention_layernorm_weight,
            memory_config=hidden_cfg,
        )
        post_norm_dram = ttnn.sharded_to_interleaved(post_norm, ttnn.DRAM_MEMORY_CONFIG)
        gate = self._decode_dram_sharded_matmul(
            post_norm_dram, self.gate_decode_weight, k=cfg.hidden_size, n=cfg.intermediate_size
        )
        up = self._decode_dram_sharded_matmul(
            post_norm_dram, self.up_decode_weight, k=cfg.hidden_size, n=cfg.intermediate_size
        )
        intermediate_cfg = _l1_width_config(self.batch, cfg.intermediate_size)
        gate_l1 = self._to_l1_width(gate, cfg.intermediate_size)
        up_l1 = self._to_l1_width(up, cfg.intermediate_size)
        gated = ttnn.multiply(
            ttnn.silu(gate_l1, memory_config=intermediate_cfg),
            up_l1,
            dtype=self.policy.activation_dtype,
            memory_config=intermediate_cfg,
        )
        gated_dram = ttnn.sharded_to_interleaved(gated, ttnn.DRAM_MEMORY_CONFIG)
        mlp_out = self._decode_dram_sharded_matmul(
            gated_dram, self.down_decode_weight, k=cfg.intermediate_size, n=cfg.hidden_size
        )
        final_l1 = ttnn.add(
            self._to_l1_width(mlp_out, cfg.hidden_size),
            attn_residual,
            dtype=self.policy.activation_dtype,
            memory_config=hidden_cfg,
        )
        return ttnn.sharded_to_interleaved(final_l1, ttnn.DRAM_MEMORY_CONFIG)


class L1ResidualCandidateDecoder(L1ChainCandidateDecoder):
    def _decode_attention_mlp(self, hidden_states: ttnn.Tensor, attn: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.cfg
        hidden_l1 = self._to_l1_width(hidden_states, cfg.hidden_size)
        attn = ttnn.reshape(
            attn,
            [1, self.batch, 1, cfg.num_attention_heads * cfg.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_out = self._decode_dram_sharded_matmul(attn, self.o_decode_weight, k=cfg.hidden_size, n=cfg.hidden_size)
        hidden_cfg = _l1_width_config(self.batch, cfg.hidden_size)
        attn_residual = ttnn.add(
            self._to_l1_width(attn_out, cfg.hidden_size),
            hidden_l1,
            dtype=self.policy.activation_dtype,
            memory_config=hidden_cfg,
        )
        post_norm = ttnn.rms_norm(
            attn_residual,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.post_attention_layernorm_weight,
            memory_config=hidden_cfg,
        )
        post_norm_dram = ttnn.sharded_to_interleaved(post_norm, ttnn.DRAM_MEMORY_CONFIG)
        gate = self._decode_dram_sharded_matmul(
            post_norm_dram, self.gate_decode_weight, k=cfg.hidden_size, n=cfg.intermediate_size
        )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = self._decode_dram_sharded_matmul(
            post_norm_dram, self.up_decode_weight, k=cfg.hidden_size, n=cfg.intermediate_size
        )
        gated = ttnn.multiply(gate, up, dtype=self.policy.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mlp_out = self._decode_dram_sharded_matmul(
            gated, self.down_decode_weight, k=cfg.intermediate_size, n=cfg.hidden_size
        )
        final_l1 = ttnn.add(
            self._to_l1_width(mlp_out, cfg.hidden_size),
            attn_residual,
            dtype=self.policy.activation_dtype,
            memory_config=hidden_cfg,
        )
        return ttnn.sharded_to_interleaved(final_l1, ttnn.DRAM_MEMORY_CONFIG)


def _build_decode_case(decoder_cls, model, mesh):
    hf_config = model.config
    torch.manual_seed(20260740)
    prefix_hidden = torch.randn(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, hf_config.hidden_size, dtype=torch.bfloat16)
    decode_hidden = torch.randn(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    reference, key_cache, value_cache, _, _ = _run_reference_decode_for_layer(
        model.model.layers[0], model.model.rotary_emb, hf_config, 0, prefix_hidden, decode_hidden
    )
    decoder = decoder_cls.from_state_dict(
        model.state_dict(),
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh,
        batch=EMITTED_BATCH_SIZE,
        max_seq_len=EMITTED_DECODE_CACHE_LEN,
    )
    return (
        decoder,
        reference,
        _tt_tensor(decode_hidden.reshape(1, EMITTED_BATCH_SIZE, 1, hf_config.hidden_size), mesh),
        _tt_tensor(key_cache, mesh),
        _tt_tensor(value_cache, mesh),
        build_update_indices(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, mesh),
        build_rope_tables(hf_config, 1, mesh, start_pos=DECODE_CACHE_POSITION),
        build_decode_mask(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, EMITTED_DECODE_CACHE_LEN, mesh),
    )


def _time_trace(decoder_cls, model, label: str):
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        decoder, reference, tt_input, key_cache, value_cache, update_idxs, rope, attention_mask = _build_decode_case(
            decoder_cls, model, mesh
        )
        position_cos, position_sin = rope
        decoder.decode_forward(
            tt_input,
            key_cache=key_cache,
            value_cache=value_cache,
            update_idxs_tensor=update_idxs,
            position_cos=position_cos,
            position_sin=position_sin,
            attention_mask=attention_mask,
        )
        ttnn.synchronize_device(mesh)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced_output = decoder.decode_forward(
            tt_input,
            key_cache=key_cache,
            value_cache=value_cache,
            update_idxs_tensor=update_idxs,
            position_cos=position_cos,
            position_sin=position_sin,
            attention_mask=attention_mask,
        )
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        rounds = []
        for _ in range(5):
            start = time.perf_counter()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            rounds.append((time.perf_counter() - start) * 1000.0)
        actual = ttnn.to_torch(traced_output).reshape_as(reference).to(torch.float32)
        _assert_pcc(reference, actual, 0.99)
        ttnn.release_trace(mesh, trace_id)
        print(f"{label}_traced_decode_ms_rounds=" + ",".join(f"{v:.6f}" for v in rounds))
        print(f"{label}_traced_decode_ms_best={min(rounds):.6f}")
    finally:
        ttnn.close_mesh_device(mesh)


def main() -> None:
    model = _load_real_model_or_skip()
    _time_trace(OptimizedDecoder, model, "default")
    try:
        _time_trace(L1ChainCandidateDecoder, model, "l1_chain_candidate")
    except Exception as exc:
        print(f"l1_chain_candidate_failed={type(exc).__name__}: {exc}")
    _time_trace(L1ResidualCandidateDecoder, model, "l1_residual_candidate")


if __name__ == "__main__":
    main()
