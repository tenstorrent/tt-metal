# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reproducible graph-fusion candidate benchmark for Qwen3-32B Stage 02."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

import models.autoports.qwen_qwen3_32b.tests.test_functional_decoder as functional_test
import ttnn
from models.autoports.qwen_qwen3_32b.tests.test_fused_decoder import _measure_decoder
from models.autoports.qwen_qwen3_32b.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_PREFILL_SEQUENCE,
    REPRESENTATIVE_LAYER,
    FunctionalDecoder,
)
from models.autoports.qwen_qwen3_32b.tt.fused_decoder import FusedDecoder


class SeparateCacheUpdateDecoder(FusedDecoder):
    """Control: retain all final rewrites except the dedicated fused cache op."""

    def _update_cache(self, key_cache, key, value_cache, value, update_indices):
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=None,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=None,
        )
        return key_cache, value_cache


class SeparateSiluDecoder(FusedDecoder):
    """Control: retain all final rewrites except the binary-NG SiLU fold."""

    def _mlp_forward(self, hidden_states):
        gate = ttnn.matmul(
            hidden_states,
            self.gate_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.matmul(
            hidden_states,
            self.up_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gated = ttnn.multiply(gate, up, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.matmul(
            gated,
            self.down_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class PackedGateUpDecoder(FusedDecoder):
    """Candidate: one shared-LHS gate/up matmul followed by two slices."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gate_up_weight = ttnn.concat(
            [self.gate_weight, self.up_weight],
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mlp_forward(self, hidden_states):
        packed = ttnn.matmul(
            hidden_states,
            self.gate_up_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        shape = list(packed.shape)
        gate_end = list(shape)
        gate_end[-1] = self.intermediate_size
        up_start = [0] * len(shape)
        up_start[-1] = self.intermediate_size
        gate = ttnn.slice(packed, [0] * len(shape), gate_end, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.slice(packed, up_start, shape, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gated = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.matmul(
            gated,
            self.down_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class UncachedMetadataDecoder(FusedDecoder):
    """Control: replay rotary and position slice/repeat ops on every call."""

    def _get_prefill_rotary_views(self, seq_len: int):
        return (
            ttnn.slice(
                self.rotary_cos,
                [0, 0, 0, 0],
                [1, 1, seq_len, self.head_dim],
                [1, 1, 1, 1],
            ),
            ttnn.slice(
                self.rotary_sin,
                [0, 0, 0, 0],
                [1, 1, seq_len, self.head_dim],
                [1, 1, 1, 1],
            ),
        )

    def _get_decode_position_views(self, current_pos: int):
        cos = ttnn.slice(
            self.rotary_cos,
            [0, 0, current_pos, 0],
            [1, 1, current_pos + 1, self.head_dim],
            [1, 1, 1, 1],
        )
        sin = ttnn.slice(
            self.rotary_sin,
            [0, 0, current_pos, 0],
            [1, 1, current_pos + 1, self.head_dim],
            [1, 1, 1, 1],
        )
        position = ttnn.slice(self.position_indices, [current_pos], [current_pos + 1], [1])
        update_indices = ttnn.repeat(
            position,
            ttnn.Shape([self.batch]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return cos, sin, update_indices


class LegacyDecodeViewsDecoder(FusedDecoder):
    """Control: restore the functional decode permute/layout/view sequence."""

    def _create_decode_heads(self, fused_qkv):
        fused_qkv = ttnn.permute(fused_qkv, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [1, 1, self.batch, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        return ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            overlap_qk_coregrid=None,
            memory_config=self.decode_heads_mem_config,
        )

    def _concatenate_decode_heads(self, attention):
        attention = ttnn.to_memory_config(attention, self.decode_concat_input_mem_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(
            attention,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_compute_core_grid,
        )
        attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.slice(
            attention,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.attention_width],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.permute(attention, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)


class LegacyCreateHeadsDecoder(LegacyDecodeViewsDecoder):
    """Control: restore only the functional pre-create-head permute/copy."""

    _concatenate_decode_heads = FusedDecoder._concatenate_decode_heads


class LegacyConcatHeadsDecoder(LegacyDecodeViewsDecoder):
    """Control: restore only the functional post-concat layout sequence."""

    _create_decode_heads = FusedDecoder._create_decode_heads


class DirectConcatViewDecoder(FusedDecoder):
    """Rejected candidate: direct concat slice-to-DRAM plus reshape view."""

    def _concatenate_decode_heads(self, attention):
        attention = ttnn.to_memory_config(attention, self.decode_concat_input_mem_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(
            attention,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_compute_core_grid,
        )
        attention = ttnn.slice(
            attention,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.attention_width],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(attention, [1, self.batch, 1, self.attention_width])


class LegacyConcatSeparateCacheDecoder(LegacyConcatHeadsDecoder):
    """Interaction control: winning concat path with two cache dispatches."""

    _update_cache = SeparateCacheUpdateDecoder._update_cache


class LegacyConcatUncachedMetadataDecoder(LegacyConcatHeadsDecoder):
    """Interaction control: winning concat path without warmed view caching."""

    _get_prefill_rotary_views = UncachedMetadataDecoder._get_prefill_rotary_views
    _get_decode_position_views = UncachedMetadataDecoder._get_decode_position_views


class ShardedQKNormDecoder(FusedDecoder):
    """Candidate: adapt Q/K through legal sharded RMSNorm/RoPE layouts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q_norm_memory_config = ttnn.create_sharded_memory_config(
            shape=(7 * 32, 32),
            core_grid=ttnn.CoreGrid(x=4, y=10),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.k_norm_memory_config = ttnn.create_sharded_memory_config(
            shape=(4 * 32, 32),
            core_grid=ttnn.CoreGrid(x=4, y=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.q_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[4, 10],
            subblock_w=1,
            block_h=7,
            block_w=1,
            inplace=False,
        )
        self.k_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[4, 8],
            subblock_w=1,
            block_h=4,
            block_w=1,
            inplace=False,
        )
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        q_grid = ttnn.num_cores_to_corerangeset(self.num_heads, device_grid, row_wise=True)
        self.q_rope_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(q_grid, [32, self.head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        rope_row_grid = ttnn.num_cores_to_corerangeset(1, device_grid, row_wise=True)
        self.rope_row_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(rope_row_grid, [32, self.head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.sharded_position_views = {}
        self.sharded_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

    def _normalize_and_rotate_decode_qk(self, query, key, current_pos: int):
        query = ttnn.to_memory_config(query, self.q_norm_memory_config)
        key = ttnn.to_memory_config(key, self.k_norm_memory_config)
        query = ttnn.rms_norm(
            query,
            epsilon=self.rms_norm_eps,
            weight=self.q_norm,
            program_config=self.q_norm_program_config,
            memory_config=self.q_norm_memory_config,
        )
        key = ttnn.rms_norm(
            key,
            epsilon=self.rms_norm_eps,
            weight=self.k_norm,
            program_config=self.k_norm_program_config,
            memory_config=self.k_norm_memory_config,
        )
        query = ttnn.to_memory_config(query, self.q_rope_memory_config)
        key = ttnn.to_memory_config(key, self.decode_key_mem_config)
        cos, sin, update_indices = self._get_decode_position_views(current_pos)
        if current_pos not in self.sharded_position_views:
            self.sharded_position_views[current_pos] = (
                ttnn.to_memory_config(cos, self.rope_row_memory_config),
                ttnn.to_memory_config(sin, self.rope_row_memory_config),
            )
        cos, sin = self.sharded_position_views[current_pos]
        query = ttnn.experimental.rotary_embedding(
            query,
            cos,
            sin,
            0,
            memory_config=self.q_rope_memory_config,
        )
        key = ttnn.experimental.rotary_embedding(
            key,
            cos,
            sin,
            0,
            memory_config=self.decode_key_mem_config,
        )
        return query, key, update_indices

    def _decode_attention(self, query, key_cache, value_cache, update_indices):
        # The decode SDPA kernel consumes this Qwen head ordering correctly
        # from interleaved DRAM.  Feeding the numerically correct height-sharded
        # Q tensor directly silently changes the batch/head interpretation, so
        # a materializing adaptation is required for a valid candidate.
        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            is_causal=True,
            cur_pos_tensor=update_indices,
            scale=self.scale,
            program_config=self.sharded_sdpa_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


CANDIDATES = {
    "functional": FunctionalDecoder,
    "fused": FusedDecoder,
    "separate_cache": SeparateCacheUpdateDecoder,
    "separate_silu": SeparateSiluDecoder,
    "packed_gate_up": PackedGateUpDecoder,
    "uncached_metadata": UncachedMetadataDecoder,
    "legacy_decode_views": LegacyDecodeViewsDecoder,
    "legacy_create_heads": LegacyCreateHeadsDecoder,
    "legacy_concat_heads": LegacyConcatHeadsDecoder,
    "direct_concat_view": DirectConcatViewDecoder,
    "legacy_concat_separate_cache": LegacyConcatSeparateCacheDecoder,
    "legacy_concat_uncached_metadata": LegacyConcatUncachedMetadataDecoder,
    "sharded_qk_norm": ShardedQKNormDecoder,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = [name for name in args.candidates.split(",") if name]
    unknown = set(selected) - set(CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64_000_000)
    try:
        config = functional_test._config()
        state = functional_test._real_state()
        reference_layer = functional_test._hf_layer(state, config)
        generator = torch.Generator().manual_seed(4401)
        prefill_host = torch.randn(
            (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        decode_host = torch.randn(
            (1, EMITTED_BATCH, 1, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        reference_prefill, _, _, reference_cache = functional_test._reference_layer(
            reference_layer,
            prefill_host,
            config,
        )
        reference_decode, _, _, _ = functional_test._reference_layer(
            reference_layer,
            decode_host,
            config,
            start_pos=EMITTED_PREFILL_SEQUENCE,
            cache=reference_cache,
        )
        results = {}
        for name in selected:
            try:
                measured = _measure_decoder(CANDIDATES[name], state, config, mesh, prefill_host, decode_host)
                prefill_passed, prefill_pcc = functional_test.comp_pcc(
                    reference_prefill.float(),
                    measured.pop("prefill_output").float(),
                    pcc=0.995,
                )
                decode_passed, decode_pcc = functional_test.comp_pcc(
                    reference_decode.float(),
                    measured.pop("decode_output").float(),
                    pcc=0.995,
                )
                measured["prefill_passed"] = bool(prefill_passed)
                measured["prefill_pcc"] = float(prefill_pcc)
                measured["decode_passed"] = bool(decode_passed)
                measured["decode_pcc"] = float(decode_pcc)
                results[name] = measured
            except RuntimeError as error:
                results[name] = {
                    "runtime_failed": True,
                    "error": str(error).splitlines()[0],
                }
            print(f"CANDIDATE_RESULT {name}={results[name]}")
            gc.collect()

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "model": "Qwen/Qwen3-32B",
                    "layer": REPRESENTATIVE_LAYER,
                    "batch": EMITTED_BATCH,
                    "prefill_sequence": EMITTED_PREFILL_SEQUENCE,
                    "precision": "BF16 weights/activations/cache",
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
