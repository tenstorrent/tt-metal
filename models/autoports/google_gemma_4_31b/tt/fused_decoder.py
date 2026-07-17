# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Graph-fused single-device decoder layer for ``google/gemma-4-31B``.

This stage preserves :class:`FunctionalDecoder`'s public contract and paged
cache semantics while replacing graph fragments with dedicated/folded TTNN
operations. Runtime inputs and outputs remain device tensors.
"""

from __future__ import annotations

import ttnn
from models.autoports.google_gemma_4_31b.tt.decode_head_grid import decode_head_core_grid, decode_head_sub_core_grids
from models.autoports.google_gemma_4_31b.tt.functional_decoder import FULL_ATTN_Q_CHUNK, MLP_CHUNK, FunctionalDecoder
from models.demos.gemma4.tt.attention.operations import (
    PREFILL_SDPA_MAX_SEQ,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    apply_rope_decode_peruser,
    chunked_prefill_sdpa_sliding,
    effective_block_size,
    prefill_sdpa_program_config,
    split_qkv_heads_decode,
    split_qkv_heads_prefill,
)

# ``nlp_concat_heads`` uses one core. These chunks keep its input at or below
# 1 MiB on P150: 32 heads * chunk * head_dim * sizeof(bf16).
SLIDING_HEAD_CONCAT_CHUNK = 64


class _FusedSharedMLP:
    """Dense GeGLU MLP with GELU folded into the gate projection matmul."""

    def __init__(self, functional_mlp):
        self.gate_proj = functional_mlp.gate_proj
        self.up_proj = functional_mlp.up_proj
        self.down_proj = functional_mlp.down_proj
        self.mesh_config = functional_mlp.mesh_config
        self.ccl_manager = functional_mlp.ccl_manager

    def __call__(self, hidden_states):
        gelu = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)
        m_tiles = (hidden_states.shape[-2] + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
        # The tuned 1D multicast geometry used by TTNN's auto-selection for
        # decode and the seq-128 acceptance workload, with GELU placed in the
        # program config so it is genuinely compiled into the matmul kernel.
        gate_program_config = None
        if m_tiles <= 4:
            gate_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=7,
                out_block_h=m_tiles,
                out_block_w=7,
                per_core_M=m_tiles,
                per_core_N=7,
                fuse_batch=False,
                fused_activation=gelu,
                mcast_in0=True,
                num_global_cb_receivers=0,
            )
        gate = ttnn.linear(hidden_states, self.gate_proj, program_config=gate_program_config)
        if gate_program_config is None:
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
        up = ttnn.linear(hidden_states, self.up_proj)
        hidden = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)
        output = ttnn.linear(hidden, self.down_proj)
        hidden.deallocate(True)
        return output


class FusedDecoder(FunctionalDecoder):
    """One real-shape Gemma 4 31B layer with graph fusions enabled."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer.shared_mlp = _FusedSharedMLP(self.layer.shared_mlp)

    @staticmethod
    def _concatenate_heads(sdpa, *, num_heads: int, head_dim: int):
        """Use the dedicated head-concat kernel in bounded-L1 sequence chunks."""
        seq_len = sdpa.shape[-2]
        # The full-attention minimum legal tile still needs 2,208,512 B of
        # single-core CB storage on P150 (limit 1,572,864 B). Keep the proven
        # multi-core structural rewrite for head_dim=512.
        if head_dim >= 512 or seq_len > 128:
            transposed = ttnn.permute(sdpa, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.reshape(transposed, [1, 1, seq_len, num_heads * head_dim])

        chunk_size = SLIDING_HEAD_CONCAT_CHUNK
        chunks = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            heads = sdpa
            if start or end != seq_len:
                heads = ttnn.slice(sdpa, [0, 0, start, 0], [1, num_heads, end, head_dim])
            chunks.append(ttnn.experimental.nlp_concat_heads(heads, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            if heads is not sdpa:
                heads.deallocate(True)
        if len(chunks) == 1:
            return chunks[0]
        output = ttnn.concat(chunks, dim=2)
        for chunk in chunks:
            chunk.deallocate(True)
        return output

    def _prefill_attention(self, hidden_states, *, rope_mats, page_table, kv_cache, user_id, valid_seq_len):
        attention = self.layer.self_attn
        config, weights = attention.config, attention.weights
        if not config.is_sliding and hidden_states.shape[-2] * config.num_attention_heads * config.head_dim >= 2**32:
            return self._streaming_full_prefill_attention(
                hidden_states,
                rope_mats=rope_mats,
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user_id,
            )

        qkv = apply_qkv_projection(hidden_states, weights)
        q, k, v = split_qkv_heads_prefill(qkv, config, weights.is_global, tp=1, kv_replicated=False)
        qkv.deallocate(True)
        q = apply_per_head_norm(q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
        k = apply_per_head_norm(k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        v = apply_per_head_norm(v, None, config.rms_norm_eps, with_scale=False)
        cos_cache, sin_cache = rope_mats
        q = apply_rope(q, cos_cache, sin_cache)
        k = apply_rope(k, cos_cache, sin_cache)

        k_cache, v_cache = kv_cache
        block_size = effective_block_size(k_cache, config.head_dim, config.num_key_value_heads)
        modulo = {"cache_position_modulo": config.cache_position_modulo} if config.cache_position_modulo else {}
        if config.cache_position_modulo is not None and valid_seq_len < k.shape[-2]:
            # Padded rows cannot be bulk-filled after a circular-window wrap:
            # their modulo slots still belong to live logical tokens. Preserve
            # the functional decoder's exact logical-tail cache ownership.
            self._fill_bounded_sliding_cache_exact(
                k_cache,
                v_cache,
                k,
                v,
                page_table,
                user_id=user_id,
                valid_seq_len=valid_seq_len,
                block_size=block_size,
                num_kv_heads=config.num_key_value_heads,
                cache_position_modulo=config.cache_position_modulo,
            )
        else:
            ttnn.experimental.paged_fill_cache(
                k_cache, k, page_table, batch_idx=user_id, block_size=block_size, **modulo
            )
            ttnn.experimental.paged_fill_cache(
                v_cache, v, page_table, batch_idx=user_id, block_size=block_size, **modulo
            )

        seq_len = q.shape[-2]
        if seq_len > PREFILL_SDPA_MAX_SEQ and config.is_sliding:
            sdpa = chunked_prefill_sdpa_sliding(q, k, v, config.sliding_window, config.head_dim, scale=1.0)
        elif seq_len > PREFILL_SDPA_MAX_SEQ:
            sdpa = self._chunked_full_attention(q, k_cache, v_cache, page_table, user_id, config.head_dim)
        else:
            sdpa = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=1.0,
                sliding_window_size=config.sliding_window if config.is_sliding else None,
                program_config=prefill_sdpa_program_config(config.head_dim, seq_len),
            )
        q.deallocate(True)
        k.deallocate(True)
        v.deallocate(True)
        concatenated = self._concatenate_heads(sdpa, num_heads=config.num_attention_heads, head_dim=config.head_dim)
        sdpa.deallocate(True)
        output = ttnn.linear(concatenated, weights.o_proj)
        concatenated.deallocate(True)
        return output

    def _streaming_full_prefill_attention(self, hidden_states, *, rope_mats, page_table, kv_cache, user_id):
        attention = self.layer.self_attn
        config, weights = attention.config, attention.weights
        k_cache, v_cache = kv_cache
        block_size = effective_block_size(k_cache, config.head_dim, config.num_key_value_heads)
        if FULL_ATTN_Q_CHUNK % block_size:
            raise ValueError("full-attention stream chunk must be page-block aligned")
        cos_cache, sin_cache = rope_mats
        seq_len = hidden_states.shape[-2]
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        projected_outputs = []
        for start in range(0, seq_len, FULL_ATTN_Q_CHUNK):
            end = min(start + FULL_ATTN_Q_CHUNK, seq_len)
            hidden_chunk = ttnn.slice(hidden_states, [0, 0, start, 0], [1, 1, end, hidden_states.shape[-1]])
            qkv = apply_qkv_projection(hidden_chunk, weights)
            hidden_chunk.deallocate(True)
            q, k, v = split_qkv_heads_prefill(qkv, config, weights.is_global, tp=1, kv_replicated=False)
            qkv.deallocate(True)
            q = apply_per_head_norm(q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
            k = apply_per_head_norm(k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
            v = apply_per_head_norm(v, None, config.rms_norm_eps, with_scale=False)
            cos_chunk = ttnn.slice(cos_cache, [0, 0, start, 0], [1, 1, end, cos_cache.shape[-1]])
            sin_chunk = ttnn.slice(sin_cache, [0, 0, start, 0], [1, 1, end, sin_cache.shape[-1]])
            q = apply_rope(q, cos_chunk, sin_chunk)
            k = apply_rope(k, cos_chunk, sin_chunk)
            cos_chunk.deallocate(True)
            sin_chunk.deallocate(True)

            first_block, last_block = start // block_size, end // block_size
            page_chunk = ttnn.slice(page_table, [user_id, first_block], [user_id + 1, last_block])
            ttnn.experimental.paged_fill_cache(k_cache, k, page_chunk, batch_idx=0, block_size=block_size)
            ttnn.experimental.paged_fill_cache(v_cache, v, page_chunk, batch_idx=0, block_size=block_size)
            page_chunk.deallocate(True)
            k.deallocate(True)
            v.deallocate(True)
            sdpa = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                k_cache,
                v_cache,
                page_table,
                chunk_start_idx=start,
                scale=1.0,
                program_config=program_config,
            )
            q.deallocate(True)
            concatenated = self._concatenate_heads(sdpa, num_heads=config.num_attention_heads, head_dim=config.head_dim)
            sdpa.deallocate(True)
            projected_outputs.append(ttnn.linear(concatenated, weights.o_proj))
            concatenated.deallocate(True)
        result = ttnn.concat(projected_outputs, dim=2)
        for output in projected_outputs:
            output.deallocate(True)
        return result

    def _decode_attention(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
        current_position,
        current_position_cache,
        batch_size,
    ):
        """Single-device decode with movement and cache-update fusions."""
        attention = self.layer.self_attn
        config, weights = attention.config, attention.weights

        # Write QKV directly to L1: the create-heads kernel's Blackhole DRAM
        # reader is both slower and affected by an odd-row alignment erratum.
        qkv = ttnn.linear(hidden_states, weights.wqkv, memory_config=ttnn.L1_MEMORY_CONFIG)
        q, k, v = split_qkv_heads_decode(qkv, config, weights.is_global, tp=1, kv_replicated=False)
        qkv.deallocate(True)

        q_sharded_mem = q.memory_config()
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        q = apply_per_head_norm(q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)
        k = apply_per_head_norm(k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        v = apply_per_head_norm(v, None, config.rms_norm_eps, with_scale=False)

        cos_cache, sin_cache = rope_mats
        cos_pos = ttnn.embedding(current_position, cos_cache, layout=ttnn.TILE_LAYOUT)
        sin_pos = ttnn.embedding(current_position, sin_cache, layout=ttnn.TILE_LAYOUT)
        cos_pos = ttnn.unsqueeze_to_4D(cos_pos)
        sin_pos = ttnn.unsqueeze_to_4D(sin_pos)
        if batch_size == 1:
            q = apply_rope(q, cos_pos, sin_pos, token_index=0)
            k = apply_rope(k, cos_pos, sin_pos, token_index=0)
        else:
            cos_b = ttnn.transpose(cos_pos, 1, 2)[:, :batch_size, :, :]
            sin_b = ttnn.transpose(sin_pos, 1, 2)[:, :batch_size, :, :]
            q = apply_rope_decode_peruser(q, cos_b, sin_b)
            k = apply_rope_decode_peruser(k, cos_b, sin_b)

        cache_position = current_position_cache if current_position_cache is not None else current_position
        k_cache, v_cache = kv_cache
        block_size = effective_block_size(k_cache, config.head_dim, config.num_key_value_heads)
        if config.cache_position_modulo is None:
            # The full-attention cache has its native head/block view, so the
            # dedicated two-cache kernel can replace two independent updates.
            # Its contract requires equal, non-overlapping K/V core grids.
            device_grid = self.mesh_device.compute_with_storage_grid_size()
            grid_x = min(batch_size, device_grid.x)
            while batch_size % grid_x:
                grid_x -= 1
            grid_h = batch_size // grid_x
            k_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_h - 1))})
            v_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, grid_h), ttnn.CoreCoord(grid_x - 1, 2 * grid_h - 1))}
            )
            k_memory_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, config.head_dim),
                core_grid=k_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            v_memory_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, config.head_dim),
                core_grid=v_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            k = ttnn.to_memory_config(k, k_memory_config)
            v = ttnn.to_memory_config(v, v_memory_config)
            ttnn.experimental.paged_fused_update_cache(
                k_cache,
                k,
                v_cache,
                v,
                update_idxs_tensor=cache_position,
                page_table=page_table,
            )
        else:
            k = ttnn.to_memory_config(k, q_sharded_mem)
            v = ttnn.to_memory_config(v, q_sharded_mem)
            modulo = {"cache_position_modulo": config.cache_position_modulo}
            ttnn.experimental.paged_update_cache(
                k_cache,
                k,
                update_idxs_tensor=cache_position,
                page_table=page_table,
                block_size=block_size,
                num_kv_heads=config.num_key_value_heads,
                **modulo,
            )
            ttnn.experimental.paged_update_cache(
                v_cache,
                v,
                update_idxs_tensor=cache_position,
                page_table=page_table,
                block_size=block_size,
                num_kv_heads=config.num_key_value_heads,
                **modulo,
            )
        k.deallocate(True)
        v.deallocate(True)

        sdpa_grid = (
            ttnn.CoreCoord(8, 4)
            if config.head_dim >= 512
            else ttnn.CoreCoord(
                self.mesh_device.compute_with_storage_grid_size().x,
                self.mesh_device.compute_with_storage_grid_size().y,
            )
        )
        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_grid,
            q_chunk_size=32,
            k_chunk_size=64,
            exp_approx_mode=False,
        )
        core_grid = decode_head_core_grid(self.mesh_device, batch_size)
        head_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, config.head_dim),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=cache_position,
            page_table_tensor=page_table,
            scale=1.0,
            sliding_window_size=config.sliding_window if config.is_sliding else None,
            # GQA validation currently rejects sharded SDPA output. Materialize
            # in DRAM and perform the one required reshard for decode concat.
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program_config,
            block_size=block_size,
            num_kv_heads=config.num_key_value_heads,
            **(
                {"cache_position_modulo": config.cache_position_modulo}
                if config.cache_position_modulo is not None
                else {}
            ),
        )
        q.deallocate(True)
        sdpa_interleaved = sdpa
        sdpa = ttnn.to_memory_config(sdpa_interleaved, head_memory_config)
        sdpa_interleaved.deallocate(True)
        concatenated = ttnn.experimental.nlp_concat_heads_decode(
            sdpa,
            num_heads=config.num_attention_heads,
            sub_core_grids=decode_head_sub_core_grids(self.mesh_device, core_grid),
        )
        sdpa.deallocate(True)
        output = ttnn.sharded_to_interleaved(concatenated, ttnn.DRAM_MEMORY_CONFIG)
        concatenated.deallocate(True)
        projected = ttnn.linear(output, weights.o_proj)
        output.deallocate(True)
        if projected.shape[2] != batch_size:
            padded = projected
            projected = padded[:, :, :batch_size, :]
            padded.deallocate(True)
        return projected

    def _forward_device(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
        is_decode,
        current_position=None,
        current_position_cache=None,
        token_index=None,
        batch_size=1,
        user_id=0,
        valid_seq_len=None,
    ):
        residual = hidden_states
        normed = self.layer.input_layernorm.forward(hidden_states)
        attn_input = normed
        if not is_decode and batch_size > 1:
            attn_input = ttnn.reshape(normed, [batch_size, 1, normed.shape[-2] // batch_size, -1])
        if is_decode:
            attn_output = self._decode_attention(
                attn_input,
                rope_mats=rope_mats,
                page_table=page_table,
                kv_cache=kv_cache,
                current_position=current_position,
                current_position_cache=current_position_cache,
                batch_size=batch_size,
            )
        else:
            attn_output = self._prefill_attention(
                attn_input,
                rope_mats=rope_mats,
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user_id,
                valid_seq_len=valid_seq_len,
            )
        attn_output = self.layer.post_attention_layernorm.forward(attn_output)
        if not is_decode and batch_size > 1:
            residual = ttnn.reshape(residual, [1, 1, residual.shape[-2] * residual.shape[-3] * residual.shape[0], -1])
        hidden_states = ttnn.add(residual, attn_output)
        attn_output.deallocate(True)

        residual = hidden_states
        normed = self.layer.pre_feedforward_layernorm.forward(hidden_states)
        if not is_decode and normed.shape[-2] > MLP_CHUNK:
            mlp_chunks = []
            for start in range(0, normed.shape[-2], MLP_CHUNK):
                end = min(start + MLP_CHUNK, normed.shape[-2])
                chunk = ttnn.slice(normed, [0, 0, start, 0], [1, 1, end, normed.shape[-1]])
                mlp_chunks.append(self.layer.shared_mlp(chunk))
                chunk.deallocate(True)
            mlp_output = ttnn.concat(mlp_chunks, dim=2)
            for chunk_output in mlp_chunks:
                chunk_output.deallocate(True)
        else:
            mlp_output = self.layer.shared_mlp(normed)
        normed.deallocate(True)
        hidden_states = self.layer.post_feedforward_layernorm.forward(mlp_output)
        mlp_output.deallocate(True)
        activations = None
        if self.layer.layer_scalar != 1.0:
            activations = [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.layer.layer_scalar)]
        combined = ttnn.add(residual, hidden_states, activations=activations)
        residual.deallocate(True)
        hidden_states.deallocate(True)
        return combined
