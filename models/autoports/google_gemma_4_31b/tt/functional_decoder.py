# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functional, single-device TTNN decoder layer for ``google/gemma-4-31B``.

The public forward paths consume and return device tensors. Weight conversion is
confined to :meth:`from_state_dict`; input/output conversion belongs to callers.
Both layer kinds use paged KV caches in the acceptance path. Decode accepts
tensor-valued current positions and is designed to be captured and replayed by
TTNN trace. Prefill accepts any logical sequence length through the TTNN tensor's
logical shape; tile padding is an internal storage detail.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention.operations import (
    PREFILL_SDPA_MAX_SEQ,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    chunked_prefill_sdpa_sliding,
    effective_block_size,
    prefill_sdpa_program_config,
    split_qkv_heads_prefill,
)
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

HF_MODEL_ID = "google/gemma-4-31B"
HF_ADVERTISED_CONTEXT = 262_144
FULL_ATTN_Q_CHUNK = 1024
MLP_CHUNK = 4096


@dataclass(frozen=True)
class DecoderContract:
    layer_kind: str
    hidden_size: int
    max_position_embeddings: int
    sliding_window: int | None


def _text_config(hf_config):
    return getattr(hf_config, "text_config", hf_config)


def _validate_target_config(hf_config, layer_idx: int) -> DecoderContract:
    tc = _text_config(hf_config)
    expected = {
        "hidden_size": 5376,
        "num_hidden_layers": 60,
        "num_attention_heads": 32,
        "max_position_embeddings": HF_ADVERTISED_CONTEXT,
    }
    for name, value in expected.items():
        actual = int(getattr(tc, name))
        if actual != value:
            raise ValueError(f"{HF_MODEL_ID} expects {name}={value}, got {actual}")
    if not 0 <= layer_idx < len(tc.layer_types):
        raise ValueError(f"layer_idx {layer_idx} is outside [0, {len(tc.layer_types)})")
    layer_kind = tc.layer_types[layer_idx]
    if layer_kind not in ("sliding_attention", "full_attention"):
        raise ValueError(f"unsupported Gemma 4 layer kind: {layer_kind}")
    return DecoderContract(
        layer_kind=layer_kind,
        hidden_size=int(tc.hidden_size),
        max_position_embeddings=int(tc.max_position_embeddings),
        sliding_window=int(tc.sliding_window) if layer_kind == "sliding_attention" else None,
    )


class FunctionalDecoder(LightweightModule):
    """One real-shape Gemma 4 31B text decoder layer on a 1x1 mesh."""

    def __init__(self, *, layer: Gemma4DecoderLayer, contract: DecoderContract, layer_idx: int, mesh_device):
        self.layer = layer
        self.contract = contract
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        tensor_cache_path=None,
        dtype=ttnn.bfloat16,
        bounded_sliding_kv_cache: bool = True,
        **kwargs,
    ) -> "FunctionalDecoder":
        """Load one canonical HF layer and convert all weights before runtime."""
        if kwargs:
            raise TypeError(f"unsupported FunctionalDecoder kwargs: {sorted(kwargs)}")
        if mesh_device.get_num_devices() != 1:
            raise ValueError("functional decoder stage requires a single 1x1 mesh")
        contract = _validate_target_config(hf_config, layer_idx)
        model_args = Gemma4ModelArgs.from_hf_config(hf_config)
        mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1))
        layer = Gemma4DecoderLayer(
            mesh_device=mesh_device,
            hf_config=model_args,
            state_dict=state_dict,
            layer_idx=layer_idx,
            ccl_manager=None,
            dtype=dtype,
            tensor_cache_path=tensor_cache_path,
            mesh_config=mesh_config,
            max_seq_len=contract.max_position_embeddings,
            max_local_batch_size=32,
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
        )
        return cls(layer=layer, contract=contract, layer_idx=layer_idx, mesh_device=mesh_device)

    def _prefill_attention(self, hidden_states, *, rope_mats, page_table, kv_cache, user_id, valid_seq_len):
        """Gemma prefill attention with a multi-core-safe head concatenation.

        The demo's legacy ``nlp_concat_heads`` uses one core and exceeds P150
        L1 for the full-attention geometry (32 heads x 512). Permute+reshape is
        mathematically identical and distributes the data movement.
        """
        attention = self.layer.self_attn
        config = attention.config
        weights = attention.weights
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
        num_kv_heads = config.num_key_value_heads
        block_size = effective_block_size(k_cache, config.head_dim, num_kv_heads)
        modulo = {"cache_position_modulo": config.cache_position_modulo} if config.cache_position_modulo else {}
        k_fill, v_fill = k, v
        if config.cache_position_modulo is not None:
            fill_len = ((min(valid_seq_len, k.shape[-2]) + block_size - 1) // block_size) * block_size
            if fill_len < k.shape[-2]:
                k_fill = ttnn.slice(k, [0, 0, 0, 0], [1, k.shape[1], fill_len, k.shape[3]])
                v_fill = ttnn.slice(v, [0, 0, 0, 0], [1, v.shape[1], fill_len, v.shape[3]])
        ttnn.experimental.paged_fill_cache(
            k_cache, k_fill, page_table, batch_idx=user_id, block_size=block_size, **modulo
        )
        ttnn.experimental.paged_fill_cache(
            v_cache, v_fill, page_table, batch_idx=user_id, block_size=block_size, **modulo
        )
        if k_fill is not k:
            k_fill.deallocate(True)
            v_fill.deallocate(True)

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
        transposed = ttnn.permute(sdpa, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sdpa.deallocate(True)
        concatenated = ttnn.reshape(transposed, [1, 1, seq_len, config.num_attention_heads * config.head_dim])
        output = ttnn.linear(concatenated, weights.o_proj)
        concatenated.deallocate(True)
        return output

    def _streaming_full_prefill_attention(self, hidden_states, *, rope_mats, page_table, kv_cache, user_id):
        """Execute advertised-context full attention without 32-bit tensor overflow."""
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
            transposed = ttnn.permute(sdpa, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            sdpa.deallocate(True)
            concatenated = ttnn.reshape(transposed, [1, 1, end - start, config.num_attention_heads * config.head_dim])
            projected_outputs.append(ttnn.linear(concatenated, weights.o_proj))
            concatenated.deallocate(True)
        result = ttnn.concat(projected_outputs, dim=2)
        for output in projected_outputs:
            output.deallocate(True)
        return result

    @staticmethod
    def _chunked_full_attention(q, k_cache, v_cache, page_table, user_id, head_dim):
        """Bound individual full-attention kernel duration for P150 watchdogs."""
        num_heads, seq_len = q.shape[1], q.shape[2]
        user_page_table = page_table
        owns_page_table = False
        if page_table.shape[0] > 1:
            user_page_table = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
            owns_page_table = True
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        outputs = []
        for start in range(0, seq_len, FULL_ATTN_Q_CHUNK):
            chunk_len = min(FULL_ATTN_Q_CHUNK, seq_len - start)
            q_chunk = ttnn.slice(q, [0, 0, start, 0], [1, num_heads, start + chunk_len, head_dim])
            output = ttnn.transformer.chunked_scaled_dot_product_attention(
                q_chunk,
                k_cache,
                v_cache,
                user_page_table,
                chunk_start_idx=start,
                scale=1.0,
                program_config=program_config,
            )
            q_chunk.deallocate(True)
            outputs.append(output)
        if owns_page_table:
            user_page_table.deallocate(True)
        result = ttnn.concat(outputs, dim=2)
        for output in outputs:
            output.deallocate(True)
        return result

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
        """Host-free decoder composition shared by prefill and decode."""
        residual = hidden_states
        normed = self.layer.input_layernorm.forward(hidden_states)
        attn_input = normed
        if not is_decode and batch_size > 1:
            attn_input = ttnn.reshape(normed, [batch_size, 1, normed.shape[-2] // batch_size, -1])
        if is_decode:
            attn_output = self.layer.self_attn(
                attn_input,
                rope_mats=rope_mats,
                position_idx=current_position,
                position_idx_cache=current_position_cache,
                page_table=page_table,
                kv_cache=kv_cache,
                is_decode=True,
                token_index=token_index,
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
                chunk_len = min(MLP_CHUNK, normed.shape[-2] - start)
                chunk = ttnn.slice(normed, [0, 0, start, 0], [1, 1, start + chunk_len, normed.shape[-1]])
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
        combined = ttnn.add(residual, hidden_states)
        residual.deallocate(True)
        hidden_states.deallocate(True)
        if self.layer.layer_scalar != 1.0:
            scaled = ttnn.mul(combined, self.layer.layer_scalar)
            combined.deallocate(True)
            combined = scaled
        return combined

    def prefill_forward(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
        batch_size: int = 1,
        user_id=0,
        valid_seq_len: int | None = None,
    ):
        """Run paged causal prefill and populate ``kv_cache`` on device."""
        logical_seq_len = int(valid_seq_len if valid_seq_len is not None else hidden_states.shape[-2] // batch_size)
        if not 1 <= logical_seq_len <= self.contract.max_position_embeddings:
            raise ValueError(f"logical prefill length must be in [1, {self.contract.max_position_embeddings}]")
        if batch_size > 1:
            outputs = []
            for batch_idx in range(batch_size):
                start = batch_idx * logical_seq_len
                user_input = ttnn.slice(
                    hidden_states,
                    [0, 0, start, 0],
                    [1, 1, start + logical_seq_len, hidden_states.shape[-1]],
                )
                user_output = self.prefill_forward(
                    user_input,
                    rope_mats=rope_mats,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    batch_size=1,
                    user_id=batch_idx,
                    valid_seq_len=logical_seq_len,
                )
                user_input.deallocate(True)
                outputs.append(user_output)
            output = ttnn.concat(outputs, dim=2)
            for user_output in outputs:
                user_output.deallocate(True)
            return output

        alignment = (
            128
            if logical_seq_len > PREFILL_SDPA_MAX_SEQ and self.contract.layer_kind == "full_attention"
            else ttnn.TILE_SIZE
        )
        padded_len = ((logical_seq_len + alignment - 1) // alignment) * alignment
        padded_input = hidden_states
        padded_rope = rope_mats
        owns_padding = padded_len != logical_seq_len
        if owns_padding:
            padded_input = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, padded_len - logical_seq_len), (0, 0)], 0.0)
            padded_rope = tuple(
                ttnn.pad(table, [(0, 0), (0, 0), (0, padded_len - logical_seq_len), (0, 0)], value)
                for table, value in zip(rope_mats, (1.0, 0.0))
            )
        output = self._forward_device(
            padded_input,
            rope_mats=padded_rope,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=False,
            batch_size=batch_size,
            user_id=user_id,
            valid_seq_len=logical_seq_len,
        )
        if owns_padding:
            sliced = ttnn.slice(output, [0, 0, 0, 0], [1, 1, logical_seq_len, output.shape[-1]])
            output.deallocate(True)
            # ``ttnn.pad`` may return a logical view over an already tile-padded
            # buffer. Deallocating that view would invalidate caller-owned input
            # or RoPE storage, which is reused across batched users and traces.
            output = sliced
        return output

    def decode_forward(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
        current_position,
        current_position_cache=None,
        token_index: int = 0,
        batch_size: int = 1,
    ):
        """Run one paged decode step; all mutable positions are device tensors.

        ``rope_mats`` must be the 2-D row-major lookup tables so capture does not
        bake a Python position into RoPE. ``current_position`` is uint32 for the
        lookup; ``current_position_cache`` is int32 for cache ops when supplied.
        """
        return self._forward_device(
            hidden_states,
            rope_mats=rope_mats,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
            current_position=current_position,
            current_position_cache=current_position_cache,
            token_index=token_index,
            batch_size=batch_size,
        )

    def forward(self, hidden_states, *, mode: str, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
