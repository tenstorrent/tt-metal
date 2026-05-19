# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Ministral3 attention: base TT Attention + Llama-4 post-RoPE Q scaling + legacy HF rope.

from __future__ import annotations

import math

import ttnn

from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.common import Mode


class TtMinistralAttention(Attention):
    """Ministral attention with Llama-4 Q scaling; legacy HF rope (not rotary_embedding_hf)."""

    def __init__(
        self,
        *args,
        llama_4_scaling_beta: float | None = None,
        original_max_position_embeddings: int | None = None,
        **kwargs,
    ):
        self.llama_4_scaling_beta = llama_4_scaling_beta
        self.original_max_position_embeddings = original_max_position_embeddings
        super().__init__(*args, **kwargs)

        if self.use_hf_rope:  # legacy rotary_embedding, not rotary_embedding_hf
            self.rotary_embedding_decode = self._hf_rope_decode_legacy
            self.rotary_embedding_prefill = self._hf_rope_prefill_legacy

        _decode_rope = self.rotary_embedding_decode
        _prefill_rope = self.rotary_embedding_prefill

        def rotary_embedding_decode_wrapped(q, k, rot_mats, current_pos):
            q, k = _decode_rope(q, k, rot_mats, current_pos)
            return self._apply_llama4_query_scale_decode(q, current_pos), k

        def rotary_embedding_prefill_wrapped(q, k, rot_mats):
            q, k = _prefill_rope(q, k, rot_mats)
            pos_tt = self._prefill_position_ids_for_llama4_scale
            return self._apply_llama4_query_scale_prefill(q, pos_tt), k

        self.rotary_embedding_decode = rotary_embedding_decode_wrapped
        self.rotary_embedding_prefill = rotary_embedding_prefill_wrapped
        self._prefill_position_ids_for_llama4_scale: ttnn.Tensor | None = None

    def _hf_rope_prefill_legacy(self, q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats):
        """Legacy HF prefill rope (full TILE cos/sin from TtMinistral3RotaryEmbedding)."""
        if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)
        if k_heads_1KSD_pre_rot.dtype != ttnn.bfloat16:
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)

        q_heads_1QSD = ttnn.experimental.rotary_embedding(q_heads_1QSD_pre_rot, rot_mats[0], rot_mats[1])
        k_heads_1KSD = ttnn.experimental.rotary_embedding(k_heads_1KSD_pre_rot, rot_mats[0], rot_mats[1])
        return q_heads_1QSD, k_heads_1KSD

    def _hf_rope_decode_legacy(self, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats, current_pos):
        """Legacy HF decode rope (per-batch slices from HfRotarySetupOld.get_rot_mats)."""
        cos, sin = rot_mats[0], rot_mats[1]
        B_iter = self.batch_size_per_device_group

        if q_heads_pre_rot_1BQD.dtype != ttnn.bfloat16:
            q_heads_pre_rot_1BQD = ttnn.typecast(q_heads_pre_rot_1BQD, dtype=ttnn.bfloat16)
        if k_heads_pre_rot_1BKD.dtype != ttnn.bfloat16:
            k_heads_pre_rot_1BKD = ttnn.typecast(k_heads_pre_rot_1BKD, dtype=ttnn.bfloat16)

        q_out_mem = q_heads_pre_rot_1BQD.memory_config()
        k_out_mem = k_heads_pre_rot_1BKD.memory_config()

        q_il_parts = []
        k_il_parts = []
        for b in range(B_iter):
            q_b = q_heads_pre_rot_1BQD[:, b : b + 1, :, :]
            k_b = k_heads_pre_rot_1BKD[:, b : b + 1, :, :]
            cos_b = cos[:, :, b : b + 1, :]
            sin_b = sin[:, :, b : b + 1, :]
            q_rot = ttnn.experimental.rotary_embedding(q_b, cos_b, sin_b, 0)
            k_rot = ttnn.experimental.rotary_embedding(k_b, cos_b, sin_b, 0)
            q_il_parts.append(ttnn.to_memory_config(q_rot, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16))
            k_il_parts.append(ttnn.to_memory_config(k_rot, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16))
            ttnn.deallocate(q_rot)
            ttnn.deallocate(k_rot)

        if B_iter == 1:
            q_merged_il = q_il_parts[0]
            k_merged_il = k_il_parts[0]
        else:
            q_merged_il = ttnn.concat(q_il_parts, dim=1)
            k_merged_il = ttnn.concat(k_il_parts, dim=1)
            for t in q_il_parts:
                ttnn.deallocate(t)
            for t in k_il_parts:
                ttnn.deallocate(t)

        q_heads_1BQD = ttnn.interleaved_to_sharded(q_merged_il, q_out_mem)
        k_heads_1BKD = ttnn.interleaved_to_sharded(k_merged_il, k_out_mem)
        ttnn.deallocate(q_merged_il)
        ttnn.deallocate(k_merged_il)

        q_heads_1BQD = ttnn.reshape(  # legacy rope pads heads to 32 tiles
            q_heads_1BQD,
            (1, self.batch_size_per_device_group, self.n_local_heads, self.head_dim),
            (1, self.batch_size_per_device_group, 32, self.head_dim),
        )
        k_heads_1BKD = ttnn.reshape(
            k_heads_1BKD,
            (1, self.batch_size_per_device_group, self.n_local_kv_heads, self.head_dim),
            (1, self.batch_size_per_device_group, 32, self.head_dim),
        )
        q_heads_1BQD = q_heads_1BQD[:, :, : self.n_local_heads]
        k_heads_1BKD = k_heads_1BKD[:, :, : self.n_local_kv_heads]
        return q_heads_1BQD, k_heads_1BKD

    def _llama4_scaling_enabled(self) -> bool:
        return self.llama_4_scaling_beta is not None and self.original_max_position_embeddings is not None

    def _llama4_scale_factor_from_positions_ttnn(self, pos_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Llama-4 scale factor in float32 (use add(1.0), not ones_like, for trace safety)."""
        orig = float(self.original_max_position_embeddings)
        beta = float(self.llama_4_scaling_beta)
        pos_f = ttnn.typecast(pos_tt, ttnn.float32)
        ratio = ttnn.divide(pos_f, orig)
        floored = ttnn.floor(ratio)
        log_term = ttnn.log1p(floored)
        scaled = ttnn.mul(log_term, beta)
        return ttnn.add(scaled, 1.0)

    def _reshape_decode_positions(self, current_pos: ttnn.Tensor, batch_dim: int) -> ttnn.Tensor:
        """Return positions with shape ``[1, batch_dim]`` for per-row scale (matches ``q`` batch axis)."""
        sh = tuple(current_pos.shape)
        numel = math.prod(sh) if sh else 0
        if numel == batch_dim:
            return ttnn.reshape(current_pos, (1, batch_dim))
        if numel > batch_dim:
            flat = ttnn.reshape(current_pos, (1, 1, 1, numel))
            sliced = flat[:, :, :, :batch_dim]
            return ttnn.reshape(sliced, (1, batch_dim))
        raise ValueError(f"Ministral Llama-4 scale: current_pos has {numel} elements but q_heads batch is {batch_dim}")

    def _apply_llama4_query_scale_decode(self, q_heads, current_pos):
        if not self._llama4_scaling_enabled():
            return q_heads
        b = int(q_heads.shape[1])
        pos_row = self._reshape_decode_positions(current_pos, b)
        scale_f = self._llama4_scale_factor_from_positions_ttnn(pos_row)
        scale_4d = ttnn.reshape(scale_f, (1, b, 1, 1))
        scale_tt = ttnn.typecast(scale_4d, ttnn.bfloat16)

        q_bf16 = q_heads if q_heads.dtype == ttnn.bfloat16 else ttnn.typecast(q_heads, dtype=ttnn.bfloat16)
        out = ttnn.mul(q_bf16, scale_tt, dtype=ttnn.bfloat16)
        if q_heads.dtype != ttnn.bfloat16:
            out = ttnn.typecast(out, dtype=q_heads.dtype)
        ttnn.deallocate(scale_tt)
        if q_bf16 is not q_heads:
            ttnn.deallocate(q_bf16)
        return out

    def _apply_llama4_query_scale_prefill(self, q_heads, position_ids: ttnn.Tensor | None = None):
        if not self._llama4_scaling_enabled():
            return q_heads
        pos_tt = position_ids
        if pos_tt is None:
            return q_heads

        sh = tuple(q_heads.shape)
        if len(sh) != 4:
            return q_heads
        batch_dim, seq_dim = sh[0], sh[2]
        shp = tuple(pos_tt.shape)
        if len(shp) == 2 and shp[0] == batch_dim and shp[1] == seq_dim:
            scale_f = self._llama4_scale_factor_from_positions_ttnn(pos_tt)
        elif len(shp) == 1 and shp[0] == seq_dim and batch_dim == 1:
            scale_f = self._llama4_scale_factor_from_positions_ttnn(ttnn.reshape(pos_tt, (1, seq_dim)))
        else:
            return q_heads

        scale_4d = ttnn.reshape(scale_f, (batch_dim, 1, seq_dim, 1))
        scale_tt = ttnn.typecast(scale_4d, ttnn.bfloat16)

        q_bf16 = q_heads if q_heads.dtype == ttnn.bfloat16 else ttnn.typecast(q_heads, dtype=ttnn.bfloat16)
        out = ttnn.mul(q_bf16, scale_tt, dtype=ttnn.bfloat16)
        if q_heads.dtype != ttnn.bfloat16:
            out = ttnn.typecast(out, dtype=q_heads.dtype)
        ttnn.deallocate(scale_tt)
        if q_bf16 is not q_heads:
            ttnn.deallocate(q_bf16)
        return out

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        position_ids: ttnn.Tensor | None = None,
    ):
        """Prefill with optional device position_ids for Llama-4 Q scaling."""
        self._prefill_position_ids_for_llama4_scale = position_ids
        try:
            return super().forward_prefill(
                x_11SH,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
            )
        finally:
            self._prefill_position_ids_for_llama4_scale = None

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode=Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        position_ids=None,
    ):
        if mode == Mode.PREFILL:
            return self.forward_prefill(
                x,
                rot_mats,
                user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
                position_ids=position_ids,
            )
        return super().forward(
            x,
            current_pos,
            rot_mats,
            user_id=user_id,
            mode=mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )


# ModelArgs.get_state_dict_prefix(self.__class__.__name__, layer_num) expects "Attention"
# (checkpoint keys are layers.{i}.attention.*, not TtMinistralAttention).
TtMinistralAttention.__name__ = "Attention"
TtMinistralAttention.__qualname__ = "Attention"


__all__ = ["TtMinistralAttention"]
