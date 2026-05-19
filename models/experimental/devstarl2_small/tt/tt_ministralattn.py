# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Tenstorrent text self-attention for Hugging Face Ministral3 (``Ministral3Attention``). Extends :class:`~models.tt_transformers.tt.attention.Attention` with Llama-4-style **query scaling after RoPE** (see ``get_llama_4_attn_scale`` in Hugging Face ``modeling_ministral3``). Weight layout and fused ``wqkv``/``wo`` handling are unchanged from the base TT Llama-family attention; HF checkpoints still map ``q_proj``/``k_proj``/``v_proj`` into ``wq``/``wk``/``wv`` via the usual loaders. Llama-4 scali...

from __future__ import annotations

import math

import ttnn

from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.common import Mode


class TtMinistralAttention(Attention):
    """TT Ministral attention like base :class:`Attention` with Llama-4 post-RoPE Q scaling.

    HF ``rope_parameters`` supply ``llama_4_scaling_beta`` and ``original_max_position_embeddings``; missing → no scaling.

    Rope override: the upstream attention was switched in the TT-Transformers
    rebase to the new ``ttnn.experimental.rotary_embedding_hf`` kernel (sharded
    HF rope). That kernel pairs with the new ``HfRotarySetup`` cos/sin layout
    (per-batch height-sharded slices). The devstral text path was validated
    against the **legacy** ``ttnn.experimental.rotary_embedding`` op with full
    ``[1, 1, max_seq_len, head_dim]`` cos/sin caches, and post-rebase the new
    kernel silently misrotates Q/K for the devstral decoder (caption decoded
    to repeating tokens like "pigeon"). To stay bit-for-bit consistent with
    the pre-rebase known-good path, we override the rope hooks here to call
    the legacy op, and pair this with
    ``HfRotarySetupOld`` inside :class:`TtMinistral3RotaryEmbedding`.
    """

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

        # Replace the upstream HF rope hooks (which target the new sharded
        # ``rotary_embedding_hf`` kernel) with the legacy implementations
        # below. Non-HF paths (mllama / qk-fused) are untouched.
        if self.use_hf_rope:
            self.rotary_embedding_decode = self._hf_rope_decode_legacy
            self.rotary_embedding_prefill = self._hf_rope_prefill_legacy

        _decode_rope = self.rotary_embedding_decode
        _prefill_rope = self.rotary_embedding_prefill

        def rotary_embedding_decode_wrapped(q, k, rot_mats, current_pos):
            q, k = _decode_rope(q, k, rot_mats, current_pos)
            return self._apply_llama4_query_scale_decode(q, current_pos), k

        def rotary_embedding_prefill_wrapped(q, k, rot_mats):
            q, k = _prefill_rope(q, k, rot_mats)
            return self._apply_llama4_query_scale_prefill(q), k

        self.rotary_embedding_decode = rotary_embedding_decode_wrapped
        self.rotary_embedding_prefill = rotary_embedding_prefill_wrapped

    def _hf_rope_prefill_legacy(self, q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats):
        """HF-style prefill rope via legacy ``ttnn.experimental.rotary_embedding``.

        Pre-rebase behaviour: cos/sin are full ``[1, 1, seq_len, head_dim]``
        TILE tensors built by :class:`TtMinistral3RotaryEmbedding`.
        """
        if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)
        if k_heads_1KSD_pre_rot.dtype != ttnn.bfloat16:
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)

        q_heads_1QSD = ttnn.experimental.rotary_embedding(q_heads_1QSD_pre_rot, rot_mats[0], rot_mats[1])
        k_heads_1KSD = ttnn.experimental.rotary_embedding(k_heads_1KSD_pre_rot, rot_mats[0], rot_mats[1])
        return q_heads_1QSD, k_heads_1KSD

    def _hf_rope_decode_legacy(self, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats, current_pos):
        """HF-style decode rope via legacy ``ttnn.experimental.rotary_embedding``.

        Pre-rebase behaviour: per-batch loop with ``token_idx=0`` because
        cos/sin are already per-position slices from
        ``HfRotarySetupOld.get_rot_mats`` (shape ``[1, 1, batch_padded,
        head_dim]`` TILE). Merge in L1, then reshard back to the original Q/K
        memory configs and reshape away the tile padding the legacy op adds.
        """
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

        # Legacy op pads head count axis up to 32 (tile alignment) — restate
        # the logical head count, then slice to the real n_local_*_heads.
        q_heads_1BQD = ttnn.reshape(
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
        """HF Llama-4 scale in TT float32 (shape matches ``pos_tt``).

        Use ``ttnn.add(..., 1.0)`` not ``ones_like``—ROW_MAJOR ``ones_like`` can host-upload inside trace capture."""
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

    def _apply_llama4_query_scale_prefill(self, q_heads):
        if not self._llama4_scaling_enabled():
            return q_heads
        pos_tt = getattr(self, "_ministral_prefill_position_ids_tt", None)
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
        """Optional device ``position_ids`` ``[batch,seq]`` or ``[seq]`` for Llama-4 prefill Q scaling.

        Ignored when Llama-4 scaling params are absent."""
        self._ministral_prefill_position_ids_tt = position_ids
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
            self._ministral_prefill_position_ids_tt = None

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


# Alias __name__ to Attention so checkpoint prefixes match layers.{i}.attention.* (see get_state_dict_prefix).
TtMinistralAttention.__name__ = "Attention"
TtMinistralAttention.__qualname__ = "Attention"


__all__ = ["TtMinistralAttention"]
