# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Tenstorrent text self-attention for Hugging Face Ministral3 (``Ministral3Attention``).

Extends :class:`~models.tt_transformers.tt.attention.Attention` with Llama-4-style **query
scaling after RoPE** (see ``get_llama_4_attn_scale`` in Hugging Face ``modeling_ministral3``).

Weight layout and fused ``wqkv``/``wo`` handling are unchanged from the base TT Llama-family
attention; HF checkpoints still map ``q_proj``/``k_proj``/``v_proj`` into ``wq``/``wk``/``wv``
via the usual loaders.

Llama-4 scaling is computed **entirely on device** with ``ttnn`` (no host position read, no
PyTorch in the scaling path).

Prefill: pass ``position_ids`` as a **device** ``ttnn.Tensor`` (integer positions, shape
``[batch, seq]``) through :meth:`forward` / :meth:`forward_prefill`. Decode uses ``current_pos``
from the device path without reading it back to the host.
"""

from __future__ import annotations

import math

import ttnn

from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.common import Mode


def _devstral_kv_cache_alloc_len(configuration) -> int:
    """
    Third dimension size for dense KV tensors when demos set ``ModelArgs.max_kv_cache_seq_len``.
    Keeps tt_transformers Attention unchanged for other models.
    """
    cap = getattr(configuration, "max_kv_cache_seq_len", None)
    if cap is None:
        return int(configuration.max_seq_len)
    kv = int(cap)
    if kv > int(configuration.max_seq_len):
        raise ValueError(f"max_kv_cache_seq_len ({kv}) exceeds max_seq_len ({configuration.max_seq_len})")
    return kv


class TtMinistralAttention(Attention):
    """
        Ministral3 attention: same TT path as :class:`Attention`, plus post-RoPE Q scaling.

    Long prefill Llama‑4 scaling: ``reshape(scale, (B,1,S,1))`` can exceed tensix L1 circular-buffer
    budgets when ``S`` is ~100k+. Apply scaling along the sequence dimension in bounded chunks instead.

    Base attention's prefill KV L1 shard path (``interleaved_to_sharded`` scaled by ``seq_len``) is disabled
    via ``min_kv_prefill_shard_seqlen``: at very long sequence lengths TT otherwise requests enormous L1 KV tiles.

        Parameters
        ----------
        llama_4_scaling_beta, original_max_position_embeddings
            From HF ``config.rope_parameters`` (e.g. ``llama_4_scaling_beta``, ``original_max_position_embeddings``).
            If either is ``None``, scaling is a no-op (not numerically equal to HF).
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

        # Tokens per Llama‑4 scale chunk on prefill; keeps reshape/mul footprints under typical L1 limits.
        self._llama4_prefill_q_scale_chunk_tokens = 2048

        # Base Attention prefill shards K/V for fill_cache via ``interleaved_to_sharded`` +
        # ``ModelArgs.get_attn_kv_prefill_mem_config(seq_len)``. That shard sizing grows ~O(seq_len);
        # at ~100k–250k+ tokens TT requests hundreds of MiB of L1 and aborts (bank_manager.cpp).
        # Keep K/V interleaved in DRAM for Ministral3 prefill fill_cache paths used by Devstral demos.
        self.min_kv_prefill_shard_seqlen = 2**62

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

    def init_kv_cache(self, configuration, weight_cache_path):
        """
        Same as ``Attention.init_kv_cache`` but allocates the dense KV buffers with length
        ``max_kv_cache_seq_len`` when set on ``configuration`` (``ModelArgs``), so RoPE /
        ``max_seq_len`` can stay at 262k while DRAM tracks a smaller run-bound cap.
        """
        if self.paged_attention_config:
            super().init_kv_cache(configuration, weight_cache_path)
            self._kv_alloc_seq_len = int(configuration.max_seq_len)
            return

        kv_slots = _devstral_kv_cache_alloc_len(configuration)

        self.layer_past = [
            ttnn.zeros(
                (self.batch_size_per_device_group, self.n_local_kv_heads, kv_slots, self.head_dim),
                dtype=self.kv_cache_dtype,
                layout=self.args.get_attn_weights_layout(),
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        ]
        self._kv_alloc_seq_len = kv_slots

    def _llama4_scaling_enabled(self) -> bool:
        return self.llama_4_scaling_beta is not None and self.original_max_position_embeddings is not None

    def _llama4_scale_factor_from_positions_ttnn(self, pos_tt: ttnn.Tensor) -> ttnn.Tensor:
        """
        ``scaling = 1 + beta * log(1 + floor(pos / original_max_position_embeddings))`` (float32),
        same as HF ``get_llama_4_attn_scale``, shape matches ``pos_tt``.
        """
        orig = float(self.original_max_position_embeddings)
        beta = float(self.llama_4_scaling_beta)
        pos_f = ttnn.typecast(pos_tt, ttnn.float32)
        ratio = ttnn.divide(pos_f, orig)
        floored = ttnn.floor(ratio)
        log_term = ttnn.log1p(floored)
        scaled = ttnn.mul(log_term, beta)
        ones = ttnn.ones_like(pos_f)
        return ttnn.add(ones, scaled)

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
        nh, hd = int(sh[1]), int(sh[3])
        shp = tuple(pos_tt.shape)
        pos_2d = None
        if len(shp) == 2 and shp[0] == batch_dim and shp[1] == seq_dim:
            pos_2d = pos_tt
        elif len(shp) == 1 and shp[0] == seq_dim and batch_dim == 1:
            pos_2d = ttnn.reshape(pos_tt, (1, seq_dim))
        else:
            return q_heads

        chunk_lim = max(512, int(self._llama4_prefill_q_scale_chunk_tokens))
        if seq_dim <= chunk_lim:
            return self._llama4_mul_q_prefill_scaled(q_heads, batch_dim, seq_dim, pos_2d, q_heads.dtype)

        parts: list = []
        for cs in range(0, seq_dim, chunk_lim):
            ce = min(cs + chunk_lim, seq_dim)
            sub = ce - cs
            q_c = ttnn.slice(q_heads, (0, 0, cs, 0), (batch_dim, nh, ce, hd))
            pos_c = ttnn.slice(pos_2d, (0, cs), (batch_dim, ce))
            parts.append(self._llama4_mul_q_prefill_scaled(q_c, batch_dim, sub, pos_c, q_heads.dtype))
        merged = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for p in parts:
            ttnn.deallocate(p)
        return merged

    def _llama4_mul_q_prefill_scaled(
        self,
        q_part,
        batch_dim: int,
        sub_seq: int,
        pos_2d_part,
        out_dtype,
    ):
        """``q_part`` [:,:,sub_seq,...], ``pos_2d_part`` [B, sub_seq]; returns scaled q dtype ``out_dtype``."""
        scale_f = self._llama4_scale_factor_from_positions_ttnn(pos_2d_part)
        scale_4d = ttnn.reshape(scale_f, (batch_dim, 1, sub_seq, 1))
        scale_tt = ttnn.typecast(scale_4d, ttnn.bfloat16)
        q_bf16 = q_part if q_part.dtype == ttnn.bfloat16 else ttnn.typecast(q_part, dtype=ttnn.bfloat16)
        out = ttnn.mul(q_bf16, scale_tt, dtype=ttnn.bfloat16)
        ttnn.deallocate(scale_tt)
        if q_bf16 is not q_part:
            ttnn.deallocate(q_bf16)
        if out_dtype != ttnn.bfloat16:
            out = ttnn.typecast(out, dtype=out_dtype)
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
        """
        Optional ``position_ids``: device ``ttnn.Tensor`` of integer positions, shape ``[batch, seq]``
        (or ``[seq]`` when ``batch == 1``), dtype typically ``uint32`` / ``int32``. Required for
        Llama-4 Q scaling on prefill when scaling is enabled.
        """
        self._ministral_prefill_position_ids_tt = position_ids
        try:
            if int(x_11SH.shape[0]) == 1:
                seq_len_chk = int(x_11SH.shape[-2])
                cap = getattr(self, "_kv_alloc_seq_len", self.max_seq_len)
                if seq_len_chk > cap:
                    raise RuntimeError(
                        f"Prefill seq_len ({seq_len_chk}) exceeds KV allocation ({cap}); "
                        "Increase ModelArgs.max_kv_cache_seq_len / max_seq_len or shorten prompt."
                    )
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


# ``Attention.__init__`` uses ``self.__class__.__name__`` for checkpoint prefixes (see ``get_state_dict_prefix``).
# Alias so this subclass resolves like ``Attention`` for ``layers.{i}.attention.*`` weight keys.
TtMinistralAttention.__name__ = "Attention"
TtMinistralAttention.__qualname__ = "Attention"


__all__ = ["TtMinistralAttention"]
