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

import torch
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

        cache_k = torch.zeros((self.batch_size_per_device_group, self.n_local_kv_heads, kv_slots, self.head_dim))
        cache_v = torch.zeros((self.batch_size_per_device_group, self.n_local_kv_heads, kv_slots, self.head_dim))

        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                dtype=self.kv_cache_dtype,
                layout=self.args.get_attn_weights_layout(),
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=(
                    f"{weight_cache_path}/kvcache_{k_or_v.shape}"
                    if weight_cache_path and not configuration.dummy_weights
                    else None
                ),
            )
            for k_or_v in [cache_k, cache_v]
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
