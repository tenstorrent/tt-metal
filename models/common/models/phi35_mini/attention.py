# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phi-3.5-mini attention: ``Attention1D`` with a model-owned single-device decode path and decode weights.

Weights. ``Attention1D`` holds ``wqkv`` / ``wo`` DRAM-interleaved and runs prefill on them (its prefill path
is inherited unchanged). Decode reads a second, DRAM width-sharded copy (``wqkv_decode`` / ``wo_decode``): the
decode matmuls are DRAM-sharded and need that layout, while the 128-row prefill matmuls run fastest as
1D-multicast matmuls, which read interleaved weights only (they take a sharded weight as L1-resident).

Prefill program configs. For the 128-row prefill bucket the config callables return the
``prefill_128_*_prg_config`` 1D-multicast configs; longer prefills keep ``Attention1D``'s defaults.

Decode. The QKV matmul output goes to ``nlp_create_qkv_heads_decode`` still width-sharded, one head per core
(``decode_xqkv_prg_config.per_core_N`` is one head). ``Attention1D`` first copies it to L1 interleaved, which
sends the head split down the op's single-core interleaved program (30.6 us per layer at batch 1 on a P150).
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig, _load_input_device_tensor
from models.common.modules.lazy_weight import LazyWeight

PREFILL_1D_SEQ_LEN = 128


@dataclass
class Phi35MiniAttentionConfig(Attention1DConfig):
    wqkv_decode: LazyWeight | None = None  # DRAM width-sharded copy of wqkv for the decode matmul
    wo_decode: LazyWeight | None = None  # DRAM width-sharded copy of wo for the decode matmul
    prefill_128_xqkv_prg_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None = None
    # Single-user 128-row QKV on the multi-row DRAM-streamed program (dram_stream_workers_per_bank), which reads
    # wqkv_decode, the DRAM width-sharded copy; None runs prefill_128_xqkv_prg_config on wqkv. Any other 128-row call
    # (Attention1D's path) keeps prefill_128_xqkv_prg_config.
    prefill_128_qkv_stream_prg_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None = None
    # Compute config of that streamed QKV (None: li_qkv_prefill_compute_kernel_cfg).
    prefill_128_qkv_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    # Single-user 128-row WO on the multi-row DRAM-streamed program over wo_decode; None runs
    # prefill_128_wo_prg_config on wo. Its compute config: prefill_128_wo_compute_kernel_cfg (None: li_o_prefill_...).
    prefill_128_wo_stream_prg_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None = None
    prefill_128_wo_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    prefill_128_wo_prg_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None = None
    # Decode SDPA reads Q from DRAM (needed when decode_sdpa_prg_config runs off the 8-wide head grid)
    decode_sdpa_q_in_dram: bool = False
    # Single-user 128-row prefill SDPA (Attention1D's default is 64 x 64 chunks). Batched prefill keeps the default:
    # with 32 users x 128 rows, 64 x 64 is faster (417 vs 485 us per layer).
    prefill_128_sdpa_prg_config: ttnn.SDPAProgramConfig | None = None
    # Batched prefill SDPA (several users, unchunked) over the whole compute grid instead of 8 x 8 (see the model's
    # config). P150, 32 users x 32 heads x 128 rows, HiFi4 with fp32 accumulation: 219 us against 429 us.
    prefill_batched_sdpa_prg_config: ttnn.SDPAProgramConfig | None = None
    # Its compute config (None: sdpa_prefill_compute_kernel_cfg)
    prefill_batched_sdpa_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    # Batched prefill QKV matmul output dtype (None: bfloat16). bfloat8_b halves the head split's traffic and gives V
    # in the cache dtype without a cast; Q and K are then rotated from bfloat8_b.
    prefill_batched_qkv_dtype: ttnn.DataType | None = None
    # Batched prefill WO with the residual add fused in (dit_minimal_matmul_addcmul_fused); None keeps a separate add.
    prefill_batched_wo_residual_minimal_matmul_config: ttnn.MinimalMatmulConfig | None = None
    # Batched prefill WO compute config (None: li_o_prefill_compute_kernel_cfg). At 32 users x 128 rows the prefill
    # matmuls are compute-bound, so fidelity sets their time.
    prefill_batched_wo_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    # Batched prefill Q dtype out of the rotary op into SDPA (None: activation_dtype, else bfloat8_b). SDPA's output
    # takes Q's dtype, so this also sets the head concat's and WO's input dtype.
    prefill_batched_q_dtype: ttnn.DataType | None = None
    # Batched prefills of up to this many rows (users x rows; 0: none) keep the QKV output, the head split, the rotated
    # Q / K and the SDPA output in L1 (interleaved over all cores) instead of DRAM. At 32 users x 128 rows these are
    # 13-40 MB each, read by the next op only; the head concat's output stays in DRAM (in L1 it does not fit beside the
    # WO matmul's buffers), and longer batches do not fit at all (16384 rows: 2.5 MB per core for the QKV output).
    prefill_batched_l1_max_rows: int = 0


def _residual_gate(mesh_device, dim: int) -> ttnn.Tensor:
    """A ``[1, 1, 1, dim]`` row of ones: the multiplier of a matmul whose residual add is fused in as an addcmul."""
    import torch

    return ttnn.from_torch(
        torch.ones(1, 1, 1, dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


# minimal_matmul blockings (M, K, N block, subblock h x w) for the packed 32-user prefill: 80..100 tile rows over the
# P150's 10 grid rows, where the default blockings leave rows of cores idle in the last M round. P150, traced, 3200 rows:
# QKV 711 -> 703, W2 570 -> 563, fused W1/W3/SwiGLU 702 -> 650 us (2560 rows: 566 -> 555, 464 -> 458, 537 -> 527 us;
# outputs pcc >= 0.99995 against the defaults); in the 32-user prefill QKV 675 -> 668, W13 698 -> 652, W2 411 -> 405 us.
# At 2048 and 4096 rows the defaults are faster (e.g. QKV 465 vs 513 and 875 vs 1028 us), so other sizes keep them. WO
# keeps its default: its residual-fused, per-user-heads matmul went 211 -> 245 us with the (8, 4, 4) blocking.
_PACKED_PREFILL_MM_BLOCKS = {
    "qkv": (8, 4, 8, 4, 2),
    "w2": (8, 8, 4, 4, 2),
    "w13": (10, 8, 8, 2, 4),
}


def _packed_prefill_mm_config(name: str, rows: int, default: ttnn.MinimalMatmulConfig | None):
    """``default``, or the packed-prefill blocking of ``name`` for 80..100 tile rows."""
    if default is None or not 80 <= rows // 32 <= 100:
        return default
    m, k, n, sh, sw = _PACKED_PREFILL_MM_BLOCKS[name]
    return ttnn.MinimalMatmulConfig(
        M_block_size=m,
        K_block_size=k,
        N_block_size=n,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=default.compute_with_storage_grid_size,
    )


def _with_128_row_config(default, config_128):
    def program_config(seq_len: int):
        return config_128 if seq_len == PREFILL_1D_SEQ_LEN else default(seq_len)

    return program_config


def _rotate_q_k(q_pre, k_pre, rot_mats, trans_mat, q_dtype, k_dtype, memory_config):
    """Prefill rotary embedding of Q and K heads, returning (Q, K) in ``q_dtype`` / ``k_dtype``.

    When Q, K and both target dtypes agree, one paired rotary op (``input_tensor_b``) rotates both tensors in place over
    their 2 x heads: P150, [1, 32, 128, 96] bfloat8_b Q and K, 12.1 us against 18.2 us for two calls. Otherwise one call
    per tensor, each writing a new tensor in its target dtype.
    """
    if q_pre.spec == k_pre.spec and q_dtype == k_dtype == q_pre.dtype:
        ttnn.experimental.rotary_embedding_llama(
            q_pre, rot_mats[0], rot_mats[1], trans_mat, is_decode_mode=False, memory_config=memory_config,
            input_tensor_b=k_pre,
        )
        return q_pre, k_pre
    q = ttnn.experimental.rotary_embedding_llama(
        q_pre, rot_mats[0], rot_mats[1], trans_mat, is_decode_mode=False, dtype=q_dtype, memory_config=memory_config
    )
    ttnn.deallocate(q_pre)
    k = ttnn.experimental.rotary_embedding_llama(
        k_pre, rot_mats[0], rot_mats[1], trans_mat, is_decode_mode=False, dtype=k_dtype, memory_config=memory_config
    )
    ttnn.deallocate(k_pre)
    return q, k


class Phi35MiniAttention(Attention1D):
    @classmethod
    def from_config(cls, config: Phi35MiniAttentionConfig) -> Phi35MiniAttention:
        if not isinstance(config, Phi35MiniAttentionConfig):
            raise TypeError("Phi35MiniAttention.from_config expects Phi35MiniAttentionConfig")
        if config.wqkv_decode is None or config.wo_decode is None:
            raise ValueError("Phi35MiniAttentionConfig needs wqkv_decode and wo_decode")
        instance = super().from_config(config)
        cfg = instance.config
        cfg.prefill_xqkv_prg_config = _with_128_row_config(cfg.prefill_xqkv_prg_config, cfg.prefill_128_xqkv_prg_config)
        cfg.prefill_wo_prg_config = _with_128_row_config(cfg.prefill_wo_prg_config, cfg.prefill_128_wo_prg_config)
        return instance

    def load_device_weights(self):
        if self._device_weights_loaded:
            return
        super().load_device_weights()
        self.wqkv_decode = self.config.wqkv_decode.get_device_weight()
        self.wo_decode = self.config.wo_decode.get_device_weight()
        self.residual_gate = _residual_gate(self.config.mesh_device, self.config.dim)
        # [users] int32 0..users-1 per batched-prefill size, for paged_fill_cache's batch_idx_tensor
        self._batch_idx_tensors: dict[int, ttnn.Tensor] = {}

    def supports_128_deferred_add(self) -> bool:
        """Whether a single-user 128-row prefill runs this module's L1 path with a streamed WO, whose output can go
        straight into the residual stream's block-sharded layout (Phi35MiniDecoderLayer.prefill_forward_128_deferred)."""
        cfg = self.config
        self.load_device_weights()
        return (
            cfg.mesh_device.get_num_devices() == 1
            and cfg.prefill_128_wo_stream_prg_config is not None
            and self.q_norm is None
            and self.k_norm is None
            and self.wqkv_bias_prefill is None
        )

    def prefill_forward_with_residual(
        self,
        x: ttnn.Tensor,
        residual: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        *,
        residual_memcfg: ttnn.MemoryConfig,
        user_id: int | list[int] = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        chunk_start_idx_tensor: ttnn.Tensor | None = None,
        batch_size: int | None = None,
        row_map: ttnn.Tensor | None = None,
        padded_sequence_length: int | None = None,
    ) -> ttnn.Tensor:
        """``residual + attention(x)``: the batched path adds the residual inside its WO matmul, others add after.

        ``row_map``: ``x`` and ``residual`` are packed rows (see ``Phi35MiniTransformer.supports_packed_batched_prefill``),
        which only the batched path takes."""
        if self.config.prefill_batched_wo_residual_minimal_matmul_config is not None and self._batched_prefill_supported(
            x, batch_size, page_table, chunk_start_idx, chunk_start_idx_tensor
        ):
            return self._prefill_forward_batched(
                x,
                rot_mats,
                user_id,
                page_table,
                chunk_page_table,
                batch_size,
                residual=residual,
                row_map=row_map,
                padded_rows=padded_sequence_length,
            )
        if row_map is not None:
            raise ValueError("packed prefill rows need the batched attention path")
        out = self.prefill_forward(
            x,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            chunk_start_idx_tensor=chunk_start_idx_tensor,
            batch_size=batch_size,
        )
        h = ttnn.add(residual, out, memory_config=residual_memcfg)
        ttnn.deallocate(out)
        return h

    def prefill_forward(
        self,
        x: ttnn.Tensor | LazyWeight,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int | list[int] = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        chunk_start_idx_tensor: ttnn.Tensor | None = None,
        batch_size: int | None = None,
        wo_memory_config: ttnn.MemoryConfig | None = None,
    ) -> ttnn.Tensor:
        """Single-user 128-row prefill with its intermediates in L1; every other case is ``Attention1D``'s.

        ``wo_memory_config`` (the 128-row path only, with ``prefill_128_wo_stream_prg_config``) is the WO output's
        layout, e.g. the residual stream's block-sharded one, which the next norm adds in place.

        The same ops as ``Attention1D.prefill_forward`` for one unchunked 128-row user (QKV, head split, rotary,
        cache-dtype casts, KV fill, SDPA, head concat, WO), but the QKV output, heads and attention output (0.3-2.4 MB
        each at 128 rows) and the output stay in L1 instead of each making a DRAM round trip.
        """
        cfg = self.config
        lean = (
            isinstance(x, ttnn.Tensor)
            and x.shape[-2] == PREFILL_1D_SEQ_LEN
            and (batch_size or x.shape[0]) == 1
            and chunk_start_idx is None
            and chunk_start_idx_tensor is None
            and cfg.mesh_device.get_num_devices() == 1
        )
        if lean:
            self.load_device_weights()
            lean = self.q_norm is None and self.k_norm is None and self.wqkv_bias_prefill is None
        if not lean and self._batched_prefill_supported(x, batch_size, page_table, chunk_start_idx, chunk_start_idx_tensor):
            return self._prefill_forward_batched(x, rot_mats, user_id, page_table, chunk_page_table, batch_size)
        if not lean:
            return super().prefill_forward(
                x,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                batch_size=batch_size,
            )

        l1 = ttnn.L1_MEMORY_CONFIG
        seq_len = x.shape[-2]  # the input may be in L1 (the 128-row layer's norm output) or DRAM
        # The same QKV / Q / K / V dtypes as the batched path (prefill_batched_qkv_dtype / prefill_batched_q_dtype):
        # bfloat8_b heads into the rotary op, Q out in bfloat8_b, K out and V in the cache dtype without casts.
        stream = cfg.prefill_128_qkv_stream_prg_config
        xqkv = ttnn.linear(
            x,
            self.wqkv_decode if stream else self.wqkv,
            dtype=cfg.prefill_batched_qkv_dtype or cfg.activation_dtype or ttnn.bfloat16,
            memory_config=l1,
            compute_kernel_config=(cfg.prefill_128_qkv_compute_kernel_cfg if stream else None)
            or cfg.li_qkv_prefill_compute_kernel_cfg,
            program_config=stream or cfg.prefill_xqkv_prg_config(seq_len),
        )
        ttnn.deallocate(x)
        q_pre, k_pre, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=cfg.n_heads, num_kv_heads=cfg.n_kv_heads, transpose_k_heads=False, memory_config=l1
        )
        ttnn.deallocate(xqkv)

        keys, values = self.kv_cache
        q_sdpa, k_fill = _rotate_q_k(
            q_pre,
            k_pre,
            rot_mats,
            cfg.transformation_mat_prefill,
            cfg.prefill_batched_q_dtype or cfg.activation_dtype or ttnn.bfloat8_b,
            keys.dtype,
            l1,
        )
        if v_heads.dtype != values.dtype:
            v_fill = ttnn.typecast(v_heads, dtype=values.dtype, memory_config=l1)
            ttnn.deallocate(v_heads)
        else:
            v_fill = v_heads
        self._kv_fill_prefill(keys, values, k_fill, v_fill, user_id, page_table, chunk_page_table)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q_sdpa,
            k_fill,
            v_fill,
            is_causal=True,
            sliding_window_size=cfg.sliding_window,
            scale=cfg.scale,
            compute_kernel_config=cfg.sdpa_prefill_compute_kernel_cfg,
            program_config=cfg.prefill_128_sdpa_prg_config or cfg.prefill_sdpa_prg_config(seq_len, None),
            memory_config=l1,
        )
        for t in (q_sdpa, k_fill, v_fill):
            ttnn.deallocate(t)
        attn = ttnn.reshape(attn, [1, cfg.n_heads, -1, cfg.head_dim])
        attn_concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=l1)
        ttnn.deallocate(attn)
        wo_stream = cfg.prefill_128_wo_stream_prg_config
        output = ttnn.linear(
            attn_concat,
            self.wo_decode if wo_stream else self.wo,
            compute_kernel_config=(cfg.prefill_128_wo_compute_kernel_cfg if wo_stream else None)
            or cfg.li_o_prefill_compute_kernel_cfg,
            dtype=cfg.activation_dtype or ttnn.bfloat8_b,
            memory_config=(wo_memory_config if wo_stream else None) or l1,
            program_config=wo_stream or cfg.prefill_wo_prg_config(seq_len),
        )
        ttnn.deallocate(attn_concat)
        return self._reduce_after_wo_prefill(output)

    def _batched_prefill_supported(self, x, batch_size, page_table, chunk_start_idx, chunk_start_idx_tensor) -> bool:
        cfg = self.config
        if not isinstance(x, ttnn.Tensor) or (batch_size or x.shape[0]) <= 1:
            return False
        if chunk_start_idx is not None or chunk_start_idx_tensor is not None or page_table is None:
            return False
        if cfg.mesh_device.get_num_devices() != 1 or cfg.paged_attention_config is None:
            return False
        self.load_device_weights()
        return self.q_norm is None and self.k_norm is None and self.wqkv_bias_prefill is None

    def _prefill_forward_batched(
        self, x, rot_mats, user_id, page_table, chunk_page_table, batch_size, residual=None, row_map=None, padded_rows=None
    ) -> ttnn.Tensor:
        """Batched (several users, unchunked, paged) prefill: ``Attention1D``'s ops with fewer copies and casts.

        ``x`` is the users' rows folded into one sequence, ``[1, 1, users x rows, dim]``. Differences from
        ``Attention1D.prefill_forward`` (P150, 32 users x 128 rows, per layer):
        - QKV and WO run as one minimal_matmul over all rows (Attention1D splits them into 2048 / 1024-row batches;
          same time).
        - With ``prefill_batched_qkv_dtype`` = bfloat8_b the QKV matmul writes bfloat8_b (1004 -> 884 us), the head
          split moves half the bytes, V is already in the cache dtype, and the rotary op reads bfloat8_b heads and writes
          Q in bfloat16 and K in the cache dtype. That drops Attention1D's K and V casts (88 us each).
        - Q goes to SDPA as it leaves the rotary op. Attention1D casts it to ``activation_dtype``, which for Phi is
          bfloat16 already, so that cast was a 121 us copy.
        - SDPA runs on the whole compute grid (``prefill_batched_sdpa_prg_config``) instead of 8 x 8 (429 -> 286 us),
          and without a sliding window when the window covers the whole prompt.
        - With ``residual`` the residual add runs inside the WO matmul (residual + 1 * (attn @ WO) * ones-row), which
          returns the new residual stream: 406 us against 559 us for the matmul and a separate add.
        WO stays bfloat16: a bfloat8_b WO output (-28 us) brought 32-user top-5 accuracy to 98.8%.
        """
        cfg = self.config
        users = batch_size or x.shape[0]
        if x.shape[0] != 1 or x.shape[1] != 1:
            x = ttnn.reshape(x, [1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]])
        seq_len = x.shape[2]
        # Packed rows (row_map): x holds the users' 32-row tiles back to back; Q / K / V, SDPA and the KV fill still use
        # the padded [users, heads, padded_rows, head_dim] layout, which the QKV matmul writes through the row map (with
        # each user's unused row tiles zeroed) and the WO matmul reads back.
        if row_map is not None and (residual is None or padded_rows is None or cfg.n_heads != cfg.n_kv_heads):
            raise ValueError("packed batched prefill needs the residual, the padded row count and equal Q / KV heads")
        rows = padded_rows if row_map is not None else seq_len // users
        keys, values = self.kv_cache
        qkv_dtype = cfg.prefill_batched_qkv_dtype or ttnn.bfloat16
        # QKV output, heads, rotated Q / K and the SDPA output: L1 interleaved or DRAM (prefill_batched_l1_intermediates)
        mid_memcfg = ttnn.L1_MEMORY_CONFIG if seq_len <= cfg.prefill_batched_l1_max_rows else ttnn.DRAM_MEMORY_CONFIG
        if cfg.n_heads == cfg.n_kv_heads:
            # Q, K and V are equal-width chunks of the QKV output, so the matmul writes each straight in head layout
            # [users, heads, rows, head_dim] (minimal_matmul_split head_split_rows): no nlp_create_qkv_heads. P150,
            # 32 users x 128 rows: 918 -> 832 us a layer, heads bit-identical.
            q_pre, k_pre, v_heads = ttnn.experimental.minimal_matmul_split(
                x,
                self.wqkv,
                chunks=3,
                compute_kernel_config=cfg.li_qkv_prefill_compute_kernel_cfg,
                config=_packed_prefill_mm_config("qkv", seq_len, cfg.prefill_xqkv_minimal_matmul_config(seq_len)),
                dtype=qkv_dtype,
                memory_config=mid_memcfg,
                head_split_rows=rows,
                head_dim=cfg.head_dim,
                head_split_users=users if row_map is not None else 0,
                row_map=row_map,
            )
            ttnn.deallocate(x)
        else:
            xqkv = ttnn.experimental.minimal_matmul(
                x,
                self.wqkv,
                compute_kernel_config=cfg.li_qkv_prefill_compute_kernel_cfg,
                config=cfg.prefill_xqkv_minimal_matmul_config(seq_len),
                dtype=qkv_dtype,
                memory_config=mid_memcfg,
            )
            ttnn.deallocate(x)
            xqkv = ttnn.reshape(xqkv, [users, 1, rows, xqkv.shape[-1]])
            q_pre, k_pre, v_heads = ttnn.experimental.nlp_create_qkv_heads(
                xqkv,
                num_heads=cfg.n_heads,
                num_kv_heads=cfg.n_kv_heads,
                transpose_k_heads=False,
                memory_config=mid_memcfg,
            )
            ttnn.deallocate(xqkv)

        # The rotary op reads bfloat16 or bfloat8_b heads and writes K straight in the cache dtype, Q in bfloat16.
        q_dtype = cfg.prefill_batched_q_dtype or cfg.activation_dtype or ttnn.bfloat8_b
        q_heads, k_fill = _rotate_q_k(
            q_pre, k_pre, rot_mats, cfg.transformation_mat_prefill, q_dtype, keys.dtype, mid_memcfg
        )
        if v_heads.dtype != values.dtype:
            v_fill = ttnn.typecast(v_heads, dtype=values.dtype)
            ttnn.deallocate(v_heads)
        else:
            v_fill = v_heads
        self._kv_fill_prefill_batched(keys, values, k_fill, v_fill, user_id, page_table, chunk_page_table)

        window = cfg.sliding_window if cfg.sliding_window is not None and cfg.sliding_window < rows else None
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_fill,
            v_fill,
            is_causal=True,
            sliding_window_size=window,
            scale=cfg.scale,
            compute_kernel_config=cfg.prefill_batched_sdpa_compute_kernel_cfg or cfg.sdpa_prefill_compute_kernel_cfg,
            program_config=cfg.prefill_batched_sdpa_prg_config or cfg.prefill_sdpa_prg_config(rows, None),
            # With the residual the WO matmul reads this output itself (in0_heads), and its static buffers clash with
            # an L1 tensor of this size, so it goes to DRAM as the concatenated heads did.
            memory_config=ttnn.DRAM_MEMORY_CONFIG if residual is not None else mid_memcfg,
        )
        for t in (q_heads, k_fill, v_fill):
            ttnn.deallocate(t)
        if residual is not None:
            # The fused WO reads the SDPA output's heads directly as the concatenated [users x rows, heads x head_dim]
            # activation (in0_heads), so there is no nlp_concat_heads: P150, 32 users x 128 rows, 284 -> 246 us a
            # layer, output bit-identical.
            output = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                attn,
                self.wo,
                1.0,
                residual,
                self.residual_gate,
                config=cfg.prefill_batched_wo_residual_minimal_matmul_config,
                compute_kernel_config=cfg.prefill_batched_wo_compute_kernel_cfg or cfg.li_o_prefill_compute_kernel_cfg,
                dtype=residual.dtype,
                in0_heads=True,
                row_map=row_map,
            )
            ttnn.deallocate(attn)
            return output
        attn_concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        attn_concat = ttnn.reshape(attn_concat, [1, 1, seq_len, attn_concat.shape[-1]])
        output = ttnn.experimental.minimal_matmul(
            attn_concat,
            self.wo,
            compute_kernel_config=cfg.prefill_batched_wo_compute_kernel_cfg or cfg.li_o_prefill_compute_kernel_cfg,
            config=cfg.prefill_wo_minimal_matmul_config(seq_len),
        )
        ttnn.deallocate(attn_concat)
        return self._reduce_after_wo_prefill(output)

    def _kv_fill_prefill_batched(self, keys, values, k_fill, v_fill, user_id, page_table, chunk_page_table) -> None:
        """Batched prefill KV fill as one paged_fill_cache call per cache instead of one per user.

        ``k_fill`` / ``v_fill`` are ``[users, kv heads, per-user rows, head dim]``. ``Attention1D`` slices and fills
        user by user (32 users: 64 slices + 64 fills a layer, ~0.75 ms with their dispatch gaps). paged_fill_cache
        also takes the whole batch with a ``batch_idx_tensor`` (one page-table row per input user, here users 0..n-1
        in order), so each cache is one call on the input as it is. A padded user's page-table row is all -1, which
        the fill skips per block, as in the per-user calls. P150, 32 users x 128 rows, per cache: 105 us against
        182 us for users laid end to end with a transpose and 1789 us for the loop (eager); caches bit-identical.
        Any other case (a subset of slots) is Attention1D's.
        """
        cfg = self.config
        users = k_fill.shape[0]
        fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
        slots = list(user_id) if isinstance(user_id, (list, tuple)) else list(range(users))
        if cfg.paged_attention_config is None or slots != list(range(users)) or fill_page_table.shape[0] != users:
            return super()._kv_fill_prefill_batched(keys, values, k_fill, v_fill, user_id, page_table, chunk_page_table)
        batch_idx = self._batch_idx_tensors.get(users)
        if batch_idx is None:
            import torch

            batch_idx = ttnn.from_torch(
                torch.arange(users, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=cfg.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(cfg.mesh_device),
            )
            self._batch_idx_tensors[users] = batch_idx
        # K and V in one program (input_tensor_b / cache_tensor_b): one dispatch instead of two, caches
        # bit-identical to two fills.
        ttnn.experimental.paged_fill_cache(
            keys, k_fill, fill_page_table, batch_idx_tensor=batch_idx, input_tensor_b=v_fill, cache_tensor_b=values
        )

    def _kv_fill_prefill_paged(self, keys, values, k_fill, v_fill, user_id, page_table, chunk_page_table) -> None:
        """Single-user paged prefill KV fill: K and V in one paged_fill_cache program.

        ``Attention1D`` fills K and V with two calls. paged_fill_cache takes a second (input, cache) pair that it
        fills with the same page table and batch index, so one program writes both: P150, 32 heads x 128 rows,
        60.6 -> 32.2 us eager for the pair, caches bit-identical. Anything else (several users, a fill longer
        than the page table, mismatched dtypes) is Attention1D's.
        """
        fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
        page_len = fill_page_table.shape[1] * keys.shape[2]
        if (
            k_fill.shape[0] != 1
            or page_len < k_fill.shape[2]
            or k_fill.dtype != keys.dtype
            or v_fill.dtype != values.dtype
            or k_fill.shape != v_fill.shape
        ):
            return super()._kv_fill_prefill_paged(keys, values, k_fill, v_fill, user_id, page_table, chunk_page_table)
        ttnn.experimental.paged_fill_cache(
            keys, k_fill, fill_page_table, batch_idx=user_id, input_tensor_b=v_fill, cache_tensor_b=values
        )

    def _fused_qk_heads_memcfg(self) -> ttnn.MemoryConfig:
        """Head-split output on the fused rope's cores: Q on the first ``batch`` cores, K on the next ``batch``.

        The fused rope reads cos/sin and the transformation matrix from the cores that hold Q and K. Both are
        sharded over ``2 * batch`` cores taken row-wise from ``decode_transformation_core_grid`` (the model sets the
        rope's ``core_grid`` to the same grid). The head split places Q on the first ``batch`` cores of its output
        grid and K on the following ``batch``, so giving it that same grid lines all three up at every batch size.
        """
        cfg = self.config
        if not cfg.use_qk_fused:
            raise ValueError("Phi35MiniAttention decode expects use_qk_fused=True")
        grid = ttnn.num_cores_to_corerangeset(2 * cfg.max_batch_size, cfg.decode_transformation_core_grid, row_wise=True)
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, cfg.head_dim),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def decode_forward(
        self,
        x: ttnn.Tensor | LazyWeight,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="decode")
        cfg = self.config
        if cfg.mesh_device.get_num_devices() != 1:
            raise ValueError("Phi35MiniAttention decode is single-device only")
        if self.q_norm is not None or self.k_norm is not None or self.wqkv_bias_decode:
            raise ValueError("Phi-3.5-mini has no q/k norm and no QKV bias")

        # QKV matmul: bfloat16, L1 width-sharded with one head per core. With fused QK the head split places Q
        # and K on separate cores and requires every shard to divide one head.
        xqkv = ttnn.linear(
            x,
            self.wqkv_decode,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=cfg.decode_xqkv_prg_config,
            compute_kernel_config=cfg.li_qkv_decode_compute_kernel_cfg,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(x)

        # Track the real batch in the logical shape; the padded tile height stays 32.
        xqkv = ttnn.reshape(xqkv, (1, 1, cfg.max_batch_size, xqkv.shape[3]), (1, 1, 32, xqkv.shape[3]))
        q_heads_pre_rot, k_heads_pre_rot, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=cfg.n_heads,
            num_kv_heads=cfg.n_kv_heads,
            overlap_qk_coregrid=False,
            memory_config=self._fused_qk_heads_memcfg(),
        )
        ttnn.deallocate(xqkv)

        q_heads, k_heads = self._rotary_embed_decode(q_heads_pre_rot, k_heads_pre_rot, rot_mats)
        ttnn.deallocate(q_heads_pre_rot)
        ttnn.deallocate(k_heads_pre_rot)

        keys, values = self.kv_cache
        self._kv_update_decode(keys, values, k_heads, v_heads, current_pos, page_table)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        if cfg.decode_sdpa_q_in_dram:
            q_sharded = q_heads
            q_heads = ttnn.to_memory_config(q_sharded, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(q_sharded)
        attn_output = self._sdpa_decode(q_heads, keys, values, current_pos, page_table)
        ttnn.deallocate(q_heads)

        attn_output_sharded = ttnn.to_memory_config(attn_output, memory_config=cfg.decode_scores_memcfg(cfg.max_batch_size))
        attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(attn_output_sharded, num_heads=cfg.n_heads)
        ttnn.deallocate(attn_output_sharded)
        ttnn.deallocate(attn_output)

        dense_out = ttnn.linear(
            attn_output_cat,
            self.wo_decode,
            program_config=cfg.decode_attn_output_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=cfg.li_o_decode_compute_kernel_cfg,
        )
        ttnn.deallocate(attn_output_cat)
        return ttnn.to_memory_config(dense_out, cfg.decode_residual_memcfg)
