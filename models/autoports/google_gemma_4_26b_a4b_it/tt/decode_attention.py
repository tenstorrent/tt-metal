# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-device attention with accurate paged cache generation and decode."""

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.precise_attention import PrecisePagedAttention
from models.autoports.google_gemma_4_26b_a4b_it.tt.precision_ops import norm_weight, rms_norm, rotary
from models.demos.gemma4.tt.attention.operations import (
    chunked_prefill_sdpa,
    concat_heads,
    prefill_sdpa_program_config,
    split_qkv_heads_decode,
    split_qkv_heads_prefill,
)


class DecodeAttention:
    """TP1 prefill and B1 decode; FunctionalDecoder supplies the slot loop."""

    def __init__(self, source, max_context):
        self.source = source
        self.compute = ttnn.init_device_compute_kernel_config(
            source.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.q_weight = norm_weight(source.weights.q_norm_weight, source.config.head_dim)
        self.k_weight = norm_weight(source.weights.k_norm_weight, source.config.head_dim)
        self.decode_sdpa = PrecisePagedAttention(source.mesh_device, source.config, max_context)
        self.tail = None

    def __getattr__(self, name):
        return getattr(self.source, name)

    def _release_sliding_prefill_tail(self, clear_persistent=True):
        self.tail = None

    def __call__(self, hidden_states, **kwargs):
        if kwargs.get("is_decode", True):
            return self.decode(hidden_states, **kwargs)
        return self.prefill(hidden_states, **kwargs)

    def heads(self, hidden_states, decode):
        cfg = self.source.config
        qkv = self.source.weights.wqkv(hidden_states)
        split = split_qkv_heads_decode if decode else split_qkv_heads_prefill
        q, k, v = split(qkv, cfg, self.source.weights.is_global)
        memory = q.memory_config()
        q, k, v = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v)]
        q = rms_norm(q, cfg.rms_norm_eps, self.q_weight)
        k = rms_norm(k, cfg.rms_norm_eps, self.k_weight)
        v = rms_norm(v, cfg.rms_norm_eps)
        return q, k, v, memory

    def project(self, attention, decode):
        cfg = self.source.config
        combined = concat_heads(
            attention,
            is_decode_mode=decode,
            num_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim,
            mesh_device=self.source.mesh_device,
        )
        return ttnn.linear(combined, self.source.weights.o_proj, dtype=ttnn.float32, compute_kernel_config=self.compute)

    def decode(self, hidden_states, **kwargs):
        if hidden_states.shape[-2] != 1:
            raise ValueError("DecodeAttention expects one request slot")
        cfg = self.source.config
        q, k, v, cache_memory = self.heads(hidden_states, True)
        cos, sin = (
            ttnn.unsqueeze_to_4D(ttnn.embedding(kwargs["position_idx"], table, layout=ttnn.TILE_LAYOUT))
            for table in kwargs["rope_mats"]
        )
        q, k = rotary(q, cos, sin, decode=True), rotary(k, cos, sin, decode=True)
        cache_pos, page_table = kwargs["position_idx_cache"], kwargs["page_table"]
        k_cache, v_cache = kwargs["kv_cache"]
        for cache, update in ((k_cache, k), (v_cache, v)):
            # Native update's Float32 repacking loses precision. Round on the
            # SFPU before handing the already-BF16 rows to the cache writer.
            update = ttnn.to_memory_config(ttnn.typecast(update, cache.dtype), cache_memory)
            ttnn.experimental.paged_update_cache(
                cache,
                update,
                update_idxs_tensor=cache_pos,
                page_table=page_table,
                block_size=cache.shape[-2],
                num_kv_heads=cfg.num_key_value_heads,
            )
        attention = self.decode_sdpa(q, k_cache, v_cache, cur_pos_tensor=cache_pos, page_table_tensor=page_table)
        return self.project(attention, True)

    def prefill(self, hidden_states, **kwargs):
        cfg = self.source.config
        q, k, v, _ = self.heads(hidden_states, False)
        cos, sin = kwargs["rope_mats"]
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        q, k, v = [ttnn.typecast(t, ttnn.bfloat16) for t in (q, k, v)]
        page_table, user_id = kwargs["page_table"], kwargs["user_id"]
        fill_table = kwargs.get("chunk_page_table")
        fill_table = page_table if fill_table is None else fill_table
        valid = kwargs["valid_seq_len"]
        fill_length = (valid + 31) // 32 * 32
        for cache, update in zip(kwargs["kv_cache"], (k, v)):
            ttnn.experimental.paged_fill_cache(
                cache, update[:, :, :fill_length, :], fill_table, batch_idx=user_id, block_size=cache.shape[-2]
            )

        start = kwargs.get("chunk_start_idx")
        if cfg.is_sliding:
            history = 0
            if self.tail is not None:
                history = self.tail[0].shape[-2]
                q_attention = ttnn.concat((q[:, :, :history, :], q), dim=2)
                k_attention = ttnn.concat((self.tail[0], k), dim=2)
                v_attention = ttnn.concat((self.tail[1], v), dim=2)
            else:
                q_attention, k_attention, v_attention = q, k, v
            attention = ttnn.transformer.scaled_dot_product_attention(
                q_attention,
                k_attention,
                v_attention,
                is_causal=True,
                scale=1.0,
                sliding_window_size=cfg.sliding_window,
                program_config=prefill_sdpa_program_config(cfg.head_dim, q_attention.shape[-2], cfg.sliding_window),
                compute_kernel_config=self.compute,
            )
            if history:
                attention = attention[:, :, history : history + q.shape[-2], :]
            # All non-final physical chunks cover a full sliding window. A
            # fresh request clears this tail; prefix continuation uses decode.
            tail_length = min(cfg.sliding_window, k.shape[-2])
            self.tail = tuple(ttnn.clone(t[:, :, -tail_length:, :]) for t in (k, v))
        elif start is not None and start != 0:
            attention = chunked_prefill_sdpa(
                q,
                *kwargs["kv_cache"],
                page_table,
                user_id,
                cfg.head_dim,
                scale=1.0,
                base_offset=start,
                num_kv_heads=cfg.num_key_value_heads,
            )
        else:
            attention = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=1.0,
                program_config=prefill_sdpa_program_config(cfg.head_dim, q.shape[-2]),
                compute_kernel_config=self.compute,
            )
        return self.project(attention, False)
