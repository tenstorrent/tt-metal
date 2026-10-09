# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 context-parallel prefill attention with local and packed global ring caches."""

import ttnn

from models.demos.gemma4_d_p.tt.ccl import ccl_reduce_scatter_rows

from .weights import load_attention_weights
from .ring_prefill import init_global_ring_kv_cache, init_sliding_ring_kv_cache

from .global_kv_cache import GLOBAL_HEAD_DIM, GLOBAL_ROTARY_DIM, pack_global_kv_device
from .operations import (
    apply_per_head_norm,
    apply_qkv_projection,
    prefill_short_lived_memcfg,
    project,
    split_qkv_heads_prefill,
)
from .ring_prefill import (
    global_query_dtype,
    global_ring_prefill_attention,
    lane_buffer_key,
    sliding_ring_prefill_attention,
    write_chunk_to_global_ring_cache,
    write_chunk_to_sliding_ring_cache,
)

# One buffer pair for each of the five sliding layers between global layers.
NUM_SWA_HALO_BUFFER_PAIRS = 5


class Gemma4AttentionConfig:
    """Configuration for a single attention layer, derived from HF config + layer type."""

    def __init__(self, hf_config, layer_idx):
        self.layer_type = hf_config.layer_types[layer_idx]
        self.hidden_size = hf_config.hidden_size
        self.num_attention_heads = hf_config.num_attention_heads
        self.rms_norm_eps = hf_config.rms_norm_eps

        self.is_sliding = self.layer_type == "sliding_attention"
        self.is_kv_tied = hf_config.attention_k_eq_v and not self.is_sliding

        if self.is_sliding:
            self.num_key_value_heads = hf_config.num_key_value_heads
            self.head_dim = hf_config.head_dim
            self.sliding_window_size = hf_config.sliding_window
            self.rope_theta = hf_config.rope_theta
            self.partial_rotary_factor = 1.0
        else:
            self.num_key_value_heads = hf_config.num_global_key_value_heads
            self.head_dim = hf_config.global_head_dim
            self.sliding_window_size = None
            self.rope_theta = hf_config.global_rope_theta
            self.partial_rotary_factor = hf_config.partial_rotary_factor

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads


class Gemma4Attention:
    def __init__(
        self,
        mesh_config,
        config,
        state_dict,
        ccl_manager,
        layer_idx,
        tensor_cache_path=None,
        max_batch_size=1,
        max_seq_len=262144,
        weight_dtype=ttnn.bfloat16,
        ring_kv_cache=None,
        ring_layer_idx=0,
        ring_num_layers=1,
    ):
        mesh_device = mesh_config.device
        self.mesh_device = mesh_device
        self.config = config
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config
        self.layer_idx = layer_idx

        if ring_kv_cache is not None:
            cache = ring_kv_cache.k if config.is_sliding else ring_kv_cache.kv
            if cache.shape[-2] * mesh_config.cp_degree < max_seq_len:
                raise ValueError("External ring cache is too small for the configured prefill capacity")

        self.weights = load_attention_weights(
            mesh_config=mesh_config,
            config=config,
            state_dict=state_dict,
            tensor_cache_path=tensor_cache_path,
            weight_dtype=weight_dtype,
        )

        self.ring_kv_cache = ring_kv_cache
        self.ring_layer_idx = ring_layer_idx
        self.ring_num_layers = ring_num_layers
        self.ring_max_seq_len = cache.shape[-2] * mesh_config.cp_degree if ring_kv_cache is not None else None
        if self.ring_kv_cache is None:
            num_local_kv_heads = (
                1 if self.weights.kv_replicated else config.num_key_value_heads // mesh_config.tp_degree
            )
            if self.weights.is_global:
                self.ring_kv_cache = init_global_ring_kv_cache(
                    mesh_config=mesh_config,
                    num_local_kv_heads=num_local_kv_heads,
                    max_seq_len=max_seq_len,
                    num_layers=1,
                    num_users=max_batch_size,
                )
            else:
                self.ring_kv_cache = init_sliding_ring_kv_cache(
                    mesh_config=mesh_config,
                    num_local_kv_heads=num_local_kv_heads,
                    head_dim=config.head_dim,
                    max_seq_len=max_seq_len,
                    num_layers=1,
                    num_users=max_batch_size,
                )
            self.ring_max_seq_len = max_seq_len

    def __call__(
        self,
        hidden_states,
        rope_mats,
        prefill_metadata,
        chunk_start_idx=0,
        packed_global_rope=None,
        packed_sliding_rope=None,
    ):
        """Write a user's chunk and attend its cached prefix."""
        if self.ring_kv_cache is None:
            raise ValueError("Galaxy prefill requires a ring KV cache")
        tp = self.mesh_config.tp_degree
        chunk_offset = int(chunk_start_idx)
        kv_tied = self.config.is_kv_tied
        xqkv = apply_qkv_projection(hidden_states, self.weights, kv_tied=kv_tied)
        # The block consumes its gathered input; at short M it sits in L1, so free it before attention's buffers.
        hidden_states.deallocate(True)

        # Short-lived prefill activations in L1 when GEMMA4_PREFILL_L1_ACT=1 (Qwen36
        # #48861). o_proj / allreduce stay DRAM (CB clash with CCL).
        act_mc = prefill_short_lived_memcfg()
        tt_q, tt_k, tt_v = split_qkv_heads_prefill(
            xqkv,
            self.config,
            self.weights.is_global,
            tp=tp,
            kv_replicated=self.weights.kv_replicated,
            memory_config=act_mc,
            kv_tied=kv_tied,
        )
        # The heads are copies: free the projection (in L1 per apply_qkv_projection) before attention's CBs allocate.
        xqkv.deallocate(True)

        tt_q = apply_per_head_norm(
            tt_q, self.config.rms_norm_eps, weight=self.weights.q_norm_weight, memory_config=act_mc
        )

        is_global = not self.config.is_sliding
        is_sliding = self.config.is_sliding
        if self.weights.is_global:
            # The tied projection is one semantic KV value. Normalize it once without
            # gamma: this entire 512-wide result is V. K branches from this value;
            # packed-only serving transforms just its active rotary quarter below.
            tt_k.deallocate(True)
            tt_v = apply_per_head_norm(tt_v, self.config.rms_norm_eps, memory_config=act_mc)
            tt_k = None
        else:
            tt_k = apply_per_head_norm(
                tt_k, self.config.rms_norm_eps, weight=self.weights.k_norm_weight, memory_config=act_mc
            )
            tt_v = apply_per_head_norm(tt_v, self.config.rms_norm_eps, memory_config=act_mc)

        # Apply RoPE to Q and the rotary part of K.
        if is_global:
            if packed_global_rope is None:
                raise RuntimeError("packed global ring attention requires pre-gathered packed RoPE tensors")
            q_cos, q_sin, _, _, trans_mat = packed_global_rope
            q_full = tt_q
            q_rotary = ttnn.slice(
                q_full,
                (0, 0, 0, 0),
                tuple(q_full.shape)[:-1] + (GLOBAL_ROTARY_DIM,),
                memory_config=act_mc,
            )
            q_nonrotary = ttnn.slice(
                q_full,
                (0, 0, 0, GLOBAL_ROTARY_DIM),
                tuple(q_full.shape)[:-1] + (GLOBAL_HEAD_DIM,),
                memory_config=act_mc,
            )
            q_rotated = ttnn.experimental.rotary_embedding_llama(
                q_rotary, q_cos, q_sin, trans_mat, is_decode_mode=False, memory_config=act_mc
            )
            tt_q = ttnn.concat((q_rotated, q_nonrotary), dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for tensor in (q_full, q_rotary, q_nonrotary, q_rotated):
                tensor.deallocate(True)
        elif is_sliding:
            if packed_sliding_rope is None:
                raise RuntimeError("packed sliding ring attention requires pre-gathered adjacent RoPE tensors")
            sliding_cos, sliding_sin, trans_mat = packed_sliding_rope
            q_unrotated = tt_q
            tt_q = ttnn.experimental.rotary_embedding_llama(
                q_unrotated,
                sliding_cos,
                sliding_sin,
                trans_mat,
                is_decode_mode=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            q_unrotated.deallocate(True)
            k_unrotated = tt_k
            tt_k = ttnn.experimental.rotary_embedding_llama(
                k_unrotated, sliding_cos, sliding_sin, trans_mat, is_decode_mode=False, memory_config=act_mc
            )
            k_unrotated.deallocate(True)
        # A PrefillLanes keeps its per-lane rows, so pass it through as is.
        lanes = prefill_metadata if isinstance(prefill_metadata, (list, tuple)) else [prefill_metadata]
        if len(lanes) > 1:
            tt_sdpa = self._batched_cache_and_attention(tt_q, tt_k, tt_v, lanes, rope_mats, packed_global_rope)
        else:
            tt_sdpa = self._cache_and_attention(tt_q, tt_k, tt_v, lanes[0], chunk_offset, rope_mats, packed_global_rope)

        # Concat heads + apply out proj + all_reduce
        tt_out = ttnn.experimental.nlp_concat_heads(tt_sdpa, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        projected = project(tt_out, self.weights.o_proj, into_reduce_scatter=True)
        tt_out.deallocate(True)
        tt_out = ccl_reduce_scatter_rows(projected, self.mesh_config, self.ccl_manager)

        return tt_out

    def _pack_global_kv(self, tt_k, tt_v, rope_mats, packed_global_rope):
        return pack_global_kv_device(
            tt_v,
            self.weights.k_norm_rotary_weight,
            rope_mats[0],
            rope_mats[1],
            canonical_k=tt_k,
            packed_rope_mats=packed_global_rope,
            value_is_packed=True,
            memory_config=prefill_short_lived_memcfg(),
        )

    def _attend(self, tt_q, prefill_metadata, kv_actual_global, num_local_kv_heads, lane=0):
        """This layer's ring SDPA of tt_q over one request's cached prefix; lane picks the receive buffers."""
        common = dict(
            mesh_config=self.mesh_config,
            prefill_metadata=prefill_metadata,
            ccl_manager=self.ccl_manager,
            num_local_kv_heads=num_local_kv_heads,
            max_seq_len=self.ring_max_seq_len,
            logical_n=self.ring_max_seq_len,
            kv_actual_global=kv_actual_global,
            scale=1.0,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                tt_q.device().arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            layer_idx=self.ring_layer_idx,
            num_layers=self.ring_num_layers,
        )
        if not self.config.is_sliding:
            return global_ring_prefill_attention(tt_q, self.ring_kv_cache.kv, lane=lane, **common)
        return sliding_ring_prefill_attention(
            tt_q,
            self.ring_kv_cache.k,
            self.ring_kv_cache.v,
            head_dim=self.config.head_dim,
            gather_buffer_key=lane_buffer_key(self.layer_idx % NUM_SWA_HALO_BUFFER_PAIRS, lane),
            sliding_window_size=self.config.sliding_window_size,
            **common,
        )

    def _cache_and_attention(self, tt_q, tt_k, tt_v, prefill_metadata, chunk_offset, rope_mats, packed_global_rope):
        """One request: write its KV chunk, then attend its cached prefix. Frees tt_q, tt_k and tt_v."""
        num_local_kv_heads = tt_v.shape[1]
        if not self.config.is_sliding:
            query_dtype = global_query_dtype(tt_q.shape[-2])
            if query_dtype is not None:
                q_bf16, tt_q = tt_q, ttnn.typecast(tt_q, query_dtype)
                q_bf16.deallocate(True)
            packed_kv = self._pack_global_kv(tt_k, tt_v, rope_mats, packed_global_rope)
            write_chunk_to_global_ring_cache(
                self.ring_kv_cache.kv,
                packed_kv,
                self.mesh_config,
                kv_actual_global=chunk_offset,
                layer_idx=self.ring_layer_idx,
                num_layers=self.ring_num_layers,
                prefill_metadata=prefill_metadata,
            )
            tt_sdpa = self._attend(tt_q, prefill_metadata, chunk_offset, num_local_kv_heads)
            packed_kv.deallocate(True)
        else:
            write_chunk_to_sliding_ring_cache(
                self.ring_kv_cache.k,
                self.ring_kv_cache.v,
                tt_k,
                tt_v,
                self.mesh_config,
                kv_actual_global=chunk_offset,
                layer_idx=self.ring_layer_idx,
                num_layers=self.ring_num_layers,
                prefill_metadata=prefill_metadata,
            )
            tt_sdpa = self._attend(tt_q, prefill_metadata, chunk_offset, num_local_kv_heads)
        tt_q.deallocate(True)
        if tt_k is not None:
            tt_k.deallocate(True)
        tt_v.deallocate(True)
        return tt_sdpa

    def _batched_cache_and_attention(self, tt_q, tt_k, tt_v, lanes, rope_mats, packed_global_rope):
        """Several requests stacked along rows, request-major: write each request's KV chunk into its own slot, then
        run each request's ring SDPA with its own metadata and concatenate the outputs back into the stacked rows.
        Frees tt_q, tt_k and tt_v.

        Each request's CP-local rows come from lanes.rows when lanes is a PrefillLanes, else an equal split. The
        stacked K/V are cast to the cache dtype once; each request's cache write reads its own row window of them
        (input_rows), and they are freed before any SDPA runs (the SDPA reads K/V from the cache), so the
        per-request calls see the L1 an unbatched call sees. The device metadata carries each
        request's prefix, so the host kv_actual_global is 0.
        """
        rows = getattr(lanes, "rows", None) or (tt_q.shape[-2] // len(lanes),) * len(lanes)
        if sum(rows) != tt_q.shape[-2]:
            raise ValueError(f"lane rows {rows} do not add up to the {tt_q.shape[-2]} stacked rows")
        spans = [(sum(rows[:lane]), rows[lane]) for lane in range(len(lanes))]
        is_global = not self.config.is_sliding
        num_local_kv_heads = tt_v.shape[1]
        # Same-width requests share one Q cast; mixed widths cast per request below.
        stacked_q_cast = is_global and len(set(rows)) == 1
        if stacked_q_cast:
            query_dtype = global_query_dtype(rows[0])
            if query_dtype is not None:
                q_bf16, tt_q = tt_q, ttnn.typecast(tt_q, query_dtype)
                q_bf16.deallocate(True)
        if is_global:
            packed_kv = self._pack_global_kv(tt_k, tt_v, rope_mats, packed_global_rope)
            tt_v.deallocate(True)
            if tt_k is not None:
                tt_k.deallocate(True)
            stacked_kv = _to_cache_dtype(packed_kv, self.ring_kv_cache.kv.dtype)
            for span, metadata in zip(spans, lanes):
                write_chunk_to_global_ring_cache(
                    self.ring_kv_cache.kv,
                    stacked_kv,
                    self.mesh_config,
                    kv_actual_global=0,
                    layer_idx=self.ring_layer_idx,
                    num_layers=self.ring_num_layers,
                    prefill_metadata=metadata,
                    input_rows=span,
                )
            stacked_kv.deallocate(True)
        else:
            stacked_k = _to_cache_dtype(tt_k, self.ring_kv_cache.k.dtype)
            stacked_v = _to_cache_dtype(tt_v, self.ring_kv_cache.v.dtype)
            for span, metadata in zip(spans, lanes):
                write_chunk_to_sliding_ring_cache(
                    self.ring_kv_cache.k,
                    self.ring_kv_cache.v,
                    stacked_k,
                    stacked_v,
                    self.mesh_config,
                    kv_actual_global=0,
                    layer_idx=self.ring_layer_idx,
                    num_layers=self.ring_num_layers,
                    prefill_metadata=metadata,
                    input_rows=span,
                )
            stacked_k.deallocate(True)
            stacked_v.deallocate(True)

        outputs = []
        for lane, (span, metadata) in enumerate(zip(spans, lanes)):
            q_rows = _lane_rows(tt_q, *span)
            if is_global and not stacked_q_cast:
                query_dtype = global_query_dtype(span[1])
                if query_dtype is not None:
                    q_bf16, q_rows = q_rows, ttnn.typecast(q_rows, query_dtype)
                    q_bf16.deallocate(True)
            outputs.append(self._attend(q_rows, metadata, 0, num_local_kv_heads, lane=lane))
            q_rows.deallocate(True)
        tt_q.deallocate(True)
        tt_sdpa = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for out in outputs:
            out.deallocate(True)
        return tt_sdpa


def _lane_rows(tensor, start, rows):
    """Rows [start, start + rows) of a tensor stacked request-major along dim 2 (a DRAM copy)."""
    shape = tuple(tensor.shape)
    return ttnn.slice(
        tensor,
        (0, 0, start, 0),
        (shape[0], shape[1], start + rows, shape[3]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_cache_dtype(tensor, dtype):
    """tensor cast to the cache dtype, freeing the original; tensor itself when it already matches."""
    if tensor.dtype == dtype:
        return tensor
    cast = ttnn.typecast(tensor, dtype)
    tensor.deallocate(True)
    return cast
