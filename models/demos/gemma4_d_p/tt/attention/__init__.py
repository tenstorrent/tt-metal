# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 context-parallel prefill attention with local and packed global ring caches."""

import ttnn

from models.demos.gemma4_d_p.tt.ccl import ccl_allreduce

from .weights import load_attention_weights
from .ring_prefill import init_global_ring_kv_cache, init_sliding_ring_kv_cache

from .global_kv_cache import GLOBAL_HEAD_DIM, GLOBAL_ROTARY_DIM, pack_global_kv_device
from .operations import (
    apply_per_head_norm,
    apply_qkv_projection,
    projection_matmul_configs,
    prefill_short_lived_memcfg,
    split_qkv_heads_prefill,
)
from .ring_prefill import (
    global_ring_prefill_attention,
    sliding_ring_prefill_attention,
    write_chunk_to_global_ring_cache,
    write_chunk_to_sliding_ring_cache,
)


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
        sliding_window_size = self.config.sliding_window_size
        if is_global:
            packed_q = tt_q
            packed_kv = pack_global_kv_device(
                tt_v,
                self.weights.k_norm_rotary_weight,
                rope_mats[0],
                rope_mats[1],
                canonical_k=tt_k,
                packed_rope_mats=packed_global_rope,
                value_is_packed=True,
                memory_config=act_mc,
            )
            write_chunk_to_global_ring_cache(
                self.ring_kv_cache.kv,
                packed_kv,
                self.mesh_config,
                kv_actual_global=chunk_offset,
                layer_idx=self.ring_layer_idx,
                num_layers=self.ring_num_layers,
                prefill_metadata=prefill_metadata,
            )
        else:
            packed_q = None
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

        sdpa_compute_config = ttnn.init_device_compute_kernel_config(
            tt_q.device().arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        num_local_kv_heads_ring = tt_v.shape[1]
        ring_logical_n = self.ring_max_seq_len
        if is_global:
            tt_sdpa = global_ring_prefill_attention(
                packed_q,
                self.ring_kv_cache.kv,
                mesh_config=self.mesh_config,
                prefill_metadata=prefill_metadata,
                ccl_manager=self.ccl_manager,
                num_local_kv_heads=num_local_kv_heads_ring,
                max_seq_len=self.ring_max_seq_len,
                logical_n=ring_logical_n,
                kv_actual_global=chunk_offset,
                scale=1.0,
                compute_kernel_config=sdpa_compute_config,
                layer_idx=self.ring_layer_idx,
                num_layers=self.ring_num_layers,
            )
            packed_kv.deallocate(True)
        else:
            tt_sdpa = sliding_ring_prefill_attention(
                tt_q,
                self.ring_kv_cache.k,
                self.ring_kv_cache.v,
                mesh_config=self.mesh_config,
                prefill_metadata=prefill_metadata,
                ccl_manager=self.ccl_manager,
                num_local_kv_heads=num_local_kv_heads_ring,
                head_dim=self.config.head_dim,
                max_seq_len=self.ring_max_seq_len,
                logical_n=ring_logical_n,
                kv_actual_global=chunk_offset,
                sliding_window_size=sliding_window_size,
                scale=1.0,
                compute_kernel_config=sdpa_compute_config,
                layer_idx=self.ring_layer_idx,
                num_layers=self.ring_num_layers,
            )
        tt_q.deallocate(True)
        if tt_k is not None:
            tt_k.deallocate(True)
        tt_v.deallocate(True)

        # Concat heads + apply out proj + all_reduce
        tt_out = ttnn.experimental.nlp_concat_heads(tt_sdpa, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        program_config, compute_kernel_config = projection_matmul_configs(tt_out, self.weights.o_proj)
        projected = ttnn.linear(
            tt_out, self.weights.o_proj, program_config=program_config, compute_kernel_config=compute_kernel_config
        )
        tt_out.deallocate(True)
        tt_out = ccl_allreduce(projected, self.mesh_config, self.ccl_manager)

        return tt_out
