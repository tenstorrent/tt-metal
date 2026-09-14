# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The attention block: QKV -> head split -> RoPE -> KV-cache write -> SDPA -> o_proj -> all-reduce.

Order matters and is fixed by the golden trace's convention: K is cached **post-RoPE** and V **raw**,
so the cache write sits between RoPE and attention. It is a single write point for every chunk —
chunk 0 writes then reads its own K/V through the live ring path, chunk N writes then reads the
accumulated prefix through the cache path — which is why there is no "first chunk" special case
anywhere above this function.
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule

from ..compute import (
    matmul_compute_config,
    plain_sdpa_program_config,
    ring_sdpa_compute_config,
    ring_sdpa_program_config,
    sdpa_compute_config,
)
from .config import AttentionConfig
from .dense_sp import ring_sdpa_cache_read, ring_sdpa_live
from .kv_cache import write_kv_chunk
from .operations import (
    apply_allreduce,
    apply_output_projection,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    split_qkv_heads,
)
from .weights import AttentionWeights, load_attention_weights


class Attention(LightweightModule):
    """One layer's attention. ``layer_idx`` is the cache slot's layer index, not a weight index."""

    def __init__(
        self,
        mesh_device,
        config: AttentionConfig,
        mesh_config,
        ccl_manager,
        rope_setup,
        state_dict=None,
        layer_idx: int = 0,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = config
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.rope_setup = rope_setup
        self.layer_idx = layer_idx
        self.n_q_local = config.num_heads // mesh_config.tp
        self.n_kv_local = config.num_kv_heads // mesh_config.tp
        self.matmul_config = matmul_compute_config(mesh_device)
        self.ring_program_config = ring_sdpa_program_config(mesh_device)
        self.ring_compute_config = ring_sdpa_compute_config()
        self.weights: AttentionWeights = load_attention_weights(
            mesh_device,
            config,
            mesh_config,
            state_dict=state_dict,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )

    def forward(
        self,
        hidden_states,
        rope_mats,
        *,
        kv_cache=None,
        user_id: int = 0,
        cached_len: int = 0,
        indexed_rope: bool = False,
        num_layers: int = None,
    ):
        """``[1, 1, s_local, hidden]`` (TP-replicated) -> same shape, TP-reduced.

        ``cached_len`` is the valid prefix already in the cache before this chunk: 0 selects the
        live ring path, > 0 the cache-read path. ``num_layers`` defaults to the cache's own count and
        exists so a test can drive the seam without a full model.
        """
        cfg = self.config
        s_local = hidden_states.shape[-2]
        if s_local <= 1:
            raise ValueError(f"prefill needs seq_len > 1 per device, got {s_local}")

        xqkv = apply_qkv_projection(hidden_states, self.weights, self.matmul_config)
        tt_q, tt_k, tt_v = split_qkv_heads(xqkv, self.n_q_local, self.n_kv_local)
        xqkv.deallocate(True)

        rope_kwargs = (
            {"kv_actual_global": cached_len, "cluster_axis": self.mesh_config.sp_axis} if indexed_rope else {}
        )
        q_pre, k_pre = tt_q, tt_k
        tt_q = apply_rope(tt_q, rope_mats, self.rope_setup.transformation_mat, **rope_kwargs)
        tt_k = apply_rope(tt_k, rope_mats, self.rope_setup.transformation_mat, **rope_kwargs)
        q_pre.deallocate(True)
        k_pre.deallocate(True)

        if kv_cache is not None:
            write_kv_chunk(
                kv_cache,
                tt_k,
                tt_v,
                slot_idx=user_id,
                layer_idx=self.layer_idx,
                kv_actual=cached_len,
                sp_axis=self.mesh_config.sp_axis,
            )

        if cfg.sequence_parallel:
            sp = self.mesh_config.sp
            if cached_len > 0:
                assert kv_cache is not None, "the cache-read path needs the cache it is reading"
                sdpa_out = ring_sdpa_cache_read(
                    tt_q,
                    kv_cache.k,
                    kv_cache.v,
                    mesh_config=self.mesh_config,
                    ccl_manager=self.ccl_manager,
                    kv_actual=cached_len,
                    logical_n=cached_len + s_local * sp,
                    cache_global=kv_cache.max_seq_len,
                    n_kv_global=cfg.num_kv_heads,
                    head_dim=cfg.head_dim,
                    scale=cfg.scale,
                    program_config=self.ring_program_config,
                    compute_kernel_config=self.ring_compute_config,
                    slot_idx=user_id,
                    layer_idx=self.layer_idx,
                    num_layers=kv_cache.num_layers if num_layers is None else num_layers,
                )
            else:
                sdpa_out = ring_sdpa_live(
                    tt_q,
                    tt_k,
                    tt_v,
                    mesh_config=self.mesh_config,
                    ccl_manager=self.ccl_manager,
                    logical_n=s_local * sp,
                    n_kv_global=cfg.num_kv_heads,
                    head_dim=cfg.head_dim,
                    scale=cfg.scale,
                    program_config=self.ring_program_config,
                    compute_kernel_config=self.ring_compute_config,
                )
        else:
            # sp=1 unit-test path: plain causal GQA flash attention, no ring, no cache read.
            sdpa_out = ttnn.transformer.scaled_dot_product_attention(
                tt_q,
                tt_k,
                tt_v,
                is_causal=True,
                scale=cfg.scale,
                program_config=plain_sdpa_program_config(self.mesh_device, s_local),
                compute_kernel_config=sdpa_compute_config(),
            )

        tt_q.deallocate(True)
        tt_k.deallocate(True)
        tt_v.deallocate(True)

        concat = concat_heads(sdpa_out)
        sdpa_out.deallocate(True)
        proj = apply_output_projection(concat, self.weights, self.matmul_config)
        concat.deallocate(True)
        return apply_allreduce(proj, self.mesh_config, self.ccl_manager)
