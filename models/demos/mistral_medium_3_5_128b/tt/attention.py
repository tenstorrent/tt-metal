# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GQA attention for Mistral-Medium-3.5 prefill on SP=8 x TP=4.

Structure from ``minimax_m3/tt/attention`` (dense ring-joint path): fused column-parallel QKV (per chip
24 q heads + 2 kv heads = 3584 columns), head split, YaRN RoPE, per-layer KV-cache write, ring-joint
SDPA over the SP axis, concat heads, row-parallel o_proj closed by a TP reduce-scatter into the sharded
residual. The llama-4 query scale is the identity for this checkpoint (beta 0, asserted).

SDPA: the first chunk (and one-shot) attends its own live bf16 K/V via the ring; a later chunk reads the
whole accumulated prefix out of the bf8 KV cache. Program config q_chunk 128 / k_chunk 512 is M3's (same
head_dim). M3 runs both ring paths with ``fp32_dest_acc_en=False``; only the cache-read path requires it
(its KV-pad rotation needs streaming compute, which fp32 accumulation disables), so the no-cache path runs
with fp32 accumulation, which measurably lifts the deep-layer KV PCC (README "Correctness knobs").
Everything else is HiFi4 + fp32 accumulation.
"""

import torch

import ttnn

from .common import cache_name, compute_config, dtype_tag
from .kv_cache import write_kv_chunk
from .precision import precision
from .rope import permute_qk_rows


def sdpa_program_config(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    # The ring's CCL workers own the last compute column; SDPA compute uses the rest.
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=precision().sdpa_q_chunk,
        k_chunk_size=precision().sdpa_k_chunk,
        exp_approx_mode=False,
    )


def sdpa_compute_config(mesh_device, cache_read: bool):
    fp32_acc = precision().sdpa_fp32_acc and not cache_read
    return compute_config(mesh_device, fp32_dest_acc_en=fp32_acc, packer_l1_acc=False)


def _ring_common(mesh_config, ccl_manager, cache_read: bool):
    return dict(
        joint_strategy="rear",
        dim=2,
        multi_device_global_semaphore=ccl_manager.ring_attention_ccl_semaphore_handles,
        num_links=ccl_manager.num_links,
        cluster_axis=mesh_config.sp_axis,
        mesh_device=ccl_manager.mesh_device,
        topology=ttnn.Topology.Linear,
        ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        is_balanced=False,
        program_config=sdpa_program_config(ccl_manager.mesh_device),
        compute_kernel_config=sdpa_compute_config(ccl_manager.mesh_device, cache_read),
    )


def ring_sdpa_nocache(tt_q, tt_k, tt_v, *, mesh_config, ccl_manager, logical_n, n_kv, head_dim, scale):
    """Causal GQA over this chunk's own SP-sharded Q/K/V (first chunk / one-shot).

    Per chip: q ``[1, nq/tp, S/sp, D]``, k/v ``[1, n_kv/tp, S/sp, D]``; ``logical_n`` = S (global)."""
    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        None,
        None,
        None,
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            "nocache_k", n_kv, logical_n, head_dim, tt_k.dtype
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            "nocache_v", n_kv, logical_n, head_dim, tt_v.dtype
        ),
        logical_n=logical_n,
        scale=scale,
        **_ring_common(mesh_config, ccl_manager, cache_read=False),
    )
    return out


def ring_sdpa_cache_read(
    tt_q, kv_cache, *, mesh_config, ccl_manager, user_id, layer_idx, kv_actual, chunk_global, scale
):
    """Causal GQA of this chunk's Q over the cached prefix [0, kv_actual + chunk_global) (chunk >= 1).

    The chunk's own K/V must already be in the cache. The ring gathers the WHOLE per-chip cache shard,
    so the gather buffers span the cache capacity, not the valid prefix (``logical_n`` masks the rest)."""
    n_kv = kv_cache.num_local_kv_heads * mesh_config.tp
    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        kv_cache.k,
        kv_cache.v,
        None,
        None,
        None,
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            "cache_k", n_kv, kv_cache.max_seq_len, kv_cache.head_dim, kv_cache.k.dtype
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            "cache_v", n_kv, kv_cache.max_seq_len, kv_cache.head_dim, kv_cache.v.dtype
        ),
        logical_n=kv_actual + chunk_global,
        scale=scale,
        kv_cache_batch_idx=user_id,
        kv_actual_isl=kv_actual,
        kv_cache_num_layers=kv_cache.num_layers,
        kv_cache_layer_idx=layer_idx,
        **_ring_common(mesh_config, ccl_manager, cache_read=True),
    )
    return out


class Attention:
    def __init__(
        self,
        mesh_device,
        mesh_config,
        ccl_manager,
        cfg,
        state_dict,
        *,
        layer_idx: int = 0,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        """``state_dict``: HF ``{q,k,v,o}_proj.weight`` ``[out, in]`` (HF half-split rotary layout); empty
        when loading from ``tensor_cache_path``."""
        assert cfg.llama_4_scaling_beta == 0, "only the identity llama-4 query scale is implemented"
        tp = mesh_config.tp
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.head_dim = cfg.head_dim
        self.n_local_heads = mesh_config.shard_size(cfg.num_attention_heads)
        self.n_local_kv_heads = mesh_config.shard_size(cfg.num_key_value_heads)
        self.scale = cfg.head_dim**-0.5
        self.compute_kernel_config = compute_config(mesh_device)

        wqkv = o_proj = None
        if state_dict:
            d = cfg.head_dim
            q = permute_qk_rows(state_dict["q_proj.weight"], d).chunk(tp, dim=0)
            k = permute_qk_rows(state_dict["k_proj.weight"], d).chunk(tp, dim=0)
            v = state_dict["v_proj.weight"].chunk(tp, dim=0)
            # Per chip [hidden, q_local | k_local | v_local]; chips concatenated then column-sharded.
            wqkv = torch.cat([torch.cat([q[c].t(), k[c].t(), v[c].t()], dim=-1) for c in range(tp)], dim=-1)
            wqkv = wqkv[None, None]
            o_proj = state_dict["o_proj.weight"].t()[None, None]  # [1, 1, heads*D, hidden], rows = heads
        tag = f"tp{tp}_{dtype_tag(weight_dtype)}"
        self.wqkv = ttnn.as_tensor(
            wqkv,
            device=mesh_device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.column_parallel(mesh_device),
            cache_file_name=cache_name(tensor_cache_path, f"wqkv_meta_{tag}"),
        )
        self.o_proj = ttnn.as_tensor(
            o_proj,
            device=mesh_device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.row_parallel(mesh_device),
            cache_file_name=cache_name(tensor_cache_path, f"o_proj_{tag}"),
        )

    def _linear(self, x, w, dtype=ttnn.bfloat16):
        return ttnn.linear(
            x,
            w,
            dtype=dtype,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x, rope, *, kv_cache=None, user_id: int = 0, cached_len: int = 0):
        """``x`` full-width ``[1, 1, s_local, hidden]`` (SP-sharded seq) -> ``[1, 1, s_local, hidden/tp]``.
        Writes this chunk's K/V into ``kv_cache`` (slot ``user_id``, layer ``layer_idx``, offset
        ``cached_len``) and, for ``cached_len > 0``, attends the cached prefix."""
        mc = self.mesh_config
        chunk_global = x.shape[-2] * mc.sp

        xqkv = self._linear(x, self.wqkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        q_rot, k_rot = rope(q, cached_len), rope(k, cached_len)
        q.deallocate(True)
        k.deallocate(True)

        if kv_cache is not None:
            write_kv_chunk(
                kv_cache, k_rot, v, user_id=user_id, layer_idx=self.layer_idx, kv_actual=cached_len, sp_axis=mc.sp_axis
            )
        if cached_len == 0:
            attn = ring_sdpa_nocache(
                q_rot,
                k_rot,
                v,
                mesh_config=mc,
                ccl_manager=self.ccl_manager,
                logical_n=chunk_global,
                n_kv=self.cfg.num_key_value_heads,
                head_dim=self.head_dim,
                scale=self.scale,
            )
        else:
            assert kv_cache is not None, "a chunk after the first needs the KV cache holding its prefix"
            attn = ring_sdpa_cache_read(
                q_rot,
                kv_cache,
                mesh_config=mc,
                ccl_manager=self.ccl_manager,
                user_id=user_id,
                layer_idx=self.layer_idx,
                kv_actual=cached_len,
                chunk_global=chunk_global,
                scale=self.scale,
            )
        for t in (q_rot, k_rot, v):
            t.deallocate(True)

        concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        out = self._linear(concat, self.o_proj, dtype=precision().proj_out_dtype)
        concat.deallocate(True)
        if mc.tp == 1:
            return out
        scattered = mc.reduce_scatter(out, self.ccl_manager, axis=mc.tp_axis, dim=3)
        out.deallocate(True)
        return scattered
