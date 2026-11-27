# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NoPE MLA attention of the GLM-5.3-Flash MLA/DSA layers (modeling_glm5_next.py:1100-1275).

    q_r    = RMSNorm_1536(x W_qa);  q = (q_r W_qb).view(64, 256)
    c      = RMSNorm_512(x W_kva)                      -> the only KV cache entry (latent, 512/token, paged)
    q_lat  = W_uk_h^T q_h  (512 per head)               (absorbed form; exact because qk_rope_head_dim = 0)
    o_lat  = softmax(q_lat . c / sqrt(256)) c           (dense causal latent attention over the visible latents)
    out    = concat_h(W_uv_h o_lat_h) W_o               [16384 -> 4096]

DSA: the indexer selects the top-512 pools (2048 tokens) + <=3 tail tokens per query. For kv_len <= 2051 every visible
token is selected, so dense causal attention is exactly the HF computation; this functional decoder implements that
regime and rejects longer contexts explicitly (`DSA_DENSE_EXACT_MAX_KV`) instead of silently computing dense attention
where HF would attend sparsely.

Cache: paged latent cache [num_blocks, 1, block_size, 512] (bf16). Prefill writes a chunk with `paged_fill_cache` through a
chunk-local page table and attends with `chunked_scaled_dot_product_attention` (K = V = latent, runtime chunk-start
tensor); decode writes the token with `paged_update_cache` at `cur_pos` and attends with
`paged_flash_multi_latent_attention_decode`.
"""

from __future__ import annotations

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.precision import LATENT_CACHE_DTYPE, hifi2_fp32, hifi4_fp32

DRAM = ttnn.DRAM_MEMORY_CONFIG


class MLA:
    def __init__(
        self,
        mesh_device,
        *,
        w_qa_kva,
        qa_norm,
        kva_norm,
        w_qb,
        w_uk,
        w_uv,
        w_o,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_head_dim: int,
        v_head_dim: int,
        eps: float,
        block_size: int,
        q_chunk_size: int = 64,
        k_chunk_size: int = 64,
        decode_k_chunk_size: int = 32,
    ):
        """w_qa_kva = [q_a | kv_a] [4096, 1536 + 512] bfloat8_b; w_uk [64, 256, 512], w_uv [64, 512, 256] bf16."""
        self.device = mesh_device
        self.h = num_heads
        self.q_lora = q_lora_rank
        self.kv_lora = kv_lora_rank
        self.dqk = qk_head_dim
        self.dv = v_head_dim
        self.eps = eps
        self.scale = qk_head_dim**-0.5
        self.block_size = block_size
        self.w_qa_kva, self.w_qb, self.w_o = w_qa_kva, w_qb, w_o
        self.qa_norm = qa_norm
        self.kva_norm = kva_norm
        self.w_uk = ttnn.reshape(w_uk, (1, num_heads, qk_head_dim, kv_lora_rank))
        self.w_uv = ttnn.reshape(w_uv, (1, num_heads, kv_lora_rank, v_head_dim))
        self.ckc_proj = hifi2_fp32(mesh_device)
        self.ckc_fp32 = hifi4_fp32(mesh_device)
        self.ckc_sdpa = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        grid = mesh_device.compute_with_storage_grid_size()
        self.q_chunk = q_chunk_size
        self.decode_heads = 32
        self.prefill_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )
        self.decode_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=0,
            k_chunk_size=decode_k_chunk_size,
            exp_approx_mode=False,
        )

    def allocate_cache(self, num_blocks: int) -> ttnn.Tensor:
        return ttnn.zeros(
            (num_blocks, 1, self.block_size, self.kv_lora),
            dtype=LATENT_CACHE_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=DRAM,
        )

    def _project(self, x):
        """x [1,1,T,4096] -> (q_lat [1,64,T,512] bf16, latent [1,1,T,512] bf16)."""
        t = x.shape[-2]
        qk = ttnn.linear(x, self.w_qa_kva, compute_kernel_config=self.ckc_proj, memory_config=DRAM, dtype=ttnn.bfloat16)
        q_r = ttnn.slice(qk, (0, 0, 0, 0), (1, 1, t, self.q_lora))
        c = ttnn.slice(qk, (0, 0, 0, self.q_lora), (1, 1, t, self.q_lora + self.kv_lora))
        ttnn.deallocate(qk)
        q_r = ttnn.rms_norm(q_r, weight=self.qa_norm, epsilon=self.eps, compute_kernel_config=self.ckc_fp32)
        latent = ttnn.rms_norm(c, weight=self.kva_norm, epsilon=self.eps, compute_kernel_config=self.ckc_fp32)
        q = ttnn.linear(q_r, self.w_qb, compute_kernel_config=self.ckc_proj, memory_config=DRAM, dtype=ttnn.bfloat16)
        q = ttnn.permute(ttnn.reshape(q, (1, t, self.h, self.dqk)), (0, 2, 1, 3))  # [1,64,T,256]
        q_lat = ttnn.matmul(q, self.w_uk, compute_kernel_config=self.ckc_fp32, dtype=ttnn.bfloat16)  # [1,64,T,512]
        return q_lat, latent

    def _output(self, o_lat):
        """o_lat [1,64,T,512] -> [1,1,T,4096]."""
        t = o_lat.shape[-2]
        o = ttnn.matmul(o_lat, self.w_uv, compute_kernel_config=self.ckc_fp32, dtype=ttnn.bfloat16)  # [1,64,T,256]
        o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), (1, 1, t, self.h * self.dv))
        return ttnn.linear(o, self.w_o, compute_kernel_config=self.ckc_proj, memory_config=DRAM, dtype=ttnn.bfloat16)

    def prefill(
        self,
        x: ttnn.Tensor,
        cache: ttnn.Tensor,
        chunk_table: ttnn.Tensor,
        user_table: ttnn.Tensor,
        chunk_start: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """One user's chunk. x [1,1,C,4096] bf16 (C a multiple of the page size).
        chunk_table [1, >= C / block_size] int32: the user's pages for this chunk's positions (paged_fill_cache, batch 0);
        user_table [1, blocks_per_seq] int32: all of the user's pages; chunk_start [1] int32: absolute position of row 0
        (multiple of the page size). All three are persistent device tensors refreshed at the chunk boundary, so the
        programs depend only on C. Rows past the logical end are padding: their latents land in the user's own
        not-yet-valid slots, are invisible to every valid query (causal) and are overwritten by later writes."""
        q_lat, latent = self._project(x)
        ttnn.experimental.paged_fill_cache(cache, ttnn.typecast(latent, LATENT_CACHE_DTYPE), chunk_table, batch_idx=0)
        # NoPE MLA == multi-query attention over the latent cache with K = V = latent (head_dim 512); the generic
        # chunked SDPA reads the chunk start from a device tensor at runtime.
        o_lat = ttnn.transformer.chunked_scaled_dot_product_attention(
            q_lat,
            cache,
            cache,
            user_table,
            chunk_start_idx_tensor=chunk_start,
            scale=self.scale,
            program_config=self.prefill_program,
            compute_kernel_config=self.ckc_sdpa,
            memory_config=DRAM,
        )
        ttnn.deallocate(q_lat)
        return self._output(o_lat)

    def decode(self, x: ttnn.Tensor, cache: ttnn.Tensor, page_table: ttnn.Tensor, cur_pos: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,B,4096] bf16; cur_pos [B] int32 device tensor = position of this token per user (its latent is
        written there, then attention covers positions 0..cur_pos)."""
        bsz = x.shape[-2]
        q_lat, latent = self._project(x)
        q_lat = ttnn.permute(q_lat, (0, 2, 1, 3))  # [1,B,64,512]
        # paged_update_cache wants [1, B, 1(32), D] height-sharded one user per core
        grid = self.device.compute_with_storage_grid_size()
        shard = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.kv_lora),
            core_grid=ttnn.num_cores_to_corerangeset(bsz, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        lat = ttnn.to_memory_config(ttnn.reshape(latent, (1, bsz, 1, self.kv_lora)), shard)
        ttnn.experimental.paged_update_cache(cache, lat, update_idxs_tensor=cur_pos, page_table=page_table)
        # 64 heads x 512 latent overflow one core's L1 in a single call: attend in groups of 32 heads (same cache,
        # same positions; the groups are independent)
        groups = []
        for h0 in range(0, self.h, self.decode_heads):
            q_g = ttnn.slice(q_lat, (0, 0, h0, 0), (1, bsz, h0 + self.decode_heads, self.kv_lora))
            o_g = ttnn.transformer.paged_flash_multi_latent_attention_decode(
                q_g,
                cache,
                head_dim_v=self.kv_lora,
                page_table_tensor=page_table,
                cur_pos_tensor=cur_pos,
                scale=self.scale,
                program_config=self.decode_program,
                compute_kernel_config=self.ckc_sdpa,
                memory_config=DRAM,
            )  # [1,B,32,512]
            groups.append(ttnn.slice(o_g, (0, 0, 0, 0), (1, bsz, self.decode_heads, self.kv_lora)))
        o_lat = groups[0] if len(groups) == 1 else ttnn.concat(groups, dim=2)
        o_lat = ttnn.permute(o_lat, (0, 2, 1, 3))  # [1,64,B,512]
        return self._output(o_lat)
