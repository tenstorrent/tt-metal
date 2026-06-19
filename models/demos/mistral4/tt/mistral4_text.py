# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-local TTNN modules for the Mistral-Small-4 (mistral4) text core: MLA + MoE + decoder layer.

Built from generic ttnn ops (no new kernels, no deepseek framework). The recipes here are the
ones PCC-verified bottom-up in tests/multimodal/mistral_24b/test_m4_mla.py and test_m4_moe.py:
  - MLA: q/kv low-rank projections + RMSNorms, interleaved RoPE applied as a permutation-matrix
    matmul (de-interleave) then standard rotate_half, MQA k_rot expand, SDPA, o_proj.
  - MoE: router (linear -> softmax -> top-4/128 kth-threshold mask -> normalize), dense weighted
    SwiGLU experts, plus a shared SwiGLU expert.
Correctness-first: weights replicated across the mesh; experts computed densely. Sharding /
sparse dispatch / flash-MLA / paged / trace are follow-up optimizations.
"""
import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def _lin(weight, mesh):
    # HF nn.Linear weight [out, in] -> ttnn [in, out] for x @ W; replicated across the mesh.
    return ttnn.as_tensor(
        weight.transpose(0, 1).contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _norm(weight, mesh):
    return ttnn.as_tensor(
        weight.reshape(1, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _to_decode_shard(t, mesh):
    """[1,B,H,dh] -> height-sharded on B cores ([H,dh] per core), the layout paged_update_cache
    requires for the decode KV write (one batch element per core)."""
    _, B, H, dh = t.shape
    cores = ttnn.num_cores_to_corerangeset(B, mesh.compute_with_storage_grid_size(), row_wise=True)
    spec = ttnn.ShardSpec(cores, [H, dh], ttnn.ShardOrientation.ROW_MAJOR)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    return ttnn.to_memory_config(t, mem)


def _rotate_half(x):
    d = x.shape[-1]
    x1 = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], d // 2])
    x2 = ttnn.slice(x, [0, 0, 0, d // 2], [x.shape[0], x.shape[1], x.shape[2], d])
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


class TtMistral4MLA(LightweightModule):
    """Multi-head Latent Attention. forward(x, cos, sin) -> [B, S, hidden]."""

    def __init__(self, mesh, sd, cfg, eps):
        super().__init__()
        self.mesh = mesh
        self.eps = eps
        self.H = cfg.num_attention_heads
        self.qk = cfg.qk_head_dim
        self.nope = cfg.qk_nope_head_dim
        self.rope = cfg.qk_rope_head_dim
        self.vd = cfg.v_head_dim
        self.kvl = cfg.kv_lora_rank
        self.scale = self.qk ** (-0.5)

        self.w_qa = _lin(sd["self_attn.q_a_proj.weight"], mesh)
        self.w_qan = _norm(sd["self_attn.q_a_layernorm.weight"], mesh)
        self.w_qb = _lin(sd["self_attn.q_b_proj.weight"], mesh)
        self.w_kva = _lin(sd["self_attn.kv_a_proj_with_mqa.weight"], mesh)
        self.w_kvan = _norm(sd["self_attn.kv_a_layernorm.weight"], mesh)
        self.w_kvb = _lin(sd["self_attn.kv_b_proj.weight"], mesh)
        self.w_o = _lin(sd["self_attn.o_proj.weight"], mesh)

        # de-interleave permutation matrix for interleaved RoPE: out[i]=x[2i] (i<d/2) else x[2(i-d/2)+1]
        perm = [2 * i for i in range(self.rope // 2)] + [2 * i + 1 for i in range(self.rope // 2)]
        P = torch.zeros(self.rope, self.rope)
        for i, p in enumerate(perm):
            P[p, i] = 1.0
        self.P = _lin(P.T, mesh)  # _lin transposes; we want the matrix itself, so pass P.T

        # === A6 compressed-latent paged flash-MLA (now CORRECT; expanded-kv remains the wired default) ===
        # forward_decode_mla caches only the kv_a latent (kvl+rope=320/token, 25.6x smaller than the
        # expanded k+v = 2*n_heads*qk = 8192/token) and absorbs kv_b into q (wkv_b1) + output (wkv_b2).
        # It is numerically verified:
        # op-isolation 0.99+ (test_m4_flash_mla_correctness), live single-MLA 0.99974 (test_m4_mla_compressed_
        # decode), 2-layer model 0.9978 (test_m4_text_decode_compressed). The earlier "0.02/bimodal/
        # nondeterministic" failure was an UNDERSIZED KV CACHE: init_compressed_cache built one block sized to
        # max_seq, which for short max_seq was < the op's k_chunk_size (128); the decode op reads a full
        # k_chunk per step, so a too-small block makes it read out-of-bounds past the page and corrupts the
        # output (the OOB region's stale L1 contents varied -> looked nondeterministic). Fixed by flooring the
        # block at 128 (see init_compressed_cache). The op itself is correct here (head_dim_v=kvl=256, vDHt=8).
        # Expanded-kv decode stays the wired default until the compressed path is full-depth + trace validated;
        # wiring it in is the long-context follow-up (25.6x smaller KV). Weights cheap to build, unconditional.
        # Split kv_b -> wkv_b1 (absorb into q) + wkv_b2 (output). kv_b [H*(nope+vd), kvl] -> [H, nope+vd, kvl].
        # Split kv_b -> wkv_b1 (absorb into q) + wkv_b2 (output). kv_b [H*(nope+vd), kvl] -> [H, nope+vd, kvl].
        def _batched(w):  # [H, m, n] per-head batched-matmul weight, replicated, TILE
            return ttnn.from_torch(
                w.contiguous().to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        kvb = sd["self_attn.kv_b_proj.weight"].reshape(self.H, self.nope + self.vd, self.kvl)
        self.wkv_b1 = _batched(kvb[:, : self.nope, :])  # [H, nope, kvl]   q_nope @ wkv_b1 -> [H,*,kvl]
        self.wkv_b2 = _batched(kvb[:, self.nope :, :].transpose(1, 2))  # [H, kvl, vd]  ctx @ wkv_b2 -> [H,*,vd]
        self._sdpa_prog = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self._sdpa_ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
        )

    def _rope(self, x, cos, sin):
        xd = ttnn.matmul(x, self.P)  # de-interleave
        return ttnn.add(ttnn.mul(xd, cos), ttnn.mul(_rotate_half(xd), sin))

    def _qkv(self, x, cos, sin):
        """x [B,S,hidden] -> q_states/k_states [B,H,S,qk_head_dim], value [B,H,S,v_head_dim]."""
        B, S = x.shape[0], x.shape[1]
        q = ttnn.linear(ttnn.rms_norm(ttnn.linear(x, self.w_qa), epsilon=self.eps, weight=self.w_qan), self.w_qb)
        qh = ttnn.transpose(ttnn.reshape(q, (B, S, self.H, self.qk)), 1, 2)
        q_nope = ttnn.slice(qh, [0, 0, 0, 0], [B, self.H, S, self.nope])
        q_rot = ttnn.slice(qh, [0, 0, 0, self.nope], [B, self.H, S, self.qk])
        q_states = ttnn.concat([q_nope, self._rope(q_rot, cos, sin)], dim=-1)

        kv_a = ttnn.linear(x, self.w_kva)
        kv_pass = ttnn.rms_norm(ttnn.slice(kv_a, [0, 0, 0], [B, S, self.kvl]), epsilon=self.eps, weight=self.w_kvan)
        kv_b = ttnn.linear(kv_pass, self.w_kvb)
        kh = ttnn.transpose(ttnn.reshape(kv_b, (B, S, self.H, self.nope + self.vd)), 1, 2)
        k_nope = ttnn.slice(kh, [0, 0, 0, 0], [B, self.H, S, self.nope])
        value = ttnn.slice(kh, [0, 0, 0, self.nope], [B, self.H, S, self.nope + self.vd])
        k_rot = ttnn.reshape(ttnn.slice(kv_a, [0, 0, self.kvl], [B, S, self.kvl + self.rope]), (B, 1, S, self.rope))
        k_rot = ttnn.repeat(self._rope(k_rot, cos, sin), ttnn.Shape([1, self.H, 1, 1]))
        k_states = ttnn.concat([k_nope, k_rot], dim=-1)
        return q_states, k_states, value

    def forward(self, x, cos, sin):
        B, S = x.shape[0], x.shape[1]
        q_states, k_states, value = self._qkv(x, cos, sin)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_states, k_states, value, is_causal=True, scale=self.scale
        )
        attn = ttnn.reshape(ttnn.transpose(attn, 1, 2), (B, S, -1))
        return ttnn.linear(attn, self.w_o)

    def init_kv_cache(self, batch, max_seq):
        """Allocate the expanded KV cache: k [batch, n_heads, max_seq, qk_head_dim], v [..., v_head_dim].
        (qk_head_dim == v_head_dim == 128 for this checkpoint, but keep the dims distinct for correctness.)"""
        mk = lambda d: ttnn.from_torch(
            torch.zeros(batch, self.H, max_seq, d, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        return [mk(self.qk), mk(self.vd)]  # [k_cache (qk_head_dim), v_cache (v_head_dim)]

    def forward_prefill(self, x, cos, sin, kv_cache):
        """Prefill: full causal attention over the prompt AND populate the KV cache (positions
        0..S-1) so decode can continue. Same math as forward(); adds the cache fill per user."""
        B, S = x.shape[0], x.shape[1]
        q_states, k_states, value = self._qkv(x, cos, sin)  # [B,H,S,qk]
        for b in range(B):
            kf = k_states if B == 1 else ttnn.slice(k_states, [b, 0, 0, 0], [b + 1, self.H, S, self.qk])
            vf = value if B == 1 else ttnn.slice(value, [b, 0, 0, 0], [b + 1, self.H, S, self.vd])
            ttnn.fill_cache(kv_cache[0], kf, b)
            ttnn.fill_cache(kv_cache[1], vf, b)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_states, k_states, value, is_causal=True, scale=self.scale
        )
        attn = ttnn.reshape(ttnn.transpose(attn, 1, 2), (B, S, -1))
        return ttnn.linear(attn, self.w_o)

    def init_paged_kv_cache(self, batch, max_seq, block_size=128):
        """Paged expanded-k/v cache for CHUNKED prefill (criteria A6/C1/C6): k/v blocks
        [max_blocks, n_heads, block_size, head_dim] + page_table [batch, blocks_per_user]. Lets prefill
        run in `block_size`-token chunks (chunked_scaled_dot_product_attention) so the prompt never
        materializes the full SxS attention in L1 — the structural cap of single-shot prefill (~4K)."""
        nblk_user = (max_seq + block_size - 1) // block_size
        max_blocks = batch * nblk_user
        mk = lambda d: ttnn.from_torch(
            torch.zeros(max_blocks, self.H, block_size, d, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        page_table = ttnn.from_torch(
            torch.arange(max_blocks, dtype=torch.int32).reshape(batch, nblk_user),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        return [mk(self.qk), mk(self.vd)], page_table

    def forward_prefill_chunked(self, x, cos, sin, paged_kv, page_table, chunk=128, block_size=128):
        """Chunked prefill: process the prompt in `chunk`-token windows against a PAGED k/v cache so L1
        holds only one chunk's attention at a time (unlocks ISL >> single-shot's ~4K cap). Same causal
        math as forward_prefill. Requires S % chunk == 0 (prompt padded upstream). q/k/v head_dim==qk."""
        B, S, hidden = x.shape[0], x.shape[1], x.shape[2]
        kc, vc = paged_kv
        outs = []
        for cs in range(0, S, chunk):
            ce = cs + chunk
            xc = ttnn.slice(x, [0, cs, 0], [B, ce, hidden])
            cosc = ttnn.slice(cos, [0, 0, cs, 0], [cos.shape[0], cos.shape[1], ce, cos.shape[3]])
            sinc = ttnn.slice(sin, [0, 0, cs, 0], [sin.shape[0], sin.shape[1], ce, sin.shape[3]])
            qc, kc_s, vc_s = self._qkv(xc, cosc, sinc)  # [B,H,chunk,*]
            chunk_pt = ttnn.slice(page_table, [0, cs // block_size], [B, ce // block_size])
            for b in range(B):
                kf = kc_s if B == 1 else ttnn.slice(kc_s, [b, 0, 0, 0], [b + 1, self.H, chunk, self.qk])
                vf = vc_s if B == 1 else ttnn.slice(vc_s, [b, 0, 0, 0], [b + 1, self.H, chunk, self.vd])
                ttnn.experimental.paged_fill_cache(kc, kf, chunk_pt, batch_idx=b)
                ttnn.experimental.paged_fill_cache(vc, vf, chunk_pt, batch_idx=b)
            ac = ttnn.transformer.chunked_scaled_dot_product_attention(
                qc, kc, vc, page_table, cs
            )  # [B,H,chunk,vd] causal; default scale == 1/sqrt(qk)==self.scale (noconvert rejects non-py-float)
            outs.append(ac)
        attn = ttnn.concat(outs, dim=2) if len(outs) > 1 else outs[0]  # [B,H,S,vd]
        attn = ttnn.reshape(ttnn.transpose(attn, 1, 2), (B, S, -1))
        return ttnn.linear(attn, self.w_o)

    def forward_decode(self, x, pos_t, cos, sin, kv_cache):
        """Single-token decode: x [B,1,hidden], pos_t an int32 device tensor [B] of cur positions.
        Writes k/v to kv_cache at pos_t (tensor-indexed -> trace-compatible) and runs flash-decode
        over the cached sequence. Expanded-k/v cache => standard decode op."""
        B = x.shape[0]
        q_states, k_states, value = self._qkv(x, cos, sin)  # [B,H,1,*]
        # tensor-indexed cache write (paged_update_cache, input [1,B,H,dh]) keeps the position out
        # of the graph so the decode step can be captured as a trace and replayed.
        k_1BKD = _to_decode_shard(ttnn.permute(k_states, (2, 0, 1, 3)), self.mesh)  # [B,H,1,qk]->[1,B,H,qk] sharded
        v_1BKD = _to_decode_shard(ttnn.permute(value, (2, 0, 1, 3)), self.mesh)
        ttnn.experimental.paged_update_cache(kv_cache[0], k_1BKD, update_idxs_tensor=pos_t)
        ttnn.experimental.paged_update_cache(kv_cache[1], v_1BKD, update_idxs_tensor=pos_t)
        q_dec = ttnn.permute(q_states, (2, 0, 1, 3))  # [B,H,1,qk] -> [1,B,H,qk]
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_dec, kv_cache[0], kv_cache[1], cur_pos_tensor=pos_t, scale=self.scale
        )
        attn = ttnn.reshape(ttnn.permute(attn, (1, 0, 2, 3)), (B, 1, self.H * self.vd))  # SDPA out is v_head_dim
        return ttnn.linear(attn, self.w_o)

    # ---- A6: compressed-latent paged MLA decode (12.8x smaller KV cache) ----
    def init_compressed_cache(self, batch, max_seq):
        """Compressed-latent KV cache [batch,1,block,kvl+rope] + page_table [batch,1]. One block per
        user sized to max_seq (tile-aligned) — the layout the flash-MLA decode op reads correctly. The
        12.8x compression comes from the latent dim (kvl+rope=320 vs expanded n_heads*qk=4096); finer
        block-paging granularity is a separate memory-management follow-up.

        CRITICAL: the block must be >= k_chunk_size (128) of the SDPA program config. With one block per
        user, the decode op reads a full k_chunk per step; if block < k_chunk it reads past the page (the
        rest of the chunk is out-of-bounds), which silently corrupts the output (this was the A6 bug —
        an undersized cache, NOT an op limit; the op is correct for head_dim_v=kvl=256, vDHt=8)."""
        block = max(128, ((max_seq + 31) // 32) * 32)
        cache = ttnn.from_torch(
            torch.zeros(batch, 1, block, self.kvl + self.rope, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        pt = ttnn.from_torch(
            torch.arange(batch, dtype=torch.int32).reshape(batch, 1),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        return cache, pt

    def _hshard(self, t, rows, width):  # height-shard [32,width] over rows//32 cores
        nc = max(1, rows // 32)
        cores = ttnn.num_cores_to_corerangeset(nc, self.mesh.compute_with_storage_grid_size(), row_wise=True)
        return ttnn.to_memory_config(
            t,
            ttnn.create_sharded_memory_config(
                shape=[32, width],
                core_grid=cores,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
        )

    def forward_decode_mla(self, x, pos_t, cos, sin, cache, page_table):
        """Compressed-latent paged flash-MLA decode (A6): caches only the kv_a latent [kvl+rope] (MQA,
        12.8x smaller than expanded), absorbs kv_b into q (wkv_b1) and output (wkv_b2). x [B,1,hidden]."""
        B, H, kvl, rope, nope, vd = x.shape[0], self.H, self.kvl, self.rope, self.nope, self.vd
        # compressed latent: [kv_pass (normed) | k_rot]
        kv_a = ttnn.linear(x, self.w_kva)  # [B,1,kvl+rope]
        kv_pass = ttnn.rms_norm(ttnn.slice(kv_a, [0, 0, 0], [B, 1, kvl]), epsilon=self.eps, weight=self.w_kvan)
        k_rot = self._rope(ttnn.reshape(ttnn.slice(kv_a, [0, 0, kvl], [B, 1, kvl + rope]), (B, 1, 1, rope)), cos, sin)
        latent = ttnn.concat([ttnn.reshape(kv_pass, (B, 1, 1, kvl)), k_rot], dim=-1)  # [B,1,1,kvl+rope]
        # [1,B,1,d] height-sharded one user per core (shard [32,d]; the seqlen-1 dim is tile-padded
        # to 32) — the 1-kv-head latent write layout (cf. deepseek mla1d.py _fwd_decode).
        cores_b = ttnn.num_cores_to_corerangeset(B, self.mesh.compute_with_storage_grid_size(), row_wise=True)
        lat_mem = ttnn.create_sharded_memory_config(
            shape=[32, kvl + rope],
            core_grid=cores_b,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        lat_in = ttnn.to_memory_config(ttnn.permute(latent, (1, 0, 2, 3)), lat_mem)  # [1,B,1,d] sharded on B cores
        ttnn.experimental.paged_update_cache(cache, lat_in, update_idxs_tensor=pos_t, page_table=page_table)
        # q absorption: q_nope @ wkv_b1 -> latent space; concat q_rope
        q = ttnn.linear(ttnn.rms_norm(ttnn.linear(x, self.w_qa), epsilon=self.eps, weight=self.w_qan), self.w_qb)
        qh = ttnn.transpose(ttnn.reshape(q, (B, 1, H, self.qk)), 1, 2)  # [B,H,1,qk]
        q_nope = ttnn.reshape(ttnn.permute(ttnn.slice(qh, [0, 0, 0, 0], [B, H, 1, nope]), (1, 0, 2, 3)), (H, B, nope))
        q_lat_nope = ttnn.reshape(ttnn.matmul(q_nope, self.wkv_b1), (H, B, 1, kvl))  # [H,B,1,kvl]
        q_lat_nope = ttnn.permute(q_lat_nope, (2, 1, 0, 3))  # [1,B,H,kvl]
        q_rope = ttnn.permute(self._rope(ttnn.slice(qh, [0, 0, 0, nope], [B, H, 1, self.qk]), cos, sin), (2, 0, 1, 3))
        q_lat = self._hshard(ttnn.concat([q_lat_nope, q_rope], dim=-1), B * H, kvl + rope)  # [1,B,H,kvl+rope] sharded
        out_mem = ttnn.create_sharded_memory_config(
            shape=[32, kvl],
            core_grid=ttnn.num_cores_to_corerangeset(
                max(1, B * H // 32), self.mesh.compute_with_storage_grid_size(), row_wise=True
            ),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        attn = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_lat,
            cache,
            page_table_tensor=page_table,
            cur_pos_tensor=pos_t,
            head_dim_v=kvl,
            scale=self.scale,
            program_config=self._sdpa_prog,
            compute_kernel_config=self._sdpa_ck,
            memory_config=out_mem,
        )  # [1,B,H,kvl]
        # output absorption: ctx @ wkv_b2 -> v_head
        ctx = ttnn.reshape(
            ttnn.permute(ttnn.to_memory_config(attn, ttnn.DRAM_MEMORY_CONFIG), (2, 1, 0, 3)), (H, B, kvl)
        )
        o = ttnn.reshape(ttnn.matmul(ctx, self.wkv_b2), (H, B, 1, vd))  # [H,B,1,vd]
        o = ttnn.reshape(ttnn.permute(o, (1, 0, 2, 3)), (B, 1, H * vd))
        return ttnn.linear(o, self.w_o)


class TtMistral4MoE(LightweightModule):
    """Ungrouped top-4/128 MoE + shared SwiGLU expert. forward(x) -> [B, S, hidden].

    shard_experts=True shards the 128 experts across the mesh (E/n_dev per device) so the full
    36-layer model fits in DRAM; partial sums are combined with all_reduce_async(Sum). Each device
    extracts its local experts' routing weights on-device via a precomputed sharded column selector
    (W_full @ select_d -> W_local), so the whole MoE stays on device and the decode step is fully
    trace-capturable (no host round-trip). Sparse dispatch is the further perf path — see #14.
    """

    def __init__(self, mesh, sd, cfg, shard_experts=False, expert_dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh = mesh
        from models.tt_transformers.tt.ccl import get_num_links

        self.n_links = get_num_links(mesh, cluster_axis=1)  # P150x8 = 2 eth links/axis; use both for CCL bandwidth
        self.E = cfg.n_routed_experts
        self.k = cfg.num_experts_per_tok
        self.interm = cfg.moe_intermediate_size
        self.hidden = cfg.hidden_size
        self.shard = shard_experts
        self.w_gate = _lin(sd["mlp.gate.weight"], mesh)
        gup, down = sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"]  # [E,2I,H],[E,H,I]
        if shard_experts:
            self.per = self.E // mesh.get_num_devices()
            self.gup_sh = ttnn.as_tensor(
                gup.transpose(1, 2).contiguous().to(torch.bfloat16),
                dtype=expert_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )  # [per,H,2I]/device
            self.down_sh = ttnn.as_tensor(
                down.transpose(1, 2).contiguous().to(torch.bfloat16),
                dtype=expert_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )  # [per,I,H]/device
            # sharded column selector: device d holds [E,per] selecting its experts' routing weights
            # (W_full @ select_d -> W_local), replacing the host round-trip with an on-device matmul.
            nd = mesh.get_num_devices()
            sel = torch.zeros(nd, self.E, self.per)
            for d in range(nd):
                for j in range(self.per):
                    sel[d, d * self.per + j, j] = 1.0
            self.expert_select = ttnn.from_torch(
                sel.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            # one-hot expert->device map [1,1,E,n_dev] (expert e on device e//per) for all_to_all
            # sparse dispatch: row e, col d == 1 iff expert e lives on device d. Replicated, uint16, RM.
            self.expert_map = ttnn.from_torch(
                torch.eye(nd, dtype=torch.int32).repeat_interleave(self.per, dim=0).reshape(1, 1, self.E, nd),
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.w_gup = [_lin(gup[e], mesh) for e in range(self.E)]
            self.w_down = [_lin(down[e], mesh) for e in range(self.E)]
        self.w_sg = _lin(sd["mlp.shared_experts.gate_proj.weight"], mesh)
        self.w_su = _lin(sd["mlp.shared_experts.up_proj.weight"], mesh)
        self.w_sd = _lin(sd["mlp.shared_experts.down_proj.weight"], mesh)

    def _route(self, x, B, S):
        # softmax -> kth-largest threshold mask -> normalize (no scatter); W replicated [B,S,E].
        # NB: routing precision is bf16-bounded — ttnn.topk requires bf16/bfp8 input, so top-k
        # selection can't be done in fp32 on device (a borderline expert can flip vs the fp32
        # reference; this is the main source of the full-depth logit-PCC gap, ~0.982). An exact
        # selection would need a host fp32 top-k round-trip per layer (not worth the perf cost).
        probs = ttnn.softmax(ttnn.linear(x, self.w_gate), dim=-1)
        kth = ttnn.slice(ttnn.topk(probs, self.k, dim=-1)[0], [0, 0, self.k - 1], [B, S, self.k])
        masked = ttnn.mul(probs, ttnn.ge(probs, kth))
        return ttnn.div(masked, ttnn.sum(masked, dim=-1, keepdim=True))

    def forward(self, x):
        B, S, I, H = x.shape[0], x.shape[1], self.interm, self.hidden
        W = self._route(x, B, S)
        if self.shard:
            # Flatten the (batch, seq) token grid to a single token axis T=B*S — experts are
            # token-wise, so this makes the batched-expert matmul batch-agnostic (B users decode
            # together) while staying identical for B=1.
            T = B * S
            xf = ttnn.reshape(x, (1, T, H))
            # extract each device's local experts' routing weights on-device (no host round-trip):
            # W_full [1,T,E] @ select_d [E,per] -> W_local [1,T,per]. Keeps the MoE trace-capturable.
            W_sh = ttnn.matmul(ttnn.reshape(W, (1, T, self.E)), self.expert_select)
            # BATCHED experts: the per local experts in 2 batched matmuls (not `per` sequential
            # linears) — the dominant decode/forward speedup. Stacked weights gup_sh [per,H,2I],
            # down_sh [per,I,H]; broadcast tokens over the expert (batch) dim.
            xb = ttnn.repeat(xf, ttnn.Shape([self.per, 1, 1]))  # [per,T,H]
            gu = ttnn.matmul(xb, self.gup_sh)  # [per,T,2I]
            h = ttnn.mul(
                ttnn.silu(ttnn.slice(gu, [0, 0, 0], [self.per, T, I])), ttnn.slice(gu, [0, 0, I], [self.per, T, 2 * I])
            )
            y = ttnn.matmul(h, self.down_sh)  # [per,T,H]
            w = ttnn.permute(W_sh, (2, 1, 0))  # [1,T,per] -> [per,T,1]
            acc = ttnn.reshape(ttnn.sum(ttnn.mul(y, w), dim=0), (B, S, H))  # weighted sum over local experts
            # trace-safe collective: async all-reduce over the 8 expert-sharded devices (cluster
            # axis 1 of the 1x8 mesh) sums each device's local-expert contribution.
            experts = ttnn.experimental.all_reduce_async(
                acc,
                cluster_axis=1,
                mesh_device=self.mesh,
                num_links=self.n_links,
                math_op=ttnn.ReduceType.Sum,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            )
        else:
            acc = None
            for e in range(self.E):
                gu = ttnn.linear(x, self.w_gup[e])
                g = ttnn.slice(gu, [0, 0, 0], [B, S, I])
                u = ttnn.slice(gu, [0, 0, I], [B, S, 2 * I])
                y = ttnn.linear(ttnn.mul(ttnn.silu(g), u), self.w_down[e])
                contrib = ttnn.mul(y, ttnn.slice(W, [0, 0, e], [B, S, e + 1]))
                acc = contrib if acc is None else ttnn.add(acc, contrib)
            experts = acc
        return ttnn.add(experts, self._shared(x))

    def _shared(self, x):
        return ttnn.linear(ttnn.mul(ttnn.silu(ttnn.linear(x, self.w_sg)), ttnn.linear(x, self.w_su)), self.w_sd)

    def _forward_sparse(self, x, sub_device_id=None):
        """Sparse expert-parallel MoE for batched decode (x sharded [bpd,1,H], B>=n_dev): dispatch each token
        to the devices holding its top-k experts (all_to_all_dispatch), compute only those, combine
        back (all_to_all_combine). Streams/computes far fewer experts than the dense-local path at
        low batch. Requires shard_experts. See deepseek_v3/tt/moe.py for the reference pipeline.

        The dispatch->experts->combine->weight+sum pipeline is VALIDATED on the native 1x8 mesh at PCC
        0.9998 vs the dense path (test_m4_moe_roundtrip). REQUIREMENT: the input tokens must be SHARDED
        1/device across the 8 (ShardTensorToMesh dim0) — NOT replicated; cluster_axis=1 spans all 8 on
        (1,8) so every expert is reachable. (cluster_axis=None hangs the device; a 2x4 mesh row-splits
        the experts.) `x` here is expected batch-sharded (B>=n_dev, 1 token/device). Dense forward() (which
        takes replicated tokens) stays the verified default; this is the B>=8 throughput path. The
        model-level reshard (replicated MLA output -> sharded for the MoE -> all-gather combine back) is
        the integration step tracked by test_m4_moe_sparse. See MISTRAL4_DESIGN.md (A10 BREAKTHROUGH)."""
        bpd, H, I = x.shape[0], self.hidden, self.interm  # bpd = batch tokens per device (x is sharded)
        sdid = sub_device_id if sub_device_id is not None else ttnn.SubDeviceId(0)
        probs = ttnn.softmax(ttnn.linear(x, self.w_gate), dim=-1)  # [bpd,1,E]
        tw, ti = ttnn.topk(probs, self.k, dim=-1)  # [bpd,1,k] weights + indices
        tw = ttnn.div(tw, ttnn.sum(tw, dim=-1, keepdim=True))  # renormalize top-k weights
        x4 = ttnn.to_layout(ttnn.reshape(x, (bpd, 1, 1, H)), ttnn.ROW_MAJOR_LAYOUT)
        idx4 = ttnn.to_layout(ttnn.typecast(ttnn.reshape(ti, (bpd, 1, 1, self.k)), ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT)
        # dispatch (return-API): tokens routed to their experts' devices. disp [1,Btot,1,H] / meta [1,Btot,1,k]
        # per device (Btot = total tokens this device's experts serve). cluster_axis=1 spans all 8 on (1,8).
        disp, meta = ttnn.all_to_all_dispatch(
            x4, idx4, self.expert_map, cluster_axis=1, num_links=self.n_links, subdevice_id=sdid
        )
        # broadcast each device's dispatched tokens across its `per` local experts -> batched-expert matmul
        d = ttnn.repeat(disp, ttnn.Shape([self.per, 1, 1, 1]))  # [per, Btot, 1, H]
        bsz = d.shape[1]
        d = ttnn.to_layout(ttnn.reshape(d, (self.per, bsz, H)), ttnn.TILE_LAYOUT)  # [per, Btot, H]
        gu = ttnn.matmul(d, self.gup_sh)  # [per,Btot,2I]
        h = ttnn.mul(
            ttnn.silu(ttnn.slice(gu, [0, 0, 0], [self.per, bsz, I])), ttnn.slice(gu, [0, 0, I], [self.per, bsz, 2 * I])
        )
        y = ttnn.matmul(h, self.down_sh)  # [per,Btot,H]
        eo = ttnn.to_layout(ttnn.reshape(y, (self.per, bsz, 1, H)), ttnn.ROW_MAJOR_LAYOUT)
        comb = ttnn.all_to_all_combine(
            eo, meta, self.expert_map, cluster_axis=1, num_links=self.n_links, subdevice_id=sdid
        )  # [k, bpd, 1, H] each token's top-k selected-expert outputs
        comb = ttnn.reshape(ttnn.to_layout(comb, ttnn.TILE_LAYOUT), (self.k, bpd, H))
        w = ttnn.reshape(ttnn.permute(tw, (2, 0, 1)), (self.k, bpd, 1))  # [bpd,1,k] -> [k,bpd,1]
        experts = ttnn.reshape(ttnn.sum(ttnn.mul(comb, w), dim=0), (bpd, 1, H))
        return ttnn.add(experts, self._shared(x))

    def forward_sparse(self, x, sub_device_id=None):
        """Replicated-in / replicated-out sparse MoE for batched decode (x replicated [B,1,H], B>=n_dev):
        ttnn.mesh_partition shards the batch 1/device (device-side, the inverse of all_gather, trace-safe),
        runs the validated sparse dispatch path, then all-gathers back to replicated for the residual.
        Matches the dense forward at PCC ~0.999; the throughput win is computing only top-k experts/token
        (vs all `per` local experts dense). Requires shard_experts + a loaded sub-device manager."""
        xs = ttnn.mesh_partition(x, dim=0, cluster_axis=1)  # [B,1,H] replicated -> [B/n_dev,1,H] sharded
        out = self._forward_sparse(xs, sub_device_id=sub_device_id)  # [B/n_dev,1,H] sharded
        return ttnn.all_gather(out, dim=0, cluster_axis=1)  # -> [B,1,H] replicated


class TtMistral4DecoderLayer(LightweightModule):
    """input_layernorm -> MLA -> +residual -> post_attention_layernorm -> MoE -> +residual."""

    def __init__(self, mesh, sd, cfg, eps, shard_experts=False, expert_dtype=ttnn.bfloat16):
        super().__init__()
        self.eps = eps
        self.w_in = _norm(sd["input_layernorm.weight"], mesh)
        self.w_post = _norm(sd["post_attention_layernorm.weight"], mesh)
        self.mla = TtMistral4MLA(mesh, sd, cfg, eps)
        self.moe = TtMistral4MoE(mesh, sd, cfg, shard_experts=shard_experts, expert_dtype=expert_dtype)

    def forward(self, x, cos, sin):
        h = ttnn.add(x, self.mla(ttnn.rms_norm(x, epsilon=self.eps, weight=self.w_in), cos, sin))
        return ttnn.add(h, self.moe(ttnn.rms_norm(h, epsilon=self.eps, weight=self.w_post)))

    def forward_decode(self, x, pos_t, cos, sin, kv_cache, use_sparse=False, sub_device_id=None):
        h = ttnn.add(
            x,
            self.mla.forward_decode(ttnn.rms_norm(x, epsilon=self.eps, weight=self.w_in), pos_t, cos, sin, kv_cache),
        )
        mn = ttnn.rms_norm(h, epsilon=self.eps, weight=self.w_post)
        # B>=n_dev: sparse top-k expert dispatch (computes only routed experts vs all `per` local dense).
        moe = self.moe.forward_sparse(mn, sub_device_id=sub_device_id) if use_sparse else self.moe(mn)
        return ttnn.add(h, moe)

    def forward_decode_mla(self, x, pos_t, cos, sin, cache, page_table):
        h = ttnn.add(
            x,
            self.mla.forward_decode_mla(
                ttnn.rms_norm(x, epsilon=self.eps, weight=self.w_in), pos_t, cos, sin, cache, page_table
            ),
        )
        return ttnn.add(h, self.moe(ttnn.rms_norm(h, epsilon=self.eps, weight=self.w_post)))

    def forward_prefill(self, x, cos, sin, kv_cache):
        h = ttnn.add(
            x, self.mla.forward_prefill(ttnn.rms_norm(x, epsilon=self.eps, weight=self.w_in), cos, sin, kv_cache)
        )
        return ttnn.add(h, self.moe(ttnn.rms_norm(h, epsilon=self.eps, weight=self.w_post)))

    def forward_prefill_chunked(self, x, cos, sin, paged_kv, page_table, chunk=128, block_size=128):
        h = ttnn.add(
            x,
            self.mla.forward_prefill_chunked(
                ttnn.rms_norm(x, epsilon=self.eps, weight=self.w_in), cos, sin, paged_kv, page_table, chunk, block_size
            ),
        )
        return ttnn.add(h, self.moe(ttnn.rms_norm(h, epsilon=self.eps, weight=self.w_post)))


class TtMistral4TextModel(LightweightModule):
    """Stacked decoder layers + final norm + LM head. forward(embedded_hidden, cos, sin) -> logits.

    Embedding is the caller's responsibility (a trivial row gather); this keeps the module a pure
    device forward over the decoder stack. `sd` is the model's named_parameters dict with HF keys
    (``model.layers.{i}.*``, ``model.norm.weight``, ``lm_head.weight``).
    """

    def __init__(self, mesh, sd, cfg, n_layers, eps, shard_experts=False, expert_dtype=ttnn.bfloat16):
        super().__init__()
        self.eps = eps
        self.layers = []
        for i in range(n_layers):
            pfx = f"model.layers.{i}."
            layer_sd = {k[len(pfx) :]: v for k, v in sd.items() if k.startswith(pfx)}
            self.layers.append(
                TtMistral4DecoderLayer(mesh, layer_sd, cfg, eps, shard_experts=shard_experts, expert_dtype=expert_dtype)
            )
        self.mesh = mesh
        # framework CCL helper — persistent global-semaphore pools (trace-safe) for the LM-head
        # all-gather; reused as-is, not modified. (The tt_all_gather *wrapper* no-ops on a 1xN mesh
        # via a `1 in shape` guard, so we call the raw op with cluster_axis=1 + these semaphores.)
        from models.tt_transformers.tt.ccl import TT_CCL, get_num_links

        self.ccl = TT_CCL(mesh)
        self.n_links = get_num_links(mesh, cluster_axis=1)  # P150x8 = 2 eth links/axis (LM-head all-gather)
        self.w_norm = _norm(sd["model.norm.weight"], mesh)
        # Sharded LM head: the [hidden, vocab=131072] weight is sharded along vocab across the 8
        # devices (16384/device, ~128MB vs 1.07GB replicated) and the per-device logit slices are
        # gathered with all_gather_async (trace-safe). bf16 + HiFi here deliberately (accuracy-
        # critical final projection), distinct from the bfp8 experts.
        self.w_lm = ttnn.as_tensor(
            sd["lm_head.weight"].transpose(0, 1).contiguous().to(torch.bfloat16),  # [hidden, vocab]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )

    def _lm_head(self, hidden):
        B, S = hidden.shape[0], hidden.shape[1]
        local = ttnn.linear(hidden, self.w_lm)  # [B,S,vocab/n_dev] (a distinct vocab slice/device)
        local = ttnn.reshape(local, (B, 1, S, local.shape[-1]))  # 4D for the line-all-gather (dim=3)
        full = ttnn.experimental.all_gather_async(
            local,
            dim=3,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=1),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
            num_links=self.n_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=1,
        )
        return ttnn.reshape(full, (B, S, full.shape[-1]))  # [B,S,vocab]

    def forward(self, hidden, cos, sin):
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)
        return self._lm_head(ttnn.rms_norm(hidden, epsilon=self.eps, weight=self.w_norm))

    def init_kv_caches(self, batch, max_seq):
        return [layer.mla.init_kv_cache(batch, max_seq) for layer in self.layers]

    def init_compressed_caches(self, batch, max_seq):
        """Per-layer compressed-latent paged caches [(cache, page_table), ...] for the A6 MLA decode."""
        return [layer.mla.init_compressed_cache(batch, max_seq) for layer in self.layers]

    def forward_decode_mla(self, hidden, pos_t, cos, sin, caches):
        """Compressed-latent (A6) decode: 12.8x smaller KV than forward_decode. `caches` is the list
        of (cache, page_table) from init_compressed_caches."""
        for layer, (cache, pt) in zip(self.layers, caches):
            hidden = layer.forward_decode_mla(hidden, pos_t, cos, sin, cache, pt)
        return self._lm_head(ttnn.rms_norm(hidden, epsilon=self.eps, weight=self.w_norm))

    def forward_prefill(self, hidden, cos, sin, kv_caches):
        for layer, kv in zip(self.layers, kv_caches):
            hidden = layer.forward_prefill(hidden, cos, sin, kv)
        return self._lm_head(ttnn.rms_norm(hidden, epsilon=self.eps, weight=self.w_norm))

    def init_paged_kv_caches(self, batch, max_seq, block_size=128):
        """Per-layer paged expanded-k/v caches [(paged_kv, page_table), ...] for chunked prefill."""
        return [layer.mla.init_paged_kv_cache(batch, max_seq, block_size) for layer in self.layers]

    def forward_prefill_chunked(self, hidden, cos, sin, caches, chunk=128, block_size=128):
        """Chunked prefill over paged k/v (criteria A6/C1/C6): lifts the single-shot ~4K L1 cap so the
        ISL sweep can reach 16K+. `caches` is the list of (paged_kv, page_table) from init_paged_kv_caches."""
        for layer, (paged_kv, pt) in zip(self.layers, caches):
            hidden = layer.forward_prefill_chunked(hidden, cos, sin, paged_kv, pt, chunk, block_size)
        return self._lm_head(ttnn.rms_norm(hidden, epsilon=self.eps, weight=self.w_norm))

    def forward_decode(self, hidden, pos_t, cos, sin, kv_caches, use_sparse=False, sub_device_id=None):
        for layer, kv in zip(self.layers, kv_caches):
            hidden = layer.forward_decode(
                hidden, pos_t, cos, sin, kv, use_sparse=use_sparse, sub_device_id=sub_device_id
            )
        return self._lm_head(ttnn.rms_norm(hidden, epsilon=self.eps, weight=self.w_norm))


class TtMistral4Projector(LightweightModule):
    """Mistral3 multi-modal projector: vision features -> text-hidden image embeds.

    norm(RMSNorm) -> 2x2 spatial patch-merge -> merging_layer -> linear_1 -> gelu -> linear_2.
    The 2x2 merge (channel-major, == HF F.unfold) is a one-time-per-image spatial rearrange done
    host-side here (negligible vs decode; an on-device reshape/permute is a perf follow-up); the
    norm + all linears run on device. `sd` keys are relative to multi_modal_projector. (norm.weight,
    patch_merger.merging_layer.weight, linear_1.weight, linear_2.weight).
    """

    def __init__(self, mesh, sd, cfg):
        super().__init__()
        self.mesh = mesh
        self.eps = cfg.text_config.rms_norm_eps
        self.sm = cfg.spatial_merge_size
        self.patch = cfg.vision_config.patch_size
        self.vh = cfg.vision_config.hidden_size
        self.w_norm = _norm(sd["norm.weight"], mesh)
        self.w_merge = _lin(sd["patch_merger.merging_layer.weight"], mesh)  # Linear(vh*sm^2 -> vh)
        self.w_l1 = _lin(sd["linear_1.weight"], mesh)  # Linear(vh -> text_hidden)
        self.w_l2 = _lin(sd["linear_2.weight"], mesh)  # Linear(text_hidden -> text_hidden)

    def _unfold_host(self, x, image_sizes):
        # x [n, vh] (single image) -> [n/sm^2, vh*sm^2], channel-major (== HF F.unfold). Host-side.
        sm, d = self.sm, x.shape[-1]
        h, w = image_sizes[0][0] // self.patch, image_sizes[0][1] // self.patch
        g = x.view(h, w, d).permute(2, 0, 1)  # [d,h,w]
        g = g.view(d, h // sm, sm, w // sm, sm).permute(0, 2, 4, 1, 3).reshape(d * sm * sm, -1).t()
        return g.contiguous()

    def forward(self, feats, image_sizes):
        n = feats.shape[0]
        x = ttnn.rms_norm(feats, epsilon=self.eps, weight=self.w_norm)  # [n, vh]
        x_host = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))[:n].float()
        merged = ttnn.from_torch(
            self._unfold_host(x_host, image_sizes).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )  # [n/sm^2, vh*sm^2]
        m = ttnn.linear(merged, self.w_merge)  # [n/sm^2, vh]
        h = ttnn.gelu(ttnn.linear(m, self.w_l1))  # [n/sm^2, text_hidden]
        return ttnn.linear(h, self.w_l2)  # [n/sm^2, text_hidden]
