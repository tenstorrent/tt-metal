# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA (DeepSeek sparse attention, MLA with NoPE) layer for GLM-5.3-Flash on one device.

HF ``Glm5NextTextAttention`` (qk_rope_head_dim = 0, so no rotary anywhere):
    q = q_b(rmsnorm(q_a(x)))                      [T, 64, 256]
    c = rmsnorm(kv_a(x))                           [T, 512]   (the latent, cached per token)
    k_h = c @ Wkb1[h]^T, v_h = c @ Wkb2[h]^T       (kv_b = [64 x (256 k | 256 v), 512])
    o = softmax(q k^T / 16) v -> concat heads -> o_proj
For context <= 2048 (+ the incomplete tail pool) the indexer selects every visible token, so
the layer is exact causal attention; longer contexts use the pooled-key indexer (sparse path,
``TtDSAIndexer``).

Device design (section 5.3 of the design doc, single chip = Phase 1):
  * prefill (T <= 2048): materialize K, V per head from the latent and run
    ``ttnn.transformer.scaled_dot_product_attention`` (causal, head_dim 256); the latent is
    written to the paged cache with ``paged_fill_cache``.
  * decode: absorbed MLA. q_lat[h] = q[h] @ Wkb1[h] (256 -> 512), then
    ``paged_flash_multi_latent_attention_decode`` over the latent cache (head_dim_v = 512),
    then o[h] = o_lat[h] @ Wkb2[h]^T (512 -> 256), concat heads, o_proj. The latent of the new
    token is appended with ``paged_update_cache`` first.

Cache: ``[max_blocks, 1, block_size, 512]`` bf16, paged; page_table ``[B, max_blocks_per_seq]`` int32.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.tt.config import GLM53FlashConfig

TILE = 32
CORES_PER_USER_DECODE = 4


def _compute(device, fidelity=ttnn.MathFidelity.HiFi2, fp32_acc=True, packer_l1_acc=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


def _upload(device, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=memory_config)


class TtLatentCache:
    """Paged latent cache for one DSA layer (and its page table), device-owned."""

    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        *,
        max_batch: int,
        max_seq_len: int,
        block_size: int = 32,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.block_size = block_size
        self.max_batch = max_batch
        self.blocks_per_seq = -(-max_seq_len // block_size)
        self.max_blocks = self.blocks_per_seq * max_batch
        self.dtype = dtype
        self.latent_dim = cfg.kv_lora_rank
        self.cache = ttnn.zeros(
            (self.max_blocks, 1, block_size, self.latent_dim),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # identity page table [B, blocks_per_seq]: user b owns blocks [b*bps, (b+1)*bps)
        pt = torch.arange(self.max_blocks, dtype=torch.int32).reshape(max_batch, self.blocks_per_seq)
        self.set_page_table(pt)

    def set_page_table(self, page_table: torch.Tensor):
        assert tuple(page_table.shape) == (self.max_batch, self.blocks_per_seq), page_table.shape
        self.page_table_host = page_table.to(torch.int32).contiguous()
        self.page_table = ttnn.from_torch(
            self.page_table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

    def fill(self, latent: ttnn.Tensor, user: int):
        """latent [1, 1, T_pad, 512] (T_pad % block_size == 0) -> pages of ``user`` from position 0."""
        src = latent if latent.dtype == self.dtype else ttnn.typecast(latent, self.dtype)
        ttnn.experimental.paged_fill_cache(self.cache, src, self.page_table, batch_idx=user)

    def update(self, latent_sharded: ttnn.Tensor, positions: ttnn.Tensor):
        """latent [1, B, 1(32), 512] height-sharded on B cores, positions [B] int32 row-major."""
        ttnn.experimental.paged_update_cache(
            self.cache, latent_sharded, update_idxs_tensor=positions, page_table=self.page_table
        )


class TtDSA(LightweightModule):
    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        attn_state_dict: dict[str, torch.Tensor],
        *,
        layer_idx: int,
        weight_dtype=ttnn.bfloat8_b,
    ):
        """``attn_state_dict``: raw checkpoint tensors of ``self_attn.`` (prefix stripped, fp8 dequantized)."""
        self.device = device
        self.cfg = cfg
        self.layer_idx = layer_idx
        H, Dq, Dv, L = cfg.num_attention_heads, cfg.qk_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        self.H, self.Dq, self.Dv, self.L = H, Dq, Dv, L
        self.scale = cfg.mla_softmax_scale
        sd = attn_state_dict
        self.w_qa = _upload(device, sd["q_a_proj.weight"].t(), weight_dtype)  # [4096,1536]
        self.qa_norm = _upload(device, sd["q_a_layernorm.weight"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.w_qb = _upload(device, sd["q_b_proj.weight"].t(), weight_dtype)  # [1536, H*256]
        self.w_kva = _upload(device, sd["kv_a_proj_with_mqa.weight"].t(), weight_dtype)  # [4096, 512]
        self.kva_norm = _upload(device, sd["kv_a_layernorm.weight"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        wkb = sd["kv_b_proj.weight"].reshape(H, Dq + Dv, L)  # [H, 512, L]
        wkb1 = wkb[:, :Dq, :]  # k_nope[h] = c @ wkb1[h]^T   -> [H, 256, L]
        wkb2 = wkb[:, Dq:, :]  # v[h]      = c @ wkb2[h]^T   -> [H, 256, L]
        # prefill: K = c @ Wk^T for all heads at once: [T, L] @ [L, H*256]; same for V
        self.w_k_all = _upload(
            device, wkb1.transpose(0, 1).reshape(Dq, H, L).permute(2, 1, 0).reshape(L, H * Dq), weight_dtype
        )
        self.w_v_all = _upload(device, wkb2.permute(2, 0, 1).reshape(L, H * Dv), weight_dtype)
        # decode (absorbed): per-head [H, 256, L] for q -> latent, and [H, L, 256] for latent -> v space
        self.w_absorb_q = _upload(device, wkb1.reshape(1, H, Dq, L), weight_dtype)
        self.w_absorb_o = _upload(device, wkb2.transpose(1, 2).reshape(1, H, L, Dv), weight_dtype)
        self.w_o = _upload(device, sd["o_proj.weight"].t(), weight_dtype)  # [H*256, 4096]
        self.hifi2 = _compute(device)
        self.hifi4 = _compute(device, ttnn.MathFidelity.HiFi4, packer_l1_acc=False)
        grid = device.compute_with_storage_grid_size()
        self.sdpa_prefill_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=128, exp_approx_mode=False
        )
        self.full_grid = grid
        self._decode_cfgs: dict[int, ttnn.SDPAProgramConfig] = {}
        self.sdpa_compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def _decode_sdpa_config(self, batch: int) -> ttnn.SDPAProgramConfig:
        """Flash MLA decode splits each user's heads over cores and tree-reduces the partial
        results; every reduction round keeps an output-sized fp32 block in L1, so at B=1 the full
        grid (120 cores, 7 rounds) overflows L1 with the 512-wide latent. Bound the cores per user
        to 16 (4 rounds) and let more users use more of the grid."""
        if batch not in self._decode_cfgs:
            # c_19 (tree-reduction staging) is sized (cores_per_user - 1) * (out + 2) tiles of fp32:
            # 36 tiles * 4 KB * 11 = 1.6 MB at 12 cores/user. 4 cores/user keeps the op at ~1.2 MB.
            want = min(self.full_grid.x * self.full_grid.y, CORES_PER_USER_DECODE * batch)
            x = min(self.full_grid.x, want)
            y = max(1, min(self.full_grid.y, want // x))
            self._decode_cfgs[batch] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(x, y),
                q_chunk_size=0,
                k_chunk_size=32,
                exp_approx_mode=False,
            )
        return self._decode_cfgs[batch]

    # ---------------------------------------------------------------- shared stems
    def _q(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,4096] -> q [1,1,T,H*256] bf16 (q_a -> rmsnorm -> q_b)."""
        qa = ttnn.linear(x, self.w_qa, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        qa = ttnn.rms_norm(qa, epsilon=self.cfg.rms_norm_eps, weight=self.qa_norm, compute_kernel_config=self.hifi4)
        q = ttnn.linear(qa, self.w_qb, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        ttnn.deallocate(qa)
        return q

    def _latent(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,4096] -> c [1,1,T,512] bf16 (kv_a -> rmsnorm)."""
        c = ttnn.linear(x, self.w_kva, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        return ttnn.rms_norm(c, epsilon=self.cfg.rms_norm_eps, weight=self.kva_norm, compute_kernel_config=self.hifi4)

    # --------------------------------------------------------------------- prefill
    def prefill_forward(self, x: ttnn.Tensor, cache: TtLatentCache, user: int, seq_len: int) -> ttnn.Tensor:
        """Dense causal MLA for one user (context <= 2048 is exact). x [1,1,T_pad,4096] with
        T_pad % block_size == 0 and the logical ``seq_len`` <= T_pad; returns [1,1,T_pad,4096]
        (rows past seq_len are garbage). Fills the user's latent cache pages [0, T_pad)."""
        T = x.shape[-2]
        H, Dq, Dv = self.H, self.Dq, self.Dv
        q = self._q(x)  # [1,1,T,H*Dq]
        c = self._latent(x)  # [1,1,T,L]
        cache.fill(c, user)
        k = ttnn.linear(c, self.w_k_all, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,T,H*Dq]
        v = ttnn.linear(c, self.w_v_all, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,T,H*Dv]
        ttnn.deallocate(c)
        # [1,1,T,H*D] -> [1,H,T,D]
        qh = ttnn.transpose(ttnn.reshape(q, [1, T, H, Dq]), 1, 2)
        kh = ttnn.transpose(ttnn.reshape(k, [1, T, H, Dq]), 1, 2)
        vh = ttnn.transpose(ttnn.reshape(v, [1, T, H, Dv]), 1, 2)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        o = ttnn.transformer.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            is_causal=True,
            scale=self.scale,
            program_config=self.sdpa_prefill_cfg,
            compute_kernel_config=self.sdpa_compute,
        )  # [1,H,T,Dv]
        ttnn.deallocate(qh)
        ttnn.deallocate(kh)
        ttnn.deallocate(vh)
        o = ttnn.reshape(ttnn.transpose(o, 1, 2), [1, 1, T, H * Dv])
        return ttnn.linear(o, self.w_o, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)

    # ---------------------------------------------------------------------- decode
    def decode_forward(self, x: ttnn.Tensor, cache: TtLatentCache, positions: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,B,4096] (one token per user), positions [B] int32 row-major (the token's
        position = current context length). Appends the latent and attends over [0, pos]."""
        B = x.shape[-2]
        H, Dq, Dv, L = self.H, self.Dq, self.Dv, self.L
        q = self._q(x)  # [1,1,B,H*Dq]
        c = self._latent(x)  # [1,1,B,L]
        # cache update wants [1, B, 1(32), L] height-sharded on B cores
        c_rows = ttnn.reshape(c, [1, B, 1, L])
        grid = self.device.compute_with_storage_grid_size()
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(TILE, L),
            core_grid=ttnn.num_cores_to_corerangeset(B, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        c_sh = ttnn.to_memory_config(c_rows, shard_cfg)
        cache.update(c_sh, positions)
        ttnn.deallocate(c_sh)
        ttnn.deallocate(c)
        # absorbed q: [1,H,B,Dq] @ [1,H,Dq,L] -> [1,H,B,L]
        qh = ttnn.transpose(ttnn.reshape(q, [1, B, H, Dq]), 1, 2)
        q_lat = ttnn.matmul(qh, self.w_absorb_q, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        ttnn.deallocate(qh)
        q_dec = ttnn.transpose(q_lat, 1, 2)  # [1,B,H,L] (decode layout: batch, padded heads, dim)
        ttnn.deallocate(q_lat)
        o_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_dec,
            cache.cache,
            head_dim_v=L,
            page_table_tensor=cache.page_table,
            cur_pos_tensor=positions,
            scale=self.scale,
            program_config=self._decode_sdpa_config(B),
            compute_kernel_config=self.sdpa_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,B,H,L]
        ttnn.deallocate(q_dec)
        oh = ttnn.transpose(o_lat, 1, 2)  # [1,H,B,L]
        o = ttnn.matmul(oh, self.w_absorb_o, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,H,B,Dv]
        ttnn.deallocate(oh)
        o = ttnn.reshape(ttnn.transpose(o, 1, 2), [1, 1, B, H * Dv])
        return ttnn.linear(o, self.w_o, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
