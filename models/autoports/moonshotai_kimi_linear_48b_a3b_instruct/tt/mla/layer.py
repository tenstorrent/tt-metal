# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NoPE multi-head latent attention (7 of 27 layers) with an absorbed, paged latent KV cache on a 1xN TP mesh.

Cache per layer: [num_blocks, 1, block_size, 576] (512 normalised latent + 64 un-rotated "rope" dims), REPLICATED on
every chip (kv_a_proj is replicated, so each chip computes the identical latent). Heads are TP-sharded (32 / tp).
Decode: q_nope @ Wk_b (absorb) -> latent q [1,B,H_loc,576] -> paged_flash_multi_latent_attention_decode -> @ Wv_b^T -> o_proj -> all-reduce.
Prefill: same absorption with flash_mla_prefill over the (padded) chunk after paged_fill_cache of the valid rows;
long prompts use chunked_flash_mla_prefill against the cache filled so far.
"""

from __future__ import annotations

from pathlib import Path

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.ccl import KimiCCL
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.weights import as_device_tensor, linear_weight, tp_of


class KimiMLA:
    def __init__(
        self,
        mesh_device,
        cfg: KimiLinearConfig,
        sd: dict | None,
        *,
        layer_idx: int,
        ccl: KimiCCL,
        cache_path: Path | None,
        dtype=ttnn.bfloat8_b,
        block_size: int = 64,
    ):
        self.mesh_device, self.cfg, self.ccl = mesh_device, cfg, ccl
        self.tp = tp_of(mesh_device)
        self.H = cfg.num_attention_heads
        assert self.H % self.tp == 0
        self.H_loc = self.H // self.tp
        self.dn, self.dr, self.dv, self.R = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        self.dq = self.dn + self.dr  # 192
        self.L = self.R + self.dr  # 576 cached width
        self.scale = self.dq**-0.5
        self.block_size = block_size
        name = f"layer_{layer_idx}.mla"
        g = (lambda k: None) if sd is None else (lambda k: sd[k])
        kw = dict(cache_path=cache_path)
        # q_proj [2304, 32*192] column-sharded by heads (head-major columns)
        self.wq = as_device_tensor(
            mesh_device,
            None if sd is None else linear_weight(g("q_proj.weight")),
            name=f"{name}.wq",
            dtype=dtype,
            shard_dim=-1,
            **kw,
        )
        self.wkv_a = as_device_tensor(
            mesh_device,
            None if sd is None else linear_weight(g("kv_a_proj_with_mqa.weight")),
            name=f"{name}.wkv_a",
            dtype=ttnn.bfloat16,
            shard_dim=None,
            **kw,
        )
        self.kv_norm = as_device_tensor(
            mesh_device,
            None if sd is None else g("kv_a_layernorm.weight").reshape(1, 1, 1, self.R),
            name=f"{name}.kv_norm",
            dtype=ttnn.bfloat16,
            shard_dim=None,
            **kw,
        )
        if sd is not None:
            wkv_b = g("kv_b_proj.weight").reshape(self.H, self.dn + self.dv, self.R)  # [H, 256, 512]
            wk_b = (
                wkv_b[:, : self.dn, :].unsqueeze(0).contiguous()
            )  # [1, H, 128(dn), 512(R)]: q_lat = q_nope @ Wk (k_nope = latent @ Wk^T)
            wv_b = wkv_b[:, self.dn :, :].unsqueeze(0).contiguous()  # [1, H, 128, 512]; used as attn_lat @ wv_b^T
            wv_bT = wv_b.transpose(-2, -1).contiguous()  # [1, H, 512, 128]
        else:
            wk_b = wv_bT = None
        self.wk_b = as_device_tensor(mesh_device, wk_b, name=f"{name}.wk_b_v2", dtype=ttnn.bfloat16, shard_dim=1, **kw)
        self.wv_bT = as_device_tensor(mesh_device, wv_bT, name=f"{name}.wv_bT", dtype=ttnn.bfloat16, shard_dim=1, **kw)
        # o_proj [32*128, 2304] row-sharded by heads
        self.wo = as_device_tensor(
            mesh_device,
            None if sd is None else linear_weight(g("o_proj.weight")),
            name=f"{name}.wo",
            dtype=dtype,
            shard_dim=-2,
            **kw,
        )
        arch = mesh_device.arch()
        self.compute = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.compute_hifi4 = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.sdpa_compute = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    # ---- cache -------------------------------------------------------------------------------
    def allocate_cache(self, num_blocks: int, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        return ttnn.zeros(
            (num_blocks, 1, self.block_size, self.L),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ---- shared pieces -----------------------------------------------------------------------
    def _latent(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,S,hidden] -> kvpe [1,1,S,576] = rmsnorm(latent) ++ k_pe (bf16)."""
        kv = ttnn.linear(
            x,
            self.wkv_a,
            compute_kernel_config=self.compute_hifi4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        S = kv.shape[2]
        lat = ttnn.slice(kv, (0, 0, 0, 0), (1, 1, S, self.R))
        pe = (
            ttnn.slice(kv, (0, 0, self.R), (1, 1, S, self.L))
            if False
            else ttnn.slice(kv, (0, 0, 0, self.R), (1, 1, S, self.L))
        )
        ttnn.deallocate(kv)
        lat = ttnn.rms_norm(lat, epsilon=self.cfg.rms_norm_eps, weight=self.kv_norm)
        out = ttnn.concat([lat, pe], dim=-1)
        ttnn.deallocate(lat)
        ttnn.deallocate(pe)
        return out

    def _q_latent(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,S,hidden] -> absorbed q [1, H_loc, S, 576] (head-major)."""
        S = x.shape[2]
        q = ttnn.linear(
            x, self.wq, compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )  # [1,1,S,H_loc*192]
        q = ttnn.reshape(q, (1, S, self.H_loc, self.dq))
        q = ttnn.permute(q, (0, 2, 1, 3))  # [1, H_loc, S, 192]
        q_nope = ttnn.slice(q, (0, 0, 0, 0), (1, self.H_loc, S, self.dn))
        q_pe = ttnn.slice(q, (0, 0, 0, self.dn), (1, self.H_loc, S, self.dq))
        ttnn.deallocate(q)
        q_lat = ttnn.matmul(
            q_nope, self.wk_b, compute_kernel_config=self.compute_hifi4, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1,H_loc,S,512]
        ttnn.deallocate(q_nope)
        out = ttnn.concat([q_lat, q_pe], dim=-1)  # [1, H_loc, S, 576]
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_pe)
        return out

    def _out_proj(self, attn_hsd: ttnn.Tensor) -> ttnn.Tensor:
        """attn [1, H_loc, S, 512] -> replicated [1,1,S,hidden]."""
        S = attn_hsd.shape[2]
        v = ttnn.matmul(
            attn_hsd, self.wv_bT, compute_kernel_config=self.compute_hifi4, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1,H_loc,S,128]
        v = ttnn.permute(v, (0, 2, 1, 3))  # [1, S, H_loc, 128]
        v = ttnn.reshape(v, (1, 1, S, self.H_loc * self.dv))
        out = ttnn.linear(
            v, self.wo, compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )
        ttnn.deallocate(v)
        return self.ccl.all_reduce(out)

    # ---- prefill -----------------------------------------------------------------------------
    def forward_prefill(
        self,
        x: ttnn.Tensor,
        cache: ttnn.Tensor,
        page_table: ttnn.Tensor,
        *,
        user_id: int = 0,
        valid_len: int | None = None,
        chunk_start: int = 0,
    ) -> ttnn.Tensor:
        """x [1,1,T,hidden] (T % 32 == 0) for positions [chunk_start, chunk_start+T); fills ``cache`` rows < valid_len via
        ``page_table`` [1|B, max_blocks] int32 (row ``user_id``) and returns [1,1,T,hidden]."""
        T = x.shape[2]
        valid_len = T if valid_len is None else valid_len
        kvpe = self._latent(x)  # [1,1,T,576]
        fill = kvpe if valid_len == T else ttnn.slice(kvpe, (0, 0, 0, 0), (1, 1, valid_len, self.L))
        fill_c = fill if fill.dtype == cache.dtype else ttnn.typecast(fill, cache.dtype)
        pt_row = (
            page_table
            if page_table.shape[0] == 1
            else ttnn.slice(page_table, (user_id, 0), (user_id + 1, page_table.shape[1]))
        )
        if chunk_start:
            # fill only the blocks of this chunk: shift the page table by chunk_start / block_size
            b0 = chunk_start // self.block_size
            pt_row = ttnn.slice(pt_row, (0, b0), (1, pt_row.shape[1]))
        ttnn.experimental.paged_fill_cache(cache, fill_c, pt_row, batch_idx=0)
        q = self._q_latent(x)  # [1,H_loc,T,576]
        if chunk_start == 0:
            attn = ttnn.transformer.flash_mla_prefill(
                q,
                kvpe,
                head_dim_v=self.R,
                is_causal=True,
                scale=self.scale,
                compute_kernel_config=self.sdpa_compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attn = ttnn.transformer.chunked_flash_mla_prefill(
                q,
                cache,
                self.R,
                page_table if page_table.shape[0] == 1 else pt_row,
                chunk_start_idx=chunk_start,
                scale=self.scale,
                compute_kernel_config=self.sdpa_compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ttnn.deallocate(q)
        ttnn.deallocate(kvpe)
        return self._out_proj(attn)

    # ---- decode ------------------------------------------------------------------------------
    def forward_decode(
        self, x: ttnn.Tensor, cache: ttnn.Tensor, page_table: ttnn.Tensor, cur_pos: ttnn.Tensor
    ) -> ttnn.Tensor:
        """x [1,1,B,hidden]; cur_pos int32 [B] (position being written/attended); page_table int32 [B, max_blocks]."""
        B = x.shape[2]
        kvpe = self._latent(x)  # [1,1,B,576]
        upd = ttnn.reshape(kvpe, (1, B, 1, self.L))  # [1, B, n_kv_heads=1, D]
        upd_c = upd if upd.dtype == cache.dtype else ttnn.typecast(upd, cache.dtype)
        ttnn.experimental.paged_update_cache(cache, upd_c, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.deallocate(kvpe)
        q = self._q_latent(x)  # [1, H_loc, B, 576]
        q = ttnn.permute(q, (0, 2, 1, 3))  # [1, B, H_loc, 576]
        attn = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q,
            cache,
            head_dim_v=self.R,
            page_table_tensor=page_table,
            cur_pos_tensor=cur_pos,
            scale=self.scale,
            compute_kernel_config=self.sdpa_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, B, H_loc, 512]
        ttnn.deallocate(q)
        attn = ttnn.permute(attn, (0, 2, 1, 3))  # [1, H_loc, B, 512]
        return self._out_proj(attn)
