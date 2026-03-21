# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 GQA Attention with QK-norm and Partial RoPE — Galaxy mesh (8,4).

KV-cache strategy:
  - Paged attention (default): [max_num_blocks, NK, block_size, D] on DRAM per layer.
    Prefill: paged_fill_cache with page_table
    Decode:  paged_update_cache + paged_scaled_dot_product_attention_decode
  - Non-paged fallback: [B, NK, max_seq_len, D] on DRAM per layer.
    Prefill: fill_cache
    Decode:  update_cache + scaled_dot_product_attention_decode
  - Trace-safe decode via forward_decode_trace with tensor positions.
"""

import torch

import ttnn
from models.demos.gpt_oss.config import MeshConfig
from models.demos.gpt_oss.tt.attention.operations import apply_allreduce
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.tt_transformers.tt.common import PagedAttentionConfig

from .model_config import MiniMaxM2TTConfig
from .rms_norm import TtRMSNorm
from .rope import apply_partial_rope


def _extract_bsh(x: ttnn.Tensor, H: int):
    """Return (B, S, x_3d) from a 3D or 4D TTNN tensor using logical shape."""
    shape = list(x.shape)
    if len(shape) == 3:
        return shape[0], shape[1], x
    elif len(shape) == 4:
        B, S = shape[0] * shape[1], shape[2]
        return B, S, ttnn.reshape(x, (B, S, H))
    raise ValueError(f"Unexpected shape {shape}")


class TtMiniMaxAttention:
    """
    MiniMax-M2.5 attention with device-resident KV cache.

    TP parallelism (gpt_oss pattern):
      - Q/K/V projections: column-parallel (shard output heads across TP)
      - O projection:      row-parallel (shard input, all-reduce partial products)
      - QK-norm weights:   sharded to match local head count per device

    KV cache modes:
      - Paged: [max_num_blocks, NK_local, block_size, D] with page_table for mapping
      - Non-paged: [B, NK_local, max_seq_len, D] static allocation

    Prefill fills via paged_fill_cache/fill_cache; decode uses paged_update_cache/update_cache.
    Trace-safe decode via forward_decode_trace with tensor positions.
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        config: MiniMaxM2TTConfig,
        layer_idx: int,
        mesh_config: MeshConfig = None,
        ccl_manager: CCLManager = None,
        max_seq_len: int = 4096,
        max_batch_size: int = 1,
        paged_attention_config: PagedAttentionConfig = None,
        weight_cache_path=None,
    ):
        self.config = config
        self.device = device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self._is_mesh = isinstance(device, ttnn.MeshDevice)

        NQ = config.num_attention_heads
        NK = config.num_key_value_heads
        D = config.head_dim
        H = config.hidden_size
        eps = config.rms_norm_eps
        tp = mesh_config.tp if mesh_config else 1

        NQ_local = NQ // tp
        NK_local = NK // tp

        prefix = f"model.layers.{layer_idx}.self_attn."
        cache_prefix = weight_cache_path / f"layer{layer_idx}.self_attn" if weight_cache_path else None

        rep_mapper = ttnn.ReplicateTensorToMesh(device) if self._is_mesh else None
        col_mapper = mesh_config.column_parallel(device) if (self._is_mesh and mesh_config) else None
        row_mapper = mesh_config.row_parallel(device) if (self._is_mesh and mesh_config) else None

        def _load_col(key):
            w = state_dict[prefix + key].T.to(torch.bfloat16)
            cache_name = cache_prefix / key.replace(".", "_") if cache_prefix else None
            return ttnn.as_tensor(
                w,
                dtype=config.weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=col_mapper,
                cache_file_name=cache_name,
            )

        self.wq = _load_col("q_proj.weight")
        self.wk = _load_col("k_proj.weight")
        self.wv = _load_col("v_proj.weight")

        wo_pt = state_dict[prefix + "o_proj.weight"].T.to(torch.bfloat16)
        wo_cache = cache_prefix / "o_proj_weight" if cache_prefix else None
        self.wo = ttnn.as_tensor(
            wo_pt,
            dtype=config.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=row_mapper,
            cache_file_name=wo_cache,
        )

        # QK-norm: row_parallel shards dim=-2 of the [1,1,N/TILE,TILE] weight,
        # giving each TP device a weight matching its local head count.
        q_norm_cache = cache_prefix / "q_norm" if cache_prefix else None
        k_norm_cache = cache_prefix / "k_norm" if cache_prefix else None
        self.q_norm = TtRMSNorm(
            device, state_dict[prefix + "q_norm.weight"], eps, mesh_mapper=row_mapper, cache_path=q_norm_cache
        )
        self.k_norm = TtRMSNorm(
            device, state_dict[prefix + "k_norm.weight"], eps, mesh_mapper=row_mapper, cache_path=k_norm_cache
        )

        self._NQ = NQ
        self._NK = NK
        self._NQ_local = NQ_local
        self._NK_local = NK_local
        self._D = D
        self._tp = tp
        self._max_seq_len = max_seq_len
        self._max_batch_size = max_batch_size
        self._paged_attention_config = paged_attention_config
        self._use_paged_attention = paged_attention_config is not None

        # Device-resident KV cache
        if self._use_paged_attention:
            # Paged: [max_num_blocks, NK_local, block_size, D]
            cache_shape = [
                paged_attention_config.max_num_blocks,
                NK_local,
                paged_attention_config.block_size,
                D,
            ]
        else:
            # Non-paged: [B, NK_local, max_seq_len, D]
            cache_shape = [max_batch_size, NK_local, max_seq_len, D]

        self.k_cache = ttnn.from_torch(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep_mapper,
        )
        self.v_cache = ttnn.from_torch(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep_mapper,
        )

        NK_local_padded = max(32, ((NK_local + 31) // 32) * 32)
        self._decode_kv_shard_mem = ttnn.create_sharded_memory_config(
            shape=(NK_local_padded, D),
            core_grid=ttnn.CoreGrid(y=max_batch_size, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    def _qkv_rope(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor):
        cfg = self.config
        NQ_l, NK_l, D, H = self._NQ_local, self._NK_local, self._D, cfg.hidden_size

        B, S, x = _extract_bsh(x, H)

        q = ttnn.linear(x, self.wq, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.linear(x, self.wk, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.linear(x, self.wv, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = ttnn.permute(ttnn.reshape(q, (B, S, NQ_l, D)), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.reshape(k, (B, S, NK_l, D)), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.reshape(v, (B, S, NK_l, D)), (0, 2, 1, 3))

        q, k = apply_partial_rope(q, k, cos, sin, cfg.rotary_dim, D)
        return q, k, v, B, S

    def _o_proj_allreduce(self, attn_out, B, S):
        cfg = self.config
        tp = self._tp
        NQ_l, D, H = self._NQ_local, self._D, cfg.hidden_size

        attn_out = ttnn.reshape(ttnn.permute(attn_out, (0, 2, 1, 3)), (B, S, NQ_l * D))
        out = ttnn.linear(attn_out, self.wo, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out.deallocate(True)

        if self._is_mesh and self.mesh_config and self.ccl_manager and tp > 1:
            out_4d = ttnn.unsqueeze_to_4D(out)
            out_4d = apply_allreduce(out_4d, self.mesh_config, self.ccl_manager, H)
            out = ttnn.reshape(out_4d, (B, S, H))

        return out

    # ------------------------------------------------------------------
    # Forward: no KV-cache (unit tests)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
        is_causal: bool = True,
    ) -> ttnn.Tensor:
        scale = self._D**-0.5
        q, k, v, B, S = self._qkv_rope(x, cos, sin)

        if attention_mask is not None:
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                is_causal=False,
                scale=scale,
            )
        else:
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=is_causal,
                scale=scale,
            )

        return self._o_proj_allreduce(attn_out, B, S)

    # ------------------------------------------------------------------
    # Prefill: fill device KV cache
    # ------------------------------------------------------------------

    def forward_prefill(self, x, cos, sin, user_id: int = 0, page_table: ttnn.Tensor = None):
        """Prefill: process S tokens, fill device-resident KV cache.

        Args:
            x: Input hidden states [B, S, H]
            cos, sin: RoPE matrices
            user_id: Batch index for non-paged attention
            page_table: [B, max_blocks_per_user] page table for paged attention
        """
        scale = self._D**-0.5
        q, k, v, B, S = self._qkv_rope(x, cos, sin)

        if self._use_paged_attention and page_table is not None:
            # Paged attention: use paged_fill_cache
            ttnn.experimental.paged_fill_cache(self.k_cache, k, page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(self.v_cache, v, page_table, batch_idx=user_id)
        else:
            # Non-paged: use fill_cache
            ttnn.fill_cache(self.k_cache, k, batch_idx=user_id)
            ttnn.fill_cache(self.v_cache, v, batch_idx=user_id)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=scale,
        )

        return self._o_proj_allreduce(attn_out, B, S)

    # ------------------------------------------------------------------
    # Decode: non-trace (uses Python int cur_pos)
    # ------------------------------------------------------------------

    def forward_decode(self, x, cos, sin, cur_pos: int):
        """Decode: single token, device-resident KV cache.

        Uses ttnn.update_cache for in-place KV update (no host reads).
        NOT trace-safe — uses Python int slice bounds.
        """
        scale = self._D**-0.5
        NK_l, D = self._NK_local, self._D
        q, k, v, B, _ = self._qkv_rope(x, cos, sin)

        ttnn.update_cache(self.k_cache, k, cur_pos)
        ttnn.update_cache(self.v_cache, v, cur_pos)
        k.deallocate(True)
        v.deallocate(True)

        k_filled = ttnn.slice(self.k_cache, (0, 0, 0, 0), (B, NK_l, cur_pos + 1, D))
        v_filled = ttnn.slice(self.v_cache, (0, 0, 0, 0), (B, NK_l, cur_pos + 1, D))

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k_filled,
            v_filled,
            is_causal=False,
            scale=scale,
        )
        q.deallocate(True)
        k_filled.deallocate(True)
        v_filled.deallocate(True)

        return self._o_proj_allreduce(attn_out, B, 1)

    # ------------------------------------------------------------------
    # Decode: trace-safe (uses tensor position_idx)
    # ------------------------------------------------------------------

    def forward_decode_trace(self, x, cos, sin, position_idx: ttnn.Tensor, page_table: ttnn.Tensor = None):
        """Trace-safe decode: uses paged_update_cache + SDPA decode with tensor positions.

        Args:
            x:            [B, 1, H] hidden states
            cos, sin:     [1, 1, B, rotary_dim] from RoPE embedding lookup
            position_idx: [B] int32 tensor with current decode position
            page_table:   [B, max_blocks_per_user] page table for paged attention (optional)

        All shapes are fixed → safe for Metal trace capture/replay.
        """
        scale = self._D**-0.5
        q, k, v, B, _ = self._qkv_rope(x, cos, sin)
        # q: [B, NQ, 1, D], k: [B, NK, 1, D], v: [B, NK, 1, D]

        # Convert K/V to decode format [1, B, NK, D] for paged_update_cache
        k_dec = ttnn.permute(k, (2, 0, 1, 3))  # [1, B, NK, D]
        v_dec = ttnn.permute(v, (2, 0, 1, 3))  # [1, B, NK, D]
        k.deallocate(True)
        v.deallocate(True)

        # HEIGHT_SHARD K/V for paged_update_cache requirement
        k_sharded = ttnn.to_memory_config(k_dec, self._decode_kv_shard_mem)
        v_sharded = ttnn.to_memory_config(v_dec, self._decode_kv_shard_mem)
        k_dec.deallocate(True)
        v_dec.deallocate(True)

        # Update cache with tensor position index (trace-safe)
        ttnn.experimental.paged_update_cache(
            self.k_cache, k_sharded, update_idxs_tensor=position_idx, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            self.v_cache, v_sharded, update_idxs_tensor=position_idx, page_table=page_table
        )
        k_sharded.deallocate(True)
        v_sharded.deallocate(True)

        # SDPA decode: Q=[1, B, NQ, D], K=[B, NK, S, D], V=[B, NK, S, D]
        q_dec = ttnn.permute(q, (2, 0, 1, 3))  # [1, B, NQ, D]
        q.deallocate(True)

        if self._use_paged_attention and page_table is not None:
            # Paged SDPA decode
            attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_dec,
                self.k_cache,
                self.v_cache,
                cur_pos_tensor=position_idx,
                page_table_tensor=page_table,
                scale=scale,
            )
        else:
            # Non-paged SDPA decode
            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q_dec,
                self.k_cache,
                self.v_cache,
                cur_pos_tensor=position_idx,
                scale=scale,
            )
        q_dec.deallocate(True)

        # attn_out: [1, B, NQ, D] → [B, NQ, 1, D] for O-proj
        attn_out = ttnn.permute(attn_out, (1, 2, 0, 3))

        return self._o_proj_allreduce(attn_out, B, 1)

    def clear_cache(self):
        """Zero the KV cache in-place on device."""
        ttnn.mul(self.k_cache, 0, output_tensor=self.k_cache)
        ttnn.mul(self.v_cache, 0, output_tensor=self.v_cache)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
