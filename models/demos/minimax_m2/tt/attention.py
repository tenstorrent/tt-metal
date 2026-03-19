# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 GQA Attention with QK-norm and Partial RoPE — Galaxy mesh (8,4).

KV-cache strategy:
  - Device-resident: pre-allocated [B, NK, max_seq_len, D] on DRAM per layer.
  - Prefill:  ttnn.fill_cache(k_cache, k, batch_idx=user_id)
  - Decode:   ttnn.update_cache / paged_update_cache with tensor index
  - Trace-safe decode via forward_decode_trace: uses paged_update_cache and
    scaled_dot_product_attention_decode with tensor positions.
"""

import torch

import ttnn
from models.demos.gpt_oss.config import MeshConfig
from models.demos.gpt_oss.tt.attention.operations import apply_allreduce
from models.demos.gpt_oss.tt.ccl import CCLManager

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

    KV cache: [B, NK, max_seq_len, D] bfloat16 on DRAM, allocated at init.
    Prefill fills via ttnn.fill_cache; decode updates via ttnn.update_cache.
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

        prefix = f"model.layers.{layer_idx}.self_attn."

        rep_mapper = ttnn.ReplicateTensorToMesh(device) if self._is_mesh else None
        col_mapper = mesh_config.column_parallel(device) if (self._is_mesh and mesh_config) else None

        def _load_rep(key):
            w = state_dict[prefix + key].T.to(torch.bfloat16)
            return ttnn.from_torch(
                w,
                dtype=config.weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep_mapper,
            )

        self.wq = _load_rep("q_proj.weight")
        self.wk = _load_rep("k_proj.weight")
        self.wv = _load_rep("v_proj.weight")

        wo_pt = state_dict[prefix + "o_proj.weight"].T.to(torch.bfloat16)
        self.wo = ttnn.from_torch(
            wo_pt,
            dtype=config.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=col_mapper,
        )

        self.q_norm = TtRMSNorm(device, state_dict[prefix + "q_norm.weight"], eps, mesh_mapper=rep_mapper)
        self.k_norm = TtRMSNorm(device, state_dict[prefix + "k_norm.weight"], eps, mesh_mapper=rep_mapper)

        self._NQ = NQ
        self._NK = NK
        self._D = D
        self._tp = tp
        self._max_seq_len = max_seq_len
        self._max_batch_size = max_batch_size

        # Device-resident KV cache [B, NK, max_seq_len, D]
        cache_shape = [max_batch_size, NK, max_seq_len, D]
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

        # Pre-compute HEIGHT_SHARDED mem config for decode KV
        NK_padded = max(32, ((NK + 31) // 32) * 32)
        grid_end_y = max_batch_size - 1
        self._decode_kv_shard_mem = ttnn.create_sharded_memory_config(
            shape=(NK_padded, D),
            core_grid=ttnn.CoreGrid(y=max_batch_size, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    def _qkv_rope(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor):
        cfg = self.config
        NQ, NK, D, H = self._NQ, self._NK, self._D, cfg.hidden_size

        B, S, x = _extract_bsh(x, H)

        q = ttnn.linear(x, self.wq, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.linear(x, self.wk, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.linear(x, self.wv, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = ttnn.permute(ttnn.reshape(q, (B, S, NQ, D)), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.reshape(k, (B, S, NK, D)), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.reshape(v, (B, S, NK, D)), (0, 2, 1, 3))

        q, k = apply_partial_rope(q, k, cos, sin, cfg.rotary_dim, D)
        return q, k, v, B, S

    def _o_proj_allreduce(self, attn_out, B, S):
        cfg = self.config
        tp = self._tp
        NQ, D, H = self._NQ, self._D, cfg.hidden_size

        attn_out = ttnn.reshape(ttnn.permute(attn_out, (0, 2, 1, 3)), (B, S, NQ * D))
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

    def forward_prefill(self, x, cos, sin, user_id: int = 0):
        """Prefill: process S tokens, fill device-resident KV cache."""
        scale = self._D**-0.5
        q, k, v, B, S = self._qkv_rope(x, cos, sin)

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
        NK, D = self._NK, self._D
        q, k, v, B, _ = self._qkv_rope(x, cos, sin)

        ttnn.update_cache(self.k_cache, k, cur_pos)
        ttnn.update_cache(self.v_cache, v, cur_pos)
        k.deallocate(True)
        v.deallocate(True)

        k_filled = ttnn.slice(self.k_cache, (0, 0, 0, 0), (B, NK, cur_pos + 1, D))
        v_filled = ttnn.slice(self.v_cache, (0, 0, 0, 0), (B, NK, cur_pos + 1, D))

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

    def forward_decode_trace(self, x, cos, sin, position_idx: ttnn.Tensor):
        """Trace-safe decode: uses paged_update_cache + SDPA decode with tensor positions.

        Args:
            x:            [B, 1, H] hidden states
            cos, sin:     [1, 1, B, rotary_dim] from RoPE embedding lookup
            position_idx: [B] int32 tensor with current decode position

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
        ttnn.experimental.paged_update_cache(self.k_cache, k_sharded, update_idxs_tensor=position_idx)
        ttnn.experimental.paged_update_cache(self.v_cache, v_sharded, update_idxs_tensor=position_idx)
        k_sharded.deallocate(True)
        v_sharded.deallocate(True)

        # SDPA decode: Q=[1, B, NQ, D], K=[B, NK, S, D], V=[B, NK, S, D]
        q_dec = ttnn.permute(q, (2, 0, 1, 3))  # [1, B, NQ, D]
        q.deallocate(True)

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
