# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 Attention with V-norm and decode workarounds for head_dim>256."""

import torch

import ttnn
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.common import Mode


def rms_norm_no_weights(x, eps=1e-6):
    if x.is_sharded():
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    x_fp32 = ttnn.typecast(x, ttnn.bfloat16) if x.dtype != ttnn.bfloat16 else x
    x_sq = ttnn.multiply(x_fp32, x_fp32)
    mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True)
    ttnn.deallocate(x_sq)
    rms = ttnn.sqrt(ttnn.add(mean_sq, eps))
    ttnn.deallocate(mean_sq)
    result = ttnn.multiply(x_fp32, ttnn.reciprocal(rms))
    ttnn.deallocate(rms)
    return result


class Gemma4Attention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_v_norm = True

    def _normalize_v(self, v_heads):
        if not self.apply_v_norm:
            return v_heads
        orig_mem = v_heads.memory_config()
        result = rms_norm_no_weights(v_heads, eps=self.args.norm_eps)
        if orig_mem != result.memory_config():
            result = ttnn.to_memory_config(result, orig_mem)
        return result

    def forward(self, x, current_pos, rot_mats=None, user_id=0, mode=Mode.DECODE,
                page_table=None, chunk_page_table=None, chunk_start_idx=None, kv_cache=None):
        # Save originals for ALL patchable ops
        orig_create_qkv = ttnn.experimental.nlp_create_qkv_heads
        orig_create_qkv_dec = ttnn.experimental.nlp_create_qkv_heads_decode
        orig_paged_update = ttnn.experimental.paged_update_cache
        orig_paged_fused = ttnn.experimental.paged_fused_update_cache
        orig_sdpa_decode = ttnn.transformer.scaled_dot_product_attention_decode
        orig_paged_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode
        attn_self = self

        # v_norm patches (always active)
        def patched_create_qkv(*a, **kw):
            q, k, v = orig_create_qkv(*a, **kw)
            v = attn_self._normalize_v(v)
            return q, k, v

        def patched_create_qkv_dec(*a, **kw):
            q, k, v = orig_create_qkv_dec(*a, **kw)
            v = attn_self._normalize_v(v)
            return q, k, v

        ttnn.experimental.nlp_create_qkv_heads = patched_create_qkv
        ttnn.experimental.nlp_create_qkv_heads_decode = patched_create_qkv_dec

        # head_dim>256 decode: no-op cache update + manual SDPA from prefilled cache
        if mode == Mode.DECODE and self.head_dim > 256:
            def noop_cache(cache, inp, update_idxs_tensor=None, page_table=None):
                pass  # Skip cache update (L1 clash with head_dim>256)
            def noop_fused(keys, k, values, v, update_idxs_tensor=None, page_table=None):
                pass

            # Cache for causal mask (shared across all full-attn layers in same decode step)
            _mask_cache = [None, None]  # [cur_pos, mask_dev]

            def manual_sdpa(q, keys, values, cur_pos_tensor=None, scale=None, **kw):
                """Manual SDPA for head_dim>256: Q @ K_cache^T -> softmax -> V_cache."""
                q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
                keys = ttnn.to_memory_config(keys, ttnn.DRAM_MEMORY_CONFIG)
                values = ttnn.to_memory_config(values, ttnn.DRAM_MEMORY_CONFIG)
                max_seq = keys.shape[2]
                keys_t = ttnn.transpose(keys, -2, -1)
                scores = ttnn.matmul(q, keys_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(keys_t)
                if scale and scale != 1.0:
                    scores = ttnn.multiply(scores, scale)
                # Causal mask (cached per decode step)
                pos_host = ttnn.to_torch(ttnn.get_device_tensors(cur_pos_tensor)[0]).flatten()
                cur_pos = int(pos_host[0].item())
                if _mask_cache[0] != cur_pos or _mask_cache[1] is None:
                    mask = torch.zeros(1, 1, 1, max_seq, dtype=torch.bfloat16)
                    mask[:, :, :, cur_pos + 1:] = -65504.0
                    _mask_cache[1] = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=attn_self.mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(attn_self.mesh_device))
                    _mask_cache[0] = cur_pos
                scores = ttnn.add(scores, _mask_cache[1])
                scores = ttnn.softmax(scores, dim=-1)
                attn_out = ttnn.matmul(scores, values, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(scores)
                return attn_out

            ttnn.experimental.paged_update_cache = noop_cache
            ttnn.experimental.paged_fused_update_cache = noop_fused
            # Manual SDPA only works for batch=1 (shape compatibility)
            if self.args.max_batch_size <= 1:
                ttnn.transformer.scaled_dot_product_attention_decode = manual_sdpa
                ttnn.transformer.paged_scaled_dot_product_attention_decode = manual_sdpa
            else:
                ttnn.transformer.scaled_dot_product_attention_decode = lambda q, *a, **kw: ttnn.clone(q)
                ttnn.transformer.paged_scaled_dot_product_attention_decode = lambda q, *a, **kw: ttnn.clone(q)

        try:
            return super().forward(x, current_pos, rot_mats, user_id, mode,
                                   page_table, chunk_page_table, chunk_start_idx, kv_cache)
        finally:
            ttnn.experimental.nlp_create_qkv_heads = orig_create_qkv
            ttnn.experimental.nlp_create_qkv_heads_decode = orig_create_qkv_dec
            ttnn.experimental.paged_update_cache = orig_paged_update
            ttnn.experimental.paged_fused_update_cache = orig_paged_fused
            ttnn.transformer.scaled_dot_product_attention_decode = orig_sdpa_decode
            ttnn.transformer.paged_scaled_dot_product_attention_decode = orig_paged_sdpa
