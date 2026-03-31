# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Optional

import torch

import ttnn

from .config import TILE_SIZE, align_to_tile, create_sharded_memory_config, get_shard_shape, get_shard_strategy
from .ops import (
    apply_dropout,
    base_context_cumsum,
    base_context_mean,
    compute_sparsity_float32,
    ensure_uint32_indices,
    linear,
    make_causal_mask_with_offset,
    make_padding_mask,
    make_sample_indices,
    make_sequence_mask,
    mask_invalid_queries_float32,
    pad_attention_mask,
    pad_last_dim_to_multiple,
    pad_qh_for_sdpa,
    pad_to_multiple,
    safe_softmax,
    select_topk_indices,
    slice_to_length,
)
from .state_io import to_float_tensor


class MultiHeadAttention:
    """
    Informer multi-head attention with optional ProbSparse query selection.

    The implementation supports two execution paths:
    - SDPA kernels (preferred): TTNN fused scaled-dot-product attention kernels.
    - Manual attention fallback: matmul + mask + softmax + matmul.

    SDPA is used when dtype/layout/mask constraints are satisfied, and manual
    fallback remains for unsupported combinations (notably FP32 compute).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        rng: torch.Generator,
        *,
        device,
        dtype: ttnn.DataType,
        prob_sparse: bool = False,
        factor: int = 5,
        mask_value: float = -1e4,
        use_sdpa: bool = False,
        random_sampling: bool = False,
        is_decoder: bool = False,
        compute_dtype: Optional[ttnn.DataType] = None,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        use_sharded: bool = False,
        shard_strategy: str = "height",
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.prob_sparse = prob_sparse
        self.factor = factor
        self.mask_value = mask_value
        self.use_sdpa = use_sdpa
        self.random_sampling = random_sampling
        self.is_decoder = is_decoder
        self.dtype = dtype
        self.compute_dtype = compute_dtype or dtype
        self.memory_config = memory_config
        self.use_sharded = use_sharded
        self.shard_strategy = get_shard_strategy(shard_strategy) if use_sharded else None
        self.device = device
        self.weight_dtype = self.compute_dtype if self.compute_dtype == ttnn.float32 else dtype
        self.core_grid = None
        self.interleaved_mem_config = None
        if self.use_sharded:
            grid_size = device.compute_with_storage_grid_size()
            self.core_grid = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)
            self.interleaved_mem_config = ttnn.L1_MEMORY_CONFIG if memory_config else ttnn.DRAM_MEMORY_CONFIG
        self.last_attention_stats: dict[str, float | int | bool | str] = {}

        self.q_weight_torch = torch.randn((d_model, d_model), generator=rng, dtype=torch.float32) * 0.02
        self.k_weight_torch = torch.randn((d_model, d_model), generator=rng, dtype=torch.float32) * 0.02
        self.v_weight_torch = torch.randn((d_model, d_model), generator=rng, dtype=torch.float32) * 0.02
        self.o_weight_torch = torch.randn((d_model, d_model), generator=rng, dtype=torch.float32) * 0.02
        self.q_bias_torch = torch.zeros((d_model,), dtype=torch.float32)
        self.k_bias_torch = torch.zeros((d_model,), dtype=torch.float32)
        self.v_bias_torch = torch.zeros((d_model,), dtype=torch.float32)
        self.o_bias_torch = torch.zeros((d_model,), dtype=torch.float32)

        self.q_weight = ttnn.from_torch(
            self.q_weight_torch, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT
        )
        self.k_weight = ttnn.from_torch(
            self.k_weight_torch, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT
        )
        self.v_weight = ttnn.from_torch(
            self.v_weight_torch, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT
        )
        self.o_weight = ttnn.from_torch(
            self.o_weight_torch, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT
        )
        self.q_bias = ttnn.from_torch(
            self.q_bias_torch, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT
        )
        self.k_bias = ttnn.from_torch(
            self.k_bias_torch, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT
        )
        self.v_bias = ttnn.from_torch(
            self.v_bias_torch, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT
        )
        self.o_bias = ttnn.from_torch(
            self.o_bias_torch, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT
        )

    def load_hf_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        mapping = (
            ("q_proj.weight", "q_weight"),
            ("q_proj.bias", "q_bias"),
            ("k_proj.weight", "k_weight"),
            ("k_proj.bias", "k_bias"),
            ("v_proj.weight", "v_weight"),
            ("v_proj.bias", "v_bias"),
            ("out_proj.weight", "o_weight"),
            ("out_proj.bias", "o_bias"),
        )
        used: set[str] = set()
        missing: list[str] = []
        for key, attr in mapping:
            tensor = state.get(key)
            if tensor is None:
                missing.append(key)
                continue
            used.add(key)
            value = to_float_tensor(tensor)
            setattr(self, f"{attr}_torch", value)
            ref = getattr(self, attr)
            setattr(
                self,
                attr,
                ttnn.from_torch(value, device=self.device, dtype=ref.dtype, layout=ttnn.TILE_LAYOUT),
            )
        unexpected = sorted(k for k in state if k not in used)
        if strict and missing:
            raise ValueError(f"Missing attention weights: {missing}")
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def load_ttnn_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        """Load TTNN-canonical attention weights owned by this module."""
        mapping = (
            ("q_weight", "q_weight"),
            ("q_bias", "q_bias"),
            ("k_weight", "k_weight"),
            ("k_bias", "k_bias"),
            ("v_weight", "v_weight"),
            ("v_bias", "v_bias"),
            ("o_weight", "o_weight"),
            ("o_bias", "o_bias"),
        )
        used: set[str] = set()
        missing: list[str] = []
        for key, attr in mapping:
            tensor = state.get(key)
            if tensor is None:
                missing.append(key)
                continue
            used.add(key)
            value = to_float_tensor(tensor)
            setattr(self, f"{attr}_torch", value)
            ref = getattr(self, attr)
            setattr(
                self,
                attr,
                ttnn.from_torch(value, device=self.device, dtype=ref.dtype, layout=ttnn.TILE_LAYOUT),
            )
        unexpected = sorted(k for k in state if k not in used)
        if strict and missing:
            raise ValueError(f"Missing attention weights: {missing}")
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    @staticmethod
    def _decode_cur_pos_list(batch: int, decode_cur_pos: int) -> list[int]:
        return [int(decode_cur_pos)] * int(batch)

    @staticmethod
    def _to_decode_q_layout(qh: ttnn.Tensor) -> ttnn.Tensor:
        # Convert [B, H, 1, D] -> [1, B, H, D] expected by SDPA decode kernels.
        qh_rm = ttnn.to_layout(qh, ttnn.ROW_MAJOR_LAYOUT)
        qh_rm = ttnn.transpose(qh_rm, 0, 2)
        qh_rm = ttnn.transpose(qh_rm, 1, 2)
        return ttnn.to_layout(qh_rm, ttnn.TILE_LAYOUT)

    @staticmethod
    def _from_decode_q_layout(qh_decode: ttnn.Tensor) -> ttnn.Tensor:
        # Convert SDPA decode output [1, B, H, D] back to [B, H, 1, D].
        qh_rm = ttnn.to_layout(qh_decode, ttnn.ROW_MAJOR_LAYOUT)
        qh_rm = ttnn.transpose(qh_rm, 1, 2)
        qh_rm = ttnn.transpose(qh_rm, 0, 2)
        return ttnn.to_layout(qh_rm, ttnn.TILE_LAYOUT)

    @staticmethod
    def sdpa_mask_matches(q: ttnn.Tensor, k: ttnn.Tensor, mask: ttnn.Tensor | None) -> bool:
        if mask is None:
            return True
        if len(mask.shape) < 4:
            return False
        return int(mask.shape[2]) == int(q.shape[2]) and int(mask.shape[3]) == int(k.shape[2])

    @staticmethod
    def _has_sdpa_prefill_kernel() -> bool:
        return hasattr(ttnn, "transformer") and hasattr(ttnn.transformer, "scaled_dot_product_attention")

    @staticmethod
    def _has_sdpa_decode_kernel() -> bool:
        return hasattr(ttnn, "transformer") and hasattr(ttnn.transformer, "scaled_dot_product_attention_decode")

    def can_use_sdpa_prefill(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        mask: ttnn.Tensor | None,
        *,
        compute_dtype: ttnn.DataType,
        use_sdpa: bool,
    ) -> bool:
        """Return whether prefill SDPA kernel is valid for current tensor/mask constraints."""
        # SDPA kernels do not support FP32 inputs; keep manual attention for FP32.
        if not use_sdpa or compute_dtype == ttnn.float32:
            return False
        if not self._has_sdpa_prefill_kernel():
            return False
        return self.sdpa_mask_matches(q, k, mask)

    def can_use_sdpa_decode(
        self,
        q: ttnn.Tensor,
        mask: ttnn.Tensor | None,
        *,
        compute_dtype: ttnn.DataType,
        use_sdpa: bool,
        decode_cur_pos: int | None,
    ) -> bool:
        """Return whether decode SDPA kernel is valid for the current decode call."""
        # Decode kernel is restricted to single-token decode without explicit mask.
        if decode_cur_pos is None or int(q.shape[2]) != 1 or mask is not None:
            return False
        if compute_dtype == ttnn.float32:
            return False
        if not use_sdpa or not self._has_sdpa_decode_kernel():
            return False
        return True

    def split_heads(self, x: ttnn.Tensor, *, batch: int, length: int) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch, length, self.n_heads, self.head_dim))
        x = ttnn.transpose(x, 1, 2)
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def merge_heads(self, x: ttnn.Tensor, *, batch: int, length: int) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.transpose(x, 1, 2)
        x = ttnn.reshape(x, (batch, length, self.d_model))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def project_qkv(
        self,
        query_in: ttnn.Tensor,
        key_in: ttnn.Tensor,
        value_in: ttnn.Tensor,
    ) -> tuple[int, int, int, int, int, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, int]:
        batch, q_len, _ = query_in.shape
        _, k_len, _ = key_in.shape

        query_rm = ttnn.to_layout(query_in, ttnn.ROW_MAJOR_LAYOUT)
        key_rm = ttnn.to_layout(key_in, ttnn.ROW_MAJOR_LAYOUT)
        value_rm = ttnn.to_layout(value_in, ttnn.ROW_MAJOR_LAYOUT)

        query_rm, _ = pad_to_multiple(query_rm, dim=1, multiple=TILE_SIZE, value=0.0)
        key_rm, _ = pad_to_multiple(key_rm, dim=1, multiple=TILE_SIZE, value=0.0)
        value_rm, _ = pad_to_multiple(value_rm, dim=1, multiple=TILE_SIZE, value=0.0)

        query_in = ttnn.to_layout(query_rm, ttnn.TILE_LAYOUT)
        key_in = ttnn.to_layout(key_rm, ttnn.TILE_LAYOUT)
        value_in = ttnn.to_layout(value_rm, ttnn.TILE_LAYOUT)
        q_len_pad = query_in.shape[1]
        k_len_pad = key_in.shape[1]

        q_memcfg = self.memory_config
        k_memcfg = self.memory_config
        v_memcfg = self.memory_config
        core_grid = None
        if self.use_sharded:
            q_memcfg = create_sharded_memory_config(
                get_shard_shape(query_in), device=self.device, strategy=self.shard_strategy
            )
            k_memcfg = create_sharded_memory_config(
                get_shard_shape(key_in), device=self.device, strategy=self.shard_strategy
            )
            v_memcfg = create_sharded_memory_config(
                get_shard_shape(value_in), device=self.device, strategy=self.shard_strategy
            )
            core_grid = self.core_grid
            query_in = ttnn.to_memory_config(query_in, q_memcfg)
            key_in = ttnn.to_memory_config(key_in, k_memcfg)
            value_in = ttnn.to_memory_config(value_in, v_memcfg)

        qk_dtype = self.compute_dtype if self.compute_dtype == ttnn.float32 else self.dtype
        v_dtype = self.dtype
        q = linear(query_in, self.q_weight, self.q_bias, dtype=qk_dtype, memory_config=q_memcfg, core_grid=core_grid)
        k = linear(key_in, self.k_weight, self.k_bias, dtype=qk_dtype, memory_config=k_memcfg, core_grid=core_grid)
        v = linear(value_in, self.v_weight, self.v_bias, dtype=v_dtype, memory_config=v_memcfg, core_grid=core_grid)
        if self.use_sharded:
            q = ttnn.to_memory_config(q, self.interleaved_mem_config)
            k = ttnn.to_memory_config(k, self.interleaved_mem_config)
            v = ttnn.to_memory_config(v, self.interleaved_mem_config)

        qh = self.split_heads(q, batch=batch, length=q.shape[1])
        kh = self.split_heads(k, batch=batch, length=k.shape[1])
        vh = self.split_heads(v, batch=batch, length=v.shape[1])
        head_dim_padded = align_to_tile(self.head_dim)
        if head_dim_padded != self.head_dim:
            qh, _ = pad_last_dim_to_multiple(qh, multiple=TILE_SIZE, value=0.0)
            kh, _ = pad_last_dim_to_multiple(kh, multiple=TILE_SIZE, value=0.0)
            vh, _ = pad_last_dim_to_multiple(vh, multiple=TILE_SIZE, value=0.0)

        return batch, q_len, k_len, q_len_pad, k_len_pad, qh, kh, vh, head_dim_padded

    def merge_attention_masks(
        self,
        mask: ttnn.Tensor | None,
        key_padding_mask: ttnn.Tensor,
        *,
        batch: int,
    ) -> ttnn.Tensor:
        if mask is not None:
            if mask.shape[0] != batch:
                mask = ttnn.repeat(mask, (batch // mask.shape[0], 1, 1, 1))
            if mask.shape[1] not in (1, self.n_heads):
                raise ValueError(
                    f"Attention mask head dimension {mask.shape[1]} does not match n_heads={self.n_heads}."
                )
            if mask.shape[1] != key_padding_mask.shape[1]:
                key_padding_mask = ttnn.repeat(key_padding_mask, (1, mask.shape[1], 1, 1))
            return mask + key_padding_mask
        return key_padding_mask

    def build_attention_mask(
        self,
        mask: ttnn.Tensor | None,
        *,
        batch: int,
        q_len: int,
        k_len: int,
        k_valid_len: int,
        device,
        causal_past_len: int | None = None,
    ) -> ttnn.Tensor:
        if mask is not None:
            mask = pad_attention_mask(mask, q_length=q_len, k_length=k_len, mask_value=self.mask_value)

        pad_heads = self.n_heads if self.prob_sparse else 1
        key_padding_mask = make_padding_mask(
            valid_length=k_valid_len,
            key_length=k_len,
            batch=batch,
            heads=pad_heads,
            device=device,
            dtype=self.dtype,
            mask_value=self.mask_value,
        )
        attn_mask = self.merge_attention_masks(mask, key_padding_mask, batch=batch)

        if causal_past_len is not None:
            causal_mask = make_causal_mask_with_offset(
                q_length=q_len,
                k_length=k_len,
                past_length=causal_past_len,
                batch=batch,
                heads=1 if self.prob_sparse else self.n_heads,
                device=device,
                dtype=self.dtype,
                mask_value=self.mask_value,
            )
            if attn_mask is not None:
                if attn_mask.shape[1] == 1 and causal_mask.shape[1] == self.n_heads:
                    attn_mask = ttnn.repeat(attn_mask, (1, self.n_heads, 1, 1))
                attn_mask = attn_mask + causal_mask
            else:
                attn_mask = causal_mask
        return attn_mask

    def get_prob_sparse_params(
        self,
        *,
        q_valid_len: int,
        k_valid_len: int,
        decode_cache_mode: bool,
    ) -> tuple[int, int]:
        if decode_cache_mode:
            top_u = min(q_valid_len, max(1, int(self.factor * math.log(max(2, q_valid_len)))))
            sample_k = min(k_valid_len, max(1, int(self.factor * math.log(max(2, k_valid_len)))))
        else:
            log_k = int(math.ceil(math.log1p(max(1, k_valid_len))))
            log_q = int(math.ceil(math.log1p(max(1, q_valid_len))))
            top_u = min(q_valid_len, max(1, int(self.factor * log_q)))
            sample_k = min(k_valid_len, max(1, int(self.factor * q_valid_len * log_k)))
        if self.is_decoder:
            top_u = q_valid_len
        return top_u, sample_k

    def sparse_selected_query_attention(
        self,
        qh_top: ttnn.Tensor,
        kh: ttnn.Tensor,
        vh: ttnn.Tensor,
        mask_top: ttnn.Tensor | None,
        *,
        kh_t: ttnn.Tensor,
        scale: float,
        compute_dtype: ttnn.DataType,
        use_sdpa: bool,
    ) -> tuple[ttnn.Tensor, bool]:
        if use_sdpa:
            qh_top_sdpa, mask_top_sdpa, q_len_orig = pad_qh_for_sdpa(qh_top, mask_top, mask_value=self.mask_value)
            if self.can_use_sdpa_prefill(
                qh_top_sdpa,
                kh,
                mask_top_sdpa,
                compute_dtype=compute_dtype,
                use_sdpa=use_sdpa,
            ):
                context_top = ttnn.transformer.scaled_dot_product_attention(
                    qh_top_sdpa,
                    kh,
                    vh,
                    attn_mask=mask_top_sdpa,
                    is_causal=False,
                    scale=scale,
                    memory_config=self.memory_config,
                )
                if qh_top_sdpa.shape[2] != q_len_orig:
                    context_top = slice_to_length(context_top, dim=2, length=q_len_orig)
                return context_top, True

        scores_top = ttnn.matmul(qh_top, kh_t, dtype=compute_dtype)
        scores_top = scores_top * scale
        if mask_top is not None:
            if mask_top.dtype != compute_dtype:
                mask_top = ttnn.typecast(mask_top, compute_dtype)
            scores_top = scores_top + mask_top
        attn_probs_top = safe_softmax(scores_top, dim=-1, numeric_stable=True)
        context_top = ttnn.matmul(attn_probs_top, vh, dtype=compute_dtype)
        return context_top, False

    def prob_sparse_attention(
        self,
        qh: ttnn.Tensor,
        kh: ttnn.Tensor,
        vh: ttnn.Tensor,
        attn_mask: ttnn.Tensor | None,
        *,
        q_valid_len: int,
        k_valid_len: int,
        q_len_pad: int,
        compute_dtype: ttnn.DataType,
        scale: float,
        use_sdpa: bool,
        decode_cache_mode: bool,
    ) -> tuple[ttnn.Tensor, int, int, bool]:
        if attn_mask is not None and attn_mask.shape[1] == 1 and self.n_heads > 1:
            attn_mask = ttnn.repeat(attn_mask, (1, self.n_heads, 1, 1))
        if attn_mask is not None and attn_mask.shape[2] == 1 and q_len_pad > 1:
            attn_mask = ttnn.repeat(attn_mask, (1, 1, q_len_pad, 1))

        top_u, sample_k = self.get_prob_sparse_params(
            q_valid_len=q_valid_len,
            k_valid_len=k_valid_len,
            decode_cache_mode=decode_cache_mode,
        )

        kh_t = ttnn.transpose(kh, -2, -1)
        sample_idx = make_sample_indices(
            k_valid_len, sample_k, device=qh.device(), random_sampling=self.random_sampling
        )
        sample_idx = ttnn.repeat(sample_idx, (qh.shape[0], self.n_heads, 1))
        sample_idx = ttnn.reshape(sample_idx, (qh.shape[0], self.n_heads, sample_k, 1))
        sample_idx = ttnn.repeat(sample_idx, (1, 1, 1, qh.shape[3]))
        sample_idx = ttnn.to_layout(sample_idx, ttnn.TILE_LAYOUT)
        sample_idx = ttnn.typecast(sample_idx, ttnn.uint32)
        kh_sample = ttnn.gather(kh, dim=2, index=sample_idx)
        kh_sample_t = ttnn.transpose(kh_sample, -2, -1)
        sample_scores = ttnn.matmul(qh, kh_sample_t, dtype=compute_dtype)
        sample_scores = sample_scores * scale

        max_scores = ttnn.max(sample_scores, 3)
        # Match HF: divide by full key length, not sampled key count.
        mean_scores = ttnn.sum(sample_scores, dim=3, scalar=1.0 / float(max(1, k_valid_len)))
        batch_heads = qh.shape[0] * self.n_heads
        if compute_dtype == ttnn.float32:
            sparsity_rm = compute_sparsity_float32(max_scores, mean_scores, device=qh.device())
            sparsity_rm = mask_invalid_queries_float32(
                sparsity_rm,
                q_valid_len=q_valid_len,
                mask_value=self.mask_value,
            )
        else:
            sparsity = max_scores - mean_scores
            q_mask = make_sequence_mask(q_len_pad, q_valid_len, device=qh.device(), dtype=ttnn.bfloat16)
            q_mask = ttnn.repeat(q_mask, (qh.shape[0], self.n_heads, 1, 1))
            q_mask = ttnn.reshape(q_mask, (qh.shape[0], self.n_heads, q_len_pad))
            q_mask_inv = ttnn.add(ttnn.mul(q_mask, -1.0), 1.0)
            sparsity = sparsity * q_mask + q_mask_inv * float(self.mask_value)
            sparsity_rm = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
            sparsity_rm = ttnn.reshape(sparsity_rm, (batch_heads, q_len_pad))

        sparsity_rm, _ = pad_to_multiple(
            sparsity_rm,
            dim=0,
            multiple=TILE_SIZE,
            value=self.mask_value,
        )
        sparsity_topk = ttnn.to_layout(sparsity_rm, ttnn.TILE_LAYOUT)
        topk_idx = select_topk_indices(
            sparsity_topk,
            k=top_u,
            dim=1,
            mask_value=self.mask_value,
            prefer_argmax=compute_dtype == ttnn.float32,
        )
        topk_idx = ttnn.slice(topk_idx, [0, 0], [batch_heads, top_u])
        topk_idx = ensure_uint32_indices(topk_idx, length=top_u)
        topk_idx = ttnn.reshape(topk_idx, (qh.shape[0], self.n_heads, top_u, 1))

        q_idx = ttnn.repeat(topk_idx, (1, 1, 1, qh.shape[3]))
        q_idx = ttnn.to_layout(q_idx, ttnn.TILE_LAYOUT)
        qh_top = ttnn.gather(qh, dim=2, index=q_idx)

        mask_top = None
        if attn_mask is not None:
            mask_idx = ttnn.repeat(topk_idx, (1, 1, 1, kh.shape[2]))
            mask_idx = ttnn.to_layout(mask_idx, ttnn.TILE_LAYOUT)
            mask_top = ttnn.gather(attn_mask, dim=2, index=mask_idx)

        context_top, used_sdpa = self.sparse_selected_query_attention(
            qh_top,
            kh,
            vh,
            mask_top,
            kh_t=kh_t,
            scale=scale,
            compute_dtype=compute_dtype,
            use_sdpa=use_sdpa,
        )

        if self.is_decoder:
            base_context = base_context_cumsum(vh)
        else:
            base_context = base_context_mean(vh, valid_length=k_valid_len)
        if base_context.shape[2] != q_len_pad:
            base_context = slice_to_length(base_context, dim=2, length=q_len_pad)
        if base_context.dtype != context_top.dtype:
            base_context = ttnn.typecast(base_context, context_top.dtype)
        scatter_idx = ttnn.repeat(topk_idx, (1, 1, 1, qh.shape[3]))
        if context_top.dtype == ttnn.float32:
            base_rm = ttnn.to_layout(base_context, ttnn.ROW_MAJOR_LAYOUT)
            src_rm = ttnn.to_layout(context_top, ttnn.ROW_MAJOR_LAYOUT)
            idx_rm = ttnn.to_layout(scatter_idx, ttnn.ROW_MAJOR_LAYOUT)
            context_rm = ttnn.scatter(base_rm, dim=2, index=idx_rm, src=src_rm)
            context = ttnn.to_layout(context_rm, ttnn.TILE_LAYOUT)
        else:
            scatter_idx = ttnn.to_layout(scatter_idx, ttnn.TILE_LAYOUT)
            context = ttnn.scatter(base_context, dim=2, index=scatter_idx, src=context_top)
        return context, top_u, sample_k, used_sdpa

    def full_attention(
        self,
        qh: ttnn.Tensor,
        kh: ttnn.Tensor,
        vh: ttnn.Tensor,
        attn_mask: ttnn.Tensor | None,
        *,
        compute_dtype: ttnn.DataType,
        scale: float,
        use_sdpa: bool,
        allow_fused_softmax: bool,
        decode_cur_pos: int | None = None,
    ) -> tuple[ttnn.Tensor, bool]:
        if self.can_use_sdpa_decode(
            qh,
            attn_mask,
            compute_dtype=compute_dtype,
            use_sdpa=use_sdpa,
            decode_cur_pos=decode_cur_pos,
        ):
            qh_decode = self._to_decode_q_layout(qh)
            context = ttnn.transformer.scaled_dot_product_attention_decode(
                qh_decode,
                kh,
                vh,
                is_causal=True,
                attn_mask=attn_mask,
                cur_pos=self._decode_cur_pos_list(qh.shape[0], decode_cur_pos),
                scale=scale,
                memory_config=self.memory_config,
            )
            return self._from_decode_q_layout(context), True

        if self.can_use_sdpa_prefill(qh, kh, attn_mask, compute_dtype=compute_dtype, use_sdpa=use_sdpa):
            context = ttnn.transformer.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                attn_mask=attn_mask,
                is_causal=False,
                scale=scale,
                memory_config=self.memory_config,
            )
            return context, True

        kh_t = ttnn.transpose(kh, -2, -1)
        if self.memory_config is None:
            attn_scores = ttnn.matmul(qh, kh_t, dtype=compute_dtype)
        else:
            attn_scores = ttnn.matmul(qh, kh_t, memory_config=self.memory_config, dtype=compute_dtype)
        scores_scaled = attn_scores * scale

        mask_for_scores = attn_mask
        use_fused_softmax = allow_fused_softmax and self.head_dim % TILE_SIZE == 0
        if mask_for_scores is not None and allow_fused_softmax:
            use_fused_softmax = use_fused_softmax and mask_for_scores.shape[1] == 1 and mask_for_scores.shape[2] <= 1
            if not use_fused_softmax and mask_for_scores.shape[1] == 1 and self.n_heads > 1:
                mask_for_scores = ttnn.repeat(mask_for_scores, (1, self.n_heads, 1, 1))
        if compute_dtype == ttnn.float32 or attn_scores.shape[-1] >= 4096:
            use_fused_softmax = False

        if use_fused_softmax:
            attn_probs = ttnn.transformer.attention_softmax_(
                attn_scores,
                attention_mask=mask_for_scores,
                head_size=self.head_dim,
            )
        else:
            scores_masked = scores_scaled
            if mask_for_scores is not None:
                if mask_for_scores.dtype != compute_dtype:
                    mask_for_scores = ttnn.typecast(mask_for_scores, compute_dtype)
                scores_masked = scores_scaled + mask_for_scores
            attn_probs = safe_softmax(scores_masked, dim=-1, numeric_stable=True)

        if self.memory_config is None:
            context = ttnn.matmul(attn_probs, vh, dtype=compute_dtype)
        else:
            context = ttnn.matmul(attn_probs, vh, memory_config=self.memory_config, dtype=compute_dtype)
        return context, False

    def _compute_attention_context(
        self,
        qh: ttnn.Tensor,
        kh: ttnn.Tensor,
        vh: ttnn.Tensor,
        attn_mask: ttnn.Tensor | None,
        *,
        q_valid_len: int,
        k_valid_len: int,
        q_len_pad: int,
        compute_dtype: ttnn.DataType,
        scale: float,
        use_sdpa: bool,
        decode_cache_mode: bool,
        allow_fused_softmax: bool,
        decode_cur_pos: int | None = None,
    ) -> tuple[ttnn.Tensor, int, int, bool]:
        if self.prob_sparse:
            context, top_u, sample_k, used_sdpa_kernel = self.prob_sparse_attention(
                qh,
                kh,
                vh,
                attn_mask,
                q_valid_len=q_valid_len,
                k_valid_len=k_valid_len,
                q_len_pad=q_len_pad,
                compute_dtype=compute_dtype,
                scale=scale,
                use_sdpa=use_sdpa,
                decode_cache_mode=decode_cache_mode,
            )
            return context, top_u, sample_k, used_sdpa_kernel

        top_u = q_valid_len
        sample_k = k_valid_len
        context, used_sdpa_kernel = self.full_attention(
            qh,
            kh,
            vh,
            attn_mask,
            compute_dtype=compute_dtype,
            scale=scale,
            use_sdpa=use_sdpa,
            allow_fused_softmax=allow_fused_softmax,
            decode_cur_pos=decode_cur_pos,
        )
        return context, top_u, sample_k, used_sdpa_kernel

    def project_attention_output(
        self,
        context: ttnn.Tensor,
        *,
        batch: int,
        q_len: int,
        projected_q_len: int,
        head_dim_padded: int,
    ) -> ttnn.Tensor:
        if head_dim_padded != self.head_dim:
            context = slice_to_length(context, dim=3, length=self.head_dim)
        context = self.merge_heads(context, batch=batch, length=projected_q_len)
        context = slice_to_length(context, dim=1, length=q_len)
        out_memcfg = self.memory_config
        core_grid = None
        if self.use_sharded:
            out_memcfg = create_sharded_memory_config(
                get_shard_shape(context), device=self.device, strategy=self.shard_strategy
            )
            core_grid = self.core_grid
            context = ttnn.to_memory_config(context, out_memcfg)
        out = linear(
            context, self.o_weight, self.o_bias, dtype=self.dtype, memory_config=out_memcfg, core_grid=core_grid
        )
        if self.use_sharded:
            out = ttnn.to_memory_config(out, self.interleaved_mem_config)
        out = apply_dropout(out, self.dropout)
        return out

    def __call__(
        self,
        query_in: ttnn.Tensor,
        key_in: ttnn.Tensor,
        value_in: ttnn.Tensor,
        mask: ttnn.Tensor | None,
        *,
        q_valid_len: int | None = None,
        k_valid_len: int | None = None,
    ) -> ttnn.Tensor:
        """Run attention for prefill/eager paths and return projected output states."""
        batch, q_len, _ = query_in.shape
        _, k_len, _ = key_in.shape
        if q_valid_len is None:
            q_valid_len = q_len
        if k_valid_len is None:
            k_valid_len = k_len
        compute_dtype = self.compute_dtype
        use_sdpa = self.use_sdpa and compute_dtype != ttnn.float32
        used_sdpa_kernel = False
        batch, q_len, _, q_len_pad, k_len_pad, qh, kh, vh, head_dim_padded = self.project_qkv(
            query_in, key_in, value_in
        )
        attn_mask = self.build_attention_mask(
            mask=mask,
            batch=batch,
            q_len=q_len_pad,
            k_len=k_len_pad,
            k_valid_len=k_valid_len,
            device=qh.device(),
        )

        scale = 1.0 / math.sqrt(self.head_dim)
        context, top_u, sample_k, used_sdpa_kernel = self._compute_attention_context(
            qh,
            kh,
            vh,
            attn_mask,
            q_valid_len=q_valid_len,
            k_valid_len=k_valid_len,
            q_len_pad=q_len_pad,
            compute_dtype=compute_dtype,
            scale=scale,
            use_sdpa=use_sdpa,
            decode_cache_mode=False,
            allow_fused_softmax=True,
        )

        self.last_attention_stats = {
            "mode": "prob_sparse" if self.prob_sparse else "full",
            "q_valid_len": int(q_valid_len),
            "k_valid_len": int(k_valid_len),
            "top_u": int(top_u),
            "sample_k": int(sample_k),
            "used_sdpa": bool(used_sdpa_kernel),
            "is_decoder": bool(self.is_decoder),
        }
        return self.project_attention_output(
            context,
            batch=batch,
            q_len=q_len,
            projected_q_len=q_len_pad,
            head_dim_padded=head_dim_padded,
        )

    def call_with_cache(
        self,
        query_in: ttnn.Tensor,
        key_in: ttnn.Tensor,
        value_in: ttnn.Tensor,
        mask: ttnn.Tensor | None,
        *,
        kv_cache: dict[str, ttnn.Tensor | int],
        q_valid_len: int | None = None,
        k_valid_len: int | None = None,
        is_causal: bool = False,
        cache_update: bool = True,
    ) -> tuple[ttnn.Tensor, dict[str, ttnn.Tensor | int]]:
        """Run decoder self-attention with KV cache update/reuse semantics."""
        if kv_cache is None:
            return (
                self(
                    query_in,
                    key_in,
                    value_in,
                    mask,
                    q_valid_len=q_valid_len,
                    k_valid_len=k_valid_len,
                ),
                {},
            )

        batch, q_len, _ = query_in.shape
        _, k_len, _ = key_in.shape
        if q_valid_len is None:
            q_valid_len = q_len
        if k_valid_len is None:
            k_valid_len = k_len
        compute_dtype = self.compute_dtype
        use_sdpa = self.use_sdpa and compute_dtype != ttnn.float32
        used_sdpa_kernel = False

        cache_k = kv_cache.get("k")
        cache_v = kv_cache.get("v")
        cache_valid_len = int(kv_cache.get("valid_len", 0))

        batch, q_len, _, q_len_pad, _, qh, kh, vh, head_dim_padded = self.project_qkv(query_in, key_in, value_in)

        if cache_k is not None and cache_v is not None:
            kh = ttnn.concat([cache_k, kh], dim=2)
            vh = ttnn.concat([cache_v, vh], dim=2)

        total_k_valid_len = cache_valid_len + k_valid_len
        k_len_full = kh.shape[2]

        attn_mask = self.build_attention_mask(
            mask=mask,
            batch=batch,
            q_len=q_len_pad,
            k_len=k_len_full,
            k_valid_len=total_k_valid_len,
            device=qh.device(),
            causal_past_len=cache_valid_len if is_causal else None,
        )

        scale = 1.0 / math.sqrt(self.head_dim)
        decode_cur_pos = cache_valid_len + max(0, q_valid_len - 1)
        context, top_u, sample_k, used_sdpa_kernel = self._compute_attention_context(
            qh,
            kh,
            vh,
            attn_mask,
            q_valid_len=q_valid_len,
            k_valid_len=total_k_valid_len,
            q_len_pad=q_len_pad,
            compute_dtype=compute_dtype,
            scale=scale,
            use_sdpa=use_sdpa,
            decode_cache_mode=True,
            allow_fused_softmax=False,
            decode_cur_pos=decode_cur_pos,
        )

        self.last_attention_stats = {
            "mode": "prob_sparse" if self.prob_sparse else "full",
            "q_valid_len": int(q_valid_len),
            "k_valid_len": int(total_k_valid_len),
            "top_u": int(top_u),
            "sample_k": int(sample_k),
            "used_sdpa": bool(used_sdpa_kernel),
            "is_decoder": bool(self.is_decoder),
            "cache_valid_len": int(total_k_valid_len),
        }

        out = self.project_attention_output(
            context,
            batch=batch,
            q_len=q_len,
            projected_q_len=q_len_pad,
            head_dim_padded=head_dim_padded,
        )

        if cache_update:
            kv_cache["k"] = kh
            kv_cache["v"] = vh
            kv_cache["valid_len"] = total_k_valid_len
        return out, kv_cache
