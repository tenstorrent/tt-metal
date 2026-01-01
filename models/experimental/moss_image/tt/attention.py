# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Grouped device attention with FP32 score arithmetic and BF16 probabilities.

This is the accurate MOSS inference path. It deliberately retains scores through
stable softmax in FP32, and FP32 PV accumulation, while preserving the reference
BF16 attention-output boundary. Head grouping bounds temporary score storage.
"""
from __future__ import annotations

import math

import torch


class AccurateAttention:
    """Caller-owned one/two-chip attention; no device opens or host activations.

    Inputs use [batch, heads, sequence, head_dim]. GQA assigns contiguous query
    heads to each KV head. No decode/KV-cache alignment is inferred: causal calls
    require equal query/key sequence lengths, matching full prompt encoding.
    """

    def __init__(self, device, compute_kernel_config, *, head_group_size=6):
        import ttnn

        if not isinstance(head_group_size, int) or head_group_size < 1:
            raise ValueError("head_group_size must be a positive integer")
        self.t = ttnn
        self.device = device
        self.compute = compute_kernel_config
        self.head_group_size = head_group_size
        self.mask_cache = {}
        self.causal_masks = {}

    def _mask(self, mask):
        """Retain the source tensor to prevent identity reuse in the local cache."""
        key = id(mask)
        if key not in self.mask_cache:
            if len(self.mask_cache) >= 2:
                self.mask_cache.clear()
            self.mask_cache[key] = (mask, self.t.typecast(mask, self.t.float32))
        return self.mask_cache[key][1]

    def _causal_mask(self, sequence):
        t = self.t
        if sequence not in self.causal_masks:
            if len(self.causal_masks) >= 2:
                self.causal_masks.clear()
            # This is position metadata, not a host attention/activation fallback.
            mask = torch.triu(torch.full((1, 1, sequence, sequence), -1e9), diagonal=1)
            self.causal_masks[sequence] = t.from_torch(
                mask,
                device=self.device,
                dtype=t.float32,
                layout=t.TILE_LAYOUT,
                memory_config=t.DRAM_MEMORY_CONFIG,
                mesh_mapper=(t.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None),
            )
        return self.causal_masks[sequence]

    def __call__(self, q, k, v, *, attn_mask=None, is_causal=False, scale=None):
        t = self.t
        if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4:
            raise ValueError("Attention inputs must have shape [batch, heads, sequence, head_dim]")
        batch, heads, queries, dim = q.shape
        kv_heads, keys = k.shape[1], k.shape[2]
        if (
            batch != 1
            or k.shape[0] != batch
            or v.shape[0] != batch
            or k.shape[-1] != dim
            or v.shape[1] != kv_heads
            or v.shape[2] != keys
            or heads % kv_heads
        ):
            raise ValueError("Incompatible MOSS attention/GQA shapes")
        if any(value.dtype != t.bfloat16 for value in (q, k, v)):
            raise ValueError("MOSS attention expects BF16 Q/K/V operands")
        if is_causal and queries != keys:
            raise ValueError("Causal prompt attention requires equal query/key sequence lengths")
        scale = dim**-0.5 if scale is None else scale
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Attention scale must be finite and positive")
        mask = None
        if attn_mask is not None:
            if (
                len(attn_mask.shape) != 4
                or tuple(attn_mask.shape)[-2:] != (queries, keys)
                or attn_mask.shape[0] != 1
                or attn_mask.shape[1] not in (1, heads)
            ):
                raise ValueError("Additive attention mask must broadcast over the query heads")
            mask = self._mask(attn_mask)
        if is_causal:
            causal = self._causal_mask(queries)
            mask = causal if mask is None else t.add(mask, causal)
        ratio = heads // kv_heads
        outputs = []
        # Equal-head attention can group neighboring heads. For GQA, keep each
        # group within a KV head and repeat that KV only for its query subgroup.
        groups = (
            [
                (start, min(heads, start + self.head_group_size), start, min(heads, start + self.head_group_size))
                for start in range(0, heads, self.head_group_size)
            ]
            if ratio == 1
            else [
                (start, min((kv + 1) * ratio, start + self.head_group_size), kv, kv + 1)
                for kv in range(kv_heads)
                for start in range(kv * ratio, (kv + 1) * ratio, self.head_group_size)
            ]
        )
        for first, last, kv_first, kv_last in groups:
            qh, kh, vh = q[:, first:last], k[:, kv_first:kv_last], v[:, kv_first:kv_last]
            if ratio > 1 and last - first > 1:
                kh = t.repeat_interleave(kh, last - first, dim=1)
                vh = t.repeat_interleave(vh, last - first, dim=1)
            scores = t.matmul(
                qh,
                t.transpose(kh, -2, -1),
                dtype=t.float32,
                compute_kernel_config=self.compute,
                memory_config=t.DRAM_MEMORY_CONFIG,
            )
            scaled = t.multiply(scores, scale)
            t.deallocate(scores)
            if mask is not None:
                selected_mask = mask if mask.shape[1] == 1 else mask[:, first:last]
                masked = t.add(scaled, selected_mask)
                t.deallocate(scaled)
                scaled = masked
            probabilities = t.softmax(scaled, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
            t.deallocate(scaled)
            rounded = t.typecast(probabilities, t.bfloat16)
            t.deallocate(probabilities)
            product = t.matmul(
                rounded, vh, dtype=t.float32, compute_kernel_config=self.compute, memory_config=t.DRAM_MEMORY_CONFIG
            )
            t.deallocate(rounded)
            outputs.append(t.typecast(product, t.bfloat16))
            t.deallocate(product)
        return outputs[0] if len(outputs) == 1 else t.concat(outputs, dim=1, memory_config=t.DRAM_MEMORY_CONFIG)
