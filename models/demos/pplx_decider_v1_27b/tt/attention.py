# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated full attention (Qwen3.5 ``self_attn``), causal prefill.

Per head: query and an output gate come from one q_proj; q/k get a zero-centred RMSNorm over
head_dim before partial RoPE (first 64 of 256 dims). GQA 24 query / 4 KV heads, scale 256^-0.5,
then ``attn * sigmoid(gate)`` and o_proj.

Prefill runs in bounded chunks. Each chunk writes its K/V into a request-local paged cache
(allocated once at setup, sized for ``max_seq_len``) and runs chunked causal SDPA over the
accumulated prefix, so any logical length up to ``max_seq_len`` is accepted. Stale cache rows
from earlier requests are never visible: every query only attends to keys at positions <= its own,
and those rows are always rewritten by the current request first.

Adapted from models/demos/qwen38_27b_qb2/tt/decoder.py (_qkv, _rope, _full_prefill), single device.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.pplx_decider_v1_27b.tt.common import prefill_linear, resolve
from models.demos.pplx_decider_v1_27b.tt.model_config import PplxDeciderArgs
from models.demos.pplx_decider_v1_27b.tt.optimizations import Optimizations
from models.demos.pplx_decider_v1_27b.tt.weight_adapter import AttentionWeights


@dataclass
class AttentionConfig:
    weights: AttentionWeights
    args: PplxDeciderArgs
    optimizations: Optimizations
    mesh_device: object | None = None


class PplxGatedAttention(LightweightModule):
    def __init__(self, weights: AttentionWeights, args: PplxDeciderArgs, optimizations: Optimizations):
        super().__init__()
        self.config = _resolve(AttentionConfig(weights=weights, args=args, optimizations=optimizations))
        self._setup()

    @classmethod
    def from_config(cls, config: AttentionConfig) -> "PplxGatedAttention":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance.config = _resolve(config)
        instance._setup()
        return instance

    def _setup(self) -> None:
        import torch

        c, opts = self.config, self.config.optimizations.attention
        self.page_size = opts.page_size
        self.num_pages = (c.args.max_seq_len + opts.page_size - 1) // opts.page_size
        shape = [self.num_pages, c.args.num_key_value_heads, opts.page_size, c.args.head_dim]
        self.key_cache, self.value_cache = [
            ttnn.zeros(
                shape,
                dtype=opts.kv_cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=c.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        ]
        # One request owns every page, in order (prefill-only model: no page sharing).
        self.page_table = ttnn.from_torch(
            torch.arange(self.num_pages, dtype=torch.int32).reshape(1, -1),
            device=c.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._loaded = False

    def load_device_weights(self) -> None:
        if self._loaded:
            return
        w = self.config.weights
        self.qkvg = w.qkvg.get_device_weight()
        self.o_proj = w.o_proj.get_device_weight()
        self.q_norm = w.q_norm.get_device_weight()
        self.k_norm = w.k_norm.get_device_weight()
        self._loaded = True

    # -- pieces -------------------------------------------------------------------------------
    def _head_norm(self, x, weight):
        return ttnn.rms_norm(
            x,
            weight=weight,
            epsilon=self.config.args.rms_norm_eps,
            compute_kernel_config=self.config.optimizations.norm.compute_kernel_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _rope(self, x, cos, sin):
        """Rotate the first ``rotary_dim`` dims of every head of x [1, H, t, 256]; keep the rest."""
        _, _, length, _ = x.shape
        width = cos.shape[-1]
        part = x[:, :, :, :width]
        rotated = ttnn.experimental.rotary_embedding(
            part, ttnn.reshape(cos, [1, 1, length, width]), ttnn.reshape(sin, [1, 1, length, width])
        )
        rotated = ttnn.reshape(rotated, part.shape, part.padded_shape)
        return ttnn.concat([rotated, x[:, :, :, width:]], dim=-1)

    def forward(self, x: ttnn.Tensor, *, start_pos: int, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """One prefill chunk. x: [1, t, 5120] (input-normed), positions start_pos..start_pos+t-1.

        ``start_pos`` must be a multiple of the page size; chunks of one request arrive in order.
        """
        self.load_device_weights()
        a, opts = self.config.args, self.config.optimizations
        b, t, _ = x.shape
        if b != 1:
            raise ValueError("Prefill attention takes one request at a time")
        if start_pos % self.page_size or start_pos + t > self.num_pages * self.page_size:
            raise ValueError(f"Chunk at {start_pos} (+{t}) is not page aligned or exceeds the cache")
        q_width, kv_width = a.num_attention_heads * a.head_dim, a.num_key_value_heads * a.head_dim

        packed = prefill_linear(x, self.qkvg, "attention_qkvg", opts.linear)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            packed[:, :, : q_width + 2 * kv_width],
            num_heads=a.num_attention_heads,
            num_kv_heads=a.num_key_value_heads,
            transpose_key=False,
        )
        gate = ttnn.reshape(packed[:, :, q_width + 2 * kv_width :], [b, t, q_width])
        q = self._rope(self._head_norm(q, self.q_norm), cos, sin)
        k = self._rope(self._head_norm(k, self.k_norm), cos, sin)

        # Cache write for this chunk's pages, then causal SDPA over the whole prefix.
        att = opts.attention
        first, last = start_pos // self.page_size, (start_pos + t + self.page_size - 1) // self.page_size
        chunk_table = self.page_table[:, first:last]
        for cache, update in ((self.key_cache, k), (self.value_cache, v)):
            if update.dtype != cache.dtype:
                update = ttnn.typecast(update, cache.dtype)
            ttnn.experimental.paged_fill_cache(cache, update, chunk_table, batch_idx=0)
        q_chunk = att.sdpa_q_chunk if t >= att.sdpa_q_chunk else 32
        k_chunk = att.sdpa_k_chunk
        capacity = self.num_pages * self.page_size
        while start_pos % q_chunk:
            q_chunk //= 2
        while start_pos % k_chunk or ((start_pos + t + k_chunk - 1) // k_chunk) * k_chunk > capacity:
            k_chunk //= 2
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            self.key_cache,
            self.value_cache,
            self.page_table,
            start_pos,
            scale=a.head_dim**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=att.sdpa_grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk
            ),
            compute_kernel_config=att.compute_kernel_cfg,
        )
        attn = ttnn.transformer.concatenate_heads(attn)
        gated = ttnn.mul(attn, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        return prefill_linear(gated, self.o_proj, "attention_out", opts.linear)


def _resolve(config: AttentionConfig) -> AttentionConfig:
    device = config.mesh_device or config.optimizations.mesh_device
    w = config.weights
    return replace(
        config,
        mesh_device=device,
        weights=AttentionWeights(
            qkvg=resolve(w.qkvg, device),
            o_proj=resolve(w.o_proj, device),
            q_norm=resolve(w.q_norm, device),
            k_norm=resolve(w.k_norm, device),
        ),
    )
