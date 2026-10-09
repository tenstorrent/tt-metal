# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 ViT attention: bidirectional over one image, 16 heads x 72 dims (padded to 96), 2D rotary.

HF ``Qwen3_5VisionAttention`` (``modeling_qwen3_5.py`` 904-983): fused ``qkv`` Linear (bias) ->
16 heads -> neox rotary on all 72 dims (cos/sin from the (row, col) patch position) -> SDPA with
scale 72^-0.5 and no mask inside one image (``cu_seqlens`` = one segment per image) -> ``proj``.

TT layout: q/k/v heads are 96 wide (zero-padded weight columns, see weights.py) and q/k head dims are
rope-permuted, so one ``rotary_embedding_llama`` per tensor with cos 1 / sin 0 on the 24 padded dims
keeps them zero; q.k and the softmax are unchanged by the zero dims. The patch sequence is padded to
a bucket; ``cu_window_seqlens = [0, n, S]`` makes SDPA block-diagonal, so the n real patches never
attend to the S - n padded keys (and the padded rows only see each other). SDPA writes the heads
concatenated ([1, 1, S, 16 x 96]); ``proj`` has zero rows on the padded dims.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.pplx_decider_v1_27b.tt.optimizations import VisionOptimizations
from models.demos.pplx_decider_v1_27b.tt.vision.common import device_linear, resolve_linear, vision_linear
from models.demos.pplx_decider_v1_27b.tt.vision.config import PplxVisionArgs
from models.demos.pplx_decider_v1_27b.tt.vision.weights import VisionAttentionWeights


@dataclass
class VisionAttentionConfig:
    weights: VisionAttentionWeights
    args: PplxVisionArgs
    optimizations: VisionOptimizations
    mesh_device: object | None = None


def rope_transformation_matrix(device) -> ttnn.Tensor:
    """One 32x32 tile for ``rotary_embedding_llama``: x @ trans maps each adjacent pair (a, b) to (-b, a)."""
    trans = torch.zeros(1, 1, 32, 32)
    trans[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1.0
    trans[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1.0
    return ttnn.from_torch(
        trans, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


class VisionAttention(LightweightModule):
    def __init__(self, weights: VisionAttentionWeights, args: PplxVisionArgs, optimizations: VisionOptimizations):
        super().__init__()
        self.config = _resolve(VisionAttentionConfig(weights=weights, args=args, optimizations=optimizations))
        self._loaded = False

    @classmethod
    def from_config(cls, config: VisionAttentionConfig) -> "VisionAttention":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance.config = _resolve(config)
        instance._loaded = False
        return instance

    def load_device_weights(self) -> None:
        if self._loaded:
            return
        w = self.config.weights
        self.qkv_w, self.qkv_b = device_linear(w.qkv)
        self.proj_w, self.proj_b = device_linear(w.proj)
        self.trans_mat = rope_transformation_matrix(self.config.mesh_device)
        self._loaded = True

    def _rope(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.experimental.rotary_embedding_llama(
            x,
            cos,
            sin,
            self.trans_mat,
            is_decode_mode=False,
            compute_kernel_config=self.config.optimizations.rope_compute_kernel_cfg,
        )

    def forward(
        self, x: ttnn.Tensor, *, cos: ttnn.Tensor, sin: ttnn.Tensor, cu_window_seqlens: ttnn.Tensor
    ) -> ttnn.Tensor:
        """x: [1, 1, S, 1152] (norm1 output), S a vision bucket; cos/sin [1, 1, S, 96]; cu [0, n, S] int32."""
        self.load_device_weights()
        a, opts = self.config.args, self.config.optimizations
        qkv = vision_linear(x, self.qkv_w, self.qkv_b, "vision_qkv", opts)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=a.num_heads,
            num_kv_heads=a.num_heads,
            transpose_k_heads=False,
            memory_config=opts.output_memcfg,
        )
        ttnn.deallocate(qkv)
        q_rot, k_rot = self._rope(q, cos, sin), self._rope(k, cos, sin)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_rot,
            v,
            is_causal=False,
            scale=a.head_dim**-0.5,
            cu_window_seqlens=cu_window_seqlens,
            program_config=opts.sdpa_program_config(),
            compute_kernel_config=opts.sdpa_compute_kernel_cfg,
            memory_config=opts.output_memcfg,
            output_concat_heads=True,
        )
        for t in (q_rot, k_rot, v):
            ttnn.deallocate(t)
        out = vision_linear(attn, self.proj_w, self.proj_b, "vision_proj", opts)
        ttnn.deallocate(attn)
        return out


def _resolve(config: VisionAttentionConfig) -> VisionAttentionConfig:
    device = config.mesh_device or config.optimizations.mesh_device
    w = config.weights
    return replace(
        config,
        mesh_device=device,
        weights=VisionAttentionWeights(qkv=resolve_linear(w.qkv, device), proj=resolve_linear(w.proj, device)),
    )
