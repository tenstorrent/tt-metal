# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Attention: `MoonViTEncoderLayer.attention_qkvpacked` in the reference.

wqkv -> split into 12 heads of 128 for q, k, v (`nlp_create_qkv_heads`) -> RoPE on q and k
(`rotary_embedding_llama`, see `rope.py`) -> non-causal ring attention over the sequence split
along `sp_axis` (`ring_joint_scaled_dot_product_attention`, `logical_n` masking the padding) ->
merge heads (`nlp_concat_heads`) -> wo. Scale is `head_dim ** -0.5`.
"""

from __future__ import annotations

import ttnn

from .linear import Linear
from .model_config import MoonViTConfig, VisionParallel
from .module import Module
from .rope import RopeTensors


class TT_Vision_Attention(Module):
    def __init__(
        self, config: MoonViTConfig, *, mesh_device: ttnn.MeshDevice, parallel: VisionParallel, compute_kernel_config
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.parallel = parallel
        self.scale = config.head_dim**-0.5
        kwargs = dict(bias=config.attn_bias, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config)
        self.wqkv = Linear(config.hidden_size, 3 * config.qkv_hidden_size, **kwargs)
        self.wo = Linear(config.qkv_hidden_size, config.hidden_size, **kwargs)

    def forward(self, x: ttnn.Tensor, rope: RopeTensors, logical_n: int) -> ttnn.Tensor:
        raise NotImplementedError("TT_Vision_Attention.forward")
