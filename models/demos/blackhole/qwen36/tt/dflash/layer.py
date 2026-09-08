# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M7: one DFlash decoder layer -- pre-norm attention and MLP, two residuals.

    x = x + attn(input_layernorm(x))
    x = x + mlp(post_attention_layernorm(x))

Identical in structure to a stock Qwen3 layer; everything unusual lives inside
``attention.py``. The residual stream is replicated across devices, so both norms are plain
local ``ttnn.rms_norm`` with no distributed statistics.
"""

from __future__ import annotations

import ttnn
from models.demos.blackhole.qwen36.tt.dflash.attention import DFlashAttention
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig
from models.demos.blackhole.qwen36.tt.dflash.mlp import DFlashMLP
from models.demos.blackhole.qwen36.tt.dflash.rms_norm import rms_norm

_MC = ttnn.DRAM_MEMORY_CONFIG


class DFlashLayer:
    def __init__(self, mesh_device, cfg: DFlashDrafterConfig, weights, layer_idx: int, tt_ccl, topology=None):
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.w = weights
        self.attention = DFlashAttention(mesh_device, cfg, weights, layer_idx, tt_ccl, topology=topology)
        self.mlp = DFlashMLP(mesh_device, cfg, weights, tt_ccl, topology=topology)

    def project_context(self, target_hidden):
        """This layer's context K/V. Block-independent, so cacheable across blocks."""
        return self.attention.project_context(target_hidden)

    def forward(self, x, ctx_k, ctx_v, rope, cos_q, sin_q, cos_k, sin_k, attn_mask):
        eps = self.cfg.rms_norm_eps

        normed = rms_norm(x, self.w.input_layernorm, eps, memory_config=_MC)
        attn_out = self.attention.forward(normed, ctx_k, ctx_v, rope, cos_q, sin_q, cos_k, sin_k, attn_mask)
        ttnn.deallocate(normed)
        x = ttnn.add(x, attn_out, memory_config=_MC)
        ttnn.deallocate(attn_out)

        normed = rms_norm(x, self.w.post_attention_layernorm, eps, memory_config=_MC)
        mlp_out = self.mlp.forward(normed)
        ttnn.deallocate(normed)
        out = ttnn.add(x, mlp_out, memory_config=_MC)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(x)
        return out
