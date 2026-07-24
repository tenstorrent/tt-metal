# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 lyric encoder (TTTv2-pattern composition module).

Reference: AceStepLyricEncoder in modeling_acestep_v15_base.py:

    x = embed_tokens(inputs_embeds)        # Linear(text_hidden_dim -> hidden), with bias
    for layer in layers:                   # N encoder layers, alternating full/sliding
        x = layer(x, rope, mask[layer.attention_type])
    x = norm(x)                            # final RMSNorm

Composes: input projection (ttnn.linear) + N x AceStepEncoderLayer (reused) + RMSNorm1D.
Per-layer mask/attention_type (full vs sliding) is selected at construction (no forward
branching): each layer is built with its own sliding_window and receives the matching mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.experimental.acestep.tt.encoder_layer import AceStepEncoderLayer


@dataclass
class AceStepLyricEncoderConfig:
    # Input projection Linear(text_hidden_dim -> hidden) [in,out] + bias [1,hidden].
    embed_weight: LazyWeight
    embed_bias: LazyWeight
    # Final norm.
    norm_weight: LazyWeight
    # Per-layer configs (already built AceStepEncoderLayerConfig, in order).
    layer_configs: list = field(default_factory=list)
    # attention_type per layer ("full_attention" | "sliding_attention"), same order.
    layer_types: list = field(default_factory=list)

    eps: float = 1e-6
    mesh_device: ttnn.MeshDevice | None = None
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    def resolved(self) -> "AceStepLyricEncoderConfig":
        if self.mesh_device is None:
            self.mesh_device = self.embed_weight.device
        if self.compute_kernel_config is None:
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        return self


class AceStepLyricEncoder(LightweightModule):
    """forward(inputs_embeds [1,1,seq,text_hidden_dim], cos, sin, sliding_mask=None) ->
    [1,1,seq,hidden]. full-attention layers get mask=None; sliding layers get sliding_mask."""

    def __init__(self, config: AceStepLyricEncoderConfig):
        self.config = config.resolved()
        cfg = self.config
        self.embed_weight = cfg.embed_weight.get_device_weight()
        self.embed_bias = cfg.embed_bias.get_device_weight()
        self.layers = [AceStepEncoderLayer(lc) for lc in cfg.layer_configs]
        self.layer_types = cfg.layer_types
        self.norm = RMSNorm1D.from_config(RMSNorm1DConfig(weight=cfg.norm_weight, eps=cfg.eps))

    @classmethod
    def from_config(cls, config: AceStepLyricEncoderConfig):
        return cls(config)

    def forward(self, inputs_embeds, cos, sin, sliding_mask=None):
        cfg = self.config
        x = ttnn.linear(
            inputs_embeds,
            self.embed_weight,
            bias=self.embed_bias,
            compute_kernel_config=cfg.compute_kernel_config,
        )  # [1,1,seq,hidden]

        for layer, attn_type in zip(self.layers, self.layer_types):
            mask = sliding_mask if attn_type == "sliding_attention" else None
            x = layer.forward(x, cos, sin, attn_mask=mask)

        x = self.norm.forward(x, mode="prefill")
        return x
