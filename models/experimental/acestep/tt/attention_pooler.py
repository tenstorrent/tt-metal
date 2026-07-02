# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 AttentionPooler (TTTv2-pattern composition module).

Reference: AttentionPooler in modeling_acestep_v15_base.py. A CLS-style pooler used to
aggregate patch-level features into a single vector per (batch, time) group:

    x = embed_tokens(x)                         # Linear(hidden -> hidden) + bias
    x = cat([special_token, x], dim=patch)      # prepend a learned CLS token
    for layer in layers: x = layer(x, rope, mask[type])   # N encoder layers (full/sliding)
    x = norm(x)
    pooled = x[:, 0, :]                          # take the CLS position

Composes: input Linear (ttnn.linear) + prepend CLS + N x AceStepEncoderLayer (reused) +
RMSNorm1D + slice position 0. Implemented for a single (B=1, T=1) pooling group — the
per-(b,t) fold in the reference is pure batching orchestration around this same core.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.experimental.acestep.tt.encoder_layer import AceStepEncoderLayer


@dataclass
class AttentionPoolerConfig:
    # Input projection Linear(hidden -> hidden) [in,out] + bias [1,hidden].
    embed_weight: LazyWeight
    embed_bias: LazyWeight
    # Learned CLS token [1, 1, hidden] as LazyWeight.
    special_token: LazyWeight
    # Final norm.
    norm_weight: LazyWeight
    # Per-layer AceStepEncoderLayerConfig (in order) + attention_type per layer.
    layer_configs: list = field(default_factory=list)
    layer_types: list = field(default_factory=list)

    hidden_size: int = 2048
    eps: float = 1e-6
    mesh_device: ttnn.MeshDevice | None = None
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    def resolved(self) -> "AttentionPoolerConfig":
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


class AttentionPooler(LightweightModule):
    """forward(x [1,1,P,hidden], cos, sin, sliding_mask=None) -> pooled [1,1,1,hidden].

    cos/sin must be sized for the CLS-prepended sequence length P+1. Full-attention layers
    get mask=None; sliding layers get sliding_mask (also sized for P+1)."""

    def __init__(self, config: AttentionPoolerConfig):
        self.config = config.resolved()
        cfg = self.config
        self.embed_weight = cfg.embed_weight.get_device_weight()
        self.embed_bias = cfg.embed_bias.get_device_weight()
        self.special_token = cfg.special_token.get_device_weight()  # [1,1,hidden]
        self.layers = [AceStepEncoderLayer(lc) for lc in cfg.layer_configs]
        self.layer_types = cfg.layer_types
        self.norm = RMSNorm1D.from_config(RMSNorm1DConfig(weight=cfg.norm_weight, eps=cfg.eps))

    @classmethod
    def from_config(cls, config: AttentionPoolerConfig):
        return cls(config)

    def forward(self, x, cos, sin, sliding_mask=None):
        cfg = self.config
        # Project patches.
        x = ttnn.linear(
            x, self.embed_weight, bias=self.embed_bias, compute_kernel_config=cfg.compute_kernel_config
        )  # [1,1,P,hidden]

        # Prepend the learned CLS token -> [1,1,P+1,hidden].
        cls = ttnn.reshape(self.special_token, (1, 1, 1, cfg.hidden_size))
        x = ttnn.concat([cls, x], dim=2)

        for layer, attn_type in zip(self.layers, self.layer_types):
            mask = sliding_mask if attn_type == "sliding_attention" else None
            x = layer.forward(x, cos, sin, attn_mask=mask)

        x = self.norm.forward(x, mode="prefill")

        # Take the CLS position (index 0) -> [1,1,1,hidden].
        pooled = x[:, :, 0:1, :]
        return pooled
