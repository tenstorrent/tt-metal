# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 AudioTokenDetokenizer (TTTv2-pattern composition module).

Reference: AudioTokenDetokenizer in modeling_acestep_v15_base.py. Reconstructs continuous
audio latents from quantized tokens by expanding each token into pool_window_size patches:

    x = embed_tokens(x)                          # Linear(hidden -> hidden) + bias, [B,T,D]
    x = x.unsqueeze(2).repeat(1,1,P,1)           # [B,T,P,D] each token -> P patches
    x = x + special_tokens                       # learned per-patch offsets [1,P,D]
    x = rearrange("b t p c -> (b t) p c")        # [(B*T), P, D]
    for layer: x = layer(x, rope, mask[type])    # 2 encoder layers (sliding/full)
    x = norm(x)
    x = proj_out(x)                              # Linear(hidden -> acoustic_dim) + bias
    x = rearrange("(b t) p c -> b (t p) c")      # [B, T*P, acoustic_dim]

Composes: input Linear + patch-expand + AceStepEncoderLayer x2 + RMSNorm1D + output Linear.
Implemented for a single (B=1, T=1) token group -> P=pool_window_size patches. The per-(b,t)
fold in the reference is batching orchestration around this same core.

NOTE (correctness): P (=5) < sliding_window (128), so the sliding mask is all-visible ->
pass sliding_mask=None (an all-zero additive mask is NOT equivalent to None in ttnn SDPA).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.experimental.acestep.tt.encoder_layer import AceStepEncoderLayer


@dataclass
class AudioTokenDetokenizerConfig:
    embed_weight: LazyWeight  # Linear(hidden->hidden) [in,out]
    embed_bias: LazyWeight  # [1, hidden]
    special_tokens: LazyWeight  # [1, P, hidden]
    norm_weight: LazyWeight
    proj_out_weight: LazyWeight  # Linear(hidden->acoustic) [in,out]
    proj_out_bias: LazyWeight  # [1, acoustic]
    layer_configs: list = field(default_factory=list)
    layer_types: list = field(default_factory=list)

    hidden_size: int = 2048
    acoustic_dim: int = 64
    pool_window_size: int = 5
    eps: float = 1e-6
    mesh_device: ttnn.MeshDevice | None = None
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    def resolved(self) -> "AudioTokenDetokenizerConfig":
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


class AudioTokenDetokenizer(LightweightModule):
    """forward(x [1,1,T=1,hidden], cos, sin, sliding_mask=None) -> [1,1,P, acoustic_dim].

    cos/sin sized for the P-patch sequence. Full layers get mask=None; sliding layers get
    sliding_mask (caller passes None when window>=P)."""

    def __init__(self, config: AudioTokenDetokenizerConfig):
        self.config = config.resolved()
        cfg = self.config
        self.embed_weight = cfg.embed_weight.get_device_weight()
        self.embed_bias = cfg.embed_bias.get_device_weight()
        self.special_tokens = cfg.special_tokens.get_device_weight()  # [1,P,hidden]
        self.layers = [AceStepEncoderLayer(lc) for lc in cfg.layer_configs]
        self.layer_types = cfg.layer_types
        self.norm = RMSNorm1D.from_config(RMSNorm1DConfig(weight=cfg.norm_weight, eps=cfg.eps))
        self.proj_out_weight = cfg.proj_out_weight.get_device_weight()
        self.proj_out_bias = cfg.proj_out_bias.get_device_weight()

    @classmethod
    def from_config(cls, config: AudioTokenDetokenizerConfig):
        return cls(config)

    def forward(self, x, cos, sin, sliding_mask=None):
        cfg = self.config
        p, d = cfg.pool_window_size, cfg.hidden_size

        # Embed the single token: [1,1,1,hidden].
        x = ttnn.linear(x, self.embed_weight, bias=self.embed_bias, compute_kernel_config=cfg.compute_kernel_config)

        # Expand token -> P patches and add learned per-patch offsets.
        # broadcast [1,1,1,hidden] over P via add with special_tokens [1,1,P,hidden].
        sp = ttnn.reshape(self.special_tokens, (1, 1, p, d))
        x = ttnn.add(x, sp)  # [1,1,P,hidden] (x broadcasts along the P axis)

        for layer, attn_type in zip(self.layers, self.layer_types):
            mask = sliding_mask if attn_type == "sliding_attention" else None
            x = layer.forward(x, cos, sin, attn_mask=mask)

        x = self.norm.forward(x, mode="prefill")
        x = ttnn.linear(
            x, self.proj_out_weight, bias=self.proj_out_bias, compute_kernel_config=cfg.compute_kernel_config
        )  # [1,1,P, acoustic_dim]
        return x
