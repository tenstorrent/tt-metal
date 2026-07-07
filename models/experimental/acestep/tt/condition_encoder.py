# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 ConditionEncoder assembly (TTTv2-pattern top-level composition).

Reference: AceStepConditionEncoder.forward in modeling_acestep_v15_base.py. Builds the
cross-attention context for the DiT by encoding and packing three conditioning streams:

    text  = text_projector(text_hidden_states)          # Linear(text_hidden_dim -> hidden), no bias
    lyric = lyric_encoder(lyric_hidden_states)           # Linear + N enc layers + norm
    timbre= timbre_encoder(refer_audio_...)              # Linear + N enc layers + norm (+ slice)
    ctx, mask = pack_sequences(lyric, timbre)            # concat + stable argsort by mask
    ctx, mask = pack_sequences(ctx, text)                # concat + stable argsort by mask

pack_sequences reorders valid (mask=1) tokens before padding (mask=0) via a stable argsort. In
the all-valid case (no padding) it is exactly a concatenation — the contract this module targets
for bring-up. The data-dependent reordering for padded batches is host orchestration handled by
the caller/pipeline (the sort indices come from CPU masks, not device compute).

Composes: text_projector (ttnn.linear) + AceStepLyricEncoder (reused for BOTH lyric and timbre,
since the timbre core is structurally identical) + concat. Zero new attention/MLP code.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.experimental.acestep.tt.lyric_encoder import AceStepLyricEncoder, AceStepLyricEncoderConfig


@dataclass
class AceStepConditionEncoderConfig:
    # text_projector Linear(text_hidden_dim -> hidden), no bias, transposed to [in,out].
    text_projector_weight: LazyWeight
    # Lyric + timbre encoders (both are AceStepLyricEncoder-shaped).
    lyric_encoder: AceStepLyricEncoderConfig
    timbre_encoder: AceStepLyricEncoderConfig

    mesh_device: ttnn.MeshDevice | None = None
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    def resolved(self) -> "AceStepConditionEncoderConfig":
        if self.mesh_device is None:
            self.mesh_device = self.text_projector_weight.device
        if self.compute_kernel_config is None:
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        return self


class AceStepConditionEncoder(LightweightModule):
    """forward(text [1,1,Lt,text_dim], lyric [1,1,Ll,text_dim], timbre [1,1,Lm,timbre_dim],
    lyric_cos, lyric_sin, timbre_cos, timbre_sin, lyric_sliding=None, timbre_sliding=None)
    -> packed context [1,1,Ll+Lm+Lt, hidden] (all-valid-mask packing == concat)."""

    def __init__(self, config: AceStepConditionEncoderConfig):
        self.config = config.resolved()
        cfg = self.config
        self.text_projector_weight = cfg.text_projector_weight.get_device_weight()
        self.lyric_encoder = AceStepLyricEncoder(cfg.lyric_encoder)
        self.timbre_encoder = AceStepLyricEncoder(cfg.timbre_encoder)

    @classmethod
    def from_config(cls, config: AceStepConditionEncoderConfig):
        return cls(config)

    def forward(
        self, text, lyric, timbre, lyric_cos, lyric_sin, timbre_cos, timbre_sin, lyric_sliding=None,
        timbre_sliding=None, timbre_cls=False
    ):
        cfg = self.config

        text_ctx = ttnn.linear(
            text, self.text_projector_weight, compute_kernel_config=cfg.compute_kernel_config
        )  # [1,1,Lt,hidden]
        lyric_ctx = self.lyric_encoder.forward(lyric, lyric_cos, lyric_sin, sliding_mask=lyric_sliding)
        timbre_ctx = self.timbre_encoder.forward(timbre, timbre_cos, timbre_sin, sliding_mask=timbre_sliding)

        # Reference AceStepTimbreEncoder slices position 0 as the single CLS timbre token per reference
        # audio (hidden_states[:, 0, :]) -> 1 token, not the whole encoded sequence. timbre_cls=True
        # reproduces this (used by the real generation path); default off preserves the per-module tests.
        if timbre_cls:
            timbre_ctx = timbre_ctx[:, :, 0:1, :]

        # pack_sequences twice, all-valid -> concat(lyric, timbre) then concat(that, text).
        ctx = ttnn.concat([lyric_ctx, timbre_ctx], dim=2)
        ctx = ttnn.concat([ctx, text_ctx], dim=2)
        return ctx
