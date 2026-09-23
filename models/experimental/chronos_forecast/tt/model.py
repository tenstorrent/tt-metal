# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip TTNN Chronos2Model (input embed -> encoder -> output embed).

reference : models/experimental/chronos_forecast/reference/chronos2/model.py
    embeds = input_embed(patches) + REG + future_embeds  # host concat
    hidden = encoder(embeds)                             # one upload/download
    quantiles = output_embed(hidden[:, -O:])             # host rearrange + unscale

v1: all-valid masks only (variable masks raise); embeddings/encoder on device, rest on host.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange

from models.experimental.chronos_forecast.tt.encoder import TtEncoder, TtEncoderWeights
from models.experimental.chronos_forecast.tt.group_attention import build_group_mask
from models.experimental.chronos_forecast.tt.model_preprocessing import (
    instance_norm_inverse,
    prepare_patched_context,
    prepare_patched_future,
)
from models.experimental.chronos_forecast.tt.residual_block import TtResidualBlock, TtResidualBlockWeights
from models.experimental.chronos_forecast.tt.time_attention import build_rope_cache


@dataclass(frozen=True)
class TtChronosConfig:
    """Host-side model geometry (mirrors ``Chronos2ForecastingConfig`` fields we need)."""

    context_length: int
    input_patch_size: int
    input_patch_stride: int
    output_patch_size: int
    time_encoding_scale: int
    use_arcsinh: bool
    use_reg_token: bool
    num_quantiles: int
    d_model: int


@dataclass(frozen=True)
class TtChronosWeights:
    """Host-side weights using ``nn.Linear`` convention."""

    input_embed: TtResidualBlockWeights
    encoder: TtEncoderWeights
    output_embed: TtResidualBlockWeights
    shared_weight: torch.Tensor  # (vocab, d) — REG token lookup stays on host
    reg_token_id: int

    @classmethod
    def from_torch_model(cls, model) -> "TtChronosWeights":
        """Extract weights from a reference ``Chronos2Model``."""
        return cls(
            input_embed=TtResidualBlockWeights.from_torch_block(model.input_patch_embedding),
            encoder=TtEncoderWeights.from_torch_encoder(model.encoder),
            output_embed=TtResidualBlockWeights.from_torch_block(model.output_patch_embedding),
            shared_weight=model.shared.weight.detach().clone(),
            reg_token_id=int(model.config.reg_token_id),
        )


def tt_chronos_config_from_torch_model(model) -> TtChronosConfig:
    """Extract geometry from a reference ``Chronos2Model``."""
    cc = model.chronos_config
    return TtChronosConfig(
        context_length=int(cc.context_length),
        input_patch_size=int(cc.input_patch_size),
        input_patch_stride=int(cc.input_patch_stride),
        output_patch_size=int(cc.output_patch_size),
        time_encoding_scale=int(cc.time_encoding_scale or cc.context_length),
        use_arcsinh=bool(cc.use_arcsinh),
        use_reg_token=bool(cc.use_reg_token),
        num_quantiles=len(cc.quantiles),
        d_model=int(model.config.d_model),
    )


class TtChronos:
    """Device Chronos-2 model. Weights move host -> device once in ``__init__``."""

    def __init__(self, device, weights: TtChronosWeights, config: TtChronosConfig):
        self.device = device
        self.weights = weights
        self.config = config
        self._input_embed = TtResidualBlock(device, weights.input_embed)
        self._encoder = TtEncoder(device, weights.encoder)
        self._output_embed = TtResidualBlock(device, weights.output_embed)

    @classmethod
    def from_torch_model(cls, device, model) -> "TtChronos":
        """Build from a reference ``Chronos2Model`` (weights + geometry)."""
        weights = TtChronosWeights.from_torch_model(model)
        return cls(device, weights, tt_chronos_config_from_torch_model(model))

    def encode(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], int]:
        """Host-orchestrated encode. Returns (hidden (B,L,d) host float, loc_scale, num_context_patches)."""
        cfg = self.config
        batch_size = context.shape[0]

        patched_context, attention_mask, loc_scale = prepare_patched_context(
            context,
            context_mask,
            patch_size=cfg.input_patch_size,
            patch_stride=cfg.input_patch_stride,
            context_length=cfg.context_length,
            time_encoding_scale=cfg.time_encoding_scale,
            use_arcsinh=cfg.use_arcsinh,
        )
        num_context_patches = attention_mask.shape[-1]

        # Input embeddings (each its own host round-trip; encoder stays single-upload below).
        input_embeds = self._input_embed.forward(patched_context)
        if cfg.use_reg_token:
            reg = self.weights.shared_weight[self.weights.reg_token_id].reshape(1, 1, -1).expand(batch_size, -1, -1)
            input_embeds = torch.cat([input_embeds, reg], dim=-2)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, dtype=attention_mask.dtype)], dim=-1)

        patched_future, _ = prepare_patched_future(
            future_covariates,
            loc_scale,
            num_output_patches=num_output_patches,
            output_patch_size=cfg.output_patch_size,
            batch_size=batch_size,
            time_encoding_scale=cfg.time_encoding_scale,
            future_covariates_mask=future_covariates_mask,
            use_arcsinh=cfg.use_arcsinh,
        )
        future_embeds = self._input_embed.forward(patched_future)

        x = torch.cat([input_embeds, future_embeds], dim=-2)
        seq_len = x.shape[-2]
        future_mask = torch.ones(batch_size, num_output_patches, dtype=attention_mask.dtype)
        full_mask = torch.cat([attention_mask, future_mask], dim=-1)
        if not bool((full_mask > 0).all()):
            raise ValueError("TtChronos v1 supports all-valid masks only (got a masked position)")

        if group_ids is None:
            group_ids = torch.arange(batch_size, dtype=torch.long)

        time_mask = torch.zeros(1, 1, seq_len, seq_len)
        group_time_mask = build_group_mask(group_ids, (full_mask > 0).float())
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        inv_freq = self.weights.encoder.blocks[0].time.inv_freq
        cos, sin = build_rope_cache(position_ids, inv_freq)

        hidden = self._encoder.forward(x, cos, sin, time_mask, group_time_mask)
        return hidden, loc_scale, num_context_patches

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
    ) -> torch.Tensor:
        """Host-orchestrated forward. Returns quantile preds (B, Q, O*P) host float."""
        cfg = self.config
        batch_size = context.shape[0]
        hidden, loc_scale, _ = self.encode(
            context,
            context_mask,
            group_ids,
            future_covariates,
            future_covariates_mask,
            num_output_patches,
        )
        forecast_embeds = hidden[:, -num_output_patches:]
        out = self._output_embed.forward(forecast_embeds)
        quantile_preds = rearrange(
            out,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=cfg.num_quantiles,
            p=cfg.output_patch_size,
        )
        quantile_preds = rearrange(quantile_preds, "b q h -> b (q h)", b=batch_size, q=cfg.num_quantiles)
        quantile_preds = instance_norm_inverse(quantile_preds, loc_scale)
        return rearrange(quantile_preds, "b (q h) -> b q h", q=cfg.num_quantiles)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
