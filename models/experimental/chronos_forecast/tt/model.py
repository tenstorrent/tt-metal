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
from models.experimental.chronos_forecast.tt.group_attention import build_group_mask, pack_group_blocks
from models.experimental.chronos_forecast.tt.mha_core import maybe_upload_mask
from models.experimental.chronos_forecast.tt.model_preprocessing import (
    instance_norm_inverse,
    prepare_patched_context,
    prepare_patched_future,
)
from models.experimental.chronos_forecast.tt.program_configs import TtChronosPrecision
from models.experimental.chronos_forecast.tt.residual_block import (
    TtResidualBlock,
    TtResidualBlockWeights,
    pad_input_features,
)
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
class TtChronosPreparedInputs:
    """Host tensors prepared once for the device-resident forward."""

    patched_context: torch.Tensor
    patched_future: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    time_mask: torch.Tensor
    group_mask: torch.Tensor | None
    reg_token: torch.Tensor | None
    loc_scale: tuple[torch.Tensor, torch.Tensor]
    num_context_patches: int
    num_output_patches: int
    unique_groups: bool
    # Grouped series are packed into ``group_block``-series blocks (see
    # ``pack_group_blocks``); ``output_rows`` maps each input series to its packed row.
    group_block: int | None = None
    output_rows: torch.Tensor | None = None


@dataclass(frozen=True)
class TtChronosDeviceInputs:
    """Address-stable device inputs used by eager-device and trace execution."""

    patched_context: object
    patched_future: object
    cos: object
    sin: object
    time_mask: object | None
    group_mask: object | None
    reg_token: object | None
    batch_size: int
    num_context_patches: int
    num_output_patches: int
    unique_groups: bool
    group_block: int | None = None


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

    def __init__(
        self,
        device,
        weights: TtChronosWeights,
        config: TtChronosConfig,
        precision: TtChronosPrecision | None = None,
        *,
        l1_chunk_tokens: int | None = None,
    ):
        """``l1_chunk_tokens`` (e.g. ``precision.l1_chunk_tokens()``) runs the encoder
        L1-resident on chunks of ``l1_chunk_tokens // padded_T`` series (rounded down
        to whole group blocks); ``None``, or a group block larger than a chunk,
        keeps activations in DRAM."""
        self.device = device
        self.weights = weights
        self.config = config
        self.precision = precision or TtChronosPrecision()
        self.l1_chunk_tokens = l1_chunk_tokens
        self._input_embed = TtResidualBlock(device, weights.input_embed)
        self._encoder = TtEncoder(device, weights.encoder, self.precision)
        self._output_embed = TtResidualBlock(device, weights.output_embed)

    @classmethod
    def from_torch_model(
        cls, device, model, precision: TtChronosPrecision | None = None, *, l1_chunk_tokens: int | None = None
    ) -> "TtChronos":
        """Build from a reference ``Chronos2Model`` (weights + geometry)."""
        weights = TtChronosWeights.from_torch_model(model)
        return cls(
            device, weights, tt_chronos_config_from_torch_model(model), precision, l1_chunk_tokens=l1_chunk_tokens
        )

    def _l1_series_chunk(self, padded_seq_len: int) -> int | None:
        if self.l1_chunk_tokens is None:
            return None
        return max(1, self.l1_chunk_tokens // padded_seq_len)

    def prepare_inputs(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
    ) -> TtChronosPreparedInputs:
        """Run host-only patching, normalization, masks, and RoPE construction."""
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

        if group_ids is None:
            group_ids = torch.arange(batch_size, dtype=torch.long)
        unique_groups = bool(torch.unique(group_ids).numel() == batch_size)
        if cfg.use_reg_token:
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, dtype=attention_mask.dtype)], dim=-1)

        future_mask = torch.ones(batch_size, num_output_patches, dtype=attention_mask.dtype)
        full_mask = torch.cat([attention_mask, future_mask], dim=-1)
        if not bool((full_mask > 0).all()):
            raise ValueError("TtChronos v1 supports all-valid masks only (got a masked position)")
        seq_len = full_mask.shape[-1]
        time_mask = torch.zeros(1, 1, seq_len, seq_len)

        group_mask, group_block, output_rows = None, None, None
        if not unique_groups:
            # All-valid masks make group attention block-diagonal over group-sorted series.
            preferred_block = 128
            l1_series_chunk = self._l1_series_chunk(-(-seq_len // 32) * 32)
            if l1_series_chunk is not None:
                preferred_block = min(preferred_block, max(32, l1_series_chunk // 32 * 32))
            packing = pack_group_blocks(group_ids, preferred_block=preferred_block)
            patched_context = patched_context[packing.rows]
            patched_future = patched_future[packing.rows]
            batch_size = packing.rows.numel()
            group_block, output_rows = packing.block, packing.output_rows
            # SDPA batch is (time, block) after the batch/time flip; one mask broadcasts when blocks match.
            group_mask = packing.mask[:1] if packing.is_uniform() else packing.mask.repeat(seq_len, 1, 1, 1)
        reg_token = None
        if cfg.use_reg_token:
            reg_token = (
                self.weights.shared_weight[self.weights.reg_token_id]
                .reshape(1, 1, -1)
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        # Every series uses positions 0..T-1, so one (1,T,Dh) cache broadcasts over the batch.
        position_ids = torch.arange(seq_len).unsqueeze(0)
        inv_freq = self.weights.encoder.blocks[0].time.inv_freq
        cos, sin = build_rope_cache(position_ids, inv_freq)
        in_features = self._input_embed.in_features
        return TtChronosPreparedInputs(
            patched_context=pad_input_features(patched_context, in_features),
            patched_future=pad_input_features(patched_future, in_features),
            cos=cos,
            sin=sin,
            time_mask=time_mask,
            group_mask=group_mask,
            reg_token=reg_token,
            loc_scale=loc_scale,
            num_context_patches=num_context_patches,
            num_output_patches=num_output_patches,
            unique_groups=unique_groups,
            group_block=group_block,
            output_rows=output_rows,
        )

    def upload_inputs(self, prepared: TtChronosPreparedInputs) -> TtChronosDeviceInputs:
        """Upload prepared tensors once; returned tensors may be reused by a trace."""
        import ttnn

        def upload(tensor: torch.Tensor):
            return ttnn.from_torch(
                tensor.detach().to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        batch_size = prepared.patched_context.shape[0]
        seq_len = (
            prepared.num_context_patches + (1 if prepared.reg_token is not None else 0) + prepared.num_output_patches
        )
        return TtChronosDeviceInputs(
            patched_context=upload(prepared.patched_context),
            patched_future=upload(prepared.patched_future),
            cos=ttnn.unsqueeze(upload(prepared.cos), 1),
            sin=ttnn.unsqueeze(upload(prepared.sin), 1),
            time_mask=(
                None
                if not bool((prepared.time_mask != 0).any())
                else maybe_upload_mask(self.device, prepared.time_mask, seq_len=seq_len)
            ),
            group_mask=(
                None
                if prepared.unique_groups
                else maybe_upload_mask(self.device, prepared.group_mask, seq_len=prepared.group_block)
            ),
            reg_token=upload(prepared.reg_token) if prepared.reg_token is not None else None,
            batch_size=batch_size,
            num_context_patches=prepared.num_context_patches,
            num_output_patches=prepared.num_output_patches,
            unique_groups=prepared.unique_groups,
            group_block=prepared.group_block,
        )

    def forward_device(self, inputs: TtChronosDeviceInputs):
        """Execute embeddings, encoder, and output head without a host round-trip."""
        import ttnn

        context_embeds = self._input_embed.forward_device(inputs.patched_context)
        future_embeds = self._input_embed.forward_device(inputs.patched_future)
        context_embeds = ttnn.reshape(
            context_embeds,
            (inputs.batch_size, inputs.num_context_patches, self.config.d_model),
        )
        future_embeds = ttnn.reshape(
            future_embeds,
            (inputs.batch_size, inputs.num_output_patches, self.config.d_model),
        )
        pieces = [context_embeds]
        if inputs.reg_token is not None:
            pieces.append(inputs.reg_token)
        pieces.append(future_embeds)
        x = ttnn.concat(pieces, dim=-2)
        ttnn.deallocate(context_embeds)
        ttnn.deallocate(future_embeds)

        l1_series_chunk = self._l1_series_chunk(x.padded_shape[-2])
        if l1_series_chunk is not None and inputs.group_block is not None:
            # Chunks hold whole group blocks; a block larger than one chunk runs from DRAM.
            l1_series_chunk = l1_series_chunk // inputs.group_block * inputs.group_block or None
        hidden = self._encoder.forward_device(
            x,
            inputs.cos,
            inputs.sin,
            inputs.time_mask,
            inputs.group_mask,
            diagonal_group_attention=inputs.unique_groups,
            group_block=inputs.group_block,
            l1_series_chunk=l1_series_chunk,
        )
        seq_len = hidden.shape[-2]
        forecast_embeds = ttnn.slice(
            hidden,
            (0, seq_len - inputs.num_output_patches, 0),
            (inputs.batch_size, seq_len, self.config.d_model),
        )
        ttnn.deallocate(hidden)
        return self._output_embed.forward_device(forecast_embeds, deallocate_input=True)

    def postprocess_output(
        self,
        output_device,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        *,
        num_output_patches: int,
        output_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Download and unscale the device output into `(B, Q, horizon)`.

        Pass ``prepared.output_rows`` so grouped (packed) batches come back in input order.
        """
        import ttnn

        out = ttnn.to_torch(output_device).float()
        if out.dim() == 4 and out.shape[0] == 1:
            out = out.squeeze(0)
        if output_rows is not None:
            out = out[output_rows]
        out = out[:, :num_output_patches, :]
        quantile_preds = rearrange(
            out,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.config.num_quantiles,
            p=self.config.output_patch_size,
        )
        batch_size = quantile_preds.shape[0]
        quantile_preds = rearrange(
            quantile_preds,
            "b q h -> b (q h)",
            b=batch_size,
            q=self.config.num_quantiles,
        )
        quantile_preds = instance_norm_inverse(
            quantile_preds,
            loc_scale,
            use_arcsinh=self.config.use_arcsinh,
        )
        return rearrange(quantile_preds, "b (q h) -> b q h", q=self.config.num_quantiles)

    @staticmethod
    def deallocate_inputs(inputs: TtChronosDeviceInputs) -> None:
        import ttnn

        for tensor in (
            inputs.patched_context,
            inputs.patched_future,
            inputs.cos,
            inputs.sin,
            inputs.time_mask,
            inputs.group_mask,
            inputs.reg_token,
        ):
            if tensor is not None:
                ttnn.deallocate(tensor)

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
        position_ids = torch.arange(seq_len).unsqueeze(0)
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

        # final prediction head
        out = self._output_embed.forward(forecast_embeds)

        # rearrange output
        quantile_preds = rearrange(
            out,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=cfg.num_quantiles,
            p=cfg.output_patch_size,
        )
        quantile_preds = rearrange(quantile_preds, "b q h -> b (q h)", b=batch_size, q=cfg.num_quantiles)
        quantile_preds = instance_norm_inverse(
            quantile_preds,
            loc_scale,
            use_arcsinh=cfg.use_arcsinh,
        )
        return rearrange(quantile_preds, "b (q h) -> b q h", q=cfg.num_quantiles)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
