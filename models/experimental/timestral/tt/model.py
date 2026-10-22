# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN Timestral: one decode block of patch embedding -> time/variate transformer -> quantile head.

reference : models/experimental/timestral/reference/src/timestral/model.py (``_predict_context_only``)
    host:   causal per-patch scaling, [values, missing] patch tokens, masks, RoPE tables
    device: x = embed(tokens) * keep + add          # decoder slots become the forecast query
            x = transformer(x)                      # 24 layers, see tt/transformer.py
            out = head(RMSNorm(x[:, L : L + h]))    # out_norm's gain folded into the head
    host:   monotone quantiles, sinh denormalization

The sequence runs padded to a whole tile with the pad tokens after the decoder tokens:
causal time attention never lets a real token read them, and every other op is per token.
``predict_block`` matches the upstream signature, so ``TtTimestralForecaster`` reuses the
upstream block-autoregressive decode, symmetric averaging and refine frame unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from models.experimental.timestral.common.timestral_src import require_timestral
from models.experimental.timestral.tt.attention import TtAttentionInputs
from models.experimental.timestral.tt.common import (
    TILE,
    TtTimestralPrecision,
    pad_to_tile,
    rms_norm,
    upload,
)
from models.experimental.timestral.tt.preprocessing import (
    TimestralHostInputs,
    forecast_token_layout,
    postprocess_head,
    prepare_inputs,
)
from models.experimental.timestral.tt.residual_block import TtResidualBlock
from models.experimental.timestral.tt.transformer import TtTransformerBlock
from models.experimental.timestral.tt.weights import TimestralWeights, rope_pair_order

require_timestral()

from timestral.forecaster import TimestralForecaster  # noqa: E402
from timestral.loading import load_model  # noqa: E402
from timestral.model import ROLE_TARGET  # noqa: E402
from timestral.scaler import apply_affine_asinh_masked, causal_patch_loc_scale  # noqa: E402


@dataclass(frozen=True)
class TtTimestralDeviceInputs:
    tokens: object  # flat (1, 1, R * S_pad, 2 * patch_size) tiles
    attention: TtAttentionInputs
    num_context_patches: int
    horizon_patches: int


def _additive(mask: Tensor) -> Tensor:
    """Boolean attend-mask -> additive SDPA mask (0 / -inf)."""
    return torch.zeros(mask.shape).masked_fill(~mask, float("-inf"))


def _pad_time_mask(mask: Tensor, padded_seq: int) -> Tensor:
    """``(R, 1, S, S)`` -> ``(R, 1, S_pad, S_pad)``: real queries never read pad keys; pads attend causally."""
    rows, _, s, _ = mask.shape
    full = torch.tril(torch.ones(padded_seq, padded_seq, dtype=torch.bool)).expand(rows, 1, -1, -1).clone()
    full[:, :, :s, :] = False
    full[:, :, :s, :s] = mask
    return full


def _pad_variate_mask(mask: Tensor, seq: int, padded_seq: int) -> Tensor:
    """``(S * B, 1, V, V)`` -> ``(S_pad * B, 1, V, V)``; pad slots see their whole series."""
    g = mask.shape[-1]
    b = mask.shape[0] // seq
    full = torch.ones(padded_seq, b, g, g, dtype=torch.bool)
    full[:seq] = mask.reshape(seq, b, g, g)
    return full.reshape(padded_seq * b, 1, g, g)


def _pad_rope(table: Tensor, padded_seq: int, identity: float) -> Tensor:
    """``(rows, S, Dh)`` -> ``(rows, 1, S_pad, Dh)``; pad positions get the identity rotation."""
    rows, s, dh = table.shape
    out = torch.full((rows, padded_seq, dh), identity)
    out[:, :s] = table
    return out.unsqueeze(1)


class TtTimestral:
    """Device Timestral model (single chip). Weights move host -> device once in ``__init__``."""

    def __init__(self, device, weights: TimestralWeights, precision: TtTimestralPrecision | None = None):
        self.device = device
        self.weights = weights
        self.config = weights.config
        self.precision = precision or TtTimestralPrecision()
        self._patch_proj = TtResidualBlock(device, weights.patch_proj, self.precision)
        self._layers = [TtTransformerBlock(device, layer, self.precision) for layer in weights.layers]
        self._head = TtResidualBlock(device, weights.head, self.precision)
        self._token_layouts: dict[tuple[int, int, int], tuple[object, object]] = {}

    @classmethod
    def from_reference(cls, device, model, precision: TtTimestralPrecision | None = None) -> "TtTimestral":
        """Build from an upstream ``TimestralModel``."""
        return cls(device, TimestralWeights.from_reference(model), precision)

    @classmethod
    def from_pretrained(
        cls, device, path_or_repo_id: str | Path, precision: TtTimestralPrecision | None = None, **hub_kwargs
    ) -> "TtTimestral":
        """Load a checkpoint directory or Hub repo id (``config.json`` + ``model.safetensors``)."""
        return cls.from_reference(device, load_model(path_or_repo_id, **hub_kwargs), precision)

    # TimestralModel duck-typing for the upstream forecaster.
    def eval(self) -> "TtTimestral":
        return self

    def scale(self, inputs: Tensor, observed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Host ``TimestralModel.scale``: causal per-patch normalization of ``(R, T)`` rows."""
        c = self.config
        loc, scale = causal_patch_loc_scale(
            inputs.unsqueeze(1), observed.unsqueeze(1), c.patch_size, c.minimum_scale, c.stabilize_with_global
        )
        loc, scale = loc.squeeze(1), scale.squeeze(1)
        return apply_affine_asinh_masked(inputs, loc, scale, observed, use_asinh=c.use_asinh), loc, scale

    def prepare_inputs(
        self, inputs: Tensor, observed: Tensor, group_ids: Tensor, horizon_patches: int
    ) -> TimestralHostInputs:
        return prepare_inputs(self.config, self.weights.inv_freq, inputs, observed, group_ids, horizon_patches)

    def upload_inputs(self, prepared: TimestralHostInputs) -> TtTimestralDeviceInputs:
        import ttnn

        rows, seq = prepared.rows, prepared.seq_len
        padded_seq = pad_to_tile(seq)
        tokens = torch.zeros(rows, padded_seq, prepared.tokens.shape[-1])
        tokens[:, :seq] = prepared.tokens
        # Tilizing on device is far cheaper than on the host for large batches.
        tokens_rm = upload(self.device, tokens.reshape(1, 1, rows * padded_seq, -1), layout=ttnn.ROW_MAJOR_LAYOUT)
        tokens_tt = ttnn.to_layout(tokens_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tokens_rm)

        time_mask = None
        if prepared.time_mask is not None:
            # Time attention runs on an (H, R, S_pad, Dh) view, so the per-row masks sit on dim 1.
            time_mask = upload(self.device, _additive(_pad_time_mask(prepared.time_mask, padded_seq)).transpose(0, 1))
        variate_mask = None
        if prepared.variate_mask is not None:
            variate_mask = self._upload_small_mask(_additive(_pad_variate_mask(prepared.variate_mask, seq, padded_seq)))
        attention = TtAttentionInputs(
            cos=self._upload_rope(prepared.cos, rows, padded_seq, 1.0),
            sin=self._upload_rope(prepared.sin, rows, padded_seq, 0.0),
            unit_stats=ttnn.full(
                ttnn.Shape((1, 1, rows * padded_seq, TILE)),
                self.config.d_model / TILE,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            time_mask=time_mask,
            variate_mask=variate_mask,
            rows=rows,
            padded_seq=padded_seq,
            group_size=prepared.group_size,
        )
        return TtTimestralDeviceInputs(
            tokens=tokens_tt,
            attention=attention,
            num_context_patches=prepared.num_context_patches,
            horizon_patches=prepared.horizon_patches,
        )

    def _upload_rope(self, table: Tensor, rows: int, padded_seq: int, identity: float):
        """``(1 or R, S, Dh)`` -> device ``(1, 1, R * S_pad, Dh)``, one row per token, in ``rope_pair_order``."""
        import ttnn

        padded = _pad_rope(rope_pair_order(table), padded_seq, identity)
        flat = upload(self.device, padded.reshape(1, 1, -1, padded.shape[-1]))
        if padded.shape[0] == rows:
            return flat
        out = ttnn.repeat(flat, ttnn.Shape((1, 1, rows, 1)), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(flat)
        return out

    def _upload_small_mask(self, mask: Tensor):
        """A mask whose last dims are not tile-aligned: the tile padding must be -inf, not 0."""
        import ttnn

        rows = upload(self.device, mask, layout=ttnn.ROW_MAJOR_LAYOUT)
        padded = [*mask.shape[:-2], *(pad_to_tile(s) for s in mask.shape[-2:])]
        out = ttnn.tilize_with_val_padding(rows, padded, float("-inf"), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rows)
        return out

    def _token_layout(self, num_context_patches: int, horizon_patches: int, padded_seq: int):
        key = (num_context_patches, horizon_patches, padded_seq)
        if key not in self._token_layouts:
            keep, add = forecast_token_layout(self.weights.forecast_query, num_context_patches, horizon_patches)
            pad = padded_seq - keep.shape[1]
            keep = torch.nn.functional.pad(keep, (0, 0, 0, pad))
            add = torch.nn.functional.pad(add, (0, 0, 0, pad))
            self._token_layouts[key] = (upload(self.device, keep), upload(self.device, add))
        return self._token_layouts[key]

    def embed(self, inputs: TtTimestralDeviceInputs):
        """Patch tokens -> flat hidden states ``(1, 1, R * S_pad, D)``; decoder slots hold the forecast query."""
        import ttnn

        att = inputs.attention
        r, sp, d = att.rows, att.padded_seq, self.config.d_model
        embedded = ttnn.reshape(self._patch_proj(inputs.tokens), (r, sp, d))
        keep, add = self._token_layout(inputs.num_context_patches, inputs.horizon_patches, sp)
        kept = ttnn.mul(embedded, keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(embedded)
        x = ttnn.add(kept, add, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(kept)
        return ttnn.reshape(x, (1, 1, r * sp, d))

    def head(self, x, inputs: TtTimestralDeviceInputs):
        """Final hidden states -> head output ``(1, 1, R * h, patch_size * Q)`` for the decoder slots; consumes ``x``."""
        import ttnn

        att = inputs.attention
        r, sp, d = att.rows, att.padded_seq, self.config.d_model
        ll, h = inputs.num_context_patches, inputs.horizon_patches
        dec = ttnn.slice(ttnn.reshape(x, (r, sp, d)), (0, ll, 0), (r, ll + h, d))
        ttnn.deallocate(x)
        dec = ttnn.reshape(dec, (1, 1, r * h, d))
        normed = rms_norm(dec, self.weights.out_norm_eps, self.precision)
        ttnn.deallocate(dec)
        out = self._head(normed)
        ttnn.deallocate(normed)
        return out

    def forward_device(self, inputs: TtTimestralDeviceInputs):
        """Device inputs -> device head output ``(1, 1, R * h, patch_size * Q)`` in normalized space."""
        x = self.embed(inputs)
        for layer in self._layers:
            x = layer(x, inputs.attention)
        return self.head(x, inputs)

    def postprocess(self, output_device, prepared: TimestralHostInputs) -> Tensor:
        """Download the head output -> raw-space quantiles ``(R, h * patch_size, Q)``."""
        import ttnn

        raw = ttnn.to_torch(output_device).float().reshape(prepared.rows, prepared.horizon_patches, -1)
        return postprocess_head(self.config, raw, prepared)

    @staticmethod
    def deallocate_inputs(inputs: TtTimestralDeviceInputs) -> None:
        import ttnn

        att = inputs.attention
        for t in (inputs.tokens, att.cos, att.sin, att.unit_stats, att.time_mask, att.variate_mask):
            if t is not None:
                ttnn.deallocate(t)

    def predict_block(
        self,
        inputs: Tensor,
        observed: Tensor,
        group_ids: Tensor,
        horizon_patches: int,
        role_ids: Tensor | None = None,
        known_future: Tensor | None = None,
        known_future_mask: Tensor | None = None,
    ) -> Tensor:
        """``TimestralModel.predict_block`` on device; raw-space quantiles ``(R, h * patch_size, Q)``."""
        import ttnn

        if role_ids is not None and self.config.covariate_enabled and not bool((role_ids == ROLE_TARGET).all()):
            raise NotImplementedError("TT Timestral runs the context-only path; covariate rows are not supported yet")
        prepared = self.prepare_inputs(inputs, observed, group_ids, horizon_patches)
        device_inputs = self.upload_inputs(prepared)
        try:
            out = self.forward_device(device_inputs)
            result = self.postprocess(out, prepared)
            ttnn.deallocate(out)
        finally:
            self.deallocate_inputs(device_inputs)
        return result.to(inputs.dtype)

    __call__ = predict_block


class TtTimestralForecaster(TimestralForecaster):
    """The upstream block-autoregressive forecaster, with ``TtTimestral`` running each decode block."""

    def __init__(self, model: TtTimestral, **kwargs) -> None:
        super().__init__(model, **kwargs)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")
