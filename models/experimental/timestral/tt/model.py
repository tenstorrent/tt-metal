# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN Timestral: one decode block of patch embedding -> time/variate transformer -> quantile head.

reference : models/experimental/timestral/reference/src/timestral/model.py (``_predict_context_only``)
    host:   causal per-patch scaling, [values, missing] patch tokens, masks, RoPE tables
    device: x = embed(tokens) * keep + add          # decoder slots become the forecast query
            x = transformer(x)[:, L : L + h]        # 24 layers; the last cuts to the decoder slots
            out = head(RMSNorm(x))                  # out_norm's gain folded into the head
    host:   monotone quantiles, sinh denormalization

The sequence runs padded to a whole tile with the pad tokens after the decoder tokens:
causal time attention never lets a real token read them, and every other op is per token.
When the decoder tokens cross a tile boundary (paper shape: 64 + 2 tokens -> 96) and pack into
whole tiles, context and decoder tokens instead run as two streams (``SplitStreams``) that meet
only in time attention, and the last layer computes just the context keys and values.
``predict_block`` matches the upstream signature, so ``TtTimestralForecaster`` reuses the
upstream block-autoregressive decode, symmetric averaging and refine frame unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from models.experimental.timestral.common.timestral_src import require_timestral
from models.experimental.timestral.tt.attention import DecoderSlots, SplitStreams, TtAttentionInputs
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
    tokens: object  # flat (1, 1, R * S_pad, 2 * patch_size) tiles; context tokens only, R * L_pad, when split
    attention: TtAttentionInputs | None  # single stream
    decoder: DecoderSlots | None  # single stream
    split: SplitStreams | None
    num_context_patches: int
    horizon_patches: int
    owned: tuple  # the tensors deallocate_inputs frees (not the per-shape caches)


def split_rows_per_block(rows: int, num_context_patches: int, horizon_patches: int) -> int | None:
    """Rows whose decoder queries fill one tile, when the split streams run fewer tokens than the
    padded single stream and the rows pack into such blocks; else None."""
    ll, h = num_context_patches, horizon_patches
    if TILE % h or rows % (TILE // h) or pad_to_tile(ll) + h >= pad_to_tile(ll + h):
        return None
    return TILE // h


def _cross_mask(time_mask: Tensor, num_context_patches: int, context_seq: int, rows_per_block: int) -> Tensor:
    """``(R, 1, S, S)`` -> ``(1, R / b, b * h, b * (L_pad + h))`` for blocks of ``b`` rows: decoder query
    ``(row, t)`` reads its row's context keys (pads excluded) and decoder keys as the time mask allows."""
    rows, _, s, _ = time_mask.shape
    ll, b = num_context_patches, rows_per_block
    h, blocks = s - ll, rows // b
    dec = time_mask[:, 0, ll:]
    ctx = torch.zeros(rows, h, context_seq, dtype=torch.bool)
    ctx[..., :ll] = dec[..., :ll]
    own_row = torch.eye(b, dtype=torch.bool).reshape(1, b, 1, b, 1)

    def packed(keys):
        n = keys.shape[-1]
        return (keys.reshape(blocks, b, h, 1, n) & own_row).reshape(blocks, b * h, b * n)

    return torch.cat([packed(ctx), packed(dec[..., ll:])], dim=-1).unsqueeze(0)


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


# Per-shape device constants kept by TtTimestral; one call uses at most 7.
_MAX_CONSTANTS = 32


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
        n = len(weights.layers)
        self._layers = [
            TtTransformerBlock(device, layer, self.precision, last=i == n - 1) for i, layer in enumerate(weights.layers)
        ]
        self._head = TtResidualBlock(device, weights.head, self.precision)
        self._constants: OrderedDict[tuple, object] = OrderedDict()

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
        rows, seq = prepared.rows, prepared.seq_len
        ll, h = prepared.num_context_patches, prepared.horizon_patches
        rows_per_block = split_rows_per_block(rows, ll, h)
        if rows_per_block is not None:
            return self._upload_split(prepared, rows_per_block)
        padded_seq = pad_to_tile(seq)
        tokens = self._upload_tokens(prepared.tokens, padded_seq)
        time_mask = None
        if prepared.time_mask is not None:
            time_mask = self._upload_time_mask(prepared.time_mask, padded_seq)
        variate_mask = decoder_variate_mask = None
        if prepared.variate_mask is not None:
            slot_major = _additive(_pad_variate_mask(prepared.variate_mask, seq, padded_seq))
            variate_mask = self._upload_small_mask(slot_major)
            series = rows // prepared.group_size
            decoder_variate_mask = self._upload_small_mask(slot_major[ll * series : (ll + h) * series])
        owned = [tokens, time_mask, variate_mask, decoder_variate_mask]
        attention = self._stream_inputs(
            prepared, padded_seq, prepared.cos, prepared.sin, time_mask, variate_mask, owned
        )
        decoder_inputs = TtAttentionInputs(
            cos=None,
            sin=None,
            unit_stats=None,
            time_mask=None,
            variate_mask=decoder_variate_mask,
            rows=rows,
            padded_seq=h,
            group_size=prepared.group_size,
        )
        return TtTimestralDeviceInputs(
            tokens=tokens,
            attention=attention,
            decoder=DecoderSlots(start=ll, count=h, inputs=decoder_inputs),
            split=None,
            num_context_patches=ll,
            horizon_patches=h,
            owned=tuple(t for t in owned if t is not None),
        )

    def _upload_split(self, prepared: TimestralHostInputs, rows_per_block: int) -> TtTimestralDeviceInputs:
        rows, ll, h = prepared.rows, prepared.num_context_patches, prepared.horizon_patches
        context_seq = pad_to_tile(ll)
        tokens = self._upload_tokens(prepared.tokens[:, :ll], context_seq)
        owned = [tokens]
        time_mask = None
        if prepared.time_mask is None:
            cross_mask = self._causal_cross_mask(ll, h, rows_per_block)
        else:
            time_mask = self._upload_time_mask(prepared.time_mask[:, :, :ll, :ll], context_seq)
            cross_mask = upload(
                self.device, _additive(_cross_mask(prepared.time_mask, ll, context_seq, rows_per_block))
            )
            owned += [time_mask, cross_mask]
        context_variate = decoder_variate = None
        if prepared.variate_mask is not None:
            split_at = ll * rows // prepared.group_size
            context_variate = self._upload_small_mask(
                _additive(_pad_variate_mask(prepared.variate_mask[:split_at], ll, context_seq))
            )
            decoder_variate = self._upload_small_mask(_additive(prepared.variate_mask[split_at:]))
            owned += [context_variate, decoder_variate]
        cos, sin = prepared.cos, prepared.sin
        context = self._stream_inputs(
            prepared, context_seq, cos[:, :ll], sin[:, :ll], time_mask, context_variate, owned
        )
        decoder = self._stream_inputs(prepared, h, cos[:, ll:], sin[:, ll:], None, decoder_variate, owned)
        return TtTimestralDeviceInputs(
            tokens=tokens,
            attention=None,
            decoder=None,
            split=SplitStreams(context=context, decoder=decoder, cross_mask=cross_mask, rows_per_block=rows_per_block),
            num_context_patches=ll,
            horizon_patches=h,
            owned=tuple(owned),
        )

    def _constant(self, key: tuple, make):
        """A device tensor that depends only on ``key``, built once; the least recently used of more
        than ``_MAX_CONSTANTS`` is freed."""
        import ttnn

        if key in self._constants:
            self._constants.move_to_end(key)
            return self._constants[key]
        value = self._constants[key] = make()
        if len(self._constants) > _MAX_CONSTANTS:
            _, evicted = self._constants.popitem(last=False)
            for t in evicted if isinstance(evicted, tuple) else (evicted,):
                ttnn.deallocate(t)
        return value

    def _upload_tokens(self, tokens: Tensor, padded_seq: int):
        """Host ``(R, S, F)`` -> device tiles ``(1, 1, R * S_pad, F)``, zero tokens in the pad slots."""
        import ttnn

        rows, seq, features = tokens.shape
        padded = torch.zeros(rows, padded_seq, features)
        padded[:, :seq] = tokens
        # Tilizing on device is far cheaper than on the host for large batches.
        tokens_rm = upload(self.device, padded.reshape(1, 1, rows * padded_seq, -1), layout=ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.to_layout(tokens_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tokens_rm)
        return out

    def _upload_time_mask(self, mask: Tensor, padded_seq: int):
        # Time attention runs on an (H, R, S_pad, Dh) view, so the per-row masks sit on dim 1.
        return upload(self.device, _additive(_pad_time_mask(mask, padded_seq)).transpose(0, 1))

    def _causal_cross_mask(self, num_context_patches: int, horizon_patches: int, rows_per_block: int):
        """The all-valid ``_cross_mask``, one block broadcast over all of them."""

        def make():
            s = num_context_patches + horizon_patches
            causal = torch.tril(torch.ones(s, s, dtype=torch.bool)).expand(rows_per_block, 1, s, s)
            mask = _cross_mask(causal, num_context_patches, pad_to_tile(num_context_patches), rows_per_block)
            return upload(self.device, _additive(mask))

        return self._constant(("cross_mask", num_context_patches, horizon_patches, rows_per_block), make)

    def _stream_inputs(self, prepared, padded_seq, cos, sin, time_mask, variate_mask, owned: list):
        """Attention inputs of ``padded_seq`` token slots per row, RoPE tables ``cos`` / ``sin`` ``(1 or R, S, Dh)``;
        per-call uploads are appended to ``owned``."""
        import ttnn

        rows, tokens = prepared.rows, prepared.rows * padded_seq
        unit_stats = self._constant(
            ("unit_stats", tokens),
            lambda: ttnn.full(
                ttnn.Shape((1, 1, tokens, TILE)),
                self.config.d_model / TILE,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )
        return TtAttentionInputs(
            cos=self._rope(cos, rows, padded_seq, 1.0, owned),
            sin=self._rope(sin, rows, padded_seq, 0.0, owned),
            unit_stats=unit_stats,
            time_mask=time_mask,
            variate_mask=variate_mask,
            rows=rows,
            padded_seq=padded_seq,
            group_size=prepared.group_size,
        )

    def _rope(self, table: Tensor, rows: int, padded_seq: int, identity: float, owned: list):
        """Shared ``(1, S, Dh)`` tables are cached per positions; per-row tables go to ``owned``."""
        if table.shape[0] == 1:
            key = ("rope", rows, padded_seq, identity, tuple(table.shape), table.contiguous().numpy().tobytes())
            return self._constant(key, lambda: self._upload_rope(table, rows, padded_seq, identity))
        out = self._upload_rope(table, rows, padded_seq, identity)
        owned.append(out)
        return out

    def _upload_rope(self, table: Tensor, rows: int, padded_seq: int, identity: float):
        """``(1 or R, S, Dh)`` -> device ``(1, 1, R * S_pad, Dh)``, one row per token, in ``rope_pair_order``."""
        import ttnn

        padded = _pad_rope(rope_pair_order(table), padded_seq, identity)
        if padded.shape[0] != rows and padded_seq % TILE:
            # A device repeat would copy the tile padding between rows.
            padded = padded.expand(rows, -1, -1, -1)
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
        def make():
            keep, add = forecast_token_layout(self.weights.forecast_query, num_context_patches, horizon_patches)
            pad = padded_seq - keep.shape[1]
            keep = torch.nn.functional.pad(keep, (0, 0, 0, pad))
            add = torch.nn.functional.pad(add, (0, 0, 0, pad))
            return upload(self.device, keep), upload(self.device, add)

        return self._constant(("token_layout", num_context_patches, horizon_patches, padded_seq), make)

    def embed(self, inputs: TtTimestralDeviceInputs):
        """Patch tokens -> flat hidden states ``(1, 1, R * S_pad, D)``; decoder slots hold the forecast query.
        With split streams: ``(context (1, 1, R * L_pad, D), decoder (1, 1, R * h, D))``."""
        import ttnn

        if inputs.split is not None:
            decoder = inputs.split.decoder
            queries = self._forecast_query_rows(decoder.rows * decoder.padded_seq)
            return self._patch_proj(inputs.tokens), ttnn.clone(queries, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        att = inputs.attention
        r, sp, d = att.rows, att.padded_seq, self.config.d_model
        embedded = ttnn.reshape(self._patch_proj(inputs.tokens), (r, sp, d))
        keep, add = self._token_layout(inputs.num_context_patches, inputs.horizon_patches, sp)
        kept = ttnn.mul(embedded, keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(embedded)
        x = ttnn.add(kept, add, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(kept)
        return ttnn.reshape(x, (1, 1, r * sp, d))

    def _forecast_query_rows(self, tokens: int):
        """Device ``(1, 1, tokens, D)`` of the forecast query."""
        query = self.weights.forecast_query.to(torch.float32)
        return self._constant(("forecast_query", tokens), lambda: upload(self.device, query.expand(1, 1, tokens, -1)))

    def run_layer(self, layer: TtTransformerBlock, x, inputs: TtTimestralDeviceInputs, *, last: bool = False):
        """One block on the residual stream, flat ``x`` or ``(context, decoder)`` with split streams;
        consumes ``x``. ``last`` returns only the decoder slots, ``(1, 1, R * h, D)``."""
        if inputs.split is None:
            return layer(x, inputs.attention, decoder=inputs.decoder if last else None)
        context, decoder = layer.split(*x, inputs.split, last=last)
        return decoder if last else (context, decoder)

    def head(self, x):
        """Decoder-slot hidden states ``(1, 1, R * h, D)`` -> head output ``(1, 1, R * h, patch_size * Q)``; consumes ``x``."""
        import ttnn

        normed = rms_norm(x, self.weights.out_norm_eps, self.precision)
        ttnn.deallocate(x)
        out = self._head(normed)
        ttnn.deallocate(normed)
        return out

    def forward_device(self, inputs: TtTimestralDeviceInputs):
        """Device inputs -> device head output ``(1, 1, R * h, patch_size * Q)`` in normalized space."""
        x = self.embed(inputs)
        *body, last = self._layers
        for layer in body:
            x = self.run_layer(layer, x, inputs)
        return self.head(self.run_layer(last, x, inputs, last=True))

    def postprocess(self, output_device, prepared: TimestralHostInputs) -> Tensor:
        """Download the head output -> raw-space quantiles ``(R, h * patch_size, Q)``."""
        import ttnn

        raw = ttnn.to_torch(output_device).float().reshape(prepared.rows, prepared.horizon_patches, -1)
        return postprocess_head(self.config, raw, prepared)

    @staticmethod
    def deallocate_inputs(inputs: TtTimestralDeviceInputs) -> None:
        import ttnn

        for t in inputs.owned:
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
