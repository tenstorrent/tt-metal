# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN t0-gamma: one forward of patch encoder -> 24 layers ([time, time, group] x 8) -> decoder -> monotone head.

reference : t0/forecaster.py (``_single_pass``, ``_backbone``) and t0/torch/model/model.py (``T0Forecaster``)
    host:   rollout buffer, causal per-step scaler + asinh, 96-wide patch tokens, masks, xPos RoPE tables
    device: x = patch_encoder(tokens) (+ type embedding)        # target rows: type_emb[0] folded in the bias
            x = layers(x)                                      # tokens padded per row to a whole tile; after the
                                                               # last time attention, the read-out patches only
            q = monotone(decoder(RMSNorm(x) * out_norm))       # [raw_0, softplus(raw_1..)] @ kron(I, U)
    host:   sinh(q) * scale + loc, the requested levels, the adjustments (``host.predict``)

Rows pad from S patches to ``S_pad`` (66 -> 96); causal time attention never lets a real patch read a pad
slot, and every other op is per token. ``tests/golden/dataflow.py`` is the fp32 spec of this decomposition.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from models.experimental.t0_forecast.tt import host
from models.experimental.t0_forecast.tt.host import monotone_matrix
from models.experimental.t0_forecast.tt.attention import TtAttention, TtAttentionInputs
from models.experimental.t0_forecast.tt.common import (
    TtT0Precision,
    pad_to_tile,
    rms_norm,
    upload,
    upload_norm,
)
from models.experimental.t0_forecast.tt.mlp import TtSwiGLU
from models.experimental.t0_forecast.tt.residual_block import TtResidualBlock
from models.experimental.t0_forecast.tt.weights import TIME, T0Weights

# Per-shape device constants kept by TtT0 (RoPE tables); one call uses 4.
_MAX_CONSTANTS = 16


@dataclass(frozen=True)
class TtT0DeviceInputs:
    tokens: object  # (1, 1, R * S_pad, 96)
    type_delta: object | None  # (1, 1, R * S_pad, d): type_emb[t] - type_emb[0]; None when every patch is a target
    attention: TtAttentionInputs
    readout_attention: TtAttentionInputs  # the layers after the cut, on the read-out patches only
    readout_start: int
    readout_count: int
    owned: tuple  # what deallocate_inputs frees (not the cached constants)


def _additive(mask: Tensor) -> Tensor:
    """Boolean attend-mask -> additive SDPA mask (0 / -inf)."""
    return torch.zeros(mask.shape).masked_fill(~mask, float("-inf"))


def _pad_time_mask(mask: Tensor, padded_seq: int) -> Tensor:
    """``(R, S, S)`` -> ``(R, 1, S_pad, S_pad)``: real queries never read pad keys; pad queries attend causally."""
    rows, s, _ = mask.shape
    full = torch.tril(torch.ones(padded_seq, padded_seq, dtype=torch.bool)).expand(rows, -1, -1).clone()
    full[:, :s, :] = False
    full[:, :s, :s] = mask
    return full.unsqueeze(1)


def _pad_group_mask(mask: Tensor, padded_seq: int) -> Tensor:
    """``(S, B, G, G)`` -> ``(S_pad * B, 1, G, G)``; pad slots see their whole series."""
    s, b, g, _ = mask.shape
    full = torch.ones(padded_seq, b, g, g, dtype=torch.bool)
    full[:s] = mask
    return full.reshape(padded_seq * b, 1, g, g)


def _packed_group_mask(mask: Tensor | None, padded_seq: int, rows: int, group_size: int) -> Tensor:
    """Group attention over a series' ``G * S_pad`` tokens as they lie (row-major ``(g, s)``): token ``(g1, s1)``
    reads ``(g2, s2)`` iff ``s1 == s2`` and ``mask[s, b, g1, g2]`` (pad slots see their whole series).
    ``(1, 1, G S_pad, G S_pad)`` without a mask, else ``(B, 1, G S_pad, G S_pad)``."""
    g, sp = group_size, padded_seq
    if mask is None:
        full = torch.ones(sp, 1, g, g, dtype=torch.bool)
    else:
        full = _pad_group_mask(mask, sp).reshape(sp, rows // g, g, g)
    blocks = torch.diag_embed(full.permute(1, 2, 3, 0))  # (B, G1, G2, S1, S2), true only on S1 == S2
    return blocks.permute(0, 1, 3, 2, 4).reshape(-1, 1, g * sp, g * sp)


class TtT0:
    """Device t0 model (single chip). Weights move host -> device once in ``__init__``."""

    def __init__(self, device, weights: T0Weights, precision: TtT0Precision | None = None):
        self.device = device
        self.weights = weights
        self.config = weights.config
        self.precision = precision or TtT0Precision.accuracy()
        self._embed = TtResidualBlock(
            device, weights.patch_encoder, self.precision, role="embed", extra_bias=weights.type_embeddings[0]
        )
        self._type_embeddings = weights.type_embeddings
        kinds = [w.kind for w in weights.layers]
        # Everything after the last time attention is per token (or per patch across a series' rows), so the stack
        # is cut to the read-out patches there (the fp32 spec ``tests/golden/dataflow.py`` does the same).
        self._cut = max((i for i, k in enumerate(kinds) if k == TIME), default=len(kinds) - 1)
        self._layers = [
            (TtAttention(device, w.attention, self.precision), TtSwiGLU(device, w.mlp, self.precision))
            for w in weights.layers
        ]
        self._out_norm = upload_norm(device, weights.out_norm)
        import ttnn

        # The head runs in fp32: in bf16, softplus(x) is 0 below x = -5 (1 + exp(x) rounds to 1), which merges
        # neighbouring quantiles whose increments are small.
        self._decoder = TtResidualBlock(device, weights.decoder, self.precision, role="decoder", out_dtype=ttnn.float32)
        first, cumulative = monotone_matrix(self.config.n_quantiles, self.config.patch_size)
        self._first = upload(device, first.reshape(1, -1), dtype=ttnn.float32)
        self._cumulative = upload(device, cumulative, dtype=ttnn.float32)
        self._constants: OrderedDict[tuple, object] = OrderedDict()

    @classmethod
    def from_pretrained(cls, device, precision: TtT0Precision | None = None) -> "TtT0":
        return cls(device, T0Weights.load(), precision)

    @property
    def layers(self):
        return self._layers

    @property
    def cut_index(self) -> int:
        """Index of the layer whose attention output is cut to the read-out patches."""
        return self._cut

    # --- host <-> device -----------------------------------------------------------------------------------

    def prepare_inputs(self, context, horizon: int, *, mask=None) -> host.T0HostInputs:
        """Array context ``[B, T]`` / ``[B, V, T]`` -> one forward's host inputs (scaler, patching, masks, RoPE)."""
        series, batch_shape, n, ctx = host.series_from_array(context, mask)
        prepared = host.prepare_series(self.config, series, horizon, ctx, range(n))
        return replace(prepared, batch_shape=batch_shape, prediction_length=horizon)

    def _constant(self, key: tuple, make):
        """A device tensor that depends only on ``key``, built once. Past ``_MAX_CONSTANTS`` entries the least
        recently used one leaves the cache but is not deallocated: device inputs that hold it keep it alive, and
        it is freed with its last reference (``test_rope_cache_eviction``)."""
        if key in self._constants:
            self._constants.move_to_end(key)
            return self._constants[key]
        value = self._constants[key] = make()
        if len(self._constants) > _MAX_CONSTANTS:
            self._constants.popitem(last=False)
        return value

    def _rope(self, prepared: host.T0HostInputs, padded_seq: int):
        """The four tables tiled over rows and heads, ``(1, 1, R * H * S_pad, hd)``: the fused QK-norm + RoPE
        reads one table row per token row. Pad positions rotate by the identity. Cached per (R, S)."""
        s, r, h = prepared.num_patches, prepared.rows, self.config.num_heads

        def make(i: int):
            def build():
                identity = 1.0 if i % 2 == 0 else 0.0  # cos tables pad with 1, sin tables with 0
                table = torch.full((padded_seq, self.config.head_dim), identity)
                table[:s] = prepared.rope[i]
                tiled = table.to(torch.bfloat16).expand(r * h, -1, -1).reshape(1, 1, r * h * padded_seq, -1)
                return upload(self.device, tiled)

            return build

        return tuple(self._constant(("rope", r, s, padded_seq, i), make(i)) for i in range(4))

    def _upload_small_mask(self, mask: Tensor):
        """A mask whose last dims are not tile-aligned: the tile padding must be -inf, not 0."""
        import ttnn

        rows = upload(self.device, mask, layout=ttnn.ROW_MAJOR_LAYOUT)
        padded = [*mask.shape[:-2], *(pad_to_tile(n) for n in mask.shape[-2:])]
        out = ttnn.tilize_with_val_padding(rows, padded, float("-inf"), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rows)
        return out

    def upload_inputs(self, prepared: host.T0HostInputs) -> TtT0DeviceInputs:
        import ttnn

        r, s, d = prepared.rows, prepared.num_patches, self.config.embed_dim
        sp = pad_to_tile(s)
        tokens = torch.zeros(r, sp, prepared.tokens.shape[-1])
        tokens[:, :s] = prepared.tokens
        tokens_rm = upload(self.device, tokens.reshape(1, 1, r * sp, -1), layout=ttnn.ROW_MAJOR_LAYOUT)
        tokens_dev = ttnn.to_layout(tokens_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tokens_rm)
        owned = [tokens_dev]
        type_delta = None
        if bool((prepared.patch_type != 0).any()):
            delta = torch.zeros(r, sp, d)
            delta[:, :s] = self._type_embeddings[prepared.patch_type] - self._type_embeddings[0]
            type_delta = upload(self.device, delta.reshape(1, 1, r * sp, d))
            owned.append(type_delta)
        time_mask = None
        if prepared.time_mask is not None:
            time_mask = upload(self.device, _additive(_pad_time_mask(prepared.time_mask, sp)))
            owned.append(time_mask)
        readout_group_mask = packed_group_mask = None
        if prepared.group_size > 1:
            packed = _packed_group_mask(prepared.group_mask, sp, r, prepared.group_size)
            packed_group_mask = upload(self.device, _additive(packed))
            owned.append(packed_group_mask)
        if prepared.group_mask is not None:
            lo, n = prepared.readout_start, prepared.readout_count
            readout_group_mask = self._upload_small_mask(
                _additive(_pad_group_mask(prepared.group_mask[lo : lo + n], n))
            )
            owned.append(readout_group_mask)
        attention = TtAttentionInputs(
            rows=r,
            padded_seq=sp,
            group_size=prepared.group_size,
            rope=self._rope(prepared, sp),
            time_mask=time_mask,
            group_mask=None,
            packed_group_mask=packed_group_mask,
        )
        readout_attention = TtAttentionInputs(
            rows=r,
            padded_seq=prepared.readout_count,
            group_size=prepared.group_size,
            rope=None,
            time_mask=None,
            group_mask=readout_group_mask,
        )
        return TtT0DeviceInputs(
            tokens=tokens_dev,
            type_delta=type_delta,
            attention=attention,
            readout_attention=readout_attention,
            readout_start=prepared.readout_start,
            readout_count=prepared.readout_count,
            owned=tuple(owned),
        )

    @staticmethod
    def deallocate_inputs(inputs: TtT0DeviceInputs) -> None:
        import ttnn

        for t in inputs.owned:
            ttnn.deallocate(t)

    # --- the network ---------------------------------------------------------------------------------------

    def embed(self, inputs: TtT0DeviceInputs):
        """Tokens -> flat hidden states ``(1, 1, R * S_pad, d)``."""
        import ttnn

        x = self._embed(inputs.tokens)
        if inputs.type_delta is not None:
            y = ttnn.add(x, inputs.type_delta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x)
            x = y
        if self.precision.fp32_residual:
            wide = ttnn.typecast(x, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x)
            x = wide
        return x

    def readout(self, x, inputs: TtT0DeviceInputs):
        """Flat ``(1, 1, R * S_pad, d)`` -> the read-out patches ``(1, 1, R * n, d)``; ``x`` is borrowed."""
        import ttnn

        att, d = inputs.attention, self.config.embed_dim
        lo, n = inputs.readout_start, inputs.readout_count
        rows = ttnn.reshape(x, (att.rows, att.padded_seq, d))
        picked = ttnn.slice(rows, (0, lo, 0), (att.rows, lo + n, d), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(picked, (1, 1, att.rows * n, d))

    def run_layer(self, index: int, x, inputs: TtT0DeviceInputs):
        """Layer ``index`` on the residual stream; consumes ``x``. At the cut layer the attention output is cut to the
        read-out patches before the MLP; later layers run on those patches only."""
        import ttnn

        attention, mlp = self._layers[index]
        y = attention(x, inputs.attention if index <= self._cut else inputs.readout_attention)
        ttnn.deallocate(x)
        if index == self._cut:
            picked = self.readout(y, inputs)
            ttnn.deallocate(y)
            y = picked
        z = mlp(y)
        ttnn.deallocate(y)
        return z

    def head(self, x):
        """Read-out hidden states ``(1, 1, R * n, d)`` -> fp32 ``(1, 1, R * n, ps * Q)`` quantiles in asinh space;
        consumes ``x``. The decoder emits fp32 and the monotone head stays in fp32."""
        return self.monotone(self.decode(x))

    def decode(self, x):
        """``out_norm`` then the decoder: the raw head values before softplus, fp32 ``(1, 1, R * n, ps * Q)``;
        consumes ``x``."""
        import ttnn

        xn = rms_norm(x, self._out_norm, 1e-8, self.precision)
        ttnn.deallocate(x)
        raw = self._decoder(xn)
        ttnn.deallocate(xn)
        return raw

    def monotone(self, raw):
        """``q = [raw_0, softplus(raw_1..)] @ kron(I, U)`` per step, in fp32; consumes ``raw``."""
        import ttnn

        p = self.precision
        soft = ttnn.softplus(raw, beta=1.0, threshold=20.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        keep = ttnn.mul(ttnn.subtract(raw, soft), self._first, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(raw)
        y = ttnn.add(soft, keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(soft)
        ttnn.deallocate(keep)
        q = ttnn.linear(
            y,
            self._cumulative,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=p.exact_kernel_config(),
        )
        ttnn.deallocate(y)
        return q

    def forward_device(self, inputs: TtT0DeviceInputs):
        """Device inputs -> device head output ``(1, 1, R * n, ps * Q)`` (asinh space)."""
        x = self.embed(inputs)
        for index in range(len(self._layers)):
            x = self.run_layer(index, x, inputs)
        return self.head(x)

    def normalized(self, output_device, prepared: host.T0HostInputs) -> Tensor:
        """Download the head output -> ``[R, n, ps, Q]`` fp32 (asinh space)."""
        import ttnn

        c = self.config
        out = ttnn.to_torch(output_device).float()
        return out.reshape(prepared.rows, prepared.readout_count, c.patch_size, c.n_quantiles)

    def postprocess(self, output_device, prepared: host.T0HostInputs) -> Tensor:
        """Download -> data-space quantiles at the trained levels, ``[B, horizon, Q]`` or ``[B, V, horizon, Q]``."""
        out = host.sanitize(host.rescale(self.config, self.normalized(output_device, prepared), prepared))
        out = out[:, : prepared.prediction_length]
        if prepared.batch_shape is not None:
            out = out.reshape(*prepared.batch_shape, *out.shape[1:])
        return out

    def network(self, prepared: host.T0HostInputs) -> Tensor:
        """``host.Network`` on device: host inputs -> ``[R, n, ps, Q]`` (asinh space)."""
        import ttnn

        inputs = self.upload_inputs(prepared)
        try:
            out = self.forward_device(inputs)
            result = self.normalized(out, prepared)
            ttnn.deallocate(out)
        finally:
            self.deallocate_inputs(inputs)
        return result

    def predict(self, context, horizon: int, quantile_levels=(0.1, 0.5, 0.9), **adjustments) -> Tensor:
        """``T0Forecaster.predict`` for array inputs, with the network on device (see ``host.predict``)."""
        return host.predict(self.config, self.network, context, horizon, quantile_levels, **adjustments)
