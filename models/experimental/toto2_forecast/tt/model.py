# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN Toto 2.0: one forecast forward (``decode_block_size=None``) on one chip.

reference : toto2/toto2/model.py @ 9213d2c (``Toto2Model.forecast``), folded by ``tt/weights.py``
    host:   causal per-patch scaler (float64) + backfill, asinh patch tokens, group ids, masks, RoPE + xPos tables
            (``tt/preprocessing.py``)
    device: x = patch_proj(tokens)                              rows of S_pad = whole tiles, pads after the real tokens
            x = time layers 0 .. L-2 (x)                         causal, or the group mask when patches are missing
                the last one cuts to the read-out tokens after its attention: nothing later mixes token slots
            x = variate layer (x)                                V = 1 shortcut, else SDPA across the variates
            out = head(RMSNorm(x))                               (R * h, 288), asinh-normalized
    host:   sinh, rescale, clamp, sort (``tt/readout.py``)

Host / device split for the benchmarks: ``prepare_inputs`` -> ``upload_inputs`` -> ``forward_device`` ->
``postprocess``, then ``deallocate_inputs``.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor

from models.experimental.toto2_forecast.tt.attention import (
    TimeInputs,
    TtTimeAttention,
    TtVariateAttention,
    VariateInputs,
)
from models.experimental.toto2_forecast.tt.common import Toto2Precision, pad_to_tile, rms_norm, take_slots, upload
from models.experimental.toto2_forecast.tt.mlp import TtResidualMLP, TtSwiGLU
from models.experimental.toto2_forecast.tt.positions import pair_rotation
from models.experimental.toto2_forecast.tt.preprocessing import ForecastOptions, Toto2HostInputs, prepare_inputs
from models.experimental.toto2_forecast.tt.readout import postprocess
from models.experimental.toto2_forecast.tt.weights import Toto2Weights

_MAX_CONSTANTS = 16


@dataclass(frozen=True)
class Toto2DeviceInputs:
    tokens: object  # flat (1, 1, R * S_pad, 64)
    time: TimeInputs
    variate: VariateInputs
    readout: tuple[int, int]  # (start, count) token slots the head reads
    owned: tuple  # the per-call tensors deallocate_inputs frees (not the per-shape caches)


def _additive(mask: Tensor) -> Tensor:
    """Boolean attend-mask -> additive SDPA mask (0 / -inf)."""
    return torch.zeros(mask.shape).masked_fill(~mask, float("-inf"))


def _pad_time_mask(mask: Tensor, padded_seq: int) -> Tensor:
    """``(R, S, S)`` -> ``(R, S_pad, S_pad)``: real queries never read pad keys; pad queries attend causally."""
    rows, s, _ = mask.shape
    full = torch.tril(torch.ones(padded_seq, padded_seq, dtype=torch.bool)).expand(rows, -1, -1).clone()
    full[:, :s, :] = False
    full[:, :s, :s] = mask
    return full


def _pad_variate_mask(mask: Tensor) -> Tensor:
    """``(N, 1, V, V)`` -> ``(N, 1, V_pad, V_pad)`` with the pad keys hidden from every query; pad queries see the
    real keys (their rows are dropped)."""
    n, _, v, _ = mask.shape
    vp = pad_to_tile(v)
    full = torch.zeros(n, 1, vp, vp, dtype=torch.bool)
    full[..., :v] = True
    full[..., :v, :v] = mask
    return full


class TtToto2:
    """Device Toto 2.0 (single chip). Weights move host -> device once in ``__init__``."""

    def __init__(
        self, device, weights: Toto2Weights, precision: Toto2Precision | None = None, fold_betas: bool | None = None
    ):
        if fold_betas is None:  # R2-1 screening switch
            fold_betas = os.environ.get("TOTO2_FOLD_BETAS", "0") == "1"
        if fold_betas:
            weights = weights.fold_residual_betas()
        self.device = device
        self.weights = weights
        self.config = c = weights.config
        self.precision = p = precision or Toto2Precision.accuracy()
        self._patch_proj = TtResidualMLP(device, weights.patch_proj, p.for_role("embed"))
        self._layers = []
        for lw in weights.layers:
            attn = (TtVariateAttention if lw.is_variate else TtTimeAttention)(device, lw, c, p)
            self._layers.append((lw.is_variate, attn, TtSwiGLU(device, lw, c, p)))
        kinds = [v for v, _, _ in self._layers]
        self._last_time = max(i for i, v in enumerate(kinds) if not v)
        assert all(kinds[self._last_time + 1 :]), "variate layers must follow the last time layer"
        import ttnn

        self._head = TtResidualMLP(
            device, weights.head, p.for_role("head"), out_dtype=ttnn.float32 if p.fp32_head_out else None
        )
        self._head_eps = c.norm_eps if weights.head_eps is None else weights.head_eps
        self._constants: OrderedDict[tuple, object] = OrderedDict()

    @classmethod
    def from_checkpoint(cls, device, name: str, precision: Toto2Precision | None = None) -> "TtToto2":
        return cls(device, Toto2Weights.from_checkpoint(name), precision)

    @property
    def layer_kinds(self) -> list[str]:
        """``time`` / ``last_time`` / ``variate`` per layer."""
        return [
            "variate" if v else ("last_time" if i == self._last_time else "time")
            for i, (v, _, _) in enumerate(self._layers)
        ]

    # ---- host / device split -------------------------------------------------------------------------------

    def prepare_inputs(
        self,
        target: Tensor,
        target_mask: Tensor,
        series_ids: Tensor,
        horizon: int,
        options: ForecastOptions = ForecastOptions(),
    ) -> Toto2HostInputs:
        c = self.config
        return prepare_inputs(
            target, target_mask, series_ids, horizon, patch_size=c.patch_size, head_dim=c.head_dim, options=options
        )

    def upload_inputs(self, host: Toto2HostInputs) -> Toto2DeviceInputs:
        import ttnn

        r, s = host.rows, host.seq_len
        sp = pad_to_tile(s)
        start, count = host.readout
        tokens = torch.zeros(r, sp, host.tokens.shape[-1])
        tokens[:, :s] = host.tokens
        # Tilizing on device is far cheaper than on the host for large batches.
        token_dtype = ttnn.float32 if self.precision.fp32_tokens else ttnn.bfloat16
        tokens_rm = upload(
            self.device, tokens.reshape(1, 1, r * sp, -1), dtype=token_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        tokens_dev = ttnn.to_layout(tokens_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tokens_rm)
        owned = [tokens_dev]
        time_mask = None
        if host.time_mask is not None:
            time_mask = upload(self.device, _additive(_pad_time_mask(host.time_mask, sp)).unsqueeze(1))
            owned.append(time_mask)
        variate_mask = None
        if host.variate_mask is not None:
            m = host.variate_mask[:, start : start + count].reshape(-1, 1, host.variates, host.variates)
            variate_mask = upload(self.device, _additive(_pad_variate_mask(m)))
            owned.append(variate_mask)
        cos_q, sin_q, cos_k, sin_k = self._rope_tables(host, sp)
        time = TimeInputs(
            rows=r,
            padded_seq=sp,
            cos_q=cos_q,
            sin_q=sin_q,
            cos_k=cos_k,
            sin_k=sin_k,
            trans_mat=self._trans_mat(),
            mask=time_mask,
        )
        variate = VariateInputs(batch=host.batch, variates=host.variates, slots=count, mask=variate_mask)
        return Toto2DeviceInputs(
            tokens=tokens_dev, time=time, variate=variate, readout=(start, count), owned=tuple(owned)
        )

    def forward_device(self, inputs: Toto2DeviceInputs):
        """Device inputs -> device head output ``(1, 1, R * h, 288)`` in asinh-normalized space."""
        x = self.embed(inputs)
        for i in range(len(self._layers)):
            x = self.run_layer(i, x, inputs)
        return self.head(x)

    def postprocess(self, out_device, host: Toto2HostInputs, *, normalized: bool = False):
        """Download the head output -> quantiles ``(9, B, V, H)``; with ``normalized`` also return the head output
        ``(9, B, V, h, 32)``."""
        import ttnn

        head = self.normalized_output(ttnn.to_torch(out_device).float(), host)
        q = postprocess(head, host.loc, host.scale, host.caps, host.horizon)
        return (q, head) if normalized else q

    def normalized_output(self, raw: Tensor, host: Toto2HostInputs) -> Tensor:
        """Head output ``(..., R * h, 288)`` -> ``(9, B, V, h, 32)``."""
        c = self.config
        out = raw.reshape(host.batch, host.variates, host.horizon_patches, c.patch_size, c.num_quantiles)
        return out.permute(4, 0, 1, 2, 3)

    @staticmethod
    def deallocate_inputs(inputs: Toto2DeviceInputs) -> None:
        import ttnn

        for t in inputs.owned:
            ttnn.deallocate(t)

    def forecast(
        self,
        target: Tensor,
        target_mask: Tensor,
        series_ids: Tensor,
        horizon: int,
        options: ForecastOptions = ForecastOptions(),
    ) -> Tensor:
        """``Toto2Model.forecast(..., decode_block_size=None)`` on device: quantiles ``(9, B, V, H)``."""
        import ttnn

        host = self.prepare_inputs(target, target_mask, series_ids, horizon, options)
        inputs = self.upload_inputs(host)
        try:
            out = self.forward_device(inputs)
            q = self.postprocess(out, host)
            ttnn.deallocate(out)
        finally:
            self.deallocate_inputs(inputs)
        return q

    # ---- sections (the profile signposts) -------------------------------------------------------------------

    def embed(self, inputs: Toto2DeviceInputs):
        """Patch tokens -> flat hidden states ``(1, 1, R * S_pad, d)``."""
        return self._patch_proj(inputs.tokens)

    def run_layer(self, i: int, x, inputs: Toto2DeviceInputs):
        """Layer ``i`` on the residual stream; consumes ``x``. The last time layer returns only the read-out slots,
        ``(1, 1, R * h, d)``."""
        import ttnn

        is_variate, attn, mlp = self._layers[i]
        if is_variate:
            x = attn(x, inputs.variate)
        else:
            x = attn(x, inputs.time)
            if i == self._last_time:
                start, count = inputs.readout
                y = take_slots(x, inputs.time.rows, inputs.time.padded_seq, start, count)
                ttnn.deallocate(x)
                x = y
        return mlp(x)

    def head(self, x):
        """Read-out hidden states ``(1, 1, R * h, d)`` -> ``(1, 1, R * h, 288)``; consumes ``x``."""
        import ttnn

        normed = rms_norm(x, self._head_eps, self.precision)
        ttnn.deallocate(x)
        out = self._head(normed)
        ttnn.deallocate(normed)
        return out

    # ---- per-shape device constants -------------------------------------------------------------------------

    def _constant(self, key: tuple, make):
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

    def _trans_mat(self):
        return self._constant(("trans_mat",), lambda: upload(self.device, pair_rotation(32).reshape(1, 1, 32, 32)))

    def _rope_tables(self, host: Toto2HostInputs, padded_seq: int):
        s = host.seq_len

        def make():
            out = []
            for t, identity in (
                (host.rope.cos_q, 1.0),
                (host.rope.sin_q, 0.0),
                (host.rope.cos_k, 1.0),
                (host.rope.sin_k, 0.0),
            ):
                padded = torch.full((padded_seq, t.shape[-1]), identity)
                padded[:s] = t
                out.append(upload(self.device, padded.reshape(1, 1, padded_seq, -1)))
            return tuple(out)

        return self._constant(("rope", s, padded_seq), make)
