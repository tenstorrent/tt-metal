# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill Engram: the n-gram memory written into the residual streams before layers 1 and 14.

The multi-token counterpart of :class:`~..decode.engram.DeepSeekV41Engram`, with the same split: the table
lookup depends only on token ids, so the host does it
(:class:`~models.experimental.deepseek_v41_flash.tt.engram_lookup.EngramHostLookup`, called with a chunk's
ids and start position) and :meth:`DeepSeekV41PrefillEngram.forward` takes the chunk's rows ``[1, 1, T, kin]``
as a device tensor. The device does the rest of the checkpoint's ``Engram.forward``::

    kv    = rows @ wkv^T                                   -> key [1, hc, T, D], value [1, 1, T, D]
    rstd  = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)       per (hc copy, token), over D
    dot   = sum(h * q_weight * k_weight * key) * rstd / sqrt(D)
    gate  = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))
    out   = h + gate * value

Decode runs everything after ``kv`` as one ``engram_gate`` op on width-sharded rows built for a few tokens; a
prefill chunk is thousands of rows, so here ``wkv`` is a plain ``ttnn.linear`` and everything after it is one
``ttnn.experimental.deepseek.engram_gate_prefill`` call on DRAM-interleaved TILE operands -- no slice/concat, no
fp32 copy of the streams. The streams are stream-major ``[1, hc, T, D]``, so the key's ``hc`` copies are column
slices of ``kv``, taken by the op itself: no tile padding, no ROW_MAJOR round trips.
"""

from typing import Optional

import torch
import ttnn

from models.experimental.deepseek_v4_flash.tt.common import DeepSeekV4Module
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache, _as_cache, _load_weight, _materialize

# Floor of |dot| before the signed sqrt (``Engram.clamp_value``).
_CLAMP = 1e-6


def _value(weight) -> torch.Tensor:
    return weight() if callable(weight) else weight


class DeepSeekV41PrefillEngram(DeepSeekV4Module):
    """One Engram layer's device half, over a chunk of tokens."""

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
    ):
        """``weights`` is :func:`~models.experimental.deepseek_v41_flash.tt.config.engram_weights` (tensors or
        thunks). ``wkv`` is ``weight_dtype`` (a ``[6144, 25600]`` projection, ~157 MB at bfloat8_b); the gate weight
        stays fp32. Under a TP mesh every rank holds the same copy, as the streams are replicated.
        """
        cache = _as_cache(cache)
        self.layer_idx = layer_idx
        self.device = device
        self.hc, self.dim, self.eps = config.hc_mult, config.hidden_size, config.rms_norm_eps
        self.kin = (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim
        replicate = ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None
        self.wkv = Linear(
            weights["wkv.weight"],
            device,
            cache.file("wkv.prefill"),
            dtype=weight_dtype,
            mesh_mapper=replicate,
        )
        # q_weight and k_weight only ever appear as a product, with the 1/sqrt(D) of ``dot``. The fused gate op takes
        # one [32, D] page per stream copy, every row holding that copy's vector, so it is a plain tile operand
        # against a 32-token block: [1, hc, 32, D].
        gate_file = cache.file("gate_weight.prefill_op")
        gate_weight = _materialize(
            lambda: (_value(weights["q_weight"]).float() * _value(weights["k_weight"]).float() * self.dim**-0.5)
            .repeat(ttnn.TILE_SIZE, 1)
            .reshape(1, self.hc, ttnn.TILE_SIZE, self.dim),
            gate_file,
            ttnn.float32,
        )
        self.gate_weight = _load_weight(
            gate_weight,
            device,
            cache_file_name=gate_file,
            dtype=ttnn.float32,
            mesh_mapper=replicate,
        )

    def forward(self, streams: ttnn.Tensor, rows: ttnn.Tensor) -> ttnn.Tensor:
        """``streams`` ``[1, hc, T, D]`` bf16 TILE and the chunk's host-looked-up ``rows`` ``[1, 1, T, kin]`` bf16
        TILE -> the streams with the lookup written in: ``[1, hc, T, D]`` bf16 TILE. Both inputs are left intact."""
        hc, d = self.hc, self.dim
        tokens = streams.shape[2]
        if tuple(streams.shape) != (1, hc, tokens, d):
            raise ValueError(f"expected streams [1, {hc}, T, {d}], got {tuple(streams.shape)}")
        if tuple(rows.shape) != (1, 1, tokens, self.kin):
            raise ValueError(f"expected rows [1, 1, {tokens}, {self.kin}], got {tuple(rows.shape)}")

        kv = self.wkv(rows)  # [1, 1, T, (hc + 1) * D]
        streams_out = ttnn.experimental.deepseek.engram_gate_prefill(
            streams, kv, self.gate_weight, self.eps, clamp_value=_CLAMP
        )
        ttnn.deallocate(kv)
        return streams_out
