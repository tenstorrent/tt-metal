# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The model terminal around the decoder blocks, in the blocks' ``[B, S, hc, D]`` stream layout.

* :class:`GLMEmbedding` -- ``embed_tokens`` then the ``hc_mult`` identical streams ``Glm5NextTextModel`` starts from
  (``inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1)``).
* :class:`GLMHead` -- ``Glm5NextTextHyperHead`` (the unweighted mean of the streams), the final
  ``Glm5NextTextRMSNorm`` and the untied ``lm_head``.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn

from models.experimental.glm53_flash.tt.ds_v4.common import DeepSeekV4Module, _HIFI4
from models.experimental.glm53_flash.tt.ds_v4.layers import Linear
from models.experimental.glm53_flash.tt.ds_v4.streams import load_norm_gamma, rows_from_tokens
from models.experimental.glm53_flash.tt.ds_v4.weight_cache import WeightCache, _as_cache, _load_weight, _materialize


class GLMEmbedding(DeepSeekV4Module):
    """``weight``: the ``[V, D]`` ``embed_tokens.weight`` (tensor or thunk), kept bf16 ROW_MAJOR in DRAM."""

    def __init__(self, config, weight, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.hc = config.hc_mult
        cache = _as_cache(cache)
        file = cache.file("embed_tokens")
        table = _materialize(weight, file, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.weight = _load_weight(
            table.detach().float() if table is not None else None,
            device,
            cache_file_name=file,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def token_ids(self, ids: torch.Tensor) -> ttnn.Tensor:
        """Host ``[B, S]`` ids -> the ``[B, S]`` uint32 ROW_MAJOR device tensor :meth:`forward` takes."""
        return ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """``input_ids`` ``[B, S]`` uint32 ROW_MAJOR -> streams ``[B, S, hc, D]`` bf16 TILE (``hc`` equal copies)."""
        b, s = input_ids.shape[0], input_ids.shape[1]
        emb = ttnn.embedding(input_ids, self.weight, layout=ttnn.ROW_MAJOR_LAYOUT)  # [B, S, D]
        d = emb.shape[-1]
        emb = ttnn.reshape(emb, [b, s, 1, d])
        streams = ttnn.repeat(emb, ttnn.Shape([1, 1, self.hc, 1]))
        ttnn.deallocate(emb)
        return ttnn.to_layout(streams, ttnn.TILE_LAYOUT)


class GLMHead(DeepSeekV4Module):
    """``weights``: ``norm.weight`` ``[D]`` and ``lm_head.weight`` ``[V, D]`` (tensors or thunks)."""

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
    ):
        cache = _as_cache(cache)
        self.eps = config.rms_norm_eps
        self.vocab = config.vocab_size
        self.norm_weight = load_norm_gamma(weights["norm.weight"], device, cache.file("norm"))
        self.lm_head = Linear(weights["lm_head.weight"], device, cache.file("lm_head"), dtype=weight_dtype)

    def hidden(self, hidden_streams: ttnn.Tensor) -> ttnn.Tensor:
        """``[B, S, hc, D]`` -> ``norm(mean over the streams)`` ``[1, 1, B*S, D]`` bf16, the reference's last hidden state."""
        mean = ttnn.mean(hidden_streams, dim=2, keepdim=True)  # [B, S, 1, D]
        rows = rows_from_tokens(mean)
        ttnn.deallocate(mean)
        h = ttnn.rms_norm(rows, weight=self.norm_weight, epsilon=self.eps, compute_kernel_config=_HIFI4)
        ttnn.deallocate(rows)
        return h

    def forward(self, hidden_streams: ttnn.Tensor) -> ttnn.Tensor:
        """``[B, S, hc, D]`` -> logits ``[1, 1, B*S, V]`` (row ``b * S + s`` is token ``(b, s)``)."""
        h = self.hidden(hidden_streams)
        logits = self.lm_head(h)
        ttnn.deallocate(h)
        return logits
