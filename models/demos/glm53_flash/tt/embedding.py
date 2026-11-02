# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Token embedding and mHC entry for GLM-5.3-Flash (single chip).

HF (``Glm5NextTextModel.forward``)::

    inputs_embeds = self.embed_tokens(input_ids)                                   # bf16 [B, S, 4096]
    hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1)        # bf16 [B, S, 4, 4096]

``embed_tokens`` is a plain row gather of ``model.language_model.embed_tokens.weight`` (bf16
``[154880, 4096]``; ``padding_idx`` only affects training). On device ``ttnn.embedding`` gathers the
bf16 table rows straight into fp32 (exact widening) and ``ttnn.repeat`` makes the 4 identical
streams: packed ``[1, 1, T, 4*4096]`` fp32, the residual contract of the decoder layers
(stream ``i`` at columns ``[i*4096, (i+1)*4096)``, token ``b*S + s`` on row ``b*S + s``).

Token ids: ``ttnn.embedding`` indexes with a UINT32 ``[1, 1, 1, T]`` tensor. It does NOT gather a
``[1, 1, B, S]`` index tensor row by row (only the last dim is used), so ``tokens_to_host`` flattens
``[B, S]`` / ``[1, 1, B, S]`` / ``[1, 1, 1, B]`` int32 host ids to ``[1, 1, 1, B*S]`` and
``tokens_to_device`` uploads them as uint32 (bit-identical to int32 for ids < 2**31). The device
id tensor can be persistent (``ttnn.copy_host_to_device_tensor``) for traced decode.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.reference.checkpoint import EMBED_KEY, default_snapshot_dir, load_tensors, shard_headers
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mhc import mhc_expand

TILE = 32


def padded_rows(n_tokens: int) -> int:
    """Prefill row count for ``n_tokens`` tokens (the layers need ``T_pad % 32 == 0``)."""
    return -(-n_tokens // TILE) * TILE


class TtEmbedding(LightweightModule):
    """``embed_tokens`` + the mHC entry: token ids -> packed streams ``[1, 1, T, hc_mult*hidden]``."""

    def __init__(
        self,
        device,
        weight: torch.Tensor,  # [vocab, hidden] bf16 (checkpoint layout)
        hc_mult: int = 4,
        *,
        stream_dtype=ttnn.float32,
        pad_token_id: int = 0,
    ):
        assert weight.ndim == 2, weight.shape
        self.device = device
        self.vocab_size, self.hidden = (int(s) for s in weight.shape)
        self.hc_mult = hc_mult
        self.stream_dtype = stream_dtype
        self.pad_token_id = int(pad_token_id)
        # ttnn.embedding reads a ROW_MAJOR bf16 table; one DRAM copy, 1.27 GB.
        self.weight = ttnn.from_torch(
            weight.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @classmethod
    def from_checkpoint(cls, device, cfg: GLM53FlashConfig, snapshot: Path | None = None, headers=None, **kw):
        snapshot = snapshot or default_snapshot_dir()
        headers = headers or shard_headers(snapshot)
        w = load_tensors(snapshot, [EMBED_KEY], headers)[EMBED_KEY]
        assert tuple(w.shape) == (cfg.vocab_size, cfg.hidden_size), w.shape
        return cls(device, w, cfg.hc_mult, pad_token_id=cfg.pad_token_id, **kw)

    # ---- token ids ---------------------------------------------------------------------------
    def tokens_to_host(self, ids: torch.Tensor, pad_to: int | None = None) -> torch.Tensor:
        """Any-shape integer ids (``[B, S]``, ``[1, 1, B, S]``, ``[1, 1, 1, B]``) -> ``[1, 1, 1, T]``
        int32 in packed-stream row order (``b*S + s``), padded with ``pad_token_id`` to ``pad_to``
        rows (prefill: ``padded_rows(T)``; decode: the user-slot count)."""
        flat = ids.reshape(-1).to(torch.int32)
        n = flat.numel()
        if pad_to is not None:
            assert pad_to >= n, (pad_to, n)
            flat = torch.cat([flat, torch.full((pad_to - n,), self.pad_token_id, dtype=torch.int32)])
        assert int(flat.min()) >= 0 and int(flat.max()) < self.vocab_size, "token id out of range"
        return flat.reshape(1, 1, 1, -1)

    def tokens_to_device(self, ids: torch.Tensor, pad_to: int | None = None) -> ttnn.Tensor:
        """Upload ids as the UINT32 ``[1, 1, 1, T]`` ROW_MAJOR tensor ``forward`` consumes."""
        return ttnn.from_torch(
            self.tokens_to_host(ids, pad_to),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ---- forward -----------------------------------------------------------------------------
    def forward(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, 1, T]`` uint32 ids on device -> packed streams ``[1, 1, T, hc_mult*hidden]`` fp32.

        Device ops only (trace-safe): embedding gather (bf16 rows widened to ``stream_dtype``),
        then the repeat on the last dim (``mhc_expand``)."""
        e = ttnn.embedding(ids, self.weight, layout=ttnn.TILE_LAYOUT, dtype=self.stream_dtype)  # [1, T, H]
        e = ttnn.unsqueeze_to_4D(e)
        streams = mhc_expand(e, self.hc_mult)
        ttnn.deallocate(e)
        return streams
