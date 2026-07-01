# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Small shared head modules for the pi0.5 mesh pipelines.

`_PrefillHead` (lang-token embedding + prefix concat) and `_DenoiseHead` (final
adaRMS-norm weights) are used by the 1×8 single-mesh pipeline (`pipeline_1x8.py`).
Kept here — free of any stage/transport/migration imports — so the 1×8 pipeline
does not depend on the legacy 28-chip `pipeline.py`.
"""
from __future__ import annotations

from typing import Dict

import torch
import ttnn


class _PrefillHead:
    """Lang token embedding + prefix concat — lives on prefill_per_chip[0]."""

    def __init__(self, vlm_language_weights: Dict[str, torch.Tensor], submesh, vlm_hidden: int):
        embed_w = vlm_language_weights.get("model.embed_tokens.weight") or vlm_language_weights.get("lm_head.weight")
        if embed_w is None:
            raise RuntimeError("vlm_embed_tokens weight not found")
        self.submesh = submesh
        self.vlm_hidden = vlm_hidden
        self.embed_scale = float(vlm_hidden) ** 0.5
        self.vlm_embed_tokens = ttnn.from_torch(
            embed_w,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # embeddings use row-major
            device=submesh,
        )

    def embed_lang(self, lang_tokens) -> "ttnn.Tensor":
        """Returns (B, lang_len, vlm_hidden) bf16 TILE.

        ``lang_tokens`` is either a torch.Tensor (eager path — uploaded inline)
        or a persistent ttnn.Tensor on self.submesh (trace path — references
        the pre-allocated buffer that copy_host_to_device_tensor refreshes
        between calls).
        """
        if isinstance(lang_tokens, torch.Tensor):
            tokens_ttnn = ttnn.from_torch(
                lang_tokens.to(torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.submesh,
            )
        else:
            tokens_ttnn = lang_tokens
        lang_emb = ttnn.embedding(tokens_ttnn, self.vlm_embed_tokens)
        if lang_emb.layout != ttnn.TILE_LAYOUT:
            lang_emb = ttnn.to_layout(lang_emb, ttnn.TILE_LAYOUT)
        return ttnn.mul(lang_emb, self.embed_scale)


class _DenoiseHead:
    """Final adaRMS norm weights for the last expert chip."""

    def __init__(self, expert_weights: Dict[str, torch.Tensor], submesh):
        from models.experimental.pi0_5.tt.ttnn_common import tensor_1d_to_2d_ttnn

        self.submesh = submesh
        mod_w = expert_weights["model.norm.dense.weight"]
        self.mod_weight = ttnn.from_torch(
            mod_w.T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=submesh
        )
        mod_b = expert_weights.get("model.norm.dense.bias")
        self.mod_bias = tensor_1d_to_2d_ttnn(mod_b, submesh, dtype=ttnn.bfloat16) if mod_b is not None else None
        device_grid = submesh.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)
