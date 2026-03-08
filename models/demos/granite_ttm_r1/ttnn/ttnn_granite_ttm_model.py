# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from models.demos.granite_ttm_r1.ttnn.common import resolve_module, run_model_as_torch_fallback
from models.demos.granite_ttm_r1.ttnn.ttnn_granite_ttm_block import TtnnGraniteTTMBlock
from models.demos.granite_ttm_r1.ttnn.ttnn_granite_ttm_embedding import TtnnGraniteTTMEmbedding
from models.demos.granite_ttm_r1.ttnn.ttnn_granite_ttm_head import TtnnGraniteTTMHead
from models.demos.granite_ttm_r1.ttnn.ttnn_granite_ttm_patching import TtnnGraniteTTMPatching

PATCHING_CANDIDATES = ["patching", "patcher", "adaptive_patching", "adaptive_patch_embedding"]
EMBEDDING_CANDIDATES = ["embedding", "patch_embedding", "patch_embed", "encoder.embedding"]
BLOCK_LIST_CANDIDATES = ["blocks", "layers", "encoder.layers", "backbone.layers", "model.layers", "mixer_layers"]
HEAD_CANDIDATES = ["head", "forecast_head", "prediction_head", "decoder", "output_projection"]


class TtnnGraniteTTMModel:
    def __init__(self, parameters=None, config=None, reference_model: torch.nn.Module | None = None):
        self.parameters = parameters
        self.config = config
        self.reference_model = reference_model.eval() if reference_model is not None else None

        self.patching = TtnnGraniteTTMPatching(
            resolve_module(self.reference_model, PATCHING_CANDIDATES) if self.reference_model is not None else None
        )
        self.embedding = TtnnGraniteTTMEmbedding(
            resolve_module(self.reference_model, EMBEDDING_CANDIDATES) if self.reference_model is not None else None
        )

        self.blocks = []
        if self.reference_model is not None:
            block_container = resolve_module(self.reference_model, BLOCK_LIST_CANDIDATES)
            if block_container is not None:
                for block in block_container:
                    self.blocks.append(TtnnGraniteTTMBlock(torch_block=block))

        self.head = TtnnGraniteTTMHead(
            resolve_module(self.reference_model, HEAD_CANDIDATES) if self.reference_model is not None else None
        )

    def __call__(self, history, *, observed_mask=None, future_values=None, device=None, extra_inputs=None):
        if self.reference_model is not None:
            return run_model_as_torch_fallback(
                self.reference_model,
                history,
                observed_mask=observed_mask,
                future_values=future_values,
                device=device,
                extra_inputs=extra_inputs,
            )

        hidden_states = self.patching(history, device=device)
        hidden_states = self.embedding(hidden_states, device=device)
        for block in self.blocks:
            hidden_states = block(hidden_states, device=device)
        return self.head(hidden_states, device=device)
