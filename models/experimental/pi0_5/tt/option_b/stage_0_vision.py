# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage 0 — vision + language embed → prefix hidden for stage 1.

Wraps Pi0_5SubmeshVisionSlice. SigLIP runs on host (PI0_SIGLIP_HF=1 default
of the upstream-openpi path) and the projected vision features + tokenized
language embeddings are uploaded to the 4x2 submesh.

On-device SigLIP (TP=8 sharded) is the follow-up — see OPTION_B_STATUS.md
"What's not yet built" item #3.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig

from .stages import StageSpec
from .vision_slice import Pi0_5SubmeshVisionSlice


class Stage0Vision:
    """SigLIP (host) + projector + embed_tokens on a 4x2 submesh."""

    def __init__(
        self,
        spec: StageSpec,
        submesh,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        embed_on_host: bool = True,
    ) -> None:
        assert spec.stage_idx == 0, "Stage0Vision must be stage 0"
        assert spec.holds_mm_projector, "Stage 0 must hold the mm_projector"
        assert spec.holds_embed_tokens, "Stage 0 must hold VLM embed_tokens (host-side lookup OK)"
        self.spec = spec
        self.submesh = submesh
        self.config = config
        self.weights = weights
        self.embed_on_host = embed_on_host
        self.slice: Optional[Pi0_5SubmeshVisionSlice] = None

    def initialize(self) -> None:
        if self.slice is not None:
            return
        self.slice = Pi0_5SubmeshVisionSlice(
            config=self.config,
            weights=self.weights,
            submesh=self.submesh,
            embed_on_host=self.embed_on_host,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        language_token_ids: torch.Tensor,
    ) -> "ttnn.Tensor":
        """Returns prefix hidden_states [B, num_patches + S_lang, vlm_W] on the
        submesh, ready to feed into stage 1.
        """
        if self.slice is None:
            raise RuntimeError("Stage0Vision.forward called before initialize()")
        return self.slice.build_prefix_hidden(pixel_values, language_token_ids)
