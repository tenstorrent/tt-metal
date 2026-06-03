# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage 0 — vision orchestrator for Option C.

Owns the 4-chip vision submesh (3 SigLIP chips + 1 mm_projector/embed chip
in the eventual on-device layout; currently host-resident SigLIP per
`vision_slice.py`'s scaffolding mode). Exposes `initialize()` and
`forward(pixel_values, language_token_ids)` returning prefix
hidden_states on the vision submesh ready to be transported to stage 1.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig

from .mesh_setup import create_per_chip_submeshes
from .stages import StageSpec
from .vision_slice import Pi0_5OptionCVisionSlice, Pi0_5OptionCVisionSliceSplit


class StageVision:
    """Stage 0: image embedding + language token embedding → prefix hidden."""

    def __init__(
        self,
        spec: StageSpec,
        submesh,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        embed_on_host: bool = True,
        device_siglip: bool = False,
        vision_weights_l1: bool = False,
    ) -> None:
        if spec.stage_idx != 0:
            raise AssertionError(f"StageVision must be stage 0, got {spec.stage_idx}")
        if not spec.holds_mm_projector:
            raise AssertionError("Stage 0 must hold the mm_projector")
        if not spec.holds_embed_tokens:
            # We accept this even when embed_on_host=True — the StageSpec
            # records the *logical* ownership of the embed table; host-side
            # lookup is just the current placement strategy.
            raise AssertionError("Stage 0 must logically own VLM embed_tokens (host-side lookup is OK)")

        self.spec = spec
        self.submesh = submesh
        self.config = config
        self.weights = weights
        self.embed_on_host = embed_on_host
        self.device_siglip = device_siglip
        self.vision_weights_l1 = vision_weights_l1
        # Populated in initialize() when device_siglip=True.
        self.micro_submeshes: Optional[list] = None
        self.slice = None  # Pi0_5OptionCVisionSlice or Pi0_5OptionCVisionSliceSplit

    def initialize(self) -> None:
        """Build the vision slice.

        With `device_siglip=True`, the SigLIP encoder runs on 3 vision chips
        (9 layers each) and the mm_projector on the 4th chip — the target
        Option C placement (deployment plan §3.1). Otherwise SigLIP runs on
        the host (the scaffolding fallback used by the current smoke tests).
        """
        if self.slice is not None:
            return
        if self.device_siglip:
            self.micro_submeshes = create_per_chip_submeshes(self.submesh, count=4)
            self.slice = Pi0_5OptionCVisionSliceSplit(
                config=self.config,
                weights=self.weights,
                micro_submeshes=self.micro_submeshes,
                weights_in_l1=self.vision_weights_l1,
            )
        else:
            self.slice = Pi0_5OptionCVisionSlice(
                config=self.config,
                weights=self.weights,
                submesh=self.submesh,
                embed_on_host=self.embed_on_host,
            )

    @property
    def last_chip_submesh(self):
        """Submesh where build_prefix_hidden's output lives.

        Device-SigLIP mode: the projector chip. Host mode: the full submesh
        (since the prefix hidden is replicated there).
        """
        if self.device_siglip:
            assert self.micro_submeshes is not None
            return self.micro_submeshes[3]
        return self.submesh

    def forward(
        self,
        pixel_values: torch.Tensor,
        language_token_ids: torch.Tensor,
    ) -> "ttnn.Tensor":
        """Returns prefix hidden_states [B, num_patches + S_lang, vlm_W] on
        the vision submesh. The pipeline driver then ships it to stage 1.
        """
        if self.slice is None:
            raise RuntimeError("StageVision.forward called before initialize()")
        return self.slice.build_prefix_hidden(pixel_values, language_token_ids)
