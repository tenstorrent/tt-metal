# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Full MoonViT vision tower.

Forward pipeline (matches MoonVitPretrainedModel.forward + final_layernorm):
    pixel_values, grid_hws
        -> patch_embed  (Conv2d + 2D learned posemb add)
        -> encoder      (27 x [norm0 -> attn -> norm1 -> mlp] + final_layernorm)
        -> patch_merger (2x2 spatial concat -> 4608-dim tokens)
        -> output: merged tokens (NOT yet projected into LLM hidden)

The multi-modal projector lives in a separate module (`projector.py`)
to match the HF layout, where `MoonVitPretrainedModel` and
`KimiVLMultiModalProjector` are independent. Callers compose them
explicitly when feeding vision tokens into the LLM.

`DropInMoonViT` is a torch.nn.Module wrapper that lets the ttnn module
be exercised through the HF KimiVLForConditionalGeneration pipeline for
end-to-end testing.
"""
from __future__ import annotations

from models.common.lightweightmodule import LightweightModule


class MoonViT(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        weight_cache_path,
    ):
        super().__init__()
        raise NotImplementedError("Phase 1 — MoonViT")

    def forward(self, pixel_values, grid_hws):
        raise NotImplementedError("Phase 1 — MoonViT.forward")


class DropInMoonViT:
    """torch.nn.Module-compatible wrapper around the ttnn MoonViT.

    Exposes the same forward signature as HF MoonVitPretrainedModel +
    KimiVLMultiModalProjector so it can be substituted into
    KimiVLForConditionalGeneration for end-to-end smoke tests.
    """

    def __init__(self, tt_model: MoonViT):
        raise NotImplementedError("Phase 1 — DropInMoonViT")
