# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
SigLIP Vision Tower — HuggingFace `SiglipVisionModel` wrapper.

Drop-in replacement for the hand-written `SigLIPVisionTower` in
`torch_siglip.py`. Uses HuggingFace's `transformers.SiglipVisionModel`
directly, matching openpi's numerical convention.

Background
----------
Our hand-written `torch_siglip.py` produces image features that differ
from HF's SigLIP by ~mean=0.51/token (max ~2.0) on the same weights,
because the two implementations have subtle ordering / numerical
convention differences. The lerobot pi05_libero_finetuned was re-trained
on OUR siglip outputs and therefore works with the custom impl, but the
upstream openpi pi05_libero was trained on HF SigLIP and fails when fed
ours. See `tests/pcc/test_pi05_upstream_vs_ours_activation_diff.py` and
the `pi05 siglip divergence` memory entry.

This module wires in HF's `SiglipVisionModel` via the `transformers`
package (which our python_env already provides at v4.53.2). We confirmed
that openpi's `transformers_replace/models/siglip/modeling_siglip.py` is
byte-identical to stock HF 4.53.2's, so no separate patches are needed
here — the patches openpi ships for Gemma do not extend to SigLIP.

Usage
-----
Gated by env var `PI0_SIGLIP_HF=1`. Default off, so the lerobot finetune
path keeps using the hand-written impl it was retrained on. The
`Pi0_5PaliGemmaBackbone` picks between `SigLIPVisionTower` (default) and
`HFSigLIPVisionTower` (opt-in) at construction.
"""

import os
from typing import Dict

import torch

from models.experimental.pi0_5.common.configs import SigLIPConfig


def use_hf_siglip() -> bool:
    """`PI0_SIGLIP_HF=1` -> use the HF SiglipVisionModel wrapper. Default off."""
    v = os.environ.get("PI0_SIGLIP_HF")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


class HFSigLIPVisionTower:
    """HuggingFace-backed SigLIP vision encoder. Same interface as the
    hand-written `SigLIPVisionTower` so it slots straight into
    `Pi0_5PaliGemmaBackbone` with no plumbing change.
    """

    def __init__(self, config: SigLIPConfig, weights: Dict[str, torch.Tensor]):
        """Construct from our config dataclass + the categorized vision weights
        from `Pi0_5WeightLoader.categorized_weights["vlm_vision"]` (keys look
        like `vision_model.encoder.layers.0.layer_norm1.weight`).
        """
        from transformers import SiglipVisionConfig, SiglipVisionModel

        self.config = config
        hf_cfg = SiglipVisionConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            image_size=config.image_size,
            patch_size=config.patch_size,
            layer_norm_eps=getattr(config, "layer_norm_eps", 1e-6),
        )
        # Disable HuggingFace's MultiheadAttentionPoolingHead init since
        # PaliGemma doesn't use it — keeps the load_state_dict noise quiet.
        # (We can't actually disable it in SiglipVisionConfig, so we just
        # accept the noise from `vision_model.head.*` missing keys.)
        self.model = SiglipVisionModel(hf_cfg)

        # The vlm_vision dict already uses `vision_model.X` prefix matching HF.
        # Filter to keys that SiglipVisionModel expects (drop `head.*` which
        # comes from openpi's HF config but is not in the checkpoint, and
        # similar) — but in practice we just pass the dict and let
        # load_state_dict report what was missing/unexpected.
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        # Expected missing: `vision_model.head.*` (the SigLIP attention-pool
        # head, unused in PaliGemma). Surface anything else loudly so we know
        # if the checkpoint is incompatible.
        unexpected_real = [k for k in unexpected if not k.startswith("vision_model.head")]
        missing_real = [k for k in missing if not k.startswith("vision_model.head")]
        if missing_real or unexpected_real:
            print(
                "[HFSigLIPVisionTower] load_state_dict notes:\n"
                f"  missing (real):    {missing_real[:5]}{' ...' if len(missing_real) > 5 else ''}\n"
                f"  unexpected (real): {unexpected_real[:5]}{' ...' if len(unexpected_real) > 5 else ''}"
            )

        # Match openpi's mixed-precision scheme (gemma_pytorch.py:62-82):
        # cast encoder to bf16, keep patch_embedding + position_embedding in fp32.
        # The 1.0 max/0.01 mean image-feature diff we measured against openpi's
        # `paligemma.model.vision_tower` was caused entirely by us running the
        # encoder in fp32 vs their bf16 — pinning matches their dtype scheme.
        self.model.to(dtype=torch.bfloat16)
        fp32_substrings = (
            "embeddings.patch_embedding.weight",
            "embeddings.patch_embedding.bias",
            "embeddings.position_embedding.weight",
        )
        for name, p in self.model.named_parameters():
            if any(s in name for s in fp32_substrings):
                p.data = p.data.to(dtype=torch.float32)

        # HF SigLIP's encoder runs in bf16 (cast above) but the patch+pos
        # embeddings are kept in fp32 to match openpi. HF doesn't bridge the
        # dtype between embeddings -> encoder, so layer_norm1 (bf16 weights)
        # would receive an fp32 input and raise "mixed dtype" mid-forward.
        # Override the embeddings module's forward to downcast its output to
        # bf16 (shows up in stack traces / dir(), unlike a forward_hook).
        _orig_embed_forward = self.model.vision_model.embeddings.forward

        def _embeddings_bf16_forward(*args, **kwargs):
            return _orig_embed_forward(*args, **kwargs).to(torch.bfloat16)

        self.model.vision_model.embeddings.forward = _embeddings_bf16_forward

        self.model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """`(B, 3, H, W)` floats in [-1, 1] -> `(B, num_patches, hidden_size)`.

        Returns `last_hidden_state` from the SigLIP encoder *before* any
        multi-modal projector — same contract as the hand-written tower.
        The downstream `MultiModalProjector` in `torch_paligemma.py` will
        project from `hidden_size` (1152) up to the VLM hidden dim (2048).

        Returns bf16 to match openpi's full bf16 VLM. Callers must handle
        downcast back to their pipeline dtype themselves — see
        `PaliGemmaBackbone.embed_image` for the bf16→fp32 reconciliation.
        """
        # patch_embedding lives in fp32; feed it fp32 input.
        x = pixel_values.to(torch.float32)
        with torch.no_grad():
            out = self.model(pixel_values=x, output_hidden_states=False)
        # Encoder output is bf16. Returned as-is so mm_projector ALSO runs
        # in bf16 (matching openpi's PaliGemmaMultiModalProjector dtype).
        return out.last_hidden_state
