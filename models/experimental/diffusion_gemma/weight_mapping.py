# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma -> Gemma-4 weight key remapping + self-conditioning loader (#47461).

The DiffusionGemma 26B-A4B checkpoint is fine-tuned from ``google/gemma-4-26B-A4B-it``;
its **text backbone weights are byte-identical in structure** to the in-repo
gemma4 backbone (``models/demos/gemma4/``). The only differences are:

1. A **prefix rename**. DiffusionGemma stores the text backbone under
   ``model.decoder.*`` (the bidirectional denoise decoder) while the encoder
   (causal prefill/commit) lives under ``model.encoder.language_model.*`` and is
   **tied** to the decoder (``convert_diffusion_gemma_weights.py`` copies
   ``model.encoder.language_model.* -> model.decoder.*``, cloning only
   ``layer_scalar``). The gemma4 loader expects ``model.language_model.*`` (HF) or
   ``model.layers.*`` (tests). So remapping ``model.decoder.* -> model.language_model.*``
   makes the DiffusionGemma backbone load through the unmodified gemma4 path.
2. **Two net-new weight groups** beyond the backbone (confirmed via the checkpoint
   ``model.safetensors.index.json`` diff vs ``gemma-4-26B-A4B-it``):
     - ``model.decoder.self_conditioning.{pre_norm,gate_proj,up_proj,down_proj}.weight``
       — the self-conditioning gated MLP (this module's :class:`SelfConditioning`
       reference; ``post_norm`` is scaleless so it has no checkpoint weight).
     - ``model.encoder.language_model.layers.{i}.layer_scalar`` — the encoder's
       own per-layer scalar (tied otherwise). Encoder/vision/multimodal — NOT on
       the text-first causal path; collected under ``ignored`` here.

This module is pure key/tensor bookkeeping — no ttnn / device / gemma4 import — so
it validates against just the checkpoint (or its index json) on any host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# The DiffusionGemma text-backbone prefix and its gemma4 equivalent.
DG_DECODER_PREFIX = "model.decoder."
GEMMA4_LM_PREFIX = "model.language_model."

# Net-new self-conditioning weights (checkpoint keys, verified). post_norm is
# scaleless (with_scale=False) -> absent from the checkpoint.
SELF_CONDITIONING_PREFIX = "model.decoder.self_conditioning."
SELF_CONDITIONING_WEIGHTS = ("pre_norm", "gate_proj", "up_proj", "down_proj")

# Everything under these prefixes is encoder/vision/multimodal — not on the
# text-first causal backbone path (#47461). Cross-referenced for #47462/#47467.
_IGNORED_PREFIXES = (
    "model.encoder.",
    "model.vision_tower.",
    "model.embed_vision.",
)


def gemma4_key_for(dg_key: str) -> Optional[str]:
    """Return the gemma4 backbone key for a DiffusionGemma **text-backbone** key.

    ``model.decoder.<rest>`` -> ``model.language_model.<rest>`` (except the
    self-conditioning sub-tree, which is net-new and has no gemma4 equivalent).
    Returns ``None`` for self-conditioning and encoder/vision keys.
    """
    if dg_key.startswith(SELF_CONDITIONING_PREFIX):
        return None
    if dg_key.startswith(DG_DECODER_PREFIX):
        return GEMMA4_LM_PREFIX + dg_key[len(DG_DECODER_PREFIX) :]
    return None


@dataclass(frozen=True)
class RemapResult:
    """Split of a DiffusionGemma key/state set into the three classes above."""

    backbone: Dict[str, str]  # dg_key -> gemma4_key  (text backbone, prefix-swapped)
    self_conditioning: List[str]  # dg self-conditioning keys (net-new)
    ignored: List[str]  # encoder / vision / multimodal keys (not text-first)

    @property
    def num_backbone(self) -> int:
        return len(self.backbone)


def classify_keys(keys) -> RemapResult:
    """Classify DiffusionGemma checkpoint keys into backbone / self-cond / ignored.

    Works on a plain iterable of key strings — e.g. the keys of
    ``model.safetensors.index.json`` ``weight_map`` — so the mapping can be
    validated WITHOUT loading the 51 GB of tensors.
    """
    backbone: Dict[str, str] = {}
    self_cond: List[str] = []
    ignored: List[str] = []
    for k in keys:
        if k.startswith(SELF_CONDITIONING_PREFIX):
            self_cond.append(k)
        elif k.startswith(DG_DECODER_PREFIX):
            backbone[k] = GEMMA4_LM_PREFIX + k[len(DG_DECODER_PREFIX) :]
        elif any(k.startswith(p) for p in _IGNORED_PREFIXES):
            ignored.append(k)
        else:
            # Unknown top-level key — surface it rather than silently dropping.
            ignored.append(k)
    return RemapResult(backbone=backbone, self_conditioning=self_cond, ignored=ignored)


def remap_state_dict(dg_state_dict: Dict) -> Tuple[Dict, Dict, List[str]]:
    """Remap a loaded DiffusionGemma state dict for the gemma4 backbone loader.

    Returns ``(backbone_state, self_cond_state, ignored_keys)`` where
    ``backbone_state`` is keyed by gemma4 ``model.language_model.*`` names (ready
    for ``Gemma4ModelArgs``/``Gemma4Model``) and ``self_cond_state`` is keyed by the
    short names ``{pre_norm,gate_proj,up_proj,down_proj}.weight`` (ready for
    :meth:`SelfConditioning.load_from_state_dict`).
    """
    result = classify_keys(dg_state_dict.keys())
    backbone_state = {g4_key: dg_state_dict[dg_key] for dg_key, g4_key in result.backbone.items()}
    self_cond_state = {k[len(SELF_CONDITIONING_PREFIX) :]: dg_state_dict[k] for k in result.self_conditioning}
    return backbone_state, self_cond_state, result.ignored


def expected_self_conditioning_shapes(hidden_size: int, intermediate_size: int) -> Dict[str, Tuple[int, ...]]:
    """The shapes the 4 self-conditioning checkpoint weights must have.

    Derived from ``DiffusionGemmaSelfConditioning(config)``:
    ``intermediate_size = config.intermediate_size`` (2112 for 26B-A4B), NOT
    ``moe_intermediate_size``.
    """
    return {
        "pre_norm.weight": (hidden_size,),
        "gate_proj.weight": (intermediate_size, hidden_size),
        "up_proj.weight": (intermediate_size, hidden_size),
        "down_proj.weight": (hidden_size, intermediate_size),
    }
