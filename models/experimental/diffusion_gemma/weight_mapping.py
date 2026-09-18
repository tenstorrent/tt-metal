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
2. **Two net-new weight groups** beyond the backbone:
     - ``model.decoder.self_conditioning.{pre_norm,gate_proj,up_proj,down_proj}.weight``
       — the self-conditioning gated MLP (this module's :class:`SelfConditioning`
       reference; ``post_norm`` is scaleless so it has no checkpoint weight).
     - ``model.encoder.language_model.layers.{i}.layer_scalar`` — the encoder's
       own per-layer scalar. This one **is** on the text path (the encoder pass is
       prefill/commit), so it is collected under ``ignored`` only because the
       conversion script *clones* it from the decoder copy and the two are equal.
       ``check_encoder_layer_scalar_tie`` below re-establishes that on load: a
       checkpoint whose copies diverge would need a separate encoder scalar
       applied in prefill and commit, and silently reusing the decoder copy would
       be a compounding per-layer error into the prompt KV. ``model.encoder.*``
       holds nothing else on the text path (its other keys are vision tower /
       embed_vision).

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

# Tied lm_head materialized at runtime by newer transformers (tie_word_embeddings=
# True). Not on disk; numerically identical to decoder.embed_tokens. Ignored.
_TIED_LM_HEAD_KEYS = frozenset({"lm_head.weight"})

# The two copies of the per-layer scalar. The encoder copy is a clone of the
# decoder copy in the shipped checkpoint; see the module docstring.
ENCODER_LAYER_SCALAR_TEMPLATE = "model.encoder.language_model.layers.{layer}.layer_scalar"
DECODER_LAYER_SCALAR_TEMPLATE = "model.decoder.layers.{layer}.layer_scalar"


def encoder_layer_scalar_key(layer: int) -> str:
    return ENCODER_LAYER_SCALAR_TEMPLATE.format(layer=int(layer))


def decoder_layer_scalar_key(layer: int) -> str:
    return DECODER_LAYER_SCALAR_TEMPLATE.format(layer=int(layer))


def check_encoder_layer_scalar_tie(get_tensor, layers) -> List[Tuple[int, float]]:
    """Return ``[(layer, max_abs_diff)]`` for each layer whose two scalars differ.

    ``get_tensor(key)`` returns the checkpoint tensor for ``key`` or ``None`` when
    the key is absent. A layer with no encoder copy is skipped (nothing to
    diverge from); a layer with no *decoder* copy is a malformed checkpoint and is
    left to the caller's normal missing-key handling. An empty result means the
    encoder scalar is tied and can keep being ignored by the loader.
    """

    divergent: List[Tuple[int, float]] = []
    for layer in layers:
        encoder = get_tensor(encoder_layer_scalar_key(layer))
        if encoder is None:
            continue
        decoder = get_tensor(decoder_layer_scalar_key(layer))
        if decoder is None:
            continue
        diff = (encoder.detach().float() - decoder.detach().float()).abs().max().item()
        if diff != 0.0:
            divergent.append((int(layer), float(diff)))
    return divergent


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
    unknown: List[str]  # unexpected keys that should be investigated

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
    unknown: List[str] = []
    for k in keys:
        if k.startswith(SELF_CONDITIONING_PREFIX):
            self_cond.append(k)
        elif k.startswith(DG_DECODER_PREFIX):
            backbone[k] = GEMMA4_LM_PREFIX + k[len(DG_DECODER_PREFIX) :]
        elif k in _TIED_LM_HEAD_KEYS:
            # tie_word_embeddings=True: newer transformers materializes a tied
            # `lm_head.weight` (a view of decoder.embed_tokens) into state_dict().
            # It is absent from the on-disk checkpoint; gemma4 reconstructs its own
            # tied lm_head from embed_tokens, so this redundant key is ignored.
            ignored.append(k)
        elif any(k.startswith(p) for p in _IGNORED_PREFIXES):
            ignored.append(k)
        else:
            # Unknown top-level key — surface it rather than silently dropping.
            unknown.append(k)
    return RemapResult(backbone=backbone, self_conditioning=self_cond, ignored=ignored, unknown=unknown)


def remap_state_dict(dg_state_dict: Dict) -> Tuple[Dict, Dict, List[str]]:
    """Remap a loaded DiffusionGemma state dict for the gemma4 backbone loader.

    Returns ``(backbone_state, self_cond_state, ignored_keys)`` where
    ``backbone_state`` is keyed by gemma4 ``model.language_model.*`` names (ready
    for ``Gemma4ModelArgs``/``Gemma4Model``) and ``self_cond_state`` is keyed by the
    short names ``{pre_norm,gate_proj,up_proj,down_proj}.weight`` (ready for
    :meth:`SelfConditioning.load_from_state_dict`).
    """
    result = classify_keys(dg_state_dict.keys())
    if result.unknown:
        raise ValueError(f"unknown DiffusionGemma checkpoint keys: {sorted(result.unknown)[:10]}")
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
