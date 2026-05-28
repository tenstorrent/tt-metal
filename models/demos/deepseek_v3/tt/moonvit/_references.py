# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
HF reference module extraction.

Pulls torch submodules out of the loaded Kimi-VL model so per-submodule
PCC tests have a known-good torch comparison target.

Attribute paths are resolved at first use against the actual HF model
graph (because moonshotai/Kimi-VL-A3B-Instruct uses trust_remote_code,
the class names live in the downloaded checkpoint rather than the
installed transformers package). If an attribute path is wrong, the
helpers below raise a clear error pointing at the expected location.

The attribute paths below match the structure described in
modeling_kimi_vl.py at the time of writing (May 2026); update if HF
re-layouts the model.
"""
from __future__ import annotations

import copy
from typing import Any


# --- Attribute-path helpers ---------------------------------------------------


def _resolve(root, dotted: str):
    """Walk a dotted attribute path. Raise a useful error if missing."""
    cur = root
    for part in dotted.split("."):
        if not hasattr(cur, part):
            raise AttributeError(
                f"HF model has no attribute `{dotted}` "
                f"(failed at `{part}` on {type(cur).__name__}). "
                "If MoonViT's HF layout has changed, update "
                "models/demos/deepseek_v3/tt/moonvit/_references.py."
            )
        cur = getattr(cur, part)
    return cur


def _vision_tower(args):
    """The MoonVitPretrainedModel instance."""
    # HF KimiVLForConditionalGeneration has .vision_tower; the AutoModel
    # variant typically exposes the same attribute. If we ever load via
    # AutoModelForCausalLM the path might be `.model.vision_tower` — fall
    # back to a couple of known locations.
    model = args.hf_model
    for path in ("vision_tower", "model.vision_tower"):
        try:
            return _resolve(model, path)
        except AttributeError:
            continue
    raise AttributeError(
        f"Could not locate vision_tower on {type(model).__name__}. "
        "Tried: vision_tower, model.vision_tower."
    )


def _projector(args):
    model = args.hf_model
    for path in ("multi_modal_projector", "model.multi_modal_projector"):
        try:
            return _resolve(model, path)
        except AttributeError:
            continue
    raise AttributeError(
        f"Could not locate multi_modal_projector on {type(model).__name__}."
    )


def _encoder_layer(args, layer_num: int):
    """One MoonVitEncoderLayer from the encoder's block list.

    Attribute name for the block list isn't fixed across HF revisions —
    try a few candidates.
    """
    enc = _resolve(_vision_tower(args), "encoder")
    for path in ("blocks", "layers", "encoder_layers"):
        if hasattr(enc, path):
            blocks = getattr(enc, path)
            if layer_num >= len(blocks):
                raise IndexError(
                    f"Requested encoder layer {layer_num} but encoder.{path} "
                    f"has only {len(blocks)} entries."
                )
            return blocks[layer_num]
    raise AttributeError(
        f"Could not find a block list on encoder ({type(enc).__name__}). "
        "Tried: blocks, layers, encoder_layers."
    )


def _clone(module):
    """Return an eval-mode deep copy on CPU for use as a PCC reference."""
    out = copy.deepcopy(module).cpu().eval()
    for p in out.parameters():
        p.requires_grad_(False)
    return out


# --- Public reference factories ----------------------------------------------


def reference_layernorm(args, layer_num: int = 0, which: str = "norm0") -> Any:
    """Return one LayerNorm submodule from an encoder layer.

    Args:
        layer_num: index into the encoder block list.
        which: "norm0" (pre-attn) or "norm1" (pre-MLP) — matches the HF
            attribute names in MoonVitEncoderLayer.
    """
    layer = _encoder_layer(args, layer_num)
    if not hasattr(layer, which):
        raise AttributeError(
            f"Encoder layer {layer_num} has no `{which}` "
            f"(found: {[n for n, _ in layer.named_children()]})."
        )
    return _clone(getattr(layer, which))


def reference_mlp(args, layer_num: int = 0) -> Any:
    """Return one MLP2 submodule."""
    layer = _encoder_layer(args, layer_num)
    if hasattr(layer, "mlp"):
        return _clone(layer.mlp)
    raise AttributeError(
        f"Encoder layer {layer_num} has no `mlp` "
        f"(found: {[n for n, _ in layer.named_children()]})."
    )


def reference_attention(args, layer_num: int = 0) -> Any:
    """Return one encoder layer; tests can call .attention_qkvpacked on it.

    MoonViT's attention isn't a clean nn.Module — it's a method on the
    layer that takes (x, cu_seqlens, rope_freqs_cis). The test driver
    invokes that method directly. Cloning the whole layer gives us
    access to wqkv / wo / norms as needed.
    """
    return _clone(_encoder_layer(args, layer_num))


def reference_block(args, layer_num: int = 0) -> Any:
    """Return one MoonVitEncoderLayer (full block including both norms)."""
    return _clone(_encoder_layer(args, layer_num))


def reference_patch_embed(args) -> Any:
    """Return MoonVisionPatchEmbed (Conv2d + learned posemb)."""
    return _clone(_resolve(_vision_tower(args), "patch_embed"))


def reference_pos_emb(args) -> Any:
    """Return the Learnable2DInterpPosEmb instance.

    Lives inside MoonVisionPatchEmbed.pos_emb in the current HF layout.
    """
    return _clone(_resolve(_vision_tower(args), "patch_embed.pos_emb"))


def reference_patch_merger(args) -> Any:
    """patch_merger is a free function in HF (not an nn.Module).

    Return the callable itself so tests can invoke it directly.
    """
    # Import lazily — trust_remote_code modules aren't on sys.path until
    # the model has been loaded.
    args.hf_model  # ensure trust_remote_code module is registered
    import importlib

    # The HF cache loads modeling_kimi_vl into a synthetic module path.
    # Walk sys.modules to find it.
    import sys

    for name, mod in sys.modules.items():
        if name.endswith("modeling_kimi_vl") and hasattr(mod, "patch_merger"):
            return mod.patch_merger
    raise RuntimeError(
        "Could not locate patch_merger function in loaded Kimi-VL modules. "
        "Searched sys.modules for a *modeling_kimi_vl module exposing patch_merger."
    )


def reference_projector(args) -> Any:
    """KimiVLMultiModalProjector."""
    return _clone(_projector(args))


def reference_vision_tower(args) -> Any:
    """Full MoonVitPretrainedModel — patch_embed + encoder, no projector."""
    return _clone(_vision_tower(args))


def reference_rope_2d(args) -> Any:
    """Rope2DPosEmb module — lives at vision_tower.encoder.rope_2d."""
    return _clone(_resolve(_vision_tower(args), "encoder.rope_2d"))


def reference_final_layernorm(args) -> Any:
    """The final LayerNorm at the end of MoonVitEncoder.

    Applied after the last encoder block, before patch_merger:
        hidden_states = self.final_layernorm(hidden_states)
    """
    return _clone(_resolve(_vision_tower(args), "encoder.final_layernorm"))
