# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""transformers 5.x compatibility helpers for the ViT reference model.

transformers 5.x removed the ``ViTEncoder`` module: ``ViTModel.encoder`` (a
``ViTEncoder`` wrapping a ``.layer`` ModuleList) became a bare ``ViTModel.layers``
ModuleList. These helpers bridge both layouts so the demos/tests work on <5 and >=5.
"""


def _vit(model):
    return model.vit if hasattr(model, "vit") else model


def vit_encoder_module(model):
    """Submodule to preprocess: ViTEncoder (<5) or the ViTModel.layers ModuleList (>=5)."""
    vit = _vit(model)
    return vit.encoder if hasattr(vit, "encoder") else vit.layers


def vit_encoder_layer(model, index=0):
    """A single transformer layer, regardless of the encoder/layers nesting."""
    vit = _vit(model)
    layers = vit.encoder.layer if hasattr(vit, "encoder") else vit.layers
    return layers[index]


def run_vit_encoder_reference(model, hidden_states, head_mask=None):
    """Run the ViT encoder forward across transformers versions.

    On <5 this calls ViTEncoder.forward; on >=5 (ViTEncoder removed) it applies the
    ViTLayer stack sequentially, matching the old encoder's per-layer head_mask use.
    """
    vit = _vit(model)
    if hasattr(vit, "encoder"):
        return vit.encoder(hidden_states, head_mask).last_hidden_state
    hidden = hidden_states
    for i, layer in enumerate(vit.layers):
        # transformers 5.x ViTLayer.forward returns a bare Tensor; <5 returned a (hidden_states, ...)
        # tuple. Only unwrap [0] when it's actually a tuple, otherwise [0] slices off the batch dim
        # (e.g. [batch, seq, dim] -> [seq, dim]).
        layer_out = layer(hidden, None if head_mask is None else head_mask[i])
        hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
    return hidden
