# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""KV head count and head dim per attention layer type, read from a HF text config.

Gemma4-family text configs give sliding and full attention layers different
``num_key_value_heads`` / ``head_dim``. transformers < 5.15 stores the full
attention values as the global fields ``num_global_key_value_heads`` and
``global_head_dim``. transformers >= 5.15 removes those fields, stores the
values in ``per_layer_config``, and raises ``AmbiguousGlobalPerLayerAttributeError``
(a ``RuntimeError``, which ``getattr(..., default)`` does not catch) when code
reads ``head_dim`` or ``num_key_value_heads`` on the top-level config.

Pure Python, no torch / ttnn / transformers import, so model packages that must
not depend on each other can share it.
"""

from typing import NamedTuple

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"


class LayerTypeAttention(NamedTuple):
    sliding_kv_heads: int
    sliding_head_dim: int
    full_kv_heads: int
    full_head_dim: int


def _first_layer_of_type(text_config, layer_type):
    layer_types = list(getattr(text_config, "layer_types", None) or ())
    if layer_type not in layer_types:
        return None
    layer_config = text_config.per_layer_config[layer_types.index(layer_type)]
    return layer_config.num_key_value_heads, layer_config.head_dim


def layer_type_attention(text_config):
    """Return a ``LayerTypeAttention`` for ``text_config``.

    ``text_config`` is the text sub-config (not the multimodal wrapper), or any
    object with the global field names (for example a model args dataclass).
    Every layer of one type has the same values in all Gemma4-family checkpoints,
    so the first layer of each type in ``layer_types`` is representative. If a
    type has no layer, it takes the values of the other type.
    """
    if getattr(text_config, "is_heterogeneous", False):
        sliding = _first_layer_of_type(text_config, SLIDING_ATTENTION)
        full = _first_layer_of_type(text_config, FULL_ATTENTION)
        if sliding is None and full is None:
            raise ValueError("per_layer_config needs layer_types with sliding_attention or full_attention layers")
        sliding = sliding or full
        full = full or sliding
        return LayerTypeAttention(sliding[0], sliding[1], full[0], full[1])

    sliding_kv_heads = text_config.num_key_value_heads
    sliding_head_dim = text_config.head_dim
    return LayerTypeAttention(
        sliding_kv_heads=sliding_kv_heads,
        sliding_head_dim=sliding_head_dim,
        full_kv_heads=getattr(text_config, "num_global_key_value_heads", None) or sliding_kv_heads,
        full_head_dim=getattr(text_config, "global_head_dim", None) or sliding_head_dim,
    )
