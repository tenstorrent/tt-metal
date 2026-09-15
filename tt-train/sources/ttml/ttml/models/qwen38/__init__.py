# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 text backbone for tt-train.

Qwen3.8-27B is a hybrid stack of 64 layers: 48 Gated DeltaNet
(``linear_attention``) layers and 16 Gated Attention (``full_attention``)
layers, one full-attention layer every four.  This package implements the text
backbone only -- the checkpoint's vision tower and MTP head are skipped.

Entry points::

    from ttml.models.qwen38 import Qwen38Config, Qwen38Transformer

    config = Qwen38Config.from_hf_json("/path/to/config.json")
    model = Qwen38Transformer(config)
"""

from .config import Qwen38Config
from .attention import Qwen38GatedAttention
from .gated_deltanet import Qwen38GatedDeltaNet
from .delta_rule import chunk_gated_delta_rule, wy_inverse
from .transformer import (
    Qwen38Block,
    Qwen38MLP,
    Qwen38RMSNorm,
    Qwen38Transformer,
)

__all__ = [
    "Qwen38Config",
    "Qwen38Transformer",
    "Qwen38Block",
    "Qwen38MLP",
    "Qwen38RMSNorm",
    "Qwen38GatedAttention",
    "Qwen38GatedDeltaNet",
    "chunk_gated_delta_rule",
    "wy_inverse",
]
