# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Functional TTNN Implementation for OpenVoice.

This module implements OpenVoice using the official TTNN functional style:
- preprocess_model_parameters for weight loading
- Stateless functional operations
- Per-op PCC validation support
"""

from .operations import (
    Flip,
    LayerNorm1d,
    ensure_conv1d_weight,
    to_torch_tensor,
    ttnn_attention_functional,
    ttnn_conv1d_functional,
    ttnn_layer_norm_functional,
)
from .preprocess import custom_preprocessor, preprocess_model_parameters

__all__ = [
    "preprocess_model_parameters",
    "custom_preprocessor",
    "ttnn_conv1d_functional",
    "ttnn_layer_norm_functional",
    "ttnn_attention_functional",
    "to_torch_tensor",
    "ensure_conv1d_weight",
    "LayerNorm1d",
    "Flip",
]
