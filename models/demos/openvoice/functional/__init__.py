# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Functional TTNN Implementation for OpenVoice.

This module implements OpenVoice using the official TTNN functional style:
- preprocess_model_parameters for weight loading
- Stateless functional operations
- Per-op PCC validation support
"""

from .preprocess import preprocess_model_parameters, custom_preprocessor
from .operations import (
    ttnn_conv1d_functional,
    ttnn_layer_norm_functional,
    ttnn_attention_functional,
)

__all__ = [
    "preprocess_model_parameters",
    "custom_preprocessor",
    "ttnn_conv1d_functional",
    "ttnn_layer_norm_functional",
    "ttnn_attention_functional",
]
