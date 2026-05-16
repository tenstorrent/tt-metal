# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RVC TTNN implementation.

Structure:
    ttnn/runtime.py    — Persistent TTNN modules (TTNNFlowDecoder, TTNNGeneratorNSF)
    ttnn/utils.py      — Tensor conversion and weight preprocessing
    ttnn/ops/          — ConvTranspose1d wrapper
"""

from models.demos.rvc.ttnn.utils import (
    to_device,
    to_host,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_linear,
    preprocess_layer_norm,
    preprocess_conv1d_weight,
    postprocess_conv_output,
    DEFAULT_MEMORY_CONFIG,
    DEFAULT_DTYPE,
    DEFAULT_LAYOUT,
)
