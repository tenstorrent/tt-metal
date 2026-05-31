# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RVC TTNN implementation.

Structure:
    tt/runtime.py    — Persistent TTNN modules (TTNNFlowDecoder, TTNNGeneratorNSF)
    tt/utils.py      — Tensor conversion and weight preprocessing
    tt/ops/          — ConvTranspose1d wrapper

The local package is named ``tt`` (matching the tt-metal audio model
convention, e.g. ``models/demos/audio/whisper/tt/``) rather than ``ttnn``
to avoid shadowing the system ``ttnn`` package.
"""

from models.demos.rvc.tt.utils import (
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
