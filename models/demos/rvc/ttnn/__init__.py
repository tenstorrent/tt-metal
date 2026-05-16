# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RVC TTNN operator wrappers and modules.

Structure:
    ttnn/utils.py      — Centralized tensor conversion, weight preprocessing
    ttnn/ops/          — Low-level operator wrappers (Conv1d, ConvTranspose1d, Linear, LayerNorm)
    ttnn/modules/      — Higher-level modules (FFN, Encoder, WaveNet, Flow)

Stage 1 bring-up: correctness-first, interleaved DRAM memory, bfloat16.
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
