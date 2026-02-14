# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
TTNN modules for OpenVoice V2.

These modules provide TTNN implementations of common building blocks
used throughout the OpenVoice model.
"""

from models.demos.openvoice.tt.modules.conv1d import ttnn_conv1d, ttnn_conv_transpose1d, Conv1dLayer
from models.demos.openvoice.tt.modules.gru import ttnn_gru, ttnn_gru_cell, GRULayer
from models.demos.openvoice.tt.modules.wavenet import WaveNetModule, fused_add_tanh_sigmoid_multiply

__all__ = [
    # Conv1d
    "ttnn_conv1d",
    "ttnn_conv_transpose1d",
    "Conv1dLayer",
    # GRU
    "ttnn_gru",
    "ttnn_gru_cell",
    "GRULayer",
    # WaveNet
    "WaveNetModule",
    "fused_add_tanh_sigmoid_multiply",
]
