# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN RVC modules — production-quality reusable components.

Each module corresponds to a validated RVC subsystem:
    - hubert_ffn: Feed-forward network (PCC=0.999976)
    - hubert_encoder: Full transformer encoder layer (PCC=0.999996)
    - wavenet: WaveNet dilated conv stack (PCC=0.999820)
    - flow: Residual coupling flow decoder (PCC=0.999995)
"""

from models.demos.rvc.ttnn.modules.hubert_ffn import (
    ttnn_hubert_ffn,
    preprocess_ffn_weights,
)
from models.demos.rvc.ttnn.modules.hubert_encoder import (
    ttnn_encoder_layer_forward,
    preprocess_encoder_layer_weights,
)
from models.demos.rvc.ttnn.modules.wavenet import (
    ttnn_wn_layer_forward,
    ttnn_wn_forward,
    preprocess_wn_weights,
)
from models.demos.rvc.ttnn.modules.flow import (
    ttnn_flow_forward,
    preprocess_flow_weights,
)

__all__ = [
    "ttnn_hubert_ffn",
    "preprocess_ffn_weights",
    "ttnn_encoder_layer_forward",
    "preprocess_encoder_layer_weights",
    "ttnn_wn_layer_forward",
    "ttnn_wn_forward",
    "preprocess_wn_weights",
    "ttnn_flow_forward",
    "preprocess_flow_weights",
]
