"""
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
"""

from dataclasses import dataclass
from typing import List


@dataclass
class YolosConfig:
    """Configuration class for YOLOS-small model."""

    # Architecture parameters
    hidden_size: int = 384  # YOLOS-small uses 384 hidden size
    num_hidden_layers: int = 12
    num_attention_heads: int = 6  # YOLOS-small uses 6 heads
    intermediate_size: int = 1536  # YOLOS-small uses 1536 (4x hidden_size)
    hidden_act: str = "gelu"

    # Regularization
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

    # Input processing
    image_size: List[int] = None  # Default [512, 864]
    patch_size: int = 16
    num_channels: int = 3

    # Detection specific
    qkv_bias: bool = True
    num_detection_tokens: int = 100
    use_mid_position_embeddings: bool = True
    auxiliary_loss: bool = False

    # Number of COCO classes
    num_labels: int = 91  # 80 classes + background

    # Loss weights for Hungarian matching
    class_cost: float = 1
    bbox_cost: float = 5
    giou_cost: float = 2

    # Loss coefficients
    bbox_loss_coefficient: float = 5
    giou_loss_coefficient: float = 2
    eos_coefficient: float = 0.1

    def __post_init__(self):
        if self.image_size is None:
            self.image_size = [512, 864]


# Default configuration for YOLOS-small
def get_yolos_small_config():
    """Returns default configuration for YOLOS-small."""
    return YolosConfig(
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=1536,
        image_size=[512, 864],
        patch_size=16,
        num_channels=3,
        num_detection_tokens=100,
        num_labels=91,
    )
