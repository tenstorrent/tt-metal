# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OWL-ViT (Open-World Localization Vision Transformer) for Tenstorrent hardware."""

from .tt.ttnn_owl_vit import (
    OwlViTTTNNConfig,
    update_model_config,
    owl_vit_for_object_detection,
    owl_vit_vision_model,
    owl_vit_vision_encoder,
    owl_vit_box_head,
    owl_vit_class_head,
    custom_preprocessor,
)

__all__ = [
    "OwlViTTTNNConfig",
    "update_model_config",
    "owl_vit_for_object_detection",
    "owl_vit_vision_model",
    "owl_vit_vision_encoder",
    "owl_vit_box_head",
    "owl_vit_class_head",
    "custom_preprocessor",
]
