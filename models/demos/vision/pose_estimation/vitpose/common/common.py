# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import VitPoseForPoseEstimation


def load_torch_model(model_name="usyd-community/vitpose-base-simple"):
    model = VitPoseForPoseEstimation.from_pretrained(model_name)
    return model.eval()


def get_vitpose_config():
    return {
        "image_size": (256, 192),
        "patch_size": (16, 16),
        "num_channels": 3,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "head_dim": 64,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-6,
        "qkv_bias": True,
        "num_labels": 17,
        "scale_factor": 4,
        "patch_height": 16,
        "patch_width": 12,
        "num_patches": 192,
    }
