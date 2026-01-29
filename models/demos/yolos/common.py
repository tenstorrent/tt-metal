# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
import transformers


def load_torch_model(model_location_generator=None):
    """
    Load the HuggingFace YOLOS-small model for object detection.
    
    Args:
        model_location_generator: Optional function to get local model path (for CI)
        
    Returns:
        Pretrained YOLOS model in eval mode
    """
    model_name = "hustvl/yolos-small"
    
    if model_location_generator is None or "TT_GH_CI_INFRA" not in os.environ:
        model = transformers.YolosForObjectDetection.from_pretrained(model_name)
        return model.eval()
    else:
        # For CI environments with local cache
        config = transformers.YolosConfig.from_pretrained(model_name)
        model = transformers.YolosForObjectDetection(config)
        weights_path = (
            model_location_generator("vision-models/yolos", model_subdir="", download_if_ci_v2=True)
            / "yolos_small.pth"
        )
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        return model.eval()


def get_yolos_config():
    """Get YOLOS-small configuration from HuggingFace."""
    return transformers.YolosConfig.from_pretrained("hustvl/yolos-small")
