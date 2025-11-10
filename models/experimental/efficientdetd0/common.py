# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch
from loguru import logger

from models.experimental.efficientdetd0.reference.efficientdet import EfficientDetBackbone
from models.experimental.efficientdetd0.reference.modules import BiFPN, Regressor, Classifier


KEY_MAPPINGS = {
    "depthwise_conv.conv": "depthwise_conv",
    "pointwise_conv.conv": "pointwise_conv",
    "p3_down_channel.0.conv": "p3_down_channel.0",
    "p4_down_channel.0.conv": "p4_down_channel.0",
    "p5_down_channel.0.conv": "p5_down_channel.0",
    "p5_to_p6.0.conv": "p5_to_p6.0",
    "p4_down_channel_2.0.conv": "p4_down_channel_2.0",
    "p5_down_channel_2.0.conv": "p5_down_channel_2.0",
}


def map_single_key(checkpoint_key):
    for key, value in KEY_MAPPINGS.items():
        checkpoint_key = checkpoint_key.replace(key, value)
    return checkpoint_key


def load_partial_state(torch_model: torch.nn.Module, state_dict, layer_name: str = ""):
    partial_state_dict = {}
    layer_prefix = layer_name + "."
    for k, v in state_dict.items():
        if k.startswith(layer_prefix):
            partial_state_dict[k[len(layer_prefix) :]] = v
    torch_model.load_state_dict(partial_state_dict, strict=True)
    return torch_model


def load_torch_model_state(torch_model: torch.nn.Module = None, layer_name: str = "", model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        model_path = "models"
    else:
        model_path = model_location_generator("vision-models/efficientdetd0", model_subdir="", download_if_ci_v2=True)
    if model_path == "models":
        if not os.path.exists(
            "models/experimental/efficientdetd0/efficientdet-d0.pth"
        ):  # check if efficientdet-d0.pth is available
            os.system(
                "models/experimental/efficientdetd0/resources/efficientdetd0_weights_download.sh"
            )  # execute the efficientdetd0_weights_download.sh file
        weights_path = "models/experimental/efficientdetd0/efficientdet-d0.pth"
    else:
        weights_path = os.path.join(model_path, "efficientdet-d0.pth")

    # Load checkpoint
    state_dict = torch.load(weights_path, map_location="cpu")

    if isinstance(
        torch_model,
        (
            BiFPN,
            Regressor,
            Classifier,
        ),
    ):
        # Get keys
        checkpoint_keys = set(state_dict.keys())

        # Get key mappings
        logger.info("Mapping keys...")
        key_mapping = {}
        for checkpoint_key in checkpoint_keys:  # pickle key
            mapped_key = map_single_key(checkpoint_key)
            key_mapping[checkpoint_key] = mapped_key

        # Apply mappings
        mapped_state_dict = {}
        for checkpoint_key, model_key in key_mapping.items():
            mapped_state_dict[model_key] = state_dict[checkpoint_key]
        logger.debug(f"Mapped {len(mapped_state_dict)} weights")

        # Load weights
        torch_model = load_partial_state(torch_model, mapped_state_dict, layer_name)
    elif isinstance(torch_model, EfficientDetBackbone):
        ds_state_dict = {k: v for k, v in state_dict.items()}
        mapped_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                mapped_state_dict[name1] = parameter2

        # Load weights
        torch_model.load_state_dict(mapped_state_dict, strict=True)
    else:
        raise NotImplementedError("Unknown torch model. Weight loading not implemented")
    logger.info(f"Successfully loaded weights: efficientdetd0 {layer_name}")

    return torch_model.eval()
