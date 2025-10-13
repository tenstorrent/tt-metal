# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch

from loguru import logger
from models.experimental.detr3d.reference.detr3d_model import (
    Model3DETR,
    SharedMLP,
    GenericMLP,
    PointnetSAModuleVotes,
    MaskedTransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerDecoder,
)
from torch.nn import MultiheadAttention


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
        model_path = model_location_generator("vision-models/detr3d", model_subdir="", download_if_ci_v2=True)
    if model_path == "models":
        if not os.path.exists(
            "models/experimental/detr3d/sunrgbd_masked_ep720.pth"
        ):  # check if sunrgbd_masked_ep720.pth is available
            os.system(
                "models/experimental/detr3d/resources/detr3d_weights_download.sh"
            )  # execute the detr3d_weights_download.sh file
        weights_path = "models/experimental/detr3d/sunrgbd_masked_ep720.pth"
    else:
        weights_path = os.path.join(model_path, "sunrgbd_masked_ep720.pth")

    # Load checkpoint
    state_dict = torch.load(weights_path, map_location="cpu")["model"]

    if isinstance(
        torch_model,
        (
            SharedMLP,
            GenericMLP,
            PointnetSAModuleVotes,
            MaskedTransformerEncoder,
            TransformerEncoderLayer,
            TransformerDecoderLayer,
            TransformerDecoder,
            MultiheadAttention,
        ),
    ):
        torch_model = load_partial_state(torch_model, state_dict, layer_name)
    elif isinstance(torch_model, Model3DETR):
        torch_model.load_state_dict(state_dict, strict=True)
    else:
        raise NotImplementedError("Unknown torch model. Weight loading not implemented")
    logger.info(f"Successfully loaded weights: {layer_name}")

    return torch_model.eval()
