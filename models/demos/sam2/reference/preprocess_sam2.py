# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Preprocessing parameters for SAM2 (sam2-hiera-tiny Image Mode) using Hugging Face weights or PyTorch reference weights.
Replaces all random weight initialization (`torch.randn`) with exact tensor mapping via ttnn.model_preprocessing."""

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias, preprocess_model_parameters


def custom_preprocessor(model, name, ttnn_module_args):
    """Custom parameter preprocessor mapping torch linear weights/biases to TTNN TILE_LAYOUT."""
    parameters = {}
    if isinstance(model, dict):
        state_dict = model
    elif hasattr(model, "state_dict"):
        state_dict = model.state_dict()
    else:
        return parameters

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        # Convert weight tensors for linear layers (transpose + tile layout ready)
        if "weight" in key and value.dim() in [2, 4]:
            if value.dim() == 2:
                parameters[key] = preprocess_linear_weight(value, dtype=ttnn.bfloat16)
            else:
                parameters[key] = value.to(torch.bfloat16)  # Keep conv/spatial weights as exact torch tensors before device upload
        elif "bias" in key:
            if value.dim() == 1:
                parameters[key] = preprocess_linear_bias(value, dtype=ttnn.bfloat16)
            else:
                parameters[key] = value.to(torch.bfloat16)
        else:
            parameters[key] = value.to(torch.bfloat16)

    return parameters


def load_and_preprocess_model_parameters(reference_model, device=None):
    """Loads reference model state dict and returns preprocessed ttnn parameter dictionary."""
    return preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
