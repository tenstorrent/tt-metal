# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.MapTR.reference.dependency import ResNet
from models.experimental.MapTR.resources.download_chkpoint import ensure_checkpoint_downloaded, MAPTR_WEIGHTS_PATH
from models.experimental.MapTR.tt.ttnn_backbone import TtResNet50
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
)

# Layer prefix for backbone (ResNet50) in mapTR
# The backbone weights are prefixed with 'img_backbone.'
BACKBONE_LAYER = "img_backbone."


def load_maptr_backbone_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    ensure_checkpoint_downloaded(weights_path)

    checkpoint = torch.load(weights_path, map_location="cpu")
    full_state_dict = checkpoint.get("state_dict", checkpoint)

    backbone_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(BACKBONE_LAYER):
            relative_key = key[len(BACKBONE_LAYER) :]
            backbone_weights[relative_key] = value

    logger.info(f"Loaded {len(backbone_weights)} weight tensors for backbone")
    return backbone_weights


def load_torch_model_maptr(torch_model: ResNet, weights_path: str = MAPTR_WEIGHTS_PATH):
    backbone_weights = load_maptr_backbone_weights(weights_path)
    state_dict = {k: v for k, v in backbone_weights.items()}
    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    return torch_model


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, ResNet):
        parameters["res_model"] = {}

        # Initial conv + bn (norm1 in dependency.py ResNet)
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.norm1)
        parameters["res_model"]["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
        }

        # Loop over all layers (layer1 to layer4)
        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            for block_idx, block in enumerate(layer):
                prefix = f"layer{layer_idx}_{block_idx}"
                parameters["res_model"][prefix] = {}

                # conv1, conv2, conv3 with norm1, norm2, norm3 (dependency.py naming)
                for conv_idx in [1, 2, 3]:
                    conv_name = f"conv{conv_idx}"
                    norm_name = f"norm{conv_idx}"
                    conv = getattr(block, conv_name)
                    bn = getattr(block, norm_name)
                    w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                    parameters["res_model"][prefix][conv_name] = {
                        "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                        "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                    }

                # downsample (if present)
                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    if isinstance(ds, torch.nn.Sequential):
                        conv = ds[0]
                        bn = ds[1]
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["res_model"][prefix]["downsample"] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

    return parameters


def create_maptr_model_parameters(model: ResNet, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 11 * 8192}], indirect=True)
def test_maptr_backbone(device, reset_seeds):
    torch_model = ResNet(depth=50, out_indices=(3,), frozen_stages=-1, norm_eval=False)
    torch_model = load_torch_model_maptr(torch_model)

    torch_input = torch.randn((6, 3, 384, 640), dtype=torch.bfloat16).float()
    torch_output = torch_model(torch_input)[0]

    # Prepare input for TT model (NHWC format, flattened)
    ttnn_input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, device=device, dtype=ttnn.bfloat16)

    parameter = create_maptr_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = TtResNet50(parameter.conv_args, parameter.res_model, device)
    ttnn_output = ttnn_model(ttnn_input_tensor, batch_size=6)[0]

    # Convert output back to PyTorch format
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.reshape(
        torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1]
    ).to(torch.float32)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.96)
    assert pcc_passed, pcc_message
