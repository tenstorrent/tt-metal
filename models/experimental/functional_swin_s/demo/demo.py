# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_swin_s.reference.swin_transformer import SwinTransformer
from models.experimental.functional_swin_s.tt.tt_swin_transformer import TtSwinTransformer
from tests.ttnn.integration_tests.swin_s.test_ttnn_swin_transformer_block import (
    create_custom_preprocessor as create_custom_preprocessor_transformer_block,
)
from tests.ttnn.integration_tests.swin_s.test_ttnn_patchmerging import (
    create_custom_preprocessor as create_custom_preprocessor_patch_merging,
)
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
)
from tests.ttnn.integration_tests.swin_s.test_ttnn_swin_transformer import (
    create_custom_preprocessor,
    preprocess_attn_mask,
)
import os
from models.utility_functions import (
    skip_for_grayskull,
)
from PIL import Image
from torchvision import transforms, models

from loguru import logger
from models.experimental.functional_swin_s.demo.demo_utils import get_data_loader, get_batch

from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES

imagenet_label_dict = {i: label for i, label in enumerate(IMAGENET2012_CLASSES)}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@skip_for_grayskull()
def test_swin_demo(device, reset_seeds):
    model = models.swin_s(weights="IMAGENET1K_V1")
    state_dict = model.state_dict()

    torch_model = SwinTransformer(
        patch_size=[4, 4], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=[7, 7]
    )

    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    batch_size = 1
    iterations = 100

    logger.info("ImageNet-1k validation Dataset")
    input_loc = "models/experimental/functional_swin_s/demo/ImageNet_data"
    data_loader = get_data_loader(input_loc, batch_size, iterations)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )

    attn_mask_tuple = preprocess_attn_mask([1, 3, 512, 512], [4, 4], [7, 7], [3, 3], device)

    ttnn_model = TtSwinTransformer(
        device,
        parameters,
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        attn_mask_tuple=attn_mask_tuple,
    )

    correct = 0

    for iter in range(iterations):
        predictions = []

        inputs, labels = get_batch(data_loader)
        torch_input_tensor = torch.unsqueeze(inputs, 0)
        input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn_model(input_tensor)

        output_tensor = ttnn.from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor).to(torch.float)

        prediction = output_tensor.argmax(dim=-1)

        for i in range(batch_size):
            predictions.append(imagenet_label_dict[prediction[i].item()])
            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
            )
            if imagenet_label_dict[labels[i]] == predictions[-1]:
                correct += 1

        del output_tensor, inputs, labels, predictions

    accuracy = correct / (batch_size * iterations)
    logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")
