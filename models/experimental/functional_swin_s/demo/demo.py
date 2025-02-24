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

    transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open("models/experimental/functional_swin_s/demo/strawberry.jpg")

    img_t = transform(img)
    torch_input_tensor = torch.unsqueeze(img_t, 0)
    print("torch_input_tensor: ", torch_input_tensor.shape)

    torch_output_tensor = torch_model(torch_input_tensor)
    print("torch_output_tensor: ", torch_output_tensor.shape)

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

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print("input_tensor: ", input_tensor.shape)

    output_tensor = ttnn_model(input_tensor)
    print("output_tensor: ", output_tensor.shape)

    #
    # Tensor Postprocessing
    #
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    with open("models/experimental/functional_swin_s/demo/imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    # Get the predicted class index and probability
    _, index = torch.max(output_tensor, 1)  # Get the index of the highest probability class
    percentage = torch.nn.functional.softmax(output_tensor, dim=1)[0] * 100  # Calculate the class probabilities
    print("percentage: ", percentage)

    # Print the predicted class and its probability
    print("\033[1m" + f"Predicted class: {classes[index[0]]}")
    print(
        "\033[1m" + f"Probability: {percentage[index[0]].item():.2f}%"
    )  # Format the probability with two decimal places

    _, indices = torch.sort(output_tensor, descending=True)
    [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
