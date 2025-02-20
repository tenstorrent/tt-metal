# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import pytest
import ttnn

from PIL import Image
from torchvision import transforms

from models.experimental.functional_mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.experimental.functional_mobilenetv2.tt.model_preprocessing import (
    create_mobilenetv2_model_parameters,
)
from models.experimental.functional_mobilenetv2.tt import ttnn_mobilenetv2
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@skip_for_grayskull()
def test_mobilenetv2_demo(device, reset_seeds):
    if not os.path.exists("models/experimental/functional_mobilenetv2/mobilenet_v2-b0353104.pth"):
        os.system(
            "bash models/experimental/functional_mobilenetv2/weights_download.sh"
        )  # execute the weights_download.sh file

    state_dict = torch.load("models/experimental/functional_mobilenetv2/mobilenet_v2-b0353104.pth")
    ds_state_dict = {k: v for k, v in state_dict.items()}
    torch_model = Mobilenetv2()

    new_state_dict = {}

    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open("models/experimental/functional_mobilenetv2/demo/images/strawberry.jpg")

    img_t = transform(img)
    torch_input_tensor = torch.unsqueeze(img_t, 0)

    torch_output_tensor = torch_model(torch_input_tensor)

    parameters = create_mobilenetv2_model_parameters(torch_model, device=device)

    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16)

    ttnn_model = ttnn_mobilenetv2.MobileNetV2(parameters, device)
    output_tensor = ttnn_model(device, ttnn_input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.reshape(torch_output_tensor.shape)
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    with open("models/experimental/functional_mobilenetv2/demo/imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(output_tensor, 1)
    percentage = torch.nn.functional.softmax(output_tensor, dim=1)[0] * 100

    _, index_torch = torch.max(torch_output_tensor, 1)
    percentage_torch = torch.nn.functional.softmax(output_tensor, dim=1)[0] * 100

    # To know whether ttnn matches with torch
    # print("\033[1m" + f"torch class: {classes[index_torch[0]]}")
    # print("\033[1m" + f"Probability_torch: {percentage[index_torch[0]].item():.2f}%")

    print("\033[1m" + f"Predicted class: {classes[index[0]]}")
    print("\033[1m" + f"Probability: {percentage[index[0]].item():.2f}%")

    _, indices = torch.sort(output_tensor, descending=True)
    [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.94)
