# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from torchvision import transforms
from PIL import Image

from ttnn.model_preprocessing import preprocess_model
from models.experimental.functional_unet.reference.unet import UNet
from models.experimental.functional_unet.tt.tt_unet import TtUnet
from tests.ttnn.integration_tests.unet.test_ttnn_unet import create_custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_l1_small_size", [32768], indirect=True)
def test_demo(device):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {k: v for k, v in state_dict.items()}

    torch_model = UNet()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]
        # print(keys[i],values[i].shape)

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open("models/experimental/functional_unet/demo/TCGA_HT_7882_19970125_1.tif")

    img_t = transform(img)
    input_tensor = torch.unsqueeze(img_t, 0)

    torch_output_tensor = torch_model(input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet(device, parameters, new_state_dict)

    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_output = ttnn_model(device, input_tensor)

    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.reshape(1, 480, 640, 1)
    tt_output = torch.permute(tt_output, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, tt_output, pcc=0.99)
