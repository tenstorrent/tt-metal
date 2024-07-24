# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.yolov4.ttnn_experiment4.common_exp4 import Conv
from models.experimental.yolov4.reference.downsample2 import DownSample2
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import time


class Down2:
    def __init__(self, model) -> None:
        if type(model) is str:
            torch_model = torch.load(model)
        else:
            torch_model = model.torch_model
        self.torch_model = torch_model
        self.conv1 = Conv(torch_model, "down2.conv1", [1, 160, 160, 64], (2, 2, 1, 1), reshard=True)
        self.conv2 = Conv(torch_model, "down2.conv2", [1, 80, 80, 128], (1, 1, 0, 0), reshard=True, deallocate=False)
        self.conv3 = Conv(torch_model, "down2.conv3", [1, 80, 80, 128], (1, 1, 0, 0))
        self.conv4 = Conv(torch_model, "down2.conv4", [1, 80, 80, 64], (1, 1, 0, 0), reshard=True, deallocate=False)

        self.res1_conv1 = Conv(
            torch_model, "down2.resblock.module_list.0.0", [1, 80, 80, 64], (1, 1, 0, 0), deallocate=False
        )
        self.res1_conv2 = Conv(torch_model, "down2.resblock.module_list.0.1", [1, 80, 80, 64], (1, 1, 1, 1))
        self.res2_conv1 = Conv(
            torch_model, "down2.resblock.module_list.1.0", [1, 80, 80, 64], (1, 1, 0, 0), deallocate=False
        )
        self.res2_conv2 = Conv(torch_model, "down2.resblock.module_list.1.1", [1, 80, 80, 64], (1, 1, 1, 1))

        self.conv5 = Conv(torch_model, "down2.conv5", [1, 80, 80, 128], (1, 1, 0, 0))

    def __call__(self, device, input_tensor):
        output_tensor_split = self.conv1(device, input_tensor)
        output_tensor_left = self.conv2(device, output_tensor_split)

        res1_split = self.conv3(device, output_tensor_split)

        output_tensor = self.res1_conv1(device, res1_split)
        output_tensor = self.res1_conv2(device, output_tensor)
        res2_split = res1_split + output_tensor
        ttnn.deallocate(res1_split)

        output_tensor = self.res2_conv1(device, res2_split)
        output_tensor = self.res2_conv2(device, output_tensor)
        output_tensor = res2_split + output_tensor

        ttnn.deallocate(res2_split)

        output_tensor = self.conv4(device, output_tensor)

        output_tensor = ttnn.experimental.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor_left = ttnn.experimental.tensor.sharded_to_interleaved(output_tensor_left, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, output_tensor_left], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(output_tensor_left)

        output_tensor = self.conv5(device, output_tensor)
        return output_tensor

    def __str__(self) -> str:
        this_str = ""
        index = 1
        for conv in self.convs:
            this_str += str(index) + " " + str(conv)
            this_str += " \n"
            index += 1
        return this_str


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_down2(device):
    ttnn_model = Down2("tests/ttnn/integration_tests/yolov4/yolov4.pth")

    torch_input = torch.randn((1, 160, 160, 64), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    torch_input = torch_input.permute(0, 3, 1, 2).float()
    torch_model = DownSample2()

    new_state_dict = {}
    ds_state_dict = {k: v for k, v in ttnn_model.torch_model.items() if (k.startswith("down2."))}

    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    result_ttnn = ttnn_model(device, ttnn_input)

    result = ttnn.to_torch(result_ttnn)
    ref = torch_model(torch_input)
    ref = ref.permute(0, 2, 3, 1)
    result = result.reshape(ref.shape)
    assert_with_pcc(result, ref, 0.97)
