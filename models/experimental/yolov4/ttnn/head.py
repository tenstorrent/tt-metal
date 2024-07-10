# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.yolov4.ttnn.common import Conv
from models.experimental.yolov4.reference.head import Head
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import time


class TtHead:
    def __init__(self, model) -> None:
        if type(model) is str:
            torch_model = torch.load(model)
        else:
            torch_model = model.torch_model
        self.torch_model = torch_model
        self.conv1 = Conv(
            torch_model, "head.conv1", [1, 40, 40, 128], (1, 1, 1, 1), reshard=True, deallocate=False, activation=""
        )
        self.conv2 = Conv(
            torch_model, "head.conv2", [1, 40, 40, 256], (1, 1, 0, 0), reshard=True, fused_op=False, activation=""
        )
        self.conv3 = Conv(
            torch_model, "head.conv3", [1, 40, 40, 128], (2, 2, 1, 1), reshard=True, deallocate=False, activation=""
        )
        self.conv4 = Conv(
            torch_model,
            "head.conv4",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            reshard=True,
            height_sharding=False,
            activation="",
        )
        self.conv5 = Conv(torch_model, "head.conv5", [1, 20, 20, 256], (1, 1, 1, 1), reshard=True, activation="")
        self.conv6 = Conv(
            torch_model,
            "head.conv6",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            reshard=True,
            height_sharding=False,
            activation="",
        )
        self.conv7 = Conv(torch_model, "head.conv7", [1, 20, 20, 256], (1, 1, 1, 1), reshard=True, activation="")
        self.conv8 = Conv(
            torch_model,
            "head.conv8",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            reshard=True,
            height_sharding=False,
            activation="",
        )
        self.conv9 = Conv(
            torch_model, "head.conv9", [1, 20, 20, 256], (1, 1, 1, 1), reshard=True, deallocate=False, activation=""
        )
        self.conv10 = Conv(
            torch_model,
            "head.conv10",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            reshard=True,
            height_sharding=False,
            fused_op=False,
            activation="",
        )
        self.conv11 = Conv(torch_model, "head.conv11", [1, 20, 20, 256], (2, 2, 1, 1), reshard=True, activation="")
        self.conv12 = Conv(
            torch_model,
            "head.conv12",
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            reshard=True,
            height_sharding=False,
            activation="",
        )
        self.conv13 = Conv(
            torch_model,
            "head.conv13",
            [1, 10, 10, 512],
            (1, 1, 1, 1),
            reshard=True,
            height_sharding=False,
            activation="",
        )
        self.conv14 = Conv(
            torch_model,
            "head.conv14",
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            reshard=True,
            height_sharding=False,
            activation="",
        )
        self.conv15 = Conv(
            torch_model,
            "head.conv15",
            [1, 10, 10, 512],
            (1, 1, 1, 1),
            reshard=True,
            height_sharding=False,
            activation="",
        )
        self.conv16 = Conv(
            torch_model,
            "head.conv16",
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            reshard=True,
            height_sharding=False,
            activation="",
        )
        self.conv17 = Conv(
            torch_model,
            "head.conv17",
            [1, 10, 10, 512],
            (1, 1, 1, 1),
            reshard=True,
            height_sharding=False,
            activation="",
        )
        self.conv18 = Conv(
            torch_model,
            "head.conv18",
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            reshard=True,
            fused_op=False,
            activation="",
            height_sharding=False,
        )

    def __call__(self, device, input_tensor, model):
        output_tensor = self.conv1(device, input_tensor[0])
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor_left_1 = self.conv2(device, output_tensor)

        output_tensor = self.conv3(device, input_tensor[0])
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)
        outfrom_Neck1 = input_tensor[2]

        output_tensor = ttnn.experimental.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        if (
            outfrom_Neck1.memory_config().is_sharded()
        ):  # This is used because test of head sub_module passes interleaved tensor
            outfrom_Neck1 = ttnn.experimental.tensor.sharded_to_interleaved(outfrom_Neck1, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.concat([output_tensor, outfrom_Neck1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(outfrom_Neck1)

        output_tensor = self.conv4(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv5(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv6(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv7(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv8(device, output_tensor)
        output_tensor_split = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv9(device, output_tensor_split)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor_left_2 = self.conv10(device, output_tensor)

        output_tensor = self.conv11(device, output_tensor_split)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        outfromNeck2 = input_tensor[1]
        output_tensor = ttnn.experimental.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        if (
            outfromNeck2.memory_config().is_sharded()
        ):  # This is used because test of head sub_module passes interleaved tensor
            outfromNeck2 = ttnn.experimental.tensor.sharded_to_interleaved(outfromNeck2, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, outfromNeck2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(outfromNeck2)

        output_tensor = self.conv12(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv13(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv14(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv15(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv16(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv17(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor_left_3 = self.conv18(device, output_tensor)

        return output_tensor_left_1, output_tensor_left_2, output_tensor_left_3


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_head(device, reset_seeds):
    ttnn_model = TtHead("tests/ttnn/integration_tests/yolov4/yolov4.pth")

    torch_input_tensor1 = torch.randn(1, 40, 40, 128, dtype=torch.bfloat16)
    torch_input_tensor2 = torch.randn(1, 10, 10, 512, dtype=torch.bfloat16)
    torch_input_tensor3 = torch.randn(1, 20, 20, 256, dtype=torch.bfloat16)

    ttnn_input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn.bfloat16)
    ttnn_input_tensor1 = ttnn.reshape(ttnn_input_tensor1, (1, 1, 1600, 128))
    ttnn_input_tensor1 = ttnn.to_layout(ttnn_input_tensor1, layout=ttnn.TILE_LAYOUT)
    ttnn_input_tensor1 = ttnn.to_device(ttnn_input_tensor1, device=device)

    ttnn_input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=ttnn.bfloat16)
    ttnn_input_tensor2 = ttnn.reshape(ttnn_input_tensor2, (1, 1, 100, 512))
    ttnn_input_tensor2 = ttnn.to_layout(ttnn_input_tensor2, layout=ttnn.TILE_LAYOUT)
    ttnn_input_tensor2 = ttnn.to_device(ttnn_input_tensor2, device=device)

    ttnn_input_tensor3 = ttnn.from_torch(torch_input_tensor3, dtype=ttnn.bfloat16)
    ttnn_input_tensor3 = ttnn.reshape(ttnn_input_tensor3, (1, 1, 400, 256))
    ttnn_input_tensor3 = ttnn.to_layout(ttnn_input_tensor3, layout=ttnn.TILE_LAYOUT)
    ttnn_input_tensor3 = ttnn.to_device(ttnn_input_tensor3, device=device)

    ttnn_input_tensor = [ttnn_input_tensor1, ttnn_input_tensor2, ttnn_input_tensor3]
    torch_input_tensor1 = torch_input_tensor1.permute(0, 3, 1, 2).float()
    torch_input_tensor2 = torch_input_tensor2.permute(0, 3, 1, 2).float()
    torch_input_tensor3 = torch_input_tensor3.permute(0, 3, 1, 2).float()
    torch_input_tensor = [torch_input_tensor1, torch_input_tensor2, torch_input_tensor3]

    torch_model = Head()

    new_state_dict = {}
    ds_state_dict = {k: v for k, v in ttnn_model.torch_model.items() if (k.startswith("head."))}

    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    result_ttnn = ttnn_model(device, ttnn_input_tensor, torch_model)
    # start_time = time.time()
    # for x in range(1):
    #     result_ttnn = ttnn_model(device, ttnn_input_tensor)
    # print(f"Time taken: {time.time() - start_time}")

    result_1 = ttnn.to_torch(result_ttnn[0])
    result_2 = ttnn.to_torch(result_ttnn[1])
    result_3 = ttnn.to_torch(result_ttnn[2])
    ref1, ref2, ref3 = torch_model(torch_input_tensor)

    result_1 = result_1.reshape(1, ref1.shape[2], ref1.shape[3], 256)
    result_1 = result_1.permute(0, 3, 1, 2)

    result_2 = result_2.reshape(1, ref2.shape[2], ref2.shape[3], 256)
    result_2 = result_2.permute(0, 3, 1, 2)

    result_3 = result_3.reshape(1, ref3.shape[2], ref3.shape[3], 256)
    result_3 = result_3.permute(0, 3, 1, 2)

    # Output is sliced because ttnn.conv returns 256 channels instead of 255.
    result_1 = result_1[:, :255, :, :]
    result_2 = result_2[:, :255, :, :]
    result_3 = result_3[:, :255, :, :]

    assert_with_pcc(result_1, ref1, 0.99)
    assert_with_pcc(result_2, ref2, 0.99)
    assert_with_pcc(result_3, ref3, 0.99)
