# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import tests.ttnn.integration_tests.fadnetpp.custom_preprocessor_dispnetc as c_dispnetc
import tests.ttnn.integration_tests.fadnetpp.custom_preprocessor_dispnetres as c_dispnetres
from models.utility_functions import skip_for_wormhole_b0
from ttnn.model_preprocessing import preprocess_model
from models.experimental.functional_fadnetpp.reference.fadnetpp import FadNetPP
from models.experimental.functional_fadnetpp.tt.tt_fadnetpp import TtFadNetPP
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2


def scale_disp(disp, output_size=(1, 960, 576)):
    # print('current shape:', disp.size())
    i_w = disp.size()[-1]
    o_w = output_size[-1]

    ## Using sklearn.transform
    # trans_disp = disp.squeeze(1).data.cpu().numpy()
    # trans_disp = transform.resize(trans_disp, output_size, preserve_range=True).astype(np.float32)
    # trans_disp = torch.from_numpy(trans_disp).unsqueeze(1).cuda()

    # Using nn.Upsample
    m = nn.Upsample(size=(output_size[-2], output_size[-1]), mode="bilinear")
    trans_disp = m(disp)

    trans_disp = trans_disp * (o_w * 1.0 / i_w)
    return trans_disp


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        parameters["dispnetc"] = c_dispnetc.custom_preprocessor(
            device, model.dispnetc, name, ttnn_module_args["dispnetc"]
        )
        parameters["dispnetres"] = c_dispnetres.custom_preprocessor(
            device, model.dispnetres, name, ttnn_module_args["dispnetres"]
        )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@skip_for_wormhole_b0()
def test_fadnetpp_model(device, reset_seeds, model_location_generator):
    in_planes = 3 * 3 + 1 + 1
    torch_model = FadNetPP(in_planes)
    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    ds_state_dict = {k: v for k, v in torch_model.state_dict().items()}
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    img_left = Image.open("models/experimental/functional_fadnetpp/demo/left_image.png")
    img_right = Image.open("models/experimental/functional_fadnetpp/demo/right_image.png")

    transform = transforms.Compose(
        [
            transforms.Resize((960, 576)),  # Resize to network input size
            transforms.ToTensor(),  # Convert to tensor
        ]
    )
    img_left = transform(img_left)
    img_right = transform(img_right)

    input = torch.cat((img_left, img_right), 0)

    print("input Shape: {}".format(input.size()))
    input = input.reshape(1, 6, 960, 576)
    input_var = input  # torch.autograd.Variable(input, volatile=True)
    input_var = F.interpolate(input_var, (960, 576), mode="bilinear")
    torch_output = torch_model(input_var)[1]
    torch_input_tensor = input_var
    # (torch_output_tensor0, torch_output_tensor1) = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(input_var),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )
    ttnn_model = TtFadNetPP(parameters, device, in_planes, torch_model)
    #
    # Tensor Preprocessing
    #
    imgs = torch.chunk(torch_input_tensor, 2, dim=1)
    img_left = imgs[0]
    img_right = imgs[1]
    img_left = torch.permute(img_left, (0, 2, 3, 1))
    img_right = torch.permute(img_right, (0, 2, 3, 1))
    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = torch_input_tensor.reshape(
        torch_input_tensor.shape[0],
        1,
        torch_input_tensor.shape[1] * torch_input_tensor.shape[2],
        torch_input_tensor.shape[3],
    )
    input_tensor1 = img_left.reshape(img_left.shape[0], 1, img_left.shape[1] * img_left.shape[2], img_left.shape[3])
    input_tensor2 = img_right.reshape(
        img_right.shape[0], 1, img_right.shape[1] * img_right.shape[2], img_right.shape[3]
    )

    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_tensor1 = ttnn.from_torch(input_tensor1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_tensor2 = ttnn.from_torch(input_tensor2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    (output_tensor0, output_tensor1) = ttnn_model(device, input_tensor, input_tensor1, input_tensor2)

    #
    # Tensor Postprocessing
    #

    output_tensor0 = ttnn.to_torch(output_tensor0)
    output_tensor0 = output_tensor0.reshape(1, 960, 576, 1)
    output_tensor0 = torch.permute(output_tensor0, (0, 3, 1, 2))
    output_tensor0 = output_tensor0.to(torch_input_tensor.dtype)

    output_tensor1 = ttnn.to_torch(output_tensor1)
    output_tensor1 = output_tensor1.reshape(1, 960, 576, 1)
    output_tensor1 = torch.permute(output_tensor1, (0, 3, 1, 2))
    output_tensor1 = output_tensor1.to(torch_input_tensor.dtype)

    ttnn_output = output_tensor1

    torch_output = scale_disp(torch_output, (torch_output.size()[-1], 960, 576))
    disp = torch_output[:, :, :, :]
    # write disparity to file
    output_disp = disp
    # np_disp = disp.float().cpu().detach().numpy()

    np_array = disp.detach().numpy()
    print(np_array.shape)
    # Convert to range [0, 255] and cast to uint8
    np_array = np_array - np_array.min()  # Shift to start at 0
    np_array = np_array / np_array.max()

    np_array = (np_array * 255).astype(np.uint8)
    np_array = np_array.reshape(960, 576)

    colored_image = cv2.applyColorMap(np_array, cv2.COLORMAP_JET)
    colored_image_rgb = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(colored_image_rgb)  # np_array)
    # Create PIL image from NumPy array

    # Save image
    pil_img.save("tests/ttnn/integration_tests/fadnetpp/demo_torch_out.png")

    ttnn_output = scale_disp(ttnn_output, (ttnn_output.size()[-1], 960, 576))
    disp = ttnn_output[:, :, :, :]
    # write disparity to file
    output_disp = disp
    # np_disp = disp.float().cpu().detach().numpy()

    np_array = disp.detach().numpy()
    print(np_array.shape)
    # Convert to range [0, 255] and cast to uint8
    np_array = np_array - np_array.min()  # Shift to start at 0
    np_array = np_array / np_array.max()

    np_array = (np_array * 255).astype(np.uint8)
    np_array = np_array.reshape(960, 576)

    colored_image = cv2.applyColorMap(np_array, cv2.COLORMAP_JET)
    colored_image_rgb = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(colored_image_rgb)  # np_array)
    # Create PIL image from NumPy array

    # Save image
    pil_img.save("tests/ttnn/integration_tests/fadnetpp/demo_ttnn_out.png")
