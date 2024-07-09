# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import timm
import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import is_grayskull, is_wormhole_b0
from models.experimental.functional_vovnet.tt import ttnn_functional_vovnet


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_effective_se_module(
    reset_seeds,
    device,
):
    rf_model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    rf_model = rf_model.eval()
    model = rf_model.stages[0].blocks[0].attn

    parameters = preprocess_model_parameters(
        initialize_model=lambda: rf_model,
        convert_to_ttnn=lambda *_: False,
    )

    torch_input = torch.randn((1, 256, 56, 56), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    torch_output = model(torch_input.float())

    tt_output = ttnn_functional_vovnet.effective_se_module(
        device=device,
        torch_model=rf_model.state_dict(),
        path="stages.0.blocks.0.attn",
        input_tensor=ttnn_input,
        conv_params=(1, 1, 0, 0),
        batch_size=ttnn_input.shape[0],
        debug=False,
        bias=True,
    )

    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv_norm_act(
    reset_seeds,
    device,
):
    rf_model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    rf_model = rf_model.eval()
    model = rf_model.stem[0]

    torch_input = torch.randn((1, 224, 224, 3), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    torch_input = torch_input.permute(0, 3, 1, 2).float()
    torch_output = model(torch_input)

    tt_output, tt_output_h, tt_output_w = ttnn_functional_vovnet.conv_norm_act(
        device=device,
        x=ttnn_input,
        torch_model=rf_model.state_dict(),
        path="stem.0",
        input_params=ttnn_input.shape,
        conv_params=[2, 2, 1, 1],
        bias=False,
        batch_size=1,
    )

    tt_output = ttnn.to_torch(tt_output)
    torch_output = torch_output.permute(0, 2, 3, 1)
    tt_output = tt_output.reshape(torch_output.shape)

    if is_grayskull():
        assert_with_pcc(torch_output, tt_output, 0.99)
    elif is_wormhole_b0():
        assert_with_pcc(torch_output, tt_output, 0.40)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_seperable_conv_norm_act(
    reset_seeds,
    device,
):
    rf_model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    rf_model = rf_model.eval()

    model = rf_model.stem[2]
    torch_input = torch.randn((1, 112, 112, 64), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    torch_input = torch_input.permute(0, 3, 1, 2)
    torch_output = model(torch_input.float())

    tt_output, conv_h, conv_w = ttnn_functional_vovnet.seperable_conv_norm_act(
        device=device,
        x=ttnn_input,
        torch_model=rf_model.state_dict(),
        path="stem.2",
        conv_params1=[2, 2, 1, 1],
        conv_params2=[1, 1, 0, 0],
        debug=False,
        groups=64,
        bias=False,
        batch_size=1,
    )

    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.reshape(1, conv_h, conv_w, tt_output.shape[-1])
    tt_output = tt_output.permute(0, 3, 1, 2)

    assert_with_pcc(torch_output, tt_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_sequential_append_list(reset_seeds, device, imagenet_sample_input):
    rf_model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True).eval()
    model = rf_model.stages[1].blocks[0].conv_mid

    torch_input = torch.randn((1, 28, 28, 160), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)

    inputs = torch.randn((1, 28, 28, 256), dtype=torch.bfloat16)
    ttnn_inputs = ttnn.from_torch(inputs, dtype=ttnn.bfloat16)

    torch_input = torch_input.permute(0, 3, 1, 2).float()
    inputs = inputs.permute(0, 3, 1, 2).float()
    torch_output = model(torch_input, [inputs])

    tt_output, h, w = ttnn_functional_vovnet.sequential_append_list(
        device=device,
        input_tensor=ttnn_input,
        torch_model=rf_model.state_dict(),
        path="stages.1.blocks.0.conv_mid",
        concat_list=[ttnn_inputs],
        conv_params1=[1, 1, 1, 1],
        conv_params2=[1, 1, 0, 0],
        groups=160,
        layers_per_block=3,
        debug=False,
        bias=False,
        batch_size=1,
    )

    tt_output = ttnn.to_torch(tt_output)
    torch_output = torch_output.permute(0, 2, 3, 1)
    tt_output = tt_output.reshape(torch_output.shape)

    assert_with_pcc(torch_output, tt_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_osa_block(reset_seeds, device, imagenet_sample_input):
    rf_model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True).eval()

    # tr_out = rf_model(imagenet_sample_input)
    model = rf_model.stages[0].blocks[0]

    torch_input = torch.randn((1, 56, 56, 64), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    torch_input = torch_input.permute(0, 3, 1, 2)
    torch_output = model(torch_input.float())

    parameters = preprocess_model_parameters(
        initialize_model=lambda: rf_model,
        convert_to_ttnn=lambda *_: False,
    )

    tt_output = ttnn_functional_vovnet.osa_block(
        device=device,
        x=ttnn_input,
        torch_model=rf_model.state_dict(),
        path="stages.0.blocks.0",
        parameters=parameters.stages[0].blocks[0],
        model=rf_model.stages[0].blocks[0],
        groups=128,
        conv_norm_act_params=[1, 1, 0, 0],
        conv_params1=[1, 1, 1, 1],
        conv_params2=[1, 1, 0, 0],
        layers_per_block=3,
        residual=False,
        depthwise=True,
        debug=False,
        bias=False,
        batch_size=1,
    )

    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.79)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_osa_stage(
    reset_seeds,
    device,
):
    rf_model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True).eval()
    model = rf_model.stages[0]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: rf_model,
        convert_to_ttnn=lambda *_: False,
    )

    torch_input = torch.randn((1, 56, 56, 64), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    torch_input = torch_input.permute(0, 3, 1, 2)
    torch_output = model(torch_input.float())

    tt_output = ttnn_functional_vovnet.osa_stage(
        device=device,
        x=ttnn_input,
        torch_model=rf_model.state_dict(),
        path=f"stages.0",
        parameters=parameters.stages[0],
        model=rf_model.stages[0],
        groups=128,
        residual=False,
        depthwise=True,
        debug=False,
        bias=False,
        downsample=False,
        layer_per_block=3,
        batch_size=1,
    )

    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.79)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_classifier_head(
    reset_seeds,
    device,
):
    rf_model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    rf_model = rf_model.eval()
    model = rf_model.head

    torch_input = torch.randn((1, 7, 7, 1024), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device)
    torch_input = torch_input.permute(0, 3, 1, 2)
    torch_output = model(torch_input.float())

    tt_output = ttnn_functional_vovnet.classifier_head(
        device=device, x=ttnn_input, torch_model=rf_model.state_dict(), path="head"
    )
    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.reshape(torch_output.shape)

    assert_with_pcc(torch_output, tt_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vovnet(reset_seeds, device, imagenet_sample_input):
    rf_model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    rf_model = rf_model.eval()

    torch_input = torch.randn((1, 224, 224, 3), dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    torch_input = torch_input.permute(0, 3, 1, 2)
    torch_output = rf_model(torch_input.float())

    parameters = preprocess_model_parameters(
        initialize_model=lambda: rf_model,
        convert_to_ttnn=lambda *_: False,
    )

    tt_output = ttnn_functional_vovnet.vovnet(
        device=device,
        x=ttnn_input,
        torch_model=rf_model.state_dict(),
        parameters=parameters,
        model=rf_model,
        layer_per_block=3,
        residual=False,
        depthwise=True,
        batch_size=ttnn_input.shape[0],
    )
    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.permute(1, 2, 0, 3)
    tt_output = tt_output.squeeze(0).squeeze(0)

    if is_grayskull():
        assert_with_pcc(torch_output, tt_output, 0.85)
    elif is_wormhole_b0():
        assert_with_pcc(torch_output, tt_output, 0.77)
