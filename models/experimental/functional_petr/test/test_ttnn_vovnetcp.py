# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.experimental.functional_petr.reference.vovnetcp import (
    VoVNetCP,
    Hsigmoid,
    eSEModule,
    _OSA_stage,
)
from models.experimental.functional_petr.tt.ttnn_vovnetcp import (
    ttnn_hsigmoid,
    ttnn_esemodule,
    ttnn_osa_stage,
    ttnn_VoVNetCP,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from loguru import logger
from models.experimental.functional_petr.tt.common import (
    create_custom_preprocessor_vovnetcp,
    stem_parameters_preprocess,
)


@pytest.mark.parametrize(
    "n, c, h, w",
    (
        (6, 256, 1, 1),
        (6, 768, 1, 1),
        (6, 512, 1, 1),
        (6, 1024, 1, 1),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_vovnetcp_hsigmoid(device, reset_seeds, n, c, h, w):
    input_tensor = torch.randn((n, c, h, w))
    torch_model = Hsigmoid()
    torch_output = torch_model(input_tensor)
    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_model = ttnn_hsigmoid(device)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
    passed, msg = check_with_pcc(torch_output, ttnn_output, pcc=0.99)

    logger.info(f"vovnetcp_hsigmoid test passed: " f"PCC={msg}")


@pytest.mark.parametrize(
    "n, c, h, w",
    (
        (6, 256, 80, 200),
        (6, 768, 20, 50),
        (6, 256, 40, 100),
        (6, 1024, 10, 25),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vovnetcp_esemodule(device, n, c, h, w):
    torch_input_tensor = torch.randn(n, c, h, w)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    torch_model = eSEModule(c)
    torch_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor_vovnetcp(None), device=None
    )

    torch_output = torch_model(torch_input_tensor)
    ttnn_model = ttnn_esemodule(parameters)

    ttnn_output = ttnn_model(device=device, x=ttnn_input_tensor)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
    passed, msg = check_with_pcc(torch_output, ttnn_output, pcc=0.99)

    logger.info(f"vovnetcp_esemodule test passed: " f"PCC={msg}")


@pytest.mark.parametrize(
    "in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num,input_shape",
    [
        (128, 128, 256, 1, 5, 2, [1, 128, 80, 200]),
        (256, 160, 512, 3, 5, 3, [1, 256, 80, 200]),
        (512, 192, 768, 9, 5, 4, [1, 512, 40, 100]),
        (768, 224, 1024, 3, 5, 5, [1, 768, 20, 50]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vovnetcp_osa_stage(
    device, reset_seeds, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, input_shape
):
    torch_input_tensor = torch.randn(input_shape)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, device=device)
    torch_model = _OSA_stage(
        in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=True, depthwise=False
    )
    torch_model.eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor_vovnetcp(None), device=None
    )

    torch_output = torch_model(torch_input_tensor)
    ttnn_model = ttnn_osa_stage(
        parameters, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=True, depthwise=False
    )
    ttnn_output = ttnn_model(device=device, x=ttnn_input_tensor)

    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    if len(ttnn_output.shape) == 4 and ttnn_output.shape[2] == 1:
        # Calculate original H and W from the torch output shape
        target_h = torch_output.shape[2]
        target_w = torch_output.shape[3]
        ttnn_output = ttnn_output.reshape(ttnn_output.shape[0], ttnn_output.shape[1], target_h, target_w)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
    passed, msg = check_with_pcc(torch_output, ttnn_output, pcc=0.99)

    logger.info(f"vovnetcp_osa_stage_{stage_num} test passed: " f"PCC={msg}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vovnetcp(
    device,
):
    torch_input_tensor = torch.randn(1, 3, 320, 800)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, device=device)
    weights_state_dict = torch.load(
        "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth", weights_only=False
    )["state_dict"]
    torch_model = VoVNetCP("V-99-eSE")
    torch_model.load_state_dict(
        {k.replace("img_backbone.", ""): v for k, v in weights_state_dict.items() if "img_backbone" in k}
    )
    torch_model.eval()
    stem_parameters = stem_parameters_preprocess(torch_model)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor_vovnetcp(None), device=None
    )

    with torch.no_grad():
        output = torch_model(torch_input_tensor)

    ttnn_model = ttnn_VoVNetCP(parameters, stem_parameters, device)

    ttnn_output = ttnn_model(device, ttnn_input_tensor)

    # Tensor Postprocessing
    #  Convert TTNN outputs to torch for comparison
    ttnn_out0_torch = ttnn.to_torch(ttnn_output[0]).permute(0, 3, 1, 2)
    ttnn_out1_torch = ttnn.to_torch(ttnn_output[1]).permute(0, 3, 1, 2)

    # Reshape if needed
    if ttnn_out0_torch.shape != output[0].shape:
        ttnn_out0_torch = ttnn_out0_torch.reshape(output[0].shape)
    if ttnn_out1_torch.shape != output[1].shape:
        ttnn_out1_torch = ttnn_out1_torch.reshape(output[1].shape)

    # Compare
    passed0, msg0 = check_with_pcc(output[0], ttnn_out0_torch, pcc=0.99)
    passed1, msg1 = check_with_pcc(output[1], ttnn_out1_torch, pcc=0.99)

    logger.info("=" * 60)
    logger.info("FINAL BACKBONE RESULTS:")
    logger.info("=" * 60)
    logger.info(f"Stage 4 output PCC: {msg0}")
    logger.info(f"Stage 5 output PCC: {msg1}")
    assert_with_pcc(output[0], ttnn_out0_torch, pcc=0.99)
    assert_with_pcc(output[1], ttnn_out1_torch, pcc=0.99)
    assert passed0, f"Stage 4 PCC failed: {msg0}"
    assert passed1, f"Stage 5 PCC failed: {msg1}"
