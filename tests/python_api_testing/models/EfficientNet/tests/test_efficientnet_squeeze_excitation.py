import tt_lib
import torch
from loguru import logger
import torchvision

from tt_models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
)
from tt_models.EfficientNet.tt.efficientnet_squeeze_excitation import (
    TtEfficientnetSqueezeExcitation,
)


def test_efficientnet_squeeze_excitation_b0():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    refence_model = torchvision.models.efficientnet_b0(pretrained=True)
    refence_model.eval()

    block = 1
    sub_block = 1
    refence_module = refence_model.features[block][0].block[sub_block]

    input_channels = refence_module.fc1.in_channels
    squeeze_channels = refence_module.fc1.out_channels

    logger.debug(f"input_channels {input_channels}")
    logger.debug(f"squeeze_channels {squeeze_channels}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 32, 64, 64)
    pt_out = refence_module(test_input)

    tt_module = TtEfficientnetSqueezeExcitation(
        state_dict=refence_model.state_dict(),
        base_address=f"features.{block}.0.block.{sub_block}",
        device=device,
        input_channels=input_channels,
        squeeze_channels=squeeze_channels,
    )

    test_input = torch2tt_tensor(
        test_input, tt_device=device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
    )
    tt_out = tt_module(test_input)
    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_efficientnet_squeeze_excitation Passed!")
    else:
        logger.warning("test_efficientnet_squeeze_excitation Failed!")

    assert does_pass


def test_efficientnet_squeeze_excitation_v2_s():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    refence_model = torchvision.models.efficientnet_v2_s(pretrained=True)
    refence_model.eval()

    block = 6
    sub_block = 2

    refence_module = refence_model.features[block][0].block[sub_block]

    input_channels = refence_module.fc1.in_channels
    squeeze_channels = refence_module.fc1.out_channels

    logger.debug(f"input_channels {input_channels}")
    logger.debug(f"squeeze_channels {squeeze_channels}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 960, 64, 64)
    pt_out = refence_module(test_input)

    tt_module = TtEfficientnetSqueezeExcitation(
        state_dict=refence_model.state_dict(),
        base_address=f"features.{block}.0.block.{sub_block}",
        device=device,
        input_channels=input_channels,
        squeeze_channels=squeeze_channels,
    )

    test_input = torch2tt_tensor(
        test_input, tt_device=device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
    )
    tt_out = tt_module(test_input)
    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_efficientnet_squeeze_excitation Passed!")
    else:
        logger.warning("test_efficientnet_squeeze_excitation Failed!")

    assert does_pass
