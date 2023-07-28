from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from torchvision import models
from loguru import logger
import pytest

import tt_lib
from utility_functions_new import comp_pcc
from tt.vgg import vgg16
from vgg_utils import get_shape

_batch_size = 1

@pytest.mark.parametrize("pcc", ((0.99),),)
def test_vgg16_inference(pcc, imagenet_sample_input):
    image = imagenet_sample_input

    batch_size = _batch_size
    with torch.no_grad():
        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)


        torch_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        torch_vgg.eval()
        torch_output = torch_vgg(image).unsqueeze(1).unsqueeze(1)

        # TODO: enable conv on tt device after adding fast dtx transform
        tt_vgg = vgg16(device, host, disable_conv_on_tt_device=True)

        tt_image = tt_lib.tensor.Tensor(
            image.reshape(-1).tolist(),
            get_shape(image.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        tt_output = tt_vgg(tt_image)

        tt_output = tt_output.cpu()
        tt_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())

        pcc_passing, pcc_output = comp_pcc(torch_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        tt_lib.device.CloseDevice(device)
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."
