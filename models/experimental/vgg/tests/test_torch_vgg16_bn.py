# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from torchvision import models
from loguru import logger

from models.experimental.vgg.reference.vgg import vgg16_bn
from models.utility_functions import comp_pcc


_batch_size = 1


@pytest.mark.parametrize("fuse_ops", [(False), (True)], ids=["Not Fused", "Ops Fused"])
def test_vgg16_bn_inference(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input

    batch_size = _batch_size
    with torch.no_grad():
        torch_vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        torch_vgg.eval()

        state_dict = torch_vgg.state_dict()

        tt_vgg = vgg16_bn(state_dict)
        tt_vgg.eval()

        if fuse_ops:
            indices = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
            modules_to_fuse = [[f"features.{ind}", f"features.{ind+1}", f"features.{ind+2}"] for ind in indices]
            tt_vgg = torch.ao.quantization.fuse_modules(tt_vgg, modules_to_fuse)

        torch_output = torch_vgg(image).unsqueeze(1).unsqueeze(1)
        tt_output = tt_vgg(image)

        passing = comp_pcc(torch_output, tt_output)
        assert passing[0], passing[1:]

    logger.info(f"vgg16_bn PASSED {passing[1]}")
