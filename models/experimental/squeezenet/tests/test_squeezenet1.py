# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import models
import pytest
from loguru import logger
from tqdm import tqdm
from models.utility_functions import (
    comp_allclose_and_pcc,
    comp_pcc,
)
from models.experimental.squeezenet.reference.squeezenet import squeezenet1_0


_batch_size = 1


@pytest.mark.parametrize("fuse_ops", [False, True], ids=["Not Fused", "Ops Fused"])
def test_squeezenet1_inference(device, fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    batch_size = _batch_size
    with torch.no_grad():
        torch_squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)

        torch_squeezenet.eval()

        state_dict = torch_squeezenet.state_dict()
        if not fuse_ops:
            tt_squeezenet = squeezenet1_0(state_dict, device=device, disable_conv_on_tt_device=fuse_ops)
        else:
            tt_squeezenet = squeezenet1_0(state_dict, device=None, disable_conv_on_tt_device=fuse_ops)
        tt_squeezenet.eval()
        if fuse_ops:
            modules_to_fuse = [
                ["features.0", "features.1"],
                ["classifier.1", "classifier.2"],
            ]
            fire_indices = [3, 4, 5, 7, 8, 9, 10, 12]
            fire_1 = [
                [
                    f"features.{ind}.squeeze",
                    f"features.{ind}.squeeze_activation",
                ]
                for ind in fire_indices
            ]
            fire_2 = [
                [
                    f"features.{ind}.expand1x1",
                    f"features.{ind}.expand1x1_activation",
                ]
                for ind in fire_indices
            ]
            fire_3 = [
                [
                    f"features.{ind}.expand3x3",
                    f"features.{ind}.expand3x3_activation",
                ]
                for ind in fire_indices
            ]
            modules_to_fuse.extend(fire_1)
            modules_to_fuse.extend(fire_2)
            modules_to_fuse.extend(fire_3)

            tt_squeezenet = torch.ao.quantization.fuse_modules(tt_squeezenet, modules_to_fuse)

        torch_output = torch_squeezenet(image).unsqueeze(1).unsqueeze(1)
        tt_output = tt_squeezenet(image)

        passing = comp_pcc(torch_output, tt_output)
        assert passing[0], passing[1:]

    logger.info(f"PASSED {passing[1]}")
