# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from loguru import logger
import torch
from torchvision import models
import pytest

from models.demos.resnet.genericResnetBlock import ResNet, Bottleneck
from models.utility_functions import comp_pcc


@pytest.mark.parametrize("fold_batchnorm", [True], ids=["Batchnorm folded"])
@pytest.mark.skip(reason="Conv disabled in main.")
def test_run_resnet50_inference(device, fold_batchnorm, imagenet_sample_input):
    image = imagenet_sample_input

    with torch.no_grad():
        torch.manual_seed(1234)

        torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        torch_resnet50.eval()

        state_dict = torch_resnet50.state_dict()

        tt_resnet50 = ResNet(
            Bottleneck,
            [3, 4, 6, 3],
            device=device,
            state_dict=state_dict,
            base_address="",
            fold_batchnorm=fold_batchnorm,
        )

        torch_output = torch_resnet50(image).unsqueeze(1).unsqueeze(1)
        tt_output = tt_resnet50(image)

        passing, info = comp_pcc(torch_output, tt_output, pcc=0.985)
        logger.info(info)
        assert passing
