# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from loguru import logger
import torch
from torchvision import models
import pytest
from models.demos.resnet.tt.genericResnetBlock import ResNet, BasicBlock

from models.utility_functions import comp_pcc, comp_allclose_and_pcc


@pytest.mark.skip(reason="Hanging post commit 8/24/23 debug war room session, see PR#2297, PR#2301")
@pytest.mark.parametrize("fold_batchnorm", [True], ids=["Batchnorm folded"])
def test_run_resnet18_inference(device, fold_batchnorm, imagenet_sample_input):
    image = imagenet_sample_input

    with torch.no_grad():
        torch.manual_seed(1234)

        torch_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        torch_resnet.eval()

        state_dict = torch_resnet.state_dict()

        tt_resnet18 = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            device=device,
            state_dict=state_dict,
            base_address="",
            fold_batchnorm=fold_batchnorm,
        )

        torch_output = torch_resnet(image).unsqueeze(1).unsqueeze(1)
        tt_output = tt_resnet18(image)

        logger.info(comp_allclose_and_pcc(torch_output, tt_output))
        passing, info = comp_pcc(torch_output, tt_output)
        logger.info(info)

        assert passing
