# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import timm
import pytest
from loguru import logger


from models.experimental.inceptionV4.reference.inception import InceptionV4
from models.utility_functions import (
    comp_pcc,
)

_batch_size = 1


@pytest.mark.parametrize("fuse_ops", [False, True], ids=["Not Fused", "Ops Fused"])
def test_inception_inference(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    batch_size = _batch_size

    with torch.no_grad():
        torch_model = timm.create_model("inception_v4", pretrained=True)
        torch_model.eval()

        torch_output = torch_model(image)

        tt_model = InceptionV4(state_dict=torch_model.state_dict())
        tt_model.eval()

        tt_output = tt_model(image)
        passing = comp_pcc(torch_output, tt_output)

        assert passing[0], passing[1:]

    logger.info(f"PASSED {passing[1]}")
