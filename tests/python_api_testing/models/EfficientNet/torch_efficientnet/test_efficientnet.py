from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
import pytest
from loguru import logger
from torchvision import models, transforms


from utility_functions import comp_allclose_and_pcc, comp_pcc
from libs import tt_lib as ttl

from efficientnet import efficientnet_v2_s



_batch_size = 1

@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', "Ops Fused"])
def test_efficient_inference(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    batch_size = _batch_size
    with torch.no_grad():


        torch_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        torch_model.eval()
        torch_output = torch_model(image)

        state_dict = torch_model.state_dict()

        tt_model = efficientnet_v2_s(state_dict)
        tt_model.eval()
        print(tt_model)
        tt_output = tt_model(image)

        passing = comp_pcc(torch_output, tt_output)

        assert passing[0], passing[1:]

    logger.info(f"PASSED {passing[1]}")
