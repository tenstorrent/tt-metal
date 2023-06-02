from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import tt_lib
import torch
import timm
import pytest
from loguru import logger

from sweep_tests.comparison_funcs import comp_pcc
from inception import InceptionV4

_batch_size = 1


@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', "Ops Fused"])
def test_inception_inference(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    batch_size = _batch_size

    with torch.no_grad():
        torch_model = timm.create_model('inception_v4', pretrained=True)
        torch_model.eval()

        torch_output = torch_model(image)

        tt_model = InceptionV4(state_dict=torch_model.state_dict())
        tt_model.eval()

        tt_output = tt_model(image)
        passing = comp_pcc(torch_output, tt_output)

        assert passing[0], passing[1:]

    logger.info(f"PASSED {passing[1]}")
