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
from inception import InceptionV4
import timm


_batch_size = 1

@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', "Ops Fused"])
def test_inception_inference(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    batch_size = _batch_size
    with torch.no_grad():

        torch_model = timm.create_model('inception_v4', pretrained=True)
        torch_model.eval()

        torch_output = torch_model(image)

        state_dict = torch_model.state_dict()

        tt_model = InceptionV4(state_dict=state_dict)
        tt_model.eval()

        tt_output = tt_model(image)

        passing = comp_pcc(torch_output, tt_output)

        assert passing[0], passing[1:]

    logger.info(f"PASSED {passing[1]}")


def imagenet_sample_input():
    from PIL import Image
    im = Image.open("/mnt/MLPerf/tt_dnn-models/samples/ILSVRC2012_val_00048736.JPEG")
    im = im.resize((224, 224))
    from torchvision import transforms
    return transforms.ToTensor()(im).unsqueeze(0)


test_inception_inference(False, imagenet_sample_input())
