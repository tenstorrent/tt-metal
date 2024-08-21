# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from torchvision import models
from loguru import logger
from models.utility_functions import comp_allclose

from models.experimental.vgg.tt.vgg import vgg16
from models.utility_functions import comp_pcc, torch_to_tt_tensor, unpad_from_zero


_batch_size = 1


@pytest.mark.parametrize(
    "dtype",
    ((ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_vgg16_inference(device, pcc, imagenet_sample_input, model_location_generator, dtype):
    image = imagenet_sample_input

    batch_size = _batch_size
    with torch.no_grad():
        torch_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        torch_vgg.eval()
        torch_output = torch_vgg(image).unsqueeze(1).unsqueeze(1)

        # TODO: enable conv on tt device after adding fast dtx transform
        cache_path = "/mnt/MLPerf/tt_dnn-models/tt/VGG/vgg16/"
        tt_vgg = vgg16(device, disable_conv_on_tt_device=True, tt_cache_path=cache_path, tt_dtype=dtype)

        tt_image = torch_to_tt_tensor(image, device=device)

        tt_output = tt_vgg(tt_image)
        tt_output = unpad_from_zero(tt_output, torch_output.shape)
        tt_output = tt_output.cpu()

        logger.info(comp_allclose(torch_output, tt_output))
        pcc_passing, pcc_output = comp_pcc(torch_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."
