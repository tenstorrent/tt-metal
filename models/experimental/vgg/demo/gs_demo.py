# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import pytest
import ttnn

from loguru import logger
from pathlib import Path

from models.experimental.vgg.tt.vgg import *
from models.utility_functions import torch_to_tt_tensor, unpad_from_zero
from models.experimental.vgg.vgg_utils import store_weights, get_tt_cache_path


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
@pytest.mark.parametrize(
    "batch_size",
    (
        (1),
        (2),
        (8),
    ),
)
def test_gs_demo(device, imagenet_sample_input, imagenet_label_dict, batch_size, dtype):
    images = batch_size * [imagenet_sample_input]
    class_labels = imagenet_label_dict

    model_version = "vgg16"
    tt_cache_path = get_tt_cache_path(model_version)
    base_addresses = [f"features", "classifier"]

    if (
        tt_cache_path == (str(Path(f"models/experimental/vgg/datasets/{model_version}")) + "/")
        and len(os.listdir(f"models/experimental/vgg/datasets/{model_version}")) < 32
    ):
        store_weights(model_version=model_version, file_name=tt_cache_path, dtype=dtype, base_addresses=base_addresses)

    with torch.no_grad():
        # TODO: enable conv on tt device after adding fast dtx transform
        tt_vgg = vgg16(device, disable_conv_on_tt_device=True, tt_cache_path=tt_cache_path)

        tt_images = [torch_to_tt_tensor(image, device=device) for image in images]
        tt_images = ttnn.concat(tt_images)

        tt_output = tt_vgg(tt_images)
        tt_output = unpad_from_zero(tt_output, tt_output.shape.with_tile_padding())
        tt_output = tt_output.cpu()

        logger.info(f"GS's predicted Output: {class_labels[torch.argmax(tt_output).item()]}\n")
