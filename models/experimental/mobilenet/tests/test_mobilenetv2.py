# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import transformers
from loguru import logger
import pytest


from models.experimental.mobilenet.reference.mobilenetv2 import MobileNetV2Model as TtMobileNetv2Model

from models.utility_functions import (
    comp_pcc,
)

_batch_size = 1


@pytest.mark.parametrize("fuse_ops", [False, True], ids=["Not Fused", "Ops Fused"])
def test_mobilenetv2_inference(fuse_ops, imagenet_sample_input, device):
    image = imagenet_sample_input
    batch_size = _batch_size

    with torch.no_grad():
        image_processor = transformers.AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
        torch_model = transformers.MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224")

        torch_model.eval()
        state_dict = torch_model.state_dict()

        if not fuse_ops:
            # TODO(nshanker): enable running of conv on tt device. Currently, it results in low PCC = 0.97 so it is disabled.
            tt_model = TtMobileNetv2Model(
                config=torch_model.config,
                state_dict=state_dict,
                device=device,
                disable_conv_on_tt_device=True,
            )
        else:
            tt_model = TtMobileNetv2Model(config=torch_model.config, state_dict=state_dict)

        tt_model.eval()

        if fuse_ops:
            modules_to_fuse = [
                [
                    "conv_stem.first_conv.convolution",
                    "conv_stem.first_conv.normalization",
                ]
            ]
            modules_to_fuse.extend([["conv_stem.conv_3x3.convolution", "conv_stem.conv_3x3.normalization"]])
            modules_to_fuse.extend(
                [
                    [
                        "conv_stem.reduce_1x1.convolution",
                        "conv_stem.reduce_1x1.normalization",
                    ]
                ]
            )

            for i in range(16):
                modules_to_fuse.extend(
                    [
                        [
                            f"layer.{i}.expand_1x1.convolution",
                            f"layer.{i}.expand_1x1.normalization",
                        ]
                    ]
                )
                modules_to_fuse.extend(
                    [
                        [
                            f"layer.{i}.conv_3x3.convolution",
                            f"layer.{i}.conv_3x3.normalization",
                        ]
                    ]
                )
                modules_to_fuse.extend(
                    [
                        [
                            f"layer.{i}.reduce_1x1.convolution",
                            f"layer.{i}.reduce_1x1.normalization",
                        ]
                    ]
                )

            modules_to_fuse.extend([[f"conv_1x1.convolution", f"conv_1x1.normalization"]])

            tt_model = torch.ao.quantization.fuse_modules(tt_model, modules_to_fuse)

        torch_output = torch_model(image).last_hidden_state
        tt_output = tt_model(image)[0]

        passing = comp_pcc(torch_output, tt_output)
        assert passing[0], passing[1:]

    logger.info(f"PASSED {passing[1]}")
