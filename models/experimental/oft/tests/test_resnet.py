# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import os
from models.experimental.oft.reference.utils import get_abs_and_relative_error
from models.experimental.oft.reference.resnet import resnet18
from models.experimental.oft.tt.tt_resnet import TTBasicBlock, TTResNetFeatures
from models.experimental.oft.reference.utils import load_image
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc

# from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters_resnet
from tests.ttnn.unit_tests.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
from loguru import logger


@pytest.mark.parametrize(
    "input_shape, layers, expected_pcc",
    [
        ((1, 3, 384, 1280), [2, 2, 2, 2], (0.998, 0.998, 0.997)),  # ResNet-18
    ],
)
@pytest.mark.parametrize(
    "input_image_path",
    [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000013.jpg")),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024}], indirect=True)
def test_resnetfeatures_forward(device, input_image_path, input_shape, layers, expected_pcc):
    skip_if_not_blackhole_20_cores(device)
    torch.manual_seed(0)

    torch_tensor = load_image(input_image_path, pad_hw=(input_shape[-2], input_shape[-1]))[None]

    model = resnet18(pretrained=False, dtype=torch.float32, return_intermediates=True)

    params = create_OFT_model_parameters_resnet(model, torch_tensor, device)
    ref_intermediates, feats8, feats16, feats32 = model.forward(torch_tensor)

    n, c, h, w = feats8.shape
    feats8 = feats8.permute(0, 2, 3, 1)
    feats8 = feats8.reshape(1, 1, n * h * w, c)
    n, c, h, w = feats16.shape
    feats16 = feats16.permute(0, 2, 3, 1)
    feats16 = feats16.reshape(1, 1, n * h * w, c)

    n, c, h, w = feats32.shape
    feats32 = feats32.permute(0, 2, 3, 1)
    feats32 = feats32.reshape(1, 1, n * h * w, c)

    ttnn_input = torch_tensor.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_module = TTResNetFeatures(
        device,
        params,
        params.conv_args,
        TTBasicBlock,
        layers,
        return_intermediates=True,
    )
    ttnn_intermediates, ttnn_feats8, ttnn_feats16, ttnn_feats32 = tt_module.forward(device, ttnn_input)

    # Flattening any nested lists in ttnn_intermediates
    def flatten_intermediates(ttnn_intermediates):
        flattened_intermediates = []
        for item in ttnn_intermediates:
            if isinstance(item, list):
                flattened_intermediates.extend(item)
            else:
                flattened_intermediates.append(item)
        return flattened_intermediates

    ref_intermediates = flatten_intermediates(ref_intermediates)
    ttnn_intermediates = flatten_intermediates(ttnn_intermediates)

    logger.info("-----------------------------------------")
    logger.info(
        f"TTNN feats8 shape: {ttnn_feats8.shape} dtype: {ttnn_feats8.dtype}, layout: {ttnn_feats8.layout}, memory_config: {ttnn_feats8.memory_config()}"
    )
    logger.info(
        f"TTNN feats16 shape: {ttnn_feats16.shape} dtype: {ttnn_feats16.dtype},layout: {ttnn_feats16.layout}, memory_config: {ttnn_feats16.memory_config()}"
    )
    logger.info(
        f"TTNN feats32 shape: {ttnn_feats32.shape} dtype: {ttnn_feats32.dtype},layout: {ttnn_feats32.layout}, memory_config: {ttnn_feats32.memory_config()}"
    )
    logger.info("------------------------------------------")

    ttnn_feats8 = ttnn.to_torch(ttnn_feats8)
    ttnn_feats16 = ttnn.to_torch(ttnn_feats16)
    logger.info(f"TTNN feats16 shape: {ttnn_feats16.shape}")
    ttnn_feats32 = ttnn.to_torch(ttnn_feats32)

    for i, (ref_tensor, ttnn_tensor, name, pcc) in enumerate(
        zip(
            [*ref_intermediates, feats8, feats16, feats32],
            [*ttnn_intermediates, ttnn_feats8, ttnn_feats16, ttnn_feats32],
            [
                "x",
                "i",
                "i",
                "i",
                "i",
                "i",
                "i",
                "i",
                "i",
                "i",
                "i",
                "i",
                "i",
                "conv1f",
                "gn",
                "relu",
                "conv1mp",
                "feats8",
                "feats16",
                "feats32",
            ],
            [
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                0.999,
                expected_pcc[0],
                expected_pcc[1],
                expected_pcc[2],
            ],
        )
    ):
        ttnn_tensor = ttnn_tensor.reshape(ref_tensor.shape)
        abs, rel = get_abs_and_relative_error(ref_tensor, ttnn_tensor)
        passed, pcc = check_with_pcc(ref_tensor, ttnn_tensor, pcc=pcc)
        special_char = "✅" if passed else "❌"
        logger.warning(f"{special_char} Output {i} {name}: {pcc=} {abs=:.3f}, {rel=:.3f}")

    message, pcc = assert_with_pcc(feats8, ttnn_feats8, expected_pcc[0])
    logger.info(f"Passing: {message}, PCC: {pcc}")
    message, pcc = assert_with_pcc(feats16, ttnn_feats16, expected_pcc[1])
    logger.info(f"Passing: {message}, PCC: {pcc}")
    message, pcc = assert_with_pcc(feats32, ttnn_feats32, expected_pcc[2])
    logger.info(f"Passing: {message}, PCC: {pcc}")
