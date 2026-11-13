# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.bevformer.model_config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CAMS
from models.experimental.bevformer.tests.common import (
    assert_pcc,
    build_reference_backbone,
    from_conv_layout,
    to_conv_layout,
)
from models.experimental.bevformer.tt.model_preprocessing import create_resnet_parameters
from models.experimental.bevformer.tt.tt_resnet import TTBottleneck, TTResLayer, TTResNet


# Every block test reads the same preprocessed backbone. Preprocessing traces a full
# reference forward, so it runs once per module rather than once per test. Being module
# scoped, it runs before the first test's device and reset_seeds fixtures. The trace only
# records shapes, so its input is zeros rather than a draw from the unseeded generator.
# It also brings up the UMD cluster (infer_ttnn_module_args does, even without a device)
# before the device fixture opens the device.
@pytest.fixture(scope="module")
def reference_and_parameters():
    reference_model = build_reference_backbone()
    parameters = create_resnet_parameters(reference_model, torch.zeros(NUM_CAMS, 3, IMAGE_HEIGHT, IMAGE_WIDTH))
    return reference_model, parameters


def _layer_kwargs(parameters, i):
    """The per-layer arguments TTResNet gives layer ``i`` for ``parameters``."""
    config = {k: v for k, v in parameters.tt_kwargs.items() if not k.startswith("stem_")}
    return TTResNet.layer_kwargs(i, **config)


def _input_layout(parameters, i):
    """The layout layer ``i`` receives in TTResNet: the max pool's TILE for layer1, else what layer
    ``i - 1`` emits for its camera groups."""
    if i == 0:
        return ttnn.TILE_LAYOUT
    conv1 = parameters.conv_args[f"layer{i + 1}"][0].conv1
    cameras = _layer_kwargs(parameters, i - 1)["cameras"]
    return TTResLayer.group_output_layout(cameras, conv1.input_height, conv1.input_width)


def _check(torch_output, ttnn_model, ttnn_output):
    assert (
        ttnn_output.dtype == ttnn_model.output_dtype
    ), f"emits {ttnn_output.dtype}, declares {ttnn_model.output_dtype}"
    assert_pcc(torch_output, from_conv_layout(ttnn_output, torch_output.shape), 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_bottleneck_layer1(device, reset_seeds, reference_and_parameters):
    reference_model, parameters = reference_and_parameters
    torch_input = torch.randn(NUM_CAMS, 64, IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4)
    torch_output = reference_model.layer1[0].eval()(torch_input)

    # layer1's first block alone, run as TTResNet runs layer1: one camera at a time in L1.
    ttnn_model = TTResLayer(
        [parameters.conv_args.layer1[0]],
        [parameters["res_model"]["layer1"][0]],
        device,
        **_layer_kwargs(parameters, 0),
    )
    ttnn_output = ttnn_model(to_conv_layout(torch_input, device, ttnn.bfloat16))
    _check(torch_output, ttnn_model, ttnn_output)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_reslayer3_first_blocks(device, reset_seeds, reference_and_parameters):
    reference_model, parameters = reference_and_parameters
    torch_input = torch.randn(NUM_CAMS, 512, IMAGE_HEIGHT // 8, IMAGE_WIDTH // 8)
    torch_output = reference_model.layer3[1].eval()(reference_model.layer3[0].eval()(torch_input))

    # layer3's first two blocks, run as TTResLayer runs layer3: on a group of cameras at a time in
    # L1, with the first block on the whole batch where TTResLayer puts it there.
    layout = _input_layout(parameters, 2)
    ttnn_model = TTResLayer(
        [parameters.conv_args.layer3[j] for j in range(2)],
        [parameters["res_model"]["layer3"][j] for j in range(2)],
        device,
        **_layer_kwargs(parameters, 2),
        input_layout=layout,
    )
    ttnn_output = ttnn_model(to_conv_layout(torch_input, device, ttnn.bfloat16, layout))
    _check(torch_output, ttnn_model, ttnn_output)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_bottleneck_layer4(device, reset_seeds, reference_and_parameters):
    """layer4's first block at stride 32, with the downsample; for the DCN variants its conv2 is a
    DCN over 512 input channels."""
    reference_model, parameters = reference_and_parameters
    torch_input = torch.randn(NUM_CAMS, 1024, IMAGE_HEIGHT // 16, IMAGE_WIDTH // 16)
    torch_output = reference_model.layer4[0].eval()(torch_input)

    kwargs = _layer_kwargs(parameters, 3)
    assert kwargs.pop("cameras") is None, "this test runs layer4's block on the whole batch, as TTResNet does"
    layout = _input_layout(parameters, 3)
    ttnn_model = TTBottleneck(
        parameters.conv_args.layer4[0],
        parameters["res_model"]["layer4"][0],
        device,
        **kwargs,
        input_layout=layout,
    )
    ttnn_output = ttnn_model(to_conv_layout(torch_input, device, ttnn.bfloat16, layout))
    _check(torch_output, ttnn_model, ttnn_output)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_reslayer1(device, reset_seeds, reference_and_parameters):
    reference_model, parameters = reference_and_parameters
    torch_input = torch.randn(NUM_CAMS, 64, IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4)
    torch_output = reference_model.layer1.eval()(torch_input)

    ttnn_model = TTResLayer(
        parameters.conv_args.layer1,
        parameters["res_model"]["layer1"],
        device,
        **_layer_kwargs(parameters, 0),
    )
    ttnn_output = ttnn_model(to_conv_layout(torch_input, device, ttnn.bfloat16))
    _check(torch_output, ttnn_model, ttnn_output)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_reslayer2(device, reset_seeds, reference_and_parameters):
    reference_model, parameters = reference_and_parameters
    torch_input = torch.randn(NUM_CAMS, 256, IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4)
    torch_output = reference_model.layer2.eval()(torch_input)

    layout = _input_layout(parameters, 1)
    ttnn_model = TTResLayer(
        parameters.conv_args.layer2,
        parameters["res_model"]["layer2"],
        device,
        **_layer_kwargs(parameters, 1),
        input_layout=layout,
    )
    ttnn_output = ttnn_model(to_conv_layout(torch_input, device, ttnn.bfloat16, layout))
    _check(torch_output, ttnn_model, ttnn_output)
