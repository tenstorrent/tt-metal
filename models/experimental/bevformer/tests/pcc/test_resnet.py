# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.bevformer.tests.backbone_common import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CAMS,
    assert_pcc,
    build_reference_backbone,
    from_conv_layout,
    random_image_batch,
    to_conv_layout,
    tt_resnet_kwargs,
)
from models.experimental.bevformer.tests.backbone_weights import full_backbone_pcc
from models.experimental.bevformer.tt.model_preprocessing_backbone import create_resnet_parameters
from models.experimental.bevformer.tt.tt_resnet import TtBottleneck, TtResLayer, TtResNet


def _reference_and_parameters():
    reference_model = build_reference_backbone()
    parameters = create_resnet_parameters(reference_model, random_image_batch())
    return reference_model, parameters


def _layer_kwargs(i):
    """The memory arguments TtResNet gives layer ``i`` in this configuration."""
    memory_config = tt_resnet_kwargs()
    memory_config.pop("out_indices")
    return TtResNet.layer_kwargs(i, **memory_config)


def _check(torch_output, ttnn_model, ttnn_output):
    assert (
        ttnn_output.dtype == ttnn_model.output_dtype
    ), f"emits {ttnn_output.dtype}, declares {ttnn_model.output_dtype}"
    assert_pcc(torch_output, from_conv_layout(ttnn_output, torch_output.shape), 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_bottleneck_layer1(device, reset_seeds):
    reference_model, parameters = _reference_and_parameters()
    torch_input = torch.randn(NUM_CAMS, 64, IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4)
    torch_output = reference_model.layer1[0].eval()(torch_input)

    # layer1 reads the max pool's bfloat16 ROW_MAJOR output.
    ttnn_model = TtBottleneck(
        parameters.conv_args.layer1[0],
        parameters["res_model"]["layer1"][0],
        device,
        **_layer_kwargs(0),
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn_output = ttnn_model(to_conv_layout(torch_input, device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT))
    _check(torch_output, ttnn_model, ttnn_output)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_bottleneck_layer3(device, reset_seeds):
    reference_model, parameters = _reference_and_parameters()
    torch_input = torch.randn(NUM_CAMS, 512, IMAGE_HEIGHT // 8, IMAGE_WIDTH // 8)
    torch_output = reference_model.layer3[0].eval()(torch_input)

    # layer3 reads layer2's bfloat8_b output.
    ttnn_model = TtBottleneck(
        parameters.conv_args.layer3[0],
        parameters["res_model"]["layer3"][0],
        device,
        **_layer_kwargs(2),
        input_dtype=ttnn.bfloat8_b,
    )
    ttnn_output = ttnn_model(to_conv_layout(torch_input, device, ttnn.bfloat8_b))
    _check(torch_output, ttnn_model, ttnn_output)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_bottleneck_layer4(device, reset_seeds):
    """layer4's DCN has 512 input channels, so it samples them in two chunks."""
    reference_model, parameters = _reference_and_parameters()
    torch_input = torch.randn(NUM_CAMS, 1024, IMAGE_HEIGHT // 16, IMAGE_WIDTH // 16)
    torch_output = reference_model.layer4[0].eval()(torch_input)

    ttnn_model = TtBottleneck(
        parameters.conv_args.layer4[0],
        parameters["res_model"]["layer4"][0],
        device,
        **_layer_kwargs(3),
    )
    ttnn_output = ttnn_model(to_conv_layout(torch_input, device, ttnn.bfloat16))
    _check(torch_output, ttnn_model, ttnn_output)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_reslayer1(device, reset_seeds):
    reference_model, parameters = _reference_and_parameters()
    torch_input = torch.randn(NUM_CAMS, 64, IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4)
    torch_output = reference_model.layer1.eval()(torch_input)

    # As in TtResNet: layer1 reads the max pool's ROW_MAJOR output and emits bfloat8_b.
    ttnn_model = TtResLayer(
        parameters.conv_args.layer1,
        parameters["res_model"]["layer1"],
        device,
        **_layer_kwargs(0),
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_dtype=ttnn.bfloat8_b,
    )
    ttnn_output = ttnn_model(to_conv_layout(torch_input, device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT))
    _check(torch_output, ttnn_model, ttnn_output)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_reslayer2(device, reset_seeds):
    reference_model, parameters = _reference_and_parameters()
    torch_input = torch.randn(NUM_CAMS, 256, IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4)
    torch_output = reference_model.layer2.eval()(torch_input)

    ttnn_model = TtResLayer(
        parameters.conv_args.layer2,
        parameters["res_model"]["layer2"],
        device,
        **_layer_kwargs(1),
        input_dtype=ttnn.bfloat8_b,
    )
    ttnn_output = ttnn_model(to_conv_layout(torch_input, device, ttnn.bfloat8_b))
    _check(torch_output, ttnn_model, ttnn_output)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_resnet(device, reset_seeds):
    reference_model, parameters = _reference_and_parameters()
    torch_input = random_image_batch()
    torch_outputs = reference_model(torch_input)

    ttnn_model = TtResNet(parameters.conv_args, parameters["res_model"], device, **tt_resnet_kwargs())
    ttnn_outputs = ttnn_model(to_conv_layout(torch_input, device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT))

    for torch_output, ttnn_output in zip(torch_outputs, ttnn_outputs, strict=True):
        assert_pcc(torch_output, from_conv_layout(ttnn_output, torch_output.shape), full_backbone_pcc())
