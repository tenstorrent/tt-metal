# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.bevformer.model_config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CAMS, tt_resnet_kwargs
from models.experimental.bevformer.tests.common import (
    assert_pcc,
    build_reference_backbone,
    from_conv_layout,
    to_conv_layout,
)
from models.experimental.bevformer.tt.model_preprocessing import create_resnet_parameters
from models.experimental.bevformer.tt.tt_common import reduced_grid
from models.experimental.bevformer.tt.tt_resnet import TTBottleneck, TTResLayer, TTResNet


@pytest.fixture(scope="module")
def reference_model():
    return build_reference_backbone()


@pytest.fixture(scope="module")
def parameters_by_grid():
    return {}


# Every block test reads the same preprocessed backbone for its grid. Preprocessing traces a full
# reference forward, so it runs once per module and grid rather than once per test. The trace only
# records shapes, so its input is zeros rather than a draw from the generator. On a reduced grid
# TTResNet runs every layer on one camera at a time (tt_resnet_kwargs), so its convs are built for
# one camera.
@pytest.fixture
def reference_and_parameters(device, reference_model, parameters_by_grid):
    on_reduced_grid = reduced_grid(device)
    if on_reduced_grid not in parameters_by_grid:
        cameras = 1 if on_reduced_grid else NUM_CAMS
        parameters_by_grid[on_reduced_grid] = create_resnet_parameters(
            reference_model, torch.zeros(cameras, 3, IMAGE_HEIGHT, IMAGE_WIDTH), tt_resnet_kwargs(on_reduced_grid)
        )
    return reference_model, parameters_by_grid[on_reduced_grid]


def _per_camera(parameters):
    return parameters.tt_kwargs.get("per_camera", False)


def _layer_kwargs(parameters, i):
    """The per-layer arguments TTResNet gives layer ``i`` for ``parameters``."""
    config = {k: v for k, v in parameters.tt_kwargs.items() if not k.startswith("stem_") and k != "per_camera"}
    return TTResNet.layer_kwargs(i, **config)


def _input_layout(parameters, i):
    """The layout layer ``i`` receives in TTResNet: the max pool's TILE for layer1, else what layer
    ``i - 1`` emits for its camera groups."""
    if i == 0:
        return ttnn.TILE_LAYOUT
    conv1 = parameters.conv_args[f"layer{i + 1}"][0].conv1
    cameras = _layer_kwargs(parameters, i - 1)["cameras"]
    return TTResLayer.group_output_layout(cameras, conv1.input_height, conv1.input_width)


def _run_and_check(ttnn_model, parameters, device, torch_input, torch_output, layout=ttnn.TILE_LAYOUT):
    """Runs ``ttnn_model`` as TTResNet runs it on this grid, on the whole batch or one camera at a time."""
    camera_inputs = torch_input.split(1) if _per_camera(parameters) else [torch_input]
    outputs = []
    for camera_input in camera_inputs:
        ttnn_output = ttnn_model(to_conv_layout(camera_input, device, ttnn.bfloat16, layout))
        assert (
            ttnn_output.dtype == ttnn_model.output_dtype
        ), f"emits {ttnn_output.dtype}, declares {ttnn_model.output_dtype}"
        outputs.append(from_conv_layout(ttnn_output, (len(camera_input), *torch_output.shape[1:])))
    assert_pcc(torch_output, torch.cat(outputs), 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_bottleneck_layer1(device, reset_seeds, reference_and_parameters):
    reference_model, parameters = reference_and_parameters
    torch_input = torch.randn(NUM_CAMS, 64, IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4)
    torch_output = reference_model.layer1[0].eval()(torch_input)

    # layer1's first block alone, run as TTResNet runs layer1.
    ttnn_model = TTResLayer(
        [parameters.conv_args.layer1[0]],
        [parameters["res_model"]["layer1"][0]],
        device,
        **_layer_kwargs(parameters, 0),
    )
    _run_and_check(ttnn_model, parameters, device, torch_input, torch_output)


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
    _run_and_check(ttnn_model, parameters, device, torch_input, torch_output, layout)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_bottleneck_layer4(device, reset_seeds, reference_and_parameters):
    """layer4's first block at stride 32, with the downsample; for the DCN variants its conv2 is a
    DCN over 512 input channels."""
    reference_model, parameters = reference_and_parameters
    torch_input = torch.randn(NUM_CAMS, 1024, IMAGE_HEIGHT // 16, IMAGE_WIDTH // 16)
    torch_output = reference_model.layer4[0].eval()(torch_input)

    kwargs = _layer_kwargs(parameters, 3)
    assert kwargs.pop("cameras") is None, "TTResNet runs layer4's blocks without camera groups"
    layout = _input_layout(parameters, 3)
    ttnn_model = TTBottleneck(
        parameters.conv_args.layer4[0],
        parameters["res_model"]["layer4"][0],
        device,
        **kwargs,
        input_layout=layout,
    )
    _run_and_check(ttnn_model, parameters, device, torch_input, torch_output, layout)


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
    _run_and_check(ttnn_model, parameters, device, torch_input, torch_output)


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
    _run_and_check(ttnn_model, parameters, device, torch_input, torch_output, layout)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
def test_stem_conv_ignores_hwc_padding(device, reset_seeds, reference_and_parameters, poison):
    """The stem conv reads convert_to_hwc's padding channels, which hold whatever L1 held (#64507),
    even Inf. Their zero weights must cancel them exactly: the output equals that of zero padding
    bit for bit. Not a PCC check, which zeroes NaN before correlating."""
    _, parameters = reference_and_parameters
    if _per_camera(parameters):
        pytest.skip("A reduced grid's stem pads the channels with zeros in DRAM, not with convert_to_hwc")
    conv = parameters.conv_args.conv1
    stem_kwargs = parameters.tt_kwargs
    stem = TTResNet.stem_conv(
        conv, parameters["res_model"].conv1, device, stem_kwargs["weights_dtype"], stem_kwargs["stem_act_block_h"]
    )
    channels = TTResNet.stem_hwc_channels(conv.in_channels)
    pixels = torch.randn(1, 1, IMAGE_HEIGHT * IMAGE_WIDTH, conv.in_channels)

    def run(padding):
        hwc = torch.cat([pixels, torch.full((*pixels.shape[:-1], channels - conv.in_channels), padding)], dim=-1)
        output, _, _ = stem(
            ttnn.from_torch(
                hwc,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=stem.input_memory_config,
            )
        )
        return ttnn.to_torch(output)

    expected = run(0.0)
    actual = run(poison)
    assert torch.isfinite(expected).all()
    assert torch.equal(actual, expected), f"{int((actual != expected).sum())} of {actual.numel()} outputs differ"
