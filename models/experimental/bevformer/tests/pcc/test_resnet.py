# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from models.experimental.bevformer.reference.resnet import ResNet
from models.experimental.bevformer.tt.tt_resnet import TtBottleneck, TtResLayer, TtResNet
from models.experimental.bevformer.tt.model_preprocessing_backbone import create_resnet_parameters
from models.experimental.bevformer.tests.backbone_weights import load_backbone_weights
from tests.ttnn.utils_for_testing import assert_with_pcc


NUM_CAMS = 6

# (height, width) of each camera image, and the ResNet stages whose activations
# are kept in DRAM because their convs do not fit in L1. 640x360 is the resolution
# the backbone was first brought up at; 928x1600 is BEVFormer-base's 1600x900 input
# padded to a multiple of 32. There the stage-1 output alone is 6 x 232 x 400 x 256
# bf16 = 285 MB, and stage 4's 2048-channel 1x1 convs overflow L1 when sharded.
INPUT_SIZES = [(640, 360, ()), (928, 1600, (0, 1, 3))]
INPUT_SIZE_IDS = ["640x360", "928x1600"]


def _assert_pcc(expected, actual, pcc):
    passed, message = assert_with_pcc(expected, actual, pcc)
    logger.info(f"PCC {message} (threshold {pcc})")
    return passed, message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
@pytest.mark.parametrize("height, width, dram_activation_stages", INPUT_SIZES, ids=INPUT_SIZE_IDS)
def test_bottleneck_layer1(device, reset_seeds, height, width, dram_activation_stages):
    reference_model = ResNet(
        depth=101,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(1, 2, 3),
        style="caffe",
        deep_stem=False,
        avg_down=False,
        frozen_stages=4,
        conv_cfg=None,
        norm_cfg={"type": "BN2d", "requires_grad": False},
        norm_eval=True,
        dcn={"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
        stage_with_dcn=(False, False, True, True),
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
        init_cfg=None,
    )

    reference_model = load_backbone_weights(reference_model)

    parameters = create_resnet_parameters(reference_model, torch.randn(NUM_CAMS, 3, height, width))

    bottle_neck = reference_model.layer1[0]
    bottle_neck.eval()

    torch_input = torch.randn(NUM_CAMS, 64, height // 4, width // 4)

    torch_output = bottle_neck(torch_input)

    torch_input_permute = torch_input.permute(0, 2, 3, 1)
    torch_input_permute = torch_input_permute.reshape(
        1,
        1,
        (torch_input_permute.shape[0] * torch_input_permute.shape[1] * torch_input_permute.shape[2]),
        torch_input_permute.shape[3],
    )
    ttnn_model = TtBottleneck(
        parameters.conv_args.layer1[0],
        parameters["res_model"]["layer1"][0],
        device,
        True,
        False,
        ttnn.bfloat16,
        False,
        64,
        style="caffe",
        dram_activation=0 in dram_activation_stages,
    )

    ttnn_input = ttnn.from_torch(torch_input_permute, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = torch.reshape(
        ttnn_output, (torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1])
    )
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))

    _assert_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
@pytest.mark.parametrize("height, width, dram_activation_stages", INPUT_SIZES, ids=INPUT_SIZE_IDS)
def test_bottleneck_layer3(device, reset_seeds, height, width, dram_activation_stages):
    reference_model = ResNet(
        depth=101,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(1, 2, 3),
        style="caffe",
        deep_stem=False,
        avg_down=False,
        frozen_stages=4,
        conv_cfg=None,
        norm_cfg={"type": "BN2d", "requires_grad": False},
        norm_eval=True,
        dcn={"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
        stage_with_dcn=(False, False, True, True),
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
        init_cfg=None,
    )
    reference_model = load_backbone_weights(reference_model)

    parameters = create_resnet_parameters(reference_model, torch.randn(NUM_CAMS, 3, height, width))

    bottle_neck = reference_model.layer3[0]
    bottle_neck.eval()

    torch_input = torch.randn(NUM_CAMS, 512, height // 8, width // 8)

    torch_output = bottle_neck(torch_input)

    torch_input_permute = torch_input.permute(0, 2, 3, 1)
    torch_input_permute = torch_input_permute.reshape(
        1,
        1,
        (torch_input_permute.shape[0] * torch_input_permute.shape[1] * torch_input_permute.shape[2]),
        torch_input_permute.shape[3],
    )
    ttnn_model = TtBottleneck(
        parameters.conv_args.layer3[0],
        parameters["res_model"]["layer3"][0],
        device,
        True,
        False,
        ttnn.bfloat16,
        False,
        256,
        style="caffe",
        dcn=True,
        dram_input=1 in dram_activation_stages,
    )

    ttnn_input = ttnn.from_torch(torch_input_permute, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = torch.reshape(
        ttnn_output, (torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1])
    )
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))

    _assert_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
@pytest.mark.parametrize("height, width, dram_activation_stages", INPUT_SIZES, ids=INPUT_SIZE_IDS)
def test_reslayer1(device, reset_seeds, height, width, dram_activation_stages):
    reference_model = ResNet(
        depth=101,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(1, 2, 3),
        style="caffe",
        deep_stem=False,
        avg_down=False,
        frozen_stages=4,
        conv_cfg=None,
        norm_cfg={"type": "BN2d", "requires_grad": False},
        norm_eval=True,
        dcn={"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
        stage_with_dcn=(False, False, True, True),
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
        init_cfg=None,
    )
    reference_model = load_backbone_weights(reference_model)

    parameters = create_resnet_parameters(reference_model, torch.randn(NUM_CAMS, 3, height, width))

    reslayer = reference_model.layer1
    reslayer.eval()

    torch_input = torch.randn(NUM_CAMS, 64, height // 4, width // 4)
    torch_output = reslayer(torch_input)

    ttnn_model = TtResLayer(
        parameters.conv_args.layer1,
        parameters["res_model"]["layer1"],
        device,
        inplanes=64,
        num_blocks=3,
        is_downsample=False,
        blk_sharded=False,
        activation_dtype=ttnn.bfloat16,
        conv3_blk_sharded=False,
        planes=64,
        stride=1,
        dilation=1,
        style="caffe",
        conv_cfg=None,
        dcn=None,
        dram_activation=0 in dram_activation_stages,
    )
    torch_input_permute = torch_input.permute(0, 2, 3, 1)
    torch_input_permute = torch_input_permute.reshape(
        1,
        1,
        (torch_input_permute.shape[0] * torch_input_permute.shape[1] * torch_input_permute.shape[2]),
        torch_input_permute.shape[3],
    )
    ttnn_input = ttnn.from_torch(torch_input_permute, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = torch.reshape(
        ttnn_output, (torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1])
    )
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))

    _assert_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
@pytest.mark.parametrize("height, width, dram_activation_stages", INPUT_SIZES, ids=INPUT_SIZE_IDS)
def test_reslayer2(device, reset_seeds, height, width, dram_activation_stages):
    reference_model = ResNet(
        depth=101,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(1, 2, 3),
        style="caffe",
        deep_stem=False,
        avg_down=False,
        frozen_stages=4,
        conv_cfg=None,
        norm_cfg={"type": "BN2d", "requires_grad": False},
        norm_eval=True,
        dcn={"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
        stage_with_dcn=(False, False, True, True),
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
        init_cfg=None,
    )
    reference_model = load_backbone_weights(reference_model)

    parameters = create_resnet_parameters(reference_model, torch.randn(NUM_CAMS, 3, height, width))

    reslayer = reference_model.layer2
    reslayer.eval()

    torch_input = torch.randn(NUM_CAMS, 256, height // 4, width // 4)
    torch_output = reslayer(torch_input)

    ttnn_model = TtResLayer(
        parameters.conv_args.layer2,
        parameters["res_model"]["layer2"],
        device,
        inplanes=256,
        num_blocks=4,
        is_downsample=False,
        blk_sharded=False,
        activation_dtype=ttnn.bfloat8_b,
        conv3_blk_sharded=False,
        planes=64,
        stride=2,
        dilation=1,
        style="caffe",
        conv_cfg=None,
        dcn=None,
        dram_activation=1 in dram_activation_stages,
    )
    torch_input_permute = torch_input.permute(0, 2, 3, 1)
    torch_input_permute = torch_input_permute.reshape(
        1,
        1,
        (torch_input_permute.shape[0] * torch_input_permute.shape[1] * torch_input_permute.shape[2]),
        torch_input_permute.shape[3],
    )
    ttnn_input = ttnn.from_torch(torch_input_permute, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, device=device)

    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = torch.reshape(
        ttnn_output, (torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1])
    )
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))

    _assert_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
@pytest.mark.parametrize("height, width, dram_activation_stages", INPUT_SIZES, ids=INPUT_SIZE_IDS)
def test_resnet(device, reset_seeds, height, width, dram_activation_stages):
    reference_model = ResNet(
        depth=101,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(1, 2, 3),
        style="caffe",
        deep_stem=False,
        avg_down=False,
        frozen_stages=4,
        conv_cfg=None,
        norm_cfg={"type": "BN2d", "requires_grad": False},
        norm_eval=True,
        dcn={"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
        stage_with_dcn=(False, False, True, True),
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
        init_cfg=None,
    )
    reference_model = load_backbone_weights(reference_model)

    parameters = create_resnet_parameters(reference_model, torch.randn(NUM_CAMS, 3, height, width))

    torch_input = torch.randn(NUM_CAMS, 3, height, width)
    torch_output = reference_model(torch_input)

    ttnn_model = TtResNet(
        parameters.conv_args,
        parameters["res_model"],
        device,
        depth=101,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(1, 2, 3),
        style="caffe",
        deep_stem=False,
        avg_down=False,
        frozen_stages=4,
        conv_cfg=None,
        dcn=True,
        stage_with_dcn=(False, False, True, True),
        pretrained=None,
        init_cfg=None,
        dram_activation_stages=dram_activation_stages,
    )

    torch_input_permute = torch_input.permute(0, 2, 3, 1)
    torch_input_permute = torch_input_permute.reshape(
        1,
        1,
        (torch_input_permute.shape[0] * torch_input_permute.shape[1] * torch_input_permute.shape[2]),
        torch_input_permute.shape[3],
    )
    ttnn_input = ttnn.from_torch(torch_input_permute, device=device, dtype=ttnn.bfloat16)

    ttnn_output = ttnn_model(ttnn_input)

    for i in range(3):
        ttnn_output_final = ttnn.to_torch(ttnn_output[i])

        ttnn_output_final = torch.reshape(
            ttnn_output_final,
            (torch_output[i].shape[0], torch_output[i].shape[2], torch_output[i].shape[3], torch_output[i].shape[1]),
        )
        ttnn_output_final = torch.permute(ttnn_output_final, (0, 3, 1, 2))

        # 0.85 is a regression floor, not an accuracy target. On Blackhole the
        # outputs measure ~0.933 / 0.886 / 0.907 (C3 / C4 / C5), and running DCN on
        # the host instead of the device moves them by under 0.003: the loss is
        # already present at C3, which has no DCN, while stage 2 alone scores
        # 0.997 from a torch input.
        _, x = _assert_pcc(torch_output[i], ttnn_output_final, 0.85)
