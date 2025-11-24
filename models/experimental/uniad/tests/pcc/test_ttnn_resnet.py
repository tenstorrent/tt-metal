# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
import pytest

import ttnn
from models.experimental.uniad.reference.resnet import ResNet, ModulatedDeformConv2dPack
from models.experimental.uniad.tt.ttnn_resnet import TtBottleneck, TtResLayer, TtResNet
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
)
from models.experimental.uniad.common import load_torch_model


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, ResNet):
        if isinstance(model, ResNet):
            parameters["res_model"] = {}

        # Initial conv + bn
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["res_model"]["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
        }

        # Loop over all layers (layer1 to layer4)
        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            prefix = f"layer{layer_idx}"  # _{block_idx}"
            parameters["res_model"][prefix] = {}
            for block_idx, block in enumerate(layer):
                parameters["res_model"][prefix][block_idx] = {}

                # conv1, conv2, conv3
                for conv_name in ["conv1", "conv2", "conv3"]:
                    conv = getattr(block, conv_name)
                    if isinstance(conv, ModulatedDeformConv2dPack):
                        parameters["res_model"][prefix][block_idx][conv_name] = {}
                        parameters["res_model"][prefix][block_idx][conv_name]["weight"] = conv.weight
                        parameters["res_model"][prefix][block_idx][conv_name]["bias"] = conv.bias
                        parameters["res_model"][prefix][block_idx][conv_name]["conv_offset"] = {
                            "weight": ttnn.from_torch(conv.conv_offset.weight, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(conv.conv_offset.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

                        bn = getattr(block, f"bn{conv_name[-1]}")
                        channel_size = bn.num_features

                        # Extract PyTorch tensors
                        weight_torch = bn.weight if bn.affine else None
                        bias_torch = bn.bias if bn.affine else None
                        batch_mean_torch = bn.running_mean
                        batch_var_torch = bn.running_var

                        # Reshape for broadcast compatibility (1, C, 1, 1)
                        batch_mean_torch = batch_mean_torch.view(1, channel_size, 1, 1)
                        batch_var_torch = batch_var_torch.view(1, channel_size, 1, 1)
                        weight_torch = weight_torch.view(1, channel_size, 1, 1) if weight_torch is not None else None
                        bias_torch = bias_torch.view(1, channel_size, 1, 1) if bias_torch is not None else None

                        parameters["res_model"][prefix][block_idx]["bn2"] = {}
                        weight = (
                            ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                            if weight_torch is not None
                            else None
                        )
                        parameters["res_model"][prefix][block_idx]["bn2"]["weight"] = weight

                        bias = (
                            ttnn.from_torch(bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                            if bias_torch is not None
                            else None
                        )
                        parameters["res_model"][prefix][block_idx]["bn2"]["bias"] = bias

                        running_mean = ttnn.from_torch(batch_mean_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                        parameters["res_model"][prefix][block_idx]["bn2"]["running_mean"] = running_mean

                        running_var = ttnn.from_torch(batch_var_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                        parameters["res_model"][prefix][block_idx]["bn2"]["running_var"] = running_var

                        parameters["res_model"][prefix][block_idx]["bn2"][
                            "eps"
                        ] = bn.eps  # scalar, used directly in ops

                    else:
                        bn = getattr(block, f"bn{conv_name[-1]}")
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["res_model"][prefix][block_idx][conv_name] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

                # downsample (if present)
                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    if isinstance(ds, torch.nn.Sequential):
                        conv = ds[0]
                        bn = ds[1]
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["res_model"][prefix][block_idx]["downsample"] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }
    return parameters


def create_uniad_model_resnet(model: ResNet, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_uniad_bottle_neck_layer_1(device, reset_seeds, model_location_generator):
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

    reference_model = load_torch_model(
        torch_model=reference_model, layer="img_backbone", model_location_generator=model_location_generator
    )

    parameters = create_uniad_model_resnet(reference_model, torch.randn(6, 3, 640, 360))

    bottle_neck = reference_model.layer1[0]
    bottle_neck.eval()

    torch_input = torch.randn(6, 64, 160, 90)

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
    )

    ttnn_input = ttnn.from_torch(torch_input_permute, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = torch.reshape(
        ttnn_output, (torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1])
    )
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_uniad_bottle_neck_layer3(device, reset_seeds, model_location_generator):
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
    reference_model = load_torch_model(
        torch_model=reference_model, layer="img_backbone", model_location_generator=model_location_generator
    )

    parameters = create_uniad_model_resnet(reference_model, torch.randn(6, 3, 640, 360))

    bottle_neck = reference_model.layer3[0]
    bottle_neck.eval()

    torch_input = torch.randn(6, 512, 80, 45)

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
    )

    ttnn_input = ttnn.from_torch(torch_input_permute, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = torch.reshape(
        ttnn_output, (torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1])
    )
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))

    assert_with_pcc(torch_output, ttnn_output, 0.88)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_uniad_reslayer1(device, reset_seeds, model_location_generator):
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
    reference_model = load_torch_model(
        torch_model=reference_model, layer="img_backbone", model_location_generator=model_location_generator
    )

    parameters = create_uniad_model_resnet(reference_model, torch.randn(6, 3, 640, 360))

    reslayer = reference_model.layer1
    reslayer.eval()

    torch_input = torch.randn(6, 64, 160, 90)
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

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_uniad_reslayer2(device, reset_seeds, model_location_generator):
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
    reference_model = load_torch_model(
        torch_model=reference_model, layer="img_backbone", model_location_generator=model_location_generator
    )

    parameters = create_uniad_model_resnet(reference_model, torch.randn(6, 3, 640, 360))

    reslayer = reference_model.layer2
    reslayer.eval()

    torch_input = torch.randn(6, 256, 160, 90)
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

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_uniad_resnet(device, reset_seeds, model_location_generator):
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
    reference_model = load_torch_model(
        torch_model=reference_model, layer="img_backbone", model_location_generator=model_location_generator
    )

    parameters = create_uniad_model_resnet(reference_model, torch.randn(6, 3, 640, 360))

    torch_input = torch.randn(6, 3, 640, 360)
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

        # We have a issue on this, issue - https://github.com/tenstorrent/tt-metal/issues/26185
        _, x = assert_with_pcc(torch_output[i], ttnn_output_final, 0.16)
