# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import timm
import transformers

from loguru import logger
from models.experimental.functional_vovnet.ref import torch_functional_vovnet
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


def test_effective_se_module(reset_seeds):
    model = timm.create_model("ese_vovnet19b_dw", pretrained=True).eval()
    torch_model = model.stages[0].blocks[0].attn
    input = torch.randn(1, 256, 56, 56)
    channels = 256

    torch_output = torch_model(input)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_functional_vovnet.effective_se_module(input, channels, parameters, add_maxpool=False)

    assert_with_pcc(torch_output, output, 1.0)


def test_batch_norm_act_2d(reset_seeds):
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    torch_model = model.stem[0].bn
    running_mean = torch_model.running_mean
    running_var = torch_model.running_var
    torch_model.eval()

    input = torch.randn(1, 64, 32, 32)
    num_features = 64
    torch_output = torch_model(input)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_functional_vovnet.batch_norm_act_2d(
        input,
        num_features,
        running_mean,
        running_var,
        parameters,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        apply_act=True,
        act_kwargs={},
        inplace=True,
    )

    assert_with_pcc(torch_output, output, 1.0)


def test_conv_norm_act(reset_seeds, imagenet_sample_input):
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    torch_model = model.stem[0]
    torch_model.eval()

    input = imagenet_sample_input
    in_channels = 3
    out_channels = 64
    torch_output = torch_model(input)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        convert_to_ttnn=lambda *_: False,
    )
    running_mean = torch_model.bn.running_mean
    running_var = torch_model.bn.running_var
    output = torch_functional_vovnet.conv_norm_act(
        input,
        in_channels,
        out_channels,
        parameters,
        running_mean,
        running_var,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
        bias=False,
        channel_multiplier=1.0,
        apply_act=True,
        groups=1,
    )

    assert_with_pcc(torch_output, output, 1.0)


def test_seperable_conv_norm_act(reset_seeds):
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    torch_model = model.stem[2]
    torch_model.eval()

    input = torch.randn(1, 64, 112, 112)
    in_channels = 64
    out_channels = 64
    torch_output = torch_model(input)

    parameters = preprocess_model_parameters(initialize_model=lambda: torch_model, convert_to_ttnn=lambda *_: False)
    running_mean = model.stem[2].bn.running_mean
    running_var = model.stem[2].bn.running_var
    num_features = 64

    output = torch_functional_vovnet.seperable_conv_norm_act(
        input,
        in_channels,
        out_channels,
        parameters,
        running_mean,
        running_var,
        groups=64,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False,
        channel_multiplier=1.0,
        pw_kernel_size=1,
        apply_act=True,
    )

    assert_with_pcc(torch_output, output, 1.0)


def test_sequential_append_list(reset_seeds):
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    torch_model = model.stages[0].blocks[0].conv_mid
    torch_model.eval()

    input = torch.randn(1, 128, 56, 56)
    inputs = torch.randn(1, 64, 56, 56)
    torch_output = torch_model(input, [inputs])

    concat_list = []
    parameters = preprocess_model_parameters(initialize_model=lambda: torch_model, convert_to_ttnn=lambda *_: False)
    running_mean = model.stages[0].blocks[0].conv_mid
    running_var = model.stages[0].blocks[0].conv_mid

    output = torch_functional_vovnet.sequential_append_list(
        input, parameters, running_mean, running_var, [inputs], in_channels=128, layer_per_block=3, groups=128
    )

    assert_with_pcc(torch_output, output, 1.0)


def test_select_adaptive_pool2d(reset_seeds):
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    torch_model = model.head.global_pool
    torch_model = torch_model.eval()
    input = torch.randn(1, 1024, 7, 7)
    torch_output = torch_model(input)

    parameters = preprocess_model_parameters(initialize_model=lambda: torch_model, convert_to_ttnn=lambda *_: False)
    output = torch_functional_vovnet.select_adaptive_pool2d(
        input, output_size=1, pool_type="fast", flatten=False, input_fmt="NCHW"
    )

    assert_with_pcc(torch_output, output, 1.0)


def test_classifier_head(reset_seeds):
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    torch_model = model.head
    torch_model.eval()

    input = torch.randn(1, 1024, 7, 7)
    in_features = 1024
    num_classes = 1000
    torch_output = torch_model(input)

    parameters = preprocess_model_parameters(initialize_model=lambda: torch_model, convert_to_ttnn=lambda *_: False)
    output = torch_functional_vovnet.classifier_head(
        input, parameters, in_features, num_classes, pool_type="avg", use_conv=False, input_fmt="NCHW", pre_logits=False
    )

    assert_with_pcc(torch_output, output, 0.9999)


def test_osa_block(reset_seeds):
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    torch_model = model.stages[0].blocks[0]
    torch_model.eval()

    input = torch.randn(1, 64, 56, 56)
    torch_output = torch_model(input)

    parameters = preprocess_model_parameters(initialize_model=lambda: torch_model, convert_to_ttnn=lambda *_: False)
    running_mean = model.stages[0].blocks[0]
    running_var = model.stages[0].blocks[0]

    output = torch_functional_vovnet.osa_block(
        input,
        parameters,
        running_mean,
        running_var,
        in_chs=64,
        mid_chs=128,
        out_chs=256,
        layer_per_block=3,
        residual=False,
        depthwise=True,
    )

    assert_with_pcc(torch_output, output, 1.0)


def test_osa_stage(reset_seeds):
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    torch_model = model.stages[0]
    torch_model.eval()

    input = torch.randn(1, 64, 56, 56)
    torch_output = torch_model(input)

    parameters = preprocess_model_parameters(initialize_model=lambda: torch_model, convert_to_ttnn=lambda *_: False)

    output = torch_functional_vovnet.osa_stage(
        input,
        parameters,
        torch_model,
        torch_model,
        in_chs=64,
        mid_chs=128,
        out_chs=256,
        block_per_stage=1,
        layer_per_block=3,
        downsample=False,
        residual=True,
        depthwise=False,
    )

    assert_with_pcc(torch_output, output, 1.0)


def test_vovnet(reset_seeds, imagenet_sample_input):
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    torch_model = model
    torch_model.eval()

    input = imagenet_sample_input
    torch_output = torch_model(input)

    parameters = preprocess_model_parameters(initialize_model=lambda: torch_model, convert_to_ttnn=lambda *_: False)

    running_mean = model
    running_var = model

    cfg = dict(
        stem_chs=[64, 64, 64],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[256, 512, 768, 1024],
        layer_per_block=3,
        block_per_stage=[1, 1, 1, 1],
        residual=True,
        depthwise=True,
        attn="ese",
    )

    output = torch_functional_vovnet.vovnet(
        cfg,
        input,
        parameters,
        running_mean,
        running_var,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        output_stride=32,
        stem_stride=4,
        depthwise=True,
    )
    assert_with_pcc(torch_output, output, 0.99)
