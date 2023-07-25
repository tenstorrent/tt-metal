import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

import tt_lib as ttl
from tt_lib.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, _nearest_y, convert_weights_2d_matrix
from python_api_testing.models.utility_functions import print_diff_argmax, is_close, comp_pcc
from tests.python_api_testing.conv.conv_unit_test_utils import create_conv_act_tensor, create_conv_weight_tensor
from python_api_testing.models.resnet.utils import conv3x3, conv1x1, fold_bn_to_conv, fold_bn_to_conv_weights_bias
from python_api_testing.models.conv_on_device_utils import is_conv_supported_on_device, run_conv_on_device_wrapper
import torch
import torch.nn as nn
from torchvision import models

def make_conv_bn_pairs_in_one_resnet_block(inplanes, planes, base_address, state_dict, stride=1):
    norm_layer = nn.BatchNorm2d
    expansion: int = 4
    base_width = 64.0
    dilation = 1
    groups = 1
    width = int(planes * (base_width / 64.0)) * groups

    conv1_weight = state_dict[f"{base_address}.conv1.weight"]
    conv1_bias = None
    conv1 = conv1x1(inplanes, width, state_dict=state_dict, base_address=f"{base_address}.conv1")

    bn1 = norm_layer(width)
    bn1.weight = nn.Parameter(state_dict[f"{base_address}.bn1.weight"])
    bn1.bias = nn.Parameter(state_dict[f"{base_address}.bn1.bias"])
    bn1.running_mean = nn.Parameter(state_dict[f"{base_address}.bn1.running_mean"])
    bn1.running_var = nn.Parameter(state_dict[f"{base_address}.bn1.running_var"])
    bn1.num_batches_tracked = nn.Parameter(state_dict[f"{base_address}.bn1.num_batches_tracked"], requires_grad=False)
    bn1.eval()

    conv2_weight = state_dict[f"{base_address}.conv2.weight"]
    conv2_bias = None
    conv2 = conv3x3(width, width, stride, groups, dilation, state_dict=state_dict, base_address=f"{base_address}.conv2")

    bn2 = norm_layer(width)
    bn2.weight = nn.Parameter(state_dict[f"{base_address}.bn2.weight"])
    bn2.bias = nn.Parameter(state_dict[f"{base_address}.bn2.bias"])
    bn2.running_mean = nn.Parameter(state_dict[f"{base_address}.bn2.running_mean"])
    bn2.running_var = nn.Parameter(state_dict[f"{base_address}.bn2.running_var"])
    bn2.num_batches_tracked = nn.Parameter(state_dict[f"{base_address}.bn2.num_batches_tracked"], requires_grad=False)
    bn2.eval()

    conv3_weight = state_dict[f"{base_address}.conv3.weight"]
    conv3_bias = None
    conv3 = conv1x1(width, planes * expansion, state_dict=state_dict, base_address=f"{base_address}.conv3")

    bn3 = norm_layer(planes * expansion)
    bn3.weight = nn.Parameter(state_dict[f"{base_address}.bn3.weight"])
    bn3.bias = nn.Parameter(state_dict[f"{base_address}.bn3.bias"])
    bn3.running_mean = nn.Parameter(state_dict[f"{base_address}.bn3.running_mean"])
    bn3.running_var = nn.Parameter(state_dict[f"{base_address}.bn3.running_var"])
    bn3.num_batches_tracked = nn.Parameter(state_dict[f"{base_address}.bn3.num_batches_tracked"], requires_grad=False)
    bn3.eval()

    return [(conv1, bn1), (conv2, bn2), (conv3, bn3)]

def test_resnet50_convs_with_folded_batch_norm():
    with torch.no_grad():
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        ttl.device.SetDefaultDevice(device)
        host = ttl.device.GetHost()

        torch.manual_seed(1234)
        torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        torch_resnet50.eval()
        state_dict = torch_resnet50.state_dict()
        layer_names = ["layer1", "layer2", "layer3", "layer4"]
        layer_planes = [64, 128, 256, 512]
        layer_blocks = [3, 4, 6, 3]
        layer_strides = [1, 2, 2, 2]
        conv_bn_pairs = []
        inplanes = 64
        base_address_with_dot = ""
        expansion = 4
        norm_layer = nn.BatchNorm2d

        # first conv and batch norm
        conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.weight = nn.Parameter(state_dict[f"{base_address_with_dot}conv1.weight"])
        bn1 = norm_layer(inplanes) # batch norm
        bn1.weight = nn.Parameter(state_dict[f"{base_address_with_dot}bn1.weight"])
        bn1.bias = nn.Parameter(state_dict[f"{base_address_with_dot}bn1.bias"])
        bn1.running_mean = nn.Parameter(state_dict[f"{base_address_with_dot}bn1.running_mean"])
        bn1.running_var = nn.Parameter(state_dict[f"{base_address_with_dot}bn1.running_var"])
        bn1.num_batches_tracked = nn.Parameter(state_dict[f"{base_address_with_dot}bn1.num_batches_tracked"], requires_grad=False)
        bn1.eval()
        conv_bn_pairs.append((conv1, bn1))
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # will run maxpool to get the correct shape for next conv
        for i, name in enumerate(layer_names):
            planes = layer_planes[i]
            stride = layer_strides[i]
            blocks = layer_blocks[i]
            conv_bn_pairs.extend(make_conv_bn_pairs_in_one_resnet_block(inplanes, planes, f"{base_address_with_dot}{name}.0", state_dict, stride))
            inplanes = planes * expansion
            for _ in range(1, blocks):
                conv_bn_pairs.extend(make_conv_bn_pairs_in_one_resnet_block(inplanes, planes,f"{base_address_with_dot}{name}.{_}", state_dict, 1))

        x_shape = [1, 3, 224, 224]

        for i, conv_bn_pair in enumerate(conv_bn_pairs):
            conv = conv_bn_pair[0]
            bn = conv_bn_pair[1]
            x = torch.randn(x_shape, dtype=torch.bfloat16).float()
            # Run pytorch golden reference -> Conv followed by BN
            x_golden = conv(x)
            x_golden = bn(x_golden)

            # Fold batchnorm into conv weights and bias
            conv_weight, conv_bias = fold_bn_to_conv_weights_bias(conv.weight, bn)

            # Run pytorch conv with folded bn
            conv.weight = nn.Parameter(conv_weight)
            conv.bias = nn.Parameter(conv_bias)
            x_pytorch_folded_bn = conv(x)

            # Compare pytorch golden vs pytorch with folded bn
            assert(x_pytorch_folded_bn.shape == x_golden.shape)
            passing_pcc, output_pcc = comp_pcc(x_golden, x_pytorch_folded_bn, 0.99)
            print("Passing (Pytorch golden vs Pytorch conv with folden batchnorm)=", passing_pcc)
            print("Output pcc=", output_pcc)
            assert passing_pcc

            # Run conv on device with folded batch norm
            conv_params = [conv.out_channels, conv.in_channels, conv.kernel_size[0], conv.kernel_size[1], conv.stride[0], conv.stride[1], conv.padding[0], conv.padding[1], 1, 1]
            assert(is_conv_supported_on_device(conv_params))
            conv_on_device = run_conv_on_device_wrapper(conv_weight.reshape(-1).tolist(), conv_params, device, host, conv_bias.reshape(-1).tolist())
            x_on_device = create_conv_act_tensor(x, x_shape[0], x_shape[1], x_shape[2], x_shape[3]).to(device)
            x_on_device = conv_on_device(x_on_device)
            # Copy output to host and convert tt tensor to pytorch tensor
            x_result = x_on_device.to(host)
            conv_output_shape_cl = [1,x_result.shape()[2],x_result.shape()[3],x_result.shape()[1]]
            out_result = torch.tensor(x_result.data()).reshape(conv_output_shape_cl)
            out_result = torch.transpose(out_result, 2, 3)
            out_result = torch.transpose(out_result, 1, 2)

            # Compare pytorch golden vs conv with folded batchnorm on device
            assert(out_result.shape == x_golden.shape)
            passing_pcc, output_pcc = comp_pcc(x_golden, out_result, 0.99)
            print("Passing (Pytorch golden vs Conv with folden batchnorm on device)=", passing_pcc)
            print("Output pcc=", output_pcc)
            assert passing_pcc

            if(i == 0):
                # run maxpool to get the correct shape for next conv
                x_golden = maxpool(x_golden)
            x_shape = x_golden.shape # for next iteration
    ttl.device.CloseDevice(device)
