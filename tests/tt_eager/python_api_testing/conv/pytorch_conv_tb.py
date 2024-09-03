# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import torch
import yaml

from enum import Enum


class TestLevel(Enum):
    INPUT_TENSOR_CREATE = 1
    OP_OUTPUT_TENSOR_CREATE = 2
    OP_PROGRAM_CREATE = 3
    OP_FULL_COMPUTE = 4


class ConvOpTestParameters:
    def __init__(self, conv_params, test_level):
        self.conv_params = conv_params
        self.test_level = test_level

    def to_string(self):
        cp = self.conv_params
        line = "Act_shape=" + str(cp.act_shape) + ", Weight_shape=" + str(cp.weight_shape)
        line += ", Stride_h=" + str(cp.stride_h) + ", Stride_w=" + str(cp.stride_w)
        line += ", Pad_h=" + str(cp.pad_h) + ", Pad_w=" + str(cp.pad_w)
        line += ", TestLevel=" + str(TestLevel(self.test_level).name)
        return line

    def print(self, d):
        print(d + self.to_string())


class ConvTestParameters:
    def __init__(self, activation_shape, weight_shape, stride_h, stride_w, pad_h, pad_w):
        assert len(activation_shape) == 4
        assert len(weight_shape) == 4
        self.act_shape = activation_shape
        self.weight_shape = weight_shape
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w


def generate_pytorch_golden(conv_test_params):
    ctp = conv_test_params
    A = torch.randn(ctp.act_shape, dtype=torch.bfloat16).float()
    B = torch.randn(ctp.weight_shape, dtype=torch.bfloat16).float()
    C = torch.nn.functional.conv2d(A, B, stride=(ctp.stride_h, ctp.stride_w), padding=(ctp.pad_h, ctp.pad_w))
    return (A, B, C)


def generate_conv_tb():
    # sweep over activation sizes, kernel sizes, stride, padding specified in test bench yaml
    with open(
        os.path.join(os.environ["TT_METAL_HOME"], "tests/tt_eager/python_api_testing/conv/conv_sweep_params.yaml"), "r"
    ) as file:
        conv_tb = yaml.safe_load(file)
    conv_op_test_bench = []
    for act_shape in conv_tb["activation_shapes"]:
        for kernel_size in conv_tb["kernel_sizes"]:
            for stride in conv_tb["strides"]:
                for pad in conv_tb["paddings"]:
                    H = act_shape[2]
                    W = act_shape[3]
                    R = kernel_size[1]
                    S = kernel_size[2]
                    # check if its a valid test
                    if (H - R + 2 * pad[0]) < 1 or (W - S + 2 * pad[1]) < 1:
                        # invalid parameters
                        continue
                    # weight shape - [K,C,R,S]
                    weight_shape = [kernel_size[0], act_shape[1], kernel_size[1], kernel_size[2]]
                    conv_test_params = ConvTestParameters(act_shape, weight_shape, stride[0], stride[1], pad[0], pad[1])
                    op_full_compute = (R == S) and (pad[0] == pad[1]) and (H == W)
                    # if(H >= 5 and act_shape[1] == 64):
                    #    op_full_compute = False
                    if op_full_compute:
                        conv_op_test_params = ConvOpTestParameters(conv_test_params, TestLevel.OP_FULL_COMPUTE)
                    else:
                        conv_op_test_params = ConvOpTestParameters(conv_test_params, TestLevel.INPUT_TENSOR_CREATE)

                    conv_op_test_bench.append(conv_op_test_params)

    # Dump test bench to yaml file for viewing

    # with open(os.path.join(os.environ['TT_METAL_HOME'], 'tests/python_api_testing/conv/generated_conv_tb.yaml'), 'w') as file:
    #     mm_yaml = yaml.dump(mm_tb_yaml_dict, file)
    # print("Total number of MM tests generated - " + str(len(mm_tb_list)))
    return conv_op_test_bench


def generate_conv_tb_with_pytorch_golden(conv_test_bench):
    test_bench_with_pytorch_golden = {}
    # Generate pytorch golden result for each test in testbench
    for conv_op_test_params in conv_test_bench:
        conv_test_params = conv_op_test_params.conv_params
        # print("Test with following parameters - ")
        # conv_op_test_params.print("   ")
        # generate_pytorch_golden returns input, weight and golden output tensors
        pytorch_golden_test = generate_pytorch_golden(conv_test_params)
        test_bench_with_pytorch_golden[conv_op_test_params] = pytorch_golden_test
    return test_bench_with_pytorch_golden
