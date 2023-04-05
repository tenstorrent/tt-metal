import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
import os
import numpy as np
import torch
import yaml

class ConvTestParameters:
    def __init__(self, activation_shape, weight_shape, stride_h, stride_w, pad_h, pad_w):
        assert(len(activation_shape) == 4)
        assert(len(weight_shape) == 4)
        self.act_shape = activation_shape
        self.weight_shape = weight_shape
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w

    def print(self, d):
        print(d + "Activation Shape - " + str(self.act_shape))
        print(d + "Weight Shape - " + str(self.weight_shape))
        print(d + "Stride H - " + str(self.stride_h))
        print(d + "Stride W - " + str(self.stride_h))
        print(d + "Pad H - " + str(self.pad_h))
        print(d + "Pad W - " + str(self.pad_w))


def generate_pytorch_golden(conv_test_params):
    ctp = conv_test_params
    A = torch.randn(ctp.act_shape, dtype=torch.bfloat16).float()
    B = torch.randn(ctp.weight_shape, dtype=torch.bfloat16).float()
    C = torch.nn.functional.conv2d(A, B, stride=(ctp.stride_h, ctp.stride_w), padding=(ctp.pad_h, ctp.pad_w))
    return (A,B,C)

def generate_pytorch_golden_tb():
    # sweep over activation sizes, kernel sizes, stride, padding specified in test bench yaml
    with open(os.path.join(sys.path[0], 'conv_tb.yaml'), 'r') as file:
        conv_tb = yaml.safe_load(file)
    pytorch_golden_test_bench = {}
    for act_shape in conv_tb["activation_shapes"]:
        for kernel_size in conv_tb["kernel_sizes"]:
            for stride in conv_tb["strides"]:
                for pad in conv_tb["paddings"]:
                    # check if its a valid test
                    output_shape_h = ((int) (act_shape[2] - kernel_size[1] + 2 * pad[0] / stride[0])) + 1
                    output_shape_w = ((int) (act_shape[3] - kernel_size[2] + 2 * pad[1] / stride[1])) + 1
                    if output_shape_h < 1 or output_shape_w < 1:
                        # invalid parameters
                        continue
                    # weight shape - [K,C,R,S]
                    weight_shape = [kernel_size[0], act_shape[1], kernel_size[1], kernel_size[2]]
                    conv_test_params = ConvTestParameters(act_shape, weight_shape, stride[0], stride[1], pad[0], pad[1])
                    # generate_pytorch_golden returns input, weight and golden output tensors
                    pytorch_golden_test = generate_pytorch_golden(conv_test_params)
                    pytorch_golden_test_bench[conv_test_params] = pytorch_golden_test
    return pytorch_golden_test_bench
