# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import yaml
from tests.tt_eager.python_api_testing.conv.pytorch_conv_tb import ConvTestParameters, generate_conv_tb
from tt_lib.utils import _nearest_32


def generate_mm_tb_using_conv_tb():
    print("Sweeping over convolution sizes and parameters in conv_tb.yaml.")
    print("Generating MM test bench with conv sweep parameters.")
    mm_tb_list = []
    conv_test_bench = generate_conv_tb()

    for ctp_ in conv_test_bench:
        ctp = ctp_.conv_params
        conv_out_h = ((int)((ctp.act_shape[2] - ctp.weight_shape[2] + 2 * ctp.pad_h) / ctp.stride_h)) + 1
        conv_out_w = ((int)((ctp.act_shape[3] - ctp.weight_shape[3] + 2 * ctp.pad_w) / ctp.stride_w)) + 1
        M = conv_out_h * conv_out_w
        K = ctp.weight_shape[1] * ctp.weight_shape[2] * ctp.weight_shape[3]
        N = ctp.weight_shape[0]
        # pad M, K, N to nearest multiple of 32
        mm_test_params = [_nearest_32(M), _nearest_32(K), _nearest_32(N)]
        if mm_test_params not in mm_tb_list:
            mm_tb_list.append(mm_test_params)

    mm_tb_yaml_dict = [{"MM test params [M,K,N]": mm_tb_list}]
    # Dump test bench to yaml file for viewing
    with open(
        os.path.join(os.environ["TT_METAL_HOME"], "tests/python_api_testing/conv/generated_mm_tb.yaml"), "w"
    ) as file:
        mm_yaml = yaml.dump(mm_tb_yaml_dict, file)
    print("Total number of MM tests generated - " + str(len(mm_tb_list)))
    return mm_tb_list
