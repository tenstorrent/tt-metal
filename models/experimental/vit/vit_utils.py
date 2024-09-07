# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.helper_funcs import Linear as linear
from models.utility_functions import torch_to_tt_tensor_rm
import ttnn


def make_address(base_address, op_name):
    return op_name if base_address == "" else f"{base_address}.{op_name}"


def make_linear(in_feature, out_feature, op_name, state_dict, base_address, device, mem_config=ttnn.DRAM_MEMORY_CONFIG):
    q_weight = state_dict[make_address(base_address, f"{op_name}.weight")]
    q_weight = torch_to_tt_tensor_rm(q_weight, device)
    if make_address(base_address, f"{op_name}.bias") in state_dict:
        q_bias = state_dict[make_address(base_address, f"{op_name}.bias")]
        q_bias = torch_to_tt_tensor_rm(q_bias, device)
    else:
        q_bias = None
    return linear(in_feature, out_feature, weight=q_weight, bias=q_bias, output_mem_config=mem_config)
