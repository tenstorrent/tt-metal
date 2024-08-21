# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from ttnn.model_preprocessing import infer_ttnn_module_args

from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn2


def create_unet_model_parameters(model: unet_shallow_torch.UNet, input_tensor: torch.Tensor, groups: int, device):
    parameters = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.keys():
        parameters[key].module = getattr(model, key)

    parameters.c1["conv_blocking_and_parallelization_config_override"] = (
        {"act_block_h": 32} if groups > 1 else ({"act_block_h": 5 * 32})
    )
    parameters.c1_2["conv_blocking_and_parallelization_config_override"] = (
        {"act_block_h": 32} if groups > 1 else ({"act_block_h": 5 * 32})
    )
    parameters.c4["conv_blocking_and_parallelization_config_override"] = {"num_cores_nhw": 42} if groups > 1 else None
    parameters.c4_2["conv_blocking_and_parallelization_config_override"] = {"num_cores_nhw": 42} if groups > 1 else None

    parameters.bnc["conv_blocking_and_parallelization_config_override"] = None
    parameters.bnc_2["conv_blocking_and_parallelization_config_override"] = None

    parameters.c5["conv_blocking_and_parallelization_config_override"] = {"num_cores_nhw": 42} if groups > 1 else None
    parameters.c5_2["conv_blocking_and_parallelization_config_override"] = {"num_cores_nhw": 42} if groups > 1 else None
    parameters.c5_3["conv_blocking_and_parallelization_config_override"] = {"num_cores_nhw": 42} if groups > 1 else None

    parameters.c6["conv_blocking_and_parallelization_config_override"] = None
    parameters.c6_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c6_3["conv_blocking_and_parallelization_config_override"] = None

    parameters.c7["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
    parameters.c7_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c7_3["conv_blocking_and_parallelization_config_override"] = None

    parameters.c8["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
    parameters.c8["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
    parameters.c8_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
    parameters.c8_3["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}

    parameters.output_layer["conv_blocking_and_parallelization_config_override"] = None

    return parameters
