# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ttnn.model_preprocessing import infer_ttnn_module_args

from models.experimental.functional_unet.tt import unet_shallow_torch


def create_unet_input_tensors(
    device,
    batch,
    groups,
    pad_input=True,
    input_channels=4,
    input_height=1056,
    input_width=160,
    mesh_mapper=None,
):
    torch_input_tensor = torch.randn(batch, input_channels * groups, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    if pad_input:
        # Pad to 16 if grayskull run and 32 for wormhole
        pad = 32 if device.arch() == ttnn.device.Arch.WORMHOLE_B0 else 16
        hpad = 0  # 96*32*64
        if ttnn_input_tensor.shape[-1] < pad or ttnn_input_tensor.shape[-2] < hpad:
            ttnn_input_tensor = torch.nn.functional.pad(
                ttnn_input_tensor,
                (0, max(0, pad - ttnn_input_tensor.shape[-1]), 0, max(0, hpad - ttnn_input_tensor.shape[-2])),
            )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, device=device, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

    return torch_input_tensor, ttnn_input_tensor


def create_unet_model_parameters(model: unet_shallow_torch.UNet, input_tensor: torch.Tensor, groups: int, device):
    assert groups == 1, "Only groups=1 is supported for now"

    parameters = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.keys():
        parameters[key].module = getattr(model, key)

    parameters.p1["parallel_config_override"] = {
        "grid_size": (8, 8),
        "num_cores_nhw": 64,
    }
    parameters.p2["parallel_config_override"] = {
        "grid_size": (8, 8),
        "num_cores_nhw": 60,
    }
    parameters.p3["parallel_config_override"] = {
        "grid_size": (8, 8),
        "num_cores_nhw": 60,
    }
    parameters.p4["parallel_config_override"] = {
        "grid_size": (8, 7),
        "num_cores_nhw": 55,
    }

    parameters.c1["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 5 * 32}
    parameters.c1_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 5 * 32}

    parameters.c2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c2_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c3["conv_blocking_and_parallelization_config_override"] = None
    parameters.c3_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c4["conv_blocking_and_parallelization_config_override"] = None
    parameters.c4_2["conv_blocking_and_parallelization_config_override"] = None

    parameters.bnc["conv_blocking_and_parallelization_config_override"] = None
    parameters.bnc_2["conv_blocking_and_parallelization_config_override"] = None

    parameters.c5["conv_blocking_and_parallelization_config_override"] = None
    parameters.c5_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c5_3["conv_blocking_and_parallelization_config_override"] = None

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
