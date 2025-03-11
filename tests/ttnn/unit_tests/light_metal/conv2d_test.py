# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import inspect
import ttnn
import tempfile
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
import os
import numpy as np
import random
import hashlib


# Trimmed - also reproduces failure after reset.
# This test aims to extract just the conv2d op from resnet that was failing.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("increase_stride", [1])
@pytest.mark.parametrize("enable_async", [False])
@pytest.mark.parametrize("blocking", [True])
def test_resnet50_first_conv2d_only_repro_nd(
    batch_size, enable_async, blocking, reset_seeds, increase_stride, device, reset_device, tmp_path
):
    # Create a dummy activation tensor that represents the already-folded activation.
    # In production, the folded activation has shape [1, 1, 211600, 16].
    # For simplicity, we simulate a small unfolded activation with shape (1, 1, 100, 16).
    dummy_activation = torch.rand((batch_size, 1, 100, 16), dtype=torch.bfloat16)
    act_tt = ttnn.from_torch(dummy_activation, ttnn.bfloat16)
    act_tt = ttnn.to_device(act_tt, device)

    # Use trimmed conv_kwargs that match the dummy activation.
    conv_kwargs = {
        "in_channels": 16,
        "out_channels": 32,
        "batch_size": 1,  # trimmed batch_size
        "input_height": 100,  # matching dummy activation height
        "input_width": 16,  # matching dummy activation width
        "kernel_size": (4, 4),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
    }

    # KCM - Quick change to reduce output size greatly.
    if increase_stride:
        conv_kwargs["stride"] = (2, 4)

    # Trimmed conv_config with only the essential parameters.
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        activation="relu",
        input_channels_alignment=16,
        act_block_h_override=100,
        act_block_w_div=2,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    mem_config = ttnn.L1_MEMORY_CONFIG

    # Create dummy weight and bias tensors as in production.
    # Weight shape (OIHW): (64, 16, 4, 4); Bias shape: (1,1,1,64)
    weight = torch.rand((32, 16, 4, 4), dtype=torch.bfloat16)
    bias = torch.rand((1, 1, 1, 32), dtype=torch.bfloat16)
    weight_tt = ttnn.from_torch(weight, ttnn.bfloat16)
    bias_tt = ttnn.from_torch(bias, ttnn.bfloat16)

    # Prepare the weights using act_tt's memory config.
    if not ttnn.is_tensor_storage_on_device(weight_tt):
        weight_tt = ttnn.prepare_conv_weights(
            weight_tensor=weight_tt,
            weights_format="OIHW",
            input_memory_config=act_tt.memory_config(),
            input_layout=act_tt.get_layout(),
            has_bias=True,
            **conv_kwargs,
        )
        bias_tt = ttnn.prepare_conv_bias(
            bias_tensor=bias_tt,
            input_memory_config=act_tt.memory_config(),
            input_layout=act_tt.get_layout(),
            **conv_kwargs,
        )
    weight_tt = ttnn.to_device(weight_tt, device)
    bias_tt = ttnn.to_device(bias_tt, device)

    # Now, call conv2d. With the trimmed conv_kwargs the op should run on a much smaller output tensor.
    output_tt, dims = ttnn.conv2d(
        input_tensor=act_tt,
        weight_tensor=weight_tt,
        bias_tensor=bias_tt,
        **conv_kwargs,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        memory_config=mem_config,
        return_output_dim=True,
    )

    output_host = ttnn.to_torch(output_tt)
    sha256_hash = hashlib.sha256(output_host.numpy().tobytes()).hexdigest()

    np.set_printoptions(threshold=np.inf)
    print(
        f"KCM conv2d output_host dtype: {output_host.dtype} shape:", output_host.shape, " data: ", output_host.numpy()
    )
    print(f"KCM SHA-256 hash of tensor: {sha256_hash}")
