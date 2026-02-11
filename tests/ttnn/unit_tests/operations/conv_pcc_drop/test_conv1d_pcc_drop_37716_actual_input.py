# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Repro for https://github.com/tenstorrent/tt-metal/issues/37716
# PCC drop in conv1d/conv2d with kernel_size=3, dilation=3, padding=3
# Original model: Vocoder (torch.nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=3, dilation=3))

import torch
import pytest
import ttnn
import tests.ttnn.unit_tests.operations.conv_pcc_drop.utils as utils
from tests.ttnn.utils_for_testing import assert_with_pcc


def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_flat, y_flat = x.flatten().double(), y.flatten().double()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return ((vx @ vy) / denom).item()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
def test_conv1d_pcc_drop_37716_actual_input(device):
    """
    Repro for issue #37716: PCC drop (~0.55) in conv with dilation=3.
    Uses ttnn.conv1d with config_tensors_in_dram=False (latest repro from issue).
    """
    # Load actual input/weight/bias from saved tensors
    torch_bias = ttnn.to_torch(utils.load_tensor("./tensors/arg0.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    torch_weight = ttnn.to_torch(utils.load_tensor("./tensors/arg1.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    torch_input_ncl = ttnn.to_torch(utils.load_tensor("./tensors/arg2.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))

    batch_size = 1
    in_channels = 128
    out_channels = 128
    input_length = 8192
    kernel_size = 3
    stride = 1
    padding = 3
    dilation = 3
    groups = 1

    # --- Golden CPU inference ---
    cpu_output = torch.nn.functional.conv1d(
        torch_input_ncl,
        torch_weight,
        bias=torch_bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    # --- TT inference ---
    # conv1d expects NLC input
    input_nlc = torch_input_ncl.permute(0, 2, 1)  # [1, 8192, 128]
    input_ttnn = ttnn.from_torch(
        input_nlc,
        layout=ttnn.Layout.ROW_MAJOR,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )

    weight_ttnn = ttnn.from_torch(torch_weight, layout=ttnn.Layout.ROW_MAJOR)

    bias_ttnn = ttnn.from_torch(
        torch_bias.reshape(1, 1, 1, out_channels),
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=True,
        config_tensors_in_dram=True,
    )

    tt_conv_output, out_length = ttnn.conv1d(
        input_tensor=input_ttnn,
        weight_tensor=weight_ttnn,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_length=input_length,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias_tensor=bias_ttnn,
        conv_config=conv_config,
        compute_config=None,
        dtype=ttnn.bfloat16,
        return_output_dim=True,
    )

    # Postprocess: NLC -> NCL
    tt_output_torch = ttnn.to_torch(tt_conv_output)
    tt_output_torch = tt_output_torch.reshape(batch_size, out_length, out_channels).permute(0, 2, 1)

    # Compare
    pcc = compute_pcc(tt_output_torch, cpu_output)
    print(f"PCC: {pcc}")
    # This should be > 0.99 but issue reports ~0.55
    assert pcc > 0.99, f"PCC too low: {pcc}"
