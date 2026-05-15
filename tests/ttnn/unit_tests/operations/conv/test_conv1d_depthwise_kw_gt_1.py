# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Repro for the Mamba conv regression introduced by commit dadf6a7bf0
# ("change depthwise condition") and tracked in issue #42163.
#
# Mamba's mixer.conv1d is a depthwise 1D conv with kernel_size=4, kernel_height=1.
# These cases cover both paths in the specialized 1D depthwise factory:
# - 2560 channels: stick_bytes * kernel_w exceeds WH NOC burst, so reads stay per tap.
# - 512 channels: stick_bytes * kernel_w fits, so the kernel-width sticks are coalesced.

import pytest
import torch
import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_length, kernel_size",
    [
        (512, 257, 2),
        (512, 257, 3),
        (512, 257, 4),
        (2560, 1027, 2),
        (2560, 1027, 3),
        (2560, 1027, 4),  # mamba 2.8B per-split shape
    ],
)
def test_depthwise_conv1d_kw_gt_1_height_sharded(device, input_channels, input_length, kernel_size):
    torch.manual_seed(0)
    batch_size = 1
    output_channels = input_channels
    groups = input_channels  # depthwise

    torch_input_ncl = torch.randn([batch_size, input_channels, input_length], dtype=torch.bfloat16).float()
    torch_input_nlc = torch_input_ncl.permute(0, 2, 1)
    torch_weight = torch.randn([output_channels, 1, kernel_size], dtype=torch.bfloat16).float()
    torch_golden = torch.nn.functional.conv1d(torch_input_ncl, torch_weight, groups=groups)

    tt_input = ttnn.from_torch(torch_input_nlc, ttnn.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight, ttnn.float32)  # bfloat8_b weights are converted from float32

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
    )

    # On main, the kernel build itself fails with:
    #   reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp:79:
    #   static_assertion failed (20480 <= 8192)
    [tt_out_dev, out_length, _] = ttnn.conv1d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=None,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        batch_size=batch_size,
        input_length=input_length,
        dtype=ttnn.bfloat8_b,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=groups,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    tt_out = ttnn.to_torch(ttnn.from_device(tt_out_dev))
    tt_out = tt_out.reshape(batch_size, out_length, output_channels).permute(0, 2, 1)

    from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
    from loguru import logger

    passing, pcc_msg = check_with_pcc_without_tensor_printout(tt_out, torch_golden, pcc=0.99)
    logger.info(f"kernel_size={kernel_size} PCC={pcc_msg}")
    assert passing, pcc_msg
