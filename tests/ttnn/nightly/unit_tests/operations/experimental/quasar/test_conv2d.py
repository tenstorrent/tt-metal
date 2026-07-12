# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


def make_depthwise_conv1d_tensors(device, channels=64, input_width=10, kernel_width=4, memory_config=None):
    torch.manual_seed(0)
    torch_input = torch.randn(1, channels, 1, input_width, dtype=torch.bfloat16).float()
    torch_weight = torch.randn(channels, 1, 1, kernel_width, dtype=torch.bfloat16).float()
    input_tt = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG,
    )
    weight_tt = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    return torch_input, torch_weight, input_tt, weight_tt


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
def test_quasar_depthwise_rejects_actual_block_layout_with_height_config(device):
    channels = 64
    input_width = 10
    block_memory_config = ttnn.create_sharded_memory_config(
        [1, 1, input_width, channels],
        core_grid=ttnn.CoreGrid(x=4, y=1),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    _, _, input_tt, weight_tt = make_depthwise_conv1d_tensors(
        device,
        channels=channels,
        input_width=input_width,
        memory_config=block_memory_config,
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )

    with pytest.raises(RuntimeError):
        ttnn.experimental.quasar.conv2d(
            input_tensor=input_tt,
            weight_tensor=weight_tt,
            device=device,
            in_channels=channels,
            out_channels=channels,
            batch_size=1,
            input_height=1,
            input_width=input_width,
            kernel_size=(1, 4),
            groups=channels,
            conv_config=conv_config,
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
def test_quasar_depthwise_accepts_public_prepared_weight(device):
    channels = 64
    input_width = 10
    kernel_width = 4
    torch_input, torch_weight, input_tt, weight_tt = make_depthwise_conv1d_tensors(
        device,
        channels=channels,
        input_width=input_width,
        kernel_width=kernel_width,
    )
    golden = torch.nn.functional.conv2d(torch_input, torch_weight, groups=channels)
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    prepared_weight = ttnn.prepare_conv_weights(
        weight_tensor=weight_tt,
        input_memory_config=ttnn.L1_MEMORY_CONFIG,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weights_format="OIHW",
        in_channels=channels,
        out_channels=channels,
        batch_size=1,
        input_height=1,
        input_width=input_width,
        kernel_size=(1, kernel_width),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        has_bias=False,
        groups=channels,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config=conv_config,
    )
    assert list(prepared_weight.shape) == [1, kernel_width, ttnn.TILE_SIZE, channels]
    prepared_weight_address = prepared_weight.buffer_address()

    def run_quasar():
        output_tt, [output_height, output_width] = ttnn.experimental.quasar.conv2d(
            input_tensor=input_tt,
            weight_tensor=prepared_weight,
            device=device,
            in_channels=channels,
            out_channels=channels,
            batch_size=1,
            input_height=1,
            input_width=input_width,
            kernel_size=(1, kernel_width),
            groups=channels,
            conv_config=conv_config,
            return_output_dim=True,
        )
        return ttnn.to_torch(output_tt).reshape(1, output_height, output_width, channels).permute(0, 3, 1, 2)

    first_output = run_quasar()
    second_output = run_quasar()
    assert prepared_weight.buffer_address() == prepared_weight_address
    first_passing, first_message = check_with_pcc_without_tensor_printout(first_output, golden, pcc=0.995)
    second_passing, second_message = check_with_pcc_without_tensor_printout(second_output, golden, pcc=0.995)
    assert first_passing, first_message
    assert second_passing, second_message
