# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 1, 32],
    ],
)
def test_tanh_range(device, shape):
    torch_input_tensor_a = torch.tensor(
        [
            [
                [
                    [
                        -1.8125,
                        -2.828125,
                        -3.125,
                        -3.234375,
                        -2.765625,
                        -1.890625,
                        -3.359375,
                        -2.0625,
                        -3.015625,
                        -2.203125,
                        -2.015625,
                        -2.9375,
                        -1.3046875,
                        -1.359375,
                        -1.3984375,
                        -1.2265625,
                        -2,
                        -3,
                        -1.5,
                        -2.5,
                        -3.5,
                        -3.75,
                        -3.359375,
                        -1.8828125,
                        -3.255,
                        -0.9,
                        -0.1,
                        0.25,
                        0.75,
                        -0.8359375,
                        -0.5,
                        0.9,
                    ]
                ]
            ]
        ],
        dtype=torch.bfloat16,
    )
    torch_output_tensor = torch.tanh(torch_input_tensor_a)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.tanh(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG, accuracy=True)

    output_tensor = ttnn.to_torch(output_tensor)

    # for i in range(1):            # Batch size
    #     for j in range(1):        # Channels
    #         for k in range(shape[-2]):   # Height
    #             for l in range(shape[-1]):  # Width
    #                 print(f"{i}-{j}-{k}-{l} input: {torch_input_tensor_a[i][j][k][l]} \t TT_out: {output_tensor[i][j][k][l]} \t torch: {torch_output_tensor[i][j][k][l]} \n")

    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    # print("pcc_msg", pcc_msg) # pcc_msg 0.9999607543069606
    assert pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([2, 4, 320, 1024])),
        (torch.Size([1, 9, 8192])),
    ),
)
@pytest.mark.parametrize(
    "input_range",
    [
        {"high": 1, "low": -1},
        {"high": 100, "low": -100},
        {"high": 4, "low": -4},
    ],
)
def test_tanh_accuracy(device, input_shapes, input_range):
    high = input_range["high"]
    low = input_range["low"]
    torch_input_tensor = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low) + low
    golden_function = ttnn.get_golden_function(ttnn.tanh)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.tanh(input_tensor, accuracy=True)
    output_tensor = ttnn.to_torch(output)
    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    # print("pcc_msg", pcc_msg)  # pcc_msg 0.9999 or above
    assert pcc
