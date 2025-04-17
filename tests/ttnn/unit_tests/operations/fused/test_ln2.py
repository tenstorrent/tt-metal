# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc

aten = torch.ops.aten
from models.utility_functions import comp_pcc


def load_tensor_from_csv(filename, shape):
    import pandas as pd

    df = pd.read_csv(filename)
    data = df.values
    tensor = torch.tensor(data)
    tensor = tensor.view(shape)
    return tensor


def run_layer_norm_tests(
    file_path,
    input_shape,
    dtype,
    dlayout,
    epsilon,
    device,
):
    x = load_tensor_from_csv(file_path[0], input_shape[0]).to(torch.bfloat16)
    weight = load_tensor_from_csv(file_path[1], input_shape[1]).to(torch.bfloat16)
    bias = load_tensor_from_csv(file_path[2], input_shape[2]).to(torch.bfloat16)

    try:
        ref_value = aten.native_layer_norm.default(x, [768], weight, bias, epsilon)[0]
        print(input_shape[0])
        print(x.shape)
        torch.set_printoptions(threshold=float("inf"))

        print(x[0, :32, :32])
        val = torch.mean(x, dim=-1, keepdim=True)
        # print(val)

        tt_x = ttnn.from_torch(x, dtype=dtype[0], layout=dlayout[0], device=device)
        tt_weight = ttnn.from_torch(weight, dtype=dtype[0], layout=dlayout[0], device=device)
        tt_bias = ttnn.from_torch(bias, dtype=dtype[0], layout=dlayout[0], device=device)

        tt_result = ttnn.layer_norm(tt_x, epsilon=epsilon, weight=tt_weight, bias=tt_bias)
        tt_result = ttnn.to_torch(tt_result)

        inds = torch.where(ref_value != tt_result)
    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.999)
    pcc_passed, pcc_message = comp_pcc(ref_value, tt_result, 0.9998)
    print(pcc_passed)


test_sweep_args = [
    (
        [
            "/localdev/vsuresh/tt-metal/tests/ttnn/unit_tests/operations/fused/inputs/mixer_ln1_input.csv",
            "/localdev/vsuresh/tt-metal/tests/ttnn/unit_tests/operations/fused/inputs/mixer_ln1_weight.csv",
            "/localdev/vsuresh/tt-metal/tests/ttnn/unit_tests/operations/fused/inputs/mixer_ln1_bias.csv",
        ],
        [(1, 196, 768), (768), (768)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        1e-6,
    ),
]
for i in range(3, 4):
    test_sweep_args.append(
        (
            [
                f"/localdev/vsuresh/tt-metal/tests/ttnn/unit_tests/operations/fused/inputs/mixer_ln{i}_input.csv",
                f"/localdev/vsuresh/tt-metal/tests/ttnn/unit_tests/operations/fused/inputs/mixer_ln{i}_weight.csv",
                f"/localdev/vsuresh/tt-metal/tests/ttnn/unit_tests/operations/fused/inputs/mixer_ln{i}_bias.csv",
            ],
            [(1, 196, 768), (768), (768)],
            [ttnn.bfloat16],
            [ttnn.TILE_LAYOUT],
            1e-6,
        )
    )


@pytest.mark.parametrize(
    "file_path, input_shape, dtype, dlayout, epsilon",
    (test_sweep_args),
)
def test_layer_norm(file_path, input_shape, dtype, dlayout, epsilon, device):
    run_layer_norm_tests(file_path, input_shape, dtype, dlayout, epsilon, device)
