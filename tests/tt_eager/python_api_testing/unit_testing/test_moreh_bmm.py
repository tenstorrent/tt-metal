# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0
from models.utility_functions import comp_pcc

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensors(b_m_k_n, device):
    input_shape = [1, b_m_k_n[0], b_m_k_n[1], b_m_k_n[2]]
    mat2_shape = [1, b_m_k_n[0], b_m_k_n[2], b_m_k_n[3]]

    torch.manual_seed(2023)
    dtype = ttl.tensor.DataType.BFLOAT16
    npu_layout = ttl.tensor.Layout.TILE
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR

    # create input tensors using torch
    input = torch.randn(input_shape, dtype=torch.bfloat16)
    mat2 = torch.randn(mat2_shape, dtype=torch.bfloat16)

    # TT matmul
    # set different padded value for tt_a and tt_b.
    tt_input = (
        ttl.tensor.Tensor(input.reshape(-1).tolist(), input_shape, dtype, cpu_layout)
        .pad_to_tile(1)
        .to(npu_layout)
        .to(device)
    )

    tt_mat2 = (
        ttl.tensor.Tensor(mat2.reshape(-1).tolist(), mat2_shape, dtype, cpu_layout)
        .pad_to_tile(float("nan"))
        .to(npu_layout)
        .to(device)
    )

    torch_input = input.reshape(-1, input_shape[2], input_shape[3])
    torch_mat2 = mat2.reshape(-1, mat2_shape[2], mat2_shape[3])

    return tt_input, tt_mat2, torch_input, torch_mat2


@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "b_m_k_n",
    (
        [1, TILE_HEIGHT - 1, TILE_WIDTH * 20 - 1, TILE_WIDTH - 1],
        [5, TILE_HEIGHT * 3 - 1, TILE_WIDTH * 13 - 1, TILE_WIDTH * 2 - 1],
        [10, TILE_HEIGHT * 6 - 1, TILE_WIDTH * 14 - 1, TILE_WIDTH * 5 - 1],
        [20, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 15 - 1, TILE_WIDTH * 8 - 1],
    ),
)
def test_moreh_bmm(b_m_k_n, device):
    # get tensors
    tt_input, tt_mat2, torch_input, torch_mat2 = get_tensors(b_m_k_n, device)
    output_shape = [1, b_m_k_n[0], b_m_k_n[1], b_m_k_n[3]]

    # tt bmm
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_out = ttl.tensor.moreh_bmm(tt_input, tt_mat2).cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()

    # torch bmm
    torch_out = torch.bmm(torch_input, torch_mat2)

    ## test for equivalance
    passing_pcc, output_pcc = comp_pcc(torch_out, tt_out)
    print(f"Passing PCC = {passing_pcc}")
    print(f"Output PCC = {output_pcc}")

    assert passing_pcc
