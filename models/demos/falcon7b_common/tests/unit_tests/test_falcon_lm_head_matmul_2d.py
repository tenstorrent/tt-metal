# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

import ttnn

from models.demos.falcon7b_common.tt.falcon_lm_head import falcon_lm_head_matmul_2d
from models.utility_functions import comp_pcc, tt2torch_tensor, torch2tt_tensor


def run_falcon_lm_head_matmul_2d(
    seq_len, device, in0_mem_config, in1_mem_config, out_mem_config, in0_dtype, in1_dtype, out_dtype
):
    pcc = 0.99
    torch.manual_seed(1234)

    M = seq_len // 32
    K = 142
    N = 2032

    a_shape = [1, 1, M * 32, K * 32]
    # weights are padded on model initialization to reach 144 inner dim
    b_shape = [1, 1, (K + 2) * 32, N * 32]
    expected_output_shape = [1, 1, M * 32, N * 32]

    # calculate parameters for the given sequence length
    num_slices = 4 if seq_len <= 1024 else 8
    # on n300 we have 8*7 grid size so we run out of memory and need more slices
    if device.compute_with_storage_grid_size().y < 8:
        num_slices *= 2

    # --- prepare activations
    A = torch.randn(a_shape)
    padding = torch.zeros([1, 1, M * 32, 64])
    A_padded = torch.cat([A, padding], -1)

    # padd in0 tensor to have last dimension 144 * 32 instead of 142 * 32
    a_t = torch2tt_tensor(
        A,
        device,
        ttnn.TILE_LAYOUT,
        in0_mem_config,
        in0_dtype,
    )

    # --- prepare weights
    B = torch.randn(b_shape)
    B_slices = torch.chunk(B, num_slices, dim=-1)
    b_t_slices = []
    for i in range(num_slices):
        b_t_slices.append(
            torch2tt_tensor(
                B_slices[i],
                device,
                ttnn.TILE_LAYOUT,
                in1_mem_config,
                in1_dtype,
            )
        )

    out = falcon_lm_head_matmul_2d(a_t, b_t_slices, num_slices, in0_mem_config, in0_dtype, out_mem_config, out_dtype)

    a_t.deallocate(True)
    for i in range(num_slices):
        b_t_slices[i].deallocate(True)

    assert out.get_legacy_shape() == expected_output_shape

    # check pcc
    out = tt2torch_tensor(out)
    ref_out = torch.matmul(A_padded, B)
    passed, output_pcc = comp_pcc(out, ref_out, pcc)
    logger.info(f"Output pcc={output_pcc}")

    assert passed


@pytest.mark.parametrize("seq_len", range(544, 2049, 32))
def test_falcon_lm_head_matmul_2d(
    seq_len,
    device,
):
    in0_mem_config = ttnn.DRAM_MEMORY_CONFIG
    in1_mem_config = ttnn.DRAM_MEMORY_CONFIG
    out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    in0_dtype = ttnn.bfloat16
    in1_dtype = ttnn.bfloat8_b
    out_dtype = ttnn.bfloat16

    run_falcon_lm_head_matmul_2d(
        seq_len, device, in0_mem_config, in1_mem_config, out_mem_config, in0_dtype, in1_dtype, out_dtype
    )
