# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import tt_lib as ttl
from loguru import logger

from models.utility_functions import (
    untilize,
    tilize_to_list,
    comp_pcc,
)
import torch


def run_tilize_matmul_test(M, K, N, device):
    a_shape = [1, 1, M, K]
    b_shape = [1, 1, K, N]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    a = ttl.tensor.Tensor(
        A.flatten().tolist(),
        a_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        device,
    )
    a_t = ttl.tensor.tilize(a)
    b_t = ttl.tensor.Tensor(
        tilize_to_list(B),
        b_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
    )
    t2 = ttl.tensor.matmul(a_t, b_t)
    assert list(t2.get_legacy_shape()) == [1, 1, M, N]
    tt_host_rm = t2.cpu().to_torch()
    pyt_got_back = tt_host_rm.reshape((1, 1, M, N))
    pyt_got_back_rm = untilize(pyt_got_back)

    ref_bmm = torch.matmul(A.reshape(1, M, K), B.reshape(1, K, N))
    ref_bmm = ref_bmm.reshape(1, 1, M, N)
    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, 0.99)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")
    assert passing_pcc


def test_run_tilize_matmul_test(device):
    run_tilize_matmul_test(32, 32, 32, device)
