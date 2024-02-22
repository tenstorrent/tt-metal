# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from models.utility_functions import untilize, comp_pcc
from models.utility_functions import is_grayskull


@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.FLOAT32),
    ids=["bfloat16", "float"],
)
@pytest.mark.parametrize(
    "nb, nc, nh, nw",
    (
        (1, 1, 1, 2),
        (5, 2, 4, 8),
        (5, 2, 4, 7),
        ## resnet shapes
        (1, 1, 1, 1),
        (1, 1, 7, 8),
        (1, 1, 49, 1),
        (1, 1, 49, 16),
        (1, 1, 49, 32),
        (1, 1, 196, 4),
        (1, 1, 196, 8),
        (1, 1, 196, 16),
        (1, 1, 784, 2),
        (1, 1, 784, 4),
        (1, 1, 784, 8),
        (1, 1, 3136, 2),
    ),
)
def test_run_untilize_test(dtype, nb, nc, nh, nw, device):
    if is_grayskull() and dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    shape = [nb, nc, 32 * nh, 32 * nw]

    torch.set_printoptions(precision=3, sci_mode=False, linewidth=3000, threshold=10000, edgeitems=128)

    torch.manual_seed(10)

    if dtype == ttl.tensor.DataType.FLOAT32:
        inp = torch.rand(*shape).float() * 1000.0
    else:
        inp = torch.rand(*shape).bfloat16()
    # for b in range(0, nb):
    #     for c in range(0, nc):
    #         for h in range(0, 32 * nh):
    #             for w in range(0, 32 * nw):
    #                 inp[b][c][h][w] = 1 + w // 32
    # print(f"{inp}")

    a = ttl.tensor.Tensor(
        inp.flatten().tolist(),
        shape,
        dtype,
        ttl.tensor.Layout.TILE,
        device,
    )

    # a_sharded = ttl.tensor.interleaved_to_sharded(
    #     a,
    #     [12, 9],
    #     [a.get_legacy_shape()[-2] // 100, a.get_legacy_shape()[-1]],
    #     ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
    #     ttl.tensor.ShardOrientation.ROW_MAJOR,
    # )
    # out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1)
    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)

    # print(f"{a}")
    b1 = ttl.tensor.untilize(a, out_mem_config, use_multicore=True, use_pack_untilize=True)
    # b2 = ttl.tensor.untilize(a_sharded, out_mem_config, use_multicore=True, use_pack_untilize=False)
    c1 = b1.cpu().to_torch()
    # c2 = b2.cpu().to_torch()

    # print(f"{c.shape}")
    # torch.save(c, "output.pt")

    untilized_inp = untilize(inp)
    # print(f"{untilized_inp.shape}")
    # torch.save(untilized_inp, "golden.pt")

    if dtype == ttl.tensor.DataType.FLOAT32:
        passing1, output = comp_pcc(untilized_inp, c1, 0.999999)
        logger.info(output)
    else:
        passing1 = torch.equal(untilized_inp, c1)
    # print(f'OUTPUT: {untilized_inp}')
    # print(f'GOLDEN: {c1}')
    # passing2 = torch.equal(untilized_inp, c2)
    assert passing1
    # assert passing1 and passing2
