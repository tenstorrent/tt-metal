# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import untilize, comp_pcc
from models.utility_functions import is_grayskull, skip_for_blackhole


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=[
        "bfloat16",
    ],
)
@pytest.mark.parametrize(
    "nb, nc, nh, nw",
    (
        # llama shapes
        (1, 1, 32, 128 * 1024),
    ),
)
def test_run_untilize_subcoregrid_test(dtype, nb, nc, nh, nw, device):
    if is_grayskull():
        pytest.skip("Skipping tests on Grayskull")
    shape = [nb, nc, nh, nw]

    torch.set_printoptions(precision=3, sci_mode=False, linewidth=3000, threshold=10000, edgeitems=128)

    torch.manual_seed(10)

    inp = torch.rand(*shape).bfloat16()

    a = ttnn.Tensor(
        inp.flatten().tolist(),
        shape,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
    )

    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    b1 = ttnn.untilize(
        a,
        memory_config=out_mem_config,
        use_multicore=True,
        sub_core_grids=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6)),
            }
        ),
    )
    c1 = b1.cpu().to_torch()

    untilized_inp = untilize(inp)

    if dtype == ttnn.float32:
        passing1, output = comp_pcc(untilized_inp, c1, 0.999999)
        logger.info(output)
    else:
        passing1 = torch.equal(untilized_inp, c1)
    assert passing1


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.float32),
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
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")
    shape = [nb, nc, 32 * nh, 32 * nw]

    torch.set_printoptions(precision=3, sci_mode=False, linewidth=3000, threshold=10000, edgeitems=128)

    torch.manual_seed(10)

    if dtype == ttnn.float32:
        inp = torch.rand(*shape).float() * 1000.0
    else:
        inp = torch.rand(*shape).bfloat16()

    a = ttnn.Tensor(
        inp.flatten().tolist(),
        shape,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
    )

    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    b1 = ttnn.untilize(a, memory_config=out_mem_config, use_multicore=True, use_pack_untilize=True)
    c1 = b1.cpu().to_torch()

    untilized_inp = untilize(inp)

    if dtype == ttnn.float32:
        passing1, output = comp_pcc(untilized_inp, c1, 0.999999)
        logger.info(output)
    else:
        passing1 = torch.equal(untilized_inp, c1)
    assert passing1


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.float32),
    ids=["bfloat16", "float"],
)
@pytest.mark.parametrize(
    "shape",
    (
        [1, 1, 1, 32 * 5, 32 * 1],
        [1, 1, 1, 32 * 4, 32 * 2],
        [1, 1, 1, 32 * 3, 32 * 3],
        [1, 1, 1, 32 * 2, 32 * 4],
        [1, 1, 1, 32 * 1, 32 * 5],
        [1, 2, 3, 32 * 2, 32 * 1],
    ),
)
def test_run_untilize_5d(dtype, shape, device):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")

    torch.set_printoptions(precision=3, sci_mode=False, linewidth=3000, threshold=10000, edgeitems=128)

    torch.manual_seed(10)

    if dtype == ttnn.float32:
        inp = torch.rand(*shape).float() * 1000.0
    else:
        inp = torch.rand(*shape).bfloat16()

    a = ttnn.from_torch(inp, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    our_untilized = ttnn.untilize(a, memory_config=out_mem_config, use_multicore=True, use_pack_untilize=True)
    our_untilized = our_untilized.cpu().to_torch()

    if dtype == ttnn.float32:
        passing1, output = comp_pcc(inp, our_untilized, 0.999999)
        logger.info(output)
    else:
        passing1 = torch.equal(inp, our_untilized)

    assert passing1


def test_regression_untilize_1d(device):
    input = ttnn.ones([1280], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    out_untilized = ttnn.to_layout(input, layout=ttnn.ROW_MAJOR_LAYOUT)
    assert out_untilized is not None
