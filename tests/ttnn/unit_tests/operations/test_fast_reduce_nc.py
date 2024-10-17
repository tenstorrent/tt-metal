# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_allclose_and_pcc, comp_pcc, skip_for_grayskull
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    TILE_HEIGHT,
    TILE_WIDTH,
)


def get_tensors(input_shape, output_shape, device, *, with_padding=True, use_randint=True, dataformat=ttnn.bfloat16):
    npu_dtype = dataformat
    cpu_dtype = torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT

    if use_randint:
        torch_input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype, requires_grad=True)
        torch_output = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
    else:
        torch_input = torch.rand(input_shape, dtype=cpu_dtype, requires_grad=True)
        torch_output = torch.rand(output_shape, dtype=cpu_dtype)

    if with_padding:
        tt_input = ttnn.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
        tt_output = ttnn.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    else:
        tt_input = ttnn.Tensor(torch_input, npu_dtype).to(npu_layout).to(device)
        tt_output = ttnn.Tensor(torch_output, npu_dtype).to(npu_layout).to(device)

    return tt_input, tt_output, torch_input


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 8, 128, 4096],
        [1, 8, 1024, 4096],
        [1, 8, 2048, 4096],
        [8, 1, 128, 4096],
        [4, 2, 1024, 4096],
    ),
    ids=[
        "mixtral_128",
        "mixtral_1k",
        "mixtral_2k",
        "dim0_reduce",
        "dim01_reduce",
    ],
)
@skip_for_grayskull()
@pytest.mark.parametrize(
    "dims",
    ([0], [1], [0, 1]),
    ids=["0", "1", "0_1"],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("dataformat", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bfloat16", "bfloat8_b"])
def test_fast_reduce_nc(input_shape, dims, compute_kernel_options, dataformat, device):
    torch.manual_seed(2023)
    output_shape = input_shape.copy()

    for dim in dims:
        output_shape[dim] = 1

    (tt_input, tt_output, torch_input) = get_tensors(input_shape, output_shape, device, dataformat=dataformat)

    torch_output = torch.sum(torch_input, dims, True)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_output = ttnn.experimental.fast_reduce_nc(
        tt_input, dims=dims, output=None, compute_kernel_config=compute_kernel_config
    )
    tt_output_cpu = tt_output.cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()

    # test for equivalance
    rtol = atol = 0.12
    if dataformat == ttnn.bfloat8_b:
        passing, output_pcc = comp_pcc(torch_output, tt_output_cpu, pcc=0.999)
    else:
        passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing


# Program caching test
@pytest.mark.parametrize(
    "dims",
    ([0], [1], [0, 1]),
    ids=["0", "1", "0_1"],
)
@skip_for_grayskull()
def test_fast_reduce_nc_with_prgm_caching(dims, device, use_program_cache):
    torch.manual_seed(2023)

    input_shape_1 = [1, 8, 128, 4096]
    output_shape_1 = input_shape_1.copy()

    for _ in range(3):
        # Apply offset in dram and l1 to test program cache
        # shift input/output tensor by creating very small tensor between loop
        inp = torch.rand(1, 1, 32, 32)
        test_tensor = (
            ttnn.Tensor(
                inp.reshape(-1).tolist(),
                inp.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        shard_spec_1_cores_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(0, 0),
                ),
            }
        )
        test_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_1_cores_grid,  # Volume must match # of attn heads
                [
                    32,  # Each core has 32 users
                    32,  # head dim
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        test_tensor = ttnn.interleaved_to_sharded(test_tensor, test_mem_cfg)

        # Test op
        for dim in dims:
            output_shape_1[dim] = 1

        (tt_input, tt_output, torch_input) = get_tensors(input_shape_1, output_shape_1, device)

        torch_output = torch.sum(torch_input, dims, True)

        cpu_layout = ttnn.ROW_MAJOR_LAYOUT
        tt_output = ttnn.experimental.fast_reduce_nc(tt_input, dims=dims, output=None)
        tt_output_cpu = tt_output.cpu().to(cpu_layout).unpad_from_tile(output_shape_1).to_torch()

        # test for equivalance
        rtol = atol = 0.12
        passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

        logger.debug(f"Out passing={passing}")
        logger.debug(f"Output pcc={output_pcc}")

        assert passing
        assert device.num_program_cache_entries() == len(dims) + 1

    input_shape_2 = [1, 8, 32, 32]
    output_shape_2 = input_shape_2.copy()

    for _ in range(2):
        # Test op
        for dim in dims:
            output_shape_2[dim] = 1

        (tt_input, tt_output, torch_input) = get_tensors(input_shape_2, output_shape_2, device)

        torch_output = torch.sum(torch_input, dims, True)

        cpu_layout = ttnn.ROW_MAJOR_LAYOUT
        tt_output = ttnn.experimental.fast_reduce_nc(tt_input, dims=dims, output=None)
        tt_output_cpu = tt_output.cpu().to(cpu_layout).unpad_from_tile(output_shape_2).to_torch()

        # test for equivalance
        rtol = atol = 0.12
        passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

        logger.debug(f"Out passing={passing}")
        logger.debug(f"Output pcc={output_pcc}")

        assert passing
        assert device.num_program_cache_entries() == 2 * len(dims) + 1
