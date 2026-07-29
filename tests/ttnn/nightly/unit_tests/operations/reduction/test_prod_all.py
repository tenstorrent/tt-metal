# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tests.ttnn.nightly.unit_tests.operations.reduction.utility_functions import ttnn_prod


def get_tensors(input_shape, output_shape, device, npu_dtype):
    torch.manual_seed(2023)
    cpu_dtype = torch.float32 if npu_dtype == ttnn.float32 else torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT

    torch_input = torch.randint(1, 5, input_shape, dtype=cpu_dtype)
    torch_output = torch.randint(1, 5, output_shape, dtype=cpu_dtype)
    tt_input = ttnn.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = ttnn.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_input, tt_output, torch_input


@pytest.mark.parametrize("npu_dtype", (ttnn.bfloat16, ttnn.float32))
@pytest.mark.parametrize(
    "shapes",
    (
        ([1, 1, 32, 32]),
        ([1, 4, 32, 32]),
        ([2, 2, 32, 32]),
        ([16, 16]),
        # ([6, 4, 32, 32]), #Fails : expected result is inf but the result generated in nan
        # ([1, 1, 320, 320]), #Fails : expected result is inf but the result generated in nan
        # ([1, 3, 320, 64]), #Fails : expected result is inf but the result generated in nan
    ),
)
def test_prod(shapes, npu_dtype, device):
    output_shape = shapes.copy()

    (tt_input, tt_output, torch_input) = get_tensors(shapes, shapes, device, npu_dtype)

    torch_output = torch.prod(torch_input)

    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_output_cpu = ttnn_prod(tt_input).cpu().to(cpu_layout).to_torch()
    N = tt_output_cpu.shape
    torch.set_printoptions(threshold=10000, precision=5, sci_mode=False)
    logger.info("Input shape")
    logger.info(torch_input.shape)
    logger.info("TT Output")
    logger.info(tt_output_cpu)
    logger.info("Torch Output")
    logger.info(torch_output)

    if torch.isfinite(torch_output).all() and torch.isfinite(tt_output_cpu).all():
        check_frobenius = True
    else:
        check_frobenius = False
    # test for equivalance
    assert_numeric_metrics(
        torch_output,
        tt_output_cpu,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_frobenius=check_frobenius,
    )


BLOCK_FLOAT_DTYPES = (ttnn.bfloat8_b, ttnn.bfloat4_b)
BLOCK_FLOAT_RESULT_ATOL = {ttnn.bfloat8_b: 5e-2, ttnn.bfloat4_b: 2e-1}


def _block_float_input(shape):
    # Block-float (bfp8_b / bfp4_b) shares one exponent per 16 elements. Mostly ones with a few 2.0s:
    # powers of two, so the whole product is exact in block-float regardless of dtype.
    torch_input = torch.ones(shape, dtype=torch.float32)
    flat = torch_input.view(-1)
    num_twos = 5
    flat[: num_twos * 7 : 7] = 2.0
    return torch_input, torch.prod(torch_input)


def _block_float_nontrivial_input(shape):
    # Non-trivial but block-float-exact input: values are integer multiples of 0.25 with |k| <= 7,
    # which are represented exactly in bfp4_b/bfp8_b.
    torch_input = torch.ones(shape, dtype=torch.float32)
    factors = torch.tensor([1.5, 0.75, 1.25, 0.75, 1.5, 0.75, -1.0, 1.25, 0.75], dtype=torch.float32)
    torch_input.view(-1)[: factors.numel()] = factors
    return torch_input, torch.prod(torch_input)


def _check_block_float_full_product(shape, npu_dtype, device, make_input, atol):
    torch_input, torch_output = make_input(shape)
    tt_input = ttnn.Tensor(torch_input, npu_dtype).to(ttnn.TILE_LAYOUT).to(device)

    tt_result = ttnn.prod(tt_input)
    assert tt_result.dtype == npu_dtype, f"expected {npu_dtype} result, got {tt_result.dtype}"

    tt_output = ttnn.to_torch(tt_result).flatten()[0]
    logger.info(f"{npu_dtype} full-product: expected={torch_output.item()} got={tt_output.item()}")
    assert torch.isclose(tt_output, torch_output, atol=atol), f"expected {torch_output.item()}, got {tt_output.item()}"


@pytest.mark.parametrize("npu_dtype", BLOCK_FLOAT_DTYPES, ids=lambda d: d.name.lower())
@pytest.mark.parametrize(
    "shapes",
    (
        ([1, 1, 32, 32]),
        ([1, 4, 32, 32]),
        ([2, 2, 32, 32]),
    ),
)
def test_prod_all_block_float(shapes, npu_dtype, device):
    # Powers-of-two data is exact in block-float, so a single tight tolerance holds for every dtype.
    _check_block_float_full_product(shapes, npu_dtype, device, _block_float_input, atol=1e-2)


@pytest.mark.parametrize("npu_dtype", BLOCK_FLOAT_DTYPES, ids=lambda d: d.name.lower())
@pytest.mark.parametrize(
    "shapes",
    (
        ([1, 4, 32, 32]),
        ([1, 16, 32, 32]),
    ),
)
def test_prod_all_block_float_nontrivial(shapes, npu_dtype, device):
    # Off-grid partial products: the result tolerance is per-dtype (see BLOCK_FLOAT_RESULT_ATOL).
    _check_block_float_full_product(
        shapes, npu_dtype, device, _block_float_nontrivial_input, atol=BLOCK_FLOAT_RESULT_ATOL[npu_dtype]
    )
