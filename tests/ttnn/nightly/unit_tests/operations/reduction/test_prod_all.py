# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tests.ttnn.nightly.unit_tests.operations.reduction.utility_functions import ttnn_prod

# Module-scoped device: these tests all run with the default device config, so the device is
# opened once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device


# Dyadic factors (k / 2**n): each needs at most two stored mantissa bits, so every *operand*
# survives the FPU source registers bit-for-bit, where those registers keep only 10 stored
# mantissa bits and truncate the rest (see test_prod_all_fp32_truncation_accuracy below).
# The running product is not exact - products of dyadic values are dyadic but their significands
# grow, e.g. 1.5**15 = 3**15 / 2**15 needs 24 bits - so a few float32 ulps still accumulate in the
# host-side reduction; measured worst case over these shapes is 3.9e-7, hence the 1e-5 tolerance.
# The point is that no error comes from the input datapath, which keeps test_prod a check on
# *what* prod computes rather than on how accurately it computes it.
_DYADIC_FACTORS = (1.5, 0.75, 1.25, 0.625, 2.0, 0.5)


def _exact_product_input(shape, cpu_dtype, seed=2023):
    """Random dyadic factors, renormalised so the full product lands in [2**-0.5, 2**0.5].

    # A product of thousands of factors drifts exponentially. For example, taking randint(1, 5)
    # as the stimulus easily overflows the golden to +inf, leaving allclose(inf, inf) as the only
    # reasonable check at the end of the test.

    # Conversely, drawing factors around 1.0 pushes the product the other way,
    # to ~1e-35 for the 4096-element shapes. That is only three orders of magnitude above the smallest
    # normal float32 (~1.2e-38, and bfloat16 has the same exponent range), so a different seed
    # or a larger shape would underflow into denormals where relative error stops being meaningful.

    # Dividing element 0 by 2**round(log2(product)) lands the product in [2**-0.5, 2**0.5].
    # Scaling by a power of two only changes the exponent field, so no element becomes inexact,
    # as long as the result stays in the normal range.
    """
    torch.manual_seed(seed)
    numel = 1
    for dim in shape:
        numel *= dim
    factors = torch.tensor(_DYADIC_FACTORS, dtype=torch.float32)
    torch_input = factors[torch.randint(0, len(factors), (numel,))].reshape(shape).contiguous()
    exponent = int(torch.round(torch.log2(torch.prod(torch_input.to(torch.float64)).abs())).item())
    torch_input.view(-1)[0] = torch_input.view(-1)[0] * (2.0**-exponent)
    return torch_input.to(cpu_dtype)


def get_tensors(input_shape, output_shape, device, npu_dtype):
    cpu_dtype = torch.float32 if npu_dtype == ttnn.float32 else torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT

    torch_input = _exact_product_input(input_shape, cpu_dtype)
    torch_output = _exact_product_input(output_shape, cpu_dtype)
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

    # Reduce in float64 and cast back.
    # torch.prod on a bfloat16 tensor would run its own serial bfloat16 reduction,
    # accumulating a different error from the device's, in a different order.
    # Comparing the two would pit two approximations against each other rather than measuring the device against the true product.
    torch_output = torch.prod(torch_input.to(torch.float64)).to(torch_input.dtype)

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

    assert torch.isfinite(torch_output).all(), "golden overflowed; the stimulus no longer bounds the product"
    assert torch.isfinite(tt_output_cpu).all(), f"device produced a non-finite result: {tt_output_cpu}"

    # float32 reproduces the exact product to within a few ulps because the stimulus is
    # source-register exact. bfloat16 cannot: the intra-tile reduction is a serial host-side
    # loop over all 1024 elements of the final tile carried out in the tensor dtype
    # (prod_op_all.cpp -> prod_result_computation_WH_B0), so a bf16 run accumulates 1024
    # round-to-nearest steps. Measured worst case over these shapes is 7.9e-2 relative.
    tolerance = 1e-05 if npu_dtype == ttnn.float32 else 2e-01
    assert_numeric_metrics(
        torch_output,
        tt_output_cpu,
        pcc_threshold=0.9999,
        rtol=tolerance,
        atol=tolerance,
        frobenius_threshold=tolerance,
    )


# --------------------------------------------------------------------------------------------
# Accuracy characterisation.
#
# Every operand reaching the FPU is first truncated to the source register's 11-bit significand
# (10 stored mantissa bits plus the leading one). ttnn.prod multiplies *every* element of the
# input together, so each element whose mantissa does not fit contributes about half an ulp of
# that format, and the loss is one-sided (truncation, not round-to-nearest) so the errors
# accumulate coherently and the result is systematically low.
#
# The bounds below record measured Blackhole behaviour so that it is visible and regressions are
# caught. They are deliberately not a statement that this accuracy is good enough; if the fold is
# ever moved off the source-register path these bounds should be tightened, not relaxed.
# --------------------------------------------------------------------------------------------


def _prod_relative_error(torch_input, device, npu_dtype=ttnn.float32):
    """Run ttnn.prod and return (relative error vs a float64 golden, golden, device result)."""
    golden = torch.prod(torch_input.to(torch.float64))
    tt_input = ttnn.from_torch(torch_input, dtype=npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.to_torch(ttnn.prod(tt_input)).flatten()[0].to(torch.float64)
    assert torch.isfinite(tt_output), f"device produced a non-finite result: {tt_output}"
    return ((tt_output - golden) / golden).item(), golden.item(), tt_output.item()


# One full-mantissa factor per tile, everything else exactly 1.0, so the error isolates the
# per-tile contribution. Measured on Blackhole: 4.8e-4 / 2.0e-3 / 1.6e-2.
TRUNCATION_REL_ERROR_BOUND_BY_TILES = {1: 1e-03, 4: 4e-03, 16: 3e-02}


@pytest.mark.parametrize("num_tiles", tuple(TRUNCATION_REL_ERROR_BOUND_BY_TILES))
def test_prod_all_fp32_truncation_accuracy(num_tiles, device):
    torch.manual_seed(0)
    torch_input = torch.ones([1, num_tiles, 32, 32], dtype=torch.float32)
    for tile in range(num_tiles):
        torch_input[0, tile, 0, 0] = float(torch.empty(1).uniform_(0.7, 1.4).item())

    rel_error, golden, got = _prod_relative_error(torch_input, device)
    logger.info(f"{num_tiles} tiles: golden={golden} got={got} rel_error={rel_error}")

    bound = TRUNCATION_REL_ERROR_BOUND_BY_TILES[num_tiles]
    assert abs(rel_error) <= bound, f"{num_tiles} tiles: relative error {rel_error} exceeds documented bound {bound}"


# The error tracks the number of elements that do not fit the source-register mantissa, not the
# number of tiles: these all live in a *single* tile, where the cross-tile fold never runs at all.
# Measured on Blackhole: 4.9e-4 / 2.3e-2 / 3.1e-1.
TRUNCATION_REL_ERROR_BOUND_BY_ELEMENTS = {1: 1e-03, 64: 5e-02, 1024: 5e-01}


@pytest.mark.parametrize("num_non_unit", tuple(TRUNCATION_REL_ERROR_BOUND_BY_ELEMENTS))
def test_prod_all_fp32_truncation_scales_with_element_count(num_non_unit, device):
    torch.manual_seed(1)
    torch_input = torch.ones([1, 1, 32, 32], dtype=torch.float32)
    torch_input.view(-1)[:num_non_unit] = torch.empty(num_non_unit).uniform_(0.95, 1.05)

    rel_error, golden, got = _prod_relative_error(torch_input, device)
    logger.info(f"{num_non_unit} non-unit elements: golden={golden} got={got} rel_error={rel_error}")

    bound = TRUNCATION_REL_ERROR_BOUND_BY_ELEMENTS[num_non_unit]
    assert abs(rel_error) <= bound, f"{num_non_unit} elements: relative error {rel_error} exceeds bound {bound}"


def test_prod_all_fp32_exact_for_representable_input(device):
    """Values that fit the source-register mantissa are reduced essentially exactly.

    This is the control for the two tests above: same element count and same magnitudes, but
    every factor is dyadic, so nothing is truncated on the way into SrcA and the accumulated
    error drops by six orders of magnitude. It pins the cause to mantissa width rather than to
    the number of multiplies.
    """
    torch_input = _exact_product_input([1, 1, 32, 32], torch.float32, seed=3)

    rel_error, golden, got = _prod_relative_error(torch_input, device)
    logger.info(f"dyadic input: golden={golden} got={got} rel_error={rel_error}")

    assert abs(rel_error) <= 1e-05, f"exactly representable input should be near-exact, got {rel_error}"


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
