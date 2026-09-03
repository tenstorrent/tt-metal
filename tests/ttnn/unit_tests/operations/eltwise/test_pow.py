# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import struct
import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from tests.ttnn.utils_for_testing import (
    assert_with_pcc,
    assert_with_ulp,
    assert_allclose,
    flush_subnormal_values_to_zero,
)
from tests.ttnn.unit_tests.operations.eltwise.test_unary_pow import generate_clean_bf16_tensor

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("exponent", [0.5, 2.0, 4])
def test_unary_pow_ttnn(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.pow(input_tensor, exponent, output_tensor=output_tensor, queue_id=cq_id)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    golden_tensor = golden_fn(in_data, exponent)

    comp_pass = compare_pcc([output_tensor], [golden_tensor], pcc=0.9)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ([20, 20], [2, 32, 320], [1, 1, 32, 32], [1, 3, 320, 384], [1, 2, 32, 64, 64]),
)
@pytest.mark.parametrize("input", [10.0, 5.5, -5.0, -2.5, -10, -3, 9.5, -7.25, -6.15])
@pytest.mark.parametrize("exponent", [2.75, 2.5, 1.5, 4, 5.75, 0, -1.5, -2.25, -3, -4.25, -5.5])
# Both input and exponent are -ve and exponent is a non-integer, TT and Torch output = nan
# input = non-zero and exponent = 0, TT and Torch output = 1
# Both input and exponent are 0, TT = 1 and Torch output = 0
def test_binary_pow_scalar_input(input_shapes, input, exponent, device):
    torch_input_tensor_b = torch.full(input_shapes, exponent, dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(input, torch_input_tensor_b)

    cq_id = 0
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.pow(input, input_tensor_b, queue_id=cq_id)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.999)


def generate_torch_tensor(shape, low, high, step=0.0025, dtype=torch.float32):
    num_elements = torch.prod(torch.tensor(shape))
    values = torch.arange(low, high + step, step, dtype=dtype)

    if values.numel() < num_elements:
        values = values.repeat((num_elements // values.numel()) + 1)
    values = values[:num_elements]
    return values.reshape(shape)


@pytest.mark.parametrize(
    "input_shapes",
    [[64, 640], [2, 32, 320], [2, 1, 32, 1024], [1, 1, 32, 32], [1, 3, 320, 384], [1, 2, 32, 64, 128]],
)
def test_binary_sfpu_pow(device, input_shapes):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, -30, 30, step=0.0022)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, -20, 20)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    [[64, 640], [2, 32, 320], [2, 1, 1024, 1024], [1, 1, 32, 32], [1, 3, 320, 384], [1, 2, 32, 64, 64]],
)
def test_binary_sfpu_pow_bf16(device, input_shapes):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, -30, 30, step=0.0021, dtype=torch.bfloat16)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, -20, 20, dtype=torch.bfloat16)
    torch_output_tensor = torch.pow(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    [[2, 1, 32, 1024], [1, 3, 320, 384], [1, 2, 32, 64, 128], [1, 1, 32, 64]],
)
def test_binary_sfpu_pow_pos(device, input_shapes):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, 0, 30, step=0.0111)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, -20, 20)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    [[2, 1, 32, 1024], [1, 3, 320, 384], [1, 2, 32, 64, 128]],
)
def test_binary_sfpu_pow_neg(device, input_shapes):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, -30, 0, step=0.0111)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, 0, 10)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_binary_pow(device, dtype):
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)
    x_torch = torch.tensor([[0.98828125, 0.47851562, 1.1875, -1.59375]], dtype=torch_dtype)
    y_torch = torch.tensor([[0.0751953125, 0.53125, -0.6640625, 0.1533203125]], dtype=torch_dtype)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    z_tt_pow = ttnn.pow(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_pow)
    # output - bfloat16
    # Due to HW limitations for bfloat16 dtype, NaN value gets packed as inf.
    # z_tt_pow ttnn.Tensor([[ 0.99609,  0.67969,  ...,  0.89844,      inf]])
    # z_torch tensor([[1.0000, 0.6758, 0.8906,    nan]], dtype=torch.bfloat16)
    # output - float32
    # z_tt_pow ttnn.Tensor([[ 0.99930,  0.68274,  ...,  0.90147,      nan]])
    # z_torch tensor([[0.9991, 0.6760, 0.8922,    nan]])

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.99
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        [32, 64],
        [1, 128, 96],
        [5, 3, 64, 128],
    ),
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_binary_sfpu_pow_bug(device, input_shapes, dtype):
    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)
    torch_input_tensor_a = torch.randn(input_shapes, dtype=torch_dtype)
    torch_input_tensor_b = torch.randn(input_shapes, dtype=torch_dtype)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.999


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_binary_sfpu_accuracy(device, dtype):
    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)
    torch_input_tensor_a = torch.tensor([[10.0, 10.0, 9.0, 9.0, 5.0, 100000, 10.0, 10.0, 2.0, 2.0]], dtype=torch_dtype)
    torch_input_tensor_b = torch.tensor([[2.0, 3.0, 2.0, 3.0, 3.0, 1.7984, -1.0, -2.0, 1.0, 10.0]], dtype=torch_dtype)

    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    if dtype == "bfloat16":
        assert_with_ulp(torch_output_tensor, output, 1)
    else:
        assert_allclose(torch_output_tensor, output, rtol=0.005, atol=1e-3)  # Ensures > 99.5% accuracy


def test_special_input_fp32(device):
    a = torch.tensor(
        [[1.0, 0.999, 0.999, 0.999, 0.999, 0.234, 0.985, 1.456, 0.0, -1.0, 1.2, -5.3, 6.7, 9.8, -10.9, 5.999]],
        dtype=torch.float32,
    )
    b = torch.tensor(
        [[0.999, 1.0, 2.0, 3.0, 9.0, 0.123, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )

    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(a, b)

    input_tensor_a = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)
    assert_with_ulp(torch_output_tensor, output, ulp_threshold=2)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_pow_zero_base_special_cases(device, dtype):
    # Needs its own inputs: the shared generators strip ±0 before any caller sees them.
    # Asserts exact equality, not allclose — a regressed 2**(-127p) is ~7.7e-20 at p=0.5,
    # well inside any absolute tolerance this file uses elsewhere.
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)
    shape = [1, 1, 32, 32]
    positive_exponents = (1e-4, 0.01, 0.25, 0.5, 1.0 / 3.0, 0.75, 0.99, 1.5, 3.0, 4.0)

    def assert_zero_power_undefined(out):
        # The unary kernel (scalar exponent) is a separate implementation, unchanged by
        # tenstorrent/tt-metal#53922 and still returning NaN here.
        if dtype == "float32":
            assert torch.isnan(out).all()
        else:
            # bf16 dest reads back as inf rather than the NaN the kernel writes
            # (tenstorrent/tt-llk#675), so only non-finiteness is assertable here.
            assert (~torch.isfinite(out)).all()

    def assert_zero_power_negative_binary(out):
        # IEEE-754 pow(0, y) = +inf for y < 0, matching torch. The fp32 binary body
        # returns that directly now; it used to return NaN, which the header documented
        # as "undefined" (tenstorrent/tt-metal#53922). The bfloat16 body still writes
        # NaN and only reads back as inf because of the packer conversion above, so the
        # observable value is the same either way.
        assert torch.isinf(out).all() and (out > 0).all(), f"expected +inf, got {out.flatten()[0].item()}"

    for exponent in positive_exponents:
        exp_t = torch.full(shape, exponent, dtype=torch_dtype)
        tt_exp = ttnn.from_torch(exp_t, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        # rpow scalar base is always +0; one run per exponent covers that kernel.
        rpow_out = ttnn.to_torch(ttnn.rpow(tt_exp, 0.0))
        assert torch.equal(
            rpow_out, torch.zeros_like(rpow_out)
        ), f"rpow({exponent}, 0.0) = {rpow_out.flatten()[0].item()}"

        for base_val in (0.0, -0.0):
            zeros = torch.full(shape, base_val, dtype=torch_dtype)
            tt_zeros = ttnn.from_torch(zeros, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
            unary_out = ttnn.to_torch(ttnn.pow(tt_zeros, exponent))
            assert torch.equal(
                unary_out, torch.zeros_like(unary_out)
            ), f"unary pow({base_val}, {exponent}) = {unary_out.flatten()[0].item()}"

            binary_out = ttnn.to_torch(ttnn.pow(tt_zeros, tt_exp))
            assert torch.equal(
                binary_out, torch.zeros_like(binary_out)
            ), f"binary pow({base_val}, {exponent}) = {binary_out.flatten()[0].item()}"

    tt_zeros = ttnn.from_torch(
        torch.zeros(shape, dtype=torch_dtype), dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    # Scalar 0.0 is an integer exponent → power_iterative, not the SFPU kernel.
    unary_zero_exp = ttnn.to_torch(ttnn.pow(tt_zeros, 0.0))
    assert torch.equal(unary_zero_exp, torch.ones_like(unary_zero_exp))

    tt_zero_exp = ttnn.from_torch(
        torch.zeros(shape, dtype=torch_dtype), dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    binary_zero_exp = ttnn.to_torch(ttnn.pow(tt_zeros, tt_zero_exp))
    assert torch.equal(binary_zero_exp, torch.ones_like(binary_zero_exp))

    unary_neg = ttnn.to_torch(ttnn.pow(tt_zeros, -1.5))
    assert_zero_power_undefined(unary_neg)

    tt_neg_zero = ttnn.from_torch(
        torch.full(shape, -0.0, dtype=torch_dtype), dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_neg_exp = ttnn.from_torch(
        torch.full(shape, -1.5, dtype=torch_dtype), dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    binary_neg = ttnn.to_torch(ttnn.pow(tt_zeros, tt_neg_exp))
    binary_neg_zero = ttnn.to_torch(ttnn.pow(tt_neg_zero, tt_neg_exp))
    assert_zero_power_negative_binary(binary_neg)
    assert_zero_power_negative_binary(binary_neg_zero)


# IEEE-754 pow(0, y) is +0 for y > 0, +inf for y < 0, 1 for y = +/-0 and NaN for a NaN
# exponent. The kernel patches all of these on after the fact, because exp_21f builds a
# result out of the biased exponent field and cannot carry a non-finite argument through
# the mainline, so every case is a separate predicate that can regress on its own. Two of
# them did: pow(0, NaN) returned 0 and pow(0, y<0) returned NaN
# (tenstorrent/tt-metal#53922). This pins the whole block rather than one case.
def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def pow_result_matches(got, want):
    """Exact match, including the sign of a zero and NaN-ness.

    `got == want` is not enough: -0.0 == 0.0 in Python, so a sign-of-zero regression
    would go unnoticed, and nan != nan would fail a correct result.
    """
    if want != want:
        return got != got
    if want == 0.0 and got == 0.0:
        return _bits(got) == _bits(want)
    return got == want


def _fmt_pow(v):
    if v != v:
        return "nan"
    if v == 0.0:
        return "-0" if _bits(v) >> 31 else "+0"
    return f"{v:g}"


# Smallest positive normal, same for fp32 and bfloat16 (both have 8 exponent bits).
SMALLEST_NORMAL = 1.1754943508222875e-38
INF = float("inf")
NAN = float("nan")

# Every exponent class that reaches a distinct path through the zero-base block, since
# each is patched on by its own predicate. Grouped by what makes them distinct.
ZERO_BASE_EXPONENTS = (
    # Positive non-integers: the mainline result, 2**(0 * -127).
    (1e-4, "1e-4"),
    (0.01, "0.01"),
    (0.1, "0.1"),
    (0.25, "0.25"),
    (1.0 / 3.0, "1/3"),
    (0.5, "0.5"),
    (0.75, "0.75"),
    (0.99, "0.99"),
    (1.5, "1.5"),
    (SMALLEST_NORMAL, "smallest normal"),
    # Positive integers, odd and even -- odd ones are where a signed base would matter.
    (1.0, "1.0"),
    (2.0, "2.0"),
    (3.0, "3.0"),
    (4.0, "4.0"),
    # Both signed zeros. `!=` is bit-exact on SFPU, so -0 does not behave like +0 here:
    # it passes the `pow != 0` gate and then the sign-bit-based `pow < 0` narrowing.
    (0.0, "+0.0"),
    (-0.0, "-0.0"),
    # Negative exponents, the `pow < 0` fill. IEEE says +inf; this returned NaN before.
    (-0.5, "-0.5"),
    (-1.0, "-1.0"),
    (-1.5, "-1.5"),
    (-2.0, "-2.0"),
    (-2.5, "-2.5"),
    (-3.0, "-3.0"),
    (-4.0, "-4.0"),
    # Non-finite. +inf must stay 0 while NaN must give NaN, and the two differ only in
    # the mantissa, so they cannot be lumped together. A sign-bit-set NaN takes a
    # different route than a positive one -- an integer compare against +inf's bit
    # pattern gets -NaN wrong -- so both are covered.
    (INF, "+inf"),
    (-INF, "-inf"),
    (NAN, "+NaN"),
    (-NAN, "-NaN"),
)


def test_pow_zero_base_exponent_matrix(device):
    # fp32 only: the guards fixed by tenstorrent/tt-metal#53922 are in
    # _sfpu_binary_power_f32_, and the bfloat16 body is left as it is in main. With them
    # in, every case below matches torch exactly, so there is nothing to waive here.
    shape = [1, 1, 32, 32]

    def to_tt(fill):
        return ttnn.from_torch(
            torch.full(shape, fill, dtype=torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )

    tt_base = to_tt(0.0)
    base_t = torch.zeros(shape, dtype=torch.float32)

    failures = []
    for exp_val, exp_label in ZERO_BASE_EXPONENTS:
        want = torch.pow(base_t, torch.full(shape, exp_val, dtype=torch.float32)).flatten()[0].item()
        got = ttnn.to_torch(ttnn.pow(tt_base, to_tt(exp_val))).flatten()[0].item()
        if not pow_result_matches(got, want):
            failures.append(f"pow(0, {exp_label}) = {_fmt_pow(got)}, expected {_fmt_pow(want)}")

    assert not failures, "; ".join(failures)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_pow_determinism(device, dtype):
    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    shape = [512, 512]

    torch_a = torch.randn(shape, dtype=torch_dtype)
    torch_b = torch.randn(shape, dtype=torch_dtype)

    # Run the operations twice, and check that results are the same
    # This ensures that, by default, ttnn.pow is deterministic (expected behavior from MLIR)

    ttnn_a = ttnn.from_torch(torch_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # First round
    ttnn_result_1 = ttnn.pow(ttnn_a, ttnn_b)
    ttnn_result_1_torch = ttnn.to_torch(ttnn_result_1)

    # Second round
    ttnn_result_2 = ttnn.pow(ttnn_a, ttnn_b)
    ttnn_result_2_torch = ttnn.to_torch(ttnn_result_2)

    mask = torch.isnan(ttnn_result_1_torch) | torch.isnan(ttnn_result_2_torch)
    assert torch.equal(ttnn_result_1_torch[~mask], ttnn_result_2_torch[~mask])


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_binary_sfpu_accuracy_pos(device, dtype):
    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    def gen_uniform(shape, dtype, a, b):
        tensor = torch.rand(shape, dtype=dtype)
        tensor = tensor * (b - a) + a
        return tensor

    torch_input_tensor_a = gen_uniform([256, 256], torch_dtype, 1e-9, 1e3)
    torch_input_tensor_b = gen_uniform([256, 256], torch_dtype, 0.125, 4)

    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    if dtype == "bfloat16":
        assert_with_ulp(torch_output_tensor, output, 5)
    else:
        assert_allclose(torch_output_tensor, output, rtol=0.02, atol=1e-5)  # Ensure > 98% accuracy


@pytest.mark.parametrize(
    "input_a, input_b",
    [
        ([32, 64], [32, 64]),
        ([1, 128, 96], [1, 128, 1]),
        ([5, 3, 1, 128], [5, 1, 64, 128]),
        ([2, 1, 1, 1, 1], [2, 1, 2, 64, 128]),
        ([], [128]),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_binary_ng_pow(device, input_a, input_b, dtype):
    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)
    torch_input_tensor_a = torch.randn(input_a, dtype=torch_dtype)
    torch_input_tensor_b = torch.randn(input_b, dtype=torch_dtype)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.999


# unary power FP32 tests
@pytest.mark.parametrize("exponent", [2.0, -2.0, -3.56, 0.5, -0.5, -0.566, -2])
def test_pow(exponent, device):
    torch.manual_seed(42)

    torch_base = torch.rand([4, 4], dtype=torch.float32)
    torch_output = torch.pow(torch_base, exponent)
    ttnn_base = ttnn.from_torch(torch_base, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_output = ttnn.pow(ttnn_base, exponent)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_allclose(torch_output, ttnn_output, atol=2.5e-4, rtol=5e-7)


# Dense log-spaced base sweep accuracy for non-integer exponents (issue #49625).
# fp32 pow(x, y) must stay < 3 ULP for the CogVideo/DeepSeek regime x in [0.5, 50000],
# while integer exponents stay bit-exact (0 ULP). Covers the long-mantissa 1.7984 case.
@pytest.mark.parametrize("exponent", [0.5, 1.5, 1.7984, 2.5])
def test_unary_pow_fp32_ulp_noninteger(exponent, device):
    base = torch.logspace(-0.3, 4.7, 1024, dtype=torch.float32).reshape(32, 32)  # ~0.5 .. ~50000
    golden = torch.pow(base, exponent)

    tt_base = ttnn.from_torch(base, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.pow(tt_base, exponent)
    result = ttnn.to_torch(tt_out)

    assert_with_ulp(golden, result, ulp_threshold=3)


@pytest.mark.parametrize("exponent", [2.0, 3.0])
def test_unary_pow_fp32_ulp_integer_exact(exponent, device):
    base = torch.logspace(-0.3, 4.7, 1024, dtype=torch.float32).reshape(32, 32)
    golden = torch.pow(base, exponent)

    tt_base = ttnn.from_torch(base, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.pow(tt_base, exponent)
    result = ttnn.to_torch(tt_out)

    assert_with_ulp(golden, result, ulp_threshold=0)


# Overflow must saturate to +inf, not wrap. The non-integer fp32 path scales by 2**k via
# setexp, which writes the 8-bit exponent field and wraps instead of saturating, so a
# missing clamp turns an overflowing result into a finite garbage value. Exponents are
# non-integer on purpose: integer exponents take a separate exact iterative path that is
# not affected. base=50000, y=9.5 gives ~1e42 -> +inf in fp32. Covers both the unary
# (scalar y) and binary (tensor y) paths, which apply the 2**k scale independently.
@pytest.mark.parametrize("exponent", [9.5, 20.5, 50.5])
def test_unary_pow_fp32_overflow_to_inf(exponent, device):
    base = torch.full((32, 32), 50000.0, dtype=torch.float32)
    golden = torch.pow(base, exponent)
    assert torch.isinf(golden).all()  # sanity: this exponent really overflows fp32

    tt_base = ttnn.from_torch(base, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.pow(tt_base, exponent))

    assert torch.isinf(result).all(), "overflow wrapped to a finite value instead of +inf"
    assert (result > 0).all(), "overflow produced -inf/NaN instead of +inf"


@pytest.mark.parametrize("exponent", [9.5, 20.5, 50.5])
def test_binary_pow_fp32_overflow_to_inf(exponent, device):
    base = torch.full((32, 32), 50000.0, dtype=torch.float32)
    exp = torch.full((32, 32), exponent, dtype=torch.float32)
    golden = torch.pow(base, exp)
    assert torch.isinf(golden).all()  # sanity: this exponent really overflows fp32

    tt_base = ttnn.from_torch(base, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_exp = ttnn.from_torch(exp, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.pow(tt_base, tt_exp))

    assert torch.isinf(result).all(), "overflow wrapped to a finite value instead of +inf"
    assert (result > 0).all(), "overflow produced -inf/NaN instead of +inf"


@pytest.mark.parametrize("exponent", [0.25, 0.5, 0.75, -0.25, -0.5, -0.75])
def test_pow_arange_masking_fp32(exponent, device):
    tt_input = generate_clean_bf16_tensor(torch.float32)

    tt_input = flush_subnormal_values_to_zero(tt_input)
    in_range = (tt_input.abs() >= 1e-4) & (tt_input.abs() <= 1e4)
    tt_input = tt_input[in_range]

    tt_in = ttnn.from_torch(
        tt_input,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = golden_function(tt_input, exponent, device=device)

    tt_result = ttnn.pow(tt_in, exponent)
    result = ttnn.to_torch(tt_result)
    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    assert_allclose(golden, result, atol=5e-4, rtol=8e-7)
