# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_allclose, assert_equal

TEST_PADDING_VALUE = -42


def assert_cumsum_quality(expected_output, torch_output):
    if torch_output.dtype == torch.int32:
        assert_equal(expected_output, torch_output)
    elif torch_output.dtype == torch.bfloat16:
        assert_with_ulp(expected_result=expected_output, actual_result=torch_output, ulp_threshold=1)
    else:
        assert_allclose(expected_output, torch_output, rtol=1e-2, atol=1e-4)


def get_backward_tensors(output_grad_shape, input_grad_shape, device):
    torch.manual_seed(2023)
    npu_dtype = ttnn.bfloat16
    cpu_dtype = torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT
    torch_output_grad = torch.randint(-2, 3, output_grad_shape, dtype=cpu_dtype, requires_grad=True)
    torch_input_grad = torch.randint(-2, 3, input_grad_shape, dtype=cpu_dtype)

    tt_output_grad = ttnn.Tensor(torch_output_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_input_grad = ttnn.Tensor(torch_input_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_output_grad, tt_input_grad, torch_output_grad


@pytest.mark.parametrize(
    "size, dim",
    [
        ([], 0),
        ([1], 0),
        ([2, 3], 0),
        ([2, 3], -1),
        ([1, 1024, 32], 0),
        ([33, 35, 37], -1),
        ([7, 13, 129, 33], 1),
        ([2, 3, 5, 33, 128], -1),
        ([5, 2, 3, 5, 33, 128], 0),
        ([19], -1),
        ([1, 19], -1),
        ([1, 4736], -1),  # long scan; the [1, 151936] vocab width is in the reduction nightly
        ([5], -1),
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [(torch.float32, None), (torch.bfloat16, ttnn.bfloat16), (torch.int32, ttnn.int32)],
)
def test_cumsum(size, dim, dtypes, device):
    torch.manual_seed(29112024)

    (torch_dtype, ttnn_dtype) = dtypes

    # Integer input on [-2; 2] keeps the accumulation exact (integer partial sums are exact in the
    # fp32 and int32 accumulators), though bf16 outputs beyond +-256 may still round. This test
    # checks indexing, dtype and padding handling; accuracy on real-valued inputs is covered by the
    # *_accuracy tests below.
    for _ in range(2):
        torch_input_tensor = torch.randint(-2, 3, size=size, dtype=torch_dtype)
        input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.Layout.TILE)
        input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

        expected_output_dtype = ttnn_dtype if ttnn_dtype is not None else input_tensor.dtype

        output_tensor = ttnn.cumsum(input_tensor, dim=dim, dtype=ttnn_dtype)

        assert output_tensor.dtype == expected_output_dtype
        assert output_tensor.shape == (size)

        torch_output = ttnn.to_torch(output_tensor, dtype=torch_dtype)

        expected_output = torch.cumsum(torch_input_tensor, dim=dim, dtype=torch_dtype)

        assert_cumsum_quality(expected_output, torch_output)


@pytest.mark.parametrize(
    "size, dim",
    [
        ([], 0),
        ([1], 0),
        ([2, 3], 0),
        ([2, 3], -1),
        ([1, 1024, 32], 0),
        ([33, 35, 37], -1),
        ([7, 13, 129, 33], 1),
        ([2, 3, 5, 33, 128], -1),
        ([5, 2, 3, 5, 33, 128], 0),
        ([1, 4736], -1),  # long scan; the [1, 151936] vocab width is in the reduction nightly
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
        (torch.float32, ttnn.bfloat16),
    ],
)
def test_cumsum_with_preallocated_output(size, dim, dtypes, device):
    torch.manual_seed(29112024)

    (torch_dtype, ttnn_dtype) = dtypes

    torch_input_tensor = torch.randint(-2, 3, size, dtype=torch_dtype)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, dtype=ttnn_dtype, layout=ttnn.Layout.TILE)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    expected_output_dtype = ttnn_dtype if ttnn_dtype is not None else input_tensor.dtype

    preallocated_output_tensor = ttnn.zeros_like(input_tensor, dtype=ttnn_dtype, layout=ttnn.Layout.TILE)

    output_tensor = ttnn.cumsum(input_tensor, dim=dim, dtype=ttnn_dtype, out=preallocated_output_tensor)
    torch_output = ttnn.to_torch(output_tensor, dtype=torch_dtype)

    expected_output = torch.cumsum(torch_input_tensor, dim=dim, dtype=torch_dtype)

    assert output_tensor.dtype == expected_output_dtype
    assert preallocated_output_tensor.dtype == expected_output_dtype

    assert output_tensor.shape == (size)
    assert preallocated_output_tensor.shape == (size)

    assert preallocated_output_tensor == output_tensor

    assert_cumsum_quality(expected_output, torch_output)

    assert device.num_program_cache_entries() >= 1


@pytest.mark.parametrize(
    "size, dim",
    [
        ([], 0),
        ([1], 0),
        ([2, 3], 0),
        ([2, 3], -1),
        ([1, 1024, 32], 0),
        ([33, 35, 37], -1),
        ([7, 13, 129, 33], 1),
        ([2, 3, 5, 33, 128], -1),
        ([5, 2, 3, 5, 33, 128], 0),
        ([1, 4736], -1),  # long scan; the [1, 151936] vocab width is in the reduction nightly
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
    ],
)
def test_cumsum_backward(size, dim, dtypes, device):
    output_shape = size.copy()

    torch.manual_seed(29112024)

    (torch_dtype, ttnn_dtype) = dtypes

    # Integer input on [-2; 2] keeps the accumulation exact, though the bf16 gradient sums may still
    # round beyond +-256; see test_cumsum.
    torch_input_tensor = torch.randint(-2, 3, size=size, dtype=torch_dtype, requires_grad=True)

    (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(size, size, device)

    torch_output = torch.cumsum(torch_input_tensor, dim)
    torch_output.backward(torch_output_grad)

    tt_input_grad_cpu = ttnn.to_torch(
        ttnn.cumsum(tt_output_grad, dim, dtype=ttnn_dtype, reverse_order=True, out=tt_input_grad)
    )

    assert tt_input_grad_cpu.shape == torch_input_tensor.grad.shape
    assert_cumsum_quality(torch_input_tensor.grad, tt_input_grad_cpu)


# The preallocated `out` tensor must live on device: the on-device check in
# validate_output_tensor fires before any other validation, so a host `out`
# would mask the check each row targets. Each row asserts the exact message of
# the check it exercises.
@pytest.mark.parametrize(
    "dim, input_shape, output_shape, torch_dtype, input_dtype, output_dtype, memory_config, layout, error_msg",
    [
        (
            -10,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
            "The requested accumulation axis is -10, while the input tensor has rank 9",
        ),  # input_rank vs dim
        (
            10,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
            "The requested accumulation axis is 10, while the input tensor has rank 9",
        ),  # input_rank vs dim
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
            "Shape mismatch: input tensor shape",
        ),  # input_shape vs output_shape
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 1],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
            "Shape mismatch: input tensor shape",
        ),  # input_shape vs output_shape
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.ROW_MAJOR,
            "The provided input tensor has a non-tile layout: ROW_MAJOR",
        ),  # unsupported layout
    ],
)
def test_cumsum_failing_cases(
    dim,
    input_shape,
    output_shape,
    torch_dtype,
    input_dtype,
    output_dtype,
    memory_config,
    layout,
    error_msg,
    device,
    expect_error,
):
    torch.manual_seed(0)
    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=layout, device=device, memory_config=memory_config
    )
    ttnn_preallocated_tensor = ttnn.zeros(output_shape, dtype=output_dtype, layout=ttnn.Layout.TILE, device=device)
    with expect_error(RuntimeError, error_msg):
        ttnn.cumsum(ttnn_input_tensor, memory_config=memory_config, dim=dim, out=ttnn_preallocated_tensor)


@pytest.mark.parametrize(
    "size, dim",
    [
        # The #55542 geometry (long thin scan on one core) at a shorter length. The full-length
        # shapes ([1, 72192, 9], [1, 151936], [4, 65536]) are in the reduction nightly.
        ([1, 8192, 9], 1),
        ([4, 8192], -1),
    ],
)
@pytest.mark.parametrize("sequence_type", ["iid", "piecewise_constant"])
def test_cumsum_fp32_long_scan_accuracy(size, dim, sequence_type, device):
    """fp32 accuracy over a LONG scan with REAL-valued inputs.

    The functional tests in this file draw integers on [-2, 2]. Integer partial sums are exact in
    fp32, so those tests cannot observe accumulation error at all. This one can.

    The metric is the compensated-summation error bound, per output element:
        |out_i - exact_i|  <=  K * eps32 * sum_{j<=i} |x_j|
    Kahan summation guarantees this with K ~ 2 independent of scan length. Plain sequential fp32
    accumulation has error growing as ~T^1.5 when consecutive rounding errors share a sign: on the
    piecewise-constant sequence below it measures K ~ 216 (#55542: 1825 ULP, sign flips near zero
    crossings). On the iid sequence the errors cancel and the plain path sits at K ~ 0.5, so that
    case discriminates nothing and is here only to show the change does not regress it. K = 8:
    4x margin over the guarantee, 25x below the plain path on the discriminating sequence at the
    #55542 length (the plain path already reaches K ~ 22 at the 8192 used here).

    Why not ULP of the reference: an iid sequence's running sum is a random walk that crosses zero,
    where one ULP of the reference is ~1e-12 and any fp32 method reads as "thousands of ULP" while
    its absolute error is ~1e-5. (torch.cumsum on CPU accumulates fp32 in double, which is why it
    reads 0.5 ULP everywhere -- it is a reference, not an fp32 peer.) The piecewise-constant
    sequence is the hard case for sequential summation: autocorrelated inputs make consecutive
    rounding errors share a sign, so they add coherently instead of partly cancelling.
    """
    torch.manual_seed(29112024)
    n = size[dim]
    if sequence_type == "iid":
        along = torch.randn(n, dtype=torch.float32)
    else:
        along = torch.randn(n // 256 + 1, dtype=torch.float32).repeat_interleave(256)[:n]
    shape_along = [1] * len(size)
    shape_along[dim] = n
    torch_input = along.view(shape_along).expand(size).contiguous()

    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)
    output = ttnn.to_torch(ttnn.cumsum(input_tensor, dim=dim)).to(torch.float64)

    x64 = torch_input.to(torch.float64)
    reference = torch.cumsum(x64, dim=dim)
    bound = 8.0 * 2.0**-23 * torch.cumsum(x64.abs(), dim=dim)
    err = (output - reference).abs()

    assert torch.isfinite(output).all()
    worst = (err / bound).max().item()
    assert (err <= bound).all(), (
        f"max error is {worst:.1f}x the compensated-summation bound 8*eps*sum|x| (sequence_type={sequence_type}); "
        "plain sequential fp32 sits near 216x on the piecewise-constant sequence"
    )


def test_cumsum_disable_compensated_sum(device, n=8192):
    """`disable_compensated_sum=True` must actually fall back to the plain sequential sum.

    Uses the discriminating sequence from the accuracy test above (piecewise-constant, where
    consecutive rounding errors share a sign). The default compensated path stays within the
    8*eps*sum|x| bound; the plain path is expected well outside it (~22x at n=8192, ~216x at the
    72192 of #55542, which the reduction nightly runs). If the flag did nothing, both would pass the
    bound and the escape hatch would be a silent no-op.
    """
    torch.manual_seed(29112024)
    along = torch.randn(n // 256 + 1, dtype=torch.float32).repeat_interleave(256)[:n]
    torch_input = along.view(1, n, 1).expand(1, n, 9).contiguous()

    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)
    compensated = ttnn.to_torch(ttnn.cumsum(input_tensor, dim=1)).to(torch.float64)
    plain = ttnn.to_torch(ttnn.cumsum(input_tensor, dim=1, disable_compensated_sum=True)).to(torch.float64)

    x64 = torch_input.to(torch.float64)
    reference = torch.cumsum(x64, dim=1)
    bound = 8.0 * 2.0**-23 * torch.cumsum(x64.abs(), dim=1)

    comp_err = (compensated - reference).abs()
    plain_err = (plain - reference).abs()

    assert torch.isfinite(compensated).all() and torch.isfinite(plain).all()
    # Default keeps the guarantee.
    assert (
        comp_err <= bound
    ).all(), f"compensated path exceeded its own bound at {(comp_err / bound).max().item():.1f}x"
    # Disabling it must degrade accuracy on the discriminating sequence -- otherwise the flag is inert.
    assert (plain_err / bound).max().item() > 4.0, (
        "disable_compensated_sum=True did not change the result: the plain path stayed within the "
        "compensated bound, so the flag is not reaching the accumulation kernel"
    )


@pytest.mark.parametrize(
    "size, dim",
    [
        ([1, 8192], -1),  # long scan; [1, 151936] and [1, 72192, 9] are in the reduction nightly
        ([1, 8192, 9], 1),  # the long thin geometry from #55542
        ([4096, 32], 0),
        ([33, 35, 37], -2),  # height not a multiple of 32: tile padding along the scan axis
        ([3, 1000], -1),  # width not a multiple of 32: tile padding along the scan axis
        ([7, 13, 129, 33], 1),
        ([5, 2, 3, 5, 33, 128], 0),  # rank > 4, accumulation axis not permuted
    ],
)
@pytest.mark.parametrize("sequence_type", ["iid", "piecewise_constant"])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_cumsum_bf16_accuracy(size, dim, sequence_type, reverse_order, device):
    """bf16 accuracy with REAL-valued inputs must match torch (#33742).

    torch.cumsum on a bf16 CPU tensor accumulates in fp32 and rounds each output to bf16 once. The
    device does the same: the accumulator is fp32 (#41371) and only the output is bf16. So torch is
    a like-for-like reference here, and both have the same error against an exact sum.

    The tolerance is 1 ULP rather than bit-exact, for two measured reasons. Short bf16 sums often
    land exactly halfway between two bf16 values; the device rounds those ties away from zero,
    torch rounds them to even. And the device's fp32 running sum is not bit-identical to the CPU's,
    which on long scans occasionally moves a value across a bf16 rounding boundary. A bf16
    accumulator, by contrast, stalls once the running sum outgrows the inputs and is off by many
    ULP on these lengths.

    The piecewise-constant sequence is the hard case for sequential summation (see
    test_cumsum_fp32_long_scan_accuracy). Padding is filled with a non-zero value, and the
    [33, 35, 37] (dim -2) and [3, 1000] (dim -1) cases have tile padding along the scan axis, so a
    reverse scan that picked up padding would fail.
    """
    torch.manual_seed(29112024)
    n = size[dim]
    if sequence_type == "iid":
        torch_input = torch.randn(size)
    else:
        along = torch.randn(n // 256 + 1).repeat_interleave(256)[:n]
        shape_along = [1] * len(size)
        shape_along[dim] = n
        torch_input = along.view(shape_along).expand(size).contiguous()
    torch_input = torch_input.to(torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    output = ttnn.to_torch(ttnn.cumsum(input_tensor, dim=dim, reverse_order=reverse_order))

    if reverse_order:
        expected = torch.flip(torch.cumsum(torch.flip(torch_input, [dim]), dim), [dim])
    else:
        expected = torch.cumsum(torch_input, dim)

    assert output.dtype == torch.bfloat16
    assert_with_ulp(expected_result=expected, actual_result=output, ulp_threshold=1)


@pytest.mark.parametrize("disable_compensated_sum", [False, True])
@pytest.mark.parametrize("dim", [0, -2])
@pytest.mark.parametrize("scan_length", [4, 8, 33])
@pytest.mark.parametrize(
    "pattern",
    [
        "inf",  # +inf at step 1: every later element stays +inf
        "neg_inf",  # -inf at step 1: every later element stays -inf
        "overflow",  # 3e38 + 3e38 overflows to +inf at step 1
        "neg_overflow",  # -3e38 + -3e38 overflows to -inf at step 1
        "inf_then_neg_max",  # +inf, then minus the largest fp32: the total stays +inf
        "neg_inf_then_pos_max",  # -inf, then the largest fp32: the total stays -inf
        "overflow_then_neg_max",  # an overflow to +inf, then minus the largest fp32 at once: still +inf
        "neg_overflow_then_pos_max",  # an overflow to -inf, then the largest fp32 at once: still -inf
        "inf_on_large_neg_total",  # +inf arriving on a total of -3e38: +inf from there on
        "neg_inf_on_large_pos_total",  # -inf arriving on a total of 3e38: -inf from there on
        "finite_with_overflowing_c",  # the total stays finite, but t - acc overflows
        "inf_then_neg_inf",  # +inf then -inf: NaN from step 2 on, as in torch
        "nan",  # NaN input: NaN from step 1 on
        "inf_last",  # +inf at the last step only
    ],
)
def test_cumsum_fp32_non_finite(pattern, scan_length, dim, disable_compensated_sum, device):
    """Once the running total is +-inf or NaN, every later element must match an fp32 running total.

    The compensated path keeps the rounding error of each add as c = (t - acc) - y. Once the running
    total t is +-inf, c is inf - inf = NaN (or +-inf after an overflow), and the next step would turn
    every later element of the scan into NaN, while the plain path keeps returning +-inf. The first
    ten patterns cover that: an inf or an overflow of either sign, then the largest fp32 with the
    other sign, and an inf arriving on a large total of the other sign. In finite_with_overflowing_c
    the total stays finite, but t - acc overflows, so c is +inf all the same. The last three already
    gave NaN or a lone inf and must keep doing so.

    The reference is torch.cumsum, except in the two overflow_then patterns. torch.cumsum on CPU
    accumulates fp32 in double (see test_cumsum_fp32_long_scan_accuracy), so there it returns a
    finite value again once the largest fp32 is subtracted, while an fp32 running total, which is
    what both device paths keep, stays at +-inf. Those two use an fp32 running total as the
    reference.
    """
    if (
        device.arch() == ttnn.device.Arch.QUASAR
        and not disable_compensated_sum
        and pattern not in ("inf_then_neg_inf", "nan", "inf_last")
    ):
        pytest.xfail("Quasar keeps the unguarded compensation: https://github.com/tenstorrent/tt-metal/issues/58986")
    torch.manual_seed(0)
    size = [scan_length, 1, 32, 64] if dim == 0 else [1, 1, scan_length, 64]
    torch_input = torch.randn(size, dtype=torch.float32)
    steps = torch_input.movedim(dim, 0)  # a view: steps[i] is step i of the scan
    fp32_max = torch.finfo(torch.float32).max
    # Each pattern sets a few of the 64 columns at the scan steps its name describes; the other columns stay random.
    # A column only gets one sign of inf, except in inf_then_neg_inf.
    if pattern == "inf":
        steps[1, ..., :5] = float("inf")
    elif pattern == "neg_inf":
        steps[1, ..., 7:12] = float("-inf")
    elif pattern == "overflow":
        steps[0, ..., 20:25] = 3e38
        steps[1, ..., 20:25] = 3e38
    elif pattern == "neg_overflow":
        steps[0, ..., 50:55] = -3e38
        steps[1, ..., 50:55] = -3e38
    elif pattern == "inf_then_neg_max":
        steps[1, ..., 34:38] = float("inf")
        steps[2, ..., 34:38] = -fp32_max
    elif pattern == "neg_inf_then_pos_max":
        steps[1, ..., 44:48] = float("-inf")
        steps[2, ..., 44:48] = fp32_max
    elif pattern == "overflow_then_neg_max":
        steps[:2, ..., 2:6] = 3e38
        steps[2, ..., 2:6] = -fp32_max
    elif pattern == "neg_overflow_then_pos_max":
        steps[:2, ..., 8:12] = -3e38
        steps[2, ..., 8:12] = fp32_max
    elif pattern == "inf_on_large_neg_total":
        steps[0, ..., 14:18] = -3e38
        steps[1, ..., 14:18] = float("inf")
    elif pattern == "neg_inf_on_large_pos_total":
        steps[0, ..., 26:30] = 3e38
        steps[1, ..., 26:30] = float("-inf")
    elif pattern == "finite_with_overflowing_c":
        steps[0, ..., 56:60] = -(2.0**126 + 3 * 2.0**103)
        steps[1, ..., 56:60] = fp32_max
    elif pattern == "inf_then_neg_inf":
        steps[1, ..., 30:33] = float("inf")
        steps[2, ..., 30:33] = float("-inf")
    elif pattern == "nan":
        steps[1, ..., 40:43] = float("nan")
    else:
        steps[-1, ..., 60:64] = float("inf")

    expected = torch.cumsum(torch_input, dim=dim)
    if pattern in ("overflow_then_neg_max", "neg_overflow_then_pos_max"):
        expected_steps = expected.movedim(dim, 0)  # a view, like steps
        total = torch.zeros_like(steps[0])
        for i in range(scan_length):
            total = total + steps[i]
            expected_steps[i] = total

    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)
    output = ttnn.to_torch(ttnn.cumsum(input_tensor, dim=dim, disable_compensated_sum=disable_compensated_sum))

    for name, kind in (("NaN", torch.isnan), ("+inf", torch.isposinf), ("-inf", torch.isneginf)):
        assert torch.equal(
            kind(output), kind(expected)
        ), f"{name} in {kind(output).sum().item()} elements, the reference has {kind(expected).sum().item()}"
    finite = torch.isfinite(expected)
    assert_allclose(expected[finite], output[finite], rtol=1e-2, atol=1e-4)


@pytest.mark.parametrize("dim", [0, -2])
def test_cumsum_fp32_finite_near_max(dim, device):
    """A finite running total in the top fp32 binade must keep the compensated accuracy.

    From 2^127 up, the rounding error of one add can reach 2^103, and the guard for non-finite
    totals (see the kernel) must leave the compensation term of a finite step alone: capping it at
    2^102 puts this scan at up to 2.0x the limit below, which is |out - exact| <= 2 * eps32 *
    |exact|, 2 to 4 ulp in this binade. (The sum-of-magnitudes bound of
    test_cumsum_fp32_long_scan_accuracy is too loose to see a drift of a few ulp here.)
    """
    torch.manual_seed(0)
    scan_length = 128
    size = [scan_length, 1, 32, 64] if dim == 0 else [1, 1, scan_length, 64]
    base = 2.0**127
    torch_input = base * (0.001 + 0.019 * torch.rand(size, dtype=torch.float64))
    torch_input *= torch.where(torch.rand(size) < 0.5, -1.0, 1.0)
    steps = torch_input.movedim(dim, 0)  # a view: steps[i] is step i of the scan
    steps[0] = base * (1.2 + 0.25 * torch.rand(steps[0].shape, dtype=torch.float64))
    steps[..., 32:] *= -1  # negative totals in half of the columns
    torch_input = torch_input.to(torch.float32)
    expected = torch.cumsum(torch_input.to(torch.float64), dim=dim)
    assert expected.abs().max() < torch.finfo(torch.float32).max, "the reference scan must stay finite"

    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)
    output = ttnn.to_torch(ttnn.cumsum(input_tensor, dim=dim)).to(torch.float64)

    assert torch.isfinite(output).all(), f"{(~torch.isfinite(output)).sum().item()} non-finite elements"
    bound = 2 * torch.finfo(torch.float32).eps * expected.abs()
    worst = ((output - expected).abs() / bound).max().item()
    assert worst <= 1.0, f"max error is {worst:.1f}x the bound 2*eps*|exact|"


@pytest.mark.parametrize("dim", [0, -2])
def test_cumsum_fp32_finite_large_inputs(dim, device):
    """A finite scan whose inputs are as large as its running total must keep the compensated accuracy.

    When an input is larger than the running total, c is the sum of two rounding errors and reaches
    2^104 in this scan. The guard for non-finite totals (see the kernel) must leave it alone:
    capping c at 2^103 puts the last element at 32.8x the limit below, the same limit as in
    test_cumsum_fp32_finite_near_max.
    """
    scan = [
        float.fromhex(h)
        for h in ("-0x1.c4dbcep+126", "0x1.f74fccp+127", "-0x1.f60404p+125", "0x1.40bc9ap+127", "-0x1.d43534p+127")
    ]
    size = [len(scan), 1, 32, 64] if dim == 0 else [1, 1, len(scan), 64]
    torch_input = torch.empty(size, dtype=torch.float32)
    steps = torch_input.movedim(dim, 0)  # a view: steps[i] is step i of the scan
    for i, value in enumerate(scan):
        steps[i] = value
    expected = torch.cumsum(torch_input.to(torch.float64), dim=dim)

    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)
    output = ttnn.to_torch(ttnn.cumsum(input_tensor, dim=dim)).to(torch.float64)

    assert torch.isfinite(output).all(), f"{(~torch.isfinite(output)).sum().item()} non-finite elements"
    bound = 2 * torch.finfo(torch.float32).eps * expected.abs()
    worst = ((output - expected).abs() / bound).max().item()
    assert worst <= 1.0, f"max error is {worst:.1f}x the bound 2*eps*|exact|"


def _cumsum_calltrace(input_tensor, **kwargs):
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        output_tensor = ttnn.cumsum(input_tensor, **kwargs)
    finally:
        captured_graph = ttnn.graph.end_graph_capture()
    return output_tensor, ttnn.graph.extract_calltrace(captured_graph)


@pytest.mark.parametrize(
    "size, dim",
    [
        ([1048576], 0),  # 1D: the permute used to pad every element out to its own tile
        ([1, 151936], -1),
        ([151936, 1], -2),
        ([33, 35, 37], -1),
        ([33, 35, 37], -2),
        ([100, 64, 64], -1),  # more rows of tiles than cores: both core groups
        ([100, 64, 64], -2),
        ([2, 3, 5, 33, 128], -1),
    ],
)
def test_cumsum_tile_axis_runs_in_place(size, dim, device):
    """A forward bf16 cumsum along H or W must scan the tiles in place (#24824, #23264).

    Before, the scan axis was permuted to dim 0, which pads every remaining element of a thin tensor
    out to its own row or tile: 32x the memory, and 1024x for a 1D tensor (a 10.8M-element int32
    scan tried to allocate 44 GB). The call trace checks the structure directly: no permute, and the
    output is the only device tensor the op creates.
    """
    torch.manual_seed(29112024)
    torch_input = torch.randn(size).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor, calltrace = _cumsum_calltrace(input_tensor, dim=dim)

    assert "PermuteDeviceOperation" not in calltrace, calltrace
    assert calltrace.count("tt::tt_metal::create_device_tensor") == 1, calltrace
    assert_with_ulp(
        expected_result=torch.cumsum(torch_input, dim),
        actual_result=ttnn.to_torch(output_tensor),
        ulp_threshold=1,
    )


@pytest.mark.parametrize(
    "torch_dtype, reverse_order",
    [
        (torch.float32, False),  # keeps the compensated (Kahan) accumulation of the permuted path
        (torch.int32, False),  # cumsum_tile only adds floats
        (torch.bfloat16, True),  # cumsum_tile only scans forward
    ],
)
def test_cumsum_tile_axis_fallback_permutes(torch_dtype, reverse_order, device):
    """Scans the tile-axis kernel does not cover yet still take the permuted path, and stay correct.

    This also shows the call-trace check in test_cumsum_tile_axis_runs_in_place can see a permute.
    """
    torch.manual_seed(29112024)
    size, dim = [33, 35, 37], -1
    torch_input = torch.randint(-2, 3, size, dtype=torch_dtype)
    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)

    output_tensor, calltrace = _cumsum_calltrace(input_tensor, dim=dim, reverse_order=reverse_order)

    assert "PermuteDeviceOperation" in calltrace, calltrace
    if reverse_order:
        expected = torch.flip(torch.cumsum(torch.flip(torch_input, [dim]), dim), [dim])
    else:
        expected = torch.cumsum(torch_input, dim)
    assert_cumsum_quality(expected, ttnn.to_torch(output_tensor, dtype=torch_dtype))
