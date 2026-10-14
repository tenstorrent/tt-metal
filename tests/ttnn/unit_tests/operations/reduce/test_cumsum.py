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
        ([1, 151936], -1),
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
        ([1, 151936], -1),
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
        ([1, 151936], -1),
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
        ([1, 72192, 9], 1),  # the shape from #55542: 2256 tiles along the scan, one core
        ([1, 151936], -1),  # the longest fp32 shape the tests above already run
        ([4, 65536], -1),
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
    4x margin over the guarantee, 25x below the plain path on the discriminating sequence.

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


def _build_special_value_row(length, pattern):
    """Build a 1D float32 row embedding a special-value case from #58986.

    - "pos_inf" / "neg_inf": a single +-inf at index 1, 1.0 elsewhere. The running total goes
      non-finite at index 1 and must stay a well-defined +-inf for every later element, matching
      torch -- this is the regression itself (the Kahan compensation term's `(t - acc) - y` landed
      NaN once `t` went infinite, which poisoned every element after it).
    - "overflow": two values near the float32 max (2**127) at indices 0 and 1, so their sum
      overflows to +inf purely from finite inputs (no inf in the input at all), 1.0 elsewhere. Same
      failure mode as "pos_inf"/"neg_inf" but triggered by overflow instead of an infinite input.
    - "nan": NaN at index 1, 1.0 elsewhere. Must keep propagating NaN (unchanged behavior: the guard
      only changes what happens to the *compensation* term, never the running total itself).
    - "inf_at_end": +inf only at the last index. Nothing downstream to poison -- unchanged behavior.
    - "pos_then_neg_inf": +inf then -inf, i.e. inf + (-inf) = NaN by IEEE 754, then 1.0 elsewhere.
      Unchanged behavior: NaN here comes from the running total itself, not from the compensation
      term, so the guard does not (and must not) suppress it.
    """
    row = torch.ones(length, dtype=torch.float32)
    if pattern == "pos_inf":
        row[1] = float("inf")
    elif pattern == "neg_inf":
        row[1] = float("-inf")
    elif pattern == "overflow":
        row[0] = 2.0**127
        row[1] = 2.0**127
    elif pattern == "nan":
        row[1] = float("nan")
    elif pattern == "inf_at_end":
        row[-1] = float("inf")
    elif pattern == "pos_then_neg_inf":
        row[0] = float("inf")
        row[1] = float("-inf")
    else:
        raise ValueError(f"unknown pattern {pattern}")
    return row


def _assert_matches_torch_special(actual, expected):
    """Elementwise compare against a golden tensor that may contain NaN/+-inf.

    NaN positions must match exactly (NaN is "equal" to NaN here, unlike IEEE compare). Remaining
    elements must match in finiteness; where both are infinite the sign must match; where both are
    finite the values must match (exact, since every scenario here only ever adds 1.0 or 2**127,
    both exact in fp32).
    """
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} vs {expected.shape}"

    expected_nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), expected_nan), (
        f"NaN positions differ: expected {expected_nan.nonzero().flatten().tolist()}, "
        f"got {torch.isnan(actual).nonzero().flatten().tolist()}"
    )

    not_nan = ~expected_nan
    exp_f = expected[not_nan]
    act_f = actual[not_nan]

    assert torch.equal(torch.isinf(act_f), torch.isinf(exp_f)), "inf positions differ outside NaN region"
    inf_mask = torch.isinf(exp_f)
    if inf_mask.any():
        assert torch.equal(
            torch.sign(exp_f[inf_mask]), torch.sign(act_f[inf_mask])
        ), f"inf sign mismatch: expected {exp_f[inf_mask]}, got {act_f[inf_mask]}"

    finite_mask = ~inf_mask
    if finite_mask.any():
        assert torch.equal(exp_f[finite_mask], act_f[finite_mask]), (
            f"finite mismatch: expected {exp_f[finite_mask]}, got {act_f[finite_mask]}"
        )


@pytest.mark.parametrize("scan_length", [4, 8, 33])
@pytest.mark.parametrize("dim", [0, -2])
@pytest.mark.parametrize(
    "pattern",
    ["pos_inf", "neg_inf", "overflow", "nan", "inf_at_end", "pos_then_neg_inf"],
)
@pytest.mark.parametrize("disable_compensated_sum", [False, True])
def test_cumsum_fp32_special_values(scan_length, dim, pattern, disable_compensated_sum, device):
    """Regression test for #58986.

    The compensated (Kahan) fp32 accumulation path computed its compensation term as
    `(t - acc) - y` without checking whether the new running total `t` was finite. Once `t` went
    +-inf -- either from an infinite input or from finite-input overflow -- that subtraction hit
    `inf - inf` (NaN), and the NaN rode the compensation FIFO into every later scan element, even
    though the running total itself remained well-defined. torch never looks at a compensation
    term at all, so it keeps emitting +-inf; this test checks ttnn now matches it.

    Runs both the default (compensated) path and the deprecated `disable_compensated_sum=True`
    fallback -- the latter was never affected by this bug (its accumulation never computes a
    compensation term), so it also serves as a control. Scan lengths 4 and 8 exercise a single
    sub-tile scan; 33 crosses a tile boundary, which is where the comp FIFO carries a poisoned
    value from one tile's last element into the next tile's first.
    """
    along = _build_special_value_row(scan_length, pattern)
    size = [scan_length, 3] if dim == 0 else [2, scan_length, 5]
    shape_along = [1] * len(size)
    shape_along[dim] = scan_length
    torch_input = along.view(shape_along).expand(size).contiguous()

    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.cumsum(input_tensor, dim=dim, disable_compensated_sum=disable_compensated_sum)
    output = ttnn.to_torch(output_tensor)

    expected = torch.cumsum(torch_input, dim=dim)

    _assert_matches_torch_special(output, expected)


def test_cumsum_disable_compensated_sum(device):
    """`disable_compensated_sum=True` must actually fall back to the plain sequential sum.

    Uses the discriminating sequence from the accuracy test above (piecewise-constant, where
    consecutive rounding errors share a sign). The default compensated path stays within the
    8*eps*sum|x| bound; the plain path is expected well outside it (~216x on this sequence). If the
    flag did nothing, both would pass the bound and the escape hatch would be a silent no-op.
    """
    torch.manual_seed(29112024)
    n = 72192
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
        ([1, 151936], -1),  # long scan
        ([1, 72192, 9], 1),  # the long thin shape from #55542
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
