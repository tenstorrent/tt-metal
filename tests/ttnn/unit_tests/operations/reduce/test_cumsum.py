# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

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

    # Generate integer input on [-2; 2];
    # by generating around 0, this avoids FP-related issues when adding large sums with small inputs
    # which are not handled yet
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

    # Generate integer input on [-2; 2];
    # by generating around 0, this avoids FP-related issues when adding large sums with small inputs
    # which are not handled yet
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
@pytest.mark.parametrize("signal", ["iid", "piecewise_constant"])
def test_cumsum_fp32_long_scan_accuracy(size, dim, signal, device):
    """fp32 accuracy over a LONG scan with REAL-valued inputs.

    Every other test in this file draws integers on [-2, 2] -- as its own comment says, to avoid
    "FP-related issues when adding large sums with small inputs which are not handled yet". Integer
    partial sums are exact in fp32, so those tests cannot observe accumulation error at all. This
    one can.

    The metric is the compensated-summation error bound, per output element:
        |out_i - exact_i|  <=  K * eps32 * sum_{j<=i} |x_j|
    Kahan summation guarantees this with K ~ 2 independent of scan length. Plain sequential fp32
    accumulation has error growing as ~T^1.5 when consecutive rounding errors share a sign: on the
    piecewise-constant signal below it measures K ~ 216 (#55542: 1825 ULP, sign flips near zero
    crossings). On the iid signal the errors cancel and the plain path sits at K ~ 0.5, so that
    case discriminates nothing and is here only to show the change does not regress it. K = 8:
    4x margin over the guarantee, 25x below the plain path on the discriminating signal.

    Why not ULP of the reference: an iid signal's running sum is a random walk that crosses zero,
    where one ULP of the reference is ~1e-12 and any fp32 method reads as "thousands of ULP" while
    its absolute error is ~1e-5. (torch.cumsum on CPU accumulates fp32 in double, which is why it
    reads 0.5 ULP everywhere -- it is a reference, not an fp32 peer.) The piecewise-constant
    signal is the hard case for sequential summation: autocorrelated inputs make consecutive
    rounding errors share a sign, so they add coherently instead of partly cancelling.
    """
    torch.manual_seed(29112024)
    n = size[dim]
    if signal == "iid":
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
        f"max error is {worst:.1f}x the compensated-summation bound 8*eps*sum|x| (signal={signal}); "
        "plain sequential fp32 sits near 216x on the piecewise-constant signal"
    )


def test_cumsum_disable_compensated_sum(device):
    """`disable_compensated_sum=True` must actually fall back to the plain sequential sum.

    Uses the discriminating signal from the accuracy test above (piecewise-constant, where
    consecutive rounding errors share a sign). The default compensated path stays within the
    8*eps*sum|x| bound; the plain path is expected well outside it (~216x on this signal). If the
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
    # Disabling it must degrade accuracy on the discriminating signal -- otherwise the flag is inert.
    assert (plain_err / bound).max().item() > 4.0, (
        "disable_compensated_sum=True did not change the result: the plain path stayed within the "
        "compensated bound, so the flag is not reaching the accumulation kernel"
    )


# --------------------------------------------------------------------------------------------------
# #58986: non-finite running totals on the compensated fp32 path.
#
# The Kahan term c = (t - acc) - y becomes NaN (inf - inf) on the step the running total turns
# infinite, or +-inf on finite overflow; fed back through y = in - c it used to turn every later
# element into NaN. These tests pin torch's IEEE behaviour for every special-value pattern, on both
# the default compensated path and the deprecated plain path.
# --------------------------------------------------------------------------------------------------

NON_FINITE_KINDS = [
    "pos_inf",  # [1, inf, 1, ...]      -> [1, inf, inf, ...]
    "neg_inf",  # [1, -inf, 1, ...]     -> [1, -inf, -inf, ...]
    "pos_overflow",  # [3e38, 3e38, 1, ...] -> [3e38, inf, inf, ...]
    "neg_overflow",  # [-3e38, -3e38, 1, ...] -> [-3e38, -inf, -inf, ...]
    "nan_input",  # [1, nan, 1, ...]     -> NaN from index 1 on (unchanged behaviour)
    "inf_last",  # [1, ..., 1, inf]     -> inf only at the end (unchanged behaviour)
    "pos_then_neg_inf",  # [1, inf, -inf, 1, ...] -> NaN from index 2 on (unchanged behaviour)
]


def _non_finite_sequence(kind, n, dtype):
    seq = torch.ones(n, dtype=dtype)
    if kind == "pos_inf":
        seq[1] = float("inf")
    elif kind == "neg_inf":
        seq[1] = float("-inf")
    elif kind == "pos_overflow":
        seq[0] = seq[1] = 3.0e38
    elif kind == "neg_overflow":
        seq[0] = seq[1] = -3.0e38
    elif kind == "nan_input":
        seq[1] = float("nan")
    elif kind == "inf_last":
        seq[-1] = float("inf")
    elif kind == "pos_then_neg_inf":
        seq[1] = float("inf")
        seq[2] = float("-inf")
    else:
        raise ValueError(kind)
    return seq


def _interleaved_scan_input(seq, size, dim, base):
    """Every even lane along `dim` carries `seq`; odd lanes keep the finite `base` values.

    Special and finite lanes share tiles, so a guard that leaked across lanes (e.g. zeroing the
    compensation of a finite neighbour) would show up as an accuracy failure in the finite lanes.
    """
    x = base.movedim(dim, -1).contiguous()
    x.view(-1, x.shape[-1])[0::2] = seq
    return x.movedim(-1, dim).contiguous()


def _assert_matches_torch_with_special_values(expected, actual):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected)), "NaN positions differ from torch"
    expected_inf = torch.isinf(expected)
    assert torch.equal(torch.isinf(actual), expected_inf), "inf positions differ from torch"
    assert torch.equal(actual[expected_inf], expected[expected_inf]), "inf signs differ from torch"
    finite = torch.isfinite(expected)
    assert_cumsum_quality(expected[finite], actual[finite])


def _scan_size(n, dim):
    # dim 0 scans across tiles; -2 scans rows inside tiles; -1 scans columns. On a 3-D tensor 0 and
    # -2 are distinct (on 2-D they would alias). n = 33 crosses a tile boundary along the scan.
    return {0: [n, 32, 32], -2: [2, n, 32], -1: [2, 32, n]}[dim]


@pytest.mark.parametrize("disable_compensated_sum", [False, True])
@pytest.mark.parametrize("kind", NON_FINITE_KINDS)
@pytest.mark.parametrize("dim", [0, -2, -1])
@pytest.mark.parametrize("n", [4, 8, 33])
def test_cumsum_fp32_non_finite_matches_torch(n, dim, kind, disable_compensated_sum, device):
    """#58986: once the running total is +-inf, fp32 cumsum must stay +-inf (not NaN) like torch.

    Covers +-inf inputs and +-finite overflow, plus the cases whose behaviour must not change (NaN
    input, inf at the last element, +inf then -inf), on the default compensated path and on
    disable_compensated_sum=True.
    """
    torch.manual_seed(58986)
    size = _scan_size(n, dim)
    seq = _non_finite_sequence(kind, n, torch.float32)
    torch_input = _interleaved_scan_input(seq, size, dim, torch.randn(size, dtype=torch.float32))

    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    output = ttnn.to_torch(ttnn.cumsum(input_tensor, dim=dim, disable_compensated_sum=disable_compensated_sum))

    expected = torch.cumsum(torch_input, dim=dim)
    _assert_matches_torch_with_special_values(expected, output)


@pytest.mark.parametrize("kind", ["pos_inf", "neg_inf", "nan_input", "pos_then_neg_inf"])
@pytest.mark.parametrize("dim", [0, -2])
def test_cumsum_bf16_non_finite_unchanged(kind, dim, device):
    """bf16 never takes the compensated path; its special-value behaviour must be unaffected."""
    torch.manual_seed(58986)
    n = 33
    size = _scan_size(n, dim)
    seq = _non_finite_sequence(kind, n, torch.bfloat16)
    # Integer-valued finite lanes keep bf16 partial sums exact, as in test_cumsum above.
    base = torch.randint(-2, 3, size=size, dtype=torch.bfloat16)
    torch_input = _interleaved_scan_input(seq, size, dim, base)

    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)
    output = ttnn.to_torch(ttnn.cumsum(input_tensor, dim=dim), dtype=torch.bfloat16)

    expected = torch.cumsum(torch_input, dim=dim, dtype=torch.bfloat16)
    _assert_matches_torch_with_special_values(expected, output)


@pytest.mark.skipif(
    os.environ.get("CUMSUM_PERF_SWEEP") != "1",
    reason="perf characterization for #58986; run under the device profiler with CUMSUM_PERF_SWEEP=1",
)
@pytest.mark.parametrize("disable_compensated_sum", [False, True])
@pytest.mark.parametrize(
    "size, dim",
    [
        ([1, 32, 32], -2),  # one tile along the scan
        ([1, 1024, 32], -2),  # 32 tiles
        ([1, 32768, 32], -2),  # 1024 tiles
        ([32, 32, 32], 0),
        ([1024, 32, 32], 0),
        ([1, 32, 32768], -1),
    ],
)
def test_cumsum_fp32_compensated_perf_sweep(size, dim, disable_compensated_sum, device):
    """Device kernel time of fp32 cumsum, for #58986's performance characterization.

    Not a correctness test. Run with:
        CUMSUM_PERF_SWEEP=1 python -m tracy -r -m pytest \
            tests/ttnn/unit_tests/operations/reduce/test_cumsum.py -k compensated_perf_sweep
    on this branch and on its base commit, then compare DEVICE KERNEL DURATION in the ops report.
    disable_compensated_sum=True is the uncompensated baseline.
    """
    torch.manual_seed(58986)
    torch_input = torch.randn(size, dtype=torch.float32)
    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)
    for _ in range(3):  # first run includes kernel compile; later runs are the steady state
        output = ttnn.cumsum(input_tensor, dim=dim, disable_compensated_sum=disable_compensated_sum)
    ttnn.synchronize_device(device)
    assert output.shape == input_tensor.shape
