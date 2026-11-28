# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device

# std_hw, var_hw and normalize_hw across the float32 exponent range. The squared deviation
# (x - mean)^2 underflows to zero below |x - mean| = 1.0842e-19, and the sum of squares saturates or
# overflows far below the largest representable standard deviation. A composite that squares
# x - mean directly therefore returns 0, a saturated constant or inf for std_hw of an ordinary 1e-20
# or 1e20 spread, and a non-finite or wrong result for normalize_hw.
#
# Alternating columns of +v and -v give a mean of exactly zero and a population standard deviation
# of exactly v at every magnitude, so the reference is analytic: normalize_hw must return the +/-1
# pattern, std_hw must return v and var_hw must return v^2. 1e36 has a scale of 2^119, where the FLOAT32 reciprocal
# op is not exact in ttsim's Blackhole model, which is why 1 / scale is formed from the bits of the scale.
DEVIATIONS = (1e-25, 1e-20, 1e-19, 1e-10, 1.0, 1e10, 1e18, 1e20, 1e25, 1e36)
DEVIATION_IDS = ["1e-25", "1e-20", "1e-19", "1e-10", "1.0", "1e10", "1e18", "1e20", "1e25", "1e36"]

# var_hw can only be right where v^2 is itself a normal float32.
VAR_DEVIATIONS = (1e-18, 1e-10, 1.0, 1e10, 1e18, 1e19)
VAR_DEVIATION_IDS = ["1e-18", "1e-10", "1.0", "1e10", "1e18", "1e19"]

DTYPES = ((torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16))
DTYPE_IDS = ("float32", "bfloat16")

# The references are exact, so the error is measured in ULP of the result's dtype: 2 in float32 and 1 in bfloat16.
# std_hw and var_hw also carry the rounding of the float32 mean of the squares, which depends on the shape and on the
# order of the reduction: on Blackhole (ttsim) up to 2 ULP for std_hw and 3.3 for var_hw on the shapes below, and up
# to 6.2 on other shapes that were tried. So in float32 they get 8, as test_mean_nc does for a float32 mean.
ULP_THRESHOLD = {torch.float32: 2, torch.bfloat16: 1}
STATISTIC_ULP_THRESHOLD = {torch.float32: 8, torch.bfloat16: 1}

NORMALIZE_SHAPE = torch.Size([1, 1, 32, 32])
UNALIGNED_NORMALIZE_SHAPE = torch.Size([1, 1, 30, 40])

# std_hw and var_hw send planes of at most 64 tiles, and 10 or more planes, to ttnn.std / ttnn.var, which run as one
# program and do not rescale the deviation yet (#55159). The statistic tests use planes of 17 x 16 tiles, which take
# the composite: 544x512, and 520x512, which is not tile-aligned, one at a time or three in one tensor.
STATISTIC_SHAPE = torch.Size([1, 1, 544, 512])
UNALIGNED_STATISTIC_SHAPE = torch.Size([1, 1, 520, 512])
PLANES_STATISTIC_SHAPE = torch.Size([1, 3, 544, 512])

# A plane of 3.0 has no deviation and an exact mean, so std_hw and var_hw must give exactly 0.
CONSTANT = 3.0


def _planes(deviations, torch_dtype, shape):
    """One plane per deviation, in order over N and C: columns of +v and -v, or CONSTANT where v is None."""
    planes = shape[0] * shape[1]
    assert len(deviations) == planes
    sign = torch.ones(shape)
    sign[..., 1::2] = -1.0
    data = sign.clone()
    for p, v in enumerate(deviations):
        plane = data.view(planes, shape[2], shape[3])[p]
        if v is None:
            plane.fill_(CONSTANT)
        else:
            plane.mul_(v)
    return sign, data.to(torch_dtype)


def _spread(v, torch_dtype):
    """v rounded the way _planes rounds it, which is the population std of that plane."""
    return 0.0 if v is None else float(torch.tensor(v, dtype=torch.float32).to(torch_dtype))


def _expected(values, shape):
    """[N, C, 1, 1] holding one value per plane, in float64, so the error is measured against the exact value."""
    return torch.tensor(values, dtype=torch.float64).reshape(shape[0], shape[1], 1, 1)


def _run(op, data, ttnn_dtype, device):
    tt_in = ttnn.from_torch(data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(op(tt_in))


def _check_normalize(deviations, torch_dtype, ttnn_dtype, shape, device):
    sign, data = _planes(deviations, torch_dtype, shape)
    got = _run(ttnn.normalize_hw, data, ttnn_dtype, device)
    assert_with_ulp(expected_result=sign.double(), actual_result=got, ulp_threshold=ULP_THRESHOLD[torch_dtype])


def _check_std(deviations, torch_dtype, ttnn_dtype, shape, device):
    _, data = _planes(deviations, torch_dtype, shape)
    got = _run(ttnn.std_hw, data, ttnn_dtype, device)
    expected = _expected([_spread(v, torch_dtype) for v in deviations], shape)
    assert_with_ulp(expected_result=expected, actual_result=got, ulp_threshold=STATISTIC_ULP_THRESHOLD[torch_dtype])


def _check_var(deviations, torch_dtype, ttnn_dtype, shape, device):
    _, data = _planes(deviations, torch_dtype, shape)
    got = _run(ttnn.var_hw, data, ttnn_dtype, device)
    expected = _expected([_spread(v, torch_dtype) ** 2 for v in deviations], shape)
    assert_with_ulp(expected_result=expected, actual_result=got, ulp_threshold=STATISTIC_ULP_THRESHOLD[torch_dtype])


def _runs_one_program(op, tt_x):
    """True when op runs ttnn.std / ttnn.var as one program instead of the composite, as in test_var_std_hw.py."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        op(tt_x)
    finally:
        captured_graph = ttnn.graph.end_graph_capture()
    return "WelfordReduceDeviceOperation" in ttnn.graph.extract_calltrace(captured_graph)


# The std_hw / var_hw tests below only test the rescale if their shapes take the composite.
@pytest.mark.parametrize(
    "shape",
    [STATISTIC_SHAPE, UNALIGNED_STATISTIC_SHAPE, PLANES_STATISTIC_SHAPE],
    ids=lambda shape: "x".join(map(str, shape)),
)
def test_statistic_shapes_take_the_composite(shape, device):
    tt_x = ttnn.from_torch(torch.ones(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for op in (ttnn.std_hw, ttnn.var_hw):
        assert not _runs_one_program(op, tt_x), f"{op} runs as one program on {list(shape)}, which does not rescale"


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("deviation", DEVIATIONS, ids=DEVIATION_IDS)
def test_normalize_hw_is_scale_invariant(deviation, torch_dtype, ttnn_dtype, device):
    _check_normalize([deviation], torch_dtype, ttnn_dtype, NORMALIZE_SHAPE, device)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("deviation", DEVIATIONS, ids=DEVIATION_IDS)
def test_std_hw_matches_the_constructed_spread(deviation, torch_dtype, ttnn_dtype, device):
    _check_std([deviation], torch_dtype, ttnn_dtype, STATISTIC_SHAPE, device)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("deviation", VAR_DEVIATIONS, ids=VAR_DEVIATION_IDS)
def test_var_hw_matches_the_constructed_spread(deviation, torch_dtype, ttnn_dtype, device):
    _check_var([deviation], torch_dtype, ttnn_dtype, STATISTIC_SHAPE, device)


# Each plane gets its own scale, so planes 50 decades apart in one tensor all come out right, and a constant plane
# next to them gives 0. Three planes stay below the ten from which std_hw / var_hw run as one program.
@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
def test_normalize_hw_planes_of_different_magnitude(torch_dtype, ttnn_dtype, device):
    _check_normalize([1e-25, 1.0, 1e18, 1e25], torch_dtype, ttnn_dtype, torch.Size([2, 2, 32, 32]), device)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
def test_std_hw_planes_of_different_magnitude(torch_dtype, ttnn_dtype, device):
    _check_std([1e-25, 1e25, None], torch_dtype, ttnn_dtype, PLANES_STATISTIC_SHAPE, device)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
def test_var_hw_planes_of_different_magnitude(torch_dtype, ttnn_dtype, device):
    _check_var([1e-18, 1e19, None], torch_dtype, ttnn_dtype, PLANES_STATISTIC_SHAPE, device)


# Not tile-aligned: the scale's max skips the tile padding and both means divide by the logical H x W.
@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("deviation", (1e-20, 1.0, 1e20), ids=["1e-20", "1.0", "1e20"])
def test_normalize_std_hw_unaligned_plane(deviation, torch_dtype, ttnn_dtype, device):
    _check_normalize([deviation], torch_dtype, ttnn_dtype, UNALIGNED_NORMALIZE_SHAPE, device)
    _check_std([deviation], torch_dtype, ttnn_dtype, UNALIGNED_STATISTIC_SHAPE, device)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("deviation", (1e-18, 1.0, 1e19), ids=["1e-18", "1.0", "1e19"])
def test_var_hw_unaligned_plane(deviation, torch_dtype, ttnn_dtype, device):
    _check_var([deviation], torch_dtype, ttnn_dtype, UNALIGNED_STATISTIC_SHAPE, device)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
def test_std_var_hw_constant_plane(torch_dtype, ttnn_dtype, device):
    _check_std([None], torch_dtype, ttnn_dtype, STATISTIC_SHAPE, device)
    _check_var([None], torch_dtype, ttnn_dtype, STATISTIC_SHAPE, device)


# One outlier sets the scale. A single 1e20 in a 544x512 plane of zeros has a variance of 3.6e34, but its scale is
# 2^66, whose square is above FLT_MAX.
@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
def test_var_std_hw_single_outlier(torch_dtype, ttnn_dtype, device):
    data = torch.zeros(STATISTIC_SHAPE)
    data[0, 0, 0, 0] = 1e20
    data = data.to(torch_dtype)
    variance = data.double().var(dim=(-2, -1), keepdim=True, correction=0)

    for op, expected, threshold in (
        (ttnn.var_hw, variance, STATISTIC_ULP_THRESHOLD),
        (ttnn.std_hw, variance.sqrt(), STATISTIC_ULP_THRESHOLD),
    ):
        got = _run(op, data, ttnn_dtype, device)
        assert_with_ulp(expected_result=expected, actual_result=got, ulp_threshold=threshold[torch_dtype])


# From max|d| = 2^127 (1.7e38) up, the scale is capped at 2^126, so that 1 / scale is still a normal float. A single
# 2e38 in a 544x512 plane of zeros has a finite mean, 7.2e32, and a standard deviation of 3.8e35. Its variance is above
# FLT_MAX, so var_hw must give inf.
@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
def test_var_std_hw_outlier_at_the_top_of_the_range(torch_dtype, ttnn_dtype, device):
    data = torch.zeros(STATISTIC_SHAPE)
    data[0, 0, 0, 0] = 2e38
    data = data.to(torch_dtype)
    std = data.double().std(dim=(-2, -1), keepdim=True, unbiased=False)

    got = _run(ttnn.std_hw, data, ttnn_dtype, device)
    assert_with_ulp(expected_result=std, actual_result=got, ulp_threshold=STATISTIC_ULP_THRESHOLD[torch_dtype])
    got = _run(ttnn.var_hw, data, ttnn_dtype, device)
    assert torch.isposinf(got).all(), f"var_hw of a variance above FLT_MAX gives {got.flatten().tolist()}, not inf"


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
def test_normalize_hw_outlier_at_the_top_of_the_range(torch_dtype, ttnn_dtype, device):
    data = torch.zeros(NORMALIZE_SHAPE)
    data[0, 0, 0, 0] = 2e38
    data = data.to(torch_dtype)
    x = data.double()
    expected = (x - x.mean(dim=(-2, -1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True, unbiased=False)

    got = _run(ttnn.normalize_hw, data, ttnn_dtype, device)
    assert_with_ulp(expected_result=expected, actual_result=got, ulp_threshold=ULP_THRESHOLD[torch_dtype])


# bfloat8_b shares one exponent across each 16 values of a tile row. The per-plane factors live on [N, C, 1, 1]
# tiles next to their padding, so they are formed in bfloat16: in bfloat8_b the padding flushed them to 0. The
# tolerance is the one test_normalize_global.py uses for bfloat8_b.
BFLOAT8_B_ATOL = 6e-2


def test_normalize_hw_bfloat8_b(device):
    torch.manual_seed(0)
    tt_in = ttnn.from_torch(torch.randn([1, 2, 64, 64]), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    x = ttnn.to_torch(tt_in).double()
    expected = (x - x.mean(dim=(2, 3), keepdim=True)) / x.std(dim=(2, 3), keepdim=True, unbiased=False)

    got = ttnn.to_torch(ttnn.normalize_hw(tt_in)).double()

    error = (got - expected).abs().max().item()
    assert error <= BFLOAT8_B_ATOL, f"normalize_hw on bfloat8_b is {error:g} off, more than {BFLOAT8_B_ATOL:g}"


# std_hw and var_hw on bfloat8_b take the composite as well: the scale is formed in bfloat16, the multiply that squares
# d / scale writes over the bfloat8_b |d| in place with that bfloat16 factor broadcast per plane, and the result keeps
# the input's dtype.
@pytest.mark.parametrize(
    "shape",
    [STATISTIC_SHAPE, UNALIGNED_STATISTIC_SHAPE, PLANES_STATISTIC_SHAPE],
    ids=lambda shape: "x".join(map(str, shape)),
)
def test_std_var_hw_bfloat8_b(shape, device):
    torch.manual_seed(0)
    tt_in = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    x = ttnn.to_torch(tt_in).double()
    variance = x.var(dim=(-2, -1), keepdim=True, correction=0)

    for op, expected in ((ttnn.var_hw, variance), (ttnn.std_hw, variance.sqrt())):
        out = op(tt_in)
        assert out.dtype == ttnn.bfloat8_b, f"{op} on bfloat8_b returns {out.dtype}"
        error = (ttnn.to_torch(out).double() - expected).abs().max().item()
        assert error <= BFLOAT8_B_ATOL, f"{op} on bfloat8_b is {error:g} off, more than {BFLOAT8_B_ATOL:g}"
