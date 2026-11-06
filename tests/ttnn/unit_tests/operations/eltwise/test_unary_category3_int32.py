# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_int32_values,
    to_int32_tt_tensor,
    wrap_int32,
    INT32_MAX,
    INT32_MIN,
)

pytestmark = pytest.mark.use_module_device


"""
Int32 accuracy sweep of the unary ops with a dim, multiple or special
parameters that accept ttnn.int32 input: the int32 subset of categories 5
and 6 of the float32 and bfloat16 files.

Inputs come from generate_int32_values: every integer in [-2^14, 2^14),
±2^k and ±(2^k ± 1) for every k (including 2^24 + 1, INT32_MIN and
INT32_MAX), and uniformly random int32 values, 65,536 in total.

Int32 category 3: ops with a dim, multiple or special parameters
(float32 / bfloat16 categories 5, dim parameter, and 6, multiple or special parameters)

| #  | Op               | Parameters                        | Output                      |
|----|------------------|-----------------------------------|-----------------------------|
| 1  | ttnn.reglu       | dim                               | int32, A * relu(B)          |
| 2  | ttnn.clamp       | min, max (int, float or Tensor)   | int32; float32 for float    |
|    |                  |                                   | bounds                      |
| 3  | ttnn.clip        | min, max (float or Tensor)        | float32 for scalar bounds;  |
|    |                  |                                   | int32 for Tensor bounds     |
| 4  | ttnn.tril        | diagonal                          | int32                       |
| 5  | ttnn.triu        | diagonal                          | int32                       |
| 6  | ttnn.unary_chain | ops_chain of int32-capable ops    | int32                       |
| 7  | ttnn.bitcast     | dtype (uint32, float32)           | reinterpreted bits          |

glu, geglu, swiglu, hardtanh, threshold, selu, softplus, xielu, tanh,
sigmoid, sigmoid_accurate, round, polygamma, logit and rdiv reject int32
input in UnaryDeviceOperation validation.

With float bounds, clamp and clip typecast the int32 input to float32, the
same promotion torch applies, so the golden is torch.clamp with float
bounds and |x| > 2^24 rounds in both. ttnn.clip has no int overload: int
bounds are taken as floats, so it cannot clip in int32.

Accuracy criteria

| Op            | Parameters                     | Check                       |
|---------------|--------------------------------|-----------------------------|
| reglu         | dim = -1, 3                    | exact (A * relu(B) wraps)   |
| clamp         | int min and max                | exact                       |
|               | int min or max only            | strict xfail                |
| clamp, clip   | float min and max              | float32, bit-equal to torch |
|               | Tensor min and max             | exact                       |
| clip          | int min and max                | strict xfail                |
| tril, triu    | diagonal 0, ±1, 5, -100, 300   | exact                       |
| unary_chain   | int32-capable ops              | exact                       |
| bitcast       | to uint32                      | bit-exact                   |
|               | to float32                     | strict xfail                |
"""


# ─────────────────────────────────────────────────────────────────────────────
# reglu — exact
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dim", [-1, 3])
def test_reglu(device, dim):
    """reglu = A * relu(B), with A and B the halves of the last dim.

    | Input range | Check | Reason                                            |
    |-------------|-------|---------------------------------------------------|
    | all         | exact | the int32 multiply wraps mod 2^32, as in torch    |

    The halves of a 256-wide row are unrelated sweep values, so the products
    cover the whole int32 range and overflow.
    """
    input_tensor = generate_int32_values().reshape(1, 1, 256, 256)

    tt_in = to_int32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.reglu)
    golden = golden_function(input_tensor, dim)

    result = ttnn.to_torch(ttnn.reglu(tt_in, dim))
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(golden, result)


# ─────────────────────────────────────────────────────────────────────────────
# clamp, clip — scalar and Tensor bounds
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (-5, 100),
        (0, 6),
        (-(2**24 + 1), 2**24 + 1),
        (INT32_MIN, INT32_MAX),
        (100, -5),
        (-1000, -1000),
    ],
    ids=["both", "relu6_bounds", "past_float32_integers", "int32_limits", "degenerate", "single_value"],
)
def test_clamp_int_bounds(device, min_val, max_val):
    """clamp(x, min_val, max_val) with int bounds.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | all         | exact | every output is x or a bound; the bounds reach  |
    |             |       | the kernel as int32                             |

    degenerate (min > max) returns max everywhere, as in torch.
    """
    input_tensor = generate_int32_values()

    tt_in = to_int32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.clamp)
    golden = golden_function(input_tensor, min_val, max_val)

    result = ttnn.to_torch(ttnn.clamp(tt_in, min_val, max_val))
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(golden, result)


@pytest.mark.xfail(
    strict=True,
    reason="#51498: int32 clamp with one int bound defaults the other to ±16775716 instead of the int32 limit",
)
@pytest.mark.parametrize("min_val, max_val", [(5, None), (None, 5)], ids=["min_only", "max_only"])
def test_clamp_one_int_bound(device, min_val, max_val):
    """clamp(x, min=5) and clamp(x, max=5) on int32.

    | Input range         | Check | Reason                                     |
    |---------------------|-------|--------------------------------------------|
    | all                 | exact | the missing bound must not clamp           |
    | |x| > 16775716      | fails | the device clamps to ±16775716, the        |
    |                     |       | default composite clamp uses for the       |
    |                     |       | missing int bound                          |
    """
    input_tensor = generate_int32_values()

    tt_in = to_int32_tt_tensor(input_tensor, device)
    golden = torch.clamp(input_tensor, min=min_val, max=max_val)

    result = ttnn.to_torch(ttnn.clamp(tt_in, min=min_val, max=max_val))
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(golden, result)


@pytest.mark.parametrize(
    "min_val, max_val",
    [(-5.0, 100.0), (-1e9, 1e9), (float(INT32_MIN), float(INT32_MAX))],
    ids=["small", "past_float32_integers", "int32_limits"],
)
@pytest.mark.parametrize("ttnn_op", [ttnn.clamp, ttnn.clip])
def test_clamp_clip_float_bounds(device, ttnn_op, min_val, max_val):
    """clamp/clip(x, min_val, max_val) on int32 with float bounds.

    | Input range  | Check                   | Reason                              |
    |--------------|-------------------------|-------------------------------------|
    | all          | float32, bit-equal to   | the input is typecast to float32,   |
    |              | torch.clamp with float  | the promotion torch applies, so     |
    |              | bounds                  | |x| > 2^24 rounds on both sides     |
    """
    input_tensor = generate_int32_values()

    tt_in = to_int32_tt_tensor(input_tensor, device)
    golden = torch.clamp(input_tensor, min_val, max_val)

    result = ttnn.to_torch(ttnn_op(tt_in, min_val, max_val))
    assert result.dtype == torch.float32, f"expected a float32 result, got {result.dtype}"
    assert torch.equal(result.view(torch.int32), golden.view(torch.int32)), (
        f"{ttnn_op.__name__}({min_val}, {max_val}) diverged from torch's float32 promotion for "
        f"{int((result != golden).sum())} of {result.numel()} elements"
    )


@pytest.mark.xfail(
    strict=True,
    reason="#64582: ttnn.clip has only float bounds, so int32 input is clipped in float32 instead of int32",
)
def test_clip_int_bounds(device):
    """clip(x, INT32_MIN, INT32_MAX) on int32 must return x unchanged as int32.

    | Input range | Check | Reason                                             |
    |-------------|-------|----------------------------------------------------|
    | all         | exact | torch.clip keeps int32 for int bounds; ttnn.clip   |
    |             |       | returns float32 and rounds |x| > 2^24              |
    """
    input_tensor = generate_int32_values()

    tt_in = to_int32_tt_tensor(input_tensor, device)
    golden = torch.clip(input_tensor, INT32_MIN, INT32_MAX)

    result = ttnn.to_torch(ttnn.clip(tt_in, INT32_MIN, INT32_MAX))
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(golden, result)


@pytest.mark.parametrize("ttnn_op", [ttnn.clamp, ttnn.clip])
def test_clamp_clip_tensor_bounds(device, ttnn_op):
    """clamp/clip(x, min, max) with per-element int32 Tensor bounds.

    | Input range | Check | Reason                                |
    |-------------|-------|---------------------------------------|
    | all         | exact | every output is x, min_i or max_i     |

    The bounds are the input rolled by ±64 columns, so each element gets
    bounds unrelated to its own value. More than 10% of the elements must
    actually be clamped.
    """
    input_tensor = generate_int32_values()
    min_tensor = -input_tensor.roll(64, dims=1).abs()
    max_tensor = input_tensor.roll(-64, dims=1).abs()

    tt_in = to_int32_tt_tensor(input_tensor, device)
    tt_min = to_int32_tt_tensor(min_tensor, device)
    tt_max = to_int32_tt_tensor(max_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, min_tensor, max_tensor)

    clamped = int((golden != input_tensor).sum().item())
    assert clamped > golden.numel() // 10, f"only {clamped} of {golden.numel()} elements are clamped"

    result = ttnn.to_torch(ttnn_op(tt_in, tt_min, tt_max))
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(golden, result)


# ─────────────────────────────────────────────────────────────────────────────
# tril, triu — exact
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("diagonal", [0, 1, -1, 5, -100, 300])
@pytest.mark.parametrize("ttnn_op", [ttnn.tril, ttnn.triu])
def test_tril_triu(device, ttnn_op, diagonal):
    """tril/triu(x, diagonal) on a 256 x 256 int32 matrix.

    | Input range | Check | Reason                                           |
    |-------------|-------|--------------------------------------------------|
    | all         | exact | every output is x or 0; diagonal 300 is past the |
    |             |       | last column, -100 is inside the matrix           |
    """
    input_tensor = generate_int32_values()

    tt_in = to_int32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, diagonal=diagonal)

    result = ttnn.to_torch(ttnn_op(tt_in, diagonal=diagonal))
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(golden, result)


# ─────────────────────────────────────────────────────────────────────────────
# unary_chain — exact
# ─────────────────────────────────────────────────────────────────────────────

U = ttnn.UnaryWithParam
T = ttnn.UnaryOpType

CHAINS = {
    "relu_neg": ([U(T.RELU), U(T.NEG)], lambda x: wrap_int32(-x.to(torch.int64).clamp(min=0))),
    "neg_relu": ([U(T.NEG), U(T.RELU)], lambda x: wrap_int32(-x.to(torch.int64)).clamp(min=0)),
    "square_relu6": ([U(T.SQUARE), U(T.RELU6)], lambda x: wrap_int32(x.to(torch.int64) ** 2).clamp(0, 6)),
    "bitwise_not_logical_not": (
        [U(T.BITWISE_NOT), U(T.LOGICAL_NOT_UNARY)],
        lambda x: (torch.bitwise_not(x) == 0).to(torch.int32),
    ),
    "relu_max_relu_min": (
        [U(T.RELU_MAX, 100), U(T.RELU_MIN, 5)],
        lambda x: x.clamp(0, 100).clamp(min=5),
    ),
}


@pytest.mark.parametrize("chain_name", list(CHAINS))
def test_unary_chain(device, chain_name):
    """unary_chain of int32-capable ops, fused into one kernel.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | all         | exact | each step is exact in int32; square wraps       |
    |             |       | mod 2^32 before relu6                           |
    """
    ops_chain, reference = CHAINS[chain_name]
    input_tensor = generate_int32_values()

    tt_in = to_int32_tt_tensor(input_tensor, device)
    golden = reference(input_tensor)

    result = ttnn.to_torch(ttnn.unary_chain(tt_in, ops_chain))
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(golden, result)


# ─────────────────────────────────────────────────────────────────────────────
# bitcast — bit-exact
# ─────────────────────────────────────────────────────────────────────────────


def test_bitcast_to_uint32(device):
    """bitcast(x, uint32) reinterprets the int32 bits.

    | Input range | Check     | Reason                                      |
    |-------------|-----------|---------------------------------------------|
    | all         | bit-exact | negative values become 2^32 + x, unchanged  |
    |             |           | bits                                        |
    """
    input_tensor = generate_int32_values()

    tt_in = to_int32_tt_tensor(input_tensor, device)
    result_tt = ttnn.bitcast(tt_in, ttnn.uint32)
    assert result_tt.dtype == ttnn.uint32, f"expected a uint32 result, got {result_tt.dtype}"

    result = ttnn.to_torch(result_tt)
    assert_equal(input_tensor, result.view(torch.int32))


@pytest.mark.xfail(
    strict=True,
    reason="#61975: bitcast int32 -> float32 does not keep the bits: float32 subnormal patterns become 0, NaN "
    "patterns become inf and normal patterns keep only the top 10 mantissa bits",
)
def test_bitcast_to_float32(device):
    """bitcast(x, float32) must keep all 32 bits.

    | Input range | Check     | Reason                                         |
    |-------------|-----------|------------------------------------------------|
    | all         | bit-exact | a bitcast changes the type, not the bits;      |
    |             |           | e.g. 0x12345678 comes back as 0x12344000       |
    """
    input_tensor = generate_int32_values()

    tt_in = to_int32_tt_tensor(input_tensor, device)
    result_tt = ttnn.bitcast(tt_in, ttnn.float32)
    assert result_tt.dtype == ttnn.float32, f"expected a float32 result, got {result_tt.dtype}"

    result = ttnn.to_torch(result_tt)
    assert_equal(input_tensor, result.view(torch.int32))
