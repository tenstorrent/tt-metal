# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_binary_grid,
    pairwise_from_values,
    pairwise_inputs,
    run_binary,
    to_float32_tt_tensor,
)

pytestmark = pytest.mark.use_module_device

"""
Category 2: binary comparison + min/max

| #  | Op            | Computes                | Check (float32)                          |
|----|---------------|-------------------------|------------------------------------------|
| 1  | ttnn.eq       | a == b                  | exact                                    |
| 2  | ttnn.ne       | a != b                  | exact                                    |
| 3  | ttnn.lt       | a < b                   | exact                                    |
| 4  | ttnn.le       | a <= b                  | exact                                    |
| 5  | ttnn.gt       | a > b                   | exact                                    |
| 6  | ttnn.ge       | a >= b                  | exact                                    |
| 7  | ttnn.isclose  | abs(a - b) <= tolerance | exact after the dest FTZ exception       |
| 8  | ttnn.minimum  | elementwise min         | exact; NaN ordered by its sign bit       |
| 9  | ttnn.maximum  | elementwise max         | exact; NaN ordered by its sign bit       |

The relational ops are covered tensor-tensor and tensor-scalar; the inplace
variants are covered tensor-tensor only.
"""


RELATIONAL_OPS = [ttnn.eq, ttnn.ne, ttnn.lt, ttnn.le, ttnn.gt, ttnn.ge]
RELATIONAL_INPLACE_OPS = [ttnn.eq_, ttnn.ne_, ttnn.lt_, ttnn.le_, ttnn.gt_, ttnn.ge_]


def _normal_values_with_normal_neighbors():
    """Finite grid values whose float32 neighbors are also normal.

    The neighbor of ±2^-126 toward zero is subnormal, and the device flushes
    subnormal inputs to zero, so those two values are left out.
    """
    values = generate_bfloat16_binary_grid(dtype=torch.float32)
    return values[torch.isfinite(values) & (values.abs() > 2.0**-126)]


@pytest.mark.parametrize("ttnn_op", RELATIONAL_OPS + RELATIONAL_INPLACE_OPS)
def test_relational_ops(device, ttnn_op):
    """Pairwise coverage of relational ops over the stratified grid at float32.

    | Case              | float32 behavior                              | Check |
    |-------------------|-----------------------------------------------|-------|
    | finite pairs      | SFPU compare matches IEEE                     | exact |
    | ±0                | compare equal                                 | exact |
    | ±inf              | matching-sign infs compare equal              | exact |
    | NaN on either side| every compare false except ne                 | exact |

    Device returns 0/1 in float32; golden is bool.
    """
    input_a, input_b = pairwise_inputs(include_spl_values=True, dtype=torch.float32)
    golden, result = run_binary(device, ttnn_op, input_a, input_b, input_dtype=ttnn.float32)
    assert_equal(golden.float(), result.float())


@pytest.mark.parametrize("ttnn_op", RELATIONAL_OPS + RELATIONAL_INPLACE_OPS)
def test_relational_ops_fp32_neighbors(device, ttnn_op):
    """Each finite grid value against itself and its float32 neighbors.

    The grid holds only bfloat16 values, so pairs from it never differ in the
    low 16 mantissa bits. Here b is a, nextafter(a, +inf) or nextafter(a, -inf),
    1 float32 ULP apart. A compare done at bfloat16 precision would call the
    neighbors equal.
    """
    values = _normal_values_with_normal_neighbors()
    up = torch.nextafter(values, torch.tensor(float("inf")))
    down = torch.nextafter(values, torch.tensor(float("-inf")))
    input_a = torch.stack([values, values, values])
    input_b = torch.stack([values, up, down])

    golden, result = run_binary(device, ttnn_op, input_a, input_b, input_dtype=ttnn.float32)
    assert_equal(golden.float(), result.float())


# One scalar per ordering / special-value class, plus scalars that only
# float32 can represent. The device packs the scalar at full float32
# precision, and torch casts it to the float32 tensor dtype, so both compare
# against the same float32 value (0.1 → 0.100000001). A scalar rounded to
# bfloat16 would make 1 + 2^-20 equal to the grid value 1.0.
_RELATIONAL_SCALAR_VALUES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.1,
    1.003,
    1.0 + 2.0**-20,
    1.0 - 2.0**-24,
    float(torch.finfo(torch.float32).tiny),
    float(torch.finfo(torch.bfloat16).max),
    float(torch.finfo(torch.float32).max),
    float(torch.finfo(torch.float32).min),
    float("inf"),
    float("-inf"),
    float("nan"),
]


@pytest.mark.parametrize("ttnn_op", RELATIONAL_OPS)
@pytest.mark.parametrize("scalar", _RELATIONAL_SCALAR_VALUES)
def test_relational_ops_scalar(device, ttnn_op, scalar):
    """Tensor-scalar coverage of the relational ops: the stratified grid,
    including ±0, ±inf and qNaN, against a scalar from every ordering /
    special-value class.

    The grid is 2048 unique values. Each is repeated across one tile width,
    giving (2048, 32): 64 full tiles, and the same (value, scalar) pairs as
    the 1-D grid. A 1-D [2048] tensor in tile layout is one tile row with
    31/32 padding, so it never reaches the multi-tile scalar path. Same
    contract as the tensor-tensor sweep: exact 0/1, ±0 compare equal, a NaN
    on either side makes every compare false except ne.
    """
    values = generate_bfloat16_binary_grid(include_spl_values=True, dtype=torch.float32)
    tile_width = 32
    input_a = values.unsqueeze(1).expand(values.numel(), tile_width).contiguous()
    tt_a = to_float32_tt_tensor(input_a, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_a, torch.tensor(scalar, dtype=input_a.dtype))
    result = ttnn.to_torch(ttnn_op(tt_a, scalar))

    assert_equal(golden.float(), result.float())


@pytest.mark.parametrize("ttnn_op", [ttnn.minimum, ttnn.maximum])
def test_minmax_ops(device, ttnn_op):
    """Pairwise coverage of ttnn.minimum / ttnn.maximum over the finite
    stratified grid (plus +0) at float32.

    The output is a select of one operand, so it matches torch exactly.
    ±0, ±inf and NaN are covered in test_minmax_special_values.
    """
    input_a, input_b = pairwise_inputs(include_zero=True, dtype=torch.float32)
    golden, result = run_binary(device, ttnn_op, input_a, input_b, input_dtype=ttnn.float32)
    assert_equal(golden, result)


@pytest.mark.parametrize("ttnn_op", [ttnn.minimum, ttnn.maximum])
def test_minmax_fp32_neighbors(device, ttnn_op):
    """Each finite grid value against its float32 neighbors, so the selected
    operand differs from the other one only in the low 16 mantissa bits.
    """
    values = _normal_values_with_normal_neighbors()
    up = torch.nextafter(values, torch.tensor(float("inf")))
    down = torch.nextafter(values, torch.tensor(float("-inf")))
    input_a = torch.stack([values, up, down, values])
    input_b = torch.stack([up, values, values, down])

    golden, result = run_binary(device, ttnn_op, input_a, input_b, input_dtype=ttnn.float32)
    assert_equal(golden, result)


def _float32_from_bits(bits):
    return torch.tensor([bits], dtype=torch.int64).to(torch.int32).view(torch.float32)[0]


_POS_NAN = _float32_from_bits(0x7FC00000)
_NEG_NAN = _float32_from_bits(0xFFC00000)


def _sign_magnitude_key(tensor):
    """Order float32 bit patterns as SFPSWAP does: by sign and magnitude bits.

    -NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN. Monotonic in the
    numeric value for non-NaN inputs, with -0 below +0.
    """
    bits = tensor.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(bits >= 0x80000000, 0xFFFFFFFF - bits, bits + 0x80000000)


@pytest.mark.parametrize("ttnn_op, is_max", [(ttnn.minimum, False), (ttnn.maximum, True)])
def test_minmax_special_values(device, ttnn_op, is_max):
    """Cross product of ±0, ±1, ±2.5, ±inf, +NaN and -NaN for ttnn.minimum /
    ttnn.maximum at float32, compared by bit pattern.

    The float32 kernel is one SFPSWAP in min/max mode with no NaN handling.
    It orders operands by their sign-magnitude bits, so a NaN acts as the
    largest value if its sign bit is clear and the smallest if it is set.
    The selected operand's bits are passed through unchanged.

    1) NaN: neither op propagates NaN the way torch does. Which op looks
       right depends on the NaN's sign bit (+NaN is 0x7FC00000, the canonical
       NaN; -NaN is 0xFFC00000, the x86 default NaN). torch returns NaN for
       every row.

    | a    | b    | device minimum | device maximum |
    |------|------|----------------|----------------|
    | x    | +NaN | x              | +NaN           |
    | +NaN | x    | x              | +NaN           |
    | x    | -NaN | -NaN           | x              |
    | -NaN | x    | -NaN           | x              |
    | +NaN | -NaN | -NaN           | +NaN           |

    x is any non-NaN value, ±inf included. Checked on device for x = ±0,
    ±1, ±2.5, ±inf.

    2) ±0 pairs: the device is sign-aware as in IEEE 754-2019, because -0
       orders below +0. torch just returns an operand, and which one depends
       on the code path, so torch cannot be the golden for the zero sign.

    | a  | b  | device minimum | device maximum |
    |----|----|----------------|----------------|
    | +0 | -0 | -0             | +0             |
    | -0 | +0 | -0             | +0             |
    | -0 | -0 | -0             | -0             |
    | +0 | +0 | +0             | +0             |

    | Call                    | 0-d tensor | 2048-element tensor |
    |-------------------------|------------|---------------------|
    | torch.minimum(-0, +0)   | -0         | +0                  |
    | torch.minimum(+0, -0)   | +0         | -0                  |
    | torch.maximum(-0, +0)   | -0         | +0                  |
    | torch.maximum(+0, -0)   | +0         | -0                  |
    """
    special_values = torch.stack(
        [
            torch.tensor(0.0),
            torch.tensor(-0.0),
            torch.tensor(1.0),
            torch.tensor(-1.0),
            torch.tensor(2.5),
            torch.tensor(-2.5),
            torch.tensor(float("inf")),
            torch.tensor(float("-inf")),
            _POS_NAN,
            _NEG_NAN,
        ]
    )
    input_a, input_b = pairwise_from_values(special_values)

    tt_a = to_float32_tt_tensor(input_a, device)
    tt_b = to_float32_tt_tensor(input_b, device)
    result = ttnn.to_torch(ttnn_op(tt_a, tt_b))

    key_a, key_b = _sign_magnitude_key(input_a), _sign_magnitude_key(input_b)
    pick_a = (key_a > key_b) if is_max else (key_a < key_b)
    expected = torch.where(pick_a, input_a, input_b)

    result_bits = result.view(torch.int32)
    expected_bits = expected.view(torch.int32)
    mismatch = result_bits != expected_bits
    assert not mismatch.any(), "bit-pattern mismatches (a, b, expected, device): " + ", ".join(
        f"({input_a[i, j].item()}, {input_b[i, j].item()}, {expected[i, j].item()}, {result[i, j].item()})"
        for i, j in torch.nonzero(mismatch).tolist()
    )


@pytest.mark.parametrize(
    "rtol, atol",
    [
        (1e-05, 1e-08),  # torch / ttnn defaults
        (0.0, 0.0),  # exact equality, plus matching-sign inf
        (1e-04, 0.0),  # relative-only
        (0.0, 1.0),  # absolute-only, large enough to fire on this grid
    ],
)
@pytest.mark.parametrize("equal_nan", [False, True])
def test_isclose(device, rtol, atol, equal_nan):
    """Pairwise coverage of ttnn.isclose over the stratified grid at float32.

    Kernel: abs(a - b) <= atol + rtol * abs(b), with an explicit Inf/NaN
    fix-up matching torch.isclose (equal_nan selects both-NaN => 1).

    | Case                      | float32 behavior                          | Handling        |
    |---------------------------|-------------------------------------------|-----------------|
    | normal abs(a - b)         | same fp32 difference as torch             | exact           |
    | subnormal abs(a - b)      | flushed to 0 in dest, so isclose is True  | expect True     |
    | tolerance fence (bf16)    | absent: torch and device both use fp32    | no exception    |

    The flush only changes the answer when the tolerance is below the
    subnormal gap, i.e. at atol = 0. On this grid that is 152 pairs with both
    abs(a) and abs(b) near 2^-126.
    Example (atol=0): 1.1755e-38 vs 1.1847e-38 → torch False, device True.
    """
    input_a, input_b = pairwise_inputs(include_spl_values=True, dtype=torch.float32)
    golden, result = run_binary(
        device,
        ttnn.isclose,
        input_a,
        input_b,
        golden_kwargs={"rtol": rtol, "atol": atol, "equal_nan": equal_nan},
        input_dtype=ttnn.float32,
    )

    golden = golden.float()
    result = result.float()

    # Min-normal abs(a - b) = 2^-126 is kept, so the bound is strict.
    diff = (input_a - input_b).abs()
    dest_sub_flushed = torch.isfinite(input_a) & torch.isfinite(input_b) & (diff < 2.0**-126)
    if atol == 0.0:
        assert (dest_sub_flushed & (golden == 0)).any(), "expected subnormal-difference pairs on this grid"
    adjusted_golden = torch.where(dest_sub_flushed, torch.ones_like(golden), golden)

    assert_equal(adjusted_golden, result)


@pytest.mark.parametrize("rtol", [1e-05, 1e-04])
def test_isclose_relative_tolerance(device, rtol):
    """ttnn.isclose with atol = 0 on pairs whose relative gap brackets rtol.

    Distinct values on the stratified grid are at least 2^-8 apart in
    relative terms, so test_isclose never accepts a pair through rtol alone.
    Here b runs over the normal grid values and a = b * (1 ± k * rtol):

    | k   | abs(a - b) vs rtol * abs(b) | expected |
    |-----|-----------------------------|----------|
    | 0.5 | half the tolerance          | True     |
    | 2   | twice the tolerance         | False    |

    k is far from 1, so float32 rounding of a cannot move a pair across the
    fence. Each pair is also checked swapped, which changes the abs(b) side.
    abs(b) is kept within [2^-100, 2^100] so rtol * abs(b) stays normal.
    """
    values = generate_bfloat16_binary_grid(dtype=torch.float32)
    b = values[torch.isfinite(values) & (values.abs() >= 2.0**-100) & (values.abs() <= 2.0**100)]
    b = b.double()
    a = torch.stack([b * (1 + sign * k * rtol) for sign in (1, -1) for k in (0.5, 2.0)]).float()
    b = b.float().expand_as(a)
    input_a = torch.cat([a, b])
    input_b = torch.cat([b, a])

    golden, result = run_binary(
        device,
        ttnn.isclose,
        input_a,
        input_b,
        golden_kwargs={"rtol": rtol, "atol": 0.0},
        input_dtype=ttnn.float32,
    )
    golden = golden.float()
    assert golden.any() and not golden.all(), "expected pairs on both sides of the rtol fence"
    assert_equal(golden, result.float())
