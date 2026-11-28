# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_zeros over the four production cases.

The reference is torch.zeros, so PCC and relative RMS are degenerate: the reference has zero
variance, and comp_pcc returns 1.0 for two equal constant tensors and also when a -0.0 is
present. The gate is therefore the bit pattern: max ULP distance 0 and no -0.0. PCC, the
absolute errors and the RMS error are still reported for the record. The got/true ratio is
undefined because every reference element is zero.

Every metric is accumulated over slices of the leading dim: p1 reads back about 2.7e8 elements,
and whole-tensor float64 temporaries would need several GB of host memory.

Cases outside SUPPORTED are refused with the typed support refusal and xfail; they run as soon as
a refinement adds their dtype.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.g6_zeros import SUPPORTED, g6_zeros

CASES = [
    pytest.param((8, 12, 128, 128), ttnn.float32, id="p3-float32"),
    pytest.param((16, 12, 128, 128), ttnn.float32, id="p2-float32"),
    pytest.param((1, 4, 256, 128), ttnn.bfloat16, id="p4-bfloat16"),
    pytest.param((32831, 1, 32, 256), ttnn.bfloat8_b, id="p1-bfloat8_b"),
]

# bfloat8_b is compared after the host unpack, widened to float32 (the widening keeps a zero's sign).
_COMPARE_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.float32}
_BITS_VIEW = {torch.float32: torch.int32, torch.bfloat16: torch.int16}
_LEADING_ROWS_PER_SLICE = 4096


def _ulp_distance(actual, expected):
    """Elementwise ULP distance on the sign-magnitude bit patterns (monotonic integer ordering)."""
    bits_view = _BITS_VIEW[actual.dtype]
    sign_mask = torch.iinfo(bits_view).min

    def ordered(t):
        bits = t.contiguous().view(bits_view).to(torch.int64)
        return torch.where(bits < 0, sign_mask - bits, bits)

    return (ordered(actual) - ordered(expected)).abs()


def _run(shape, dtype, device):
    return g6_zeros(
        list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize("shape, dtype", CASES)
def test_g6_zeros_precision_baseline(shape, dtype, device, expect_error):
    if dtype not in SUPPORTED["dtype"]:
        with expect_error(SupportRefusal, ""):
            _run(shape, dtype, device)
        pytest.xfail(f"{dtype} is outside SUPPORTED")

    output = _run(shape, dtype, device)
    compare_dtype = _COMPARE_DTYPE[dtype]
    bits_view = _BITS_VIEW[compare_dtype]
    readback = ttnn.to_torch(output)
    assert list(readback.shape) == list(shape)

    min_pcc, allclose_ok = 1.0, True
    max_abs_err, abs_sum, sq_sum, max_ulp, negative_zeros = 0.0, 0.0, 0.0, 0, 0
    for start in range(0, shape[0], _LEADING_ROWS_PER_SLICE):
        actual = readback[start : start + _LEADING_ROWS_PER_SLICE].to(compare_dtype).contiguous()
        expected = torch.zeros(actual.shape, dtype=compare_dtype)
        _, pcc = comp_pcc(expected, actual)
        allclose_slice, _ = comp_allclose(expected, actual)
        min_pcc = min(min_pcc, float(pcc))
        allclose_ok = allclose_ok and bool(allclose_slice)
        error = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
        max_abs_err = max(max_abs_err, float(error.max()))
        abs_sum += float(error.sum())
        sq_sum += float(error.square().sum())
        max_ulp = max(max_ulp, int(_ulp_distance(actual, expected).max()))
        negative_zeros += int((actual.view(bits_view) == torch.iinfo(bits_view).min).sum())
    numel = readback.numel()
    mean_abs_err = abs_sum / numel
    rms_err = (sq_sum / numel) ** 0.5

    print(
        f"\nPRECISION {list(shape)} {dtype}: pcc={min_pcc} max_abs_err={max_abs_err} mean_abs_err={mean_abs_err} "
        f"rms_err={rms_err} rel_rms=undefined(ref std 0) max_ulp={max_ulp} negative_zeros={negative_zeros} "
        f"allclose={allclose_ok} ratio=undefined(ref all zero)"
    )

    assert max_ulp == 0, f"{max_ulp} ULP from +0.0"
    assert negative_zeros == 0, f"{negative_zeros} elements are -0.0"
    assert min_pcc == 1.0 and allclose_ok
    assert max_abs_err == 0.0 and rms_err == 0.0
