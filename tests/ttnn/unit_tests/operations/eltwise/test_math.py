# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random

from loguru import logger

pytestmark = pytest.mark.use_module_device


def run_math_unary_test(device, h, w, ttnn_function, layout=ttnn.TILE_LAYOUT, pcc=0.9999):
    torch.manual_seed(0)

    # Generate random [0; 1] tensor
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    if "digamma" in str(ttnn_function):
        # Scale and shift range to [2; 102] for digamma
        torch_input_tensor = torch_input_tensor * 100.0 + 2.0

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    output_tensor = ttnn_function(input_tensor)
    # Verify output layout matches input layout
    assert output_tensor.layout == layout, f"Output layout {output_tensor.layout} should match input layout {layout}"
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_i0(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.i0, layout=layout, pcc=0.998)


@pytest.mark.parametrize("h", [5])
@pytest.mark.parametrize("w", [5])
def test_lgamma(device, h, w):
    run_math_unary_test(device, h, w, ttnn.lgamma, pcc=0.999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.uint16, ttnn.uint32])
def test_eq(device, h, w, output_dtype):
    torch.manual_seed(0)

    same = 50
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_a[0, 0] = same
    torch_input_tensor_a[0, 1] = same
    torch_input_tensor_a[0, 2] = same

    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b[0, 0] = same
    torch_input_tensor_b[0, 1] = same
    torch_input_tensor_b[0, 2] = same

    golden_function = ttnn.get_golden_function(ttnn.eq)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    output_tensor = ttnn.eq(input_tensor_a, input_tensor_b, dtype=output_dtype)
    assert output_tensor.get_dtype() == output_dtype
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device)) - 1
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)

    # EQ with a preallocated output tensor
    output_tensor_preallocated_bfloat16 = ttnn.ones(
        [h, w], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    output_tensor_preallocated = output_tensor_preallocated_bfloat16
    # There is no good way to create uint16 tensor in ttnn/torch, so we create bfloat16 and typecast to target
    if output_dtype != ttnn.bfloat16:
        output_tensor_preallocated = ttnn.typecast(
            output_tensor_preallocated_bfloat16, output_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.eq(input_tensor_a, input_tensor_b, dtype=output_dtype, output_tensor=output_tensor_preallocated)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))
    torch_output_tensor_preallocated = ttnn.to_torch(output_tensor_preallocated)
    assert_with_pcc(torch_output_tensor, torch_output_tensor_preallocated, 0.999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log10(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.log10, layout=layout)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log1p(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.log1p, layout=layout, pcc=0.999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log2(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.log2, layout=layout, pcc=0.999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_neg(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.neg, layout=layout)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_abs(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.abs, layout=layout)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rad2deg(device, h, w):
    run_math_unary_test(device, h, w, ttnn.rad2deg)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_cbrt(device, h, w):
    run_math_unary_test(device, h, w, ttnn.cbrt, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tril(device, h, w):
    run_math_unary_test(device, h, w, ttnn.tril)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_deg2rad(device, h, w):
    run_math_unary_test(device, h, w, ttnn.deg2rad)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sqrt(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.sqrt, layout=layout)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_digamma(device, h, w):
    run_math_unary_test(device, h, w, ttnn.digamma)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_erf(device, h, w):
    run_math_unary_test(device, h, w, ttnn.erf)


@pytest.mark.parametrize("h", [2])
@pytest.mark.parametrize("w", [3])
def test_erfc(device, h, w):
    run_math_unary_test(device, h, w, ttnn.erfc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_erfinv(device, h, w):
    run_math_unary_test(device, h, w, ttnn.erfinv, pcc=0.999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_square(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.square, layout=layout)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_exp2(device, h, w):
    run_math_unary_test(device, h, w, ttnn.exp2, pcc=0.98)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_expm1(device, h, w):
    run_math_unary_test(device, h, w, ttnn.expm1, pcc=0.99)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_triu(device, h, w):
    run_math_unary_test(device, h, w, ttnn.triu)


def run_math_unary_test_recip(device, h, w, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    low = -100
    high = 100

    torch_input_tensor = torch.empty((h, w), dtype=torch.bfloat16).uniform_(low, high) + 0.0001
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_math_unary_test_fixed_val(device, h, w, fill_value, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)
    torch_input_tensor = torch.full((h, w), fill_value, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_recip(device, h, w):
    class reciprocal_golden_wrapper:
        def __call__(self, input_tensor):
            return ttnn.reciprocal.golden_function(input_tensor, device=device)

    class reciprocal_wrapper:
        def __init__(self):
            self.golden_function = reciprocal_golden_wrapper()

        def __call__(self, input_tensor):
            return ttnn.reciprocal(input_tensor)

    run_math_unary_test_recip(device, h, w, reciprocal_wrapper(), pcc=0.999)


def run_math_unary_test_range(device, h, w, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)
    low = 1.6
    high = 100

    torch_input_tensor = torch_random((h, w), low, high, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [5])
@pytest.mark.parametrize("w", [5])
def test_multigammaln(device, h, w):
    run_math_unary_test_range(device, h, w, ttnn.multigammaln, pcc=0.999)


def _float_to_bf16_bits(f: float) -> int:
    """Convert float to BFloat16 bit representation."""
    import struct

    f32_bits = struct.unpack(">I", struct.pack(">f", f))[0]
    return f32_bits >> 16


def _bf16_bits_to_float(bits: int) -> float:
    """Convert BFloat16 bits to float."""
    import struct

    f32_bits = bits << 16
    return struct.unpack(">f", struct.pack(">I", f32_bits))[0]


def _bf16_daz_normalize(bits: int) -> int:
    """Apply DAZ (Denormals-Are-Zero) normalization to BF16 bits."""
    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    if (exp == 0) and (mantissa != 0):
        return 0x0000
    if bits == 0x8000:
        return 0x0000
    return bits


def _bf16_value_order_index_daz(bits: int) -> int:
    """Calculate the value order index for a BFloat16 value with DAZ."""
    bits = _bf16_daz_normalize(bits)
    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    if exp == 0xFF and mantissa != 0:
        return -1  # NaN
    if bits == 0x7F80:
        return 65281  # +inf
    if bits == 0xFF80:
        return -1  # -inf
    if bits == 0x0000:
        return 32640  # Zero
    if bits & 0x8000:
        magnitude = bits & 0x7FFF
        return 0x7F7F - magnitude
    else:
        return 32640 + bits - 0x007F


def _ulp_distance_bf16_daz(a: float, b: float) -> int:
    """Calculate ULP distance with DAZ+FTZ model."""
    a_bits = _bf16_daz_normalize(_float_to_bf16_bits(a))
    b_bits = _bf16_daz_normalize(_float_to_bf16_bits(b))
    a_exp = (a_bits >> 7) & 0xFF
    b_exp = (b_bits >> 7) & 0xFF
    if (a_exp == 0xFF and (a_bits & 0x7F) != 0) or (b_exp == 0xFF and (b_bits & 0x7F) != 0):
        return -1
    idx_a = _bf16_value_order_index_daz(a_bits)
    idx_b = _bf16_value_order_index_daz(b_bits)
    if idx_a < 0 or idx_b < 0:
        return -1
    return abs(idx_a - idx_b)


def run_math_test_polygamma_ulp(device, h, w, scalar, ttnn_function, max_ulp=1):
    torch.manual_seed(0)

    low = 1
    high = 10

    torch_input_tensor = torch_random((h, w), low, high, dtype=torch.bfloat16)

    # High-precision reference using float64
    ref_f64 = torch.special.polygamma(scalar, torch_input_tensor.to(torch.float64))

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor, scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    # Compute per-element ULP errors
    result_flat = output_tensor.flatten()
    ref_flat = ref_f64.flatten()

    worst_ulp = 0
    worst_idx = 0
    ulp_errors = []

    for i in range(len(result_flat)):
        res_val = result_flat[i].item()
        ref_val = ref_flat[i].item()
        if not torch.isfinite(torch.tensor(res_val)) or not torch.isfinite(torch.tensor(ref_val)):
            continue
        # Convert reference to bf16 (round-to-nearest) for fair comparison
        ref_bf16 = torch.tensor(ref_val, dtype=torch.bfloat16).item()
        ulp = _ulp_distance_bf16_daz(res_val, ref_bf16)
        if ulp < 0:
            continue
        ulp_errors.append(ulp)
        if ulp > worst_ulp:
            worst_ulp = ulp
            worst_idx = i

    import numpy as np

    ulp_arr = np.array(ulp_errors)
    logger.info(
        f"polygamma(n={scalar}) ULP stats — max: {worst_ulp}, "
        f"mean: {np.mean(ulp_arr):.2f}, p99: {np.percentile(ulp_arr, 99):.1f}, "
        f"p50: {np.percentile(ulp_arr, 50):.1f}"
    )

    assert worst_ulp <= max_ulp, (
        f"polygamma(n={scalar}) max ULP {worst_ulp} exceeds threshold {max_ulp} "
        f"at index {worst_idx}: got {result_flat[worst_idx].item()}, "
        f"expected {ref_flat[worst_idx].item():.6e}"
    )


@pytest.mark.parametrize("scalar", [1, 2, 5, 10])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_polygamma(device, h, w, scalar):
    run_math_test_polygamma_ulp(device, h, w, scalar, ttnn.polygamma, max_ulp=1)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_recip_fixed(device, h, w):
    run_math_unary_test_fixed_val(device, h, w, 0, ttnn.reciprocal, pcc=0.999)
