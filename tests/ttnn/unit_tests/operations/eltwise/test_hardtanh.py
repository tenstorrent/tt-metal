# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_pcc,
    assert_with_ulp,
    assert_allclose,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)

pytestmark = pytest.mark.use_module_device


def _golden(input_tensor, min_val, max_val):
    return torch.nn.functional.hardtanh(input_tensor, min_val=min_val, max_val=max_val)


def _run_bfloat16(device, input_tensor, min_val, max_val):
    golden = _golden(input_tensor, min_val, max_val)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.hardtanh(tt_input, min_val=min_val, max_val=max_val)
    result = ttnn.to_torch(tt_output)
    return golden, result


@pytest.mark.parametrize("h,w", [(64, 128)])
def test_hardtanh_default_bfloat16(device, h, w):
    torch.manual_seed(0)
    input_tensor = torch.randn((h, w), dtype=torch.bfloat16)
    golden, result = _run_bfloat16(device, input_tensor, min_val=-1.0, max_val=1.0)
    assert_with_ulp(golden, result, 2)
    assert_with_pcc(golden, result, 0.99)


@pytest.mark.parametrize("h,w", [(64, 128)])
def test_hardtanh_default_fp32(device, h, w):
    torch.manual_seed(0)
    input_tensor = torch.randn((h, w), dtype=torch.float32)
    golden = _golden(input_tensor, -1.0, 1.0)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.hardtanh(tt_input, min_val=-1.0, max_val=1.0)
    result = ttnn.to_torch(tt_output)
    assert_allclose(golden, result, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "min_val,max_val",
    [
        (-1.0, 1.0),
        (-0.5, 0.5),
        (-2.0, 2.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (-5.5, 3.7),
    ],
)
@pytest.mark.parametrize("h,w", [(64, 128)])
def test_hardtanh_params_bfloat16(device, h, w, min_val, max_val):
    torch.manual_seed(42)
    input_tensor = torch.randn((h, w), dtype=torch.bfloat16) * 3.0
    golden, result = _run_bfloat16(device, input_tensor, min_val, max_val)
    assert_with_ulp(golden, result, 2)
    assert_with_pcc(golden, result, 0.99)


@pytest.mark.parametrize(
    "min_val,max_val",
    [
        (-1.0, 1.0),
        (-0.5, 0.5),
        (-2.0, 2.0),
        (0.0, 1.0),
        (-5.5, 3.7),
    ],
)
@pytest.mark.parametrize("h,w", [(64, 128)])
def test_hardtanh_params_fp32(device, h, w, min_val, max_val):
    torch.manual_seed(42)
    input_tensor = torch.randn((h, w), dtype=torch.float32) * 3.0
    golden = _golden(input_tensor, min_val, max_val)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.hardtanh(tt_input, min_val=min_val, max_val=max_val)
    result = ttnn.to_torch(tt_output)
    assert_allclose(golden, result, rtol=1.6e-2, atol=1e-2)


def test_hardtanh_exhaustive_bfloat16(device):
    input_tensor = generate_all_bfloat16_bitpatterns(torch.bfloat16).flatten()
    input_tensor = flush_subnormal_values_to_zero(input_tensor)
    # Filter out NaN and Inf values: hardtanh behavior on non-finite inputs is undefined
    finite_mask = torch.isfinite(input_tensor)
    input_tensor = input_tensor[finite_mask]
    golden = _golden(input_tensor, -1.0, 1.0)
    tt_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = ttnn.hardtanh(tt_input, min_val=-1.0, max_val=1.0)
    result = ttnn.to_torch(tt_output)
    assert_with_ulp(golden, result, 2)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 64, 64),
        (1, 1, 320, 384),
        (1, 3, 320, 384),
        (2, 4, 64, 128),
    ],
)
def test_hardtanh_shapes_bfloat16(device, shape):
    torch.manual_seed(0)
    input_tensor = torch.randn(shape, dtype=torch.bfloat16) * 2.0
    golden = _golden(input_tensor, -1.0, 1.0)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.hardtanh(tt_input, min_val=-1.0, max_val=1.0)
    result = ttnn.to_torch(tt_output)
    assert_with_pcc(golden, result, 0.99)


def test_hardtanh_golden_function_consistency(device):
    torch.manual_seed(0)
    input_tensor = torch.randn((64, 128), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.hardtanh)
    golden_from_ttnn = golden_function(input_tensor, min_val=-1.0, max_val=1.0)
    golden_direct = torch.nn.functional.hardtanh(input_tensor, min_val=-1.0, max_val=1.0)
    assert torch.allclose(golden_from_ttnn, golden_direct)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.hardtanh(tt_input, min_val=-1.0, max_val=1.0)
    result = ttnn.to_torch(tt_output)
    assert_with_pcc(golden_direct, result, 0.99)
