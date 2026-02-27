# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

pytestmark = pytest.mark.use_module_device


def compute_ulp_distance(output, expected):
    """Compute max ULP distance between two float tensors."""
    output_f32 = output.to(torch.float32)
    expected_f32 = expected.to(torch.float32)
    output_bits = output_f32.view(torch.int32)
    expected_bits = expected_f32.view(torch.int32)
    return (output_bits - expected_bits).abs().max().item()


@pytest.mark.parametrize(
    "shape",
    [
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 320])),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        (ttnn.bfloat16, ttnn.uint8),
        (ttnn.uint8, ttnn.bfloat16),
        (ttnn.float32, ttnn.uint8),
        (ttnn.uint8, ttnn.float32),
    ],
)
def test_typecast_uint8(shape, in_dtype, out_dtype, device):
    if in_dtype == ttnn.uint8:
        torch_input = torch.randint(0, 256, shape, dtype=torch.uint8)
    else:
        torch_input = torch.randn(shape).to(torch.bfloat16 if in_dtype == ttnn.bfloat16 else torch.float32)

    # Reference
    if out_dtype == ttnn.uint8:
        # Pytorch clamping/truncation for uint8
        torch_output = torch_input.to(torch.float32).clamp(0, 255).to(torch.uint8)
    else:
        torch_output = torch_input.to(torch.float32)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=in_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn.typecast(input_tensor, out_dtype)

    # For comparison, convert everything to a comparable torch format
    if out_dtype == ttnn.uint8:
        output_tensor = ttnn.to_torch(output_tensor).to(torch.uint8)
        assert torch.equal(output_tensor, torch_output)
    else:
        output_tensor = ttnn.to_torch(output_tensor).to(torch.float32)
        # ULP-based comparison: allow up to 1 ULP of error
        ulp_dist = compute_ulp_distance(output_tensor, torch_output)
        assert ulp_dist <= 1, f"ULP distance {ulp_dist} exceeds threshold"


def test_typecast_bf16_to_uint8_specific_case(device):
    # Case from issue #36219: 1.0 -> 1 (not 3)
    torch_ones = torch.ones([1, 1, 32, 32], dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_ones,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device
    )

    output = ttnn.typecast(input_tensor, ttnn.uint8)
    output_torch = ttnn.to_torch(output)

    expected = torch.ones([1, 1, 32, 32], dtype=torch.uint8)

    # Check if any element is 3 (the original bug)
    assert not torch.any(output_torch == 3), f"Found 3 in output: {output_torch}"
    assert torch.equal(output_torch.to(torch.uint8), expected)
