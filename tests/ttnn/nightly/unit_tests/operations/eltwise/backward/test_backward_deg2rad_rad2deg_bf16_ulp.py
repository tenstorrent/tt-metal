# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import ulp_distance


def _all_bf16_values_without_subnormals():
    bits = torch.arange(1 << 16, dtype=torch.int32)
    exponent = (bits >> 7) & 0xFF
    bits = torch.where(exponent == 0, bits & 0x8000, bits)
    return bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)


@pytest.mark.parametrize(
    "operation_name,scale",
    (
        ("deg2rad", 0.017453292519943295),
        ("rad2deg", 57.29577951308232),
    ),
)
def test_backward_bf16_all_values_match_multiply(operation_name, scale, device):
    """Check optimized scalar multiplication against the legacy path for all BF16 values."""
    gradient = _all_bf16_values_without_subnormals()
    input_tensor = torch.zeros_like(gradient).requires_grad_(True)

    gradient_device = ttnn.from_torch(
        gradient, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    input_device = ttnn.from_torch(
        input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    operation = getattr(ttnn, f"{operation_name}_bw")
    actual = ttnn.to_torch(operation(gradient_device, input_device)[0], dtype=torch.bfloat16)
    reference = ttnn.to_torch(ttnn.multiply(gradient_device, scale), dtype=torch.bfloat16)

    finite = torch.isfinite(reference) & torch.isfinite(actual)
    assert torch.equal(torch.isfinite(reference), torch.isfinite(actual))
    distances = ulp_distance(reference[finite], actual[finite])
    values, counts = torch.unique(distances, return_counts=True)
    histogram = {int(value): int(count) for value, count in zip(values, counts)}
    print(f"{operation_name}_bw BF16 ULP histogram: {histogram}")
    assert int(distances.max()) <= 4
