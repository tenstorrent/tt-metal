# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Regression test for ttnn.remainder large-ratio bug (#54048).
# Tests that remainder produces correct results for |dividend/divisor| > 2^23,
# where the initial quotient estimate can be off by more than one divisor width.

import pytest
import ttnn
import torch


@pytest.mark.parametrize(
    "dividend, divisor, expected",
    [
        # Cases from the issue reproduction
        (14260600.0, 3.0, 1.0),
        (57042500.0, 3.0, 2.0),
        # Large ratios across sign combinations
        (67108864.0, 7.0, 4.0),
        (-67108864.0, 7.0, 3.0),
        (134217728.0, 7.0, 1.0),
        (-134217728.0, 7.0, 6.0),
        (268435456.0, 11.0, 3.0),
        (536870912.0, 11.0, 6.0),
        # Normal range still works
        (100000.0, 3.0, 1.0),
        (-100000.0, 3.0, 2.0),
    ],
)
def test_remainder_large_ratio(dividend, divisor, expected, device):
    torch.manual_seed(0)

    cpu_input = torch.tensor([dividend], dtype=torch.float32)
    gpu_input = ttnn.from_torch(cpu_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.remainder(gpu_input, divisor)
    output = ttnn.to_torch(output)

    assert (
        output.item() == expected
    ), f"remainder({dividend}, {divisor}) = {output.item()}, expected {expected}"

