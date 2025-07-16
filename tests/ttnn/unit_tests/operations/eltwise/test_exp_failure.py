# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_exp_failure(device):
    a_vals = [
        0.7734375,
        0.30859375,
        0.890625,
        0.71875,
        0.66015625,
        0.26953125,
        0.05078125,
        0.71484375,
        0.4296875,
        0.3671875,
        0.31640625,
        0.5703125,
        0.1875,
        0.6015625,
        0.671875,
        0.1640625,
        0.625,
        0.76953125,
        0.9609375,
    ]

    b_vals = [
        0.99609375,
        0.90234375,
        0.98828125,
        0.79296875,
        0.81640625,
        0.390625,
        0.48828125,
        0.3046875,
        0.75,
        0.25,
        0.18359375,
        0.171875,
        0.14453125,
        0.0859375,
        0.08203125,
        0.04296875,
        0.72265625,
        0.0078125,
        0.69921875,
    ]

    a = torch.tensor(a_vals, dtype=torch.bfloat16)
    b = torch.tensor(b_vals, dtype=torch.bfloat16)

    print("Original A:", a)
    print("Original B:", b)

    tt_in_a = ttnn.from_torch(
        a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn.set_printoptions(profile="full")
    print("TTNN A:", tt_in_a)
    print("TTNN B:", tt_in_b)

    print("Apply pre-activations : ")
    a = torch.floor(a)
    print("\nAfter torch.floor(a):", a)
    tt_in_a = ttnn.floor(tt_in_a)
    print("After ttnn.floor(a):", tt_in_a)

    a = torch.exp(a)
    print("\nAfter torch.exp(a):", a)
    tt_in_a = ttnn.exp(tt_in_a)
    print("After ttnn.exp(a):", tt_in_a)

    b = torch.exp(b)
    print("\nAfter torch.exp(b):", b)
    tt_in_b = ttnn.exp(tt_in_b)
    print("After ttnn.exp(b):", tt_in_b)

    a = torch.ldexp(a, b)
    print("\nAfter torch.ldexp(a, b):", a)
    tt_in_a = ttnn.ldexp(tt_in_a, tt_in_b)
    print("After ttnn.ldexp(a, b):", tt_in_a)

    result = ttnn.to_torch(tt_in_a)
    assert_with_pcc(a, result, 0.999)
