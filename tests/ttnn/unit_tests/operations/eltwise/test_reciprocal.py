# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device


def test_reciprocal_fp32_faithful(device):
    if device.arch() != ttnn.device.Arch.WORMHOLE_B0:
        pytest.skip("Wormhole reciprocal precision regression")

    # Exhaust every FP32 mantissa in [1, 2), in eight chunks to bound memory use.
    chunk_size = 1 << 20
    for start in range(0x3F800000, 0x40000000, chunk_size):
        bits = torch.arange(start, start + chunk_size, dtype=torch.int32)
        values = bits.view(torch.float32).reshape(1024, 1024)
        input_tensor = ttnn.from_torch(values, layout=ttnn.TILE_LAYOUT, device=device)
        actual = ttnn.to_torch(ttnn.reciprocal(input_tensor)).to(torch.float64)
        exact = 1.0 / values.to(torch.float64)
        # Reciprocal results lie in [0.5, 1], with spacing 2**-24 below 1.
        assert torch.all(
            (actual - exact).abs() < 2.0**-24
        ), f"Unfaithful reciprocal in chunk starting at {start:#010x}"


def test_reciprocal_shared_seed_polygamma_fp32(device):
    if device.arch() != ttnn.device.Arch.WORMHOLE_B0:
        pytest.skip("Wormhole shared reciprocal precision regression")

    # Guards the BF16-only scoping of the reciprocal seed: polygamma calls
    # recip_init<..., false> but takes FP32 reciprocals, so applying the
    # BF16-tuned seed there raised the maximum from 19 to 22 ULP.
    # Inputs are neighborhoods of the worst cases from an exhaustive
    # polygamma(10) sweep over FP32 [1, 10]. The threshold is the current
    # maximum, so a polygamma accuracy change may legitimately move it.
    centers = torch.tensor([1.9826958179473877, 1.7422690391540527], dtype=torch.float32).view(torch.int32)
    bits = centers[:, None] + torch.arange(-256, 256, dtype=torch.int32)
    values = bits.view(torch.float32).reshape(32, 32)
    input_tensor = ttnn.from_torch(values, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.polygamma(input_tensor, 10))
    expected = torch.polygamma(10, values.to(torch.float64)).to(torch.float32)
    assert_with_ulp(expected_result=expected, actual_result=actual, ulp_threshold=19)
