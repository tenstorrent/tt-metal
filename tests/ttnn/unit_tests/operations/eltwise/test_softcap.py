# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp


def softcap_golden(x, cap):
    return cap * torch.tanh(x / cap)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 32, 32),
        (1, 1, 64, 64),
        (1, 3, 128, 128),
        (4, 7, 64, 128),
    ],
)
@pytest.mark.parametrize(
    "cap",
    [1.0, 5.0, 10.0, 50.0],
)
class TestSoftcapBfloat16:
    def test_softcap_bf16(self, device, input_shape, cap):
        torch.manual_seed(0)
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        expected = softcap_golden(torch_input.float(), cap).bfloat16()

        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output = ttnn.softcap(tt_input, cap=cap)
        actual = ttnn.to_torch(tt_output)

        assert_with_ulp(expected, actual, ulp_threshold=2)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 32, 32),
        (1, 1, 64, 64),
    ],
)
@pytest.mark.parametrize(
    "cap",
    [1.0, 10.0, 50.0],
)
class TestSoftcapFloat32:
    def test_softcap_fp32(self, device, input_shape, cap):
        torch.manual_seed(0)
        torch_input = torch.randn(input_shape, dtype=torch.float32)
        expected = softcap_golden(torch_input, cap)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output = ttnn.softcap(tt_input, cap=cap)
        actual = ttnn.to_torch(tt_output)

        torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 32, 32),
    ],
)
@pytest.mark.parametrize(
    "cap",
    [1.0, 50.0],
)
class TestSoftcapAllclose:
    def test_softcap_allclose(self, device, input_shape, cap):
        torch.manual_seed(0)
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        expected = softcap_golden(torch_input.float(), cap).bfloat16()

        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output = ttnn.softcap(tt_input, cap=cap)
        actual = ttnn.to_torch(tt_output)

        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=1.6e-2,
            atol=1e-2,
        )


class TestSoftcapEdgeCases:
    def test_softcap_zeros(self, device):
        torch_input = torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16)
        expected = softcap_golden(torch_input.float(), 50.0).bfloat16()

        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output = ttnn.softcap(tt_input, cap=50.0)
        actual = ttnn.to_torch(tt_output)

        assert_with_ulp(expected, actual, ulp_threshold=2)

    def test_softcap_large_values(self, device):
        torch.manual_seed(42)
        torch_input = torch.randn((1, 1, 32, 32), dtype=torch.bfloat16) * 100.0
        cap = 10.0
        expected = softcap_golden(torch_input.float(), cap).bfloat16()

        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output = ttnn.softcap(tt_input, cap=cap)
        actual = ttnn.to_torch(tt_output)

        assert_with_ulp(expected, actual, ulp_threshold=2)

    def test_softcap_small_cap(self, device):
        torch.manual_seed(7)
        torch_input = torch.randn((1, 1, 32, 32), dtype=torch.bfloat16)
        cap = 0.5
        expected = softcap_golden(torch_input.float(), cap).bfloat16()

        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output = ttnn.softcap(tt_input, cap=cap)
        actual = ttnn.to_torch(tt_output)

        assert_with_ulp(expected, actual, ulp_threshold=2)

    def test_softcap_default_cap(self, device):
        torch.manual_seed(0)
        torch_input = torch.randn((1, 1, 32, 32), dtype=torch.bfloat16)
        expected = softcap_golden(torch_input.float(), 50.0).bfloat16()

        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output = ttnn.softcap(tt_input)
        actual = ttnn.to_torch(tt_output)

        assert_with_ulp(expected, actual, ulp_threshold=2)
