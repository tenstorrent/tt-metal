# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_allclose


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("use_multicore", [False, True])
@pytest.mark.parametrize("N", [1, 3, 5])
@pytest.mark.parametrize("C", [1, 2])
@pytest.mark.parametrize("H", [32, 448])
@pytest.mark.parametrize("W", [256, 672])
def test_untilize(device, in_dtype, use_multicore, N, C, H, W):
    torch_input_shape = [N, C, H, W]

    torch_input = torch.randn(torch_input_shape).bfloat16()

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.TILE_LAYOUT)

    output_tt = ttnn.untilize(ttnn_input, use_multicore=use_multicore)
    output_torch = ttnn.to_torch(output_tt)

    assert_allclose(torch_input, output_torch, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("use_multicore", [False, True])
@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("C", [1, 7])
@pytest.mark.parametrize("H", [128, 480])
@pytest.mark.parametrize("W", [32, 416])
@pytest.mark.parametrize("i_h", [10, 20])
@pytest.mark.parametrize("i_w", [2, 3])
def test_untilize_with_unpadding(device, in_dtype, use_multicore, N, C, H, W, i_h, i_w):
    torch_input_shape = [N, C, H, W]

    torch_input = torch.randn(torch_input_shape).bfloat16()

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.TILE_LAYOUT)

    output_tt = ttnn.untilize_with_unpadding(ttnn_input, [N - 1, C - 1, H - i_h, W - i_w], use_multicore=use_multicore)
    output_torch = ttnn.to_torch(output_tt)
    torch_input = torch_input[:, :, : H - i_h + 1, : W - i_w + 1]
    assert_allclose(torch_input, output_torch, rtol=1e-2, atol=1e-2)
