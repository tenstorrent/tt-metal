# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from models.common.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_allclose, assert_equal

TEST_PADDING_VALUE = -42


@pytest.mark.parametrize("input_shape", [(16, 2, 32, 3), (16, 2, 32, 24), (1, 1, 64, 64)])
@pytest.mark.parametrize("dim", [None, -1, -2])
@pytest.mark.parametrize("scalar", [1.0, 2.5, -2.5])
@pytest.mark.parametrize("fast_and_approximate_mode", [False, True], ids=["accurate", "fast"])
def test_max_fp32_fast_and_approximate_mode(device, input_shape, dim, scalar, fast_and_approximate_mode):
    """FLOAT32 max with both values of fast_and_approximate_mode.
    - False (default): accurate SFPU path - result matches torch exactly.
    - True: faster FPU/TF32 path - result is approximate.
    """
    torch.manual_seed(1)

    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    torch_output_tensor = torch.amax(scalar * torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.max(input_tensor, fast_and_approximate_mode=fast_and_approximate_mode, dim=dim, scalar=scalar)
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor)).reshape(torch_output_tensor.shape)

    # A negative scalar dispatches to min, and fp32 min is not routed to the SFPU yet,
    # so it stays on the FPU path.
    if fast_and_approximate_mode or scalar < 0 or device.arch() == ttnn.device.Arch.QUASAR:
        assert_allclose(torch_output_tensor, output_tensor, rtol=1e-3, atol=1e-2)
    else:
        assert_equal(torch_output_tensor, output_tensor)
