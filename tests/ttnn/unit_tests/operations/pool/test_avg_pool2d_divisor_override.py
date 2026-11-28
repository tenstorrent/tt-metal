# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from ttnn.operations.pool import golden_avg_pool2d

pytestmark = pytest.mark.use_module_device

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _run_pool(device, torch_input, divisor_override):
    in_n, in_c, in_h, in_w = 1, 32, 16, 16
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return ttnn.avg_pool2d(
        input_tensor=ttnn_input,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=[2, 2],
        stride=[2, 2],
        padding=[0, 0],
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=divisor_override,
        applied_shard_scheme=HS,
        config_tensor_in_dram=True,
    )


def _pool_inputs():
    torch.manual_seed(1e3)
    return torch.rand((1, 1, 16 * 16, 32)).bfloat16()


@pytest.mark.parametrize("divisor_override", [0])
def test_avg_pool2d_zero_divisor_raises(device, expect_error, divisor_override):
    with expect_error(RuntimeError, "divisor_override must be non-zero"):
        _run_pool(device, _pool_inputs(), divisor_override)


@pytest.mark.parametrize("divisor_override", [None, 1, 5, -4])
def test_avg_pool2d_divisor_override_matches_torch(device, divisor_override):
    torch_input = _pool_inputs()
    ttnn_output = _run_pool(device, torch_input, divisor_override)

    golden = golden_avg_pool2d(
        input_tensor=torch_input,
        batch_size=1,
        input_h=16,
        input_w=16,
        channels=32,
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=(0, 0),
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=divisor_override,
    )
    out = ttnn.to_torch(ttnn_output).float()
    assert out.shape == golden.shape
    assert torch.allclose(out, golden.float(), atol=0.05, rtol=0.05)
