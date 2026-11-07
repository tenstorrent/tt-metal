# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

MAX_RELATIVE_ERROR = 0.02


def _max_relative_error(device, conv, cin, cout, hw, shard_layout=None):
    torch.manual_seed(0)
    x = torch.randn(1, cin, hw, hw)
    transpose = conv is ttnn.conv_transpose2d
    w = (torch.randn(cin, cout, 3, 3) if transpose else torch.randn(cout, cin, 3, 3)) / (cin * 9) ** 0.5
    torch_conv = torch.nn.functional.conv_transpose2d if transpose else torch.nn.functional.conv2d
    ref = torch_conv(x.double(), w.double(), padding=1).permute(0, 2, 3, 1).reshape(-1, cout)
    out = conv(
        input_tensor=ttnn.from_torch(
            x.permute(0, 2, 3, 1).reshape(1, 1, hw * hw, cin),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        ),
        weight_tensor=ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT),
        in_channels=cin,
        out_channels=cout,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=1,
        input_height=hw,
        input_width=hw,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.float32, shard_layout=shard_layout),
        groups=1,
        dtype=ttnn.float32,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    out = ttnn.to_torch(out).double().reshape(-1, cout)[: ref.shape[0]]
    return ((out - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "cin, cout, hw, shard_layout",
    [
        (320, 320, 64, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        (256, 256, 32, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        (256, 256, 32, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ],
)
def test_conv2d_float32_default_compute_config(device, cin, cout, hw, shard_layout):
    err = _max_relative_error(device, ttnn.conv2d, cin, cout, hw, shard_layout)
    assert err <= MAX_RELATIVE_ERROR, f"max relative error {err:.5f} > {MAX_RELATIVE_ERROR}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("cin, cout, hw", [(256, 256, 32)])
def test_conv_transpose2d_float32_default_compute_config(device, cin, cout, hw):
    err = _max_relative_error(device, ttnn.conv_transpose2d, cin, cout, hw)
    assert err <= MAX_RELATIVE_ERROR, f"max relative error {err:.5f} > {MAX_RELATIVE_ERROR}"
