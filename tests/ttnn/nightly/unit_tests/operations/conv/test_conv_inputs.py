# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for Conv-family OPs (#31714).
Tests: conv2d, pool2d, upsample (conv context), halo

Note: Conv ops have unique requirements - activation must typically be sharded.
This tests whether interleaved inputs are also accepted (or auto-resharded).
"""

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Conv DRAM path still runs conv2d_L1 per slice, which needs L1_SMALL for config/scratch.
# So the module-scoped device must be created with l1_small_size.
pytestmark = pytest.mark.use_module_device({"l1_small_size": 16384})


@pytest.mark.parametrize(
    "input_memory_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["dram", "l1"],
)
def test_manual_conv2d(device, input_memory_config):
    """Minimal conv2d: input [1, 32, 32, 32] bfloat16, kernel (2,2).

    Runs with activation in DRAM (DRAM conv path) or L1 (L1 path; device needs l1_small_size).
    """
    torch.manual_seed(42)
    batch_size = 1
    in_channels = 32
    out_channels = 32
    input_h = 32
    input_w = 32
    kernel_size = 2

    # Constant values for deterministic comparison (1.0 for input, 1.0 for weights)
    # torch_input = torch.ones(batch_size, in_channels, input_h, input_w, dtype=torch.bfloat16) * 10
    # torch_weight = torch.ones(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16)
    torch_input = torch.randn(batch_size, in_channels, input_h, input_w, dtype=torch.bfloat16) * 10
    torch_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16)

    print("torch_input shape", torch_input.shape)
    print("torch_weight shape", torch_weight.shape)

    # NHWC for ttnn conv, then flatten to [1, 1, N*H*W, C]
    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, in_channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    print("torch_input_flat shape/ tt_input", torch_input_flat.shape)

    # DRAM input; device has l1_small_size via use_module_device({"l1_small_size": 16384})
    input_mem = ttnn.DRAM_MEMORY_CONFIG
    tt_input = ttnn.from_torch(
        torch_input_flat,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )
    # Host weights must be ROW_MAJOR; conv2d will prepare and move to device internally.
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_output = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_h,
        input_width=input_w,
        kernel_size=(kernel_size, kernel_size),
        stride=(1, 1),
        padding=(1, 1),  # padding 1 for 2x2 kernel keeps output 32x32
    )

    print("tt_output shape", tt_output.shape)
    # Golden: PyTorch conv2d output is NCHW [1, out_ch, out_h, out_w]
    torch_output = torch.nn.functional.conv2d(torch_input.float(), torch_weight.float(), padding=1).bfloat16()
    out_h = (input_h + 2 * 1 - kernel_size) // 1 + 1
    out_w = (input_w + 2 * 1 - kernel_size) // 1 + 1
    print("torch_output shape", torch_output.shape)

    # TT conv2d returns flattened [1, 1, out_h*out_w, out_ch]. Convert to NCHW to match torch.
    # Flattened order may be row-major (H, W) or column-major (W, H); try row-major first.
    tt_result = ttnn.to_torch(tt_output)
    tt_result = tt_result.reshape(batch_size, out_h, out_w, out_channels)
    # If TT uses column-major spatial, swap H and W: (0, 2, 1, 3) then (0, 3, 1, 2) -> NCHW
    tt_result = tt_result.permute(0, 3, 1, 2)  # (N,H,W,C) -> NCHW

    print("tt_result shape", tt_result.shape)

    # print("torch_output", torch_output)
    # print("tt_result", tt_result)
    assert_with_pcc(torch_output, tt_result, 0.99)
