# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 7: TILE activations (+ bf8b). Pins the three TILE paths:
  technique 1 — sharded TILE: own shard untilized in place + halo tile-rows (HEIGHT / BLOCK / WIDTH);
  technique 2 — DRAM-interleaved TILE: per-m-block window tile-rows from DRAM (R0);
  technique 3 — R9: 1x1 stride-1 TILE, the shard tiles are in0 (HEIGHT in place, BLOCK multicast);
plus the 4-D TILE input form (W, W_out tile multiples) and its refusal otherwise."""
import pytest
import torch
import ttnn

from eval.golden_tests.conv2d_nhwc.helpers import run_conv2d_nhwc
from ttnn.operations.conv2d_nhwc import conv2d_nhwc, prepare_conv2d_weights

ML = ttnn.TensorMemoryLayout
TL = ttnn.TILE_LAYOUT

CASES = [
    pytest.param(((2, 28, 28, 64), 64, 3, 1, 1, 1, 1), ML.HEIGHT_SHARDED, id="height_3x3"),
    pytest.param(((2, 28, 28, 64), 256, 1, 0, 1, 1, 1), ML.HEIGHT_SHARDED, id="height_1x1_direct"),
    pytest.param(((2, 56, 56, 64), 128, 3, 1, 2, 1, 1), ML.HEIGHT_SHARDED, id="height_s2"),
    pytest.param(((1, 28, 28, 256), 256, 3, 1, 1, 1, 1), ML.BLOCK_SHARDED, id="block_3x3_ragged"),
    pytest.param(((1, 14, 14, 1024), 256, 1, 0, 1, 1, 1), ML.BLOCK_SHARDED, id="block_1x1_direct"),
    pytest.param(((1, 8, 8, 512), 512, 3, 1, 1, 1, 1), ML.WIDTH_SHARDED, id="width_3x3"),
    pytest.param(((1, 32, 32, 320), 320, 3, 1, 2, 1, 1), ML.INTERLEAVED, id="dram_s2"),
    pytest.param(((1, 14, 14, 64), 64, 3, 1, 1, 1, 1), ML.INTERLEAVED, id="dram_3x3"),
]


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bf8b", "bf16"])
@pytest.mark.parametrize("inputs,memory_layout", CASES)
def test_conv2d_nhwc_tile(device, inputs, memory_layout, dtype):
    run_conv2d_nhwc(
        inputs,
        dtype=dtype,
        weight_dtype=ttnn.bfloat8_b,
        layout=TL,
        bias_mode="with_bias",
        memory_layout=memory_layout,
        activation="relu",
        device=device,
    )


def test_conv2d_nhwc_tile_4d(device, expect_error):
    """A 4-D TILE input is accepted when W and W_out are tile multiples; the output is 4-D TILE."""
    N, H, W, C, C_out = 1, 8, 32, 64, 64
    torch.manual_seed(0)
    x = torch.randn(N, H, W, C)
    w = torch.randn(C_out, C, 3, 3)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=TL, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tw = prepare_conv2d_weights(
        w,
        kernel_size=3,
        padding=1,
        input_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        input_memory_config=tx.memory_config(),
        device=device,
    )
    out = conv2d_nhwc(tx, tw, 3, padding=1)
    assert list(out.shape) == [N, H, W, C_out] and out.layout == TL
    ref = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, padding=1).permute(0, 2, 3, 1)
    got = ttnn.to_torch(out).float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1]
    assert pcc > 0.995, pcc
    # W_out = 30: a 4-D TILE tensor would straddle image rows -> refused.
    with expect_error(ValueError, "4-D TILE activation needs W and W_out multiples of 32"):
        conv2d_nhwc(tx, tw, 3, padding=0)
