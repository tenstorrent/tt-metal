# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

# The first six shapes run ttnn.var / ttnn.std (at most 64 tiles per plane; 256x256 is exactly 64); the rest run the
# composite. 30x70, 300x700 and 600x600 are not tile-aligned.
SHAPES = [
    [1, 1, 32, 32],
    [2, 4, 64, 64],
    [1, 2, 30, 70],
    [2, 64, 64],
    [1, 2, 3, 32, 32],
    [1, 1, 256, 256],
    [1, 1, 256, 512],
    [1, 1, 512, 512],
    [1, 1, 300, 700],
    [1, 600, 600],
]


@pytest.mark.parametrize("dtype, rtol", [(ttnn.bfloat16, 1.6e-2), (ttnn.float32, 1e-5)], ids=["bf16", "fp32"])
@pytest.mark.parametrize("offset", [1.0, 100.0])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: "x".join(map(str, shape)))
def test_var_std_hw(device, dtype, rtol, offset, shape):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)
    x = (torch.randn(shape) * 3 + offset).to(torch_dtype)
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    variance = torch.var(x.double(), dim=(-2, -1), keepdim=True, correction=0)

    for op, expected in ((ttnn.var_hw, variance), (ttnn.std_hw, variance.sqrt())):
        actual = ttnn.to_torch(op(tt_x)).double()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=0)


# A bfloat16 mean of about 100 is a multiple of 0.5, so subtracting it as a bfloat16 adds its rounding error squared to
# every squared deviation: 4% of the variance for a std of 1, 15% for a std of 0.5. The composite subtracts an FP32
# mean. 256x256 runs ttnn.var / ttnn.std; 512x512 and 300x700 run the composite.
@pytest.mark.parametrize("std", [1.0, 0.5])
@pytest.mark.parametrize(
    "shape", [[1, 1, 256, 256], [1, 1, 512, 512], [1, 1, 300, 700]], ids=lambda shape: "x".join(map(str, shape))
)
def test_var_std_hw_bf16_large_mean(device, shape, std):
    torch.manual_seed(0)
    x = (torch.randn(shape) * std + 100.3).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    variance = torch.var(x.double(), dim=(-2, -1), keepdim=True, correction=0)

    for op, expected in ((ttnn.var_hw, variance), (ttnn.std_hw, variance.sqrt())):
        actual = ttnn.to_torch(op(tt_x)).double()
        torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=0)


def _input_memory_config(shape, placement):
    if placement == "height_sharded":
        return ttnn.create_sharded_memory_config(shape, ttnn.CoreGrid(y=1, x=8), ttnn.ShardStrategy.HEIGHT)
    if placement == "block_sharded":
        return ttnn.create_sharded_memory_config(shape, ttnn.CoreGrid(y=2, x=2), ttnn.ShardStrategy.BLOCK)
    return {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}[placement]


# Without memory_config the output stays where the input is; with one it goes there. 64x64, 8 height-sharded 64x64
# planes and a block-sharded 256x256 run ttnn.var / ttnn.std; 512x512, 8 height-sharded 512x512 planes and a
# block-sharded 512x512 run the composite.
@pytest.mark.parametrize(
    "shape, placement",
    [
        ([1, 1, 64, 64], "dram"),
        ([1, 1, 64, 64], "l1"),
        ([1, 1, 512, 512], "dram"),
        ([1, 1, 512, 512], "l1"),
        ([1, 8, 64, 64], "height_sharded"),
        ([1, 8, 512, 512], "height_sharded"),
        ([1, 1, 256, 256], "block_sharded"),
        ([1, 1, 512, 512], "block_sharded"),
    ],
    ids=lambda value: "x".join(map(str, value)) if isinstance(value, list) else value,
)
@pytest.mark.parametrize(
    "memory_config", [None, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["none", "dram", "l1"]
)
def test_var_std_hw_memory_config(device, shape, placement, memory_config):
    input_memory_config = _input_memory_config(shape, placement)
    torch.manual_seed(0)
    x = (torch.randn(shape) * 3 + 1).to(torch.bfloat16)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )
    variance = torch.var(x.double(), dim=(-2, -1), keepdim=True, correction=0)
    expected_memory_config = input_memory_config if memory_config is None else memory_config

    for op, expected in ((ttnn.var_hw, variance), (ttnn.std_hw, variance.sqrt())):
        output = op(tt_x, memory_config=memory_config)
        assert output.memory_config().buffer_type == expected_memory_config.buffer_type
        assert output.memory_config().memory_layout == expected_memory_config.memory_layout
        torch.testing.assert_close(ttnn.to_torch(output).double(), expected, rtol=1.6e-2, atol=0)


def _runs_one_program(op, tt_x):
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        op(tt_x)
    finally:
        captured_graph = ttnn.graph.end_graph_capture()
    return "WelfordReduceDeviceOperation" in ttnn.graph.extract_calltrace(captured_graph)


# Both sides of each limit: one program for planes of at most 64 tiles, and for larger planes from 10 planes up.
@pytest.mark.parametrize(
    "planes, height, width, one_program",
    [
        (1, 256, 256, True),  # 64 tiles
        (1, 32, 2080, False),  # 65 tiles
        (1, 256, 512, False),  # 128 tiles
        (9, 256, 512, False),  # 128 tiles each
        (10, 256, 512, True),  # 128 tiles each
        ("cores_plus_one", 32, 2048, True),  # 64 tiles each
    ],
)
def test_var_std_hw_path_at_limits(device, planes, height, width, one_program):
    if device.arch() == ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar has no two-pass Std/Var kernel, so var_hw / std_hw always run the composite")
    if planes == "cores_plus_one":
        grid = device.compute_with_storage_grid_size()
        planes = grid.x * grid.y + 1
    tt_x = ttnn.from_torch(
        torch.randn(planes, 1, height, width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    for op in (ttnn.var_hw, ttnn.std_hw):
        assert _runs_one_program(op, tt_x) == one_program, f"{op} on {planes} x {height} x {width}"
