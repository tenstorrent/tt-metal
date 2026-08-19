# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics

TEST_PADDING_VALUE = -42


@pytest.mark.parametrize("mem_config", [None, ttnn.DRAM_MEMORY_CONFIG, "block", "height"])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_mean_shard(device, mem_config, keepdim, dtype):
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    if mem_config == "height":
        # Height 100 is intentionally non-tile-aligned (not a multiple of 32).
        # Physical height pads to 128, so shard height 32 across 4 cores is valid.
        # After reducing dim=-1 with keepdim=False the output shape is (1, 100),
        # which exercises reshape_tiled's shard spec recomputation for HEIGHT_SHARDED.
        torch_input_tensor = torch.randn(1, 100, 160, dtype=torch_dtype)
        sharded_config = ttnn.create_sharded_memory_config(
            shape=(32, 160),
            core_grid=ttnn.CoreGrid(x=1, y=4),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
    else:
        torch_input_tensor = torch.randn(1, 1024, 160, dtype=torch_dtype)
        sharded_config = ttnn.create_sharded_memory_config(
            shape=(1, 1024, 160),
            core_grid=ttnn.CoreGrid(x=5, y=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_config,
    )

    if mem_config in ("block", "height"):
        memory_config = sharded_config
    else:
        memory_config = mem_config

    output_tensor = ttnn.mean(
        input_tensor,
        dim=-1,
        keepdim=keepdim,
        memory_config=memory_config,
    )
    tt_output_torch = ttnn.to_torch(output_tensor)
    torch_output = torch.mean(torch_input_tensor, -1, keepdim)
    # test for equivalance; FLOAT32 runs the accurate SFPU path, so its abs tolerance is far tighter.
    assert_numeric_metrics(
        torch_output,
        tt_output_torch,
        pcc_threshold=0.999,
        rtol=0.610,
        atol=1e-4 if dtype == ttnn.float32 else 0.002,
        frobenius_threshold=0.0055,
    )

    output_mem_config = output_tensor.memory_config()
    if mem_config == ttnn.DRAM_MEMORY_CONFIG:
        assert output_mem_config == mem_config
    else:
        assert output_mem_config.buffer_type == ttnn.BufferType.L1
        assert output_mem_config.is_sharded()
