# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58722 review): does a block-pack program change the time of the next program on its cores? The
measured op is a bfp8 add (no block pack in its program) on a 4-tile-per-core height-sharded tensor, alone or each time
after a bf16 add whose program packs with the block pack."""
import pytest
import torch
import ttnn


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


def _t(device, shape, dt, mc, seed):
    torch.manual_seed(seed)
    return ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)


@pytest.mark.parametrize("t_rows", [256, 1024])
@pytest.mark.parametrize("before", ["alone", "after_block_pack", "after_bfp8_out"])
def test_next_program(device, t_rows, before):
    shape = (1, 1, t_rows, 128)
    mc = ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=2, x=4), strategy=ttnn.ShardStrategy.HEIGHT)
    a8, b8 = _t(device, shape, ttnn.bfloat8_b, mc, 1), _t(device, shape, ttnn.bfloat8_b, mc, 2)
    a16, b16 = _t(device, shape, ttnn.bfloat16, mc, 3), _t(device, shape, ttnn.bfloat16, mc, 4)
    for _ in range(5):
        if before == "after_block_pack":
            ttnn.subtract(a16, b16, dtype=ttnn.bfloat16, memory_config=mc)
        elif before == "after_bfp8_out":
            ttnn.subtract(a8, b8, dtype=ttnn.bfloat8_b, memory_config=mc)
        ttnn.add(a8, b8, dtype=ttnn.bfloat8_b, memory_config=mc)


@pytest.mark.parametrize("t_rows", [256, 1024])
@pytest.mark.parametrize("before", ["alone", "after_block_pack", "after_bfp8_out"])
@pytest.mark.parametrize("dt", ["bf16", "bfp8"])
def test_next_unary(device, t_rows, before, dt):
    """The measured op is a unary relu (UnaryDeviceOperation) on the same cores, alone or after a binary op."""
    shape = (1, 1, t_rows, 128)
    mc = ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=2, x=4), strategy=ttnn.ShardStrategy.HEIGHT)
    a8, b8 = _t(device, shape, ttnn.bfloat8_b, mc, 1), _t(device, shape, ttnn.bfloat8_b, mc, 2)
    a16, b16 = _t(device, shape, ttnn.bfloat16, mc, 3), _t(device, shape, ttnn.bfloat16, mc, 4)
    x = a8 if dt == "bfp8" else a16
    for _ in range(5):
        if before == "after_block_pack":
            ttnn.subtract(a16, b16, dtype=ttnn.bfloat16, memory_config=mc)
        elif before == "after_bfp8_out":
            ttnn.subtract(a8, b8, dtype=ttnn.bfloat8_b, memory_config=mc)
        ttnn.relu(x, memory_config=mc)
