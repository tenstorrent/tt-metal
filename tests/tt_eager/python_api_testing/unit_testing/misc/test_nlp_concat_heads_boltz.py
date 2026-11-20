# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, tt2torch_tensor

# nlp_concat_heads_boltz: [num_heads, S, S, head_dim] -> [1, S, S, num_heads * head_dim]


def concat_heads_boltz_ref(x):
    num_heads, s0, s1, head_dim = x.shape
    return x.permute(1, 2, 0, 3).reshape(1, s0, s1, num_heads * head_dim)


@pytest.mark.parametrize("num_heads, seq_len, head_dim", [(2, 32, 64), (4, 64, 64)])
def test_nlp_concat_heads_boltz(device, num_heads, seq_len, head_dim):
    torch.manual_seed(1234)

    spacers, input_addresses = [], set()
    num_entries_before = device.num_program_cache_entries()
    for i in range(3):
        # A growing live allocation moves every tensor below to a new address on each cache hit, and fresh
        # data per iteration makes a stale address show up as a mismatch.
        spacers.append(ttnn.Tensor(torch.zeros(1, 1, 32, 32 * (i + 1)), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device))
        x = torch.randn((num_heads, seq_len, seq_len, head_dim)).bfloat16().float()
        xt = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
        input_addresses.add(xt.buffer_address())
        out = ttnn.experimental.nlp_concat_heads_boltz(xt)

        assert list(out.padded_shape) == [1, seq_len, seq_len, num_heads * head_dim]
        assert torch.equal(tt2torch_tensor(out), concat_heads_boltz_ref(x))

    assert len(input_addresses) > 1
    assert device.num_program_cache_entries() - num_entries_before == 1


def run_nlp_concat_heads_boltz_interleaved(num_heads, seq_len, head_dim, dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)
    in0_shape = [num_heads, seq_len, seq_len, head_dim]
    torch_input = torch.randn(in0_shape)

    in0_t = ttnn.Tensor(torch_input, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)
    out = ttnn.experimental.nlp_concat_heads_boltz(in0_t, memory_config=out_mem_config)

    assert in0_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert out.memory_config().buffer_type == out_mem_config.buffer_type
    assert list(out.padded_shape) == [1, seq_len, seq_len, num_heads * head_dim]

    tt_out = tt2torch_tensor(out)
    ref_out = concat_heads_boltz_ref(torch_input)
    pcc = 0.99 if dtype == ttnn.bfloat8_b else 1.0
    passing_pcc, output_pcc = comp_pcc(tt_out, ref_out, pcc)
    logger.debug(f"passing={passing_pcc} output pcc={output_pcc}")
    assert passing_pcc


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "num_heads, seq_len, head_dim",
    (
        (2, 32, 64),
        (4, 32, 64),
        (4, 64, 64),  # issue #51284
    ),
)
def test_nlp_concat_heads_boltz_interleaved(
    num_heads, seq_len, head_dim, dtype, in0_mem_config, out_mem_config, device
):
    run_nlp_concat_heads_boltz_interleaved(num_heads, seq_len, head_dim, dtype, in0_mem_config, out_mem_config, device)


@pytest.mark.parametrize(
    "num_heads, seq_len, head_dim, grid_size",
    (
        (4, 64, 64, (1, 4)),  # 1 head/core, issue #51284
        (4, 32, 64, (1, 4)),
        (4, 32, 64, (1, 2)),  # 2 heads/core
    ),
)
def test_nlp_concat_heads_boltz_sharded(num_heads, seq_len, head_dim, grid_size, device):
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    torch.manual_seed(1234)
    num_cores = grid_size[0] * grid_size[1]
    in0_shape = [num_heads, seq_len, seq_len, head_dim]
    torch_input = torch.randn(in0_shape).bfloat16().float()

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    output_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0_t = ttnn.Tensor(torch_input, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, interleaved_mem_config)
    shard_height = num_heads * seq_len * seq_len // num_cores
    in0_t = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size,
        [shard_height, head_dim],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )

    out = ttnn.experimental.nlp_concat_heads_boltz(in0_t, memory_config=output_mem_config)
    out = ttnn.sharded_to_interleaved(out, interleaved_mem_config)

    assert list(out.padded_shape) == [1, seq_len, seq_len, num_heads * head_dim]
    tt_out = tt2torch_tensor(out)
    ref_out = concat_heads_boltz_ref(torch_input)
    passing_pcc, output_pcc = comp_pcc(tt_out, ref_out, 1.0)
    logger.info(output_pcc)
    assert passing_pcc


def test_nlp_concat_heads_boltz_sharded_rejects_partial_head_shard(device, expect_error):
    # 2 heads / 4 cores: shard height is half a head
    torch.manual_seed(1234)
    num_heads, seq_len, head_dim = 2, 32, 64
    grid_size = (1, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    num_cores = grid_size[0] * grid_size[1]
    in0_shape = [num_heads, seq_len, seq_len, head_dim]
    torch_input = torch.randn(in0_shape).bfloat16().float()
    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    in0_t = ttnn.Tensor(torch_input, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, interleaved_mem_config)
    shard_height = num_heads * seq_len * seq_len // num_cores
    in0_t = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size,
        [shard_height, head_dim],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )

    output_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    with expect_error(RuntimeError, "rows per head"):
        ttnn.experimental.nlp_concat_heads_boltz(in0_t, memory_config=output_mem_config)


def test_nlp_concat_heads_boltz_sharded_rejects_interleaved_output(device, expect_error):
    num_heads, seq_len, head_dim = 4, 32, 64
    grid_size = (1, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    num_cores = grid_size[0] * grid_size[1]
    torch_input = torch.randn([num_heads, seq_len, seq_len, head_dim]).bfloat16().float()
    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    in0_t = ttnn.Tensor(torch_input, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, interleaved_mem_config)
    in0_t = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size,
        [num_heads * seq_len * seq_len // num_cores, head_dim],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )

    with expect_error(RuntimeError, "Sharded input requires a sharded output"):
        ttnn.experimental.nlp_concat_heads_boltz(in0_t, memory_config=interleaved_mem_config)
