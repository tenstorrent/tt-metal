# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_program_cache_flat, assert_with_pcc


@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize("sequence", [1, 2])
@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [64, 32])
def test_concatenate_heads(device, batch, sequence, height, width):
    torch_input_tensor = torch.rand((batch, sequence, height, width), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.transformer.concatenate_heads)
    torch_output_tensor = golden_function(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.transformer.concatenate_heads(input_tensor)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("batch, sequence, height, width", [(1, 2, 32, 64), (5, 1, 32, 32)])
def test_concatenate_heads_program_cache(device, batch, sequence, height, width):
    torch.manual_seed(0)
    golden_function = ttnn.get_golden_function(ttnn.transformer.concatenate_heads)

    spacers, input_addresses = [], set()
    num_entries_before = device.num_program_cache_entries()
    for i in range(3):
        # A growing live allocation moves every tensor below to a new address on each cache hit, and fresh
        # data per iteration makes a stale address show up as a mismatch.
        spacers.append(ttnn.from_torch(torch.zeros((32, 32 * (i + 1))), layout=ttnn.TILE_LAYOUT, device=device))
        torch_input_tensor = torch.rand((batch, sequence, height, width), dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        input_addresses.add(input_tensor.buffer_address())
        output = ttnn.to_torch(ttnn.transformer.concatenate_heads(input_tensor))

        assert torch.equal(golden_function(torch_input_tensor), output)

    assert len(input_addresses) > 1
    assert device.num_program_cache_entries() - num_entries_before == 1


def _h3_concat_rows():
    # Per-chip packed rows served by the MiniMax-H3 DiT on 4x8 (74..470 tile rows), 148 as the warm point.
    return [148 * 32, 74 * 32, 287 * 32, 427 * 32, 470 * 32]


@pytest.mark.parametrize("batch", [1, 2])
def test_concatenate_heads_h3_dynamic_rows_flat_cache(device, batch):
    """H3 concat ([B,14,M,128] -> [B,M,1792]) with runtime_rows: one program over every served M, bit-exact per length."""
    torch.manual_seed(0)
    golden_function = ttnn.get_golden_function(ttnn.transformer.concatenate_heads)

    def run(rows):
        torch_input_tensor = torch.rand((batch, 14, rows, 128), dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        output = ttnn.to_torch(ttnn.transformer.concatenate_heads(input_tensor, runtime_rows=True))
        assert torch.equal(golden_function(torch_input_tensor), output), f"mismatch at rows={rows}"

    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    assert_program_cache_flat(device, run, _h3_concat_rows())
    assert device.num_program_cache_entries() == 1


def test_concatenate_heads_width_or_dtype_change_misses(device):
    """With runtime_rows only the row count is runtime: a new head size, head count or dtype needs its own program."""
    torch.manual_seed(0)
    golden_function = ttnn.get_golden_function(ttnn.transformer.concatenate_heads)

    def run(variant):
        heads, rows, width, dtype = variant
        torch_input_tensor = torch.rand((1, heads, rows, width), dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
        output = ttnn.to_torch(ttnn.transformer.concatenate_heads(input_tensor, runtime_rows=True))
        expected = golden_function(ttnn.to_torch(input_tensor))
        assert torch.equal(expected, output), f"mismatch for {variant}"

    variants = [
        (14, 256, 128, ttnn.bfloat16),
        (14, 64, 128, ttnn.bfloat16),
        (14, 256, 64, ttnn.bfloat16),  # head size: new program
        (8, 256, 128, ttnn.bfloat16),  # head count: new program
        (14, 256, 128, ttnn.bfloat8_b),  # dtype: new program
    ]
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    assert_program_cache_flat(device, run, variants, miss_indices=(2, 3, 4))


def test_concatenate_heads_crosses_head_parallel_threshold(device):
    """With runtime_rows fewer tile rows than cores splits by head on device; both sides of the threshold share one program."""
    torch.manual_seed(0)
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    golden_function = ttnn.get_golden_function(ttnn.transformer.concatenate_heads)

    def run(rows):
        torch_input_tensor = torch.rand((1, 4, rows, 64), dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        output = ttnn.to_torch(ttnn.transformer.concatenate_heads(input_tensor, runtime_rows=True))
        assert torch.equal(golden_function(torch_input_tensor), output), f"mismatch at rows={rows}"

    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    assert_program_cache_flat(device, run, [32 * num_cores, 32, 32 * (num_cores - 1), 32 * (num_cores + 3)])
    assert device.num_program_cache_entries() == 1


def test_concatenate_heads_rows_change_misses_by_default(device):
    """Without runtime_rows every row count builds its own program (the default path), bit-exact each."""
    torch.manual_seed(0)
    golden_function = ttnn.get_golden_function(ttnn.transformer.concatenate_heads)

    def run(rows):
        torch_input_tensor = torch.rand((1, 14, rows, 128), dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        output = ttnn.to_torch(ttnn.transformer.concatenate_heads(input_tensor))
        assert torch.equal(golden_function(torch_input_tensor), output), f"mismatch at rows={rows}"

    rows = [128, 32, 256, 416]
    assert_program_cache_flat(device, run, rows, miss_indices=range(1, len(rows)))
    assert device.num_program_cache_entries() == len(rows)


def test_concatenate_heads_runtime_rows_ignored_on_sharded(device):
    """The sharded program keeps its full-spec hash: with runtime_rows a row change still builds a new program."""
    torch.manual_seed(0)
    heads, head_dim = 32, 64
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))])
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)

    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    for rows in (32, 64):
        torch_input_tensor = torch.rand((1, heads, rows, head_dim), dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            torch_input_tensor,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [heads * rows // 4, head_dim], ttnn.ShardOrientation.COL_MAJOR),
            ),
        )
        entries_before = device.num_program_cache_entries()
        output = ttnn.experimental.nlp_concat_heads(input_tensor, memory_config=output_memory_config, runtime_rows=True)
        assert device.num_program_cache_entries() == entries_before + 1, f"rows={rows} reused a program"
        output = ttnn.to_torch(ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG))
        expected = torch_input_tensor.transpose(1, 2).reshape(1, 1, rows, heads * head_dim)
        assert torch.equal(expected, output), f"mismatch at rows={rows}"


def test_concatenate_heads_sharded_input_interleaved_output(device):
    """Regression test for hang when input is height-sharded L1 and output is DRAM interleaved.
    See https://github.com/tenstorrent/tt-metal/issues/40925
    """
    torch_input_tensor = torch.randn(32, 32, 18, 64, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.transformer.concatenate_heads)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )

    input_tensor = ttnn.to_memory_config(
        input_tensor,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [512, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )

    output = ttnn.transformer.concatenate_heads(
        input_tensor,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
