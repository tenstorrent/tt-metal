# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from math import prod

import pytest
import torch
import ttnn

from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program


@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK])
@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("output_memory", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("preallocate", [False, True])
def test_repeat_uneven_logical_width(device, strategy, rank, output_memory, preallocate):
    shape = [1] * (rank - 2) + [4, 7]
    repeats = [1] * rank
    repeats[-2] = 2
    layout = (
        ttnn.TensorMemoryLayout.WIDTH_SHARDED
        if strategy == ttnn.ShardStrategy.WIDTH
        else ttnn.TensorMemoryLayout.BLOCK_SHARDED
    )
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    input_memory = ttnn.MemoryConfig(
        layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, [4, 5], ttnn.ShardOrientation.ROW_MAJOR)
    )
    values = (torch.arange(prod(shape), dtype=torch.int64).reshape(shape) * 1000003 + 7).to(torch.int32)
    source = ttnn.from_torch(
        values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_memory
    )
    expected = values.to(torch.int64).repeat(*repeats)
    assert torch.equal(ttnn.to_torch(source).to(torch.int64), values.to(torch.int64))
    kwargs = {"memory_config": output_memory}
    if preallocate:
        target = ttnn.from_torch(
            torch.full(expected.shape, -1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=output_memory,
        )
        kwargs["optional_output_tensor"] = target
    result = ttnn.repeat(source, repeats, **kwargs)
    assert result.memory_config() == output_memory
    assert torch.equal(ttnn.to_torch(result).to(torch.int64), expected)
    if preallocate:
        assert result.buffer_address() == target.buffer_address()
        assert torch.equal(ttnn.to_torch(target).to(torch.int64), expected)


@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK])
@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize("output_memory", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("preallocate", [False, True])
def test_repeat_split_shards_to_interleaved(device, strategy, rank, output_memory, preallocate):
    shape = [1] * (rank - 2) + ([7, 40] if strategy == ttnn.ShardStrategy.WIDTH else [4, 10])
    grid = ttnn.CoreGrid(y=4, x=2) if strategy == ttnn.ShardStrategy.WIDTH else ttnn.CoreGrid(y=1, x=2)
    repeats = [1] * rank
    repeats[1] = 2
    input_memory = ttnn.create_sharded_memory_config(
        shape, core_grid=grid, strategy=strategy, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    values = (torch.arange(prod(shape), dtype=torch.int64).reshape(shape) * 1000003 + 7).to(torch.int32)
    source = ttnn.from_torch(
        values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_memory
    )
    expected = values.to(torch.int64).repeat(*repeats)
    kwargs = {"memory_config": output_memory}
    if preallocate:
        target = ttnn.from_torch(
            torch.full(expected.shape, -1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=output_memory,
        )
        kwargs["optional_output_tensor"] = target
    if ttnn.device.IsProgramRealtimeProfilerActive():
        result, records = profile_realtime_program(
            device, lambda: ttnn.repeat(source, repeats, **kwargs), collect_all=True
        )
        sources = {
            path.rsplit("/", 1)[-1] for record in records for path in record["kernel_sources"] if "/repeat/" in path
        }
        expected_kernel = "repeat_higher_dim_rm_sharded_staged.cpp" if rank == 5 else "reader_repeat_higherdim_rm.cpp"
        assert sources == {expected_kernel}, sources
    else:
        result = ttnn.repeat(source, repeats, **kwargs)
    assert result.memory_config() == output_memory
    assert torch.equal(ttnn.to_torch(result).to(torch.int64), expected)
    if preallocate:
        assert result.buffer_address() == target.buffer_address()
        assert torch.equal(ttnn.to_torch(target).to(torch.int64), expected)


@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK])
@pytest.mark.parametrize("shard_width", [5, 12])
def test_repeat_split_row_kernel(device, strategy, shard_width, record_property):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.skip("Kernel-source observation requires the real-time profiler")
    shape = [1, 4, 2 * shard_width]
    grid = ttnn.CoreGrid(y=1, x=2)
    placement = ttnn.create_sharded_memory_config(
        shape, core_grid=grid, strategy=strategy, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    output_placement = ttnn.create_sharded_memory_config(
        [1, 8, 2 * shard_width], core_grid=grid, strategy=strategy, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    values = torch.arange(prod(shape), dtype=torch.int32).reshape(shape)
    source = ttnn.from_torch(
        values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=placement
    )
    result, records = profile_realtime_program(device, lambda: ttnn.repeat(source, [1, 2, 1]), collect_all=True)
    sources = {path for record in records for path in record["kernel_sources"] if "/repeat/" in path}
    kernel = "repeat_higher_dim_rm_sharded_staged.cpp" if shard_width == 5 else "repeat_higher_dim_rm_sharded.cpp"
    assert {path.rsplit("/", 1)[-1] for path in sources} == {kernel}, sources
    record_property("repeat_kernel_sources", sorted(sources))
    assert result.memory_config() == output_placement
    assert torch.equal(ttnn.to_torch(result).to(torch.int64), values.to(torch.int64).repeat(1, 2, 1))
