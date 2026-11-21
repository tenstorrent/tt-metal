# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design asks for beyond the acceptance test: the thread plan ran, the block knob, the caller's
core counts (R10), and the op in one device session with tt-metal programs before and after it."""

import pytest
import torch

import ttnn
from ttnn.operations.padded_slice import padded_slice
from ttnn.operations.padded_slice.padded_slice import _dispatch, validate
from ttnn.operations.padded_slice.padded_slice_program_artifacts import create_program_artifacts, K_GATHER

SENTINEL = 1.0e4


def _height_config(device, rows, width, cores):
    grid = device.compute_with_storage_grid_size()
    assert rows % cores == 0
    return ttnn.create_sharded_memory_config(
        shape=(rows // cores, width),
        core_grid=ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _expected(torch_input, start, end):
    block = torch_input[start[0] : end[0], start[1] : end[1], start[2] : end[2], start[3] : end[3]]
    return block.reshape(1, 1, -1, end[3] - start[3])


def _input(device, shape, seed=0):
    torch.manual_seed(seed)
    t = torch.randn(shape, dtype=torch.bfloat16)
    return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _sentinel_then_op(device, tt_input, start, end, config, **knobs):
    """Fill the shard memory the result will take with a sentinel first, so a row no hart wrote shows."""
    rows = 1
    for s, e in zip(start[:3], end[:3]):
        rows *= e - s
    width = int(config.shard_spec.shape[1])
    filler = ttnn.from_torch(torch.full((1, 1, rows, width), SENTINEL, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=config)
    address = filler.buffer_address()
    ttnn.deallocate(filler)
    validate(tt_input, start, end, [1, 1, 1, 1], memory_config=config)
    out = _dispatch(tt_input, tuple(start), tuple(end), config, **knobs)
    assert out.buffer_address() == address, "the result did not reuse the sentinel's memory: the test shows nothing"
    return out


@pytest.mark.parametrize("threads", [1, 2, 3, 6])
@pytest.mark.parametrize(
    "shape,start,end,cores",
    [
        ((1, 64, 32, 32), (0, 0, 0, 0), (1, 32, 32, 32), 32),  # 32 rows per core
        ((1, 17, 50, 64), (0, 3, 7, 0), (1, 12, 40, 64), 1),  # odd w-runs, 297 rows on one core
        ((1, 64, 32, 32), (0, 0, 0, 0), (1, 32, 32, 32), 2),  # a traced config: 512 rows on two cores
    ],
    ids=["32cores", "odd_1core", "traced_2cores"],
)
def test_every_hart_writes_its_rows(device, shape, start, end, cores, threads):
    """Each hart's loop is sized from the plan's count: a hart that did not run leaves sentinel rows."""
    torch_input, tt_input = _input(device, shape)
    rows = 1
    for s, e in zip(start[:3], end[:3]):
        rows *= e - s
    config = _height_config(device, rows, end[3] - start[3], cores)
    out = _sentinel_then_op(device, tt_input, start, end, config, gather_threads=threads)
    got = ttnn.to_torch(out)
    assert torch.equal(got, _expected(torch_input, start, end)), (got == SENTINEL).sum()


@pytest.mark.parametrize("threads", [1, 2, 6])
def test_the_kernel_has_the_plan_threads(device, threads):
    """The built program's gather kernel runs `threads` harts."""
    _, tt_input = _input(device, (1, 64, 32, 32))
    config = _height_config(device, 1024, 32, 32)
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape([1, 1, 1024, 32]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, config), device)
    spec, _, _ = create_program_artifacts(tt_input, out, (0, 0, 0, 0), (1, 32, 32, 32), gather_threads=threads)
    (kernel,) = [k for k in spec.kernels if k.unique_id == K_GATHER]
    assert kernel.num_threads == threads


@pytest.mark.parametrize("block_rows", [1, 5, 7, 64])
@pytest.mark.parametrize("threads", [2, 3])
def test_block_knob(device, block_rows, threads):
    """Blocks smaller than the shard, including a ragged last block and blocks with fewer rows than harts."""
    shape, start, end = (1, 17, 50, 64), (0, 3, 7, 0), (1, 12, 40, 64)  # 297 rows, 27 cores of 11
    torch_input, tt_input = _input(device, shape)
    config = _height_config(device, 297, 64, 27)
    out = _sentinel_then_op(device, tt_input, start, end, config, gather_threads=threads, max_block_rows=block_rows)
    assert torch.equal(ttnn.to_torch(out), _expected(torch_input, start, end))


def test_traced_stem_call_one_core(device):
    """The ResNet-50 stem call: 6,785 rows of 32 on one core (R10)."""
    shape, start, end = (1, 115, 115, 32), (0, 0, 0, 0), (1, 59, 115, 32)
    torch_input, tt_input = _input(device, shape)
    config = _height_config(device, 6785, 32, 1)
    out = padded_slice(tt_input, list(start), list(end), [1, 1, 1, 1], memory_config=config)
    assert out.memory_config() == config
    assert torch.equal(ttnn.to_torch(out), _expected(torch_input, start, end))


def test_one_device_session_with_tt_metal_programs(device):
    """A tt-metal program before the op and one after it, in one device session (target page, ring protocols):
    the op's harts must neither inherit a ring's tag nor leave one for untagged transfers."""
    torch.manual_seed(1)
    tile = torch.randn((1, 1, 64, 128), dtype=torch.bfloat16)

    def tt_metal_program():
        tiled = ttnn.from_torch(tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        rm = ttnn.to_layout(tiled, ttnn.ROW_MAJOR_LAYOUT)
        assert torch.equal(ttnn.to_torch(rm), tile)

    shape, start, end = (1, 64, 32, 32), (0, 0, 0, 0), (1, 32, 32, 32)
    for seed in range(2):
        tt_metal_program()
        torch_input, tt_input = _input(device, shape, seed)
        out = padded_slice(tt_input, start, end, [1, 1, 1, 1], memory_config=_height_config(device, 1024, 32, 32))
        assert torch.equal(ttnn.to_torch(out), _expected(torch_input, start, end))
        # A tt-metal program on the op's own output, then another unrelated one.
        back = ttnn.to_torch(ttnn.to_layout(ttnn.to_layout(out, ttnn.TILE_LAYOUT), ttnn.ROW_MAJOR_LAYOUT))
        assert torch.equal(back, _expected(torch_input, start, end))
        tt_metal_program()
