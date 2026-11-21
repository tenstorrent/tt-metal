# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 0: both operands height-sharded in L1 (the YOLOv8 bottleneck add).

1. R9: the result of a height-sharded call is height-sharded in L1 with the first operand's shard shape and
   cores (a same-shape result allocated with a's memory config);
2. the op does not depend on the shard grid: operands sharded over one core, two cores, the full grid, a grid
   offset from the cores the op works on, and operands sharded over different grids;
3. the thread plan (4, 4, 2) on sharded operands into a NaN-prefilled sharded output (every thread wrote);
4. one device session with tt-metal programs on sharded tensors: interleaved_to_sharded before add, add twice,
   to_layout and sharded_to_interleaved of add's sharded result after.
"""

import pytest
import torch

import ttnn
from ttnn.operations.add import add, default_compute_kernel_config
from ttnn.operations.add.add_program_artifacts import create_program_artifacts

from eval.memory import build, memory_axes, place


def _height_config(rows_per_shard, width, cores):
    return ttnn.create_sharded_memory_config(
        shape=(rows_per_shard, width),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _first_cores(device, n):
    return ttnn.num_cores_to_corerangeset(n, device.compute_with_storage_grid_size(), row_wise=True)


def _put(t, device, memory_config):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _expected(ta, tb):
    # The fp32 DEST sum of two bf16 values, rounded once to bf16.
    return (ta.float() + tb.float()).to(torch.bfloat16).float()


def _check(got, ta, tb):
    exp = _expected(ta, tb)
    assert not torch.isnan(got).any(), "NaN pages left: some thread did not write its pages"
    assert torch.allclose(got, exp, rtol=1e-2, atol=1e-2), (got - exp).abs().max()


def _assert_sharded_like(out, a):
    """R9: height-sharded in L1 with a's shard shape and cores."""
    mc, amc = out.memory_config(), a.memory_config()
    assert memory_axes(out) == {"memory_config": "L1", "sharding": "height"}
    assert mc.buffer_type == ttnn.BufferType.L1
    assert mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert list(mc.shard_spec.shape) == list(amc.shard_spec.shape)
    assert mc.shard_spec.grid == amc.shard_spec.grid
    assert mc.shard_spec.orientation == amc.shard_spec.orientation


def _inputs(shape, seed=0):
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.bfloat16), torch.randn(*shape, dtype=torch.bfloat16)


# The runner's placement (eval/memory.py, as the golden suite builds it): the traced YOLOv8 call
# ([1, 1, 64, 128], two cores of [32, 128]), a non-aligned W, a non-aligned H, rank 2.
@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (1, 2, 400), (1, 1, 50, 128), (128, 100)], ids=str)
def test_result_is_sharded_like_a(device, shape):
    cfg = place(build("L1", "height"), shape, ttnn.TILE_LAYOUT, device)
    ta, tb = _inputs(shape)
    a, b = _put(ta, device, cfg), _put(tb, device, cfg)
    out = add(a, b)
    _assert_sharded_like(out, a)
    assert list(out.shape) == list(shape)
    _check(ttnn.to_torch(out).float(), ta, tb)


SHAPE = (1, 1, 1024, 64)  # 32 tile rows: shardable over 1, 2, ..., 32 cores


def _grids(device):
    grid = device.compute_with_storage_grid_size()
    full = grid.x * grid.y
    offset = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(grid.x - 2, grid.y - 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    return {"one_core": (1, _first_cores(device, 1)), "two_cores": (2, _first_cores(device, 2)),
            "full_grid": (full, _first_cores(device, full)), "offset_two_cores": (2, offset)}


@pytest.mark.parametrize("grid_name", ["one_core", "two_cores", "full_grid", "offset_two_cores"])
def test_shard_grid_independence(device, grid_name):
    n, cores = _grids(device)[grid_name]
    rows = SHAPE[-2] // n
    cfg = _height_config(rows, SHAPE[-1], cores)
    ta, tb = _inputs(SHAPE, seed=3)
    a, b = _put(ta, device, cfg), _put(tb, device, cfg)
    out = add(a, b)
    _assert_sharded_like(out, a)
    _check(ttnn.to_torch(out).float(), ta, tb)


def test_operands_on_different_shard_grids(device):
    """a over two cores, b over the full grid: the result follows a (R9) and the values are right."""
    grids = _grids(device)
    n_b, cores_b = grids["full_grid"]
    ta, tb = _inputs(SHAPE, seed=4)
    a = _put(ta, device, _height_config(SHAPE[-2] // 2, SHAPE[-1], grids["two_cores"][1]))
    b = _put(tb, device, _height_config(SHAPE[-2] // n_b, SHAPE[-1], cores_b))
    out = add(a, b)
    _assert_sharded_like(out, a)
    _check(ttnn.to_torch(out).float(), ta, tb)


@pytest.mark.parametrize("threads", [(1, 1, 1), (4, 4, 2)], ids=lambda p: "R%dC%dW%d" % p)
def test_thread_plan_sharded(device, threads):
    shape = (1, 3, 320, 320)  # 300 tiles over the full grid: every NEO of a C=4 plan gets pages
    cfg = place(build("L1", "height"), shape, ttnn.TILE_LAYOUT, device)
    ta, tb = _inputs(shape, seed=5)
    a, b = _put(ta, device, cfg), _put(tb, device, cfg)
    out = _put(torch.full(shape, float("nan"), dtype=torch.bfloat16), device, cfg)
    spec, run_args, idx = create_program_artifacts(a, b, out, default_compute_kernel_config(), threads=threads)
    res = ttnn.generic_op([a, b, out], spec, run_args, idx)
    _check(ttnn.to_torch(res).float(), ta, tb)


def test_one_device_session_sharded(device):
    """tt-metal programs on sharded tensors before and after add, in one device session."""
    shape = (1, 1, 64, 128)  # the traced YOLOv8 call
    cfg = place(build("L1", "height"), shape, ttnn.TILE_LAYOUT, device)
    ta, tb = _inputs(shape, seed=6)
    # before: tt-metal's own program makes the sharded operands
    a = ttnn.interleaved_to_sharded(_put(ta, device, ttnn.DRAM_MEMORY_CONFIG), cfg)
    b = ttnn.interleaved_to_sharded(_put(tb, device, ttnn.DRAM_MEMORY_CONFIG), cfg)
    out1 = add(a, b)
    out2 = add(out1, b)
    _assert_sharded_like(out2, a)
    # after: tt-metal's own programs on add's sharded result, issuing untagged NoC transfers
    out_rm = ttnn.to_layout(out2, ttnn.ROW_MAJOR_LAYOUT)
    out_il = ttnn.sharded_to_interleaved(out1, ttnn.L1_MEMORY_CONFIG)
    exp2 = (_expected(ta, tb) + tb.float()).to(torch.bfloat16).float()
    got2 = ttnn.to_torch(ttnn.from_device(out_rm)).float()
    assert torch.allclose(got2, exp2, rtol=1e-2, atol=1e-2), (got2 - exp2).abs().max()
    _check(ttnn.to_torch(out_il).float(), ta, tb)
    # and an interleaved add after the sharded ones, in the same session
    tc = torch.randn(shape, dtype=torch.bfloat16)
    c = _put(tc, device, ttnn.DRAM_MEMORY_CONFIG)
    out3 = add(c, c)
    assert out3.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    _check(ttnn.to_torch(out3).float(), tc, tc)


def test_block_sharding_refused(device):
    """R10: a sharding the op does not support (block) is refused by name. (Width became supported in
    Refinement 20: test_add_models.py.)"""
    shape = (1, 1, 64, 128)
    try:
        cfg = ttnn.create_sharded_memory_config(
            shape=(32, 64), core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))]),
            strategy=ttnn.ShardStrategy.BLOCK, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        ta, tb = _inputs(shape)
        a, b = _put(ta, device, cfg), _put(tb, device, cfg)
    except Exception as e:  # a placement this device cannot build is not the op's to refuse
        pytest.skip(f"block-sharded tensor not buildable here: {e}")
    with pytest.raises(NotImplementedError, match="sharding"):
        add(a, b)
