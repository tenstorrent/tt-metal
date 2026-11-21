# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 20: the Llama and Qwen calls.

1. `memory_config=` the first operand's own config (R11): same result and placement as leaving it out (R9);
   any other config is refused by name;
2. `dtype=ttnn.bfloat16` on bfloat16 operands (R8);
3. both operands width-sharded in L1: the result width-sharded with a's shard shape, cores and orientation (R9),
   on the runner's placement and on explicit grids (one core, two cores, an offset pair, a and b on different
   grids), with the thread plan (4, 4, 2) into a NaN-prefilled output, and in one device session with
   tt-metal programs before and after;
4. a second operand of lower rank that equals the first once leading 1s are added (`broadcast` = "none"),
   in every supported placement, and the first operand of lower rank (the output has the broadcast shape).
"""

import pytest
import torch

import ttnn
from ttnn.operations.add import add, default_compute_kernel_config
from ttnn.operations.add.add_program_artifacts import create_program_artifacts

from eval.memory import build, memory_axes, place


def _put(t, device, memory_config):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _expected(ta, tb):
    # The fp32 DEST sum of two bf16 values, rounded once to bf16.
    return (ta.float() + tb.float()).to(torch.bfloat16).float()


def _within_one_ulp(got, exp):
    """The specification's bfloat16 requirement: <= 1 bf16 ulp of the single-rounded sum (the packer's rounding
    differs from torch's on rare elements)."""
    ulp = torch.where(exp == 0, torch.tensor(2.0**-133), 2.0 ** (torch.floor(torch.log2(exp.abs())) - 7))
    return bool(((got - exp).abs() <= ulp).all())


def _check(out, ta, tb):
    got, exp = ttnn.to_torch(out).float(), _expected(ta, tb)
    assert list(got.shape) == list(exp.shape)
    assert not torch.isnan(got).any(), "NaN pages left: some thread did not write its pages"
    assert _within_one_ulp(got, exp), (got - exp).abs().max()


def _width_config(rows, cols_per_shard, cores):
    return ttnn.create_sharded_memory_config(
        shape=(rows, cols_per_shard), core_grid=cores, strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
    )


def _first_cores(device, n):
    return ttnn.num_cores_to_corerangeset(n, device.compute_with_storage_grid_size(), row_wise=True)


def _assert_placed_like(out, a):
    """R9: the result has a's buffer, memory layout, and (sharded) a's shard shape, cores and orientation."""
    mc, amc = out.memory_config(), a.memory_config()
    assert memory_axes(out) == memory_axes(a)
    assert mc.buffer_type == amc.buffer_type and mc.memory_layout == amc.memory_layout
    if amc.shard_spec is not None:
        assert list(mc.shard_spec.shape) == list(amc.shard_spec.shape)
        assert mc.shard_spec.grid == amc.shard_spec.grid
        assert mc.shard_spec.orientation == amc.shard_spec.orientation


def _inputs(shape_a, shape_b, seed=0):
    torch.manual_seed(seed)
    return torch.randn(*shape_a, dtype=torch.bfloat16), torch.randn(*shape_b, dtype=torch.bfloat16)


# --- the models' calls ------------------------------------------------------------------------------------------


def test_llama_dram_memory_config(device):
    """Llama: add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG) on DRAM operands -> a DRAM interleaved result."""
    shape = (1, 1, 32, 2048)
    ta, tb = _inputs(shape, shape)
    a, b = _put(ta, device, ttnn.DRAM_MEMORY_CONFIG), _put(tb, device, ttnn.DRAM_MEMORY_CONFIG)
    out = add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    _check(out, ta, tb)


def test_qwen_width_sharded_with_memory_config_and_dtype(device):
    """Qwen: two [1, 1, 32, 256] operands width-sharded over two cores (shard [32, 128]),
    add(a, b, memory_config=<their config>, dtype=ttnn.bfloat16) -> width-sharded the same way."""
    shape = (1, 1, 32, 256)
    cfg = _width_config(32, 128, _first_cores(device, 2))
    ta, tb = _inputs(shape, shape, seed=1)
    a, b = _put(ta, device, cfg), _put(tb, device, cfg)
    out = add(a, b, memory_config=a.memory_config(), dtype=ttnn.bfloat16)
    mc = out.memory_config()
    assert mc.buffer_type == ttnn.BufferType.L1
    assert mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert list(mc.shard_spec.shape) == [32, 128]
    assert mc.shard_spec.grid == _first_cores(device, 2)
    assert out.dtype == ttnn.bfloat16
    _check(out, ta, tb)


@pytest.mark.parametrize("placement", ["DRAM", "L1", "height", "width"])
@pytest.mark.parametrize("shape_b", [(128, 512), (1, 128, 512)], ids=str)
def test_qwen_lower_rank_b(device, placement, shape_b):
    """Qwen: [1, 1, 128, 512] + [128, 512] (and a rank-3 b), served like two operands of one shape,
    in every supported placement (runner-sized shards, as the golden suite builds them)."""
    shape = (1, 1, 128, 512)
    buffer, sharding = (placement, "interleaved") if placement in ("DRAM", "L1") else ("L1", placement)
    ta, tb = _inputs(shape, shape_b, seed=2)
    a = _put(ta, device, place(build(buffer, sharding), shape, ttnn.TILE_LAYOUT, device))
    b = _put(tb, device, place(build(buffer, sharding), shape_b, ttnn.TILE_LAYOUT, device))
    out = add(a, b)
    _assert_placed_like(out, a)
    assert list(out.shape) == list(shape)
    _check(out, ta, tb)


def test_lower_rank_a_output_has_broadcast_shape(device):
    """The first operand of lower rank: the output has the broadcast (rank-4) shape, placed where a is."""
    shape_a, shape_b = (64, 128), (1, 1, 64, 128)
    ta, tb = _inputs(shape_a, shape_b, seed=3)
    a, b = _put(ta, device, ttnn.L1_MEMORY_CONFIG), _put(tb, device, ttnn.L1_MEMORY_CONFIG)
    out = add(a, b)
    assert list(out.shape) == list(shape_b)
    _assert_placed_like(out, a)
    _check(out, ta, tb)


# --- memory_config (R11) and dtype (R8) -------------------------------------------------------------------------


@pytest.mark.parametrize("placement", ["DRAM", "L1", "height", "width"])
def test_memory_config_input_equals_none(device, placement):
    shape = (1, 1, 64, 128)
    buffer, sharding = (placement, "interleaved") if placement in ("DRAM", "L1") else ("L1", placement)
    cfg = place(build(buffer, sharding), shape, ttnn.TILE_LAYOUT, device)
    ta, tb = _inputs(shape, shape, seed=4)
    a, b = _put(ta, device, cfg), _put(tb, device, cfg)
    out_none, out_input = add(a, b), add(a, b, memory_config=a.memory_config(), dtype=ttnn.bfloat16)
    _assert_placed_like(out_none, a)
    _assert_placed_like(out_input, a)
    assert torch.equal(ttnn.to_torch(out_none), ttnn.to_torch(out_input))
    _check(out_input, ta, tb)


def test_other_memory_config_refused(device):
    """R11: a config that is not a's own reads as "other" and is refused by name."""
    shape = (1, 1, 32, 64)
    ta, tb = _inputs(shape, shape)
    a, b = _put(ta, device, ttnn.DRAM_MEMORY_CONFIG), _put(tb, device, ttnn.DRAM_MEMORY_CONFIG)
    with pytest.raises(NotImplementedError, match="memory_argument"):
        add(a, b, memory_config=ttnn.L1_MEMORY_CONFIG)
    with pytest.raises(NotImplementedError, match="output_dtype"):
        add(a, b, dtype=ttnn.float32)


# --- width sharding: the shard grid, the thread plan, one device session ----------------------------------------

SHAPE = (1, 1, 64, 1024)  # 32 tile columns: width-shardable over 1, 2, ..., 32 cores


def _grids(device):
    grid = device.compute_with_storage_grid_size()
    full = grid.x * grid.y
    offset = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(grid.x - 2, grid.y - 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    return {"one_core": (1, _first_cores(device, 1)), "two_cores": (2, _first_cores(device, 2)),
            "full_grid": (full, _first_cores(device, full)), "offset_two_cores": (2, offset)}


@pytest.mark.parametrize("grid_name", ["one_core", "two_cores", "full_grid", "offset_two_cores"])
def test_width_shard_grid_independence(device, grid_name):
    n, cores = _grids(device)[grid_name]
    cfg = _width_config(SHAPE[-2], SHAPE[-1] // n, cores)
    ta, tb = _inputs(SHAPE, SHAPE, seed=5)
    a, b = _put(ta, device, cfg), _put(tb, device, cfg)
    out = add(a, b)
    _assert_placed_like(out, a)
    _check(out, ta, tb)


def test_width_operands_on_different_shard_grids(device):
    grids = _grids(device)
    n_b, cores_b = grids["full_grid"]
    ta, tb = _inputs(SHAPE, SHAPE, seed=6)
    a = _put(ta, device, _width_config(SHAPE[-2], SHAPE[-1] // 2, grids["two_cores"][1]))
    b = _put(tb, device, _width_config(SHAPE[-2], SHAPE[-1] // n_b, cores_b))
    out = add(a, b)
    _assert_placed_like(out, a)
    _check(out, ta, tb)


@pytest.mark.parametrize("threads", [(1, 1, 1), (4, 4, 2)], ids=lambda p: "R%dC%dW%d" % p)
def test_thread_plan_width_sharded(device, threads):
    shape = (1, 3, 320, 320)  # 300 tiles: every NEO of a C=4 plan gets pages
    cfg = place(build("L1", "width"), shape, ttnn.TILE_LAYOUT, device)
    ta, tb = _inputs(shape, shape, seed=7)
    a, b = _put(ta, device, cfg), _put(tb, device, cfg)
    out = _put(torch.full(shape, float("nan"), dtype=torch.bfloat16), device, cfg)
    spec, run_args, idx = create_program_artifacts(a, b, out, default_compute_kernel_config(), threads=threads)
    _check(ttnn.generic_op([a, b, out], spec, run_args, idx), ta, tb)


def test_one_device_session_width_sharded(device):
    """tt-metal programs on width-sharded tensors before and after add, in one device session."""
    shape = (1, 1, 32, 256)
    cfg = _width_config(32, 128, _first_cores(device, 2))
    ta, tb = _inputs(shape, shape, seed=8)
    a = ttnn.interleaved_to_sharded(_put(ta, device, ttnn.DRAM_MEMORY_CONFIG), cfg)
    b = ttnn.interleaved_to_sharded(_put(tb, device, ttnn.DRAM_MEMORY_CONFIG), cfg)
    out1 = add(a, b, memory_config=cfg, dtype=ttnn.bfloat16)
    out2 = add(out1, b)
    _assert_placed_like(out2, a)
    out_rm = ttnn.to_layout(out2, ttnn.ROW_MAJOR_LAYOUT)
    out_il = ttnn.sharded_to_interleaved(out1, ttnn.L1_MEMORY_CONFIG)
    exp2 = (_expected(ta, tb) + tb.float()).to(torch.bfloat16).float()
    assert _within_one_ulp(ttnn.to_torch(ttnn.from_device(out_rm)).float(), exp2)
    _check(out_il, ta, tb)
