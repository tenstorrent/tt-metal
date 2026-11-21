# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The tests op_design.md's Work Distribution and Regimes ask for, beside the acceptance test:

- every NEO of an engaged core owns a real stream tile, and a missing thread would leave rows unwritten;
- the op in one device session after a program of tt-metal's own and before one (ring tag hand-back);
- rule R10: sharded inputs on one core, and a pool of a pool's result (height- and width-sharded);
- one cell per regime: staged-resident (interleaved, height-sharded), in-place-shard with staged spare-core
  sub-ranges (width-sharded), staged-streamed (forced with _STAGING_BUDGET_OVERRIDE), in both layouts;
- the kernels under other thread plans (R < C, W < C, one thread of each kind).
"""

import pytest
import torch

import ttnn
from ttnn.operations.max_pool2d import max_pool2d
from ttnn.operations.max_pool2d import max_pool2d_program_artifacts as artifacts
from ttnn.operations.max_pool2d import default_compute_kernel_config
from ttnn.operations.max_pool2d.max_pool2d import validate


def reference(t, n, h, w, c, k, s, p, d=(1, 1)):
    images = t.double().reshape(n, h, w, c).permute(0, 3, 1, 2)
    pooled = torch.nn.functional.max_pool2d(images, k, stride=s, padding=p, dilation=d)
    N, C, Ho, Wo = pooled.shape
    return pooled.permute(0, 2, 3, 1).reshape(1, 1, N * Ho * Wo, C)


def draw(n, h, w, c, *, seed=0, negative=False):
    torch.manual_seed(seed)
    t = torch.rand(1, 1, n * h * w, c) - 1.0 if negative else torch.randn(1, 1, n * h * w, c)
    return t.to(torch.bfloat16)


def sharded_config(strategy, shard_shape, num_cores, device):
    grid = device.compute_with_storage_grid_size()
    return ttnn.create_sharded_memory_config(
        shape=shard_shape, core_grid=ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True),
        strategy=strategy, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)


def put(t, device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    x = ttnn.from_torch(t, layout=layout, device=device, dtype=ttnn.bfloat16)
    return ttnn.to_memory_config(x, memory_config) if memory_config is not ttnn.DRAM_MEMORY_CONFIG else x


def run_and_check(x, t, args):
    out = max_pool2d(x, *args)
    got = ttnn.to_torch(out).double()
    expected = reference(t, *args)
    assert got.shape == expected.shape, (got.shape, expected.shape)
    mismatches = (got != expected).sum().item()
    assert mismatches == 0, f"{mismatches} elements differ, max |diff| {(got - expected).abs().max().item()}"
    return out


def assignments_of(x, args):
    """The host plan of a call: its assignments, with tiles and block sizes filled in as the program has them."""
    geo = validate(x, *args)
    plan = artifacts.plan_output(x, geo, x.device())
    out = ttnn.allocate_tensor_on_device(plan.spec, x.device())
    _, run_args, _ = artifacts.create_program_artifacts(x, out, geo, plan, default_compute_kernel_config())
    ttnn.deallocate(out)
    return run_args


SPPF = (1, 20, 20, 256, (5, 5), (1, 1), (2, 2), (1, 1))
RESNET = (1, 32, 32, 64, (3, 3), (2, 2), (1, 1), (1, 1))


def test_every_neo_owns_a_real_tile(device):
    """SPPF interleaved in DRAM: 8 width shards x 4 sub-ranges = 32 cores, most with exactly C = 4 stream tiles,
    so NEO t of such a core computes tile t and nothing else. The output's memory first holds another pool's
    positive result; the input is all negative, so a tile a missing reader, NEO or writer left unwritten shows."""
    R, C, W = artifacts.THREAD_PLAN
    x_pos = put(draw(*SPPF[:4], seed=1).abs(), device)
    first = max_pool2d(x_pos, *SPPF)
    ttnn.deallocate(first)

    t = draw(*SPPF[:4], seed=2, negative=True)
    x = put(t, device)
    run_args = assignments_of(x, SPPF)
    reader = {k.kernel: k for k in run_args.kernel_run_args}["reader"].runtime_arg_values
    tiles = list(reader["num_tiles"].values())
    assert len(tiles) == 32, tiles
    assert sum(1 for n in tiles if n == C) >= 16, tiles  # every NEO owns exactly one real tile there
    run_and_check(x, t, SPPF)


def test_one_device_session_with_tt_metal_programs(device):
    """A tt-metal program before (to_layout of the input), ours, then tt-metal programs after (a tilize and a move
    to DRAM, both with untagged transfers), and our op again on its own result: all in one session."""
    args = (1, 8, 8, 64, (5, 5), (1, 1), (2, 2), (1, 1))
    t = draw(*args[:4], seed=3)
    x_tile = put(t, device, layout=ttnn.TILE_LAYOUT)
    x = ttnn.to_layout(x_tile, ttnn.ROW_MAJOR_LAYOUT)  # tt-metal before ours
    out = run_and_check(x, t, args)
    tiled = ttnn.to_layout(out, ttnn.TILE_LAYOUT)  # tt-metal after ours
    in_dram = ttnn.to_memory_config(tiled, ttnn.DRAM_MEMORY_CONFIG)
    expected = reference(t, *args)
    assert torch.equal(ttnn.to_torch(in_dram).double(), expected)
    # Ours after tt-metal's, on our own result (width-sharded in L1): YOLOv8 pools three times in a row.
    again = max_pool2d(out, *args)
    assert torch.equal(ttnn.to_torch(again).double(), reference(expected.to(torch.bfloat16), *args))
    back = ttnn.to_layout(again, ttnn.TILE_LAYOUT)
    assert torch.equal(ttnn.to_torch(back).double(), reference(expected.to(torch.bfloat16), *args))


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "ROW_MAJOR"])
def test_height_sharded_on_one_core_then_pooled_again(device, layout):
    """R10: ResNet-50's [1, 1, 1024, 64] as one shard on one core (position sub-ranges on spare cores), then the
    result (height-sharded on that core) pooled again."""
    t = draw(*RESNET[:4], seed=4)
    x = put(t, device, layout=layout,
            memory_config=sharded_config(ttnn.ShardStrategy.HEIGHT, (1024, 64), 1, device))
    out = run_and_check(x, t, RESNET)
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert out.memory_config().shard_spec.grid.num_cores() == 1
    first = reference(t, *RESNET).to(torch.bfloat16)
    second_args = (1, 16, 16, 64, (3, 3), (2, 2), (1, 1), (1, 1))
    again = max_pool2d(out, *second_args)
    assert torch.equal(ttnn.to_torch(again).double(), reference(first, *second_args))


def test_width_sharded_on_one_core(device):
    """R10: a width-sharded input held by one core (in place on its owner, staged on the spare cores)."""
    args = (1, 8, 8, 64, (5, 5), (1, 1), (2, 2), (1, 1))
    t = draw(*args[:4], seed=5)
    x = put(t, device, memory_config=sharded_config(ttnn.ShardStrategy.WIDTH, (64, 64), 1, device))
    out = run_and_check(x, t, args)
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "ROW_MAJOR"])
def test_in_place_owner_and_staged_spare_cores(device, layout):
    """Width-sharded on two cores: each owner pools its first sub-range in place from its own shard, a spare core
    pools the second from a staging it fills over the NoC."""
    t = draw(*RESNET[:4], seed=6)
    x = put(t, device, layout=layout, memory_config=sharded_config(ttnn.ShardStrategy.WIDTH, (1024, 32), 2, device))
    run_args = assignments_of(x, RESNET)
    reader = {k.kernel: k for k in run_args.kernel_run_args}["reader"].runtime_arg_values
    in_place = list(reader["in_place"].values())
    assert in_place.count(1) == 2 and in_place.count(0) >= 2, in_place
    run_and_check(x, t, RESNET)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "ROW_MAJOR"])
@pytest.mark.parametrize("placement", ["interleaved", "height"])
def test_staged_streamed(device, monkeypatch, layout, placement):
    """The staged-streamed regime: a staging budget too small for one block per core, so the reader stages the
    receptive field block by block (two sync_threads per block) and every core runs several blocks."""
    args = (1, 32, 64, 32, (3, 5), (2, 1), (1, 2), (1, 1))
    t = draw(*args[:4], seed=7)
    config = (ttnn.DRAM_MEMORY_CONFIG if placement == "interleaved"
              else sharded_config(ttnn.ShardStrategy.HEIGHT, (2048, 32), 1, device))
    x = put(t, device, layout=layout, memory_config=config)
    # One core per output shard (no position sub-ranges), so a core has Q = 8 tiles per NEO to block; a budget
    # that holds about one round's receptive field (RM: 4 input rows of 64 B sticks; TILE: 10 tile pages).
    monkeypatch.setattr(artifacts, "MAX_POSITION_SPLITS", 1)
    monkeypatch.setattr(artifacts, "_STAGING_BUDGET_OVERRIDE", 32 * 1024 if layout == ttnn.TILE_LAYOUT else 24 * 1024)
    run_args = assignments_of(x, args)
    reader = {k.kernel: k for k in run_args.kernel_run_args}["reader"].runtime_arg_values
    C = artifacts.THREAD_PLAN[1]
    blocks = [-(-(-(-n // C)) // bt) for n, bt in zip(reader["num_tiles"].values(), reader["block_tiles"].values())]
    assert max(blocks) > 1, blocks
    run_and_check(x, t, args)


@pytest.mark.parametrize("threads", [(1, 1, 1), (2, 2, 2), (2, 4, 2), (4, 4, 1), (1, 4, 1)],
                         ids=lambda p: "R%dC%dW%d" % p)
def test_thread_plans(device, monkeypatch, threads):
    """The kernels size their loops from the plan's counts: other plans than THREAD_PLAN, including R < C (a reader
    feeding several NEOs in rotation) and one thread of each kind."""
    monkeypatch.setattr(artifacts, "THREAD_PLAN", threads)
    t = draw(*SPPF[:4], seed=8)
    run_and_check(put(t, device), t, SPPF)
    t2 = draw(*RESNET[:4], seed=9)
    x2 = put(t2, device, layout=ttnn.TILE_LAYOUT,
             memory_config=sharded_config(ttnn.ShardStrategy.HEIGHT, (1024, 64), 1, device))
    run_and_check(x2, t2, RESNET)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "ROW_MAJOR"])
@pytest.mark.parametrize("strategy", ["height", "width"])
def test_col_major_input_gives_row_major_result(device, layout, strategy):
    """Rule R9: the result's shard orientation is row-major, also for an input sharded col-major on a 2-D grid
    (2 x 4 cores). There input shard g and output shard g sit on different cores, so a width-sharded owner may not
    pool in place: it stages its channel run from the core that holds it."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3))])
    if strategy == "height":
        args, shard, memory_layout = RESNET, (128, 64), ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    else:
        args, shard, memory_layout = (1, 8, 8, 256, (5, 5), (1, 1), (2, 2), (1, 1)), (64, 32), \
            ttnn.TensorMemoryLayout.WIDTH_SHARDED
    t = draw(*args[:4], seed=10)
    config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1,
                               ttnn.ShardSpec(grid, list(shard), ttnn.ShardOrientation.COL_MAJOR))
    x = ttnn.from_torch(t, layout=layout, device=device, dtype=ttnn.bfloat16, memory_config=config)
    out = run_and_check(x, t, args)
    assert out.memory_config().memory_layout == memory_layout
    assert out.memory_config().shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    assert out.memory_config().shard_spec.grid == grid
