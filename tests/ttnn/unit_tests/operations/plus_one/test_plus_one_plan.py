# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests of plus_one's thread plan and host split (op_design.md, Work Distribution, "Tests the plan needs").

1. every thread ran: thread t adds 1 + t under the builder's `probe_tag_threads`, and the test recomputes
   each unit's owner from the host split;
2. one device session: plus_one after a tt-metal program and before one, twice on the same tensor;
3. pinned split shapes: (4096,) multi-core, (100,) one worker, (100,) cut into segments with a ragged tail;
   and the block path (several units per block) the rank-2 knob-turn will use, through the builder.
"""

import pytest
import torch

import ttnn
from ttnn.operations.plus_one import plus_one
from ttnn.operations.plus_one.plus_one import _plus_one
from ttnn.operations.plus_one.plus_one_program_artifacts import WORKER_THREADS, create_program_artifacts, plan_for

MASK = 0xFFFFFFFF
INT32_MAX = 2**31 - 1
EDGES = [-1, 0, INT32_MAX, -(2**31), INT32_MAX - 1, 1]


def _host(dtype):
    return torch.int32 if dtype == ttnn.int32 else torch.int64


def _values(shape, dtype, seed=0, edges_at=()):
    """Integers over the whole 32-bit range, with the edge values at the start and at each of `edges_at`."""
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2**32, shape, generator=g, dtype=torch.int64)
    flat = bits.view(-1)
    for pos in (0, *edges_at):
        for i, v in enumerate(EDGES):
            if pos + i < flat.numel():
                flat[pos + i] = v & MASK
    if dtype == ttnn.int32:
        return torch.where(bits >= 2**31, bits - 2**32, bits).to(torch.int32)
    return bits


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.int64) & MASK


def _expected(x, increments, skip):
    u = _bits(x).reshape(-1)
    new = (u + increments) & MASK
    return torch.where(u < INT32_MAX, new, u) if skip else new


def _put(x, dtype, device):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _check(got, expected):
    got = _bits(ttnn.to_torch(got)).view(-1)
    expected = expected.view(-1)
    bad = (got != expected).nonzero().view(-1)
    assert bad.numel() == 0, f"{bad.numel()} wrong elements, first at {bad[:8].tolist()}: " \
                             f"got {got[bad[:8]].tolist()} expected {expected[bad[:8]].tolist()}"


def _increments_by_thread(plan, numel):
    """Per element, the increment a tag_threads run adds: 1 + the thread that owns the element's unit."""
    inc = torch.empty(numel, dtype=torch.int64)
    for unit in range(plan.num_units):
        row, off, n = plan.unit_extent(unit)
        start = row * plan.row_elems + off
        inc[start : start + n] = 1 + plan.owner_thread(unit)
    return inc


def test_every_thread_ran(device):
    """A correct result does not show that four threads ran; the tag does: thread t adds 1 + t."""
    shape = (4096,)
    x = _values(shape, ttnn.int32)
    t = _put(x, ttnn.int32, device)
    plan = plan_for(t)
    assert (plan.num_units, plan.num_cores, plan.seg_elems) == (16, 4, 256), plan
    inc = _increments_by_thread(plan, x.numel())
    assert set(inc.unique().tolist()) == {1 + i for i in range(WORKER_THREADS)}
    out = _plus_one(t, probe_tag_threads=True)
    assert out is t
    _check(out, _expected(x, inc, skip=False))


def test_one_device_session(device):
    """tt-metal program -> plus_one -> tt-metal program -> plus_one again on the same tensor."""
    torch.manual_seed(1)
    tile_src = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)

    def untilize():
        tiled = ttnn.from_torch(tile_src, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        rm = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(tiled, ttnn.ROW_MAJOR_LAYOUT)))
        assert torch.equal(rm, tile_src)

    x = _values((4096,), ttnn.uint32, seed=2)
    t = _put(x, ttnn.uint32, device)
    untilize()
    assert plus_one(t) is t
    _check(t, _expected(x, 1, skip=False))
    untilize()
    assert plus_one(t) is t
    _check(t, _expected(x, 2, skip=False))
    untilize()


PINNED = [
    # (shape, builder knobs, (units, cores, seg_elems, last segment))
    ((4096,), {}, (16, 4, 256, 256)),
    ((100,), {}, (1, 1, 100, 100)),
    ((100,), {"min_worker_bytes": 64}, (7, 2, 16, 4)),
]


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32], ids=["int32", "uint32"])
@pytest.mark.parametrize("shape,knobs,expect", PINNED, ids=["4096", "100", "100-seg16"])
def test_pinned_split(device, shape, knobs, expect, dtype, skip):
    t = _put(torch.zeros(shape, dtype=_host(dtype)), dtype, device)
    plan = plan_for(t, **knobs)
    last = plan.unit_extent(plan.num_units - 1)[2]
    assert (plan.num_units, plan.num_cores, plan.seg_elems, last) == expect, plan
    # The edge values at every segment boundary.
    x = _values(shape, dtype, seed=3, edges_at=[u * plan.seg_elems for u in range(plan.num_units)])
    t = _put(x, dtype, device)
    assert _plus_one(t, skip_negative_entries=skip, **knobs) is t
    _check(t, _expected(x, 1, skip))


@pytest.mark.parametrize("scratch_depth", [1, 2])
@pytest.mark.parametrize("shape", [(512, 8), (1, 1, 17, 50)], ids=["512x8", "17x50"])
def test_block_path(device, shape, scratch_depth):
    """Several units per block (block_units > 1), the rank-2+ knob-turn, through the builder (the op's
    validate() still refuses rank > 1 in this phase). Tagged, so the owner thread is checked too."""
    x = _values(shape, ttnn.int32, seed=4)
    t = _put(x, ttnn.int32, device)
    plan = plan_for(t, scratch_depth=scratch_depth)
    if shape == (512, 8):
        assert (plan.num_units, plan.num_cores, plan.block_units) == (512, 4, 32), plan
    spec, run_args, indices = create_program_artifacts(t, probe_tag_threads=True, scratch_depth=scratch_depth)
    ttnn.generic_op([t], spec, run_args, indices)
    _check(t, _expected(x, _increments_by_thread(plan, x.numel()), skip=False))
