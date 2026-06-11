# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for the descriptor-framework slow-path rebuild perf bug (issue #46506), for
hand-migrated data-movement ops.

These ops are SHARED device operations (one op, several program factories). Each op declares a
single op-level get_dynamic_runtime_args() that dispatches on the same condition as
select_program_factory:
  - Slice (SliceRmProgramFactory): the row-major interleaved factory bakes ADDRESS-DERIVED runtime
    args (reader common arg 0 = start_addr + begins_bytes - misalignment; writer per-core arg 0 =
    raw output address) that no Buffer*/CB binding can patch, so they are re-derived from the live
    buffers on every cache hit. The other slice factories already fast-path via Buffer*/CB bindings,
    so get_dynamic returns {} for them.
  - Concat (ConcatS2IProgramFactory): sharded-input -> interleaved-output factory bakes a raw output
    address into the writer's per-core arg 0 (its input addresses ARE CB-bound). That address is
    re-applied on every cache hit; every other concat factory returns {}.

This module sets TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT=1 BEFORE importing ttnn, so that
ANY op falling to the slow-path rebuild raises (TT_FATAL) instead of silently rebuilding.

To actually exercise the address-derived re-application (not just a stable-address no-op), each
test allocates FRESH input/output tensors on every iteration so the device buffer addresses change
across cache hits. If the op left a stale baked address, the result would be wrong (caught by the
torch comparison) or the dispatch would fault.
"""

import os

# Must be set before ttnn first dispatches (the adapter reads it via getenv on first slow-path hit).
os.environ["TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT"] = "1"

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


# ---------------------------------------------------------------------------
# Slice (row-major, interleaved) -- SliceRmProgramFactory: address-derived rt-args.
# ---------------------------------------------------------------------------


def _slice_rm_once(device, torch_input, begins, ends):
    """Fresh device tensor each call (so the input/output buffer addresses change across cache hits),
    then a row-major slice with an OFFSET start (begins[-1] != 0 exercises the misalignment-adjusted
    reader address)."""
    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = ttnn.slice(tt_in, begins, ends)
    out = ttnn.to_torch(tt_out)
    # Drop the device tensors so the next iteration reallocates at (likely) different addresses.
    ttnn.deallocate(tt_out)
    ttnn.deallocate(tt_in)
    return out


def test_slice_rm_no_rebuild(device):
    # Row-major slice with a non-zero last-dim start so reader common arg 0 is misalignment-adjusted.
    shape = (1, 2, 8, 64)
    begins = (0, 0, 1, 8)
    ends = (1, 2, 7, 56)

    for i in range(3):
        # Fresh, distinct data each iteration; correctness is verified against torch every time.
        torch_input = torch.randn(shape, dtype=torch.bfloat16) + float(i)
        out = _slice_rm_once(device, torch_input, begins, ends)
        ref = torch_input[
            begins[0] : ends[0],
            begins[1] : ends[1],
            begins[2] : ends[2],
            begins[3] : ends[3],
        ]
        assert_with_pcc(ref, out, 0.9999)


# ---------------------------------------------------------------------------
# Concat (sharded inputs -> interleaved output) -- ConcatS2IProgramFactory: raw output address.
# ---------------------------------------------------------------------------


def _make_height_sharded(device, torch_tensor, shard_shape, shard_grid):
    mem_cfg = ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    t = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return ttnn.to_memory_config(t, mem_cfg)


def _height_sharded_mem_config(shard_shape, shard_grid):
    return ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )


def test_concat_s2i_no_rebuild(device):
    # Two height-sharded row-major inputs concatenated along width (dim=3) into a height-sharded
    # output. This routes through the migrated sharded-concat descriptor fast path (input/output
    # shards are CB `.buffer`-bound, re-patched on every cache hit). Fresh input tensors each
    # iteration so the shard buffer addresses change across cache hits -> any descriptor rebuild on
    # a cache hit raises under the guard, and a stale baked address would corrupt the result.
    #
    # NOTE: a sharded-input -> INTERLEAVED-output concat (the literal ConcatS2IProgramFactory) is
    # rejected by ConcatDeviceOperation::validate ("Sharded output and inputs must have the same
    # memory layout."), i.e. that factory is unreachable via ttnn.concat. We therefore exercise the
    # reachable sharded->sharded concat, which still validates the cache-hit fast path for concat.
    shape = (1, 1, 160, 32)
    shard_shape = (80, 32)
    out_shard_shape = (80, 64)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
    out_mem_cfg = _height_sharded_mem_config(out_shard_shape, shard_grid)

    for i in range(3):
        torch_a = torch.randn(shape, dtype=torch.bfloat16) + float(i)
        torch_b = torch.randn(shape, dtype=torch.bfloat16) - float(i)
        tt_a = _make_height_sharded(device, torch_a, shard_shape, shard_grid)
        tt_b = _make_height_sharded(device, torch_b, shard_shape, shard_grid)

        tt_out = ttnn.concat([tt_a, tt_b], dim=3, memory_config=out_mem_cfg)
        out = ttnn.to_torch(tt_out)

        ref = torch.concat([torch_a, torch_b], dim=3)
        assert_with_pcc(ref, out, 0.9999)

        ttnn.deallocate(tt_out)
        ttnn.deallocate(tt_b)
        ttnn.deallocate(tt_a)
