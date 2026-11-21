# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design's thread plan requires (op_design.md, Work Distribution, "Tests the plan needs").

1. Every reader and writer thread of the plan moves exactly its own blocks (a correct result does not show it).
2. The op shares a device session with tt-metal's own programs, before and after them.
3. The stick_stream regime predicate refuses a stick too wide for the ring (deferred channel_chunk).
Plus: R2 (int and pair factor alike) and R5 (an L1 input gives an L1 result).
"""

import math

import pytest
import torch

import ttnn
from ttnn.operations.upsample import upsample
from ttnn.operations.upsample.upsample import _dispatch
from ttnn.operations.upsample.upsample_program_artifacts import (
    THREAD_PLAN,
    block_sticks_for,
    split_sticks,
    stick_geometry,
)

SENTINEL = 12345.0  # exactly representable in bfloat16; never drawn from a standard normal


def _put(t, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )


def _reference(t, sh, sw):
    return t.repeat_interleave(sh, dim=1).repeat_interleave(sw, dim=2)


def _thread_of_stick(x, num_threads, device):
    """For each input stick (flattened N*H*W), the thread index (block mod num_threads) that owns it."""
    num_sticks = math.prod(tuple(x.shape)[:3])
    _, counts, starts = split_sticks(num_sticks, device.compute_with_storage_grid_size())
    _, stride = stick_geometry(x)
    block = block_sticks_for(max(counts.values()), stride)
    owner = torch.empty(num_sticks, dtype=torch.long)
    for core, n in counts.items():
        local = torch.arange(n)
        owner[starts[core] : starts[core] + n] = (local // block) % num_threads
    # every thread of the plan must own a block somewhere, or the test proves nothing
    assert set(owner.tolist()) == set(range(num_threads)), owner.unique()
    return owner


SHAPE, SH, SW = (1, 40, 40, 512), 2, 2  # 50 sticks per cluster: 8 blocks of 7, every thread busy


@pytest.mark.parametrize("masked", range(THREAD_PLAN[1]))
def test_each_writer_thread_writes_its_blocks(device, masked):
    """Writer thread `masked` takes its entries but writes nothing: exactly the output pages of its blocks
    keep the sentinel, every other page matches the reference."""
    torch.manual_seed(100 + masked)
    t = torch.randn(*SHAPE, dtype=torch.bfloat16)
    x = _put(t, device)
    out_shape = (SHAPE[0], SHAPE[1] * SH, SHAPE[2] * SW, SHAPE[3])
    out = _put(torch.full(out_shape, SENTINEL, dtype=torch.bfloat16), device)
    W = THREAD_PLAN[1]
    y = ttnn.to_torch(_dispatch(x, SH, SW, output=out, writer_thread_mask=((1 << W) - 1) & ~(1 << masked)))

    owner = _thread_of_stick(x, W, device).reshape(SHAPE[:3])
    skipped = _reference((owner == masked).unsqueeze(-1), SH, SW).expand(out_shape)
    ref = _reference(t, SH, SW)
    assert torch.all(y[skipped] == SENTINEL), "a masked writer thread's page was written"
    assert torch.equal(y[~skipped], ref[~skipped]), "a page of another writer thread is wrong"
    assert skipped.any() and (~skipped).any()


@pytest.mark.parametrize("masked", range(THREAD_PLAN[0]))
def test_each_reader_thread_reads_its_blocks(device, masked):
    """Reader thread `masked` takes its entries but reads nothing: the pages of its blocks are written from a
    stale ring entry (not the reference), every other page matches the reference."""
    torch.manual_seed(200 + masked)
    t = torch.randn(*SHAPE, dtype=torch.bfloat16)
    x = _put(t, device)
    R = THREAD_PLAN[0]
    y = ttnn.to_torch(_dispatch(x, SH, SW, reader_thread_mask=((1 << R) - 1) & ~(1 << masked)))

    owner = _thread_of_stick(x, R, device).reshape(SHAPE[:3])
    stale = _reference((owner == masked).unsqueeze(-1), SH, SW).expand(y.shape)
    ref = _reference(t, SH, SW)
    assert torch.equal(y[~stale], ref[~stale]), "a page of another reader thread is wrong"
    # stale sticks: no stick of a masked block may carry its own fresh input
    y_sticks = y.reshape(-1, SHAPE[3])
    r_sticks = ref.reshape(-1, SHAPE[3])
    stale_sticks = stale.reshape(-1, SHAPE[3])[:, 0]
    assert not torch.any(torch.all(y_sticks[stale_sticks] == r_sticks[stale_sticks], dim=1)), (
        "a masked reader thread's stick arrived anyway"
    )


def test_session_order(device):
    """In one device session: a tt-metal program, then upsample twice, then tt-metal's own to_layout
    (tilize, then untilize) on the result. The kernels tag every transfer and hand the tag back."""
    torch.manual_seed(7)
    first = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    tiled = ttnn.from_torch(first, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    assert torch.equal(ttnn.to_torch(ttnn.to_layout(tiled, ttnn.ROW_MAJOR_LAYOUT)), first)

    a = torch.randn(1, 20, 20, 512, dtype=torch.bfloat16)
    b = torch.randn(2, 8, 8, 128, dtype=torch.bfloat16)
    ya = upsample(_put(a, device), (2, 2))
    yb = upsample(_put(b, device), 3)
    assert torch.equal(ttnn.to_torch(ya), _reference(a, 2, 2))

    yb_tiled = ttnn.to_layout(yb, ttnn.TILE_LAYOUT)
    yb_back = ttnn.to_layout(yb_tiled, ttnn.ROW_MAJOR_LAYOUT)
    assert torch.equal(ttnn.to_torch(yb_back), _reference(b, 3, 3))


def test_wide_stick_refused_by_regime_predicate(device):
    """A stick whose 8-entry ring exceeds the ring cap is the deferred channel_chunk regime."""
    x = _put(torch.zeros(1, 1, 1, 65536, dtype=torch.bfloat16), device)
    with pytest.raises(NotImplementedError, match="channel_chunk"):
        upsample(x, 2)


def test_int_and_pair_factor_agree_and_l1_stays_l1(device):
    """R2: 3 and (3, 3) read alike. R5: an L1 interleaved input gives an L1 interleaved output."""
    torch.manual_seed(11)
    t = torch.randn(1, 7, 9, 64, dtype=torch.bfloat16)
    x = _put(t, device, ttnn.L1_MEMORY_CONFIG)
    y_int, y_pair = upsample(x, 3), upsample(x, (3, 3))
    for y in (y_int, y_pair):
        assert y.memory_config().buffer_type == ttnn.BufferType.L1
        assert y.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        assert torch.equal(ttnn.to_torch(y), _reference(t, 3, 3))
