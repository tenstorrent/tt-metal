# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Implementer tests of nlp_concat_heads_decode beyond the acceptance test: every first-phase INPUTS entry, the
thread plan (every reader thread ran), the padding rows, the session order of the target page, rejections."""

import pytest
import torch

import ttnn
from ttnn.operations.nlp_concat_heads_decode import nlp_concat_heads_decode
from ttnn.operations.nlp_concat_heads_decode import nlp_concat_heads_decode_program_artifacts as artifacts
from ttnn.operations.nlp_concat_heads_decode.nlp_concat_heads_decode import output_memory_config

# PHASE0 entries of the specification's INPUTS (H = 32, head_dim % 32 == 0).
PHASE0_INPUTS = [
    ((1, 1, 32, 64), 2),
    ((1, 1, 32, 64), 32),
    ((1, 1, 32, 128), 1),
    ((1, 1, 32, 128), 2),
    ((1, 1, 32, 128), 32),
    ((1, 1, 32, 32), 1),
    ((1, 1, 32, 64), 1),
    ((1, 1, 32, 64), 4),
    ((1, 1, 32, 96), 3),
    ((1, 1, 32, 160), 8),
    ((1, 1, 32, 256), 16),
    ((1, 1, 32, 256), 32),
]


def _input(device, shape, seed=0, dtype=ttnn.bfloat16):
    torch.manual_seed(seed)
    host = torch.randn(*shape, dtype=torch.bfloat16)
    grid = device.compute_with_storage_grid_size()
    config = ttnn.create_sharded_memory_config(
        shape=(32 * shape[1], shape[3]),
        core_grid=ttnn.num_cores_to_corerangeset(1, grid, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return host, ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)


def _check(device, host, out, num_heads, zero_padding=True):
    _, users, _, head_dim = host.shape
    got = ttnn.to_torch(out)
    assert list(got.shape) == [1, 1, 32, num_heads * head_dim]
    expected = host[:, :, :num_heads, :].reshape(1, 1, users, num_heads * head_dim)
    assert torch.equal(got[:, :, :users, :], expected), (got[:, :, :users, :] - expected).abs().max()
    if zero_padding:
        assert torch.count_nonzero(got[:, :, users:, :]) == 0, "padding rows must be zero"
    assert out.memory_config() == output_memory_config(device, num_heads, head_dim)


@pytest.mark.parametrize("shape,num_heads", PHASE0_INPUTS, ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else f"h{v}")
def test_phase0_inputs(device, shape, num_heads):
    host, x = _input(device, shape)
    _check(device, host, nlp_concat_heads_decode(x, num_heads=num_heads), num_heads)


@pytest.mark.parametrize("reader_threads", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("head_dim", [256, 160])
def test_every_reader_thread_ran(device, monkeypatch, reader_threads, head_dim):
    """READER_THREADS split a head's tiles into contiguous ranges, one per thread (ragged last range at 160 = 5
    tiles). Each thread zeroes and gathers only its own tiles, so a thread that did not run leaves its tiles'
    row 0 wrong: an exact match over the whole row shows that every thread of the plan ran."""
    monkeypatch.setattr(artifacts, "READER_THREADS", reader_threads)
    host, x = _input(device, (1, 1, 32, head_dim), seed=reader_threads)
    _check(device, host, nlp_concat_heads_decode(x, num_heads=8), 8)


def test_thread_split_covers_every_tile_once():
    """Host-side: the ranges [r*block, min((r+1)*block, head_tiles)) of the threads the plan engages partition
    the head's tiles, and no engaged thread is empty."""
    for head_tiles in range(1, 9):
        for reader_threads in range(1, 7):
            block = -(-head_tiles // min(reader_threads, head_tiles))
            threads = -(-head_tiles // block)
            ranges = [range(r * block, min((r + 1) * block, head_tiles)) for r in range(threads)]
            assert all(len(rg) > 0 for rg in ranges)
            assert [t for rg in ranges for t in rg] == list(range(head_tiles))
            assert threads <= reader_threads


def test_padding_left_unwritten(device, monkeypatch):
    """zero_padding_rows = 0 (R3 allows it): rows 0..users-1 still exact."""
    monkeypatch.setattr(artifacts, "ZERO_PADDING_ROWS", 0)
    host, x = _input(device, (1, 1, 32, 128))
    _check(device, host, nlp_concat_heads_decode(x, num_heads=4), 4, zero_padding=False)


def test_session_order(device):
    """Target page: in one device session, the op after another program and before a tt-metal program
    (to_layout of its result), values checked after each."""
    torch.manual_seed(7)
    other = torch.randn(1, 1, 64, 96, dtype=torch.bfloat16)
    t = ttnn.from_torch(other, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    assert torch.equal(ttnn.to_torch(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)), other)

    for seed, (shape, num_heads) in enumerate([((1, 1, 32, 128), 32), ((1, 1, 32, 64), 2), ((1, 1, 32, 256), 16)]):
        host, x = _input(device, shape, seed=seed)
        out = nlp_concat_heads_decode(x, num_heads=num_heads)
        _check(device, host, out, num_heads)
        # A tt-metal program on the result: untagged transfers after our tagged ones.
        rm = ttnn.to_layout(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT)
        expected = host[:, :, :num_heads, :].reshape(1, 1, 1, num_heads * shape[3])
        assert torch.equal(ttnn.to_torch(rm)[:, :, :1, :], expected)
        rm2 = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        assert torch.equal(ttnn.to_torch(rm2)[:, :, :1, :], expected)


@pytest.mark.parametrize(
    "shape,num_heads,kwargs,error",
    [
        ((1, 1, 32, 64), 0, {}, ValueError),  # X1
        ((1, 1, 8, 64), 9, {}, ValueError),  # X2
        ((1, 1, 32, 64), 2, {"sub_core_grids": ((0, 0), (1, 0))}, ValueError),  # X4
        ((1, 1, 64, 64), 2, {}, ValueError),  # X7
    ],
)
def test_rejections(device, shape, num_heads, kwargs, error):
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(error):
        nlp_concat_heads_decode(x, num_heads=num_heads, **kwargs)


def test_interleaved_refused(device):
    """X6: an interleaved input is outside SUPPORTED (sharding axis)."""
    x = ttnn.from_torch(torch.randn(1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=ttnn.L1_MEMORY_CONFIG)
    with pytest.raises(NotImplementedError):
        nlp_concat_heads_decode(x, num_heads=2)


def test_float32_refused_in_phase0(device):
    _, x = _input(device, (1, 1, 32, 64), dtype=ttnn.float32)
    with pytest.raises(NotImplementedError):
        nlp_concat_heads_decode(x, num_heads=2)


class _SmallGridDevice:
    """A device whose compute grid is 2 x 1, like the callers' small-grid tests (spec R5)."""

    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(2, 1)


def test_r5_too_few_cores_raises_target_number_of_cores():
    """R5 (a `review` check in the spec; no INPUTS cell can reach it on a 32-core part, since num_heads <= H <= 32):
    the op compares num_heads with the device's compute cores before it allocates and raises RuntimeError with
    tt-metal's words, which the callers' tests turn into a skip."""
    with pytest.raises(RuntimeError, match="Target number of cores 32 is greater than total number of available cores 2"):
        output_memory_config(_SmallGridDevice(), 32, 64)
    config = output_memory_config(_SmallGridDevice(), 2, 64)
    assert list(config.shard_spec.shape) == [32, 64]
