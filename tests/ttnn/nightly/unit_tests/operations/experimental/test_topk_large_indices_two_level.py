# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""topk_large_indices two-level selection (block_select=32).

A two-level program picks the top 2048 of the 32-key block maxima, then the exact top 2048 among those blocks' keys,
and decides at run time from the search length whether to run that way or single-level. Results are checked with the
value oracle of topk_two_level_reference: the value multiset must equal the exact top-k; which of several tied keys
comes back may differ from single-level selection.
"""

import pytest
import torch
import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.topk_two_level_reference import (
    BLOCK,
    GENERATORS,
    K,
    assert_topk_exact,
    generate,
    normalize,
)

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="topk_large_indices is Blackhole-only"),
]

FORCE_TWO_LEVEL = 0
FORCE_SINGLE_LEVEL = 0xFFFFFFFF


def _dev(tensor: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _scalar(device, value: int) -> ttnn.Tensor:
    return _dev(torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1), device, ttnn.uint32)


def _indices(tt_out: ttnn.Tensor, rows: int) -> torch.Tensor:
    return ttnn.to_torch(tt_out, dtype=torch.uint32).reshape(rows, K).to(torch.int64)


def _mode(mode_tensor: ttnn.Tensor) -> int:
    return int(ttnn.to_torch(mode_tensor, dtype=torch.uint32).reshape(-1)[0])


def _grid(device, columns: int | None):
    if columns is None:
        return None
    grid = device.compute_with_storage_grid_size()
    if grid.x < columns or grid.y < 10:
        pytest.skip(f"needs a {columns}x10 worker grid, got {grid}")
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(columns - 1, 9))])


def _block_maxima(row: torch.Tensor, n: int) -> torch.Tensor:
    blocks = (n + BLOCK - 1) // BLOCK
    padded = torch.full((blocks * BLOCK,), float("-inf"))
    padded[:n] = normalize(row[:n])
    return padded.reshape(blocks, BLOCK).amax(dim=1)


def _check_rows(torch_input: torch.Tensor, indices: torch.Tensor, n: int, rows) -> None:
    for row in rows:
        assert_topk_exact(torch_input[row], indices[row], n)


# (valid length, physical width, rows, core-grid columns): lengths across both sides of the block and chunk
# boundaries (65,536 = 2048 blocks; 524,304 = 16 mod 32), the 1M capacity, both top-k grids of the sparse-MLA
# overlap layout, and row counts with and without split rows.
SHAPES = [
    (1, 65_536, 2, None),
    (K - 1, 65_536, 2, None),
    (5_000, 65_536, 3, None),
    (65_535, 65_536, 2, None),
    (65_536, 131_072, 2, None),
    (65_537, 131_072, 2, None),
    (70_000, 70_016, 2, None),
    (131_072, 131_072, 161, 12),
    (524_304, 1_049_600, 160, 12),
    (600_000, 1_049_600, 160, 11),
    (1_048_575, 1_049_600, 2, None),
    (1_048_576, 1_049_600, 160, 12),
]


@pytest.mark.parametrize("n,width,rows,columns", SHAPES)
def test_two_level_matches_oracle(device, n, width, rows, columns):
    torch_input = generate("iid", rows, width, 11, n=n)
    mode = _scalar(device, 0)
    blocks = (width + BLOCK - 1) // BLOCK
    maxima = _dev(torch.zeros(1, 1, rows, blocks, dtype=torch.bfloat16), device)
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length=n,
        sub_core_grids=_grid(device, columns),
        block_select=32,
        two_level_cutover=FORCE_TWO_LEVEL,
        block_max_output=maxima,
        mode_output=mode,
    )
    assert _mode(mode) == 1
    indices = _indices(out, rows)
    check = range(rows) if rows <= 4 else (0, 1, 107, 108, rows - 2, rows - 1)
    _check_rows(torch_input, indices, n, check)
    got = ttnn.to_torch(maxima).reshape(rows, blocks).float()
    nb = (n + BLOCK - 1) // BLOCK
    for row in check:
        assert torch.equal(normalize(got[row, :nb]), _block_maxima(torch_input[row], n)), f"row {row} maxima"


@pytest.mark.parametrize("kind", GENERATORS)
@pytest.mark.parametrize("n", [131_072, 524_304])
def test_two_level_generators(device, kind, n):
    rows, width = 8, 524_320
    torch_input = generate(kind, rows, width, 5, n=n)
    blocks = width // BLOCK
    maxima = _dev(torch.zeros(1, 1, rows, blocks, dtype=torch.bfloat16), device)
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length=n,
        block_select=32,
        two_level_cutover=FORCE_TWO_LEVEL,
        block_max_output=maxima,
    )
    _check_rows(torch_input, _indices(out, rows), n, range(rows))
    got = ttnn.to_torch(maxima).reshape(rows, blocks).float()
    nb = (n + BLOCK - 1) // BLOCK
    for row in range(rows):
        assert torch.equal(normalize(got[row, :nb]), _block_maxima(torch_input[row], n)), f"row {row} maxima"


# Stage 3 stages the selected blocks that share an aligned span (2 KB for rows of at most 8 x 2048 blocks, 1 KB for
# longer ones) and copies them to their slots. These inputs pin the 2048 winning blocks so every chunk is staged:
# - n = 524,272 (16 past a block boundary, 16,384 blocks, 2 KB spans): blocks 0, 10-13, 20, 21 and 31 of the last 256
#   spans, so each staged range is a whole span holding runs of 4 and 2 consecutive members (one local copy each);
#   the partial last block is a member of a staged group, and the keys past n in it are larger than every valid key;
# - n = 1,048,576 (32,768 blocks, 1 KB spans): blocks 0 and 15 of the first 1024 spans, 32 staged pairs per chunk.
@pytest.mark.parametrize(
    "n,members,first_span", [(524_272, (0, 10, 11, 12, 13, 20, 21, 31), 256), (1_048_576, (0, 15), 0)]
)
def test_two_level_staged_gather_patterns(device, n, members, first_span):
    rows, width = 2, (n + BLOCK - 1) // BLOCK * BLOCK + BLOCK
    span_blocks = 32 if n <= 8 * 2048 * BLOCK else 16
    g = torch.Generator().manual_seed(21)
    x = torch.randn(rows, width, generator=g) * 0.1 - 5.0
    spans = range(first_span, first_span + 2048 // len(members))
    blocks = torch.tensor([s * span_blocks + m for s in spans for m in members])
    assert int(blocks.max()) < (n + BLOCK - 1) // BLOCK
    hot = 1.0 + torch.rand(rows, blocks.numel(), generator=g)
    for row in range(rows):
        # Each winning block's maximum sits at its own lane (below 16, inside the last block's valid prefix), so a
        # block copied to the wrong slot, or a slot left with an earlier chunk's data, maps to a background key.
        x[row, blocks * BLOCK + (blocks * 7) % 16] = hot[row]
    x[:, n:] = 100.0
    torch_input = x.to(torch.bfloat16)
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length=n,
        block_select=32,
        two_level_cutover=FORCE_TWO_LEVEL,
    )
    _check_rows(torch_input, _indices(out, rows), n, range(rows))


def test_two_level_split_row_ties_order_as_a_whole_row(device):
    """A row halved across a core pair returns the same ordered indices, ties included, as on one core."""
    width, n = 262_144, 262_144
    row = generate("ties8", 1, width, 22, n=n)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1))])
    results = []
    for rows in (1, 2):  # one row on two cores is a split row; two rows on two cores are whole rows
        out = ttnn.experimental.topk_large_indices(
            _dev(row.repeat(rows, 1).reshape(1, 1, rows, width), device),
            k=K,
            valid_length=n,
            block_select=32,
            two_level_cutover=FORCE_TWO_LEVEL,
            sub_core_grids=grid,
        )
        results.append(_indices(out, rows)[0])
    _check_rows(row, results[0].reshape(1, -1), n, range(1))
    assert torch.equal(results[0], results[1])


def test_two_level_poisoned_tail_and_last_lane(device):
    """The last valid key sits in a partially valid block next to stale keys larger than everything valid."""
    rows, width, n = 2, 131_072, 100_017
    torch_input = generate("iid", rows, width, 3, n=n)
    torch_input[:, n:] = 1000.0
    torch_input[:, n - 1] = 500.0
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length=n,
        block_select=32,
        two_level_cutover=FORCE_TWO_LEVEL,
    )
    indices = _indices(out, rows)
    _check_rows(torch_input, indices, n, range(rows))
    assert (indices[:, 0] == n - 1).all()


def test_two_level_tied_blocks(device):
    """Far more than 2048 blocks share the top maximum: any 2048 of them hold the exact top values."""
    rows, width, n = 2, 262_144, 262_144
    torch_input = torch.full((rows, width), -1.0, dtype=torch.bfloat16)
    torch_input[:, ::32] = 2.0
    torch_input[:, 5::97] = 1.5
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length=n,
        block_select=32,
        two_level_cutover=FORCE_TWO_LEVEL,
    )
    _check_rows(torch_input, _indices(out, rows), n, range(rows))


@pytest.mark.parametrize("rows,columns", [(4, None), (160, 3)])
def test_two_level_supplied_block_maxima(device, rows, columns):
    """The caller's maxima: a few rows over the default grid, and 160 rows over 27 cores (six on most), past the
    four rows per core that phase 1 allows. The second call hits the cache with new input and maxima tensors, whose
    addresses the program must take up."""
    width, n = 1_049_600, 700_000
    blocks = width // BLOCK
    kept = []
    entries = []
    for seed, kind in ((9, "sinks"), (19, "iid")):
        torch_input = generate(kind, rows, width, seed, n=n)
        maxima = torch.full((rows, blocks), float("-inf"))
        for row in range(rows):
            nb = (n + BLOCK - 1) // BLOCK
            maxima[row, :nb] = _block_maxima(torch_input[row], n)
        # Both calls' tensors stay allocated, so the second call's addresses differ from the first's.
        tt_input = _dev(torch_input.reshape(1, 1, rows, width), device)
        tt_maxima = _dev(maxima.to(torch.bfloat16).reshape(1, 1, rows, blocks), device)
        kept.append((tt_input, tt_maxima))
        out = ttnn.experimental.topk_large_indices(
            tt_input,
            k=K,
            valid_length=n,
            block_select=32,
            two_level_cutover=FORCE_TWO_LEVEL,
            block_max_tensor=tt_maxima,
            sub_core_grids=_grid(device, columns),
        )
        entries.append(device.num_program_cache_entries())
        _check_rows(torch_input, _indices(out, rows), n, range(rows))
    assert entries[0] > 0, "the program cache must be on"
    assert entries[1] == entries[0], "the second call must reuse the cached program"
    (first_input, first_maxima), (second_input, second_maxima) = kept
    assert first_input.buffer_address() != second_input.buffer_address()
    assert first_maxima.buffer_address() != second_maxima.buffer_address()


@pytest.mark.parametrize("n,cutover,expected_mode", [(200_000, 200_001, 0), (200_001, 200_001, 1), (5_000, 0, 1)])
def test_two_level_cutover_selects_mode(device, n, cutover, expected_mode):
    rows, width = 2, 262_144
    torch_input = generate("iid", rows, width, 1, n=n)
    mode = _scalar(device, 7)
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length=n,
        block_select=32,
        two_level_cutover=cutover,
        mode_output=mode,
    )
    assert _mode(mode) == expected_mode
    _check_rows(torch_input, _indices(out, rows), n, range(rows))


def test_two_level_empty_row_is_sentinels(device):
    """A real-token end of 0 searches nothing: every index is the sentinel."""
    rows, width = 2, 131_072
    torch_input = generate("iid", rows, width, 1)
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length_tensor=_scalar(device, 1000),
        valid_end_tensor=_scalar(device, 0),
        block_select=32,
    )
    assert (_indices(out, rows) == 0xFFFFFFFF).all()


def test_two_level_scalar_and_metadata_agree_and_repeat(device):
    """C2 and C3: the same bound as a scalar and as metadata gives bit-identical ordered indices, run after run."""
    rows, width, n, offset = 160, 1_049_600, 524_304, 160
    torch_input = generate("narrow", rows, width, 4, n=n)
    tt_input = _dev(torch_input.reshape(1, 1, rows, width), device)
    common = dict(k=K, block_select=32, two_level_cutover=FORCE_TWO_LEVEL, sub_core_grids=_grid(device, 12))
    scalar = _indices(ttnn.experimental.topk_large_indices(tt_input, valid_length=n, **common), rows)
    metadata = _indices(
        ttnn.experimental.topk_large_indices(
            tt_input,
            valid_length_tensor=_scalar(device, n - offset),
            valid_length_offset=offset,
            valid_end_tensor=_scalar(device, n),
            **common,
        ),
        rows,
    )
    again = _indices(ttnn.experimental.topk_large_indices(tt_input, valid_length=n, **common), rows)
    assert torch.equal(scalar, metadata)
    assert torch.equal(scalar, again)
    _check_rows(torch_input, scalar, n, (0, 107, 108, 159))


def test_two_level_trace_crosses_cutover(device):
    """One captured program serves lengths on both sides of the cutover, in both directions, with both metadata
    words changing between replays; the mode output says which way each replay ran."""
    device.enable_program_cache()
    rows, width, cutover = 160, 1_049_600, 300_000
    torch_input = generate("iid", rows, width, 6)
    tt_input = _dev(torch_input.reshape(1, 1, rows, width), device)
    length = _scalar(device, 5_000)
    valid_end = _scalar(device, 5_000)
    mode = _scalar(device, 7)

    def run():
        return ttnn.experimental.topk_large_indices(
            tt_input,
            k=K,
            valid_length_tensor=length,
            valid_end_tensor=valid_end,
            block_select=32,
            two_level_cutover=cutover,
            mode_output=mode,
            sub_core_grids=_grid(device, 12),
        )

    run()
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    output = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        for n, end in (
            (5_000, 5_000),
            (1_048_576, 1_048_576),
            (400_000, 390_000),
            (299_999, 299_999),
            (700_000, 250_000),
            (5_000, 5_000),
        ):
            for tensor, value in ((length, n), (valid_end, end)):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    ),
                    tensor,
                )
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            bound = min(n, (end + 31) // 32 * 32)
            assert _mode(mode) == (1 if bound >= cutover else 0), f"replay n={n} end={end}"
            _check_rows(torch_input, _indices(output, rows), bound, (0, 107, 108, 159))
    finally:
        ttnn.release_trace(device, trace_id)


def test_two_level_leaves_the_next_op_intact(device):
    """A single-level op after a two-level one on the same cores still gives the exact result."""
    rows, width = 160, 131_072
    torch_input = generate("iid", rows, width, 8)
    tt_input = _dev(torch_input.reshape(1, 1, rows, width), device)
    ttnn.experimental.topk_large_indices(tt_input, k=K, block_select=32, two_level_cutover=FORCE_TWO_LEVEL)
    out = ttnn.experimental.topk_large_indices(tt_input, k=K)
    _check_rows(torch_input, _indices(out, rows), width, (0, 50, 120, 159))


def test_two_level_and_single_level_programs_are_cached_apart(device):
    device.enable_program_cache()
    device.clear_program_cache()
    rows, width = 4, 131_072
    tt_input = _dev(generate("iid", rows, width, 2).reshape(1, 1, rows, width), device)
    ttnn.experimental.topk_large_indices(tt_input, k=K)
    single = device.num_program_cache_entries()
    ttnn.experimental.topk_large_indices(tt_input, k=K, block_select=32)
    assert device.num_program_cache_entries() == single + 1
    ttnn.experimental.topk_large_indices(tt_input, k=K, block_select=32, two_level_cutover=FORCE_TWO_LEVEL)
    assert device.num_program_cache_entries() == single + 1, "the cutover is a runtime argument"


@pytest.mark.parametrize("k,width", [(1024, 131_072), (K, 131_088), (K, 65_504), (K, 2_097_184)])
def test_two_level_falls_back_outside_its_range(device, expect_error, k, width):
    """block_select on an input the two-level program does not cover runs the single-level program; the two-level
    test outputs need a two-level program."""
    rows = 2
    torch_input = generate("iid", rows, width, 4)
    tt_input = _dev(torch_input.reshape(1, 1, rows, width), device)
    out = ttnn.experimental.topk_large_indices(tt_input, k=k, block_select=32)
    indices = ttnn.to_torch(out, dtype=torch.uint32).reshape(rows, k).to(torch.int64)
    for row in range(rows):
        assert_topk_exact(torch_input[row], indices[row], width, k)
    with expect_error(RuntimeError, "need a two-level program"):
        ttnn.experimental.topk_large_indices(tt_input, k=k, block_select=32, mode_output=_scalar(device, 0))


def test_two_level_rejects_other_block_sizes(device, expect_error):
    """block_select other than 32 raises, also when the single-level program for the shape is already cached."""
    tt_input = _dev(torch.zeros(1, 1, 1, 131_072, dtype=torch.bfloat16), device)
    with expect_error(RuntimeError, "supports 32 only"):
        ttnn.experimental.topk_large_indices(tt_input, k=K, block_select=16)
    narrow = _dev(torch.zeros(1, 1, 1, 4096, dtype=torch.bfloat16), device)
    ttnn.experimental.topk_large_indices(narrow, k=K)
    with expect_error(RuntimeError, "supports 32 only"):
        ttnn.experimental.topk_large_indices(narrow, k=K, block_select=16)


def test_two_level_row_count_changes(device):
    """The program's own maxima tensor is sized by the row count, which keys the program: more rows at the same
    width get a program of their own."""
    # A width no other test uses, so neither row count's program is cached already.
    width, n = 99_360, 99_360
    entries = None
    for rows in (2, 3, 2):
        torch_input = generate("iid", rows, width, 18 + rows, n=n)
        out = ttnn.experimental.topk_large_indices(
            _dev(torch_input.reshape(1, 1, rows, width), device),
            k=K,
            valid_length=n,
            block_select=32,
            two_level_cutover=FORCE_TWO_LEVEL,
        )
        _check_rows(torch_input, _indices(out, rows), n, range(rows))
        if entries is None:
            entries = device.num_program_cache_entries()
    assert device.num_program_cache_entries() == entries + 1


def test_two_level_rejects_maxima_input_with_maxima_output(device, expect_error):
    rows, width = 2, 131_072
    tt_input = _dev(torch.zeros(1, 1, rows, width, dtype=torch.bfloat16), device)
    maxima = lambda: _dev(torch.zeros(1, 1, rows, width // 32, dtype=torch.bfloat16), device)
    with expect_error(RuntimeError, "with block_max_tensor none are computed"):
        ttnn.experimental.topk_large_indices(
            tt_input, k=K, block_select=32, block_max_tensor=maxima(), block_max_output=maxima()
        )


@pytest.mark.parametrize("rows", [100, 90])
def test_two_level_cores_with_several_rows(device, rows):
    """Up to four rows per core on 27 cores, each tracked by its own byte of the owner's readiness count. 100 rows: 19
    left over after three rounds, too many to pair, so four whole rows on some cores. 90 rows: 9 split rows on 9
    pairs, whose first cores wait on byte 3."""
    width, n = 131_072, 120_000
    torch_input = generate("iid", rows, width, 17, n=n)
    mode = _scalar(device, 7)
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length=n,
        block_select=32,
        two_level_cutover=FORCE_TWO_LEVEL,
        sub_core_grids=_grid(device, 3),
        mode_output=mode,
    )
    assert _mode(mode) == 1
    _check_rows(torch_input, _indices(out, rows), n, range(rows))


def test_two_level_rejects_more_than_four_rows_per_core(device, expect_error):
    rows, width = 120, 65_536
    tt_input = _dev(torch.zeros(1, 1, rows, width, dtype=torch.bfloat16), device)
    with expect_error(RuntimeError, "at most 4 rows"):
        ttnn.experimental.topk_large_indices(tt_input, k=K, block_select=32, sub_core_grids=_grid(device, 3))


def test_two_level_fallback_reuses_the_single_level_program(device):
    """Outside the two-level range a block_select call is the single-level program, cache entry included."""
    tt_input = _dev(generate("iid", 2, 131_088, 4).reshape(1, 1, 2, 131_088), device)
    ttnn.experimental.topk_large_indices(tt_input, k=K)
    entries = device.num_program_cache_entries()
    ttnn.experimental.topk_large_indices(tt_input, k=K, block_select=32)
    assert device.num_program_cache_entries() == entries


def test_two_level_rejects_bad_block_max_tensor(device, expect_error):
    rows, width = 2, 131_072
    tt_input = _dev(torch.zeros(1, 1, rows, width, dtype=torch.bfloat16), device)
    with expect_error(RuntimeError, "block_max_tensor has shape"):
        ttnn.experimental.topk_large_indices(
            tt_input,
            k=K,
            block_select=32,
            block_max_tensor=_dev(torch.zeros(1, 1, rows, width // 32 - 32, dtype=torch.bfloat16), device),
        )
    with expect_error(RuntimeError, "mode_output must not share a buffer"):
        shared = _scalar(device, 70_000)
        ttnn.experimental.topk_large_indices(
            tt_input, k=K, block_select=32, valid_length_tensor=shared, mode_output=shared
        )


def test_two_level_width_changes_on_cache_hits(device):
    """Widths sharing every other program class get their own programs: bitmap, scratch and maxima pitch follow the
    width. Fresh input, output, maxima and mode buffers on a hit are picked up."""
    device.enable_program_cache()
    rows = 160
    for width, n in ((70_016, 70_000), (131_072, 131_072), (70_016, 65_000), (131_072, 100_000)):
        torch_input = generate("iid", rows, width, width % 97, n=n)
        blocks = width // BLOCK
        maxima = _dev(torch.zeros(1, 1, rows, blocks, dtype=torch.bfloat16), device)
        mode = _scalar(device, 0)
        out = ttnn.experimental.topk_large_indices(
            _dev(torch_input.reshape(1, 1, rows, width), device),
            k=K,
            valid_length=n,
            block_select=32,
            two_level_cutover=FORCE_TWO_LEVEL,
            block_max_output=maxima,
            mode_output=mode,
            sub_core_grids=_grid(device, 12),
        )
        assert _mode(mode) == 1
        _check_rows(torch_input, _indices(out, rows), n, (0, 107, 108, 159))
        got = ttnn.to_torch(maxima).reshape(rows, blocks).float()
        nb = (n + BLOCK - 1) // BLOCK
        for row in (0, 13, 108, 159):
            assert torch.equal(normalize(got[row, :nb]), _block_maxima(torch_input[row], n)), f"{width} row {row}"


def test_two_level_largest_width_class(device):
    """The 2,097,152-wide class over the whole width: 32 maxima chunks, the largest scratch and bitmap, and a winner
    in the last key."""
    rows, width = 2, 2_097_152
    n = width
    torch_input = generate("iid", rows, width, 12, n=n)
    torch_input[:, width - 1] = 100.0
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length=n,
        block_select=32,
        two_level_cutover=FORCE_TWO_LEVEL,
    )
    _check_rows(torch_input, _indices(out, rows), n, range(rows))


def test_two_level_noc1_reads(device):
    rows, width, n = 160, 524_320, 400_000
    torch_input = generate("iid", rows, width, 13, n=n)
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length=n,
        block_select=32,
        two_level_cutover=FORCE_TWO_LEVEL,
        dram_read_noc=ttnn.NOC.NOC_1,
        sub_core_grids=_grid(device, 12),
    )
    _check_rows(torch_input, _indices(out, rows), n, (0, 107, 108, 159))


@pytest.mark.parametrize("winning_blocks", [2047, 2048, 2049])
def test_two_level_winning_block_count_at_the_limit(device, winning_blocks):
    """The top value sits in exactly `winning_blocks` blocks: one fewer than, equal to, and one more than the 2048
    blocks selected."""
    rows, width = 2, 131_072
    torch_input = torch.full((rows, width), -2.0, dtype=torch.bfloat16)
    blocks = torch.randperm(width // BLOCK, generator=torch.Generator().manual_seed(winning_blocks))[:winning_blocks]
    torch_input[:, blocks * BLOCK + 7] = 3.0
    torch_input[:, 1::64] = torch.maximum(torch_input[:, 1::64], torch.tensor(1.0, dtype=torch.bfloat16))
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length=width,
        block_select=32,
        two_level_cutover=FORCE_TWO_LEVEL,
    )
    _check_rows(torch_input, _indices(out, rows), width, range(rows))


# The default cutover per top-k grid (two_level_default_cutover): 12 columns are 108 cores, 11 are 99.
DEFAULT_CUTOVER = {12: 196_608, 11: 147_456}


@pytest.mark.parametrize("columns", [12, 11])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_two_level_default_cutover(device, columns, offset):
    """The default cutover on both top-k grids: rows run two-level from exactly the cutover, and a scalar and a
    metadata bound choose the same mode and the same ordered indices."""
    rows, width = 160, 262_144
    n = DEFAULT_CUTOVER[columns] + offset
    torch_input = generate("ties64", rows, width, 14, n=n)
    tt_input = _dev(torch_input.reshape(1, 1, rows, width), device)
    results = []
    for length in ({"valid_length": n}, {"valid_length_tensor": _scalar(device, n)}):
        mode = _scalar(device, 7)
        out = ttnn.experimental.topk_large_indices(
            tt_input,
            k=K,
            **length,
            block_select=32,
            sub_core_grids=_grid(device, columns),
            dram_read_noc=ttnn.NOC.NOC_0,
            mode_output=mode,
        )
        assert _mode(mode) == (1 if offset >= 0 else 0)
        results.append(_indices(out, rows))
    assert torch.equal(results[0], results[1]), "scalar and metadata bounds returned different ordered indices"
    _check_rows(torch_input, results[0], n, (0, 98, 107, 108, 159))


@pytest.mark.parametrize("columns", [12, 11])
def test_two_level_default_long_rows(device, columns):
    """The production long case on both top-k grids under the default cutover."""
    rows, width, n = 160, 1_049_600, 1_048_576
    torch_input = generate("iid", rows, width, 16, n=n)
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length_tensor=_scalar(device, n),
        block_select=32,
        sub_core_grids=_grid(device, columns),
        dram_read_noc=ttnn.NOC.NOC_0,
    )
    _check_rows(torch_input, _indices(out, rows), n, (0, 51, 52, 98, 107, 108, 143, 144, 159))


@pytest.mark.parametrize("n", [1_500, 3_000])
def test_two_level_program_short_single_level_rows(device, n):
    """A two-level program running single-level on rows of one or two chunks, with split pairs, in scalar mode."""
    rows, width = 3, 131_072
    torch_input = generate("iid", rows, width, 15, n=n)
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length=n,
        block_select=32,
        two_level_cutover=FORCE_SINGLE_LEVEL,
    )
    _check_rows(torch_input, _indices(out, rows), n, range(rows))


def test_single_level_load_order_zero_class(device):
    """The load order two-level selection relies on: topk_large_indices ranks subnormals of either sign and -0 equal
    to +0, between -min_normal and +min_normal."""
    n = 4096
    words = torch.full((n,), 0xBF80, dtype=torch.int32)
    zero_class = {10: 0x0001, 20: 0x0040, 30: 0x007F, 40: 0x0000, 50: 0x8000, 60: 0x8001}
    for position, word in zero_class.items():
        words[position] = word
    words[70] = 0x0080
    words[80] = 0x8080
    # The control row holds +0 at every zero-class position: both rows must rank the same.
    control = words.clone()
    for position in zero_class:
        control[position] = 0x0000
    results = []
    for row_words in (words, control):
        row = (row_words << 16).view(torch.float32).to(torch.bfloat16)
        out = ttnn.experimental.topk_large_indices(_dev(row.reshape(1, 1, 1, n), device), k=16)
        results.append(ttnn.to_torch(out, dtype=torch.uint32).reshape(-1).to(torch.int64).tolist())
    indices, control_indices = results
    assert indices[0] == 70
    assert set(indices[1:7]) == set(zero_class)
    assert indices[7] == 80
    # Equal values, so the same tie order as the control: not ranked by magnitude.
    assert indices[:8] == control_indices[:8]
