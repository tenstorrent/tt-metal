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
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length=n,
        block_select=32,
        two_level_cutover=FORCE_TWO_LEVEL,
    )
    _check_rows(torch_input, _indices(out, rows), n, range(rows))


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


def test_two_level_supplied_block_maxima(device):
    rows, width, n = 4, 1_049_600, 700_000
    torch_input = generate("sinks", rows, width, 9, n=n)
    blocks = width // BLOCK
    maxima = torch.full((rows, blocks), float("-inf"))
    for row in range(rows):
        nb = (n + BLOCK - 1) // BLOCK
        maxima[row, :nb] = _block_maxima(torch_input[row], n)
    out = ttnn.experimental.topk_large_indices(
        _dev(torch_input.reshape(1, 1, rows, width), device),
        k=K,
        valid_length=n,
        block_select=32,
        two_level_cutover=FORCE_TWO_LEVEL,
        block_max_tensor=_dev(maxima.to(torch.bfloat16).reshape(1, 1, rows, blocks), device),
    )
    _check_rows(torch_input, _indices(out, rows), n, range(rows))


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
        for n, end in ((5_000, 5_000), (1_048_576, 1_048_576), (400_000, 399_990), (299_999, 299_999), (5_000, 5_000)):
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


@pytest.mark.parametrize(
    "k,width,block_select,message",
    [
        (1024, 131_072, 32, "requires k = 2048"),
        (K, 131_088, 32, "multiple of 32"),
        (K, 65_504, 32, "requires a last dimension"),
        (K, 131_072, 16, "supports 32 only"),
    ],
)
def test_two_level_rejects_unsupported_shapes(device, expect_error, k, width, block_select, message):
    tt_input = _dev(torch.zeros(1, 1, 1, width, dtype=torch.bfloat16), device)
    with expect_error(RuntimeError, message):
        ttnn.experimental.topk_large_indices(tt_input, k=k, block_select=block_select)
