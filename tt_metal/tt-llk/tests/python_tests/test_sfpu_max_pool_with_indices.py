# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# max_pool_with_indices (Wormhole): column-wise arg-max over the rows of a values tile, carrying
# an indices tile in lockstep; both are reduced in place into their row 0. Covers the TILE and
# ROW_MAJOR 9-row networks, the ROW_MAJOR 32-row walk and 32-row accumulation (chunk 0 seeds the
# running max, chunk 1 folds into it). The 32-row paths pin the walk's tie order: the row each
# tie pattern resolves to and, on accumulation, the current chunk winning a positive tie with
# the running max and losing a negative one (SFPSWAP keeps its first operand only on positive
# ties).

from dataclasses import dataclass

import torch
from helpers.data_format_inference import unpacks_to_dest
from helpers.format_config import DataFormat
from helpers.golden_generators import MaxPoolWithIndicesGolden, get_golden_generator
from helpers.llk_params import ApproximationMode, DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DEST_SYNC,
    MAX_POOL_CHUNK,
    MAX_POOL_WITH_INDICES,
    SFPU_TILE_INDICES,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block, untilize_block

_CPP_SOURCE = "sources/sfpu_max_pool_with_indices_test.cpp"
_TILE_DIMS = [32, 32]
_TILE_ELEMENTS = 32 * 32
# Every variant stages 4 Dest tiles so the running-max tiles above the operands exist.
_TILE_CNT = 4
# Rows past a 9-row window exceed every reduced value, so reducing past row 8 changes row 0.
_OUT_OF_WINDOW_VALUE = 1000.0
# Chunk-0 running tiles hold this; seeding must overwrite it rather than fold it in.
_STALE_RUNNING_VALUE = 5000.0
# 32-row tie patterns (column i takes pattern i; even columns tie on a positive maximum, odd
# columns on a negative one) and the row the walk returns for each.
_TIE_ROWS = [
    ((0, 31), 31),
    ((1, 2), 1),
    ((7, 8), 8),
    ((15, 16), 15),
    ((30, 31), 30),
    ((1, 3), 3),
    ((2, 30), 30),
    ((5, 6, 7), 7),
    ((9, 10, 11), 10),
    ((3, 28), 3),
    ((17, 29), 17),
    ((14, 15), 15),
    (tuple(range(32)), 31),
    (tuple(range(1, 32, 2)), 31),
]
# Columns that tie on -inf over every row; the walk returns row 31.
_NEG_INF_COLUMNS = (30, 31)
_NEG_INF_ROW = 31
# -inf columns whose other rows hold the all-ones padding sentinel (a negative NaN that SFPSWAP
# orders below -inf), so a -inf row must win: column -> (sentinel rows, row returned).
_SENTINEL_COLUMNS = {
    28: (tuple(range(0, 32, 2)), 31),
    29: (tuple(range(16)) + tuple(range(17, 32, 2)), 30),
}


@dataclass(frozen=True)
class MaxPoolBuild:
    """Compile-time kernel configuration. ``num_rows`` is the 9-versus-32 dispatch selector."""

    num_rows: int
    row_major: bool
    accumulate: bool = False

    def __str__(self):
        layout = "ROW_MAJOR" if self.row_major else "TILE"
        return f"{layout}-{self.num_rows}{'-acc' if self.accumulate else ''}"


def _tile_index_variants(build):
    """(values_tile, indices_tile, checked_tile) triples: every tile the kernel writes.
    ROW_MAJOR uses compute_mpwi's values=0 / indices=2 layout, whose accumulating variant keeps
    the running max in 1 / 3."""
    if not build.row_major:
        return [(0, 1, 0), (0, 1, 1)]
    if build.accumulate:
        return [(0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3)]
    return [(0, 2, 0), (0, 2, 2)]


_BUILDS = [
    MaxPoolBuild(num_rows=9, row_major=False),
    MaxPoolBuild(num_rows=9, row_major=True),
    MaxPoolBuild(num_rows=32, row_major=True),
    MaxPoolBuild(num_rows=32, row_major=True, accumulate=True),
]
_VARIANTS = [
    (build, chunk, tile_indices)
    for build in _BUILDS
    for chunk in ((0, 1) if build.accumulate else (0,))
    for tile_indices in _tile_index_variants(build)
]


def _index_codes():
    """Distinct positive index-tile entries, exact in every float format: the row picks the
    exponent and the column a 5-bit mantissa. The kernel moves them as raw bits."""
    rows = torch.arange(32, dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(32, dtype=torch.float32).unsqueeze(0)
    return torch.pow(2.0, rows - 4) * (1 + cols / 32)


def _distinct_column(num_rows, col, torch_format):
    """num_rows distinct log-uniform values; columns cycle through all-negative, all-positive
    and mixed signs."""
    while True:
        magnitude = torch.pow(2.0, torch.empty(num_rows).uniform_(-6.0, 6.0))
        if col % 3 == 0:
            sign = -torch.ones(num_rows)
        elif col % 3 == 1:
            sign = torch.ones(num_rows)
        else:
            sign = torch.where(torch.rand(num_rows) < 0.5, -1.0, 1.0)
        column = (sign * magnitude).to(torch_format).to(torch.float32)
        if torch.unique(column).numel() == num_rows:
            return column[torch.randperm(num_rows)]


def _prepare_values(num_rows, data_format):
    """Window rows of every column hold distinct values with the maximum walking the window.
    32-row windows give the first columns the _TIE_ROWS patterns (alternating sign) and end on
    all -inf columns."""
    torch_format = format_dict[data_format]
    values = torch.full((32, 32), _OUT_OF_WINDOW_VALUE, dtype=torch.float32)
    for col in range(32):
        column = _distinct_column(num_rows, col, torch_format)
        top = int(torch.argmax(column))
        max_row = (7 * col) % num_rows
        column[[top, max_row]] = column[[max_row, top]]
        if num_rows == 32 and col < len(_TIE_ROWS):
            tie_rows = list(_TIE_ROWS[col][0])
            tie_value = column.max() if col % 2 == 0 else -column.abs().min()
            others = [r for r in range(32) if r not in tie_rows]
            column[others] = tie_value - 1 - column[others].abs()
            column[tie_rows] = tie_value
        if num_rows == 32 and (col in _NEG_INF_COLUMNS or col in _SENTINEL_COLUMNS):
            column[:] = float("-inf")
        values[:num_rows, col] = column
    values = values.to(torch_format)
    if num_rows == 32:
        sentinel = _padding_sentinel(torch_format)
        for col, (rows, _) in _SENTINEL_COLUMNS.items():
            values[list(rows), col] = sentinel
    return values


def _padding_sentinel(torch_format):
    """All-ones word: the value ttnn pads max-pool-with-indices windows with."""
    int_format = torch.int16 if torch_format == torch.bfloat16 else torch.int32
    return torch.tensor(-1, dtype=int_format).view(torch_format)


def _expected_row(window, col):
    """Row the 32-row walk returns for column col."""
    if col in _NEG_INF_COLUMNS:
        return _NEG_INF_ROW
    if col in _SENTINEL_COLUMNS:
        return _SENTINEL_COLUMNS[col][1]
    if col < len(_TIE_ROWS):
        return _TIE_ROWS[col][1]
    return int(torch.argmax(window[:, col]))


def _prepare_running(window_max, data_format):
    """Previous running max for chunk > 0: columns cycle through just above the window's maximum
    (the previous chunk wins), equal to it (a tie) and just below it (the current chunk wins).
    """
    torch_format = format_dict[data_format]
    window_max = window_max.to(torch.float32)
    step = window_max.abs().clamp(min=1.0) * 0.25
    phase = torch.arange(32) % 3
    running = torch.where(
        phase == 0,
        window_max + step,
        torch.where(phase == 1, window_max, window_max - step),
    )
    running = torch.where(torch.isinf(window_max), window_max, running)
    values = torch.full((32, 32), _STALE_RUNNING_VALUE, dtype=torch.float32)
    values[0] = running
    # Negated codes cannot collide with any window index code.
    indices = torch.zeros((32, 32), dtype=torch.float32)
    indices[0] = -_index_codes()[0]
    return values.to(torch_format), indices.to(torch_format)


def _stale_running(data_format):
    torch_format = format_dict[data_format]
    return (
        torch.full((32, 32), _STALE_RUNNING_VALUE).to(torch_format),
        torch.full((32, 32), -_STALE_RUNNING_VALUE).to(torch_format),
    )


def _to_l1(tile, row_major, data_format):
    """ROW_MAJOR tiles sit in Dest as logical rows (Face 0 Row r, Face 1 Row r, ...), which is
    what unpacking a plain row-major buffer makes of them; TILE tiles are tilized."""
    if row_major:
        return tile.flatten()
    return tilize_block(tile.flatten(), _TILE_DIMS, data_format).flatten()


def _row0(result, row_major, data_format):
    tile = torch.tensor(result, dtype=format_dict[data_format])
    if row_major:
        return tile[:32]
    return untilize_block(tile, data_format, _TILE_DIMS).reshape(32, 32)[0]


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    variant=_VARIANTS,
)
def test_sfpu_max_pool_with_indices(formats, variant):
    """max_pool_with_indices over every layout / row-count / accumulate path, Float16_b and
    Float32, checking every tile the kernel writes."""
    torch.manual_seed(42)
    build, chunk, (values_idx, indices_idx, checked_idx) = variant
    folds = build.accumulate and chunk > 0

    fmt = formats.input_format
    torch_format = format_dict[fmt]
    dest_acc = (
        DestAccumulation.Yes if fmt == DataFormat.Float32 else DestAccumulation.No
    )

    values = _prepare_values(build.num_rows, fmt)
    indices = _index_codes().to(torch_format)
    # The padding sentinel (NaN) ranks below every value, -inf included.
    ranked_values = values.to(torch.float32).nan_to_num(
        nan=float("-inf"), posinf=float("inf"), neginf=float("-inf")
    )
    window_values = ranked_values[: build.num_rows]
    window_indices = indices.to(torch.float32)[: build.num_rows]

    if folds:
        running_values, running_indices = _prepare_running(
            window_values.max(dim=0).values, fmt
        )
    else:
        running_values, running_indices = _stale_running(fmt)

    tiles = [torch.zeros(_TILE_ELEMENTS, dtype=torch_format) for _ in range(_TILE_CNT)]
    tiles[values_idx] = _to_l1(values, build.row_major, fmt)
    tiles[indices_idx] = _to_l1(indices, build.row_major, fmt)
    if build.accumulate:
        tiles[values_idx + 1] = _to_l1(running_values, build.row_major, fmt)
        tiles[indices_idx + 1] = _to_l1(running_indices, build.row_major, fmt)
    buffer_A = torch.cat(tiles)

    configuration = TestConfig(
        test_name=_CPP_SOURCE,
        formats=formats,
        templates=[
            MAX_POOL_WITH_INDICES(build.num_rows, build.row_major, build.accumulate),
            APPROX_MODE(ApproximationMode.No),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(_TILE_CNT),
            SFPU_TILE_INDICES(values_idx, indices_idx, checked_idx),
            MAX_POOL_CHUNK(chunk),
        ],
        variant_stimuli=StimuliConfig(
            buffer_A,
            fmt,
            buffer_A[:_TILE_ELEMENTS],  # dummy buffer_B (unused by the kernel)
            fmt,
            formats.output_format,
            tile_count_A=_TILE_CNT,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpacks_to_dest(fmt, dest_acc),
    )

    result = configuration.run().result
    assert len(result) == _TILE_ELEMENTS
    res_row = _row0(result, build.row_major, formats.output_format)

    golden_values, _, _ = get_golden_generator(MaxPoolWithIndicesGolden)(
        ranked_values, indices, build.num_rows, formats.output_format
    )
    prev_values = running_values.to(torch.float32)[0]
    prev_indices = running_indices.to(torch.float32)[0]
    if folds:
        golden_values = torch.maximum(golden_values.to(torch.float32), prev_values).to(
            format_dict[formats.output_format]
        )

    desc = (
        f"{build}, chunk={chunk}, format={fmt}, dest_acc={dest_acc}, "
        f"tiles=({values_idx}, {indices_idx}), checked={checked_idx}"
    )
    checks_values = checked_idx == values_idx or (
        build.accumulate and checked_idx == values_idx + 1
    )
    if checks_values:
        assert torch.equal(
            res_row, golden_values
        ), f"max_pool values mismatch ({desc}):\n  got {res_row}\n  exp {golden_values}"
        return

    got = res_row.to(torch.float32)
    wrong_columns = []
    for col in range(32):
        if build.num_rows == 32:
            expected = window_indices[_expected_row(window_values, col), col]
            window_max = window_values[:, col].max()
            prev_wins = prev_values[col] > window_max or (
                prev_values[col] == window_max and window_max < 0
            )
            if folds and prev_wins:
                expected = prev_indices[col]
            if got[col] != expected:
                wrong_columns.append((col, float(got[col]), float(expected)))
            continue
        # 9-row networks: any row holding the column maximum is a valid result.
        rows = (window_indices[:, col] == got[col]).nonzero().flatten().tolist()
        if len(rows) != 1 or window_values[rows[0], col] != window_values[:, col].max():
            wrong_columns.append((col, float(got[col]), rows))
    assert (
        not wrong_columns
    ), f"max_pool indices mismatch ({desc}); (col, got, expected): {wrong_columns}"
