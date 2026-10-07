# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# max_pool_with_indices: column-wise arg-max over the rows of a values tile, carrying an
# indices tile in lockstep; both are reduced in place into their row 0. Covers every path of
# ckernel_sfpu_max_pool_indices.h: the TILE 9-row network, the ROW_MAJOR 9-row network, the
# ROW_MAJOR 32-row walk, and 32-row accumulation (chunk 0 seeding, chunk > 0 folding) - the
# ROW_MAJOR paths are the ones ttnn's Quasar compute_mpwi.cpp uses.

from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import MaxPoolWithIndicesGolden, get_golden_generator
from helpers.llk_params import (
    ApproximationMode,
    ImpliedMathFormat,
    PerfRunType,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_quasar_sfpu_format_variants,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    MAX_POOL_CHUNK,
    MAX_POOL_WITH_INDICES,
    NUM_FACES,
    SFPU_TILE_INDICES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import MAX_NUM_FACES, MAX_TILE_ELEMENTS
from helpers.tilize_untilize import tilize_block, untilize_block


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed the RNG once per test so stimuli are deterministic across runs."""
    torch.manual_seed(42)


_CPP_SOURCE = "sources/quasar/sfpu_max_pool_with_indices_quasar_test.cpp"
_TILE_DIMS = [32, 32]
# Every variant stages 4 Dest tiles so the running-max tiles above the operands exist.
_TILE_CNT = 4
# Logical column -> the two in-window rows that both hold that column's maximum.
_TIE_COLUMNS = {3: (2, 7), 20: (0, 8)}
# Rows past a 9-row window exceed every reduced value, so reducing past row 8 changes row 0.
_OUT_OF_WINDOW_VALUE = 1000.0
# Chunk-0 running tiles hold this; seeding must overwrite it rather than fold it in.
_STALE_RUNNING_VALUE = 5000.0


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

    TILE layout also checks operands placed at Dest tiles 2/3. ROW_MAJOR uses compute_mpwi's
    values=0 / indices=2 layout, whose accumulating variant keeps the running max in 1 / 3.
    """
    if not build.row_major:
        return [(0, 1, 0), (0, 1, 1), (2, 3, 2), (2, 3, 3)]
    if build.accumulate:
        return [(0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3)]
    return [(0, 2, 0), (0, 2, 2)]


_BUILDS = [
    MaxPoolBuild(num_rows=9, row_major=False),
    MaxPoolBuild(num_rows=9, row_major=True),
    MaxPoolBuild(num_rows=32, row_major=True),
    MaxPoolBuild(num_rows=32, row_major=True, accumulate=True),
]
# (build, chunk, tile indices): chunk and tile indices are runtime, so each build compiles once.
# Accumulating builds run chunk 0 (seed the running max) and chunk 1 (fold into it).
_VARIANTS = [
    (build, runtime(chunk), runtime(tile_indices))
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


def _prepare_values(num_rows, data_format):
    """Rows 0 to num_rows-1 of every column hold distinct log-uniform values; columns cycle
    through all-negative, all-positive and mixed signs, and the maximum walks the window so
    every row wins in some column. _TIE_COLUMNS repeat the maximum on two rows."""
    torch_format = format_dict[data_format]
    values = torch.full((32, 32), _OUT_OF_WINDOW_VALUE, dtype=torch.float32)
    for col in range(32):
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
                break
        column = column[torch.randperm(num_rows)]
        tie_rows = _TIE_COLUMNS.get(col)
        max_row = tie_rows[0] if tie_rows else col % num_rows
        top = int(torch.argmax(column))
        column[[top, max_row]] = column[[max_row, top]]
        if tie_rows:
            column[tie_rows[1]] = column[max_row]
        values[:num_rows, col] = column
    return values.to(torch_format)


def _prepare_running(window_max, data_format):
    """Previous running max for chunk > 0: even columns sit just above the window's maximum
    (the previous chunk wins), odd columns just below it (the current chunk wins)."""
    torch_format = format_dict[data_format]
    window_max = window_max.to(torch.float32)
    step = window_max.abs().clamp(min=1.0) * 0.25
    above = torch.arange(32) % 2 == 0
    running = torch.where(above, window_max + step, window_max - step)
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
    what Unpack-to-Dest makes of a plain row-major buffer; TILE tiles are tilized."""
    if row_major:
        return tile.flatten()
    return tilize_block(tile.flatten(), _TILE_DIMS, data_format).flatten()


def _row0(result, row_major, data_format):
    tile = torch.tensor(result, dtype=format_dict[data_format])
    if row_major:
        return tile[:32]
    return untilize_block(tile, data_format, _TILE_DIMS).reshape(32, 32)[0]


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=[
        fmt_variant
        for fmt_variant in generate_quasar_sfpu_format_variants(
            None, input_output_formats([DataFormat.Float16_b, DataFormat.Float32])
        )
        if fmt_variant.unpack_to_dest
    ],
    variant=_VARIANTS,
)
def test_sfpu_max_pool_with_indices_quasar(formats_dest_acc, variant):
    """max_pool_with_indices over every layout / row-count / accumulate path, Float16_b and
    Float32, checking every tile the kernel writes."""
    build, chunk, (values_idx, indices_idx, checked_idx) = variant
    folds = build.accumulate and chunk > 0

    formats = formats_dest_acc.formats
    fmt = formats.input_format
    torch_format = format_dict[fmt]

    values = _prepare_values(build.num_rows, fmt)
    indices = _index_codes().to(torch_format)
    window_values = values.to(torch.float32)[: build.num_rows]
    window_indices = indices[: build.num_rows]

    if folds:
        running_values, running_indices = _prepare_running(
            window_values.max(dim=0).values, fmt
        )
    else:
        running_values, running_indices = _stale_running(fmt)

    tiles = [
        torch.zeros(MAX_TILE_ELEMENTS, dtype=torch_format) for _ in range(_TILE_CNT)
    ]
    tiles[values_idx] = _to_l1(values, build.row_major, fmt)
    tiles[indices_idx] = _to_l1(indices, build.row_major, fmt)
    if build.accumulate:
        tiles[values_idx + 1] = _to_l1(running_values, build.row_major, fmt)
        tiles[indices_idx + 1] = _to_l1(running_indices, build.row_major, fmt)
    buffer_A = torch.cat(tiles)

    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=(PerfRunType.L1_TO_L1,),
        test_config_kwargs={
            "test_name": _CPP_SOURCE,
            "formats": formats,
            "templates": [
                MAX_POOL_WITH_INDICES(
                    build.num_rows, build.row_major, build.accumulate
                ),
                APPROX_MODE(ApproximationMode.No),
                IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
                UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
                DEST_SYNC(),
            ],
            "runtimes": [
                TILE_COUNT(_TILE_CNT),
                NUM_FACES(MAX_NUM_FACES),
                TEST_FACE_DIMS(),
                SFPU_TILE_INDICES(values_idx, indices_idx, checked_idx),
                MAX_POOL_CHUNK(chunk),
                LOOP_FACTOR(1),
            ],
            "variant_stimuli": StimuliConfig(
                buffer_A,
                fmt,
                buffer_A[:MAX_TILE_ELEMENTS],  # dummy buffer_B (unused by kernel)
                fmt,
                formats.output_format,
                tile_count_A=_TILE_CNT,
                tile_count_B=1,
                tile_count_res=1,
                num_faces=MAX_NUM_FACES,
                sfpu=True,
            ),
            "unpack_to_dest": True,
            "dest_acc": formats_dest_acc.dest_acc,
        },
    )
    formats_dest_acc.apply_formats(configuration.formats_config)

    result = configuration.run().result
    assert len(result) == MAX_TILE_ELEMENTS
    res_row = _row0(result, build.row_major, formats.output_format)

    golden_values, _, _ = get_golden_generator(MaxPoolWithIndicesGolden)(
        values, indices, build.num_rows, formats.output_format
    )
    # Candidate (value, index) entries per column: the window rows, plus the previous running
    # max when a later chunk folds into it.
    cand_values, cand_indices = window_values, window_indices.to(torch.float32)
    if folds:
        prev_values = running_values.to(torch.float32)[:1]
        golden_values = torch.maximum(
            golden_values.to(torch.float32), prev_values[0]
        ).to(format_dict[formats.output_format])
        cand_values = torch.cat([cand_values, prev_values])
        cand_indices = torch.cat([cand_indices, running_indices.to(torch.float32)[:1]])

    desc = (
        f"{build}, chunk={chunk}, format={fmt}->{formats.output_format}, "
        f"dest_acc={formats_dest_acc.dest_acc}, tiles=({values_idx}, {indices_idx}), "
        f"checked={checked_idx}"
    )
    # The running-values tile sits at values_idx + 1 only when accumulating; otherwise TILE
    # layout's indices operand may occupy that slot.
    checks_values = checked_idx == values_idx or (
        build.accumulate and checked_idx == values_idx + 1
    )
    if checks_values:
        assert torch.equal(
            res_row, golden_values
        ), f"max_pool values mismatch ({desc}):\n  got {res_row}\n  exp {golden_values}"
        return

    # Indices: the returned entry must come from exactly one candidate holding the column max.
    got = res_row.to(torch.float32)
    wrong_columns = []
    for col in range(32):
        rows = (cand_indices[:, col] == got[col]).nonzero().flatten().tolist()
        if len(rows) != 1 or cand_values[rows[0], col] != cand_values[:, col].max():
            wrong_columns.append((col, float(got[col]), rows))
    assert (
        not wrong_columns
    ), f"max_pool indices mismatch ({desc}); (col, got, matching rows): {wrong_columns}"
