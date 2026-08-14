# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RTL tests for Quasar max_pool_with_indices."""

import sys

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import MaxPoolWithIndicesGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MAX_POOL_CONFIG,
    NUM_FACES,
    SFPU_TILE_INDICES,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.utils import passed_test

TILE_ROWS = 32
TILE_COLS = 32
FACE_DIM = 16
OUT_OF_WINDOW_SENTINEL = 100.0
BF16_FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
FP32_FORMATS = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)

# Exercise every supported row count, both short-path layouts, the three generic
# sizes, both accumulation phases, and 16-bit and 32-bit Dest index transport.
CASES = [
    pytest.param(BF16_FORMATS, DestAccumulation.No, 4, False, False, 0, id="tile-4"),
    pytest.param(BF16_FORMATS, DestAccumulation.No, 8, False, False, 0, id="tile-8"),
    pytest.param(BF16_FORMATS, DestAccumulation.No, 9, False, False, 0, id="tile-9"),
    pytest.param(
        BF16_FORMATS, DestAccumulation.No, 4, True, False, 0, id="row-major-4"
    ),
    pytest.param(
        BF16_FORMATS, DestAccumulation.No, 8, True, False, 0, id="row-major-8"
    ),
    pytest.param(
        BF16_FORMATS, DestAccumulation.No, 9, True, False, 0, id="row-major-9"
    ),
    pytest.param(
        BF16_FORMATS, DestAccumulation.No, 16, True, False, 0, id="row-major-16"
    ),
    pytest.param(
        BF16_FORMATS, DestAccumulation.No, 20, True, False, 0, id="row-major-20"
    ),
    pytest.param(
        BF16_FORMATS, DestAccumulation.No, 32, True, False, 0, id="row-major-32"
    ),
    pytest.param(
        BF16_FORMATS,
        DestAccumulation.No,
        32,
        True,
        True,
        0,
        id="row-major-32-accumulate-seed",
    ),
    pytest.param(
        BF16_FORMATS,
        DestAccumulation.No,
        32,
        True,
        True,
        1,
        id="row-major-32-accumulate-fold",
    ),
    pytest.param(
        FP32_FORMATS,
        DestAccumulation.Yes,
        32,
        True,
        True,
        1,
        id="row-major-32-fp32-accumulate-fold",
    ),
]


def _tilize(tile_2d: torch.Tensor) -> torch.Tensor:
    """Face-major flattening used by the TILE-layout reducer."""
    return torch.cat(
        [
            tile_2d[
                face_row * FACE_DIM : (face_row + 1) * FACE_DIM,
                face_col * FACE_DIM : (face_col + 1) * FACE_DIM,
            ].reshape(-1)
            for face_row in range(2)
            for face_col in range(2)
        ]
    )


def _stage(tile_2d: torch.Tensor, row_major: bool) -> torch.Tensor:
    """Arrange L1 chunks in the order consumed by the selected Dest layout."""
    return tile_2d.reshape(-1) if row_major else _tilize(tile_2d)


def _result_row0(tile: torch.Tensor, row_major: bool) -> torch.Tensor:
    """Read the 32 result columns from the physical layout used by the kernel."""
    if row_major:
        return tile[:TILE_COLS]
    return torch.cat([tile[:FACE_DIM], tile[256 : 256 + FACE_DIM]])


def _encode_indices(indices: torch.Tensor, row_major: bool, torch_format):
    index_dtype = (
        torch.int32
        if torch.empty(0, dtype=torch_format).element_size() == 4
        else torch.int16
    )
    return _stage(indices, row_major).to(index_dtype).contiguous().view(torch_format)


def _decode_indices(indices: torch.Tensor, torch_format) -> torch.Tensor:
    index_dtype = (
        torch.int32
        if torch.empty(0, dtype=torch_format).element_size() == 4
        else torch.int16
    )
    return indices.contiguous().view(index_dtype).to(torch.int64)


def _make_values(num_rows: int):
    """Negative and mixed-sign windows plus a larger out-of-window value.

    Even columns stay entirely negative to expose Quasar's raw SFPSWAP ordering
    bug. Odd columns contain both signs to cover the uncorrected comparison path.
    The positive sentinel immediately exposes any row beyond ``num_rows`` being
    included in the reduction.
    """
    generator = torch.Generator().manual_seed(1000 + num_rows)
    order = torch.argsort(torch.rand(num_rows, TILE_COLS, generator=generator), dim=0)
    column_fraction = torch.arange(TILE_COLS, dtype=torch.float32).remainder(4) / 8

    values = torch.full(
        (TILE_ROWS, TILE_COLS), OUT_OF_WINDOW_SENTINEL, dtype=torch.float32
    )
    values[:num_rows] = -(order.to(torch.float32) + 1 + column_fraction)
    values[:num_rows, 1::2] += 2.5
    return values


def _make_accumulator(poison: bool):
    """Create either a fold input or values that chunk zero must ignore."""
    fill_value = OUT_OF_WINDOW_SENTINEL if poison else -200.0
    values = torch.full((TILE_ROWS, TILE_COLS), fill_value, dtype=torch.float32)
    if not poison:
        values[0, 0::2] = -0.25
    indices = torch.zeros((TILE_ROWS, TILE_COLS), dtype=torch.int64)
    indices[0] = 2000 + torch.arange(TILE_COLS)
    return values, indices


@pytest.mark.quasar
@pytest.mark.parametrize(
    "formats,dest_acc,num_rows,row_major,accumulate,chunk",
    CASES,
)
def test_max_pool_indices_quasar(
    formats, dest_acc, num_rows, row_major, accumulate, chunk
):
    torch_format = format_dict[formats.input_format]

    values = _make_values(num_rows)
    indices = torch.arange(TILE_ROWS * TILE_COLS, dtype=torch.int64).reshape(
        TILE_ROWS, TILE_COLS
    )
    accum_values, accum_indices = _make_accumulator(poison=accumulate and chunk == 0)

    # Dest tiles are values/current, values/accumulator, indices/current,
    # indices/accumulator. Chunk zero gets a winning poison value that must be
    # ignored, while later chunks get a real prior result to fold.
    src_A = torch.cat(
        [
            _stage(values, row_major).to(torch_format),
            _stage(accum_values, row_major).to(torch_format),
            _encode_indices(indices, row_major, torch_format),
            _encode_indices(accum_indices, row_major, torch_format),
        ]
    )
    src_B = torch.zeros(TILE_ROWS * TILE_COLS, dtype=torch_format)

    configuration = TestConfig(
        "sources/quasar/sfpu_max_pool_indices_quasar_test.cpp",
        formats,
        templates=[
            DEST_SYNC(),
            MAX_POOL_CONFIG(num_rows, row_major, accumulate, chunk),
        ],
        runtimes=[
            TILE_COUNT(4),
            NUM_FACES(4),
            TEST_FACE_DIMS(),
            SFPU_TILE_INDICES(0, 2, 0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=4,
            tile_count_B=1,
            tile_count_res=2,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    result = torch.tensor(configuration.run().result, dtype=torch_format)
    assert len(result) == 2 * TILE_ROWS * TILE_COLS

    golden = get_golden_generator(MaxPoolWithIndicesGolden)(
        values.reshape(-1),
        indices.reshape(-1),
        num_rows,
        formats.output_format,
        "bits",
    )
    golden_values = golden[:TILE_COLS]
    golden_indices = _decode_indices(golden[TILE_COLS:], torch_format)

    if accumulate and chunk > 0:
        prior_values = accum_values[0].to(torch_format)
        prior_indices = accum_indices[0]
        prior_wins = prior_values > golden_values
        golden_values = torch.where(prior_wins, prior_values, golden_values)
        golden_indices = torch.where(prior_wins, prior_indices, golden_indices)

    result_values = _result_row0(result[: TILE_ROWS * TILE_COLS], row_major)
    result_indices = _decode_indices(
        _result_row0(result[TILE_ROWS * TILE_COLS :], row_major), torch_format
    )

    if not torch.equal(result_indices, golden_indices):
        print(
            f"max_pool_with_indices index mismatch: rows={num_rows}, "
            f"row_major={row_major}, accumulate={accumulate}, chunk={chunk}, "
            f"dest_acc={dest_acc.name}",
            file=sys.stderr,
        )
        print(f"got:  {result_indices.tolist()}", file=sys.stderr)
        print(f"want: {golden_indices.tolist()}", file=sys.stderr)
        assert False, "Winning indices do not match golden"

    assert passed_test(
        golden_values,
        result_values,
        formats.output_format,
        print_errors=True,
    ), "Reduced values do not match golden"
