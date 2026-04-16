# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Phase 1+2 test for BH fast-tilize unpack + math.

Math accumulates 8 dvalids into one DEST half-bank (512 rows for bf16).
Standard pack reads 4 tiles from the DEST half.

DEST layout per unit (8 dvalids → 512 rows):
  dvalid 0: rows 0-7 (data), 8-15 (gap), 16-23 (data), 24-31 (gap),
            32-39 (data), 40-47 (gap), 48-55 (data), 56-63 (gap)
  dvalid 1: rows 64-71 (data), 72-79 (gap), ...
  ...
  dvalid 7: rows 448-455, 464-471, 480-487, 496-503

Standard pack reads 4 tiles sequentially (tile t = DEST rows t*128..(t+1)*128-1):
  Tile 0: 128 rows = face0 (rows 0-15), face1 (16-31), ... face7? No —
  Actually standard pack reads tile t as 4 faces × 16 rows starting at row t*64.

For validation: dump the DEST contents and check expected pattern.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)

TILE_R = 32
TILE_C = 32
FACE_C = 16
UNIT_DIM = 4


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    dest_acc=[DestAccumulation.No],
    dimensions=[(1, 4)],
)
def test_fast_tilize_unpack(formats, dest_acc, dimensions, workers_tensix_coordinates):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    input_height_tiles, input_width_tiles = dimensions
    assert input_width_tiles % 4 == 0, "ct_dim must be divisible by 4"

    input_dimensions = [input_height_tiles * TILE_R, input_width_tiles * TILE_C]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Pack outputs 4 tiles per unit (1 dest section)
    num_output_tiles = input_height_tiles * (input_width_tiles // UNIT_DIM) * UNIT_DIM

    configuration = TestConfig(
        "sources/fast_tilize_phase1_test.cpp",
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(num_output_tiles),
            LOOP_FACTOR(1),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=num_output_tiles,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates).result
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    # Dump output for analysis: map each output row to input data
    src_2d = src_A.reshape(input_height_tiles * TILE_R, input_width_tiles * TILE_C)

    # Build lookup: input row + col_group → key
    input_rows = {}
    for r in range(input_height_tiles * TILE_R):
        for cg in range(input_width_tiles * TILE_C // FACE_C):
            key = tuple(
                round(v, 3) for v in src_2d[r, cg * FACE_C : (cg + 1) * FACE_C].tolist()
            )
            input_rows[key] = (r, cg)

    num_out_rows = len(res_tensor) // FACE_C
    print(
        f"\nOutput: {len(res_tensor)} datums = {num_out_rows} rows of 16, {num_output_tiles} tiles"
    )

    found_data = 0
    found_zero = 0
    for out_row in range(min(64, num_out_rows)):
        start = out_row * FACE_C
        row_data = tuple(
            round(v, 3) for v in res_tensor[start : start + FACE_C].tolist()
        )
        match = input_rows.get(row_data, None)
        all_zero = all(abs(v) < 1e-6 for v in row_data)
        tile = out_row // 16
        face_row = out_row % 16

        if match:
            label = f"input[{match[0]},cols {match[1]*16}:{match[1]*16+15}]"
            found_data += 1
        elif all_zero:
            label = "ZERO"
            found_zero += 1
        else:
            label = "???"

        print(f"  out_row {out_row:3d} (tile {tile} face_row {face_row:2d}): {label}")

    print(
        f"\nFound {found_data} data rows, {found_zero} zero rows out of {min(64, num_out_rows)} checked"
    )

    # Validate the exact expected pattern per the plan:
    # Standard pack reads 4 tiles from DEST. Each tile = 4 faces × 16 rows.
    # Face i of tile t = DEST rows (t*64 + i*16)..(t*64 + i*16 + 15).
    # Data rows: (t*64 + i*16 + 0..7) = tensor row (t*4 + i), all 128 cols.
    # Gap rows: (t*64 + i*16 + 8..15) = zeros.
    mismatches = 0
    for tile_idx in range(min(num_output_tiles, 4)):
        for face_idx in range(4):
            tensor_row_idx = tile_idx * 4 + face_idx
            if tensor_row_idx >= input_height_tiles * TILE_R:
                break
            base = (tile_idx * 4 + face_idx) * 256  # 16 rows × 16 cols per face
            # Check data rows (first 8 rows of face)
            for r in range(8):
                out_start = base + r * FACE_C
                expected_col_start = r * FACE_C
                expected = src_2d[
                    tensor_row_idx, expected_col_start : expected_col_start + FACE_C
                ]
                actual = res_tensor[out_start : out_start + FACE_C]
                if not torch.allclose(
                    actual.float(),
                    expected.to(format_dict[formats.output_format]).float(),
                    atol=0,
                    rtol=0,
                ):
                    mismatches += 1
                    if mismatches <= 3:
                        print(
                            f"Mismatch tile {tile_idx} face {face_idx} row {r}: "
                            f"got {actual[:4].tolist()}, expected {expected[:4].tolist()}"
                        )
            # Check gap rows (last 8 rows of face = zeros)
            gap_start = base + 8 * FACE_C
            gap = res_tensor[gap_start : gap_start + 8 * FACE_C]
            if not torch.all(gap == 0):
                mismatches += 1
                if mismatches <= 3:
                    print(f"Non-zero gap tile {tile_idx} face {face_idx}")

    assert mismatches == 0, f"{mismatches} mismatches"
