# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Full fast-tilize test: unpack + math + pack → standard tilized output.

Input: row-major bf16 tensor [height, width].
Expected output: standard tilized tiles (4 faces of 16x16 per tile).
Uses TilizeGolden from the existing test infrastructure.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TilizeGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    NUM_GUARD_TILES,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test

# Warm reset is no longer needed per-test: _llk_pack_fast_tilize_uninit_ now
# restores all modified config (strides, addr_mods, X counter, MOP, replay buf).


TILE_R = 32
TILE_C = 32


@parametrize(
    formats=[
        *input_output_formats([DataFormat.Float16_b], same=True),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp8_b),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp4_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Bfp8_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Bfp4_b),
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dimensions=[
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 9),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 8),
        (3, 4),
        (3, 8),
        (4, 4),
        (4, 8),
    ],
)
def test_fast_tilize_full(formats, dest_acc, dimensions):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    input_height_tiles, input_width_tiles = dimensions
    assert input_width_tiles >= 1, "ct_dim must be >= 1"

    input_dimensions = [input_height_tiles * TILE_R, input_width_tiles * TILE_C]
    tile_count = input_height_tiles * input_width_tiles

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Standard tilize golden: row-major → tile format
    generate_golden = get_golden_generator(TilizeGolden)
    golden_tensor = generate_golden(src_A, input_dimensions, formats.output_format)

    configuration = TestConfig(
        "sources/fast_tilize_bh_test.cpp",
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(1),
            NUM_FACES(4),
            NUM_GUARD_TILES(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_count,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result length {len(res_from_L1)} != golden length {len(golden_tensor)}"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    # Dump output mapping for debugging pack
    FACE_C = 16
    src_2d = src_A.reshape(input_height_tiles * TILE_R, input_width_tiles * TILE_C)
    input_rows = {}
    for r in range(input_height_tiles * TILE_R):
        for cg in range(input_width_tiles * TILE_C // FACE_C):
            key = tuple(
                round(v, 3) for v in src_2d[r, cg * FACE_C : (cg + 1) * FACE_C].tolist()
            )
            input_rows[key] = (r, cg)

    print(f"\nOutput: {len(res_tensor)} datums = {len(res_tensor)//FACE_C} rows")
    for out_row in range(min(128, len(res_tensor) // FACE_C)):
        start = out_row * FACE_C
        row_data = tuple(
            round(v, 3) for v in res_tensor[start : start + FACE_C].tolist()
        )
        match = input_rows.get(row_data)
        all_zero = all(abs(v) < 1e-6 for v in row_data)
        tile = out_row // (4 * 16)  # tile = every 64 rows
        face = (out_row % 64) // 16
        fr = out_row % 16
        label = (
            f"input[{match[0]},cols {match[1]*16}:{match[1]*16+15}]"
            if match
            else ("ZERO" if all_zero else "???")
        )
        print(f"  row {out_row:3d} (tile {tile} face {face} r{fr:2d}): {label}")

    print()

    # For Bfp4_b, _bfp4_block_aware_compare inside passed_test tilizes its
    # inputs to align BFP blocks, but our golden/result are already tilized.
    # Compare directly on the tilized data using the block-aware ULP comparator
    # without the extra tilize step.
    if formats.output_format == DataFormat.Bfp4_b:
        import math

        BLOCK = 16
        BFP4_MANTISSA_BITS = 3
        MAX_ULP_DIFF = 2
        g = golden_tensor.float()
        r = res_tensor.float()
        n = g.numel()
        all_ok = True
        for blk in range(0, n, BLOCK):
            g_blk = g[blk : blk + BLOCK]
            r_blk = r[blk : blk + BLOCK]
            finite = torch.cat(
                [g_blk[torch.isfinite(g_blk)].abs(), r_blk[torch.isfinite(r_blk)].abs()]
            )
            if finite.numel() == 0:
                continue
            bmax = finite.max().item()
            if bmax == 0:
                if not (g_blk == r_blk).all():
                    all_ok = False
                continue
            one_ulp = 2.0 ** (math.floor(math.log2(bmax)) - BFP4_MANTISSA_BITS + 1)
            if not ((g_blk - r_blk).abs() <= MAX_ULP_DIFF * one_ulp).all():
                all_ok = False
                break
        assert all_ok, "Bfp4_b output doesn't match golden (block-aware ULP check)"
    else:
        assert passed_test(
            golden_tensor, res_tensor, formats.output_format
        ), "Output doesn't match TilizeGolden"


# ============================================================
# Large-dimension fast tilize accuracy.
# Tests wide tile rows (20×4 to 120×4) to catch address overflow
# or configuration bugs that only manifest with many tiles.
# ============================================================
@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.No],
    dimensions=[
        # ct_dim=4: 40 rows passed before, find boundary
        (4, 1),
        (8, 1),
        (20, 1),
        (40, 1),
        (4, 2),
        (8, 2),
        (20, 2),
        (40, 2),
        (4, 4),
        (20, 4),
        (40, 4),
        (50, 4),
    ],
)
def test_fast_tilize_large(formats, dest_acc, dimensions):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    input_height_tiles, input_width_tiles = dimensions
    input_dimensions = [input_height_tiles * TILE_R, input_width_tiles * TILE_C]
    tile_count = input_height_tiles * input_width_tiles

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(TilizeGolden)
    golden_tensor = generate_golden(src_A, input_dimensions, formats.output_format)

    configuration = TestConfig(
        "sources/fast_tilize_bh_test.cpp",
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(1),
            NUM_FACES(4),
            NUM_GUARD_TILES(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_count,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result length {len(res_from_L1)} != golden length {len(golden_tensor)}"
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), f"Output doesn't match TilizeGolden for {dimensions}"


# ============================================================
# Fast tilize with L1 overflow detection.
# Allocates 1 extra guard tile after the result buffer.
# After tilize, checks the guard tile is all zeros (untouched).
# Catches PACR_FLUSH overflow that writes beyond the last tile.
# ============================================================
@parametrize(
    formats=[
        *input_output_formats([DataFormat.Float16_b], same=True),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp8_b),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp4_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b),
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dimensions=[
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 2),
        (2, 4),
        (4, 4),
    ],
)
def test_fast_tilize_overflow_guard(formats, dest_acc, dimensions):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    input_height_tiles, input_width_tiles = dimensions
    input_dimensions = [input_height_tiles * TILE_R, input_width_tiles * TILE_C]
    tile_count = input_height_tiles * input_width_tiles
    guard_tiles = (
        5  # enough to capture full ct_dim overflow + 1 sentinel validation tile
    )

    # Use constant input (0.5) so corrupted values are identifiable:
    # if guard has 0.5 → tilize output data leaked; if 0.0 → ZeroWrite from PACR_FLUSH
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        const_face=True,
        const_value_A=0.5,
    )

    # Golden for the data tiles
    generate_golden = get_golden_generator(TilizeGolden)
    golden_tensor = generate_golden(src_A, input_dimensions, formats.output_format)

    configuration = TestConfig(
        "sources/fast_tilize_bh_test.cpp",
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(1),
            NUM_FACES(4),
            NUM_GUARD_TILES(guard_tiles),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_count + guard_tiles,  # extra guard tile
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    tile_size = TILE_R * TILE_C

    # Skip accuracy check here — test_fast_tilize_full already covers that.
    # This test only checks for L1 overflow via sentinel guard.

    # Print buffer layout for debugging
    stim = configuration.variant_stimuli
    guard_byte_addr = stim.buf_res_addr + tile_count * stim.buf_res_tile_size
    print(
        f"  buf_res=0x{stim.buf_res_addr:x} tile_size={stim.buf_res_tile_size} tiles={tile_count} guard=0x{guard_byte_addr:x}"
    )

    # Read corruption counts from the last guard tile.
    # C++ checks sentinel at raw uint16 level and writes:
    #   word[0] = 0x4680 (marker), word[1..4] = corrupted count per guard tile.
    import struct

    from ttexalens.tt_exalens_lib import read_from_device

    stim = configuration.variant_stimuli
    last_g_addr = (
        stim.buf_res_addr + (tile_count + guard_tiles - 1) * stim.buf_res_tile_size
    )
    core_loc = "0,0"  # already "x,y" string
    raw = read_from_device(core_loc, last_g_addr, num_bytes=10)
    marker = struct.unpack_from("<H", raw, 0)[0]
    assert (
        marker == 0x4680
    ), f"Sentinel check marker missing (got 0x{marker:04x}), C++ check didn't run"
    for g in range(guard_tiles - 1):
        count = struct.unpack_from("<H", raw, (g + 1) * 2)[0]
        tile_words = stim.buf_res_tile_size // 2
        print(f"  Guard[{g}]: {count}/{tile_words} uint16 corrupted (raw byte check)")
        assert (
            count == 0
        ), f"L1 overflow: Guard[{g}] has {count}/{tile_words} corrupted uint16 words (dims={dimensions})"
