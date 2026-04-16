# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Probe: does BH pack HW auto-advance L1_Dest_addr on PACR Last=1?

Packs 2 tiles using a MOP with outerloop=2, BFP replay (Last=1), NO end_ops.
If tile 1 has correct data at tile_size offset → HW auto-advances.
If tile 1 is zeros or duplicate of tile 0 → HW does NOT auto-advance.
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

TILE_R = 32
TILE_C = 32


@parametrize(
    formats=[
        *input_output_formats([DataFormat.Float16_b], same=True),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp8_b),
    ],
    dest_acc=[DestAccumulation.No],
)
def test_pack_l1_autoadvance(formats, dest_acc, workers_tensix_coordinates):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    ct_dim = 2
    rt_dim = 1
    input_dimensions = [rt_dim * TILE_R, ct_dim * TILE_C]
    tile_count = rt_dim * ct_dim

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(TilizeGolden)
    golden_tensor = generate_golden(src_A, input_dimensions, formats.output_format)

    configuration = TestConfig(
        "sources/pack_l1_autoadvance_probe.cpp",
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

    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result length {len(res_from_L1)} != golden length {len(golden_tensor)}"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    # Check tile 0
    tile_size = TILE_R * TILE_C
    tile0 = res_tensor[:tile_size]
    tile1 = res_tensor[tile_size : 2 * tile_size]
    golden0 = golden_tensor[:tile_size]
    golden1 = golden_tensor[tile_size : 2 * tile_size]

    tile0_nonzero = tile0.abs().sum().item() > 0
    tile1_nonzero = tile1.abs().sum().item() > 0
    tile0_matches = torch.allclose(tile0.float(), golden0.float(), atol=0.1)
    tile1_matches = torch.allclose(tile1.float(), golden1.float(), atol=0.1)
    tiles_identical = torch.allclose(tile0.float(), tile1.float(), atol=1e-6)

    print(f"\n=== L1 Auto-Advance Probe ({formats.output_format}) ===")
    print(f"  Tile 0: nonzero={tile0_nonzero}, matches_golden={tile0_matches}")
    print(f"  Tile 1: nonzero={tile1_nonzero}, matches_golden={tile1_matches}")
    print(f"  Tiles identical: {tiles_identical}")

    if tile0_matches and tile1_matches:
        print("  VERDICT: HW AUTO-ADVANCES L1_Dest_addr on Last=1")
    elif tile0_matches and not tile1_nonzero:
        print("  VERDICT: HW does NOT auto-advance (tile 1 is zeros)")
    elif tile0_matches and tiles_identical:
        print("  VERDICT: HW does NOT auto-advance (tile 1 overwrote tile 0)")
    else:
        print(
            f"  VERDICT: INCONCLUSIVE (tile0_ok={tile0_matches}, tile1_ok={tile1_matches})"
        )
        print(f"  Tile 0 first 8: {tile0[:8].tolist()}")
        print(f"  Tile 1 first 8: {tile1[:8].tolist()}")
        print(f"  Golden 0 first 8: {golden0[:8].tolist()}")
        print(f"  Golden 1 first 8: {golden1[:8].tolist()}")

    # The test "passes" regardless — the verdict is informational.
    # The caller reads the verdict to decide Option A vs B.
    # But assert tile 0 is correct (sanity check that the probe ran).
    assert tile0_matches, "Probe sanity: tile 0 should match golden"
