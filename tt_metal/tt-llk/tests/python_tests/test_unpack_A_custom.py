# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Standalone correctness test for the Blackhole experimental _llk_unpack_A_custom_ LLK.
# _llk_unpack_A_custom_(address) programs the SrcA base address and unpacks a full
# 32x32 tile to SrcA with dvalid. We drive it through an identity path
# (unpack_A_custom -> A2D datacopy to DEST -> pack out) and compare the packed
# result to the input via DataCopyGolden. This is the same kernel pipeline as
# sources/eltwise_unary_datacopy_custom_test.cpp (which already wires
# _llk_unpack_A_custom_); we reuse that source and drive it from a focused sweep
# so the two coverage paths cannot drift apart.
#
# The LLK is fixed to a full 4-face 32x32 tile (it hardcodes SETADCXX to 1023
# datums), and the kernel hardcodes 16-bit destination (dest_acc=No), so:
#   - num_faces is not swept (always 4);
#   - formats are restricted to combinations that do not require 32-bit dest
#     accumulation (i.e. the harness would not flip dest_acc on us).
# Instead we sweep formats and multiple tile counts, exercising the per-tile
# source L1 base-address programming inside _llk_unpack_A_custom_.

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    DataCopyGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
)
from helpers.utils import passed_test

# _llk_unpack_A_custom_ always unpacks a full 4-face 32x32 tile.
NUM_FACES_FULL_TILE = 4

# The custom kernel hardcodes 16-bit destination (dest_acc=No), so drop any
# format pair the harness would otherwise force onto a 32-bit destination.
UNPACK_A_CUSTOM_FORMATS = [
    fmt
    for fmt in input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float16, DataFormat.Bfp8_b]
    )
    if not is_dest_acc_needed(fmt)
]


@skip_for_wormhole
@skip_for_quasar
@parametrize(
    formats=UNPACK_A_CUSTOM_FORMATS,
    # Sweep tile counts to exercise several distinct source L1 addresses inside
    # _llk_unpack_A_custom_ (one base-address program per tile).
    input_dimensions=[[32, 32], [64, 64], [32, 256]],
    dest_acc=[DestAccumulation.No],
)
def test_unpack_A_custom(
    formats,
    input_dimensions,
    dest_acc,
):
    num_faces = NUM_FACES_FULL_TILE

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(
        src_A, formats.output_format, num_faces, input_dimensions
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS
    )

    configuration = TestConfig(
        "sources/eltwise_unary_datacopy_custom_test.cpp",
        formats,
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    outcome = configuration.run()
    res_from_L1 = outcome.result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result length {len(res_from_L1)} != golden length {len(golden_tensor)}"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "unpack_A_custom identity result does not match DataCopyGolden"
