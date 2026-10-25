# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Datacopy whose math/pack dest handshake is re-initialised between every block.

Functional smoke test of the per-block re-init: the pack thread calls ``_llk_pack_dest_init_`` (and the
math thread ``_llk_math_pack_sync_init_``) between consecutive blocks with no reconfig in between, so the
re-init lands directly behind the previous block's section-done. Both re-inits reset to DEST half 0, and
the math-side init waits for the MATH_PACK semaphore to drain, so no pack is in flight when the re-init
runs. The copy must equal its input across every block boundary; a re-init that broke the handshake or
the dest-offset registers shows up as a hang or a wrong tile.
"""

import torch
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.data_format_inference import unpacks_to_dest
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    DataCopyGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestSync,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test

REINIT_FORMATS = input_output_formats(
    [DataFormat.Float16_b, DataFormat.Float32, DataFormat.Bfp8_b], same=True
)

DEST_SYNC_MODE = DestSync.Half


@parametrize(
    formats=REINIT_FORMATS,
    dest_acc=get_valid_dest_accumulation_modes,
    input_dimensions=[[128, 128], [64, 512]],
)
def test_pack_dest_reinit(formats, dest_acc, input_dimensions):
    num_faces = 4

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        num_faces,
        input_dimensions,
        input_format=formats.input_format,
    )

    unpack_to_dest = unpacks_to_dest(formats.input_format, dest_acc)

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DEST_SYNC_MODE,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )
    assert num_blocks >= 2, "the test needs at least one re-init between blocks"

    configuration = TestConfig(
        test_name="sources/pack_dest_reinit_test.cpp",
        formats=formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            # Single-valued across the sweep: emitted as constexprs the kernel reads directly.
            DEST_SYNC(DEST_SYNC_MODE),
            DEST_INDEX(0),
            NUM_FACES(num_faces),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
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
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(
        golden_tensor
    ), f"result has {len(res_from_L1)} datums, golden has {len(golden_tensor)}"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(golden_tensor, res_tensor, formats.output_format), (
        "datacopy output differs from its input after the per-block _llk_pack_dest_init_ / "
        "_llk_math_pack_sync_init_ re-init"
    )
