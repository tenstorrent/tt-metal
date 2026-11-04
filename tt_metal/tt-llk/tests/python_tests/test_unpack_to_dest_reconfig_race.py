# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Race probe by Anil Mahmud (#62121), with unpacker 0's format words checked too, the result words read
# through ttexalens, and Wormhole enabled. Float32 unpack to Dest, then an immediate configure_unpack_AB for
# Float16_b; the kernel drains and checks the unpacker-0 ch1 Z-stride and format words after each reconfigure.

import struct

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import BlocksCalculationAlgorithm, DestAccumulation, DestSync, Tilize
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)
from ttexalens.tt_exalens_lib import read_from_device


@pytest.mark.parametrize("loop_factor", [64])
def test_unpack_to_dest_reconfig_race(loop_factor):
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes
    input_dimensions = [64, 64]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS, BlocksCalculationAlgorithm.Standard
    )
    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=None,
        test_config_kwargs={
            "test_name": "sources/unpack_to_dest_reconfig_race_test.cpp",
            "formats": formats,
            "templates": [generate_input_dim(input_dimensions, input_dimensions), TILIZE(Tilize.No)],
            "runtimes": [
                DEST_INDEX(0),
                TILE_COUNT(tile_cnt_A),
                NUM_FACES(4),
                NUM_BLOCKS(num_blocks),
                NUM_TILES_IN_BLOCK(num_tiles_in_block),
                LOOP_FACTOR(loop_factor),
            ],
            "variant_stimuli": StimuliConfig(
                src_A, formats.input_format, src_B, formats.input_format, formats.output_format,
                tile_count_A=tile_cnt_A, tile_count_B=tile_cnt_B, tile_count_res=tile_cnt_A, num_faces=4,
            ),
            "dest_acc": dest_acc,
            "unpack_to_dest": True,
        },
    )
    err = None
    try:
        configuration.run()
    except Exception as e:
        err = e
    raw = read_from_device(TestConfig.TENSIX_LOCATION, 0xA0000, num_bytes=20)
    stale, last_seen, trials, magic, stale_fmt = struct.unpack("<5I", bytes(raw))
    print(f"\nRACE PROBE: stale={stale}/{trials} last_seen_zstride={last_seen} stale_fmt={stale_fmt} magic={magic:#x} err={err!r}")
    assert magic == 0xC0FFEE00, f"probe words not written (run error: {err})"
    assert stale == 0 and stale_fmt == 0, f"race observed: Z-stride {stale}/{trials} (last {last_seen}), formats {stale_fmt}/{trials}"
