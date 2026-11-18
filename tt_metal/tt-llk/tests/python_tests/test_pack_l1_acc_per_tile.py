# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.data_format_inference import unpacks_to_dest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    DataCopyGolden,
    TilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    L1Accumulation,
    PerfRunType,
    Tilize,
    format_dict,
)
from helpers.param_config import (
    BlocksCalculationAlgorithm,
    DestSync,
    get_num_blocks_and_num_tiles_in_block,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    L1_ACC,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PERF_RUN_TYPE,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)
from helpers.utils import passed_test, result_tensor


# Full-sync datacopy packed one tile per call (the byte-wise destination write), plain or after
# Blackhole's tilize init, with and without packer L1 accumulation into a zeroed output.
@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    l1_acc=[L1Accumulation.No, L1Accumulation.Yes],
    tilize=[Tilize.No, Tilize.Yes],
    input_dimensions=[[64, 64], [32, 256]],
)
def test_pack_l1_acc_per_tile(formats, dest_acc, l1_acc, tilize, input_dimensions):
    if tilize == Tilize.Yes and get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("Blackhole's tilize packs through the default pack")

    num_faces = 4
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    if tilize == Tilize.Yes:
        golden_tensor = get_golden_generator(TilizeGolden)(
            src_A, input_dimensions, formats.output_format
        )
    else:
        golden_tensor = get_golden_generator(DataCopyGolden)(
            src_A, formats.output_format, num_faces, input_dimensions
        )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Full,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        (
            BlocksCalculationAlgorithm.Standard
            if tilize == Tilize.No
            else BlocksCalculationAlgorithm.Tilize
        ),
    )

    configuration = TestConfig(
        "sources/pack_l1_acc_per_tile_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILIZE(tilize),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
        ],
        runtimes=[
            DEST_INDEX(0),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            LOOP_FACTOR(1),
            L1_ACC(l1_acc),
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
        l1_acc=l1_acc,
        unpack_to_dest=unpacks_to_dest(formats.input_format, dest_acc, tilize),
    )

    configuration.generate_variant_hash()
    if TestConfig.BUILD_MODE in [BuildMode.PRODUCE, BuildMode.DEFAULT]:
        configuration.build_elfs()

    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    configuration.write_runtimes_to_L1()
    configuration.variant_stimuli.write(TestConfig.TENSIX_LOCATION)

    torch_format = format_dict[formats.output_format]
    zero_tiles = torch.zeros(tile_cnt_A * 32 * 32, dtype=torch_format)
    StimuliConfig.write_matrix(
        zero_tiles,
        tile_cnt_A,
        StimuliConfig.get_packer(formats.output_format),
        configuration.variant_stimuli.buf_res_addr,
        configuration.variant_stimuli.buf_res_tile_size,
        num_faces,
        configuration.variant_stimuli.face_r_dim,
        TestConfig.TENSIX_LOCATION,
        configuration.variant_stimuli.write_full_tiles,
    )

    configuration.run_elf_files()
    configuration.wait_for_tensix_operations_finished()
    res_from_L1 = configuration.variant_stimuli.collect_results(
        TestConfig.TENSIX_LOCATION
    )

    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
