# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.data_format_inference import unpacks_to_dest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import DestAccumulation, L1Accumulation, Tilize, format_dict
from helpers.param_config import (
    BlocksCalculationAlgorithm,
    DestSync,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    L1_ACC,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)
from helpers.utils import passed_test, result_tensor


def get_valid_tilize_datacopy(formats):
    """
    Get valid tilize options for _llk_math_eltwise_unary_datacopy_

    - Blackhole and Wormhole have differing APIs:
        - Blackhole: Has tilize argument (SW workaround for HW bug)
        - Wormhole: No tilize argument (No SW workaround needed)
    - Tilize cannot be enabled if input format is Bfp8_b (HW limitation)

    Therefore we only test tilization on Blackhole
    """

    chip_arch = get_chip_architecture()

    if chip_arch == ChipArchitecture.WORMHOLE:
        return [Tilize.No]

    if formats.input_format == DataFormat.Bfp8_b:
        return [Tilize.No]

    return [Tilize.No, Tilize.Yes]


def get_valid_num_faces_datacopy(tilize):
    """
    Get valid num_faces options for _llk_math_eltwise_unary_datacopy_

    - Number of faces must be 4 when tilization is enabled (SW limitation)
    - Otherwise num_faces can be 1, 2, or 4
    """

    if tilize == Tilize.Yes:
        return [4]

    return [1, 2, 4]


@parametrize(
    formats=InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    l1_acc=[L1Accumulation.No, L1Accumulation.Yes],
    num_faces=4,
    tilize=[Tilize.No],
    dest_index=0,
    input_dimensions=[[32, 32], [32, 64], [128, 32], [128, 64], [128, 256]],
)
def test_pack_dest_bank(
    formats, dest_acc, l1_acc, num_faces, tilize, dest_index, input_dimensions
):
    _run_pack_dest_bank(
        formats, dest_acc, l1_acc, num_faces, tilize, dest_index, input_dimensions
    )


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_pack_dest_bank_sync_full(formats, dest_acc):
    """SyncFull pack blocks with a 16-bit (Float16_b) pack source, [512, 32] = 16 tiles.

    dest_acc=No: one block of 16 tiles, the whole 16-bit DEST, twice the width the Blackhole
    init used to cap a 16-bit pack source at. dest_acc=Yes: two blocks of 8 tiles on a 32-bit
    DEST; the pack source stays Float16_b (it is the output format under dest accumulation), so
    the block width equals the old cap and this cell is a regression guard, not a fail-without.
    """
    _run_pack_dest_bank(
        formats,
        dest_acc,
        L1Accumulation.No,
        4,  # num_faces
        Tilize.No,
        0,  # dest_index
        [512, 32],  # input_dimensions
        dest_sync=DestSync.Full,
        pack_src=DataFormat.Float16_b,
    )


@parametrize(
    formats=input_output_formats([DataFormat.Float32, DataFormat.Int32], same=True),
    dest_acc=DestAccumulation.Yes,
)
def test_pack_dest_bank_sync_full_32bit_source(formats, dest_acc):
    """One SyncFull block of 8 tiles, the whole 32-bit DEST, with a 32-bit pack source.

    [256, 32] = 8 tiles. Float32: twice the width the Blackhole init used to cap a Float32 pack
    source at. Int32: the init used to have no bound for an integer source at all; the whole DEST
    has to stay accepted. The pack source is asserted to be the 32-bit output format so each case
    keeps exercising the 32-bit bound.
    """
    _run_pack_dest_bank(
        formats,
        dest_acc,
        L1Accumulation.No,
        4,  # num_faces
        Tilize.No,
        0,  # dest_index
        [256, 32],  # input_dimensions
        dest_sync=DestSync.Full,
        pack_src=formats.output_format,
    )


def _run_pack_dest_bank(
    formats,
    dest_acc,
    l1_acc,
    num_faces,
    tilize,
    dest_index,
    input_dimensions,
    dest_sync=DestSync.Half,
    pack_src=None,
):
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    torch_format = format_dict[formats.output_format]
    golden_tensor = src_A.to(torch_format)

    unpack_to_dest = unpacks_to_dest(formats.input_format, dest_acc, tilize)

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
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
        "sources/pack_dest_bank_test.cpp",
        formats,
        templates=[
            TILIZE(tilize),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            DEST_INDEX(dest_index),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            L1_ACC(l1_acc),
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
        l1_acc=l1_acc,
        unpack_to_dest=unpack_to_dest,
    )

    if pack_src is not None:
        # The init bound under test is keyed on the pack source's datum width; pin the inferred
        # source so the case cannot silently drift to a different bound.
        inferred_pack_src = configuration.formats_config[0].pack_src
        assert (
            inferred_pack_src == pack_src
        ), f"pack source inferred as {inferred_pack_src}, the case needs {pack_src}"

    configuration.generate_variant_hash()
    if TestConfig.BUILD_MODE in [BuildMode.PRODUCE, BuildMode.DEFAULT]:
        configuration.build_elfs()

    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    configuration.write_runtimes_to_L1()
    configuration.variant_stimuli.write(TestConfig.TENSIX_LOCATION)

    zero_tiles = torch.zeros(tile_cnt_A * 32 * 32, dtype=torch_format)
    zero_pack = StimuliConfig.get_packer(formats.output_format)
    StimuliConfig.write_matrix(
        zero_tiles,
        tile_cnt_A,
        zero_pack,
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


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    l1_acc=[L1Accumulation.No, L1Accumulation.Yes],
    num_faces=4,
    tilize=[Tilize.No],
    dest_index=0,
)
def test_pack_dest_bank_two_blocked_packs_of_4(
    formats, dest_acc, l1_acc, num_faces, tilize, dest_index
):
    if get_chip_architecture() != ChipArchitecture.WORMHOLE:
        pytest.skip("Wormhole-specific blocked pack regression")

    input_dimensions = [128, 64]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    torch_format = format_dict[formats.output_format]
    golden_tensor = src_A.to(torch_format)

    unpack_to_dest = unpacks_to_dest(formats.input_format, dest_acc)

    configuration = TestConfig(
        "sources/pack_dest_bank_test.cpp",
        formats,
        templates=[
            TILIZE(tilize),
            DEST_SYNC(DestSync.Half),
        ],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            DEST_INDEX(dest_index),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            L1_ACC(l1_acc),
            NUM_BLOCKS(2),
            NUM_TILES_IN_BLOCK(4),
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
        unpack_to_dest=unpack_to_dest,
    )

    configuration.generate_variant_hash()
    if TestConfig.BUILD_MODE in [BuildMode.PRODUCE, BuildMode.DEFAULT]:
        configuration.build_elfs()

    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    configuration.write_runtimes_to_L1()
    configuration.variant_stimuli.write(TestConfig.TENSIX_LOCATION)

    zero_tiles = torch.zeros(tile_cnt_A * 32 * 32, dtype=torch_format)
    zero_pack = StimuliConfig.get_packer(formats.output_format)
    StimuliConfig.write_matrix(
        zero_tiles,
        tile_cnt_A,
        zero_pack,
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
