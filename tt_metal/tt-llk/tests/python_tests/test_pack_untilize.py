# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.data_format_inference import infer_data_formats
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    NarrowTile,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    NARROW_TILE,
    NUM_FACES,
    TILE_COUNT,
    TILE_DST_CT_OFFSET,
    generate_input_dim,
)
from helpers.tile_constants import get_tile_params
from helpers.utils import passed_test


def _skip_unsupported(formats, dest_acc, input_dimensions, data_formats):
    if TestConfig.WITH_COVERAGE and input_dimensions == [64, 512]:
        pytest.skip(
            "Skipping large dimension test in coverage mode, check issue: #1063 on TT-LLK repo"
        )

    if get_chip_architecture() == ChipArchitecture.WORMHOLE and (
        formats.input_format == DataFormat.Fp8_e4m3
        or formats.output_format == DataFormat.Fp8_e4m3
    ):
        pytest.skip("Fp8_e4m3 not supported on wormhole")

    if formats.output_format == DataFormat.Bfp8_b:
        pytest.skip("Pack Untilize does not support Bfp8_b format")

    if (formats.input_format == DataFormat.Int32) ^ (
        formats.output_format == DataFormat.Int32
    ):
        pytest.skip("Pack Untilize does not support mixing Int32 with other formats")

    # Handling a hardware limitation: cannot convert 8-bit exponent datums to Float16 without storing them as intermediate Float32 in dest register.
    # For wormhole architecture, gasket cannot perform this conversion and packer takes input Float32 (from dest register) converting to Float16_A.
    # For blackhole architecture, gasket is able to convert Float32 to Float16_A before packing (reduces work on packer).`
    if (
        formats.input_format == DataFormat.Float16
        and data_formats.pack_src.is_32_bit()
        and dest_acc == DestAccumulation.No
    ):
        pytest.skip(
            "Due to hardware limitation, cannot convert 8-bit exponent datums to Float16 without storing them as intermediate Float32 in dest register. Therefore using dest_acc=No is not supported in this case."
        )

    # TODO: Checkout issue #1405 on tt-llk.
    if (
        get_chip_architecture() == ChipArchitecture.WORMHOLE
        and formats.input_format
        in (DataFormat.Float16_b, DataFormat.Float16, DataFormat.Bfp8_b)
        and formats.output_format == DataFormat.Float32
        and dest_acc == DestAccumulation.No
        and input_dimensions == [64, 512]
    ):
        pytest.skip("Wormhole pack_untilize does not support this format combination.")


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,  # Test Float32 with both 32bit mode dest (full precision) and 16bit mode dest (precision loss)
            DataFormat.Int32,
            DataFormat.Bfp8_b,
            DataFormat.Fp8_e4m3,
        ]  # Pack Untilize doesn't work for block float formats (Bfp8_b); we only include as input format in our test
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    input_dimensions=[[64, 64], [32, 128], [128, 128], [32, 64]],
    #  TODO add DestSync::Full tests when we have a solution for the static_assert in _llk_pack_untilize_init_ that requires block_ct_dim to be less or equal to 8,
    #  which is currently a limitation for testing DestSync::Full with the Untilize blocks calculation algorithm.
    dest_sync=[DestSync.Half],
    tile_dst_ct_offset=[0],  # Non-zero offsets are tracked in #1449
)
def test_pack_untilize(
    formats,
    dest_acc,
    input_dimensions,
    dest_sync,
    tile_dst_ct_offset,
):
    data_formats = infer_data_formats(
        formats.input_format,
        formats.output_format,
        dest_acc,
        False,
    )
    _skip_unsupported(formats, dest_acc, input_dimensions, data_formats)

    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )

    generate_golden = get_golden_generator(UntilizeGolden)

    golden_tensor = generate_golden(src_A, formats.output_format, input_dimensions)

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    # _llk_pack_untilize_init_ has a static_assert that checks if block_ct_dim is less or equal to 8.
    # TODO: Update this logic to accept more than 8 tiles per block if the static_assert changes in the future.
    _, block_ct_dim = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Untilize,
    )

    configuration = TestConfig(
        "sources/pack_untilize_test.cpp",
        formats,
        templates=[
            generate_input_dim(
                input_dimensions,
                input_dimensions,
                block_ct_dim,
            ),
            DEST_SYNC(dest_sync),
            TILE_DST_CT_OFFSET(tile_dst_ct_offset),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A), NUM_FACES(4), NARROW_TILE(NarrowTile.No)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            sfpu=False,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Int32,
        ]
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    # 32x16 tiles: 2, 4 and 8 tiles per tile-row; 8 with a 32-bit Dest splits the tile-row into two
    # blocks (block_ct_dim < full_ct_dim), so both the one-block and the per-row L1 stride paths run.
    input_dimensions=[[64, 32], [64, 64], [32, 128]],
)
def test_pack_untilize_narrow_tile(formats, dest_acc, input_dimensions):
    """Untilize of 32x16 (narrow) tiles: the two faces of a tile are stacked, so every packed row
    is 16 datums of one face and the second face follows below the first, not next to it.
    """
    tile_dimensions = [32, 16]
    data_formats = infer_data_formats(
        formats.input_format,
        formats.output_format,
        dest_acc,
        False,
    )
    _skip_unsupported(formats, dest_acc, input_dimensions, data_formats)

    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim

    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
        tile_dimensions=tile_dimensions,
    )

    generate_golden = get_golden_generator(UntilizeGolden)
    golden_tensor = generate_golden(
        src_A, formats.output_format, input_dimensions, tile_dimensions=tile_dimensions
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    _, block_ct_dim = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Untilize,
    )

    configuration = TestConfig(
        "sources/pack_untilize_test.cpp",
        formats,
        templates=[
            generate_input_dim(
                input_dimensions,
                input_dimensions,
                block_ct_dim,
                tile_dimensions=tuple(tile_dimensions),
            ),
            DEST_SYNC(DestSync.Half),
            TILE_DST_CT_OFFSET(0),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            NARROW_TILE(NarrowTile.Yes),
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
            face_r_dim=face_r_dim,
            tile_dimensions=tile_dimensions,
            sfpu=False,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
