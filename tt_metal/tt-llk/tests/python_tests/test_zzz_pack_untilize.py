# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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
    NUM_FACES,
    TILE_COUNT,
    TILE_DST_CT_OFFSET,
    generate_input_dim,
)
from helpers.utils import passed_test


def _pack_untilize_formats():
    base = [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Int32,
        DataFormat.Bfp8_b,
    ]
    if get_chip_architecture() != ChipArchitecture.WORMHOLE:
        base.append(DataFormat.Fp8_e4m3)
    fmts = input_output_formats(base)
    return [
        f
        for f in fmts
        if f.output_format != DataFormat.Bfp8_b
        and not (
            (f.input_format == DataFormat.Int32) ^ (f.output_format == DataFormat.Int32)
        )
    ]


def _pack_untilize_dest_acc(formats):
    modes = get_valid_dest_accumulation_modes(formats)
    filtered = []
    for da in modes:
        try:
            df = infer_data_formats(
                formats.input_format, formats.output_format, da, False
            )
        except ValueError:
            continue
        if (
            formats.input_format == DataFormat.Float16
            and df.pack_src.is_32_bit()
            and da == DestAccumulation.No
        ):
            continue
        filtered.append(da)
    return filtered


def _pack_untilize_input_dimensions(formats, dest_acc):
    dims = [[64, 64], [32, 128], [128, 128], [32, 64]]
    if TestConfig.WITH_COVERAGE:
        dims = [d for d in dims if d != [64, 512]]
    if (
        get_chip_architecture() == ChipArchitecture.WORMHOLE
        and formats.input_format
        in (DataFormat.Float16_b, DataFormat.Float16, DataFormat.Bfp8_b)
        and formats.output_format == DataFormat.Float32
        and dest_acc == DestAccumulation.No
    ):
        dims = [d for d in dims if d != [64, 512]]
    return dims


@parametrize(
    formats=_pack_untilize_formats(),
    dest_acc=lambda formats: _pack_untilize_dest_acc(formats),
    input_dimensions=lambda formats, dest_acc: _pack_untilize_input_dimensions(
        formats, dest_acc
    ),
    dest_sync=[DestSync.Half],
    tile_dst_ct_offset=[0],
)
def test_pack_untilize(
    formats,
    dest_acc,
    input_dimensions,
    dest_sync,
    tile_dst_ct_offset,
):

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
        runtimes=[TILE_COUNT(tile_cnt_A), NUM_FACES(4)],
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
