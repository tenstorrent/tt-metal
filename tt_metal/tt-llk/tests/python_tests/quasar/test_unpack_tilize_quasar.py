# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import (
    TilizeGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import (
    DataCopyType,
    DestSync,
    Fp32DestMode,
    ImpliedMathFormat,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
)
from helpers.utils import passed_test


def generate_unpack_tilize_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate unpack_tilize combinations.

    Rules:
    1. 32-bit formats require Fp32DestMode.Yes

    Args: List of input-output format pairs

    Returns: List of (format, is_32b_dest_en, unpacker_sel, input_dimensions) tuples
    """
    dimensions_cache = {
        (is_32b_dest_en, dest_sync): tuple(
            generate_unary_input_dimensions(is_32b_dest_en, dest_sync)
        )
        for is_32b_dest_en in (Fp32DestMode.No, Fp32DestMode.Yes)
        for dest_sync in (DestSync.Half, DestSync.Full)
    }

    combinations = []

    for fmt in formats_list:
        in_fmt = fmt.input_format

        fp32_dest_modes = (
            (Fp32DestMode.Yes,)
            if in_fmt.is_32_bit()
            else (
                (Fp32DestMode.No,)
                if in_fmt in [DataFormat.Float16, DataFormat.Int16]
                else (Fp32DestMode.No, Fp32DestMode.Yes)
            )
        )
        # 32-bit tilize uses unpack_to_dest (UNP_DEST)
        unpacker_engines = (
            (UnpackerEngine.UnpDest,)
            if in_fmt.is_32_bit()
            else (UnpackerEngine.UnpA, UnpackerEngine.UnpB)
        )

        for is_32b_dest_en in fp32_dest_modes:
            for dest_sync in (DestSync.Half, DestSync.Full):
                for unpacker_sel in unpacker_engines:
                    for dimensions in dimensions_cache[(is_32b_dest_en, dest_sync)]:
                        combinations.append(
                            (fmt, is_32b_dest_en, dest_sync, unpacker_sel, dimensions)
                        )

    return combinations


UNPACK_TILIZE_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Int32,
        DataFormat.Int16,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ],
    same=True,  # Input format and output format are the same
)
ALL_UNPACK_TILIZE_COMBINATIONS = generate_unpack_tilize_combinations(
    UNPACK_TILIZE_FORMATS
)


@pytest.mark.quasar
@parametrize(
    formats_32b_dest_sync_unpack_sel_dimensions=ALL_UNPACK_TILIZE_COMBINATIONS,
)
def test_unpack_tilize_quasar(
    formats_32b_dest_sync_unpack_sel_dimensions, boot_mode=BootMode.DEFAULT
):
    (formats, is_32b_dest_en, dest_sync_mode, unpacker_sel, input_dimensions) = (
        formats_32b_dest_sync_unpack_sel_dimensions[0]
    )

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(TilizeGolden)
    golden_src = src_B if unpacker_sel == UnpackerEngine.UnpB else src_A
    if formats.input_format.is_mx_format():
        golden_src = quantize_mx_tensor_chunked(golden_src, formats.input_format)
    golden_tensor = generate_golden(
        golden_src, input_dimensions, formats.output_format, num_faces=4
    )

    configuration = TestConfig(
        "sources/quasar/unpack_tilize_quasar_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            UNPACKER_ENGINE_SEL(unpacker_sel),
            DATA_COPY_TYPE(
                DataCopyType.B2D
                if unpacker_sel == UnpackerEngine.UnpB
                else DataCopyType.A2D
            ),
            DEST_SYNC(dest_sync_mode),
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(),
            NUM_FACES(),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=4,
        ),
        unpack_to_dest=(
            formats.input_format.is_32_bit() and is_32b_dest_en == Fp32DestMode.Yes
        ),
        is_32b_dest_en=is_32b_dest_en,
        boot_mode=boot_mode,
        # MX formats require disable_format_inference to match C++ IMPLIED_MATH_FORMAT setting.
        disable_format_inference=(formats.input_format.is_mx_format()),
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
