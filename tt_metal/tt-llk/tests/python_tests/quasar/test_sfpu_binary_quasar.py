# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import BinarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import InputOutputFormat
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    SFPU_TILE_INDICES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test


@pytest.mark.quasar
@pytest.mark.parametrize(
    "data_format, dest_acc",
    [
        (DataFormat.Int32, DestAccumulation.Yes),
        (DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float16_b, DestAccumulation.No),
    ],
)
@pytest.mark.parametrize(
    "src0_idx, src1_idx, dst_idx",
    [
        (0, 1, 0),
        (1, 0, 0),
        (0, 1, 1),
        (0, 2, 1),
        (2, 3, 0),
    ],
)
def test_sfpu_binary_add_quasar(data_format, dest_acc, src0_idx, src1_idx, dst_idx):
    """
    Test binary SFPU ADD on Quasar architecture.

    Loads tiles directly into DEST via unpack-to-dest,
    performs elementwise add via SFPU using programmable
    tile indices, and verifies the result against a golden reference.
    """
    num_tiles_needed = max(src0_idx, src1_idx, dst_idx) + 1
    formats = InputOutputFormat(input_format=data_format, output_format=data_format)

    input_dimensions = [num_tiles_needed * 32, 32]

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=data_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=data_format,
        input_dimensions_B=input_dimensions,
        sfpu=False,
        full_range=True,
    )

    num_faces = 4
    mathop = MathOperation.SfpuElwadd

    elements_per_tile = 1024  # 4 faces * 16 rows * 16 cols
    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_full = generate_golden(
        mathop,
        src_A,
        src0_idx,
        src1_idx,
        dst_idx,
        32,  # num_iterations: 32 rows = 1 full tile
        input_dimensions,
        data_format,
    ).flatten()
    dst_start = dst_idx * elements_per_tile
    golden_tensor = golden_full[dst_start : dst_start + elements_per_tile]

    tile_count_res = 1  # we only pack the single output tile

    configuration = TestConfig(
        "sources/quasar/sfpu_binary_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            SFPU_TILE_INDICES(src0_idx, src1_idx, dst_idx),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            data_format,
            src_B,
            data_format,
            data_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_count_res,
            num_faces=num_faces,
        ),
        unpack_to_dest=True,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[data_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, data_format
    ), "Assert against golden failed"
