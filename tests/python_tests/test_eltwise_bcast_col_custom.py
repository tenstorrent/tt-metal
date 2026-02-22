# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    MATH_FIDELITY,
    MATH_OP,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

logger = logging.getLogger(__name__)


@dataclass
class CT_DIM(TemplateParameter):
    ct_dim: int

    def covert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t CT_DIM = {self.ct_dim};"


@parametrize(
    cpp_source=[
        "sources/multiple_tiles_eltwise_custom_test.cpp",
    ],
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
        ]
    ),
    mathop=[MathOperation.Elwsub],
    dest_acc=[DestAccumulation.No],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    broadcast_type=[BroadcastType.Column],
    input_dimensions_A=[[32, w] for w in range(32, 257, 32)],
    input_dimensions_B=[[32, 32]],
)
def test_eltwise_bcast_col_custom(
    cpp_source,
    formats,
    mathop,
    dest_acc,
    math_fidelity,
    broadcast_type,
    input_dimensions_A,
    input_dimensions_B,
    workers_tensix_coordinates,
):
    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE
        and cpp_source == "sources/multiple_tiles_eltwise_custom_test.cpp"
    ):
        pytest.skip("Custom test not supported on Wormhole")

    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")

    ct_dim = input_dimensions_A[1] // 32
    logger.info(
        "Running ct_dim=%d  srcA=%s  srcB=%s",
        ct_dim,
        input_dimensions_A,
        input_dimensions_B,
    )

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions_A,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions_B,
    )

    src_A_tilized = tilize_block(
        src_A, input_dimensions_A, formats.input_format
    ).flatten()
    src_B_tilized = tilize_block(
        src_B, input_dimensions_B, formats.input_format
    ).flatten()

    broadcast_golden = get_golden_generator(BroadcastGolden)
    src_B_broadcasted_tilized = broadcast_golden(
        broadcast_type,
        src_B_tilized,
        formats.input_format,
        num_faces=4,
        tile_cnt=tile_cnt_B,
        face_r_dim=16,
    )

    src_B_golden = untilize_block(
        src_B_broadcasted_tilized, formats.input_format, input_dimensions_B
    ).flatten()

    src_B_2d = src_B_golden.view(input_dimensions_B[0], input_dimensions_B[1])
    src_B_expanded = src_B_2d.repeat(1, input_dimensions_A[1] // input_dimensions_B[1])
    src_B_golden_expanded = src_B_expanded.flatten()

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        mathop, src_A, src_B_golden_expanded, formats.output_format, math_fidelity
    )

    configuration = TestConfig(
        cpp_source,
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            generate_input_dim(input_dimensions_A, input_dimensions_A),
            MATH_OP(mathop=mathop),
            BROADCAST_TYPE(broadcast_type),
            CT_DIM(ct_dim),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A)],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B_tilized,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
    )
    res_from_L1 = configuration.run(workers_tensix_coordinates)

    res_from_L1 = untilize_block(
        res_from_L1, formats.output_format, input_dimensions_A
    ).flatten()

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Spot-check: log a few elements per tile to verify srcA - bcast(srcB)
    for t in range(ct_dim):
        tile_start = t * 32 * 32
        a_sample = src_A[tile_start : tile_start + 4].tolist()
        b_sample = src_B_golden_expanded[tile_start : tile_start + 4].tolist()
        g_sample = golden_tensor[tile_start : tile_start + 4].tolist()
        r_sample = res_tensor[tile_start : tile_start + 4].tolist()
        logger.info(
            "  tile %d: srcA[:4]=%s  bcast_srcB[:4]=%s  golden[:4]=%s  result[:4]=%s",
            t,
            a_sample,
            b_sample,
            g_sample,
            r_sample,
        )

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
