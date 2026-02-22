# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    PerfRunType,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    TILE_COUNT,
    TemplateParameter,
)


@dataclass
class CT_DIM(TemplateParameter):
    ct_dim: int

    def covert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t CT_DIM = {self.ct_dim};"


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    mathop=[MathOperation.Elwsub],
    dest_acc=[DestAccumulation.No],
    math_fidelity=[MathFidelity.LoFi],
    broadcast_type=[BroadcastType.Column],
    ct_dim=[1, 8],
    loop_factor=[16],
)
def test_perf_eltwise_bcast_col_custom(
    perf_report,
    formats,
    mathop,
    dest_acc,
    math_fidelity,
    broadcast_type,
    ct_dim,
    loop_factor,
    workers_tensix_coordinates,
):
    if TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE:
        pytest.skip("Custom blocked sub_bcast_cols not supported on Wormhole")

    configuration = PerfConfig(
        "sources/eltwise_bcast_col_custom_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop),
            BROADCAST_TYPE(broadcast_type),
            CT_DIM(ct_dim),
        ],
        runtimes=[
            TILE_COUNT(ct_dim),
            LOOP_FACTOR(loop_factor),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=ct_dim,
            tile_count_B=1,
            tile_count_res=ct_dim,
        ),
        dest_acc=dest_acc,
    )

    configuration.run(perf_report, location=workers_tensix_coordinates)
