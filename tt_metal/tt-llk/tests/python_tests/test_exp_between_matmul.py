# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Accurate exp / exp2 / exp with base on a bf16 Dest between two K steps of one matmul.

The matmul replays its MVMUL sequence from the FPU half of the replay buffer, recorded once in
its init. Exp records its body into the replay buffer on every call; a record longer than the
SFPU half overwrites the start of the matmul sequence, and the second K step then computes
garbage. Exp writes a spare Dest tile, so the packed result is the plain matmul.

A throttled matmul (THROTTLE_LEVEL > 0, built with MM_THROTTLE as ttnn does) programs ADDR_MOD_6,
which selects the exp body's MM_THROTTLE dest-advance path.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    PerfRunType,
    Transpose,
    format_dict,
)
from helpers.matmul_sweep import generate_tile_dims
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    PERF_RUN_TYPE,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    TemplateParameter,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test


@dataclass
class EXP_VARIANT(TemplateParameter):
    """SfpuType run between the K steps, and its CLAMP_NEGATIVE."""

    op: str = "exponential"
    clamp_negative: bool = True

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr SfpuType EXP_OP = SfpuType::{self.op};\n"
            f"constexpr bool EXP_CLAMP_NEGATIVE = {str(self.clamp_negative).lower()};"
        )


@dataclass
class MM_THROTTLE_DEFINE(TemplateParameter):
    """MM_THROTTLE as ttnn defines it for a throttled matmul kernel."""

    level: int = 0

    def convert_to_cpp(self) -> str:
        return f"#define MM_THROTTLE {self.level}" if self.level else ""


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
    # (A, B) dimensions: K of two and four tiles, so exp runs between K steps.
    dims=[([32, 64], [64, 32]), ([64, 128], [128, 64])],
    variant=[
        ("exponential", True),
        ("exponential", False),
        ("exp2", True),
        ("exp_with_base", True),
    ],
    throttle=[0, 3, 5],
)
def test_exp_between_matmul(formats, math_fidelity, dims, variant, throttle):
    input_A_dimensions, input_B_dimensions = dims
    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=spec,
        spec_B=spec,
    )
    matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))

    golden_tensor = get_golden_generator(MatmulGolden)(
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )
    tilized_A = tilize_block(
        src_A, dimensions=input_A_dimensions, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_B_dimensions, stimuli_format=formats.input_format
    )

    configuration = TestConfig(
        "sources/exp_between_matmul_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
            DEST_SYNC(),
            THROTTLE_LEVEL(throttle),
            MM_THROTTLE_DEFINE(throttle),
            EXP_VARIANT(*variant),
        ],
        runtimes=[
            NUM_FACES(),
            TILE_COUNT(matmul_dims.output_tile_cnt),
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
            LOOP_FACTOR(1),
            UNPACK_TRANS_FACES(Transpose.No),
        ],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=matmul_dims.output_tile_cnt,
        ),
        dest_acc=DestAccumulation.No,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
