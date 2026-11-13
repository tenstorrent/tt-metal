# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 matmul through the Compute API, against the same golden as test_matmul.py.

Shapes, formats and fidelity are compile time in LLK 2.0, so the tile dims reach the kernel as constexprs
(LLK2_MATMUL_DIMS) and the formats with compile_time_formats=True.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.matmul_sweep import (
    generate_matmul_dimension_combinations,
    generate_tile_dims,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    TemplateParameter,
)
from helpers.utils import passed_test
from llk2_common import DEST_HALF_TILES, tilize_operands
from llk2_paths import cpp_source, include_dirs


@dataclass
class LLK2_MATMUL_DIMS(TemplateParameter):
    rt_dim: int
    ct_dim: int
    kt_dim: int

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t RT_DIM = {self.rt_dim};\n"
            f"constexpr std::uint32_t CT_DIM = {self.ct_dim};\n"
            f"constexpr std::uint32_t KT_DIM = {self.kt_dim};"
        )


def _combinations():
    combos = []
    for fmt in input_output_formats(
        [DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Float32], same=True
    ):
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes):
            if fmt.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
                continue
            for dims in generate_matmul_dimension_combinations(DEST_HALF_TILES[dest_acc]):
                combos.append((fmt, dest_acc, dims))
    return combos


@parametrize(
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    format_dest_acc_and_dims=_combinations(),
)
def test_llk2_matmul(math_fidelity, format_dest_acc_and_dims):
    formats, dest_acc, (dims_A, dims_B) = format_dest_acc_and_dims
    torch_format = format_dict[formats.output_format]

    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims_A,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims_B,
        spec_A=spec,
        spec_B=spec,
    )
    matmul_dims = generate_tile_dims((dims_A, dims_B))

    golden = get_golden_generator(MatmulGolden)(
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=dims_A,
        input_B_dimensions=dims_B,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )

    tilized_A, tilized_B = tilize_operands(src_A, dims_A, src_B, dims_B, formats.input_format)

    configuration = TestConfig(
        cpp_source("llk2_matmul_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(),
            LLK2_MATMUL_DIMS(matmul_dims.rt_dim, matmul_dims.ct_dim, matmul_dims.kt_dim),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_A,
            formats.input_format,
            tilized_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=matmul_dims.output_tile_cnt,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(golden), "Result tensor and golden tensor are not of the same length"
    assert passed_test(golden, torch.tensor(res, dtype=torch_format), formats.output_format), "Assert against golden failed"
