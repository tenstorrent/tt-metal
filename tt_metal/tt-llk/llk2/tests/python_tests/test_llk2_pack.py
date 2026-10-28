# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 pack modifiers through the Compute API: ReLU and L1 accumulation, behind a matmul.

The kernel runs the matmul and packs it PASSES times, one Dest section per pass: the first pack overwrites, later
ones add into L1 (l1_acc). Goldens: the harness's MatmulGolden, then PackGolden's ReLU and L1 accumulation.
"""

from dataclasses import dataclass

import torch
from helpers.data_format_inference import infer_data_formats
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, PackGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    PackerReluType,
    format_dict,
)
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
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test
from llk2_paths import cpp_source, include_dirs

ACCUMULATING_PASSES = 3  # > 2: the Dest sections wrap around


@dataclass
class LLK2_PACK_TEST(TemplateParameter):
    rt_dim: int
    ct_dim: int
    kt_dim: int
    relu: bool
    passes: int

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t RT_DIM = {self.rt_dim};\n"
            f"constexpr std::uint32_t CT_DIM = {self.ct_dim};\n"
            f"constexpr std::uint32_t KT_DIM = {self.kt_dim};\n"
            f"constexpr bool PACK_RELU = {'true' if self.relu else 'false'};\n"
            f"constexpr std::uint32_t PASSES = {self.passes};"
        )


def _combinations():
    combos = []
    for fmt in input_output_formats(
        [DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Float32], same=True
    ):
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes):
            if fmt.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
                continue
            for relu in (False, True):
                for l1_acc in (False, True):
                    # L1 adds Float32, Float16_b and Float16 only (Packers/Accumulation.md).
                    if l1_acc and fmt.output_format == DataFormat.Bfp8_b:
                        continue
                    for dims in generate_matmul_dimension_combinations(4, kt_dims=[1, 2]):
                        combos.append((fmt, dest_acc, relu, l1_acc, dims))
    return combos


@parametrize(case=_combinations())
def test_llk2_pack(case):
    (formats, dest_acc, relu, l1_acc, (dims_A, dims_B)),  = case  # parametrize wraps one-axis tuples
    torch_format = format_dict[formats.output_format]
    math_fidelity = MathFidelity.LoFi
    passes = ACCUMULATING_PASSES if l1_acc else 1

    # With ReLU, negative values so it has something to clamp. Otherwise [0, 1] as test_llk2_matmul.py: signed sums
    # near 0, added up PASSES times in BF16, can miss the relative tolerance by one rounding step.
    spec = StimuliSpec.uniform(low=-1.0 if relu else 0.0, high=1.0)
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
    if relu:
        pack_src = infer_data_formats(
            input_format=formats.input_format,
            output_format=formats.output_format,
            is_fp32_dest_acc_en=dest_acc,
        ).pack_src
        relu_config = PackGolden.generate_relu_config(
            PackerReluType.ZeroRelu, relu_threshold=0.0, intermediate_format=pack_src
        )
        golden = PackGolden.apply_relu(golden, relu_config, pack_src)
    if passes > 1:
        golden = PackGolden.accumulate_l1([golden] * passes, formats.output_format)

    if formats.input_format != DataFormat.Bfp8_b:
        tilized_A = tilize_block(src_A, dimensions=dims_A, stimuli_format=formats.input_format)
        tilized_B = tilize_block(src_B, dimensions=dims_B, stimuli_format=formats.input_format)
    else:
        tilized_A, tilized_B = src_A, src_B

    configuration = TestConfig(
        cpp_source("llk2_pack_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(),
            LLK2_PACK_TEST(matmul_dims.rt_dim, matmul_dims.ct_dim, matmul_dims.kt_dim, relu, passes),
        ],
        runtimes=[],
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
        dest_acc=dest_acc,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(golden), "Result tensor and golden tensor are not of the same length"
    res_tensor = torch.tensor(res, dtype=torch_format)
    if relu:
        assert (res_tensor >= 0).all(), "ReLU left negative values"
    assert passed_test(golden, res_tensor, formats.output_format), "Assert against golden failed"
