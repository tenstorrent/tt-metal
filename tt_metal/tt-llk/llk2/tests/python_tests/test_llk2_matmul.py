# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 matmul through the Compute API, against the same golden as test_matmul.py.

Shapes, formats and fidelity are compile time in LLK 2.0, so the tile dims reach the kernel as constexprs
(LLK2_MATMUL_DIMS) and the formats with compile_time_formats=True.

test_llk2_matmul_transpose_in1 covers matmul_modifiers{.transpose_in1 = true}: in1 is a [K, N] grid of tiles, each
used transposed (the tile grid is not). That is how SDPA's Q K^T reads K: the reader lays K's tiles out as a K^T
grid, but each tile still holds 32 keys x 32 head-dim datums, and the matmul transposes it on the way in.
"""

from dataclasses import dataclass

import torch
from helpers.bfp_format_utils import bfp4b_to_float16b, bfp8b_to_float16b
from helpers.format_config import DataFormat, InputOutputFormat
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
from llk2_common import (
    DEST_HALF_TILES,
    TILE_DIM,
    as_packed,
    lattice_only,
    tilize_operands,
    unwrap_case,
)
from llk2_paths import cpp_source, include_dirs


@dataclass
class LLK2_MATMUL_DIMS(TemplateParameter):
    rt_dim: int
    ct_dim: int
    kt_dim: int
    transpose_in1: bool = False

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t RT_DIM = {self.rt_dim};\n"
            f"constexpr std::uint32_t CT_DIM = {self.ct_dim};\n"
            f"constexpr std::uint32_t KT_DIM = {self.kt_dim};\n"
            f"constexpr bool TRANSPOSE_IN1 = {'true' if self.transpose_in1 else 'false'};"
        )


def _combinations():
    combos = []
    for fmt in input_output_formats(
        [DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Float32], same=True
    ):
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes):
            if (
                fmt.input_format == DataFormat.Float32
                and dest_acc == DestAccumulation.No
            ):
                continue
            for dims in generate_matmul_dimension_combinations(
                DEST_HALF_TILES[dest_acc]
            ):
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

    golden = as_packed(golden, formats.output_format)

    tilized_A, tilized_B = tilize_operands(
        src_A, dims_A, src_B, dims_B, formats.input_format
    )

    configuration = TestConfig(
        cpp_source("llk2_matmul_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(),
            LLK2_MATMUL_DIMS(
                matmul_dims.rt_dim, matmul_dims.ct_dim, matmul_dims.kt_dim
            ),
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
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"
    assert passed_test(
        golden,
        torch.tensor(res, dtype=torch_format),
        formats.output_format,
        **lattice_only(formats.output_format),
    ), "Assert against golden failed"


# L1's block-float formats as plain values: a tensor in tile order quantized per block, back in row-major order.
BLOCK_FLOAT_QUANTIZERS = {
    DataFormat.Bfp8_b: bfp8b_to_float16b,
    DataFormat.Bfp4_b: bfp4b_to_float16b,
}


def transpose_each_tile(src, dims):
    """src [rows, cols] row-major with every 32x32 tile transposed in place; the tile grid stays as it is."""
    rows, cols = dims
    grid = src.view(rows // TILE_DIM, TILE_DIM, cols // TILE_DIM, TILE_DIM)
    return grid.permute(0, 3, 2, 1).contiguous().view(-1)


# (in0 tiles [rt, kt], in1 tiles [kt, ct]). The first is SDPA's Q K^T for one Q tile row: head_dim 128 (4 tiles) and
# a K chunk of 8 key tiles, the most a 16-bit Dest section holds. The others cover each reuse direction (ct >= rt
# holds in0 in SrcB; rt > ct holds in1 in SrcA) and K > 1 accumulation.
TRANSPOSE_IN1_SHAPES = [
    ((1, 4), (4, 8)),
    ((1, 1), (1, 1)),
    ((2, 3), (3, 2)),
    ((4, 2), (2, 1)),
    ((2, 2), (2, 4)),
]


def _transpose_in1_combinations():
    combos = []
    fp16b = input_output_formats([DataFormat.Float16_b], same=True)[0]
    fp32 = input_output_formats([DataFormat.Float32], same=True)[0]
    for fidelity in (MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4):
        for shape in TRANSPOSE_IN1_SHAPES:
            combos.append((fp16b, DestAccumulation.No, fidelity, shape))
    # 32-bit Dest, and Float32 (TF32 in SrcA): the SDPA shape and one with SrcA held.
    for fmt in (fp16b, fp32):
        for shape in (TRANSPOSE_IN1_SHAPES[0], TRANSPOSE_IN1_SHAPES[3]):
            if shape[0][0] * shape[1][1] <= DEST_HALF_TILES[DestAccumulation.Yes]:
                combos.append((fmt, DestAccumulation.Yes, MathFidelity.HiFi2, shape))
    return combos


@parametrize(case=_transpose_in1_combinations())
def test_llk2_matmul_transpose_in1(case):
    run_matmul_transpose_in1(case)


# A 32-bit Dest with the formats SDPA has there: InputOutputFormat(in0, out, in1). Q K^T is
# Float16_b or block-float operands into Float32 scores; P V is Float32 probabilities times Float16_b or block-float
# values. A Float32 operand goes through its Src register as TF32.
FP32_DEST_FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
    InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Float32),
    InputOutputFormat(DataFormat.Bfp4_b, DataFormat.Float32),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b, DataFormat.Bfp8_b),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b, DataFormat.Bfp4_b),
]


@parametrize(
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    formats=FP32_DEST_FORMATS,
    shape=[((1, 4), (4, 4))],
    transpose=[False, True],
)
def test_llk2_matmul_fp32_dest(math_fidelity, formats, shape, transpose):
    run_matmul_transpose_in1(
        (formats, DestAccumulation.Yes, math_fidelity, shape), transpose
    )


# Block-float operands in a 16-bit Dest, as SDPA's Q, K and V may be: Bfp4_b and Bfp8_b in either operand, plain
# (P V) and transposed (Q K^T).
@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Bfp4_b, DataFormat.Float16_b),
        InputOutputFormat(
            DataFormat.Float16_b, DataFormat.Float16_b, DataFormat.Bfp4_b
        ),
        InputOutputFormat(
            DataFormat.Float16_b, DataFormat.Float16_b, DataFormat.Bfp8_b
        ),
        InputOutputFormat(DataFormat.Bfp4_b, DataFormat.Bfp4_b),
    ],
    transpose=[False, True],
)
def test_llk2_matmul_block_float_inputs(formats, transpose):
    case = (formats, DestAccumulation.No, MathFidelity.HiFi2, ((1, 4), (4, 4)))
    run_matmul_transpose_in1(case, transpose)


def run_matmul_transpose_in1(case, transpose_in1=True):
    """One matmul, with transpose_in1 or plain, against the golden. in1 may have its own format."""
    formats, dest_acc, math_fidelity, ((rt, kt), (kt_b, ct)) = unwrap_case(case)
    assert kt == kt_b
    torch_format = format_dict[formats.output_format]
    dims_A = [rt * TILE_DIM, kt * TILE_DIM]
    dims_B = [kt * TILE_DIM, ct * TILE_DIM]

    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims_A,
        stimuli_format_B=formats.input_format_B,
        input_dimensions_B=dims_B,
        spec_A=spec,
        spec_B=spec,
    )

    # The kernel gets in1 as stored; the golden multiplies by in1 with each tile transposed, which is what tt-metal's
    # matmul with transpose (SDPA's Q K^T) computes for the same L1 layout.
    golden_B, golden_B_format = src_B, formats.input_format_B
    if transpose_in1:
        # A block-float in1 shares an exponent per 16 datums of a face row as L1 holds it, before the transpose: the
        # golden takes it quantized that way (the stimulus is in tile order already), then transposed, as plain values.
        if golden_B_format in BLOCK_FLOAT_QUANTIZERS:
            golden_B = BLOCK_FLOAT_QUANTIZERS[golden_B_format](src_B.flatten(), dims_B)
            golden_B_format = DataFormat.Float16_b
        golden_B = transpose_each_tile(golden_B, dims_B)
    golden = get_golden_generator(MatmulGolden)(
        src_A,
        golden_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=dims_A,
        input_B_dimensions=dims_B,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=golden_B_format,
    )
    golden = as_packed(golden, formats.output_format)

    tilized_A, _ = tilize_operands(src_A, dims_A, src_A, dims_A, formats.input_format)
    tilized_B, _ = tilize_operands(src_B, dims_B, src_B, dims_B, formats.input_format_B)

    configuration = TestConfig(
        cpp_source("llk2_matmul_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(),
            LLK2_MATMUL_DIMS(rt, ct, kt, transpose_in1=transpose_in1),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_A,
            formats.input_format,
            tilized_B,
            formats.input_format_B,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=rt * ct,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"
    assert passed_test(
        golden,
        torch.tensor(res, dtype=torch_format),
        formats.output_format,
        **lattice_only(formats.output_format),
    ), "Assert against golden failed"
