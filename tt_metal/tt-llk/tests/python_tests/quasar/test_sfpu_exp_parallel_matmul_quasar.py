# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Parallel FPU matmul + isolated SrcS SFPU exp on Quasar.

TRISC0-2: matmul A×B -> buffer_C (Dest, PACK0).
TRISC3: UNP_S -> SrcS -> exp -> PACK1 -> buffer_Res.
Both outputs verified after a single configuration.run().
"""

import pytest
import torch
from helpers.data_format_inference import data_formats
from helpers.format_config import FormatConfig
from helpers.golden_generators import (
    MatmulGolden,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    Transpose,
    format_dict,
)
from helpers.matmul_sweep import generate_tile_dims
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    generate_sfpu_format_dest_acc_combinations,
    parametrize,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    ENABLE_2X_FORMAT,
    ENABLE_DIRECT_INDEXING,
    IMPLIED_MATH_FORMAT,
    MATH_FIDELITY,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test
from test_eltwise_unary_sfpu_quasar import (
    SFPU_UNARY_FORMATS,
    prepare_inputs_for_operation,
)

# (exp_dims, input_A_dims, input_B_dims)
DIMENSION_PROFILES = (
    ([32, 32], [32, 32], [32, 32]),
    ([32, 256], [64, 64], [64, 128]),
)


def _matmul_output_fits_dest(
    input_A_dimensions: list[int],
    input_B_dimensions: list[int],
    dest_acc: DestAccumulation,
    dest_sync: DestSync,
) -> bool:
    matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))
    capacity_divisor = 2 if dest_acc == DestAccumulation.Yes else 1
    max_tiles = DEST_SYNC_TILE_LIMITS[dest_sync] // capacity_divisor
    return matmul_dims.output_tile_cnt <= max_tiles


def generate_parallel_matmul_exp_combinations(formats_list: list[FormatConfig]):
    combinations = []
    for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(formats_list):
        if not fmt.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes:
            continue
        for dest_sync in (DestSync.Half, DestSync.Full):
            for implied_math_format in (
                ImpliedMathFormat.No,
                ImpliedMathFormat.Yes,
            ):
                for (
                    exp_input_dimensions,
                    input_A_dimensions,
                    input_B_dimensions,
                ) in DIMENSION_PROFILES:
                    if not _matmul_output_fits_dest(
                        input_A_dimensions,
                        input_B_dimensions,
                        dest_acc,
                        dest_sync,
                    ):
                        continue
                    combinations.append(
                        (
                            fmt,
                            dest_acc,
                            dest_sync,
                            implied_math_format,
                            runtime(exp_input_dimensions),
                            runtime(input_A_dimensions),
                            runtime(input_B_dimensions),
                        )
                    )
    return combinations


PARALLEL_MATMUL_EXP_COMBINATIONS = generate_parallel_matmul_exp_combinations(
    SFPU_UNARY_FORMATS
)


@pytest.mark.quasar
@parametrize(
    format_dest_acc_sync_implied_math=PARALLEL_MATMUL_EXP_COMBINATIONS,
)
def test_sfpu_exp_parallel_matmul_quasar(format_dest_acc_sync_implied_math):
    (
        formats,
        dest_acc,
        dest_sync,
        implied_math_format,
        exp_input_dimensions,
        input_A_dimensions,
        input_B_dimensions,
    ) = format_dest_acc_sync_implied_math[0]

    matmul_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=matmul_spec,
        spec_B=matmul_spec,
        output_format=formats.output_format,
    )

    exp_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_exp, tile_cnt_exp, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=exp_input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=exp_input_dimensions,
        spec_A=exp_spec,
        spec_B=exp_spec,
        output_format=formats.output_format,
    )
    src_exp = prepare_inputs_for_operation(
        src_exp, MathOperation.Exp, formats.input_format, formats.output_format
    )

    tilized_A = tilize_block(
        src_A, dimensions=input_A_dimensions, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_B_dimensions, stimuli_format=formats.input_format
    )

    matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))

    formats_config = data_formats(
        input_format=formats.input_format,
        input_format_B=formats.input_format,
        output_format=formats.output_format,
        is_fp32_dest_acc_en=dest_acc,
        num_iterations=1,
        unpacking_to_dest=False,
        unpacking_to_srcs=True,
    )[0]
    pack_src_format = formats_config.pack_src

    generate_matmul_golden = get_golden_generator(MatmulGolden)
    golden_matmul = generate_matmul_golden(
        src_A,
        src_B,
        formats.output_format,
        MathFidelity.LoFi,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
        math_format=pack_src_format,
        dest_acc=dest_acc,
    )

    generate_exp_golden = get_golden_generator(UnarySFPUGolden)
    golden_exp = generate_exp_golden(
        MathOperation.Exp,
        src_exp,
        formats.output_format,
        dest_acc,
        formats.input_format,
        exp_input_dimensions,
    )

    num_faces = 4
    torch_format = format_dict[formats.output_format]

    stimuli = StimuliConfig(
        tilized_A.flatten(),
        formats.input_format,
        tilized_B.flatten(),
        formats.input_format,
        formats.output_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_B,
        buffer_S=src_exp,
        stimuli_S_format=formats.input_format,
        tile_count_S=tile_cnt_exp,
        buffer_C=torch.zeros(matmul_dims.output_tile_cnt * 1024, dtype=torch_format),
        stimuli_C_format=formats.output_format,
        tile_count_C=matmul_dims.output_tile_cnt,
        tile_count_res=tile_cnt_exp,
        num_faces=num_faces,
        srcs_layout_operands=frozenset({"S", "Res"}),
    )

    configuration = TestConfig(
        "sources/quasar/sfpu_exp_parallel_matmul_quasar_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(MathFidelity.LoFi),
            IMPLIED_MATH_FORMAT(implied_math_format),
            ENABLE_2X_FORMAT(False),
            ENABLE_DIRECT_INDEXING(False),
            DEST_SYNC(dest_sync),
            UNPACK_TRANS_FACES(Transpose.No),
        ],
        runtimes=[
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
            NUM_FACES(num_faces, num_faces, num_faces),
            TILE_COUNT(tile_cnt_exp),
        ],
        variant_stimuli=stimuli,
        unpack_to_srcs=True,
        dest_acc=dest_acc,
    )

    outcome = configuration.run()

    res_exp = torch.tensor(outcome.result, dtype=torch_format)
    res_matmul = torch.tensor(stimuli.collect_buffer_c_results(), dtype=torch_format)

    assert len(res_exp) == len(golden_exp), "exp"
    assert len(res_matmul) == len(golden_matmul), "matmul"
    assert passed_test(golden_exp, res_exp, formats.output_format), "exp"
    assert passed_test(golden_matmul, res_matmul, formats.output_format), "matmul"
