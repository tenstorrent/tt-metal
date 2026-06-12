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
from helpers.format_config import DataFormat, FormatConfig
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
from helpers.param_config import parametrize
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
from test_sfpu_nonlinear_quasar import (
    SFPU_NONLINEAR_FORMATS,
    prepare_inputs_for_operation,
)

EXP_INPUT_DIMENSIONS = [32, 32]
MATMUL_A_DIMENSIONS = [32, 32]
MATMUL_B_DIMENSIONS = [32, 32]


def generate_parallel_matmul_exp_combinations(formats_list: list[FormatConfig]):
    combinations = []
    for fmt in formats_list:
        dest_acc_modes = (
            (DestAccumulation.Yes,)
            if fmt.input_format.is_32_bit()
            else (DestAccumulation.No, DestAccumulation.Yes)
        )
        for dest_acc in dest_acc_modes:
            if (
                fmt.input_format != DataFormat.Float32
                and fmt.output_format == DataFormat.Float32
                and dest_acc == DestAccumulation.No
            ):
                continue
            for dest_sync in (DestSync.Half, DestSync.Full):
                for implied_math_format in (
                    ImpliedMathFormat.No,
                    ImpliedMathFormat.Yes,
                ):
                    combinations.append((fmt, dest_acc, dest_sync, implied_math_format))
    return combinations


PARALLEL_MATMUL_EXP_COMBINATIONS = generate_parallel_matmul_exp_combinations(
    SFPU_NONLINEAR_FORMATS
)


@pytest.mark.quasar
@parametrize(
    format_dest_acc_sync_implied_math=PARALLEL_MATMUL_EXP_COMBINATIONS,
)
def test_sfpu_exp_parallel_matmul_quasar(format_dest_acc_sync_implied_math):
    (formats, dest_acc, dest_sync, implied_math_format) = (
        format_dest_acc_sync_implied_math[0]
    )

    torch.manual_seed(42)

    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=MATMUL_A_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=MATMUL_B_DIMENSIONS,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
        output_format=formats.output_format,
    )

    src_exp, tile_cnt_exp, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=EXP_INPUT_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=EXP_INPUT_DIMENSIONS,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )
    src_exp = prepare_inputs_for_operation(
        src_exp, MathOperation.Exp, formats.input_format, formats.output_format
    )

    tilized_A = tilize_block(
        src_A, dimensions=MATMUL_A_DIMENSIONS, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=MATMUL_B_DIMENSIONS, stimuli_format=formats.input_format
    )
    tilized_exp = tilize_block(
        src_exp, dimensions=EXP_INPUT_DIMENSIONS, stimuli_format=formats.input_format
    )

    matmul_dims = generate_tile_dims((MATMUL_A_DIMENSIONS, MATMUL_B_DIMENSIONS))

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
        input_A_dimensions=MATMUL_A_DIMENSIONS,
        input_B_dimensions=MATMUL_B_DIMENSIONS,
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
        EXP_INPUT_DIMENSIONS,
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
        buffer_S=tilized_exp.flatten(),
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
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
            TILE_COUNT(tile_cnt_exp),
            NUM_FACES(num_faces, num_faces, num_faces),
        ],
        runtimes=[],
        variant_stimuli=stimuli,
        unpack_to_srcs=True,
        dest_acc=dest_acc,
    )

    outcome = configuration.run()

    res_exp = torch.tensor(outcome.result, dtype=torch_format)
    res_matmul = torch.tensor(stimuli.collect_buffer_c_results(), dtype=torch_format)

    assert len(res_exp) == len(golden_exp)
    assert len(res_matmul) == len(golden_matmul)
    assert passed_test(golden_exp, res_exp, formats.output_format)
    assert passed_test(golden_matmul, res_matmul, formats.output_format)
