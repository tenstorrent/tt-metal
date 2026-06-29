# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Parallel FPU matmul + isolated SrcS SFPU add on Quasar.

TRISC0-2: matmul A×B -> buffer_C (Dest, PACK0).
TRISC3: UNP_S x2 -> SrcS -> add -> PACK1 -> buffer_Res.
Both outputs verified after a single configuration.run().
"""

import pytest
import torch
from helpers.data_format_inference import data_formats
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BinarySFPUGolden,
    MatmulGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
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
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import (
    StimuliSpec,
    apply_log_uniform_magnitudes,
    compute_safe_input_magnitude_range,
    format_elem_max,
    generate_stimuli,
)
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
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# (ADD_INPUT_DIMENSIONS, MATMUL_A_DIMENSIONS, MATMUL_B_DIMENSIONS)
DIMENSION_PROFILES = (
    ([32, 32], [32, 32], [32, 32]),
    ([32, 256], [64, 64], [64, 128]),
)

# Caps each add operand at 45% of format max so |a|+|b| stays <= 90% with rounding headroom.
ADD_RANGE_SAFETY_FACTOR = 0.45

SFPU_ADD_FORMATS = input_output_formats(
    [
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
    ]
)


def _matmul_output_fits_dest(
    MATMUL_A_DIMENSIONS: list[int],
    MATMUL_B_DIMENSIONS: list[int],
    dest_acc: DestAccumulation,
    dest_sync: DestSync,
) -> bool:
    matmul_dims = generate_tile_dims((MATMUL_A_DIMENSIONS, MATMUL_B_DIMENSIONS))
    capacity_divisor = 2 if dest_acc == DestAccumulation.Yes else 1
    max_tiles = DEST_SYNC_TILE_LIMITS[dest_sync] // capacity_divisor
    return matmul_dims.output_tile_cnt <= max_tiles


def generate_parallel_matmul_add_combinations(formats_list):
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
                    ADD_INPUT_DIMENSIONS,
                    MATMUL_A_DIMENSIONS,
                    MATMUL_B_DIMENSIONS,
                ) in DIMENSION_PROFILES:
                    if not _matmul_output_fits_dest(
                        MATMUL_A_DIMENSIONS,
                        MATMUL_B_DIMENSIONS,
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
                            ADD_INPUT_DIMENSIONS,
                            MATMUL_A_DIMENSIONS,
                            MATMUL_B_DIMENSIONS,
                        )
                    )
    return combinations


PARALLEL_MATMUL_ADD_COMBINATIONS = generate_parallel_matmul_add_combinations(
    SFPU_ADD_FORMATS
)


@pytest.mark.quasar
@parametrize(
    format_dest_acc_sync_implied_math=PARALLEL_MATMUL_ADD_COMBINATIONS,
)
def test_sfpu_add_parallel_matmul_quasar(format_dest_acc_sync_implied_math):
    (
        formats,
        dest_acc,
        dest_sync,
        implied_math_format,
        ADD_INPUT_DIMENSIONS,
        MATMUL_A_DIMENSIONS,
        MATMUL_B_DIMENSIONS,
    ) = format_dest_acc_sync_implied_math[0]

    matmul_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=MATMUL_A_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=MATMUL_B_DIMENSIONS,
        spec_A=matmul_spec,
        spec_B=matmul_spec,
        output_format=formats.output_format,
    )

    src_add_in0, tile_cnt_add, src_add_in1, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=ADD_INPUT_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=ADD_INPUT_DIMENSIONS,
        output_format=formats.output_format,
    )

    min_magnitude, max_magnitude = compute_safe_input_magnitude_range(
        formats.input_format,
        formats.output_format,
        input_magnitude_cap=format_elem_max(formats.input_format)
        * ADD_RANGE_SAFETY_FACTOR,
        output_magnitude_cap=format_elem_max(formats.output_format)
        * ADD_RANGE_SAFETY_FACTOR,
    )
    src_add_in0 = apply_log_uniform_magnitudes(
        src_add_in0,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        cast_to_format=formats.input_format,
        alternate_sign_every_n=3,
    )
    src_add_in1 = apply_log_uniform_magnitudes(
        src_add_in1,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        cast_to_format=formats.input_format,
        alternate_sign_every_n=3,
    )

    tilized_A = tilize_block(
        src_A, dimensions=MATMUL_A_DIMENSIONS, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=MATMUL_B_DIMENSIONS, stimuli_format=formats.input_format
    )

    matmul_dims = generate_tile_dims((MATMUL_A_DIMENSIONS, MATMUL_B_DIMENSIONS))

    torch_format = format_dict[formats.output_format]

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

    src_A_golden = src_A
    src_B_golden = src_B
    if formats.input_format.is_mx_format():
        tilized_A_golden = quantize_mx_tensor_chunked(
            tilized_A.flatten().to(torch.bfloat16), formats.input_format
        ).reshape(tilized_A.shape)
        tilized_B_golden = quantize_mx_tensor_chunked(
            tilized_B.flatten().to(torch.bfloat16), formats.input_format
        ).reshape(tilized_B.shape)
        src_A_golden = untilize_block(
            tilized_A_golden,
            stimuli_format=formats.input_format,
            dimensions=MATMUL_A_DIMENSIONS,
        )
        src_B_golden = untilize_block(
            tilized_B_golden,
            stimuli_format=formats.input_format,
            dimensions=MATMUL_B_DIMENSIONS,
        )

    generate_matmul_golden = get_golden_generator(MatmulGolden)
    golden_matmul = generate_matmul_golden(
        src_A_golden,
        src_B_golden,
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
    if formats.output_format.is_mx_format():
        golden_matmul = quantize_mx_tensor_chunked(
            golden_matmul.to(format_dict[pack_src_format]), formats.output_format
        ).to(torch_format)

    generate_add_golden = get_golden_generator(BinarySFPUGolden)
    golden_add = generate_add_golden(
        MathOperation.SfpuElwadd,
        torch.cat([src_add_in0, src_add_in1]),
        0,
        tile_cnt_add,
        0,
        tile_cnt_add * 32,
        [ADD_INPUT_DIMENSIONS[0] * 2, ADD_INPUT_DIMENSIONS[1]],
        formats.output_format,
        skip_tilize=True,
        input_format=formats.input_format,
    )[: src_add_in0.numel()]

    num_faces = 4

    stimuli = StimuliConfig(
        tilized_A.flatten(),
        formats.input_format,
        tilized_B.flatten(),
        formats.input_format,
        formats.output_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_B,
        buffer_S=src_add_in0,
        stimuli_S_format=formats.input_format,
        tile_count_S=tile_cnt_add,
        buffer_T=src_add_in1,
        stimuli_T_format=formats.input_format,
        tile_count_T=tile_cnt_add,
        buffer_C=torch.zeros(matmul_dims.output_tile_cnt * 1024, dtype=torch_format),
        stimuli_C_format=formats.output_format,
        tile_count_C=matmul_dims.output_tile_cnt,
        tile_count_res=tile_cnt_add,
        num_faces=num_faces,
        srcs_layout_operands=frozenset({"S", "T", "Res"}),
    )

    configuration = TestConfig(
        "sources/quasar/sfpu_add_parallel_matmul_quasar_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(MathFidelity.LoFi),
            IMPLIED_MATH_FORMAT(implied_math_format),
            ENABLE_2X_FORMAT(False),
            ENABLE_DIRECT_INDEXING(False),
            DEST_SYNC(dest_sync),
            UNPACK_TRANS_FACES(Transpose.No),
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
            NUM_FACES(num_faces, num_faces, num_faces),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_add),
        ],
        variant_stimuli=stimuli,
        unpack_to_srcs=True,
        dest_acc=dest_acc,
        disable_format_inference=formats.input_format.is_mx_format(),
    )

    outcome = configuration.run()

    res_add = torch.tensor(outcome.result, dtype=torch_format)
    res_matmul = torch.tensor(stimuli.collect_buffer_c_results(), dtype=torch_format)

    assert len(res_add) == len(golden_add), "add"
    assert len(res_matmul) == len(golden_matmul), "matmul"
    assert passed_test(golden_add, res_add, formats.output_format), "add"
    assert passed_test(golden_matmul, res_matmul, formats.output_format), "matmul"
