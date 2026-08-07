# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# 4row_arch INT8_2x matmul test. L1 holds plain dense INT8; the unpacker packs two int8
# (sign+magnitude) per SrcA/SrcB datum into the Int8_2x register format, and the 4row_arch
# matmul (16-MVMULDI DI+X2 traversal) does the 32-way dot-product reduction into INT32.
# Mirrors test_matmul_quasar.py but is scoped to the INT8 -> INT8_2x register path.

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.data_format_inference import data_formats
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIM,
    MatmulGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    Transpose,
    format_dict,
)
from helpers.matmul_sweep import generate_tile_dims
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import BootMode, InputOutputFormat, TestConfig
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

kt_dims = [1, 2, 4]


def matmul_dimensions_dest_sync(dest_acc_modes):
    return [
        (
            [mt_dim * TILE_DIM, kt_dim * TILE_DIM],
            [kt_dim * TILE_DIM, nt_dim * TILE_DIM],
            dest_acc,
            dest_sync,
        )
        for dest_sync in (DestSync.Half, DestSync.Full)
        for dest_acc in dest_acc_modes
        for max_tiles in (
            DEST_SYNC_TILE_LIMITS[dest_sync]
            // (2 if dest_acc == DestAccumulation.Yes else 1),
        )
        for mt_dim in range(1, max_tiles + 1)
        for nt_dim in range(1, max_tiles // mt_dim + 1)
        for kt_dim in kt_dims
    ]


# INT8_2x is an input-register packing of INT8: L1 holds plain INT8 (input), the unpacker
# produces Int8_2x in SrcA/SrcB, and the result accumulates in INT32.
MATMUL_FORMAT = [InputOutputFormat(DataFormat.Int8, DataFormat.Int32)]

_ARCH = get_chip_architecture()


@pytest.mark.quasar
@parametrize(
    format=MATMUL_FORMAT,
    # Integer matmul is LoFi-only.
    math_fidelity=[MathFidelity.LoFi],
    # INT8 matmul requires INT32 dest accumulation.
    dimensions_dest_acc_dest_sync=matmul_dimensions_dest_sync((DestAccumulation.Yes,)),
    implied_math_format=[ImpliedMathFormat.Yes],
    # Opt the INT8 input into the Int8_2x src-register packing (gated on QUASAR; the 4row_arch
    # target build runs on the Quasar-arch emulator).
    register_format_hint=lambda format: (
        [DataFormat.Int8_2x]
        if format.input_format == DataFormat.Int8 and _ARCH == ChipArchitecture.QUASAR
        else [None]
    ),
    enable_direct_indexing=lambda register_format_hint: (
        [False] if register_format_hint is None else [True, False]
    ),
    transpose=[Transpose.No],
)
def test_matmul(
    math_fidelity,
    dimensions_dest_acc_dest_sync,
    format,
    implied_math_format,
    register_format_hint,
    enable_direct_indexing,
    transpose,
):

    # Reassign format with register_format_hint so that test config generation and stimulus generation are aware of the register format hint.
    format = InputOutputFormat(
        format.input_format,
        format.output_format,
        input_format_B=format.input_format_B,
        register_format_hint=register_format_hint,
    )

    input_A_dimensions, input_B_dimensions, dest_acc, dest_sync_mode = (
        dimensions_dest_acc_dest_sync
    )

    torch_format = format_dict[format.output_format]

    if format.input_format == DataFormat.Int8:
        stimuli_spec = StimuliSpec.uniform(low=-127.0, high=127.0)
    else:
        stimuli_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=format.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=format.input_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=stimuli_spec,
        spec_B=stimuli_spec,
        output_format=format.output_format,
    )

    tilized_A = tilize_block(
        src_A, dimensions=input_A_dimensions, stimuli_format=format.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_B_dimensions, stimuli_format=format.input_format
    )

    src_A_golden = src_A
    src_B_golden = src_B

    if transpose == Transpose.Yes:
        t_matrix = get_golden_generator(TransposeGolden)

        src_B_golden = t_matrix.transpose_faces_multi_tile(
            src_B_golden,
            format.input_format,
            num_tiles=tile_cnt_B,
            tilize=True,
            input_dimensions=input_B_dimensions,
        )
        src_B_golden = t_matrix.transpose_within_faces_multi_tile(
            src_B_golden,
            format.input_format,
            num_tiles=tile_cnt_B,
            untilize=True,
            input_dimensions=input_B_dimensions,
        )

    # Calculate all matmul dimensions using helper function
    matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))

    formats_config = data_formats(
        input_format=format.input_format,
        input_format_B=format.input_format_B,
        output_format=format.output_format,
        is_fp32_dest_acc_en=dest_acc,
        num_iterations=1,
        unpacking_to_dest=False,
        # 2x register-format opt-in needs to flow through inference.
        disable_format_inference=False,
        register_format_hint=format.register_format_hint,
    )[0]
    pack_src_format = formats_config.pack_src

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_A_golden,
        src_B_golden,
        format.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        tilize=True,  # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
        input_A_format=format.input_format,
        input_B_format=format.input_format,
        math_format=pack_src_format,  # For accumulation of results in matmul we require to calculate in pack_src_format.
        dest_acc=dest_acc,
    )

    num_faces = 4

    configuration = TestConfig(
        "sources/quasar/matmul_4row_arch_test.cpp",
        format,
        templates=[
            MATH_FIDELITY(math_fidelity),
            IMPLIED_MATH_FORMAT(implied_math_format),
            ENABLE_2X_FORMAT(format.register_format_hint == DataFormat.Int8_2x),
            ENABLE_DIRECT_INDEXING(enable_direct_indexing),
            DEST_SYNC(dest_sync_mode),
            UNPACK_TRANS_FACES(transpose),
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
            TILE_COUNT(matmul_dims.output_tile_cnt),
            NUM_FACES(num_faces, num_faces, num_faces),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            format.input_format,
            tilized_B.flatten(),
            format.input_format,
            format.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=matmul_dims.output_tile_cnt,
            num_faces=num_faces,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
        boot_mode=BootMode.TRISC,
        disable_format_inference=False,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        format.output_format,
    ), "Assert against golden failed"
