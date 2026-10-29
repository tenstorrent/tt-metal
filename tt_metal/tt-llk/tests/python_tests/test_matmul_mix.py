# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.device import BootMode
from helpers.format_config import DataFormat, FormatConfig, InputOutputFormat, is_dest_acc_needed
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    Transpose,
    format_dict,
)
from helpers.matmul_sweep import (
    generate_matmul_dimension_combinations,
    generate_tile_dims,
    matmul_pack_form,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATMUL_MIXED_STRIDE,
    MATMUL_PHASE_ELIDE,
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATMUL_PACK_BLOCK,
    MATMUL_ROW_MOP,
    MATMUL_FORM_KEYS,
    MATMUL_K_LOOP,
    MATMUL_ROW_END_FOLD,
    MATMUL_STREAM_NARROW,
    MATMUL_REUSE_DIR,
    MATMUL_STREAM_Z,
    MATMUL_UNPACK_TTSYNC,
    MATMUL_UNPACK_TTSYNC_EXCL,
    NUM_FACES,
    PERF_RUN_TYPE,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test


def generate_format_aware_matmul_combinations(
    formats_list: List[FormatConfig],
    dest_acc_modes: List[DestAccumulation],
):
    """
    Generate matmul dimension combinations for multiple tiles.

    Rules:
    1. Format outliers (Float16_b->Float16, Bfp8_b->Float16) MUST use dest_acc=Yes
    2. Running matmul tests on DestSync.Half, max tile count is 8
    3. When dest_acc=Yes: max 4 tiles (32-bit dest register)
    4. When dest_acc=No: max 8 tiles (16-bit dest register)

    Returns: List of (format, dest_acc, dimensions) tuples
    """
    combinations = []

    for fmt in formats_list:
        base_max_tiles = 4 if is_dest_acc_needed(fmt) else 8

        for dest_acc in dest_acc_modes:
            max_tiles = 4 if dest_acc == DestAccumulation.Yes else base_max_tiles
            dimensions_list = generate_matmul_dimension_combinations(max_tiles)
            combinations.extend([(fmt, dest_acc, dims) for dims in dimensions_list])

    return combinations


DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
# Local mixed-format functional test (not committed): input_format feeds SrcA (matmul in1), input_format_B SrcB (matmul in0), buffer A is in0.
F = DataFormat
PAIRS = [(F.Float32, F.Float16_b), (F.Float16_b, F.Float32)]
MIX_FORMATS = [InputOutputFormat(a, o, input_format_B=b) for a, b in PAIRS for o in (F.Float16_b, F.Float32)]
MIX_COMBINATIONS = generate_format_aware_matmul_combinations(MIX_FORMATS, DEST_ACC_MODES)


@parametrize(
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi2],
    format_dest_acc_and_dims=MIX_COMBINATIONS,
    dest_sync=[DestSync.Half, DestSync.Full],
)
# Note: this test is used to test boot modes, that is why it has them piped as default arguments to the test itself
def test_matmul_mix(
    math_fidelity,
    format_dest_acc_and_dims,
    dest_sync,
    boot_mode=BootMode.DEFAULT,
    row_mop=False,
    transpose=False,
    unpack_ttsync=False,
    unpack_ttsync_excl=False,
):
    torch_format = format_dict[format_dest_acc_and_dims[0].output_format]

    formats = format_dest_acc_and_dims[0]
    dest_acc = format_dest_acc_and_dims[1]
    input_A_dimensions = format_dest_acc_and_dims[2][0]
    input_B_dimensions = format_dest_acc_and_dims[2][1]

    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format_B,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )

    # Calculate all matmul dimensions using helper function
    matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))

    # with transpose the unpacker transposes every in1 tile (faces and within faces); the golden multiplies by that
    golden_B = src_B
    if transpose:
        k_dim, n_dim = input_B_dimensions
        golden_B = (
            src_B.reshape(k_dim // 32, 32, n_dim // 32, 32)
            .permute(0, 3, 2, 1)
            .reshape(-1)
        )

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_A,
        golden_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
        tilize=True,
        input_A_format=formats.input_format_B,
        input_B_format=formats.input_format,
    )

    tilized_A = (
        src_A
        if formats.input_format_B.is_block_float()
        else tilize_block(src_A, dimensions=input_A_dimensions, stimuli_format=formats.input_format_B)
    )
    tilized_B = (
        src_B
        if formats.input_format.is_block_float()
        else tilize_block(src_B, dimensions=input_B_dimensions, stimuli_format=formats.input_format)
    )


    configuration = TestConfig(
        "sources/matmul_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATMUL_MIXED_STRIDE(),
            *MATMUL_PHASE_ELIDE.for_block(formats, math_fidelity, dest_acc, 0),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(),
            *([MATMUL_ROW_MOP()] if row_mop else []),
            *([MATMUL_UNPACK_TTSYNC()] if unpack_ttsync else []),
            *([MATMUL_UNPACK_TTSYNC_EXCL()] if unpack_ttsync_excl else []),
            *([] if unpack_ttsync else MATMUL_UNPACK_TTSYNC.for_block(formats, dest_acc, math_fidelity, matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim, dest_sync)),
            *MATMUL_STREAM_NARROW.for_block(
                formats, dest_acc, dest_sync, matmul_dims.ct_dim, matmul_dims.rt_dim,
                math_fidelity=math_fidelity,
            ),
            *MATMUL_REUSE_DIR.for_block(formats, math_fidelity, matmul_dims.ct_dim, matmul_dims.rt_dim, dest_sync),
            *MATMUL_STREAM_Z.for_block(
                formats, dest_acc, dest_sync, matmul_dims.ct_dim, matmul_dims.rt_dim,
                math_fidelity=math_fidelity,
                kt_dim=matmul_dims.kt_dim,
            ),
            *MATMUL_K_LOOP.for_block(
                formats, dest_acc, math_fidelity, matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim
            ),
            *MATMUL_ROW_END_FOLD.for_block(matmul_dims.ct_dim, matmul_dims.rt_dim),
            *MATMUL_FORM_KEYS.for_block(
                formats, matmul_dims.kt_dim, matmul_dims.ct_dim, matmul_dims.rt_dim,
                math_fidelity=math_fidelity, dest_sync=dest_sync, dest_acc=dest_acc, unfuse_one_tile=True,
            ),
            MATMUL_PACK_BLOCK(
                matmul_pack_form(
                    formats,
                    dest_sync,
                    math_fidelity,
                    matmul_dims.rt_dim,
                    matmul_dims.ct_dim,
                    matmul_dims.kt_dim,
                )
            ),
        ],
        runtimes=[
            NUM_FACES(),
            TILE_COUNT(matmul_dims.output_tile_cnt),
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
            LOOP_FACTOR(1),
            UNPACK_TRANS_FACES(Transpose.Yes if transpose else Transpose.No),
        ],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            formats.input_format_B,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=matmul_dims.output_tile_cnt,
        ),
        dest_acc=dest_acc,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


