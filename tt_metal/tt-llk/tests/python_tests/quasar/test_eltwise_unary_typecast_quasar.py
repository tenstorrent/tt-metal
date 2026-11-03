# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    MathOperation,
    PerfRunType,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import parametrize, resolve_quasar_sfpu_variant, runtime
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import MAX_NUM_FACES
from helpers.utils import passed_test, result_tensor
from quasar.test_eltwise_unary_sfpu_quasar import (
    DEST_SYNC_MODES,
    TENSOR_DIMS,
    _dest_acc_for_typecast_formats,
    _prepare_typecast_input,
    run_eltwise_unary_sfpu_quasar,
    typecast_formats,
)


def _has_int32_endpoint(formats: InputOutputFormat) -> bool:
    return DataFormat.Int32 in (formats.input_format, formats.output_format)


def _sign_magnitude_typecast_formats():
    """Pairs the shared runner drives with its sign-magnitude integer staging. Int32 is
    two's complement on Quasar, so its pairs run in test_eltwise_unary_typecast_int32_quasar.
    """
    return [f for f in typecast_formats() if not _has_int32_endpoint(f)]


@pytest.mark.quasar
@parametrize(
    mathop=[MathOperation.Typecast],
    formats=_sign_magnitude_typecast_formats(),
    dest_acc=_dest_acc_for_typecast_formats,
    approx_mode=[ApproximationMode.No],
    dest_sync=list(DEST_SYNC_MODES),
    input_dimensions=runtime(list(TENSOR_DIMS)),
)
def test_eltwise_unary_typecast_quasar(
    mathop,
    formats,
    dest_acc,
    approx_mode,
    dest_sync,
    input_dimensions,
):
    """Typecast family: same unary SFPU kernel, paired with BH typecast module."""
    run_eltwise_unary_sfpu_quasar(
        formats,
        dest_acc,
        mathop,
        approx_mode,
        input_dimensions,
        dest_sync,
    )


# Int32 is two's complement in L1 and in Dest (unpack-to-dest copies the bits; tenstorrent/tt-metal#59016),
# so these pairs stage and read back Int32 as two's complement, with negative values. The Int16 pairs keep
# their stimuli non-negative (the harness has one twos_complement switch for every integer operand and Quasar
# Int16 stays sign-magnitude in L1), and so do the UInt16 pairs (a negative golden is unrepresentable there).
_INT32_TYPECAST_PAIRS = [
    InputOutputFormat(DataFormat.Int32, DataFormat.Float32),
    InputOutputFormat(DataFormat.Float32, DataFormat.Int32),
    InputOutputFormat(DataFormat.Int32, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Int32),
    InputOutputFormat(DataFormat.Int32, DataFormat.UInt16),
    InputOutputFormat(DataFormat.UInt16, DataFormat.Int32),
    InputOutputFormat(DataFormat.Int32, DataFormat.Int16),
    InputOutputFormat(DataFormat.Int16, DataFormat.Int32),
]

# Planted first: -1 is 0xFFFFFFFF in two's complement and -(2**31 - 1) read as sign-magnitude.
_INT32_SIGN_SEEDS = (-1, 1, -2, 0)


def _prepare_int32_typecast_input(src_A, src_B, formats: InputOutputFormat):
    values = _prepare_typecast_input(
        src_A, src_B, formats.input_format, formats.output_format
    )
    endpoints = (formats.input_format, formats.output_format)
    if DataFormat.Int16 in endpoints or DataFormat.UInt16 in endpoints:
        return values.abs()
    flat = values.flatten()
    for i, seed in enumerate(_INT32_SIGN_SEEDS):
        flat[i] = seed
    return flat.reshape(values.shape)


@pytest.mark.quasar
@parametrize(
    formats=_INT32_TYPECAST_PAIRS,
    dest_sync=list(DEST_SYNC_MODES),
    input_dimensions=runtime(list(TENSOR_DIMS)),
)
def test_eltwise_unary_typecast_int32_quasar(formats, dest_sync, input_dimensions):
    """Int32 typecast pairs with two's-complement Int32 staging and negative values."""
    dest_acc = DestAccumulation.Yes  # a 32-bit endpoint needs a 32-bit Dest
    variant = resolve_quasar_sfpu_variant(MathOperation.Typecast, formats, dest_acc)
    assert variant is not None, f"no Quasar SFPU variant for Typecast {formats}"
    formats = variant.formats

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    src_A = _prepare_int32_typecast_input(src_A, src_B, formats)
    # Integer-valued stimuli, so the torch cast is the exact conversion (RNE narrows to bf16).
    golden_tensor = src_A.to(torch.float32).to(format_dict[formats.output_format])

    num_faces = MAX_NUM_FACES
    test_config_kwargs = {
        "test_name": "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp",
        "formats": formats,
        "templates": [
            MATH_OP(mathop=MathOperation.Typecast),
            APPROX_MODE(ApproximationMode.No),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest
                if variant.unpack_to_dest
                else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync),
            TYPECAST_FORMATS(
                input_format=variant.sfpu_src, output_format=variant.sfpu_dst
            ),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            LOOP_FACTOR(1),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
            twos_complement=True,
        ),
        "unpack_to_dest": variant.unpack_to_dest,
        "dest_acc": dest_acc,
    }

    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=(PerfRunType.L1_TO_L1,),
        test_config_kwargs=test_config_kwargs,
    )
    variant.apply_formats(configuration.formats_config)

    res_from_L1 = configuration.run().result
    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
