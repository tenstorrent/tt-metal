# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# AI-generated — run_id: 2026-04-24_trigonometry_quasar_6e898d97

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test


# Trigonometry ops are SFPU transcendentals. Per llk-tester playbook §1A.8:
# always run with unpack_to_dest=True and pre-filter the matrix to bit-width
# matched (input_format.is_32_bit() == (dest_acc == Yes)).
def _is_invalid_quasar_combination(
    fmt: FormatConfig, dest_acc: DestAccumulation
) -> bool:
    in_fmt = fmt.input_format
    out_fmt = fmt.output_format

    # SFPU test uses unpack_to_dest=True: exclude bit-width mismatches.
    if in_fmt.is_32_bit() != (dest_acc == DestAccumulation.Yes):
        return True

    # Quasar packer does not support non-Float32 -> Float32 without dest_acc.
    if (
        in_fmt != DataFormat.Float32
        and out_fmt == DataFormat.Float32
        and dest_acc == DestAccumulation.No
    ):
        return True

    # Quasar SFPU Float32 -> Float16 needs dest_acc=Yes.
    if (
        in_fmt == DataFormat.Float32
        and out_fmt == DataFormat.Float16
        and dest_acc == DestAccumulation.No
    ):
        return True

    return False


TRIGONOMETRY_MATHOPS = [
    MathOperation.Sin,
    MathOperation.Cos,
    MathOperation.Acosh,
    MathOperation.Asinh,
    MathOperation.Atanh,
]


def generate_trigonometry_combinations(formats_list: List[FormatConfig]):
    """
    Generate trigonometry test combinations.

    Returns: List of (format, dest_acc, dest_sync, implied_math_format,
    input_dimensions, mathop) tuples.
    """
    combinations = []

    dest_sync_modes = (DestSync.Half, DestSync.Full)
    for fmt in formats_list:
        in_fmt = fmt.input_format

        dest_acc_modes = (
            (DestAccumulation.Yes,) if in_fmt.is_32_bit() else (DestAccumulation.No,)
        )
        for dest_acc in dest_acc_modes:
            if _is_invalid_quasar_combination(fmt, dest_acc):
                continue

            for dest_sync in dest_sync_modes:
                for implied_math_format in [
                    ImpliedMathFormat.No,
                    ImpliedMathFormat.Yes,
                ]:
                    for input_dimensions in [[32, 32]]:
                        for mathop in TRIGONOMETRY_MATHOPS:
                            combinations.append(
                                (
                                    fmt,
                                    dest_acc,
                                    dest_sync,
                                    implied_math_format,
                                    input_dimensions,
                                    mathop,
                                )
                            )

    return combinations


def prepare_trig_inputs(
    src_A: torch.Tensor,
    mathop: MathOperation,
    input_format: DataFormat,
    output_format: DataFormat,
) -> torch.Tensor:
    """
    Prepare inputs per-op with safe value ranges that the Quasar kernel can handle:
      sin / cos — [-pi, pi] (argument reduction is valid up to |y|<2^22 but we keep
                   the domain small so Maclaurin precision suffices).
      asinh    — [-10, 10] (log polynomial stable away from overflow).
      acosh    — [1.1, 50] (x >= 1 domain; avoid near-1 where the 3rd-order log loses
                   precision per analysis §6e.8).
      atanh    — [-0.9, 0.9] (|x| < 1 domain with margin so RECIP doesn't blow up).
    """
    torch_format = format_dict[input_format]
    src_A_f32 = src_A.to(torch.float32)

    # Normalize to [0, 1]
    mn = src_A_f32.min()
    mx = src_A_f32.max()
    if mx > mn:
        u = (src_A_f32 - mn) / (mx - mn)
    else:
        u = torch.zeros_like(src_A_f32)

    import math as _math

    if mathop == MathOperation.Sin or mathop == MathOperation.Cos:
        lo, hi = -_math.pi, _math.pi
        out = lo + u * (hi - lo)
    elif mathop == MathOperation.Asinh:
        lo, hi = -10.0, 10.0
        out = lo + u * (hi - lo)
    elif mathop == MathOperation.Acosh:
        lo, hi = 1.1, 50.0
        out = lo + u * (hi - lo)
    elif mathop == MathOperation.Atanh:
        lo, hi = -0.9, 0.9
        out = lo + u * (hi - lo)
    else:
        out = src_A_f32

    return out.to(torch_format)


SFPU_TRIGONOMETRY_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Float16_b,
    ]
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_implied_math_input_dims_mathop=generate_trigonometry_combinations(
        SFPU_TRIGONOMETRY_FORMATS
    ),
)
def test_sfpu_trigonometry_quasar(formats_dest_acc_sync_implied_math_input_dims_mathop):
    """
    Test trigonometry SFPU operations (sin, cos, acosh, asinh, atanh) on Quasar.

    Uses sfpu_trigonometry_quasar_test.cpp which dispatches on SfpuType at
    compile time. The five trig ops' dispatcher specializations are defined in
    that source.
    """
    (formats, dest_acc, dest_sync, implied_math_format, input_dimensions, mathop) = (
        formats_dest_acc_sync_implied_math_input_dims_mathop[0]
    )

    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=False,
    )

    src_A = prepare_trig_inputs(
        src_A, mathop, formats.input_format, formats.output_format
    )

    num_faces = 4

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    # SFPU tests: unpack_to_dest=True; matrix pre-filtered to matched bit widths.
    configuration = TestConfig(
        "sources/quasar/sfpu_trigonometry_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_dest=True,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Quasar SFPNONLINEAR modes are always approximate; accuracy is measurably
    # below Blackhole's accurate path. Relax tolerances for the inverse-hyperbolics
    # which cascade through an inlined 3rd-order log polynomial.
    custom_atol = None
    custom_rtol = None
    if mathop in (MathOperation.Acosh, MathOperation.Asinh, MathOperation.Atanh):
        custom_atol = 0.1
        custom_rtol = 0.1

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
    ), "Assert against golden failed"
