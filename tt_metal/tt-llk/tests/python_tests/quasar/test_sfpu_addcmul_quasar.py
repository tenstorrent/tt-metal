# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import TernarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    VectorMode,
    format_dict,
)
from helpers.param_config import (
    generate_sfpu_format_dest_acc_combinations,
    input_output_formats,
    parametrize,
)
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
    SFPU_TERNARY_SCALAR,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    VECTOR_MODE,
)
from helpers.utils import passed_test

FACE_SIZE = 16 * 16

# Scalar multipliers, as raw fp32 bit patterns (the kernel takes `std::uint32_t value`
# and reinterprets it). 1.5f is deliberately not a power of two so a botched scalar
# decode cannot pass; -1.0f additionally covers a negative multiplier.
_SCALAR_BITS_1_5 = 0x3FC00000
_SCALAR_BITS_NEG_1 = 0xBF800000

# Map each VectorMode to the set of face indices the LLK dispatch processes.
# Faces outside the set keep whatever the producer left in Dest (the `a` tile
# in this test, since the output aliases it), so the per-mode assertion is
# restricted to processed faces.
_PROCESSED_FACES = {
    VectorMode.None_: (0,),
    VectorMode.R: (0, 1),
    VectorMode.C: (0, 2),
    VectorMode.RC: (0, 1, 2, 3),
}


def _processed_face_mask(vector_mode: VectorMode, num_faces: int) -> torch.Tensor:
    """1-D bool mask selecting the elements of a flat tile that ``vector_mode`` writes."""
    mask = torch.zeros(num_faces * FACE_SIZE, dtype=torch.bool)
    for face in _PROCESSED_FACES[vector_mode]:
        mask[face * FACE_SIZE : (face + 1) * FACE_SIZE] = True
    return mask


def _get_valid_formats_dest_acc():
    # addcmul is lanewise FP32 math (SFPMUL + SFPMAD) with an fp32 scalar, so the
    # kernel's static_assert allows only Float32 / Float16_b / Float16 Dest formats.
    formats = input_output_formats(
        [
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Float32,
        ]
    )
    return [
        (fmt, dest_acc)
        for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(formats)
        if not (
            fmt.input_format == DataFormat.Float16 and dest_acc == DestAccumulation.Yes
        )
    ]


def _get_valid_implied_math_formats(fmt: FormatConfig):
    if fmt.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def _is_unpack_to_dest(fmt: FormatConfig, dest_acc: DestAccumulation) -> bool:
    """UNPACK→DEST is selected only for 32-bit inputs with dest_acc=Yes."""
    return fmt.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=_get_valid_formats_dest_acc(),
    implied_math_format=lambda formats_dest_acc: _get_valid_implied_math_formats(
        formats_dest_acc[0]
    ),
    vector_mode=[VectorMode.None_, VectorMode.R, VectorMode.C, VectorMode.RC],
)
def test_sfpu_addcmul_quasar(formats_dest_acc, implied_math_format, vector_mode):
    """
    Test ternary `addcmul(a, b, c, value) -> a + value * b * c` on Quasar.

    The C++ test source packs 3 input tiles (a, b, c) into `buffer_A`, gets them
    into DEST at tile indices 0, 1, 2 (FPU datacopy, or UNPACK→DEST for 32-bit
    inputs), then runs the SFPU `addcmul` kernel over the faces selected by
    `vector_mode`, writing output back to DEST tile 0. PACK writes DEST tile 0
    out to `buffer_Res`.

    `vector_mode` covers all four face-selection modes; unprocessed faces are
    excluded from the golden assertion since Dest retains the `a` tile there.
    Both `dest_acc` arms are swept: `dest_acc=No` is the only path that emits the
    kernel's SFP_STOCH_RND narrowing, and Float16 vs Float16_b inputs select its
    two different rounding modes (fp32→fp16a vs fp32→fp16b).
    """
    formats, dest_acc = formats_dest_acc
    input_dimensions = [32, 32]
    torch_format_in = format_dict[formats.input_format]

    # Default float stimuli are uniform in [0.1, 1.1], so a + 1.5*b*c stays inside
    # [0.11, 2.92] — no overflow/underflow in any of the three swept Dest formats.
    torch.manual_seed(42)
    src_a_raw, tile_cnt_single, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    torch.manual_seed(43)
    src_b_raw, _, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    torch.manual_seed(44)
    src_c_raw, _, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    operand_a = src_a_raw.to(torch_format_in)
    operand_b = src_b_raw.to(torch_format_in)
    operand_c = src_c_raw.to(torch_format_in)

    src_A = torch.cat([operand_a, operand_b, operand_c])
    tile_cnt_A = tile_cnt_single * 3
    num_faces = 4

    generate_golden = get_golden_generator(TernarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.SfpuAddcmul,
        operand_a,
        operand_b,
        operand_c,
        _SCALAR_BITS_1_5,
        formats.output_format,
    )
    torch_format_out = format_dict[formats.output_format]

    unpack_to_dest = _is_unpack_to_dest(formats, dest_acc)
    src_B_dummy = torch.zeros_like(operand_a)

    configuration = TestConfig(
        "sources/quasar/sfpu_addcmul_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.SfpuAddcmul),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
            VECTOR_MODE(vector_mode),
            SFPU_TERNARY_SCALAR(ternary_scalar_bits=_SCALAR_BITS_1_5),
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
            src_B_dummy,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_single,
            tile_count_res=1,
            num_faces=num_faces,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format_out)
    mask = _processed_face_mask(vector_mode, num_faces)
    assert passed_test(
        golden_tensor[mask], res_tensor[mask], formats.output_format
    ), "Assert against golden failed"


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=_get_valid_formats_dest_acc()[:3],
    implied_math_format=lambda formats_dest_acc: _get_valid_implied_math_formats(
        formats_dest_acc[0]
    ),
    vector_mode=[VectorMode.None_, VectorMode.R, VectorMode.C, VectorMode.RC],
    # Not wrapped in runtime(): the compile-producer collection must select the same
    # variant count the simulator executes, or run_test.sh classifies the run
    # `execution_count_exceeds_selection` (infra_error) even when every variant passes.
    dest_index=[0, 1],
)
def test_sfpu_addcmul_mcw_quasar(
    formats_dest_acc, implied_math_format, vector_mode, dest_index
):
    """
    Deterministic addcmul test — alternating 0/1 `a` pattern with known b and c
    scalars (2 and 11) and a negative multiplier (-1.0f), so every output is an
    exactly representable -22 / -21.

    Runs through the same C++ harness as `test_sfpu_addcmul_quasar`, including a
    nonzero Dest tile offset. If this fails but the stimulus-driven test passes,
    the problem is in stimulus generation rather than the kernel.
    """
    formats, dest_acc = formats_dest_acc
    torch_format_in = format_dict[formats.input_format]
    input_dimensions = [32, 32]
    height, width = input_dimensions

    pattern = torch.arange(height * width) % 2
    operand_a = pattern.view(height, width).to(torch_format_in).flatten()
    operand_b = (torch.ones(height, width, dtype=torch_format_in) * 2).flatten()
    operand_c = (torch.ones(height, width, dtype=torch_format_in) * 11).flatten()

    tile_cnt_single = 1
    src_A = torch.cat([operand_a, operand_b, operand_c])
    tile_cnt_A = tile_cnt_single * 3
    num_faces = 4

    generate_golden = get_golden_generator(TernarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.SfpuAddcmul,
        operand_a,
        operand_b,
        operand_c,
        _SCALAR_BITS_NEG_1,
        formats.output_format,
    )
    torch_format_out = format_dict[formats.output_format]

    unpack_to_dest = _is_unpack_to_dest(formats, dest_acc)
    src_B_dummy = torch.zeros_like(operand_a)

    configuration = TestConfig(
        "sources/quasar/sfpu_addcmul_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.SfpuAddcmul),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
            VECTOR_MODE(vector_mode),
            SFPU_TERNARY_SCALAR(ternary_scalar_bits=_SCALAR_BITS_NEG_1),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(dest_index),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B_dummy,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_single,
            tile_count_res=1,
            num_faces=num_faces,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format_out)
    mask = _processed_face_mask(vector_mode, num_faces)
    assert passed_test(
        golden_tensor[mask], res_tensor[mask], formats.output_format
    ), "Assert against golden failed"
