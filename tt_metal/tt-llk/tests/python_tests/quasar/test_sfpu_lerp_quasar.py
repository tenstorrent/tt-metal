# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct

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
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
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
    VECTOR_MODE,
)
from helpers.utils import passed_test

FACE_SIZE = 16 * 16

# TernarySFPUGolden takes the addcmul/addcdiv scalar as a raw fp32 bit pattern; lerp
# ignores it, so pass a well-formed 1.0f rather than an undecodable value.
_UNUSED_SCALAR_BITS = struct.unpack("<I", struct.pack("<f", 1.0))[0]

# Map each VectorMode to the set of face indices the LLK dispatch processes.
# Faces outside the set keep whatever the producer left in Dest (the start tile
# in this test), so the per-mode assertion is restricted to processed faces.
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
    """Float32 / Float16_b only — calculate_lerp() static_asserts on that pair.

    Float16 (fp16a) is a legal Quasar Dest format but is rejected by the kernel: its
    16-bit-Dest arm narrows with float32_to_bf16_rne(), a bf16 rounding that would
    corrupt an E5M10 store. Integer and MX formats are out for a float-only op.
    """
    formats = input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
        ]
    )
    return generate_sfpu_format_dest_acc_combinations(formats)


def _get_valid_implied_math_formats(fmt: FormatConfig):
    if fmt.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def _is_unpack_to_dest(fmt: FormatConfig, dest_acc: DestAccumulation) -> bool:
    """UNPACK→DEST is selected only for 32-bit inputs with dest_acc=Yes."""
    return fmt.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes


def _weight_for_test_case(
    base: torch.Tensor, input_format: DataFormat, test_case: str
) -> torch.Tensor:
    """Apply the weight regime (mixed / weight_zero / weight_one) for a variant."""
    torch_format = format_dict[input_format]
    if test_case == "weight_zero":
        # out must reduce to the start tensor exactly.
        return torch.zeros_like(base, dtype=torch_format)
    if test_case == "weight_one":
        # out must reduce to the end tensor exactly.
        return torch.ones_like(base, dtype=torch_format)
    # "mixed" — interior weights in [0, 1], the actual interpolation regime.
    return base.to(torch_format)


def _run_lerp(
    formats,
    dest_acc,
    implied_math_format,
    vector_mode,
    start,
    end,
    weight,
    dest_index,
):
    """Stage start/end/weight into Dest tiles base+0/1/2, run lerp, assert on Dest tile base+0."""
    torch_format_out = format_dict[formats.output_format]

    src_A = torch.cat([start, end, weight])
    tile_cnt_A = 3
    num_faces = 4

    generate_golden = get_golden_generator(TernarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.SfpuLerp,
        start,
        end,
        weight,
        _UNUSED_SCALAR_BITS,
        formats.output_format,
    )
    golden_tensor = golden_tensor.to(torch_format_out)

    unpack_to_dest = _is_unpack_to_dest(formats, dest_acc)
    src_B_dummy = torch.zeros_like(start)

    configuration = TestConfig(
        "sources/quasar/sfpu_lerp_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.SfpuLerp),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
            VECTOR_MODE(vector_mode),
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
            tile_count_B=1,
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
    formats_dest_acc=_get_valid_formats_dest_acc(),
    implied_math_format=lambda formats_dest_acc: _get_valid_implied_math_formats(
        formats_dest_acc[0]
    ),
    # `test_case` is a stimulus-only axis and could carry runtime(), but leaving every
    # axis compile-time keeps the collected variant count equal to the executed one, which
    # the codegen verification schema requires.
    test_case=["mixed", "weight_zero", "weight_one"],
    vector_mode=[VectorMode.None_, VectorMode.R, VectorMode.C, VectorMode.RC],
)
def test_sfpu_lerp_quasar(
    formats_dest_acc, implied_math_format, test_case, vector_mode
):
    """
    Test ternary `lerp(start, end, weight) -> start + weight * (end - start)` on Quasar.

    The C++ test source packs 3 input tiles (start, end, weight) into `buffer_A`,
    stages them into DEST tile indices 0, 1, 2 (UNPACK→DEST for 32-bit inputs,
    FPU datacopy otherwise), runs the SFPU `lerp` kernel over the faces selected by
    `vector_mode` writing output to DEST tile 0, and PACK writes DEST tile 0 out to
    `buffer_Res`.

    `test_case` pins the weight regime: `weight_zero` and `weight_one` collapse the
    interpolation onto a single endpoint, so a bad operand order or a dropped term
    shows up as an exact-tensor mismatch rather than a small PCC drift; `mixed`
    exercises real interpolation. Unprocessed faces are excluded from the golden
    assertion since Dest retains the producer-written start tile there.
    """
    formats, dest_acc = formats_dest_acc
    input_dimensions = [32, 32]
    torch_format_in = format_dict[formats.input_format]

    # start/end in [-1, 1] and weight in [0, 1] keep `start + weight * (end - start)`
    # inside [-1, 1] — no overflow or catastrophic cancellation on either dest_acc arm.
    endpoint_spec = StimuliSpec.uniform(low=-1.0, high=1.0)
    weight_spec = StimuliSpec.uniform(low=0.0, high=1.0)

    torch.manual_seed(42)
    start_raw, _, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=endpoint_spec,
        spec_B=endpoint_spec,
    )
    torch.manual_seed(43)
    end_raw, _, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=endpoint_spec,
        spec_B=endpoint_spec,
    )
    torch.manual_seed(44)
    weight_raw, _, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=weight_spec,
        spec_B=weight_spec,
    )

    start = start_raw.to(torch_format_in)
    end = end_raw.to(torch_format_in)
    weight = _weight_for_test_case(weight_raw, formats.input_format, test_case)

    _run_lerp(
        formats,
        dest_acc,
        implied_math_format,
        vector_mode,
        start,
        end,
        weight,
        dest_index=0,
    )


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=_get_valid_formats_dest_acc(),
    implied_math_format=lambda formats_dest_acc: _get_valid_implied_math_formats(
        formats_dest_acc[0]
    ),
    vector_mode=[VectorMode.RC],
    dest_index=[0, 1],
)
def test_sfpu_lerp_mcw_quasar(
    formats_dest_acc, implied_math_format, vector_mode, dest_index
):
    """
    Deterministic lerp test — start=2, end=10 and a repeating weight ramp
    (0, 0.25, 0.5, 0.75) so every element has a hand-checkable answer
    (2, 4, 6, 8).

    Runs through the same C++ harness as `test_sfpu_lerp_quasar`, including a
    nonzero Dest tile offset. If this fails but the stimulus-driven test passes,
    the problem is in stimulus generation rather than the kernel. Only
    `VectorMode.RC` is swept here — face selection is already covered in full by
    `test_sfpu_lerp_quasar`; this test adds the Dest-offset axis.
    """
    formats, dest_acc = formats_dest_acc
    if dest_index != 0 and _is_unpack_to_dest(formats, dest_acc):
        pytest.skip(
            "UNPACK→DEST always lands the three staged tiles at Dest 0/1/2 — "
            "_llk_unpack_unary_operand_ takes no Dest tile-offset argument — so a "
            "nonzero DEST_INDEX would make MATH read one tile past the staged set. "
            "The Dest-offset axis is exercised on the FPU datacopy path instead, "
            "which writes DST_INDEX + i explicitly."
        )

    torch_format_in = format_dict[formats.input_format]
    height, width = 32, 32

    start = (torch.ones(height * width, dtype=torch_format_in) * 2).flatten()
    end = (torch.ones(height * width, dtype=torch_format_in) * 10).flatten()
    ramp = (torch.arange(height * width) % 4).to(torch.float32) * 0.25
    weight = ramp.to(torch_format_in).flatten()

    _run_lerp(
        formats,
        dest_acc,
        implied_math_format,
        vector_mode,
        start,
        end,
        weight,
        dest_index=dest_index,
    )
