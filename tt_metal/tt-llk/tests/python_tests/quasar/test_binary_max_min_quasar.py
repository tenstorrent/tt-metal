# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# AI-generated — run_id: 2026-04-29_binary_max_min_quasar_dualpath
#
# Three-operand SFPU test for binary_max_min on Quasar.
#
# Buffer layout (in buffer_A, tile_count_A=2):
#   tile 0 (in0) + tile 1 (in1) both in buffer_A
#   SFPU computes max/min(Dest[0], Dest[1]) → Dest[2]
#   PACK reads result from Dest[2]   (tile_count_res=1)
#
# Two execution paths in the kernel, picked from the format:
#   * 32-bit formats (Float32, Int32) with dest_acc=Yes   → unpack_to_dest=True
#   * Non-32-bit / MX formats with dest_acc=No            → unpack_to_dest=False
#                                                           (UNPACK→SrcA→FPU datacopy→Dest)
#
# Format matrix:
#   Float variant: Float16, Float16_b, Float32, MxFp8R, MxFp8P
#   Int variant:   Int32, UInt8 (UInt32 deferred — not in VALID_QUASAR_DEST_REG_FORMATS)

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import quantize_mx_stimuli
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    input_output_formats,
    is_invalid_quasar_sfpu_format_combination,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import (
    apply_log_uniform_magnitudes,
    compute_safe_input_magnitude_range,
    format_elem_max,
    generate_stimuli,
)
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    IS_MAX_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test

# ─── Input preparation ────────────────────────────────────────────────────────


def prepare_binary_max_min_inputs(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
) -> tuple:
    """
    Prepare two input tensors for binary max/min with safe value ranges.

    Returns (in0, in1) clamped to safe representable range for the given
    (input_format, output_format) pair. Both tensors have the same format.

    For float formats: log-uniform magnitudes inside the format-aware safe
    range from `compute_safe_input_magnitude_range`. The result of max/min
    is one of the operands verbatim, so the per-operand cap is just the
    smaller of the input and output format max magnitudes — this prevents
    MX block-shared-exponent flushing when adjacent lanes within a 32-element
    block straddle ranges the block exponent and 2-/3-bit mantissa cannot
    span.
    For integer formats: values inside ~1/8 of the format's representable
    range to keep clear of extremes that trigger sign-magnitude edge cases.
    """
    torch_fmt = format_dict[input_format]

    # Integer formats: scale to a safe fraction of the representable range.
    # iinfo.max // 8 keeps Int32 at ~2^28 and shrinks proportionally for UInt8.
    if not torch_fmt.is_floating_point:
        iinfo = torch.iinfo(torch_fmt)
        max_val = iinfo.max // 8
        in0 = (
            torch.clamp(src_A * max_val, iinfo.min, iinfo.max)
            .to(torch_fmt)
            .to(torch.float32)
        )
        in1 = (
            torch.clamp(src_B * max_val, iinfo.min, iinfo.max)
            .to(torch_fmt)
            .to(torch.float32)
        )
        return in0, in1

    # max/min returns one operand verbatim → result magnitude == max(|in0|, |in1|)
    cap = min(format_elem_max(input_format), format_elem_max(output_format))
    min_mag, max_mag = compute_safe_input_magnitude_range(
        input_format,
        output_format,
        input_magnitude_cap=cap,
        output_magnitude_cap=cap,
    )

    in0 = apply_log_uniform_magnitudes(
        src_A,
        min_magnitude=min_mag,
        max_magnitude=max_mag,
        cast_to_format=input_format,
        sign_source=src_A,
    )
    in1 = apply_log_uniform_magnitudes(
        src_B,
        min_magnitude=min_mag,
        max_magnitude=max_mag,
        cast_to_format=input_format,
        sign_source=src_B,
    )
    return in0, in1


# ─── Format lists ─────────────────────────────────────────────────────────────

# Float variant — pairs covered by the dual-path kernel.
SFPU_BINARY_MAX_MIN_FLOAT_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
    ],
)

# Int variant: Int32 + UInt8 (UInt32 deferred — not in VALID_QUASAR_DEST_REG_FORMATS).
SFPU_BINARY_MAX_MIN_INT32_FORMATS = input_output_formats(
    [DataFormat.Int32],
    same=True,
)


# ─── Combination generators ────────────────────────────────────────────────────


def generate_binary_max_min_float_combinations(formats_list: List[FormatConfig]):
    """
    Generate (format, dest_acc, implied_math_format, is_max_op, input_dimensions) tuples
    for the float variant (non-Int32 formats).

    Yes-count: 5 pairs. dest_acc filtered per SFPU bit-width rule.
    """
    combinations = []
    for fmt in formats_list:
        in_fmt = fmt.input_format

        # SFPU bit-width rule: 32-bit input → dest_acc=Yes only; else dest_acc=No only
        dest_acc_modes = (
            (DestAccumulation.Yes,) if in_fmt.is_32_bit() else (DestAccumulation.No,)
        )

        for dest_acc in dest_acc_modes:
            if is_invalid_quasar_sfpu_format_combination(fmt, dest_acc):
                continue

            for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]:
                if (
                    in_fmt.is_mx_format()
                    and implied_math_format == ImpliedMathFormat.No
                ):
                    continue  # MX formats require implied_math_format=Yes

                for is_max_op in [True, False]:  # max AND min
                    for input_dimensions in [[32, 32]]:
                        combinations.append(
                            (
                                fmt,
                                dest_acc,
                                implied_math_format,
                                is_max_op,
                                input_dimensions,
                            )
                        )

    return combinations


def generate_binary_max_min_int32_combinations(formats_list: List[FormatConfig]):
    """
    Generate combinations for the int variant.

    Int32 / UInt8 both use dest_acc=Yes so the math thread enables
    EN_INT32_MATH_FORMAT and Dest holds INT32 sign-mag values. Without it,
    sub-32-bit integer src would go through the FP16 datacopy path and the byte
    values would be re-encoded as FP16, corrupting integer ordering.
    """
    combinations = []
    for fmt in formats_list:
        dest_acc_modes = (DestAccumulation.Yes,)

        for dest_acc in dest_acc_modes:
            if is_invalid_quasar_sfpu_format_combination(fmt, dest_acc):
                continue

            for is_max_op in [True, False]:  # max AND min
                for input_dimensions in [[32, 32]]:
                    combinations.append((fmt, dest_acc, is_max_op, input_dimensions))

    return combinations


# ─── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_is_max_input_dims=generate_binary_max_min_float_combinations(
        SFPU_BINARY_MAX_MIN_FLOAT_FORMATS
    ),
)
def test_binary_max_min_float_quasar(formats_dest_acc_implied_math_is_max_input_dims):
    """
    Test float variant of binary_max_min (calculate_binary_max_min) on Quasar.

    Verifies element-wise max and min across two Dest tile regions using
    SFPSWAP (sign-magnitude comparison = FP32 total order).

    Golden: torch.maximum / torch.minimum applied element-wise.
    """
    (formats, dest_acc, implied_math_format, is_max_op, input_dimensions) = (
        formats_dest_acc_implied_math_is_max_input_dims[0]
    )

    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
        negative_values=True,
    )

    # Prepare in0 and in1 inputs with safe value ranges
    in0, in1 = prepare_binary_max_min_inputs(
        src_A, src_B, formats.input_format, formats.output_format
    )

    # Convert to target format
    torch_fmt = format_dict[formats.input_format]
    in0 = in0.to(torch_fmt)
    in1 = in1.to(torch_fmt)

    num_faces = 4

    # MX formats use a 32-element shared block exponent. Quantizing both inputs
    # through the same pack→unpack roundtrip the hardware applies makes the
    # golden compare against what the kernel actually sees in Dest, not the
    # higher-precision bfloat16 source values.
    if formats.input_format.is_mx_format():
        in0_for_golden = quantize_mx_stimuli(
            in0.flatten(), formats.input_format, num_faces
        ).reshape(in0.shape)
        in1_for_golden = quantize_mx_stimuli(
            in1.flatten(), formats.input_format, num_faces
        ).reshape(in1.shape)
    else:
        in0_for_golden = in0
        in1_for_golden = in1

    in0_f32 = in0_for_golden.to(torch.float32)
    in1_f32 = in1_for_golden.to(torch.float32)
    if is_max_op:
        golden_f32 = torch.maximum(in0_f32, in1_f32)
    else:
        golden_f32 = torch.minimum(in0_f32, in1_f32)

    output_torch_fmt = format_dict[formats.output_format]
    golden_tensor = golden_f32.to(output_torch_fmt)

    # Mirror the output-side MX pack quantization: the kernel re-quantizes the
    # max/min result with a fresh per-block exponent on the way back to L1.
    if formats.output_format.is_mx_format():
        golden_tensor = quantize_mx_stimuli(
            golden_tensor.flatten(), formats.output_format, num_faces
        ).reshape(golden_tensor.shape)

    # buffer_A has 2 tiles: tile 0 = in0, tile 1 = in1, concatenated.
    # StimuliConfig with tile_count_A=2 reads buffer_A[0:1024] as tile 0
    # and buffer_A[1024:2048] as tile 1 (stride = MAX_TILE_ELEMENTS=1024).
    # buffer_B is a dummy (required by StimuliConfig; not written to the kernel).
    buffer_A_combined = torch.cat([in0.flatten(), in1.flatten()])

    # If in0/in1 have fewer than 1024 elements (e.g. partial faces), pad to 1024 each
    max_tile_elements = 1024
    if len(in0.flatten()) < max_tile_elements:
        pad_len = max_tile_elements - len(in0.flatten())
        buffer_A_combined = torch.cat(
            [
                in0.flatten(),
                torch.zeros(pad_len, dtype=in0.dtype),
                in1.flatten(),
                torch.zeros(pad_len, dtype=in1.dtype),
            ]
        )

    # 32-bit input + dest_acc=Yes  → unpack_to_dest=True
    # Everything else (incl. Float16_b, MxFp8R, MxFp8P) → unpack_to_dest=False
    # (UNPACK→SrcA→FPU datacopy→Dest path; required for MX block formats).
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        "sources/quasar/sfpu_binary_max_min_quasar_test.cpp",
        formats,
        templates=[
            IS_MAX_OP(is_max_op=is_max_op),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(2),  # 2 input tiles in buffer_A
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            buffer_A_combined,  # in0 (tile 0) + in1 (tile 1) concatenated
            formats.input_format,
            in1,  # dummy in buffer_B (not used by kernel)
            formats.input_format,
            formats.output_format,
            tile_count_A=2,
            tile_count_B=1,
            tile_count_res=1,  # kernel packs only the SFPU output (Dest[2])
            num_faces=num_faces,
            sfpu=True,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        disable_format_inference=formats.input_format.is_mx_format(),
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result length {len(res_from_L1)} != golden length {len(golden_tensor)}"

    res_tensor = torch.tensor(res_from_L1, dtype=output_torch_fmt)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
    ), (
        f"Assert against golden failed for is_max_op={is_max_op}, "
        f"format={formats.input_format}->{formats.output_format}, dest_acc={dest_acc}"
    )


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_is_max_input_dims=generate_binary_max_min_int32_combinations(
        SFPU_BINARY_MAX_MIN_INT32_FORMATS
    ),
)
def test_binary_max_min_int32_quasar(formats_dest_acc_is_max_input_dims):
    """
    Test int variant of binary_max_min (calculate_binary_max_min_int32) on Quasar.

    Covers Int32 and UInt8: both rebase into INT32 sign-mag in Dest via
    EN_INT32_MATH_FORMAT, so the same kernel handles both. Verifies element-wise
    max and min using SFPSWAP + CC-guarded correction to reconcile sign-magnitude
    vs two's-complement ordering for negative pairs.

    UInt32 is deferred — not in VALID_QUASAR_DEST_REG_FORMATS.

    Golden: torch.maximum / torch.minimum on int32 values.
    """
    (formats, dest_acc, is_max_op, input_dimensions) = (
        formats_dest_acc_is_max_input_dims[0]
    )

    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
        negative_values=True,
    )

    # Prepare integer inputs: scaled into ~1/8 of the format's representable
    # range, returned as float32-container tensors.
    in0_f, in1_f = prepare_binary_max_min_inputs(
        src_A, src_B, formats.input_format, formats.output_format
    )

    # Convert to actual int32 for packing and golden computation
    in0_int = in0_f.to(torch.int32)
    in1_int = in1_f.to(torch.int32)

    # Golden: element-wise signed max or min in two's-complement int32
    # (the hardware kernel corrects sign-magnitude to two's-complement ordering)
    if is_max_op:
        golden_int = torch.maximum(in0_int, in1_int)
    else:
        golden_int = torch.minimum(in0_int, in1_int)

    golden_tensor = golden_int.to(torch.float32)

    num_faces = 4

    output_torch_fmt = format_dict[formats.output_format]

    # Concatenate in0 and in1 into buffer_A (tile_count_A=2)
    # pack_int32 expects int32 tensors (converts two's-complement → sign-magnitude for HW)
    max_tile_elements = 1024
    buffer_A_combined = torch.cat([in0_int.flatten(), in1_int.flatten()])
    if len(in0_int.flatten()) < max_tile_elements:
        pad_len = max_tile_elements - len(in0_int.flatten())
        buffer_A_combined = torch.cat(
            [
                in0_int.flatten(),
                torch.zeros(pad_len, dtype=torch.int32),
                in1_int.flatten(),
                torch.zeros(pad_len, dtype=torch.int32),
            ]
        )

    # Int32 is 32-bit with dest_acc=Yes → unpack_to_dest=True path.
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        "sources/quasar/sfpu_binary_max_min_quasar_test.cpp",
        formats,
        templates=[
            IS_MAX_OP(is_max_op=is_max_op),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(2),  # 2 input tiles in buffer_A
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            buffer_A_combined,  # in0 (tile 0) + in1 (tile 1) concatenated
            formats.input_format,
            in1_int,  # dummy in buffer_B (not used by kernel)
            formats.input_format,
            formats.output_format,
            tile_count_A=2,
            tile_count_B=1,
            tile_count_res=1,  # kernel packs only the SFPU output (Dest[2])
            num_faces=num_faces,
            sfpu=True,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result length {len(res_from_L1)} != golden length {len(golden_tensor)}"

    res_tensor = torch.tensor(res_from_L1, dtype=output_torch_fmt)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), f"Assert against golden failed for is_max_op={is_max_op}, int32 format"
