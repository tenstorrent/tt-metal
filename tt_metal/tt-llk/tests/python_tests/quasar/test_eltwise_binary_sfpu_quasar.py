# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import (
    BinarySFPUGolden,
    get_golden_generator,
    quantize_mx_stimuli,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    InputOutputFormat,
    generate_sfpu_format_dest_acc_combinations,
    input_output_formats,
    is_invalid_quasar_sfpu_format_combination,
    parametrize,
    runtime,
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
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    SFPU_BINARY_OP,
    SFPU_TILE_INDICES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
    ZERO_POINT,
)
from helpers.tile_constants import MAX_NUM_FACES, MAX_TILE_ELEMENTS
from helpers.utils import passed_test


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed the RNG once per test so stimuli are deterministic across runs."""
    torch.manual_seed(42)


_CPP_SOURCE = "sources/quasar/eltwise_binary_sfpu_quasar_test.cpp"

# Shared (src0_idx, src1_idx, dst_idx) tile-index variants exercised by every
# binary SFPU family
_TILE_INDEX_VARIANTS = [(0, 1, 0), (2, 3, 0)]


def _stage_binary_operands(op0_flat, op1_flat, tile_indices, dtype):
    """Lay two binary operands into a TILE_CNT-tile buffer_A so buffer_A[i] maps
    to Dest[i]: op0 at tile src0_idx, op1 at tile src1_idx, gaps zero-filled.
    The kernel writes the result to Dest[dst_idx], which may alias an operand
    tile (e.g. (0, 1, 0)) or be disjoint (e.g. (2, 3, 0)). Shared by the max/min
    (float/int) and quant (raw-int32) families."""
    src0_idx, src1_idx, dst_idx = tile_indices
    tile_cnt = max(src0_idx, src1_idx, dst_idx) + 1

    def _pad_to_tile(flat):
        if len(flat) < MAX_TILE_ELEMENTS:
            return torch.cat(
                [flat, torch.zeros(MAX_TILE_ELEMENTS - len(flat), dtype=dtype)]
            )
        return flat

    tiles = [torch.zeros(MAX_TILE_ELEMENTS, dtype=dtype) for _ in range(tile_cnt)]
    tiles[src0_idx] = _pad_to_tile(op0_flat.flatten())
    tiles[src1_idx] = _pad_to_tile(op1_flat.flatten())
    return torch.cat(tiles), tile_cnt


def _get_valid_implied_math_formats(fmt: FormatConfig):
    """Valid IMPLIED_MATH_FORMAT settings for a format, shared by every binary
    family: MX formats run only with ImpliedMathFormat.Yes; all others run both."""
    if fmt.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def _run_sfpu_binary_llk_golden(
    formats,
    dest_acc,
    implied_math_format,
    tile_indices,
    mathop,
    binary_op,
    prepare_stimuli,
    post_check=None,
):
    """Shared driver for the unpack-to-dest, LLK-golden binary SFPU ops.

    ``prepare_stimuli(formats, input_dimensions, src0_idx, src1_idx, mathop)``
    returns ``(src_A, tile_cnt_A, src_B)``; both operands live in ``src_A`` at
    tiles ``src0_idx`` / ``src1_idx``. ``post_check(res_tensor)`` is an optional
    extra assertion (e.g. div's x/x special-case lanes).
    """
    src0_idx, src1_idx, dst_idx = tile_indices
    input_dimensions = [(max(src0_idx, src1_idx, dst_idx) + 1) * 32, 32]
    num_faces = MAX_NUM_FACES

    src_A, tile_cnt_A, src_B = prepare_stimuli(
        formats, input_dimensions, src0_idx, src1_idx, mathop
    )

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_full = generate_golden(
        mathop,
        src_A,
        src0_idx,
        src1_idx,
        dst_idx,
        32,
        input_dimensions,
        formats.input_format,
    ).flatten()
    dst_start = dst_idx * MAX_TILE_ELEMENTS
    torch_format_out = format_dict[formats.output_format]
    golden_tensor = golden_full[dst_start : dst_start + MAX_TILE_ELEMENTS].to(
        torch_format_out
    )

    configuration = TestConfig(
        _CPP_SOURCE,
        formats,
        templates=[
            SFPU_BINARY_OP(binary_op),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
            # The shared unary-SFPU dispatch in sfpu_operations_quasar.h has a typecast
            # branch that references the non-dependent globals TYPECAST_IN_FORMAT /
            # TYPECAST_OUT_FORMAT, so every build that includes it must define them.
            TYPECAST_FORMATS(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            SFPU_TILE_INDICES(src0_idx, src1_idx, dst_idx),
            ZERO_POINT(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=1,
            num_faces=num_faces,
            twos_complement=formats.input_format.is_integer(),
        ),
        unpack_to_dest=True,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format_out)
    assert passed_test(golden_tensor, res_tensor, formats.output_format)

    if post_check is not None:
        post_check(res_tensor)


# ===========================================================================
# Family 1 — integer ops (add, mul, gt, lt, le, ge), Int32 only.
# Ported from test_sfpu_binary_quasar.py.
# ===========================================================================
def _prepare_int_stimuli(
    formats, input_dimensions, src0_idx, src1_idx, mathop, clamp_inputs
):
    """Integer stimuli: uniform over the dtype range, optionally clamped (int MUL
    clamps to keep the product representable). Both operands live in src_A."""
    data_format = formats.input_format
    iinfo = torch.iinfo(format_dict[data_format])
    spec = StimuliSpec.uniform(low=float(iinfo.min), high=float(iinfo.max - 1))
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=data_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=data_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )
    if clamp_inputs is not None:
        src_A = torch.clamp(src_A, -clamp_inputs, clamp_inputs)
        src_B = torch.clamp(src_B, -clamp_inputs, clamp_inputs)
    return src_A, tile_cnt_A, src_B


# (binary_op, mathop, clamp_inputs) — int MUL clamps to keep the product in range.
_INT_OPS = [
    ("ADD", MathOperation.SfpuElwadd, None),
    ("MUL", MathOperation.SfpuElwmulInt, 1000),
    ("GT", MathOperation.SfpuGtInt, None),
    ("LT", MathOperation.SfpuLtInt, None),
    ("LE", MathOperation.SfpuLeInt, None),
    ("GE", MathOperation.SfpuGeInt, None),
]


@pytest.mark.quasar
@pytest.mark.parametrize("tile_indices", _TILE_INDEX_VARIANTS)
@pytest.mark.parametrize(
    "binary_op, mathop, clamp_inputs", _INT_OPS, ids=[op for op, _, _ in _INT_OPS]
)
@pytest.mark.parametrize(
    "data_format, dest_acc", [(DataFormat.Int32, DestAccumulation.Yes)]
)
def test_eltwise_binary_sfpu_int_quasar(
    data_format, dest_acc, binary_op, mathop, clamp_inputs, tile_indices
):
    """Binary SFPU integer ops (add, mul, gt, lt, le, ge), Int32."""
    formats = InputOutputFormat(input_format=data_format, output_format=data_format)
    _run_sfpu_binary_llk_golden(
        formats,
        dest_acc,
        ImpliedMathFormat.No,
        tile_indices,
        mathop,
        binary_op,
        prepare_stimuli=lambda f, dims, s0, s1, op: _prepare_int_stimuli(
            f, dims, s0, s1, op, clamp_inputs
        ),
    )


# ===========================================================================
# Family 2 — float ops (mul, div). Ported from test_sfpu_binary_float_quasar.py.
# Operand/result tile-index variants exercise result-over-operand aliasing.
# ===========================================================================
# Crafted lanes in face 0 exercising the div special-case branches.
_DIV_SPECIAL_CASE_LANES = [
    (0, 0.0, 0.0, "nan"),
    (1, 1.5, 0.0, "pos_inf"),
    (2, -1.5, 0.0, "neg_inf"),
    (3, 2.7, 2.7, "one"),
    (4, -3.3, -3.3, "one"),
]


def _get_valid_float_formats_dest_acc():
    """Float16 + DestAccumulation.Yes is not supported."""
    formats = input_output_formats(
        [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
    )
    return [
        (fmt, dest_acc)
        for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(formats)
        if not (
            fmt.input_format == DataFormat.Float16 and dest_acc == DestAccumulation.Yes
        )
    ]


def _prepare_float_inputs(src_A, data_format, src0_idx, src1_idx, mathop):
    """Map [0,1) uniform stimuli into op-appropriate ranges (div: ±[0.25,4] + special
    lanes; mul: ±250)."""
    torch_format = format_dict[data_format]
    if mathop == MathOperation.SfpuElwdiv:
        scaled = (src_A.to(torch.float32) - 0.5) * 8.0
        sign = torch.where(scaled >= 0, torch.tensor(1.0), torch.tensor(-1.0))
        abs_scaled = torch.maximum(scaled.abs(), torch.tensor(0.25))
        scaled = (sign * abs_scaled).to(torch_format)
        flat = scaled.flatten()
        for lane, dividend, divisor, _ in _DIV_SPECIAL_CASE_LANES:
            flat[src0_idx * MAX_TILE_ELEMENTS + lane] = dividend
            flat[src1_idx * MAX_TILE_ELEMENTS + lane] = divisor
        return flat.reshape(scaled.shape)
    # SfpuElwmul
    scaled = ((src_A.to(torch.float32) - 0.5) * 500.0).to(torch_format)
    return scaled.flatten().reshape(scaled.shape)


def _prepare_float_stimuli(formats, input_dimensions, src0_idx, src1_idx, mathop):
    """Float stimuli: uniform [0,1) mapped to op-appropriate ranges by
    _prepare_float_inputs (div: ±[0.25,4] + special lanes; mul: ±250)."""
    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )
    src_A = _prepare_float_inputs(
        src_A, formats.input_format, src0_idx, src1_idx, mathop
    )
    return src_A, tile_cnt_A, src_B


def _check_div_special_cases(res_tensor):
    """The kernel's x/x branch forces an exact 1.0 regardless of reciprocal
    rounding — verify the 'one' special-case lanes."""
    for lane, _, _, kind in _DIV_SPECIAL_CASE_LANES:
        if kind != "one":
            continue
        actual = res_tensor[lane].item()
        assert (
            actual == 1.0
        ), f"x/x special case at lane {lane}: expected 1.0, got {actual}"


_FLOAT_OPS = [
    ("MUL", MathOperation.SfpuElwmul),
    ("DIV", MathOperation.SfpuElwdiv),
]


@pytest.mark.quasar
@pytest.mark.parametrize(
    "binary_op, mathop", _FLOAT_OPS, ids=[op for op, _ in _FLOAT_OPS]
)
@parametrize(
    formats_dest_acc=_get_valid_float_formats_dest_acc(),
    implied_math_format=lambda formats_dest_acc: _get_valid_implied_math_formats(
        formats_dest_acc[0]
    ),
    tile_indices=runtime(_TILE_INDEX_VARIANTS),
)
def test_eltwise_binary_sfpu_float_quasar(
    formats_dest_acc, implied_math_format, tile_indices, binary_op, mathop
):
    """Binary SFPU float ops (mul, div)."""
    formats, dest_acc = formats_dest_acc
    post_check = (
        _check_div_special_cases if mathop == MathOperation.SfpuElwdiv else None
    )
    _run_sfpu_binary_llk_golden(
        formats,
        dest_acc,
        implied_math_format,
        tile_indices,
        mathop,
        binary_op,
        prepare_stimuli=_prepare_float_stimuli,
        post_check=post_check,
    )


# ===========================================================================
# Family 3 — max / min (float + Int32). Ported from test_binary_max_min_quasar.py.
# Layout in0=Dest[0], in1=Dest[1], out=Dest[2]; dual unpack path; torch golden.
# ===========================================================================
SFPU_BINARY_MAX_MIN_FLOAT_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
    ],
)
SFPU_BINARY_MAX_MIN_INT32_FORMATS = input_output_formats([DataFormat.Int32], same=True)


def prepare_binary_max_min_inputs(src_A, src_B, input_format, output_format):
    """Two safe-range inputs for max/min (result == one operand verbatim)."""
    torch_fmt = format_dict[input_format]

    if not torch_fmt.is_floating_point:
        iinfo = torch.iinfo(torch_fmt)
        in0 = torch.clamp(src_A, iinfo.min, iinfo.max).to(torch_fmt).to(torch.float32)
        in1 = torch.clamp(src_B, iinfo.min, iinfo.max).to(torch_fmt).to(torch.float32)
        return in0, in1

    cap = min(format_elem_max(input_format), format_elem_max(output_format))
    min_mag, max_mag = compute_safe_input_magnitude_range(
        input_format, output_format, input_magnitude_cap=cap, output_magnitude_cap=cap
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


def _generate_max_min_combinations(
    formats_list: List[FormatConfig],
    dest_acc_for_format,
    implied_math_formats=(ImpliedMathFormat.No, ImpliedMathFormat.Yes),
    input_dimensions_list=([32, 32],),
):
    """Generate max/min (fmt, dest_acc, implied_math_format, is_max_op, input_dims) tuples."""
    combinations = []
    for fmt in formats_list:
        for dest_acc in dest_acc_for_format(fmt):
            if is_invalid_quasar_sfpu_format_combination(fmt, dest_acc):
                continue
            for implied_math_format in implied_math_formats:
                if implied_math_format not in _get_valid_implied_math_formats(fmt):
                    continue
                for is_max_op in [True, False]:
                    for input_dimensions in input_dimensions_list:
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


def _run_max_min(
    formats,
    dest_acc,
    implied_math_format,
    is_max_op,
    input_dimensions,
    spec,
    tile_indices,
    is_int,
):
    binary_op = "MAX" if is_max_op else "MIN"
    src0_idx, src1_idx, dst_idx = tile_indices
    num_faces = MAX_NUM_FACES

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    in0, in1 = prepare_binary_max_min_inputs(
        src_A, src_B, formats.input_format, formats.output_format
    )
    output_torch_fmt = format_dict[formats.output_format]

    if is_int:
        in0_int = in0.to(torch.int32)
        in1_int = in1.to(torch.int32)
        golden_int = (
            torch.maximum(in0_int, in1_int)
            if is_max_op
            else torch.minimum(in0_int, in1_int)
        )
        golden_tensor = golden_int.to(torch.float32)
        buffer_A_combined, tile_cnt = _stage_binary_operands(
            in0_int, in1_int, tile_indices, torch.int32
        )
        buffer_B_dummy = in1_int
        disable_format_inference = False
    else:
        torch_fmt = format_dict[formats.input_format]
        in0 = in0.to(torch_fmt)
        in1 = in1.to(torch_fmt)
        if formats.input_format.is_mx_format():
            in0_g = quantize_mx_stimuli(
                in0.flatten(), formats.input_format, num_faces
            ).reshape(in0.shape)
            in1_g = quantize_mx_stimuli(
                in1.flatten(), formats.input_format, num_faces
            ).reshape(in1.shape)
        else:
            in0_g, in1_g = in0, in1
        in0_f32 = in0_g.to(torch.float32)
        in1_f32 = in1_g.to(torch.float32)
        golden_f32 = (
            torch.maximum(in0_f32, in1_f32)
            if is_max_op
            else torch.minimum(in0_f32, in1_f32)
        )
        golden_tensor = golden_f32.to(output_torch_fmt)
        if formats.output_format.is_mx_format():
            golden_tensor = quantize_mx_stimuli(
                golden_tensor.flatten(), formats.output_format, num_faces
            ).reshape(golden_tensor.shape)
        buffer_A_combined, tile_cnt = _stage_binary_operands(
            in0, in1, tile_indices, in0.dtype
        )
        buffer_B_dummy = in1
        disable_format_inference = formats.input_format.is_mx_format()

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        _CPP_SOURCE,
        formats,
        templates=[
            SFPU_BINARY_OP(binary_op),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
            # The shared unary-SFPU dispatch in sfpu_operations_quasar.h has a typecast
            # branch that references the non-dependent globals TYPECAST_IN_FORMAT /
            # TYPECAST_OUT_FORMAT, so every build that includes it must define them.
            TYPECAST_FORMATS(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            SFPU_TILE_INDICES(src0_idx, src1_idx, dst_idx),
            ZERO_POINT(0),
        ],
        variant_stimuli=StimuliConfig(
            buffer_A_combined,
            formats.input_format,
            buffer_B_dummy,  # dummy buffer_B (unused by kernel)
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=num_faces,
            sfpu=True,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        disable_format_inference=disable_format_inference,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=output_torch_fmt)
    assert passed_test(golden_tensor, res_tensor, formats.output_format), (
        f"max/min failed for is_max_op={is_max_op}, "
        f"format={formats.input_format}->{formats.output_format}, dest_acc={dest_acc}"
    )


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_is_max_input_dims=_generate_max_min_combinations(
        SFPU_BINARY_MAX_MIN_FLOAT_FORMATS,
        dest_acc_for_format=lambda fmt: (
            (DestAccumulation.Yes,)
            if fmt.input_format.is_32_bit()
            else (DestAccumulation.No,)
        ),
    ),
    tile_indices=runtime(_TILE_INDEX_VARIANTS),
)
def test_eltwise_binary_sfpu_max_min_float_quasar(
    formats_dest_acc_implied_math_is_max_input_dims,
    tile_indices,
):
    """Binary SFPU max/min (float + MX)."""
    formats, dest_acc, implied_math_format, is_max_op, input_dimensions = (
        formats_dest_acc_implied_math_is_max_input_dims
    )
    spec = StimuliSpec.uniform(low=-0.9, high=1.1)
    _run_max_min(
        formats,
        dest_acc,
        implied_math_format,
        is_max_op,
        input_dimensions,
        spec,
        tile_indices,
        is_int=False,
    )


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_is_max_input_dims=_generate_max_min_combinations(
        SFPU_BINARY_MAX_MIN_INT32_FORMATS,
        dest_acc_for_format=lambda fmt: (DestAccumulation.Yes,),
        implied_math_formats=(ImpliedMathFormat.No,),
    ),
    tile_indices=runtime(_TILE_INDEX_VARIANTS),
)
def test_eltwise_binary_sfpu_max_min_int32_quasar(
    formats_dest_acc_implied_math_is_max_input_dims,
    tile_indices,
):
    """Binary SFPU max/min (Int32)."""
    formats, dest_acc, _implied_math_format, is_max_op, input_dimensions = (
        formats_dest_acc_implied_math_is_max_input_dims
    )
    iinfo = torch.iinfo(format_dict[formats.input_format])
    spec = StimuliSpec.uniform(low=float(iinfo.min), high=float(iinfo.max - 1))
    _run_max_min(
        formats,
        dest_acc,
        ImpliedMathFormat.No,
        is_max_op,
        input_dimensions,
        spec,
        tile_indices,
        is_int=True,
    )


# ===========================================================================
# Family 4 — quant / requant / dequant. Mixed int/float operands with a runtime
# fp32 zero-point. Ported from Blackhole ckernel_sfpu_quant.h.
#
#   quant:   out_int32 = clamp(round(A_fp32 * scale_fp32 + zp), -127, 127)
#   requant: out_int32 = clamp(round(int32_to_fp32(A) * scale_fp32 + zp), -127, 127)
#   dequant: out_fp32  = (int32_to_fp32(A) - zp) * scale_fp32
#
# Operand A and the fp32 scale B are both staged into buffer_A as raw 32-bit words
# (Int32 tag, twos_complement=True => identity bit copy). The unpack-to-dest path
# moves the raw 32 bits into Dest unchanged; the kernel re-interprets each operand
# via its explicit per-operand SFPLOAD sfpmem mode (FP32 vs INT32). The packer
# format is set per op (Int32 for quant/requant, Float32 for dequant) by patching
# the inferred FormatConfig, since input and output formats differ.
# ===========================================================================
# The binary SFPU params wrapper calls the kernel once per face (4 faces); each
# call runs ITERATIONS=8 unrolled rows, covering the whole 1024-element dst tile.
_PROCESSED_ELEMS = MAX_TILE_ELEMENTS

# SMAG8 saturation: the STOCH_RND fp32->sint8 path clamps magnitude to 127, so the
# representable output range is [-127, +127] (-128 is NOT representable).
_QUANT_INT8_MIN = -127
_QUANT_INT8_MAX = 127


def _fp32_bits_as_int32(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a float32 tensor's bits as int32 (no numeric conversion)."""
    return t.to(torch.float32).contiguous().view(torch.int32)


_QUANT_OPS = ["QUANT", "REQUANT", "DEQUANT"]


def _run_quant(binary_op, tile_indices):
    src0_idx, src1_idx, dst_idx = tile_indices
    dest_acc = DestAccumulation.Yes  # all quant endpoints are 32-bit
    num_faces = MAX_NUM_FACES

    is_dequant = binary_op == "DEQUANT"
    a_is_int = binary_op in ("REQUANT", "DEQUANT")
    output_format = DataFormat.Float32 if is_dequant else DataFormat.Int32

    # ---- stimuli (one tile's worth of active datums) ----
    torch.manual_seed(42)
    n = MAX_TILE_ELEMENTS

    # Scale: small positive fp32 so the quantized magnitudes stay inside [-127,127].
    scale = torch.rand(n, dtype=torch.float32) * 0.04 + 0.01  # [0.01, 0.05)
    # Zero-point: a small fp32 offset.
    zero_point = 3.0

    if a_is_int:
        # int32 operand A in a modest range so int*scale stays well within int8.
        a_vals = torch.randint(-2000, 2000, (n,), dtype=torch.int32)
        a_float = a_vals.to(torch.float32)
        a_staged = a_vals  # already int32 (2's-comp; twos_complement=True stages raw)
    else:
        # fp32 operand A.
        a_float = (torch.rand(n, dtype=torch.float32) - 0.5) * 4000.0  # +/-2000
        a_vals = a_float
        a_staged = _fp32_bits_as_int32(a_float)

    b_staged = _fp32_bits_as_int32(scale)

    buffer_A, tile_cnt = _stage_binary_operands(
        a_staged, b_staged, tile_indices, torch.int32
    )

    # ---- golden ----
    if is_dequant:
        golden_active = (a_float - zero_point) * scale  # fp32
        golden_active = golden_active.to(torch.float32)
        out_dtype = torch.float32
    else:
        prod = a_float * scale + zero_point
        rounded = torch.round(prod)  # round-half-to-even matches SFP STOCH_RND NearEven
        clamped = torch.clamp(rounded, _QUANT_INT8_MIN, _QUANT_INT8_MAX)
        golden_active = clamped.to(torch.int32)
        # STOCH_RND emits a sign-magnitude result: a negative value whose magnitude
        # rounds to 0 becomes sign-magnitude negative-zero (0x80000000). The
        # SM32_TO_2SC cast maps that to 2's-complement -1 (not +0), so model it.
        neg_zero = (golden_active == 0) & (prod < 0)
        golden_active[neg_zero] = -1
        out_dtype = torch.int32

    golden_tensor = torch.zeros(MAX_TILE_ELEMENTS, dtype=out_dtype)
    golden_tensor[:_PROCESSED_ELEMS] = golden_active[:_PROCESSED_ELEMS]

    # zero_point bits passed to the kernel init (DEQUANT negates the contract).
    import struct as _struct

    zp_for_kernel = -zero_point if is_dequant else zero_point
    zp_bits = _struct.unpack("<I", _struct.pack("<f", zp_for_kernel))[0]

    # buffer_A is raw int32; operand B's fp32 bits ride along as int32 words.
    formats = InputOutputFormat(
        input_format=DataFormat.Int32, output_format=output_format
    )

    configuration = TestConfig(
        _CPP_SOURCE,
        formats,
        templates=[
            SFPU_BINARY_OP(binary_op),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
            TYPECAST_FORMATS(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            SFPU_TILE_INDICES(src0_idx, src1_idx, dst_idx),
            ZERO_POINT(zp_bits),
        ],
        variant_stimuli=StimuliConfig(
            buffer_A,
            DataFormat.Int32,
            buffer_A[:MAX_TILE_ELEMENTS],  # dummy buffer_B (unused by kernel)
            DataFormat.Int32,
            output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=num_faces,
            twos_complement=True,  # raw 32-bit identity staging
        ),
        unpack_to_dest=True,
        dest_acc=dest_acc,
        disable_format_inference=True,
    )

    # Packer reads the SFPU result tile from Dest. For dequant the result is fp32;
    # for quant/requant it is int32. disable_format_inference ties pack_src to the
    # (Int32) input format, so override it to match the actual output datum width.
    fc = configuration.formats_config[0]
    fc.pack_src = output_format
    fc.pack_dst = output_format
    fc.pack_S_src = output_format
    fc.pack_S_dst = output_format

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=out_dtype)
    # Compare only the rows the kernel processes (8 iterations * faces).
    assert passed_test(
        golden_tensor[:_PROCESSED_ELEMS],
        res_tensor[:_PROCESSED_ELEMS],
        output_format,
    ), f"{binary_op} mismatch (tile_indices={tile_indices})"


@pytest.mark.quasar
@pytest.mark.parametrize("tile_indices", _TILE_INDEX_VARIANTS)
@pytest.mark.parametrize("binary_op", _QUANT_OPS, ids=_QUANT_OPS)
def test_eltwise_binary_sfpu_quant_quasar(binary_op, tile_indices):
    """Binary SFPU quant family (quant / requant / dequant), Int32-staged operands
    with a runtime fp32 zero-point; output Int32 (quant/requant) or Float32
    (dequant)."""
    _run_quant(binary_op, tile_indices)
