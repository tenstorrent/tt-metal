# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Ternary half of the SFPU parity set on Quasar: addcmul / addcdiv / lerp / snake_beta.

These four are Blackhole kernels written in pure ``sfpi::`` that Quasar has not received
yet. The op list comes from ``helpers/sfpu_port_quasar.py`` and every op is filtered
through :func:`is_ported`, so until a kernel header lands this module collects nothing and
the suite stays green; when one lands, its full sweep activates with no edit here.

Quasar already carries the ternary plumbing that ``where`` uses
(``llk_math_eltwise_ternary_sfpu_macros.h``, ``_llk_math_eltwise_ternary_sfpu_init_``), so
this is harness assembly rather than new infrastructure. The C++ side is
``sources/quasar/sfpu_ternary_quasar_test.cpp``, structurally a copy of the ``where`` test
with the op selected at compile time.
"""

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import TernarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_sfpu_format_dest_acc_combinations,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.sfpu_port_quasar import Arity, entries, is_ported
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    SFPU_TERNARY_OP,
    SFPU_TERNARY_SCALAR,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test

_CPP_SOURCE = "sources/quasar/sfpu_ternary_quasar_test.cpp"

# The addc kernels' fp32 multiplier. 2.0 rather than 1.0 so a kernel that ignored the
# scalar entirely would still fail: with 1.0, addcmul(a, b, c) and a + b * c coincide.
_TERNARY_SCALAR = 2.0
_TERNARY_SCALAR_BITS = 0x40000000

# Per-operand domains, as (low, high) per operand (a, b, c). Chosen so the result stays
# representable and each kernel's preconditions hold:
#   addcdiv    -- operand c is a divisor, so it must stay away from zero;
#   lerp       -- operand c is the interpolation weight, natural range [0, 1];
#   snake_beta -- operand c divides sin(b*a)^2, so it too must avoid zero.
_TERNARY_DOMAINS = {
    MathOperation.SfpuAddcmul: ((-8.0, 8.0), (-4.0, 4.0), (-4.0, 4.0)),
    MathOperation.SfpuAddcdiv: ((-8.0, 8.0), (-4.0, 4.0), (0.5, 4.0)),
    MathOperation.SfpuLerp: ((-8.0, 8.0), (-8.0, 8.0), (0.0, 1.0)),
    MathOperation.SfpuSnakeBeta: ((-4.0, 4.0), (-2.0, 2.0), (0.5, 4.0)),
}

# Ops that carry coverage for ckernel_sfpu_conversions.h, which has no kernel of its own
# (see CONVERSIONS_COVERAGE in helpers/sfpu_port_quasar.py). Both reach
# float32_to_bf16_rne when narrowing their fp32 result, so their stimuli seed exact
# round-nearest-even ties -- values whose fp32 -> bf16 rounding is decided by the
# even-mantissa rule rather than by magnitude. Without seeding, a tie is vanishingly
# unlikely to appear in random stimuli and the rounding path would go untested.
_RNE_TIE_OPS = (MathOperation.SfpuAddcdiv, MathOperation.SfpuLerp)

# fp32 values sitting exactly halfway between two bf16 neighbours: mantissa bit 16 set,
# all lower bits clear. RNE must round each to the neighbour with an even mantissa.
_RNE_TIE_BITS = (
    0x3F808000,  # 1.00390625  -> ties between 1.0 and 1.0078125
    0x3F818000,  # 1.01171875
    0x40008000,  # 2.00390625
    0x40808000,  # 4.0078125
    0xBF808000,  # -1.00390625 (sign must not change the tie decision)
    0xBF818000,  # -1.01171875
)


def _rne_tie_values(dtype: torch.dtype) -> torch.Tensor:
    """The tie values above, as a tensor of *dtype*."""
    bits = torch.tensor(list(_RNE_TIE_BITS), dtype=torch.int64).to(torch.int32)
    return bits.view(torch.float32).to(dtype)


def _get_valid_formats_dest_acc():
    """Float format x dest_acc matrix, minus the combination Quasar does not support."""
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


def _get_valid_implied_math_formats(fmt: FormatConfig):
    if fmt.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def _is_unpack_to_dest(fmt: FormatConfig, dest_acc: DestAccumulation) -> bool:
    """UNPACK->DEST is selected only for 32-bit inputs with dest_acc=Yes."""
    return fmt.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes


def _operand(
    mathop: MathOperation,
    which: int,
    input_format: DataFormat,
    input_dimensions,
    seed: int,
):
    """One operand tile, drawn uniformly from this op's band for that operand."""
    low, high = _TERNARY_DOMAINS[mathop][which]
    torch.manual_seed(seed)
    src, tile_cnt, _, _ = generate_stimuli(
        stimuli_format_A=input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec.uniform(low=0.0, high=1.0),
        spec_B=StimuliSpec.uniform(low=0.0, high=1.0),
    )
    scaled = low + src.to(torch.float32) * (high - low)
    return scaled.to(format_dict[input_format]), tile_cnt


def _ternary_cases():
    """(mathop, formats, dest_acc) for every ported ternary parity op.

    Empty while the parity kernels are unported, which is the normal state today.
    """
    cases = []
    for entry in entries(Arity.TERNARY):
        for mathop in entry.ops:
            if not is_ported(mathop):
                continue
            for fmt, dest_acc in _get_valid_formats_dest_acc():
                cases.append((mathop, fmt, dest_acc, entry.has_approx))
    return cases


_TERNARY_CASES = _ternary_cases()


@pytest.mark.quasar
@pytest.mark.skipif(
    not _TERNARY_CASES,
    reason="no ternary SFPU parity kernel is ported to Quasar yet",
)
@parametrize(
    ternary_case=_TERNARY_CASES,
    implied_math_format=lambda ternary_case: _get_valid_implied_math_formats(
        ternary_case[1]
    ),
    approx_mode=lambda ternary_case: (
        [ApproximationMode.No, ApproximationMode.Yes]
        if ternary_case[3]
        else [ApproximationMode.No]
    ),
    input_dimensions=runtime([[32, 32], [64, 64]]),
)
def test_sfpu_ternary_quasar(
    ternary_case, implied_math_format, approx_mode, input_dimensions
):
    """Ternary SFPU parity ops on Quasar, validated against TernarySFPUGolden.

    Three operand tiles are staged into buffer_A, datacopied into Dest at tile indices
    0/1/2, and the compile-time-selected op writes its result to Dest tile 0, which PACK
    writes back. Only ops whose Quasar kernel header exists are collected.
    """
    mathop, formats, dest_acc, _has_approx = ternary_case
    torch_format_in = format_dict[formats.input_format]

    operand_a, tile_cnt_single = _operand(
        mathop, 0, formats.input_format, input_dimensions, seed=42
    )
    operand_b, _ = _operand(mathop, 1, formats.input_format, input_dimensions, seed=43)
    operand_c, _ = _operand(mathop, 2, formats.input_format, input_dimensions, seed=44)

    if mathop in _RNE_TIE_OPS:
        # Seed the fp32->bf16 round-nearest-even ties that exercise
        # ckernel_sfpu_conversions.h's float32_to_bf16_rne through this carrier op.
        ties = _rne_tie_values(torch_format_in)
        flat_a = operand_a.flatten()
        n = min(ties.numel(), flat_a.numel())
        flat_a[:n] = ties[:n]
        operand_a = flat_a.reshape(operand_a.shape)

    src_A = torch.cat([operand_a, operand_b, operand_c])
    tile_cnt_A = tile_cnt_single * 3
    num_faces = 4

    generate_golden = get_golden_generator(TernarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        operand_a,
        operand_b,
        operand_c,
        _TERNARY_SCALAR_BITS,
        formats.output_format,
    )

    unpack_to_dest = _is_unpack_to_dest(formats, dest_acc)
    src_B_dummy = torch.zeros_like(operand_a)

    configuration = TestConfig(
        _CPP_SOURCE,
        formats,
        templates=[
            SFPU_TERNARY_OP(ternary_mathop=mathop),
            SFPU_TERNARY_SCALAR(ternary_scalar_bits=_TERNARY_SCALAR_BITS),
            APPROX_MODE(approx_mode),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
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
            tile_count_res=tile_cnt_single,
            num_faces=num_faces,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format_out = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format_out)
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
