# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from helpers.bf16_ties import (
    BF16_FRAC_BITS,
    BF16_SIG_ONE,
    assert_is_bf16_tie,
    fp32_bits,
)
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.golden_generators import ScalarBinopGolden, get_golden_generator
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.sfpu_domains import (
    SPECIALS_READY_OPS,
    edge_spec,
    specials_after_nan_sign_gate,
    specials_safe,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    SFPU_BINOP_MODE,
    SFPU_UNARY_SCALAR,
)
from helpers.tile_constants import MAX_TILE_ELEMENTS
from helpers.utils import passed_test

# The scalar is a swept axis: zero, unity, a sign flip, a large multiplier, and a value
# small enough to matter against the tolerance. Kept deliberately small -- inputs are
# uniform(-1, 1), so |scalar| <= 8 keeps every op's result inside the range where the
# default bf16 tolerance is meaningful.
#
# Split across two tests rather than swept in one: the full axis is 6 scalars x 5 ops x
# 2 formats x 2 dest modes, which is more hardware variants than presubmit should spend on
# one kernel parameter. Presubmit drives the ops at a single representative scalar and the
# remaining values run nightly.
_PRESUBMIT_SCALAR = 2.0
_SCALARS = (0.0, 1.0, 2.0, -2.0, 8.0, 0.25)
_NIGHTLY_SCALARS = tuple(s for s in _SCALARS if s != _PRESUBMIT_SCALAR)

# ScalarDiv is the one op whose scalar is not the value the kernel sees: the host inverts the
# divisor at compile time and the kernel only multiplies, so `d` never reaches the device.
# That also means a divide-by-zero cannot be reached through this op at all -- 1/0 would be
# computed on the host -- so 0.0 is not a legal divisor here rather than an untested edge.
_ZERO_DIVISOR_UNREACHABLE = (
    "ScalarDiv inverts the divisor on the host; d=0 is not a device path"
)


def _scalar_bits_for(mathop, scalar):
    """The 32-bit pattern the kernel is given for *mathop* at *scalar*."""
    if mathop == MathOperation.ScalarDiv:
        return fp32_bits(1.0 / scalar)
    return fp32_bits(scalar)


# Keep inputs small and bounded so the bf16 result stays accurate across all five scalar
# ops (add/sub/mul/div/rsub) and both dest-accumulation modes.
_DEFAULT_TENSOR_SPEC = StimuliSpec.uniform(low=-1.0, high=1.0)


def _run_sfpu_binop_scalar(
    formats,
    dest_acc,
    mathop,
    scalar=_PRESUBMIT_SCALAR,
    input_dimensions=[32, 32],
    spec_A=None,
    src_A=None,
    exact=False,
):
    """Drive one scalar binop variant.

    *spec_A* overrides the tensor operand. The scalar axis has been swept since the
    presubmit/nightly split, but the tensor operand had no knob at all and was pinned to
    the default above, so the only way to reach an edge on it was to edit this function.

    *src_A* replaces the tensor operand with explicit values (one tile, *input_dimensions*
    elements); *exact* then compares bit for bit instead of within tolerance, which only a
    kernel that narrows with round-to-nearest-even can pass.
    """
    torch.manual_seed(0)
    scalar_bits = _scalar_bits_for(mathop, scalar)

    spec_a = _DEFAULT_TENSOR_SPEC if spec_A is None else spec_A

    generated_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_a,
    )
    if src_A is None:
        src_A = generated_A
    else:
        assert src_A.numel() == generated_A.numel(), "src_A must fill the whole operand"
        src_A = src_A.to(generated_A.dtype).reshape(generated_A.shape)

    generate_golden = get_golden_generator(ScalarBinopGolden)
    golden = generate_golden(
        mathop, src_A, scalar_bits, formats.output_format, dest_acc
    )

    configuration = TestConfig(
        "sources/sfpu_binop_scalar_test.cpp",
        formats,
        templates=[
            SFPU_BINOP_MODE(mathop),
            SFPU_UNARY_SCALAR(scalar_bits),
            APPROX_MODE(ApproximationMode.No),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    res_from_L1 = res_from_L1[:1024]

    assert len(res_from_L1) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    golden_tensor = torch.tensor(golden, dtype=torch_format).flatten()
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    if exact:
        # max_ulp=0 is the bit-exact gate: zero representable steps between result and golden.
        assert passed_test(
            golden_tensor, res_tensor, formats.output_format, max_ulp=0
        ), "Result is not bit-identical to the golden"
        return

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


_SCALAR_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float32,
    ],
    same=True,
)

_SCALAR_OPS = [
    MathOperation.ScalarAdd,
    MathOperation.ScalarSub,
    MathOperation.ScalarMul,
    MathOperation.ScalarDiv,
    MathOperation.ScalarRsub,
]


def _skip_unsupported(formats, dest_acc, mathop, scalar):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")
    if (
        formats.input_format == DataFormat.Float16_b
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip("Float16_b not supported with DestAccumulation.Yes")
    if mathop == MathOperation.ScalarDiv and scalar == 0.0:
        pytest.skip(_ZERO_DIVISOR_UNREACHABLE)


@parametrize(
    formats=_SCALAR_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=_SCALAR_OPS,
)
def test_sfpu_binop_scalar(formats, dest_acc, mathop):
    _skip_unsupported(formats, dest_acc, mathop, _PRESUBMIT_SCALAR)
    _run_sfpu_binop_scalar(formats, dest_acc, mathop, scalar=_PRESUBMIT_SCALAR)


@pytest.mark.nightly
@parametrize(
    formats=_SCALAR_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=_SCALAR_OPS,
    scalar=list(_NIGHTLY_SCALARS),
)
def test_sfpu_binop_scalar_values(formats, dest_acc, mathop, scalar):
    """The rest of the scalar axis: zero, unity, a sign flip and a fractional multiplier."""
    _skip_unsupported(formats, dest_acc, mathop, scalar)
    _run_sfpu_binop_scalar(formats, dest_acc, mathop, scalar=scalar)


# Edge values on the *tensor* operand. All five ops are x (+|-|*|/) c for a compile-time c,
# which is smooth in x -- no pole, no knee -- so cat A and cat D contribute nothing and
# edge_spec() returns None unless specials are on. Cat B is their entire edge story, which is
# why this test only exists now the scalar ops are enrolled in SPECIALS_READY_OPS.
#
# Two of the eight (format, dest_acc) pairs survive both gates, and they are complementary
# rather than redundant -- between them they cover both sides of the delivery split:
#
#   Float32->Float32   dest_acc=Yes  unpack-to-dest, so a real -0.0 arrives
#   Float16_b->Float16_b dest_acc=No datacopy, so -0.0 arrives as +0.0 and is not sent
#
# The other six are excluded by _skip_unsupported (Float32 needs a 32-bit dest, Float16_b
# cannot use one) or by specials_safe.
#
# Still out of scope, and both need a per-op tolerance first -- the default bf16 tolerance is
# only meaningful while the result stays in range: |scalar| > 8, and +/-tiny / +/-large on the
# tensor operand. That is the pattern BINARY_CUSTOM_TOLERANCES uses for pow and xlogy.
@pytest.mark.nightly
@parametrize(
    formats=_SCALAR_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=_SCALAR_OPS,
)
def test_sfpu_binop_scalar_edges(formats, dest_acc, mathop):
    """Drive IEEE specials through the tensor operand of each scalar binop."""
    _skip_unsupported(formats, dest_acc, mathop, _PRESUBMIT_SCALAR)

    # Both gates, as everywhere else: SPECIALS_READY_OPS says the *golden* defines a result
    # for a non-finite input, specials_safe() says the *pipeline* delivers one intact.
    specials = mathop in SPECIALS_READY_OPS and specials_safe(
        formats.input_format, formats.output_format, dest_acc
    )

    # The unary sweep's gate, same rule and same helper: ScalarRsub builds `c - x` through
    # SFPMAD, so a NaN operand comes back as a NaN of the kernel's own making rather than the
    # one it was handed. See sfpu_domains.GENERATED_NAN_SIGN_OPS.
    specials = specials_after_nan_sign_gate(
        mathop,
        formats.input_format,
        formats.output_format,
        dest_acc,
        specials,
        TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE,
    )

    spec_A = edge_spec(
        mathop,
        formats.input_format,
        formats.output_format,
        specials=specials,
        dest_acc=dest_acc,
    )
    if spec_A is None:
        pytest.skip(
            reason=f"{mathop.name} has no edge values for this pipeline "
            "(smooth in x, and specials not preserved here)"
        )

    _run_sfpu_binop_scalar(
        formats, dest_acc, mathop, scalar=_PRESUBMIT_SCALAR, spec_A=spec_A
    )


# =============================================================================
# bf16 round-to-nearest-even narrowing. ScalarRsub is the one scalar op that rounds its fp32
# result into a bf16 Dest with RNE (the others truncate), and the default bf16 tolerance cannot
# tell RNE from truncation or from a miswired rounding bias, so its ties are pinned bit for bit.
# =============================================================================

# Half a bf16 ULP of [1, 2): s - x lands exactly between two bf16 neighbours for every
# x = +/-(1 + m / 128) whose result stays in that binade.
_RSUB_TIE_SCALAR = math.ldexp(1.0, -(BF16_FRAC_BITS + 1))


def _rsub_tie_tensor(count):
    """*count* bf16 values x for which s - x is an exact fp32 tie, s = _RSUB_TIE_SCALAR.

    x = -(1 + m/128) gives 1 + m/128 + s in [1, 2) for every m; x = +(1 + m/128) gives
    1 + m/128 - s, in [1, 2) for m >= 1 (m = 0 lands in [0.5, 1), where s is a whole ULP and
    the result is exact). Both parities of the bf16 LSB occur, which is where round-to-even and
    round-half-away differ.
    """
    values = [-(1.0 + m / BF16_SIG_ONE) for m in range(BF16_SIG_ONE)]
    values += [1.0 + m / BF16_SIG_ONE for m in range(1, BF16_SIG_ONE)]
    # Self-check on the host: every lane must be an exact tie, else the test proves nothing.
    for x in values:
        assert_is_bf16_tie(_RSUB_TIE_SCALAR - x, f"s - x for x = {x}")
    return torch.tensor(
        [values[i % len(values)] for i in range(count)], dtype=torch.bfloat16
    )


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    mathop=[MathOperation.ScalarRsub],
)
def test_sfpu_binop_scalar_bf16_rne_ties(formats, mathop):
    """Exact bf16 ties through ScalarRsub's narrowing must round to even, bit for bit.

    Guards the RNE helper the kernel shares with the binary and ternary SFPU ops: a truncating
    store or SFPSTOCHRND (ties away from zero on Blackhole) fails half the lanes.
    """
    _run_sfpu_binop_scalar(
        formats,
        DestAccumulation.No,
        mathop,
        scalar=_RSUB_TIE_SCALAR,
        src_A=_rsub_tie_tensor(MAX_TILE_ELEMENTS),  # one tile, the runner's default
        exact=True,
    )
