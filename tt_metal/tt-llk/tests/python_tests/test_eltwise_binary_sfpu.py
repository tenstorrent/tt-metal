# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import struct
from dataclasses import replace
from fractions import Fraction
from typing import Dict

import numpy as np
import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.constraints import skip_if_32bit_input_without_dest_acc
from helpers.data_format_inference import effective_dest_acc, unpacks_to_dest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    BinarySFPUGolden,
    BroadcastGolden,
    get_golden_generator,
    quantize_input_to_unpack_format,
)
from helpers.llk_params import (
    ApproximationMode,
)
from helpers.llk_params import BroadcastType as LlkBroadcastType
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    DstRoundingMode,
    MathOperation,
    PerfRunType,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.sfpu_accuracy_budget import assert_against_contract
from helpers.sfpu_domains import (
    _OP_DOMAIN_REGISTRY,
    _SFPU_BINARY_OPS,
    BINARY_SPECIALS_READY_OPS,
    SHIFT_EDGE_AMOUNTS,
    edge_pair_values,
    exclude_undefined_pair,
    for_op,
    generated_nan_sign_is_asserted,
    integer_specials,
    ops_with_singularity,
    specials_safe,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import DistributionKind, StimuliSpec, generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    BROADCAST_TYPE,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    OUT_ZERO_POINT,
    SFPU_BCAST_DIM,
    SFPU_DST_ROUNDING_MODE,
    SIGN_MAGNITUDE_FORMAT,
    TILE_COUNT,
    ZERO_POINT,
    generate_input_dim,
)
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.tilize_untilize import tilize
from helpers.utils import (
    assert_bf16_separates_rne,
    assert_bit_exact,
    bf16_from_bits,
    fp32_bits_as_int32,
    int32_to_smag32,
    passed_test,
)

# =============================================================================
# Shared skip helpers
# =============================================================================


def _resolve_perf_args(is_perf, perf_report, run_types):
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")
    if run_types is None:
        run_types = [PerfRunType.L1_TO_L1]
    return run_types


def _skip_sfpu_lcm_dest_acc_bh(mathop, dest_acc):
    """SfpuLcm dest_acc=Yes is codegen-sensitive on Blackhole and hangs. See tt-metal#52997."""
    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and mathop == MathOperation.SfpuLcm
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip(
            "SfpuLcm dest_acc=Yes is codegen-sensitive and hangs on Blackhole; see tt-metal#52997"
        )


# Shared crafted-stimuli helpers. Several predicate/paired ops (mask, isclose, eq/ne,
# lt/gt/le/ge) need their two operand tiles filled from different per-position data, which
# the default random sweep cannot express. These builders produce those StimuliSpecs.

# Number of faces per tile for the [64, 32] two-tile binary harness layout
# (a 32x32 tile is 4 faces of 16x16, and input_dimensions=[64, 32] is 8 faces).
_FACES_PER_TILE = 4

#: The approximation mode the contract-gated drivers below compile, and so the one their
#: contract names.
_APPROX_MODE = ApproximationMode.No
_ELEMENTS_PER_TILE = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM


def _build_paired_tile_override(pairs, dtype):
    """Two-tile raw override from a list of (A, B) pairs: tile 0 holds every A, tile 1
    every B, paired by index (tilize pairs them that way).

    *pairs* is cycled to fill a whole tile rather than zero-padded, so every element is a
    pair the caller meant to drive.
    """
    if not pairs:
        raise ValueError("_build_paired_tile_override() needs at least one pair")
    a = [pairs[i % len(pairs)][0] for i in range(_ELEMENTS_PER_TILE)]
    b = [pairs[i % len(pairs)][1] for i in range(_ELEMENTS_PER_TILE)]
    return torch.tensor(a + b, dtype=dtype)


def _pair_operand_specs(spec_A, spec_B, input_dimensions):
    """Interleave two per-operand specs across every tile pair in the buffer.

    The kernel reads operand 0 from the even tile of each pair and operand 1 from the odd
    one, so the specs alternate every 4 faces across the whole buffer. `face_specs` is
    applied positionally and not cycled, so the list has to cover the real tile count.
    """
    tiles = (input_dimensions[0] * input_dimensions[1]) // _ELEMENTS_PER_TILE
    if tiles % 2:
        raise ValueError(
            f"SFPU binary needs a whole number of tile pairs, got {tiles} tiles "
            f"from input_dimensions={input_dimensions}"
        )
    face_specs = ([None] * _FACES_PER_TILE + [spec_B] * _FACES_PER_TILE) * (tiles // 2)
    return replace(spec_A, face_specs=face_specs)


def _face_spec(dist):
    """A per-face callable distribution as a StimuliSpec, with a fixed seed."""
    return StimuliSpec(distribution=dist, seed=0)


def _positions_and_ramp(size):
    """The (positions, 1..8 ramp) pair every paired builder below is built from.

    `size` is whatever the generator passes per face, so the builders never assume a face
    size. The ramp is non-zero everywhere and of order 1, so the offsets the other operand
    adds stay unambiguous against any rounding.
    """
    positions = torch.arange(size, dtype=torch.float32)
    return positions, 1.0 + (positions % 8)


# Which ops take their domain from _OP_DOMAIN_REGISTRY. Every op driven through sfpu_binary()
# belongs to exactly one of the three sets below, and _classify_stimuli_source() enforces that
# in the driver, so an unclassified op is a coverage question nobody asked rather than a
# default anybody chose:
#
#   _REGISTRY_DOMAIN_OPS            - rerouted onto the op's registered domain.
#   _UNREGISTERED_BINARY_OPS        - no registry entry exists; keeps the format default.
#   _REGISTERED_DEFAULT_STIMULI_OPS - a registry entry exists but is deliberately not used
#                                     here, with the reason recorded per op.
#
# The reroute is float-only on purpose: several of these ops also run on an integer format,
# where a float domain would collapse to a handful of values and gut the int coverage.

_REGISTRY_DOMAIN_OPS = frozenset(
    {
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
        MathOperation.SfpuElwrsub,
        MathOperation.SfpuElwdiv,
        MathOperation.SfpuElwpow,
        MathOperation.SfpuXlogy,
        MathOperation.SfpuLogaddexp,
        MathOperation.SfpuLogaddexp2,
    }
)

# Ops this suite drives that have no _OP_DOMAIN_REGISTRY entry at all, and so keep the
# format default. Not a TODO list — several are int-only or carry crafted stimuli — but it
# is checked below so that registering a domain for one of them shows up here as a diff
# rather than silently changing what that op is fed.
_UNREGISTERED_BINARY_OPS = frozenset(
    {
        MathOperation.SfpuAtan2,
        MathOperation.SfpuBinaryFmod,
        MathOperation.SfpuBinaryMax,
        MathOperation.SfpuBinaryMin,
        MathOperation.SfpuBinaryRemainder,
        MathOperation.SfpuBitwiseAnd,
        MathOperation.SfpuBitwiseOr,
        MathOperation.SfpuBitwiseXor,
        MathOperation.SfpuDivInt32,
        MathOperation.SfpuDivInt32Floor,
        MathOperation.SfpuElwEq,
        MathOperation.SfpuElwGe,
        MathOperation.SfpuElwGt,
        MathOperation.SfpuElwLe,
        MathOperation.SfpuElwLt,
        MathOperation.SfpuElwNe,
        MathOperation.SfpuEqInt,
        MathOperation.SfpuFmodInt32,
        MathOperation.SfpuGcd,
        MathOperation.SfpuIsclose,
        MathOperation.SfpuIscloseEqualNan,
        MathOperation.SfpuLcm,
        MathOperation.SfpuLogsigmoid,
        MathOperation.SfpuMask,
        MathOperation.SfpuMaskPosinf,
        MathOperation.SfpuMaxInt32,
        MathOperation.SfpuMaxUint32,
        MathOperation.SfpuMinInt32,
        MathOperation.SfpuMinUint32,
        MathOperation.SfpuMulInt32,
        MathOperation.SfpuNeInt,
        MathOperation.SfpuRemainderInt32,
        MathOperation.SfpuRemainderUint32,
        MathOperation.SfpuRsubInt32,
        MathOperation.SfpuCopyDest,
    }
)

# Ops with an _OP_DOMAIN_REGISTRY entry that this suite deliberately does not read, and why.
# Distinct from _UNREGISTERED_BINARY_OPS: there the registry has nothing to offer, here it
# has something we are choosing not to take.
_REGISTERED_DEFAULT_STIMULI_OPS: Dict[MathOperation, str] = {
    MathOperation.SfpuElwLeftShift: "driven as Int32 here; the registered float domain "
    "uniform(0, 255) is for the shift *amount* and the value operand needs the full "
    "int32 range, so crafted stimuli / the format default are what these want",
    MathOperation.SfpuElwRightShift: "as SfpuElwLeftShift",
    MathOperation.SfpuElwLogicalRightShift: "as SfpuElwLeftShift",
}

# Ops this file tests without going through sfpu_binary(), so no stimulus classification
# applies to them. Declared rather than merely absent, so that routing one of them onto
# the shared driver later forces a decision instead of silently picking the default.
_OPS_NOT_USING_SHARED_DRIVER: Dict[MathOperation, str] = {
    MathOperation.SfpuAddTopRow: "test_eltwise_binary_sfpu_add_top_row builds its own stimuli, "
    "golden and TestConfig (the top-row semantics need a [64, 32] single-pair layout)",
    MathOperation.SfpuQuant: "test_eltwise_binary_sfpu_quant stages fp32 bits in an Int32 tile "
    "and passes a zero-point the shared float driver does not have",
    MathOperation.SfpuRequant: "as SfpuQuant",
    MathOperation.SfpuDequant: "as SfpuQuant; the output tile is Float32",
    MathOperation.SfpuQuantUint8: "test_eltwise_binary_sfpu_quant_byte, as SfpuQuant",
    MathOperation.SfpuQuantInt8: "as SfpuQuantUint8",
    MathOperation.SfpuRequantUint8: "as SfpuQuantUint8",
    MathOperation.SfpuRequantInt8: "as SfpuQuantUint8",
    MathOperation.SfpuRequantInt8In: "as SfpuQuantUint8",
    MathOperation.SfpuRequantInt8InUint8Out: "as SfpuQuantUint8",
    MathOperation.SfpuRequantInt8InInt8Out: "as SfpuQuantUint8",
    MathOperation.SfpuDequantInt8: "as SfpuQuantUint8; the output tile is Float32",
    MathOperation.SfpuFloatToInt32Positive: "test_eltwise_binary_sfpu_float_to_int32_positive "
    "stages raw fp32 words in an Int32 tile and checks the int32 result bit for bit",
}


def _assert_domain_sets_consistent():
    """The three stimulus-source sets must be disjoint and consistent with the registry.

    Every way of getting this wrong fails quietly otherwise -- an op keeps the positive-only
    default while looking, to a reader, like it was considered. Totality over the ops this
    suite actually drives cannot be checked until pytest has collected, so
    _classify_stimuli_source() enforces that half in the driver; what is checked here is
    family totality against sfpu_domains._SFPU_BINARY_OPS.
    """
    missing = sorted(
        op.name
        for op in _REGISTRY_DOMAIN_OPS | _REGISTERED_DEFAULT_STIMULI_OPS.keys()
        if op not in _OP_DOMAIN_REGISTRY
    )
    assert not missing, (
        "these ops are declared as registered but have no entry in "
        f"sfpu_domains._OP_DOMAIN_REGISTRY: {missing}"
    )
    now_registered = sorted(
        op.name for op in _UNREGISTERED_BINARY_OPS if op in _OP_DOMAIN_REGISTRY
    )
    assert not now_registered, (
        "these ops now have a domain in _OP_DOMAIN_REGISTRY but are still on the "
        "positive-only fallback list; move them to _REGISTRY_DOMAIN_OPS (float ops) or "
        f"to _REGISTERED_DEFAULT_STIMULI_OPS with a reason: {now_registered}"
    )
    unregistered = sorted(
        op.name
        for op in _REGISTERED_DEFAULT_STIMULI_OPS
        if op not in _OP_DOMAIN_REGISTRY
    )
    assert not unregistered, (
        "these ops are listed as registered-but-not-read, yet have no entry in "
        f"_OP_DOMAIN_REGISTRY; move them to _UNREGISTERED_BINARY_OPS: {unregistered}"
    )
    sets = {
        "_REGISTRY_DOMAIN_OPS": set(_REGISTRY_DOMAIN_OPS),
        "_UNREGISTERED_BINARY_OPS": set(_UNREGISTERED_BINARY_OPS),
        "_REGISTERED_DEFAULT_STIMULI_OPS": set(_REGISTERED_DEFAULT_STIMULI_OPS),
    }
    for a, b in itertools.combinations(sorted(sets), 2):
        overlap = sorted(op.name for op in sets[a] & sets[b])
        assert not overlap, f"{a} and {b} both claim: {overlap}"
    undeclared = sorted(
        op.name
        for op in _SFPU_BINARY_OPS
        - set().union(*sets.values())
        - _OPS_NOT_USING_SHARED_DRIVER.keys()
    )
    assert not undeclared, (
        "these ops are in sfpu_domains._SFPU_BINARY_OPS but are in none of this suite's "
        "three stimulus-source sets, so nothing states what they are fed; classify them "
        f"or declare them in _OPS_NOT_USING_SHARED_DRIVER: {undeclared}"
    )


_assert_domain_sets_consistent()

_CLASSIFIED_STIMULI_OPS = (
    set(_REGISTRY_DOMAIN_OPS)
    | set(_UNREGISTERED_BINARY_OPS)
    | set(_REGISTERED_DEFAULT_STIMULI_OPS)
)


def _classify_stimuli_source(mathop):
    """True if *mathop* reads its domain from the registry; False if it keeps the default.

    Raises for an op in none of the three declared sets -- the case the collection-time
    assertion cannot see, since adding an op to a parametrize list is enough to drive it and
    an unclassified op would silently inherit the positive-only format default.
    """
    if mathop not in _CLASSIFIED_STIMULI_OPS:
        raise KeyError(
            f"MathOperation.{mathop.name} is driven through sfpu_binary() but is in none "
            "of _REGISTRY_DOMAIN_OPS / _UNREGISTERED_BINARY_OPS / "
            "_REGISTERED_DEFAULT_STIMULI_OPS. Add it to whichever describes what it "
            "should be fed; the last two want a recorded reason."
        )
    return mathop in _REGISTRY_DOMAIN_OPS


def _mask_stimuli_specs():
    # mask forces data (in0) to the op's fill value (0 for MASK, +inf for MASK_POSINF) where
    # mask (in1) is 0. Data and mask are separate tiles: keep data strictly non-zero (1..8) and
    # zero ~1/3 of the mask, so a passthrough kernel fails.
    def data_face(size, dtype, generator):
        _, ramp = _positions_and_ramp(size)
        return ramp.to(dtype)  # 1..8, always non-zero

    def mask_face(size, dtype, generator):
        j, _ = _positions_and_ramp(size)
        return torch.where(j % 3 == 0, 0.0, 1.0).to(dtype)  # ~1/3 exact zeros

    return _face_spec(data_face), _face_spec(mask_face)


def _isclose_stimuli_specs():
    # isclose is a predicate on paired operands (a = tile0, b = tile1). Fill the two tiles
    # from different data so even p -> identical (isclose 1), odd p -> differ by 2.0
    # (isclose 0); the 2.0 gap dwarfs the tolerance so the decision is unambiguous.
    def a_face(size, dtype, generator):
        _, ramp = _positions_and_ramp(size)
        return ramp.to(dtype)  # 1..8, strictly positive

    def b_face(size, dtype, generator):
        j, ramp = _positions_and_ramp(size)
        return (ramp + torch.where(j % 2 == 0, 0.0, 2.0)).to(dtype)

    return _face_spec(a_face), _face_spec(b_face)


def _isclose_nan_stimuli_specs():
    # The NaN arm of isclose, which _isclose_stimuli_specs (finite ramps only) never reaches.
    # Eight lanes per period, so every pairing the EQUAL_NAN template flag changes -- and every
    # one it must not -- appears in every face:
    #   p%8: 0 (+NaN,+NaN) 1 (+NaN,x) 2 (x,+NaN) 3 (-NaN,+NaN) 4 (-NaN,-NaN)
    #        5 (x,x) 6 (x,x+2) 7 (+NaN,-NaN)
    # equal_nan=False answers 0 on lanes 0-4 and 7, equal_nan=True answers 1 on 0, 3, 4 and 7
    # (both NaN, whatever the signs) and 0 on the one-sided lanes 1 and 2; 5 and 6 are the
    # finite 1/0 controls. The goldens are torch.isclose(equal_nan=False/True). The NaNs are
    # built from explicit fp32 bit patterns (0x7FC00000 / 0xFFC00000); torch's bfloat16 cast
    # turns every NaN into one pattern, so the mixed-sign lanes are distinct at Float32 only.
    nan = torch.tensor([0x7FC00000], dtype=torch.int32).view(torch.float32)
    neg_nan = torch.tensor([0xFFC00000 - (1 << 32)], dtype=torch.int32).view(
        torch.float32
    )

    def a_face(size, dtype, generator):
        j, ramp = _positions_and_ramp(size)
        k = j % 8
        a = ramp.clone()
        a = torch.where((k == 0) | (k == 1) | (k == 7), nan, a)
        a = torch.where((k == 3) | (k == 4), neg_nan, a)
        return a.to(dtype)

    def b_face(size, dtype, generator):
        j, ramp = _positions_and_ramp(size)
        k = j % 8
        b = ramp + torch.where(k == 6, 2.0, 0.0)
        b = torch.where((k == 0) | (k == 2) | (k == 3), nan, b)
        b = torch.where((k == 4) | (k == 7), neg_nan, b)
        return b.to(dtype)

    return _face_spec(a_face), _face_spec(b_face)


def _xlogy_denormal_stimuli_specs():
    # The log operand (in1) on the biased-exponent-0 lane, which the kernel maps to ln = -inf:
    # +0, the smallest and the largest positive denormal, then FLT_MIN and 1.0 as finite
    # controls, then the smallest and the largest negative denormal. in0 is the positive ramp,
    # so the golden is -inf on lanes 0-2, in0 * ln(FLT_MIN) on lane 3, 0 on lane 4 and NaN on
    # lanes 5-6. Lanes 1-2 pin the exexp(Biased) == 0 test of _calculate_log_body_on_reg_: an
    # `in1 == 0.0F` test would return a finite in0 * ~-88 there (see its docstring), and
    # _xlogy's golden flushes a positive denormal in1 to match the kernel. Lanes 5-6 pin the
    # NaN guard: `in1 < 0.0f` reads the sign bit of the unflushed register, so a negative
    # denormal is NaN, as log of a negative is, and the golden does not flush it. Only a
    # 32-bit input into a 32-bit Dest carries a denormal to the SFPU; on the datacopy path a
    # bf16 denormal is flushed before SFPLOAD.
    lanes = torch.tensor(
        [
            0x00000000,
            0x00000001,
            0x007FFFFF,
            0x00800000,
            0x3F800000,
            -0x7FFFFFFF,  # 0x80000001
            -0x7F800001,  # 0x807FFFFF
        ],
        dtype=torch.int32,
    ).view(torch.float32)

    def a_face(size, dtype, generator):
        _, ramp = _positions_and_ramp(size)
        return ramp.to(dtype)

    def b_face(size, dtype, generator):
        j, _ = _positions_and_ramp(size)
        return lanes[(j % len(lanes)).long()].to(dtype)

    return _face_spec(a_face), _face_spec(b_face)


def _eq_ne_stimuli_specs():
    # Eq/Ne compare paired operands (a = tile0, b = tile1). Fill the two tiles so even p ->
    # identical (Eq 1), odd p -> differ by 1.0 (Eq 0), a clean ~50/50 mix.
    def a_face(size, dtype, generator):
        _, ramp = _positions_and_ramp(size)
        return ramp.to(dtype)  # 1..8

    def b_face(size, dtype, generator):
        j, ramp = _positions_and_ramp(size)
        return (ramp + torch.where(j % 2 == 0, 0.0, 1.0)).to(dtype)

    return _face_spec(a_face), _face_spec(b_face)


def _comparison_stimuli_specs():
    """Three-way paired stimuli for lt/gt/le/ge: a < b, a == b, a > b in equal thirds.

    The exact-equality third is the point: it is the only input where lt/gt and le/ge
    disagree, and a random sweep essentially never produces it. The +/-1.0 gaps are far
    wider than any rounding, so every other element's verdict is unambiguous too.
    """

    def a_face(size, dtype, generator):
        _, ramp = _positions_and_ramp(size)
        return ramp.to(dtype)  # 1..8

    def b_face(size, dtype, generator):
        j, ramp = _positions_and_ramp(size)
        # j % 3 == 0 -> equal, 1 -> b greater (a < b), 2 -> b smaller (a > b)
        delta = torch.where(j % 3 == 0, 0.0, torch.where(j % 3 == 1, 1.0, -1.0))
        return (ramp + delta).to(dtype)

    return _face_spec(a_face), _face_spec(b_face)


def _logsigmoid_stimuli_spec():
    # logsigmoid(x) = -softplus(-x). in1 is only read in the x > 4 branch, so x is restricted
    # to [-8, 3.9], which sweeps the passthrough and polynomial branches without it.
    def dist(size, dtype, generator):
        return torch.linspace(-8.0, 3.9, size).to(dtype)

    return StimuliSpec(distribution=dist, seed=0)


# =============================================================================
# Shared driver
#
# Every test below (except add_top_row, the quant family, and the separate
# broadcast kernel) runs through sfpu_binary(): it builds the two-tile stimuli,
# computes the golden per tile-pair, and drives sources/sfpu_binary_test.cpp.
# Perf counterparts call the same helper with is_perf=True.
# =============================================================================


def _override_src_A(src_A, src_A_override, *, repeat):
    """Replace the generated operand buffer with an explicit one, cast to its dtype.

    With ``repeat`` the override is tiled across the buffer (its length must evenly divide
    the buffer length); without it the override must be the whole buffer.
    """
    override = src_A_override.to(src_A.dtype).flatten()
    if repeat:
        if src_A.numel() % override.numel() != 0:
            raise ValueError(
                "SFPU binary override length must evenly divide the buffer length "
                f"({src_A.numel()} elements), got {override.numel()}"
            )
        return override.repeat(src_A.numel() // override.numel())
    if override.numel() != src_A.numel():
        raise ValueError(
            f"SFPU binary override must hold the whole buffer ({src_A.numel()} "
            f"elements), got {override.numel()}"
        )
    return override


def sfpu_binary(
    formats,
    dest_acc,
    mathop,
    broadcast_type=None,
    src_A_override=None,
    spec_A=None,
    spec_B=None,
    twos_complement=False,
    input_dimensions=None,
    unspecified_nonfinite_sign=False,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
    iterations=32,
    approx_mode=ApproximationMode.No,
    dst_rounding_mode=DstRoundingMode.Default,
    check_contract=True,
):
    """*unspecified_nonfinite_sign* compares a non-finite result by magnitude only.

    For the one case where the sign genuinely is not specified: a NaN the kernel generated,
    packed as a signed infinity through a pipeline too narrow to hold it, on Wormhole. The
    magnitude, the finiteness and every finite lane stay checked. Scoped per lane, because one
    tensor can hold both a non-finite whose sign the ISA leaves open and one it specifies.
    """

    # Seed the draw so the stimuli are identical run to run; an unseeded redraw makes a
    # variant near its tolerance fail unreproducibly.
    torch.manual_seed(0)

    # Default tensor size. 32-bit formats use [128, 128]. 16-bit formats use
    # [256, 128]. dest_acc=No fills that tensor as four blocks of eight tiles.
    # dest_acc=Yes holds four tiles, so the same tensor is eight blocks of four.
    if input_dimensions is None:
        input_dimensions = (
            [128, 128] if formats.input_format.is_32_bit() else [256, 128]
        )

    # Different L1 formats cannot share buffer_A: Operand strides one tile size.
    # Each operand gets its own buffer, and the golden is interleaved afterwards.
    # Only Float16_b x Bfp8_b is modeled: a 32-bit operand would set unpack_to_dest
    # and hang on the dest handshake.
    mixed_formats = formats.input_format != formats.input_format_B
    if mixed_formats:
        if (formats.input_format, formats.input_format_B) != (
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ):
            raise ValueError("mixed input formats only support Float16_b x Bfp8_b")
        if broadcast_type not in (None, LlkBroadcastType.None_):
            raise ValueError("mixed input formats do not support broadcast")
        if src_A_override is not None or spec_A is not None or spec_B is not None:
            raise ValueError(
                "mixed input formats take neither an override nor a domain spec"
            )
        src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=input_dimensions,
            stimuli_format_B=formats.input_format_B,
            input_dimensions_B=input_dimensions,
        )
        elements_per_tile = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM
        golden_a = quantize_input_to_unpack_format(
            src_A.flatten(), formats.input_format
        )
        golden_b = quantize_input_to_unpack_format(
            src_B.flatten(), formats.input_format_B
        )
        golden_src = torch.stack(
            (
                golden_a.flatten().view(tile_cnt_A, elements_per_tile),
                golden_b.flatten().view(tile_cnt_B, elements_per_tile),
            ),
            dim=1,
        ).reshape(-1)
        stimuli_b_format = formats.input_format_B
        result_tile_cnt = tile_cnt_A + tile_cnt_B
        block_dimensions = [
            DEFAULT_TILE_R_DIM,
            DEFAULT_TILE_C_DIM * result_tile_cnt,
        ]
    else:
        # Per-operand domains. Both operands live in buffer_A (even tile = in0, odd tile = in1),
        # so there is no spec_B knob in generate_stimuli -- the two specs are interleaved into one
        # spec's face_specs here, where the tile count is known. Unlike the unary sweep, most ops
        # here are not registered, and a missing entry falls back to the format default.
        reads_registry = _classify_stimuli_source(mathop)
        if spec_A is None and reads_registry and not formats.input_format.is_integer():
            specs = exclude_undefined_pair(mathop, for_op(mathop, formats.input_format))
            spec_A, spec_B = specs.spec_A, specs.spec_B

        if spec_B is not None:
            if spec_A is None:
                raise ValueError(
                    "spec_B requires spec_A (it fills the odd tile of each pair)"
                )
            spec_A = _pair_operand_specs(spec_A, spec_B, input_dimensions)

        src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=input_dimensions,
            stimuli_format_B=formats.input_format,
            input_dimensions_B=input_dimensions,
            spec_A=spec_A,
        )

        # The kernel only consumes buffer_A (operand 0 = tile 0, operand 1 = tile 1), so an
        # explicit src_A fully controls inputs for edge cases; src_B stays random but unused.
        if src_A_override is not None:
            src_A = _override_src_A(src_A, src_A_override, repeat=True)

        # generate_stimuli does not round-trip Bfp8_b stimuli through their pack/unpack
        # quantization, so a registry domain or an override would leave the device seeing
        # quantized operands while the golden sees the unrounded originals. Quantize the golden's
        # copy only: src_A keeps the unrounded values, since the packer rounds on the way to L1.
        golden_src = src_A
        if formats.input_format == DataFormat.Bfp8_b:
            golden_src = quantize_input_to_unpack_format(golden_src, DataFormat.Bfp8_b)
        stimuli_b_format = formats.input_format
        result_tile_cnt = tile_cnt_A
        block_dimensions = input_dimensions

    if broadcast_type is not None and broadcast_type != LlkBroadcastType.None_:
        generate_broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_src = generate_broadcast_golden(
            broadcast_type,
            golden_src,
            (
                formats.input_format
                if formats.input_format != DataFormat.Bfp8_b
                else DataFormat.Float16_b
            ),
            tile_cnt=tile_cnt_A,
        )

    # The kernel runs with a 32-bit Dest where the hardware needs one -- an exponent-B
    # input packed to Float16 (`effective_dest_acc`, the rule TestConfig applies
    # silently), a Float16/Float32 input on Blackhole -- so dest_acc
    # is promoted here too. Hoisted above the golden, which models the Dest width from
    # it, and above the contract, whose `dest: "Yes"` rows describe the kernel that ran.
    # Left at No, both would describe a variant that never executed.
    dest_acc = effective_dest_acc(
        formats.input_format, formats.output_format, dest_acc, TestConfig.CHIP_ARCH
    )
    if (
        formats.input_format in [DataFormat.Float16, DataFormat.Float32]
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        dest_acc = DestAccumulation.Yes

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_format = (
        DataFormat.Float16_b
        if formats.input_format == DataFormat.Bfp8_b
        or formats.input_format_B == DataFormat.Bfp8_b
        else formats.input_format
    )
    elements_per_pair = 2 * _ELEMENTS_PER_TILE
    golden_chunks = []
    generated_nan_chunks = []
    # WH/BH copy_dest_value copies src1 onto the even dest tile. The shared
    # _copy_dest golden still returns its first argument, which Quasar uses for
    # a src0 copy, so only this call swaps the pair.
    src1_idx, src2_idx = (1, 0) if mathop == MathOperation.SfpuCopyDest else (0, 1)
    for offset in range(0, golden_src.numel(), elements_per_pair):
        chunk = generate_golden(
            mathop,
            golden_src[offset : offset + elements_per_pair],
            src1_idx,
            src2_idx,
            0,
            32,
            [64, 32],
            golden_format,
            dest_acc=dest_acc,
            output_format=formats.output_format,
            collect_generated_nan=unspecified_nonfinite_sign,
        )
        # Asked of the return value rather than of the build mode, since the golden is stubbed
        # out during --compile-producer.
        if isinstance(chunk, tuple):
            chunk, generated_nan = chunk
            generated_nan_chunks.append(generated_nan)
        golden_chunks.append(chunk.flatten())
    golden_tensor = torch.cat(golden_chunks)

    bcast = broadcast_type if broadcast_type else LlkBroadcastType.None_

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, block_dimensions, TILE_DIMENSIONS
    )

    run_types = _resolve_perf_args(is_perf, perf_report, run_types)

    # Int32 ADD/SUB are sign-magnitude end to end. Other ops here ignore the flag.
    sign_magnitude = formats.input_format == DataFormat.Int32 and mathop in (
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
    )

    test_config_kwargs = {
        "test_name": "sources/sfpu_binary_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(block_dimensions, block_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
            BROADCAST_TYPE(bcast),
            SFPU_BCAST_DIM(LlkBroadcastType.None_),
            SFPU_DST_ROUNDING_MODE(dst_rounding_mode),
            SIGN_MAGNITUDE_FORMAT(sign_magnitude),
        ],
        "runtimes": [
            TILE_COUNT(result_tile_cnt),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(),
            ZERO_POINT(0),
            OUT_ZERO_POINT(0),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            stimuli_b_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=result_tile_cnt,
            twos_complement=twos_complement,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": formats.input_format.is_32_bit(),
        "compile_time_formats": True,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    # The contract tolerance cannot tell truncation from round-to-nearest-even.
    # Callers that pass check_contract=False compare bit-exactly themselves.
    if not check_contract:
        return res_tensor, golden_tensor

    if unspecified_nonfinite_sign and generated_nan_chunks:
        # Clear the sign only on lanes that held a generated NaN and where both sides are
        # non-finite, so this excuses one bit on the lanes the ISA declines to pin and nothing
        # else: a golden +inf against a hardware 5.0, or against a NaN, still fails.
        unspecified = (
            torch.cat(generated_nan_chunks)
            & ~torch.isfinite(golden_tensor)
            & ~torch.isfinite(res_tensor)
        )
        golden_tensor = torch.where(unspecified, golden_tensor.abs(), golden_tensor)
        res_tensor = torch.where(unspecified, res_tensor.abs(), res_tensor)

    assert_against_contract(
        mathop,
        formats,
        dest_acc,
        golden_tensor,
        res_tensor,
        approx_mode=_APPROX_MODE,
    )


# =============================================================================
# Float ops
# =============================================================================

# The bf16/fp32 matrix most of the sweeps below share.
_BF16_FP32_FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32])

# Shared with perf_eltwise_binary_sfpu.py so the two sweeps stay aligned.
FLOAT_SWEEP = dict(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    bcast_dim=[
        LlkBroadcastType.None_,
        LlkBroadcastType.Row,
        LlkBroadcastType.Column,
        LlkBroadcastType.Scalar,
    ],
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
        MathOperation.SfpuElwrsub,
        MathOperation.SfpuElwpow,
        MathOperation.SfpuXlogy,
        MathOperation.SfpuLogaddexp,
        MathOperation.SfpuLogaddexp2,
        MathOperation.SfpuCopyDest,
        # Eq/Ne and Lt/Gt/Le/Ge are excluded from this *random* sweep: independent draws
        # are never equal (the Eq/Ne golden collapses to a constant) and near-ties that
        # the kernel and the total-order golden round differently read as failures. They
        # are covered with crafted paired stimuli by test_eltwise_binary_sfpu_eq_ne and
        # test_eltwise_binary_sfpu_float_comparison below.
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
DIV_SWEEP = dict(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16_b,
        ]
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
FLOAT_EXTENDED_SWEEP = dict(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    mathop=[
        MathOperation.SfpuBinaryMax,
        MathOperation.SfpuBinaryMin,
        MathOperation.SfpuBinaryFmod,
        MathOperation.SfpuBinaryRemainder,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
MASK_SWEEP = dict(
    formats=_BF16_FP32_FORMATS,
    mathop=[MathOperation.SfpuMask, MathOperation.SfpuMaskPosinf],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
ATAN2_SWEEP = dict(
    formats=_BF16_FP32_FORMATS,
    mathop=[MathOperation.SfpuAtan2],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
EQ_NE_SWEEP = dict(
    formats=_BF16_FP32_FORMATS,
    mathop=[MathOperation.SfpuElwEq, MathOperation.SfpuElwNe],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
FLOAT_COMPARISON_SWEEP = dict(
    formats=_BF16_FP32_FORMATS,
    mathop=[
        MathOperation.SfpuElwLt,
        MathOperation.SfpuElwGt,
        MathOperation.SfpuElwLe,
        MathOperation.SfpuElwGe,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
ISCLOSE_SWEEP = dict(
    formats=_BF16_FP32_FORMATS,
    mathop=[MathOperation.SfpuIsclose, MathOperation.SfpuIscloseEqualNan],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
LOGSIGMOID_SWEEP = dict(
    formats=_BF16_FP32_FORMATS,
    mathop=[MathOperation.SfpuLogsigmoid],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)


@parametrize(**FLOAT_SWEEP)
def test_eltwise_binary_sfpu_float(
    formats,
    dest_acc,
    mathop,
    bcast_dim,
    **run_kwargs,
):
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    # copy_dest_value copies src1 onto src0 and does not broadcast. A broadcast
    # row would compare against the broadcast golden, which this kernel does not
    # compute.
    if mathop == MathOperation.SfpuCopyDest and bcast_dim != LlkBroadcastType.None_:
        pytest.skip("COPY_DEST does not broadcast; the None_ row covers the kernel")

    # Bfp8_b quantization can map small positive operands to zero, making xlogy's
    # logarithm -inf. LOGADDEXP and LOGADDEXP2 are skipped here too: their +/-200
    # domain under Bfp8_b's shared-exponent quantization collapses most of the
    # |a - b| < 20 correction band this sweep exists to exercise. Perf compares
    # no golden, so those rows still run.
    if (
        not run_kwargs.get("is_perf")
        and formats.input_format == DataFormat.Bfp8_b
        and mathop
        in (
            MathOperation.SfpuXlogy,
            MathOperation.SfpuLogaddexp,
            MathOperation.SfpuLogaddexp2,
        )
    ):
        pytest.skip(
            "Bfp8_b input is not supported for XLOGY/LOGADDEXP/LOGADDEXP2 coverage"
        )

    if (
        bcast_dim == LlkBroadcastType.Row
        and effective_dest_acc(formats.input_format, formats.output_format, dest_acc)
        == DestAccumulation.Yes
    ):
        pytest.skip(
            "Row broadcast with FP32 dest: B2D datacopy uses MOVB2D which can't handle FP32 dest format conversion"
        )

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        broadcast_type=bcast_dim,
        **run_kwargs,
    )


# One directed point: Float16_b x Bfp8_b -> Float16_b. Quasar has no block-float.
# Shared with perf_eltwise_binary_sfpu.py so the two sweeps stay aligned.
MIXED_FORMAT_SWEEP = dict(
    formats=[
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b, DataFormat.Bfp8_b)
    ],
    mathop=[MathOperation.SfpuElwadd],
    dest_acc=[DestAccumulation.No],
)


@parametrize(**MIXED_FORMAT_SWEEP)
def test_eltwise_binary_sfpu_mixed_format(formats, mathop, dest_acc, **run_kwargs):
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        input_dimensions=[32, 32],
        **run_kwargs,
    )


def _tile_from_pairs(pairs):
    """Row-major operand tiles from cycled (a, b) pairs. One tile each."""
    src = _build_paired_tile_override(pairs, torch.float32)
    a = src[:_ELEMENTS_PER_TILE].to(torch.bfloat16)
    b = src[_ELEMENTS_PER_TILE:].to(torch.bfloat16)
    return src, a, b


# One in0/in1 pair. The golden's first tile is the result in row-major order.
_EXACT_DIMS = [64, 32]
_BF16_FORMATS = InputOutputFormat(
    input_format=DataFormat.Float16_b, output_format=DataFormat.Float16_b
)
# Sums whose low 16 bits are a tie and whose kept mantissa bit is odd, so
# round-to-nearest-even steps up and truncation does not.
_RNE_ADD_PAIRS = (
    (0x3F00, 0x3F03),
    (0x3F00, 0x3F0F),
    (0x3F00, 0x3F1B),
    (0x3F00, 0x3F27),
)
# Finite products that separate RNE from truncation, then 0 * +inf, stored as +0.
_PRODUCTION_MUL_PAIRS = (
    (0x0000, 0x7F80),
    (0x3F01, 0x3F42),
    (0x3F01, 0x3F45),
    (0x3F01, 0x3F48),
)


def test_eltwise_binary_sfpu_bf16_rne_exact():
    """NearestEven ADD on Float16_b with a 16-bit Dest, bit-exact against RNE.

    The contract tolerance also passes a truncated store, so this compares
    the device result to the golden with torch.equal.
    """
    src, a, b = _tile_from_pairs(
        [(bf16_from_bits(x), bf16_from_bits(y)) for x, y in _RNE_ADD_PAIRS]
    )
    assert_bf16_separates_rne(a.float() + b.float())
    res, golden = sfpu_binary(
        _BF16_FORMATS,
        DestAccumulation.No,
        MathOperation.SfpuElwadd,
        src_A_override=src,
        input_dimensions=_EXACT_DIMS,
        dst_rounding_mode=DstRoundingMode.NearestEven,
        check_contract=False,
    )
    assert_bit_exact(res, golden, "NearestEven ADD")


def test_eltwise_binary_sfpu_mul_production_exact():
    """Production mul is RNE, and 0 * inf stores 0 rather than a NaN."""
    src, a, b = _tile_from_pairs(
        [(bf16_from_bits(x), bf16_from_bits(y)) for x, y in _PRODUCTION_MUL_PAIRS]
    )
    finite = torch.isfinite(a.float() * b.float())
    assert_bf16_separates_rne((a.float() * b.float())[finite])
    res, golden = sfpu_binary(
        _BF16_FORMATS,
        DestAccumulation.No,
        MathOperation.SfpuElwmul,
        src_A_override=src,
        input_dimensions=_EXACT_DIMS,
        check_contract=False,
    )
    zero = (a == 0) | (b == 0)
    if not bool(zero.any()):
        raise AssertionError("production mul stimulus has no zero input")
    # IEEE 0 * inf is not +0, so a golden that keeps it does not check the store.
    got_zero = golden[:_ELEMENTS_PER_TILE][zero]
    if not torch.equal(got_zero, torch.zeros_like(got_zero)):
        raise AssertionError("golden did not model production 0 * x = +0")
    assert_bit_exact(res, golden, "production mul")


@parametrize(**DIV_SWEEP)
def test_eltwise_binary_sfpu_div(formats, dest_acc, **run_kwargs):
    # DIV routes through the dedicated production kernel (calculate_sfpu_binary_div);
    # split out from the float sweep since the reciprocal path is precision-sensitive.
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    sfpu_binary(
        formats,
        dest_acc,
        MathOperation.SfpuElwdiv,
        broadcast_type=LlkBroadcastType.None_,
        **run_kwargs,
    )


@parametrize(**FLOAT_EXTENDED_SWEEP)
def test_eltwise_binary_sfpu_float_extended(formats, dest_acc, mathop, **run_kwargs):
    # max/min (SFPSWAP) and fmod/remainder (fp32 reciprocal) binary kernels with no
    # dedicated production BinaryOp; driven through the same in-DST harness as add/sub.
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    # fmod/remainder divide by b via a reciprocal, and Bfp8_b's coarse quantization blows up
    # the quotient for small divisors. It only shows when the output is Bfp8_b as well: a
    # Bfp8_b input into any of the float outputs passes, so the guard is scoped to
    # Bfp8_b -> Bfp8_b rather than to every Bfp8_b input.
    if (
        formats.input_format == DataFormat.Bfp8_b
        and formats.output_format == DataFormat.Bfp8_b
        and mathop
        in (
            MathOperation.SfpuBinaryFmod,
            MathOperation.SfpuBinaryRemainder,
        )
    ):
        pytest.skip("Bfp8_b -> Bfp8_b is not supported for fmod/remainder coverage")

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        broadcast_type=LlkBroadcastType.None_,
        **run_kwargs,
    )


@parametrize(**MASK_SWEEP)
def test_eltwise_binary_sfpu_mask(formats, dest_acc, mathop, **run_kwargs):
    # float mask: data at tile0, mask at tile1. Output is data where mask != 0, else 0
    # (MASK) or +inf (MASK_POSINF).
    # Crafted stimuli so the mask carries real zeros.
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    # One tile pair only, unlike every other op here: calculate_mask hard-codes its operands
    # and ignores the forwarded dst indices, so only the first placement computes anything.
    # [64, 32] is one pair, which is the only one the adapter supports.
    spec_A, spec_B = _mask_stimuli_specs()
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        broadcast_type=LlkBroadcastType.None_,
        spec_A=spec_A,
        spec_B=spec_B,
        input_dimensions=[64, 32],
        **run_kwargs,
    )


@parametrize(**ATAN2_SWEEP)
def test_eltwise_binary_sfpu_atan2(formats, dest_acc, mathop, **run_kwargs):
    # atan2(y, x): y = tile0, x = tile1. Signed [-5, 5] gives mixed signs so all quadrants
    # (and the |y|>=|x| / x<0 branches) are exercised. A minimax approximation: gated by
    # its step budget on Wormhole and Blackhole, and by tolerance + PCC on Quasar.
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0),
        **run_kwargs,
    )


@parametrize(**EQ_NE_SWEEP)
def test_eltwise_binary_sfpu_eq_ne(formats, dest_acc, mathop, **run_kwargs):
    # Eq/Ne(a, b) with a = tile0, b = tile1. Crafted paired stimuli give a non-constant 0/1
    # golden so the equal branch is exercised (the default random sweep never is).
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    spec_A, spec_B = _eq_ne_stimuli_specs()
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B, **run_kwargs)


@parametrize(**FLOAT_COMPARISON_SWEEP)
def test_eltwise_binary_sfpu_float_comparison(formats, dest_acc, mathop, **run_kwargs):
    # lt/gt/le/ge(a, b) with a = tile0, b = tile1. Crafted so a third of the elements are
    # exactly equal and the rest differ by +/-1.0: the tie is what distinguishes the strict
    # comparisons from the non-strict ones, and the wide gaps keep every other element's
    # verdict independent of rounding. See _comparison_stimuli_specs.
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    spec_A, spec_B = _comparison_stimuli_specs()
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B, **run_kwargs)


@parametrize(**ISCLOSE_SWEEP)
def test_eltwise_binary_sfpu_isclose(formats, dest_acc, mathop, **run_kwargs):
    # isclose(a, b) = |a - b| <= atol + rtol*|b|, a = tile0, b = tile1. torch default
    # tolerances (fixed in the C++ dispatch); crafted stimuli give a non-constant 0/1 mix.
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    spec_A, spec_B = _isclose_stimuli_specs()
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B, **run_kwargs)


@parametrize(**ISCLOSE_SWEEP)
def test_eltwise_binary_sfpu_isclose_nan(formats, dest_acc, mathop):
    # Both-NaN, one-sided-NaN and mixed-sign-NaN pairs, which is where EQUAL_NAN changes the
    # answer; see _isclose_nan_stimuli_specs for the lane table. Only on pipelines that deliver
    # a NaN operand to the SFPU intact.
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)
    if not specials_safe(formats.input_format, formats.output_format, dest_acc):
        pytest.skip("this pipeline does not deliver a NaN operand to the SFPU intact")

    spec_A, spec_B = _isclose_nan_stimuli_specs()
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B)


# dest_acc stays an axis: a single-axis `parametrize` hands pytest 1-tuples, which `--op` cannot
# read, so this is the same shape as test_eltwise_binary_sfpu_int_uniform.
@parametrize(mathop=[MathOperation.SfpuXlogy], dest_acc=[DestAccumulation.Yes])
def test_eltwise_binary_sfpu_xlogy_denormal(mathop, dest_acc):
    # xlogy(x, denormal) = x * ln(denormal) = -inf, because the kernel flushes its log operand
    # (see _xlogy_denormal_stimuli_specs). Float32 into a 32-bit Dest is the only pipeline in
    # this harness that delivers a denormal to the SFPU, and the flush is measured on Blackhole.
    if TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "the denormal flush of the xlogy log operand is measured on Blackhole only"
        )

    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    spec_A, spec_B = _xlogy_denormal_stimuli_specs()
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B)


@parametrize(**LOGSIGMOID_SWEEP)
def test_eltwise_binary_sfpu_logsigmoid(formats, dest_acc, mathop, **run_kwargs):
    # logsigmoid(x) with x = tile0. Piecewise poly/passthrough approximation, gated by its
    # step budget on Wormhole and Blackhole and by tolerance + PCC on Quasar; x swept
    # over [-8, 3.9].
    # The x > 4 (-exp(-x)) branch needs a device-computed
    # exp(-x) operand the shared harness can't provide, left to a future driver.
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        spec_A=_logsigmoid_stimuli_spec(),
        **run_kwargs,
    )


# =============================================================================
# Quant family. One tile pair: operand A in tile 0, the fp32 scale in tile 1.
# Quant and requant run in 2's complement only; dequant also runs sign-magnitude.
# =============================================================================

_QUANT_INT8_MIN = -128
_QUANT_INT8_MAX = 127
_QUANT_UINT8_MAX = 255
_QUANT_INT8_BIAS = 0x80  # the offset-128 pack's bias; also the int8 byte of -128
_QUANT_OPS = [
    MathOperation.SfpuQuant,
    MathOperation.SfpuRequant,
    MathOperation.SfpuDequant,
]
QUANT_SWEEP = dict(
    mathop=_QUANT_OPS,
    sign_magnitude=lambda mathop: (
        [False, True] if mathop == MathOperation.SfpuDequant else [False]
    ),
)


def _run_sfpu_binary_word_pairs(
    mathop,
    in0_words,
    in1_words,
    zero_point_bits,
    output_format,
    *,
    sign_magnitude=False,
    out_zero_point_bits=0,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
    iterations=32,
    approx_mode=ApproximationMode.No,
):
    """Run *mathop* on (in0, in1) tile pairs of raw 32-bit words; return the in0-tile result words.

    The result overwrites the in0 tile; the in1 tiles are packed back untouched and checked to round-trip.
    """
    pairs = in0_words.size // _ELEMENTS_PER_TILE
    assert pairs * _ELEMENTS_PER_TILE == in0_words.size == in1_words.size
    tile_cnt_A = 2 * pairs
    input_dimensions = (
        [32 * tile_cnt_A // 4, 128] if tile_cnt_A % 4 == 0 else [32 * tile_cnt_A, 32]
    )
    staged = np.empty((pairs, 2, _ELEMENTS_PER_TILE), dtype=np.int32)
    staged[:, 0, :] = in0_words.reshape(pairs, _ELEMENTS_PER_TILE)
    staged[:, 1, :] = in1_words.reshape(pairs, _ELEMENTS_PER_TILE)
    src_A = torch.from_numpy(staged.reshape(-1).copy())

    formats = InputOutputFormat(DataFormat.Int32, output_format)
    dest_acc = DestAccumulation.Yes
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS
    )
    run_types = _resolve_perf_args(is_perf, perf_report, run_types)
    test_config_kwargs = {
        "test_name": "sources/sfpu_binary_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
            BROADCAST_TYPE(LlkBroadcastType.None_),
            SFPU_BCAST_DIM(LlkBroadcastType.None_),
            # The shared math driver reads this for every op. These ops do not use it.
            SFPU_DST_ROUNDING_MODE(),
            SIGN_MAGNITUDE_FORMAT(sign_magnitude),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(),
            ZERO_POINT(zero_point_bits),
            OUT_ZERO_POINT(out_zero_point_bits),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            DataFormat.Int32,
            src_A[:_ELEMENTS_PER_TILE],
            DataFormat.Int32,
            output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=1,
            tile_count_res=tile_cnt_A,
            twos_complement=True,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": True,
        "disable_format_inference": True,
        "compile_time_formats": True,
    }
    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    fc = configuration.formats_config[0]
    fc.pack_src = output_format
    fc.pack_dst = output_format
    fc.pack_S_src = output_format
    fc.pack_S_dst = output_format
    if is_perf:
        configuration.run(perf_report)
        return None

    res = configuration.run().result
    if output_format == DataFormat.Float32:
        words = fp32_bits_as_int32(torch.tensor(res, dtype=torch.float32).flatten())
    else:
        words = torch.tensor(res, dtype=torch.int32).flatten()
    words = words.numpy().reshape(pairs, 2, _ELEMENTS_PER_TILE)
    assert np.array_equal(
        words[:, 1, :].reshape(-1), in1_words
    ), f"{mathop.name}: the untouched in1 tiles did not round-trip bit-exactly"
    return words[:, 0, :].reshape(-1).copy()


def _run_quant(mathop, sign_magnitude, **run_kwargs):
    """Stage the Quasar quant stimuli on the shared WH/BH binary kernel."""
    n = _ELEMENTS_PER_TILE
    is_dequant = mathop == MathOperation.SfpuDequant
    a_is_int = mathop != MathOperation.SfpuQuant
    output_format = DataFormat.Float32 if is_dequant else DataFormat.Int32
    # Wide enough that |a| near 2000 saturates the int8 clamp on both signs.
    scale = torch.rand(n, dtype=torch.float32) * 0.2 + 0.01
    zero_point = 3.0
    out_zero_point = 1.0 if mathop == MathOperation.SfpuRequant else 0.0

    if a_is_int:
        a_float = torch.randint(-2000, 2000, (n,), dtype=torch.int32).to(torch.float32)
    else:
        a_float = (torch.rand(n, dtype=torch.float32) - 0.5) * 4000.0

    # Dyadic edges, so they do not depend on the random draw. 0.25 and 0.5 are
    # exact in fp32. Lane 2 is an exact half for requant (4.5 = 3 * 0.5 + zp);
    # 5.5 would round to 6 under both tie rules. Lane 4 is an exact half for quant (2.5 = 5 * 0.5).
    a_float[0], scale[0] = 4000.0, 0.25
    a_float[1], scale[1] = -4000.0, 0.25
    a_float[2], scale[2] = 3.0, 0.5
    a_float[3], scale[3] = -13.0, 0.25
    a_float[4], scale[4] = 5.0, 0.5

    if a_is_int:
        a_vals = a_float.to(torch.int32)
        a_staged = int32_to_smag32(a_vals) if sign_magnitude else a_vals
    else:
        a_staged = fp32_bits_as_int32(a_float)

    if is_dequant:
        golden_active = ((a_float - zero_point) * scale).to(torch.float32)
    else:
        assert not sign_magnitude, f"{mathop.name} runs only in 2's complement"
        exact = a_float.double() * scale.double()  # exact in float64
        if mathop == MathOperation.SfpuQuant:
            rounded = torch.round(exact) + zero_point
        else:
            # requant_init adds zero_point before rounding and out_zero_point after.
            rounded = (
                torch.round((exact + zero_point).float().double()) + out_zero_point
            )
        clamped = torch.clamp(rounded, _QUANT_INT8_MIN, _QUANT_INT8_MAX).to(torch.int32)
        golden_active = clamped.clone()
        assert (clamped == _QUANT_INT8_MIN).any() and (
            clamped == _QUANT_INT8_MAX
        ).any(), f"{mathop.name} stimuli never reach the int8 clamp"
        # Ties round to even: quant 2.5 -> 2 (+3), requant 4.5 -> 4 (+1).
        tie_lane, tie_value = (4, 5) if mathop == MathOperation.SfpuQuant else (2, 5)
        assert (
            clamped[tie_lane] == tie_value
        ), f"{mathop.name} exact half must round to even"

    zp_for_kernel = -zero_point if is_dequant else zero_point
    zp_bits = struct.unpack("<I", struct.pack("<f", zp_for_kernel))[0]
    out_zp_bits = struct.unpack("<I", struct.pack("<f", out_zero_point))[0]
    words = _run_sfpu_binary_word_pairs(
        mathop,
        a_staged.numpy().astype(np.int32),
        fp32_bits_as_int32(scale).numpy(),
        zp_bits,
        output_format,
        sign_magnitude=sign_magnitude,
        out_zero_point_bits=out_zp_bits,
        **run_kwargs,
    )
    if words is None:
        return
    res_tensor = torch.from_numpy(words)
    if is_dequant:
        res_tensor = res_tensor.view(torch.float32)
    assert passed_test(
        golden_active, res_tensor, output_format
    ), f"{mathop.name} sign_magnitude={sign_magnitude} mismatch"


@parametrize(**QUANT_SWEEP)
def test_eltwise_binary_sfpu_quant(mathop, sign_magnitude, **run_kwargs):
    _run_quant(mathop, sign_magnitude, **run_kwargs)


# =============================================================================
# Quant family, 8-bit entry points: the quantization.h wrappers with uint8 / int8 output or int8
# input, on raw Int32 words against an exact emulation. Production runs them without sign-magnitude.
# =============================================================================

_QUANT_BYTE_OPS = [
    MathOperation.SfpuQuantUint8,
    MathOperation.SfpuQuantInt8,
    MathOperation.SfpuRequantUint8,
    MathOperation.SfpuRequantInt8,
    MathOperation.SfpuRequantInt8In,
    MathOperation.SfpuRequantInt8InUint8Out,
    MathOperation.SfpuRequantInt8InInt8Out,
    MathOperation.SfpuDequantInt8,
]
QUANT_BYTE_SWEEP = dict(mathop=_QUANT_BYTE_OPS)

# Operand-0 width and result width of each entry point.
_QUANT_FP32_IN = frozenset({MathOperation.SfpuQuantUint8, MathOperation.SfpuQuantInt8})
_QUANT_INT8_IN = frozenset(
    {
        MathOperation.SfpuRequantInt8In,
        MathOperation.SfpuRequantInt8InUint8Out,
        MathOperation.SfpuRequantInt8InInt8Out,
        MathOperation.SfpuDequantInt8,
    }
)
_QUANT_INT8_OUT = frozenset(
    {
        MathOperation.SfpuQuantInt8,
        MathOperation.SfpuRequantInt8,
        MathOperation.SfpuRequantInt8InInt8Out,
    }
)
_QUANT_UINT8_OUT = frozenset(
    {
        MathOperation.SfpuQuantUint8,
        MathOperation.SfpuRequantUint8,
        MathOperation.SfpuRequantInt8InUint8Out,
    }
)
_QUANT_FP32_OUT = frozenset({MathOperation.SfpuDequantInt8})

_F32_MAX = Fraction(float(np.finfo(np.float32).max))


def _f32_bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).view(np.int32)


def _bits_f32(words) -> np.ndarray:
    """fp32 values from 32-bit patterns given as int32 or as unsigned Python ints."""
    return (
        (np.asarray(words, dtype=np.int64) & 0xFFFFFFFF)
        .astype(np.uint32)
        .view(np.float32)
    )


def _exact_f32(fr: Fraction):
    """The fp32 equal to *fr*; +-inf past FLT_MAX; None when finite but not representable."""
    if abs(fr) > _F32_MAX:
        return np.float32(np.inf) if fr > 0 else np.float32(-np.inf)
    f = np.float32(float(fr))
    return f if Fraction(float(f)) == fr else None


def _round_half_even(fr: Fraction) -> tuple[int, bool]:
    """(round-half-to-even of *fr*, whether *fr* was an exact tie)."""
    return round(fr), fr.denominator == 2


def _quant_byte_golden(mathop, in0_words, scale_words, zero_point_bits):
    """Bit-exact emulation of one 8-bit quant entry point on raw Int32 words.

    Returns (expected words, defined mask, tie mask). A lane is undefined when the result needs an
    unspecified hardware choice: a non-finite operand or an inexact fp32 multiply-add. Ties are
    returned so the caller asserts the tie rule.
    """
    in0_words = np.asarray(in0_words, dtype=np.int32)
    scales = _bits_f32(scale_words)
    n = in0_words.size
    expected = np.zeros(n, dtype=np.int32)
    defined = np.ones(n, dtype=bool)
    tie = np.zeros(n, dtype=bool)
    zp = float(_bits_f32([zero_point_bits])[0])
    # Quant adds its zero point after rounding. The kernel adds it exactly only when it is an integer.
    if not math.isfinite(zp) or (mathop in _QUANT_FP32_IN and not zp.is_integer()):
        defined[:] = False
        return expected, defined, tie
    zp_fr = Fraction(zp)
    in0_f32 = _bits_f32(in0_words) if mathop in _QUANT_FP32_IN else None
    if mathop in _QUANT_UINT8_OUT:
        lo, hi = 0, _QUANT_UINT8_MAX
    else:
        lo, hi = _QUANT_INT8_MIN, _QUANT_INT8_MAX

    def store(i, q):
        """Saturate q to the output range. Int8 output keeps the 2's complement byte."""
        q = min(max(q, lo), hi)
        expected[i] = q & 0xFF if mathop in _QUANT_INT8_OUT else q

    def saturate(i, v_inf):
        """An infinite value saturates each int path by its sign; the fp32 path keeps it."""
        if mathop in _QUANT_FP32_OUT:
            expected[i] = _f32_bits([v_inf])[0]
        else:
            store(i, hi if v_inf > 0 else lo)

    for i in range(n):
        s = float(scales[i])
        if not math.isfinite(s):
            defined[i] = False
            continue
        s_fr = Fraction(s)
        if mathop in _QUANT_FP32_IN:
            a = float(in0_f32[i])
            if not math.isfinite(a):
                if math.isnan(a) or s == 0.0:
                    defined[i] = False  # NaN in, or inf * 0
                else:
                    saturate(i, a * s)
                continue
            # One SFPMAD rounds the exact A * B to nearest even. The zero point is added after.
            r, tie[i] = _round_half_even(Fraction(a) * s_fr)
            store(i, r + int(zp))
            continue
        elif mathop in _QUANT_INT8_IN:
            # The byte as the UInt8 unpacker delivers it, after the kernel's `^ 0x80` unbias.
            a_fr = Fraction((int(in0_words[i]) & 0xFF) ^ _QUANT_INT8_BIAS)
        else:
            a_fr = Fraction(int(in0_words[i]))
            if a_fr == -(2**31):
                # No sign-magnitude encoding: the input SFPCAST + SFPSETSGN turns it into -0.
                defined[i] = False
                continue

        if mathop in _QUANT_FP32_OUT:
            # dequant: (A + LREG2) * B, LREG2 holding the bits the caller passed.
            summ = _exact_f32(a_fr + zp_fr)
            if summ is None or (np.isinf(summ) and s == 0.0):
                defined[i] = False
                continue
            if np.isinf(summ):
                saturate(i, float(summ) * (1.0 if s > 0 else -1.0))
                continue
            prod = _exact_f32(Fraction(float(summ)) * s_fr)
            if prod is None:
                defined[i] = False
                continue
            expected[i] = _f32_bits([prod])[0]
            continue

        # Requant: SFPMAD gives v = A * B + zp in fp32, and v is rounded to nearest even.
        # The output zero point is 0 here.
        v = _exact_f32(a_fr * s_fr + zp_fr)
        if v is None:
            defined[i] = False
            continue
        if np.isinf(v):
            saturate(i, float(v))
            continue
        r, tie[i] = _round_half_even(Fraction(float(v)))
        store(i, r)
    return expected, defined, tie


# Power-of-two scales and odd-sixteenth (fp32) or 1/16-offset zero-point (integer) operands keep
# every multiply-add exact in fp32 and never land on a .5 tie.
_QUANT_BYTE_SCALES = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float32)


def _quant_byte_sweep_stimuli(mathop, n, generator):
    """(in0 words, scale words, zero-point bits) reaching the saturation sides."""
    scales = generator.choice(_QUANT_BYTE_SCALES, size=n).astype(np.float32)
    if mathop in _QUANT_FP32_IN:
        # Odd sixteenths in [-512, 512): every output saturates on both sides.
        k = generator.integers(-4096, 4096, size=n)
        in0 = ((2 * k + 1) / 16.0).astype(np.float32)
        in0_words = _f32_bits(in0)
        zp = 5.0
    elif mathop in _QUANT_INT8_IN:
        # Every byte value; the unbiased byte is in [0, 255], so a negative zero-point reaches the low side.
        in0_words = generator.integers(0, 256, size=n).astype(np.int32)
        zp = -300.0625
        if mathop in _QUANT_FP32_OUT:
            zp = -128.0625  # dequant: the bits of -zero_point
    else:
        # Int32 two's complement in [-2048, 2048): exact in fp32, saturates for scale >= 1/8.
        in0_words = generator.integers(-2048, 2048, size=n).astype(np.int32)
        zp = 5.0625
    return in0_words, _f32_bits(scales), int(np.uint32(_f32_bits([zp])[0]))


# Special lanes, against zero-point 0: the edges each kernel's clamp / saturation /
# conversion path is built around.
_QUANT_BYTE_F32_SPECIALS = [
    0.0, -0.0, float("inf"), float("-inf"),
    1.0e-45, -1.0e-45,  # smallest denormal
    1.17549435e-38, -1.17549435e-38,  # FLT_MIN
    3.4028235e38, -3.4028235e38,  # FLT_MAX
    1.0e6, -1.0e6, 300.0, -300.0, 256.0, 255.5, 255.0, 254.5,
    129.0, 128.5, 128.0, 127.5, 127.25, 127.0, 126.75, 126.5,
    -126.5, -126.75, -127.0, -127.25, -127.5, -127.75,
    -128.0, -128.25, -128.5, -128.75, -129.0, -200.0, 200.0,
    0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 1.0, -1.0,
]  # fmt: skip
_QUANT_BYTE_F32_SPECIAL_WORDS = [
    0x7FC00000, 0xFFC00000,  # canonical quiet NaN, both signs
    0x7F800001, 0xFF800001,  # signalling NaN payloads, both signs
    0x7FFFFFFF, 0xFFFFFFFF,  # all-ones payload NaN, both signs
]  # fmt: skip
_QUANT_BYTE_INT32_SPECIALS = [
    0, 1, -1, 127, 128, -127, -128, -129, 255, 256, -255, -256, 1000, -1000,
    2**23, -(2**23), 2**24, -(2**24), 2**31 - 1, -(2**31), -(2**31) + 1,
]  # fmt: skip
_QUANT_BYTE_INT8_SPECIALS = [0x00, 0x01, 0x7F, 0x80, 0x81, 0xFF, 0xFE, 0x40, 0xC0]


_QUANT_BYTE_PAIRS = 8  # sweep size, in (in0, scale) tile pairs


def _quant_byte_special_stimuli(mathop):
    """(in0 words, scale words, number of real lanes); the rest of the tile is padding."""
    if mathop in _QUANT_FP32_IN:
        words = list(_f32_bits(_QUANT_BYTE_F32_SPECIALS)) + [
            int(np.int32(np.uint32(w))) for w in _QUANT_BYTE_F32_SPECIAL_WORDS
        ]
        scales = [1.0, 0.5, 2.0]
    elif mathop in _QUANT_INT8_IN:
        words, scales = _QUANT_BYTE_INT8_SPECIALS, [1.0]
    else:
        words, scales = _QUANT_BYTE_INT32_SPECIALS, [1.0]
    words = np.array(words, dtype=np.int32)
    in0 = np.concatenate([words for _ in scales])
    sc = np.concatenate([np.full(words.size, s, dtype=np.float32) for s in scales])
    pad = (-in0.size) % _ELEMENTS_PER_TILE
    in0 = np.concatenate([in0, np.zeros(pad, dtype=np.int32)])
    sc = np.concatenate([sc, np.ones(pad, dtype=np.float32)])
    return in0, _f32_bits(sc), in0.size - pad


def _hex_words(words) -> str:
    return ", ".join(f"0x{int(np.uint32(np.int32(w))):08x}" for w in words)


def _assert_quant_byte_lanes(
    mathop, in0_words, scale_words, result, expected, defined, tie, label, n_real=None
):
    """Assert the defined lanes, assert one tie rule for the variant, log the rest."""
    n_real = result.size if n_real is None else n_real
    live = np.zeros(result.size, dtype=bool)
    live[:n_real] = True

    def lanes(mask, limit=16):
        idx = np.flatnonzero(mask)[:limit]
        return "\n".join(
            f"  lane {i}: in0={_hex_words([in0_words[i]])} "
            f"scale={float(_bits_f32([scale_words[i]])[0]):g} "
            f"got={_hex_words([result[i]])} expected={_hex_words([expected[i]])}"
            for i in idx
        )

    # Quant and requant round ties to nearest even.
    ties = tie & live
    if ties.any():
        assert np.array_equal(
            result[ties], expected[ties]
        ), f"{mathop.name} {label}: ties do not round to even\n" + lanes(
            ties & (result != expected)
        )
    undefined = ~defined & live
    if undefined.any():
        print(
            f"{mathop.name} {label}: implementation-defined lanes ->\n"
            + lanes(undefined, limit=64)
        )
    check = defined & ~tie & live
    mismatch = check & (result != expected)
    assert not mismatch.any(), (
        f"{mathop.name} {label}: {int(mismatch.sum())} of {int(check.sum())} defined lanes "
        "differ\n" + lanes(mismatch)
    )


def _run_quant_byte(mathop, in0_words, scale_words, zero_point_bits, **run_kwargs):
    """Run one 8-bit quant entry point on (in0, scale) tile pairs; returns the result words."""
    output_format = (
        DataFormat.Float32 if mathop in _QUANT_FP32_OUT else DataFormat.Int32
    )
    return _run_sfpu_binary_word_pairs(
        mathop, in0_words, scale_words, zero_point_bits, output_format, **run_kwargs
    )


# A lone axis goes through plain pytest.mark.parametrize: the harness @parametrize hands it over as a
# 1-tuple that --op does not see. The perf twin adds its own axes and uses QUANT_BYTE_SWEEP.
@pytest.mark.parametrize("mathop", _QUANT_BYTE_OPS)
def test_eltwise_binary_sfpu_quant_byte(mathop, **run_kwargs):
    """Sweep: eight pairs of exact, tie-free stimuli that reach the saturation sides."""
    generator = np.random.default_rng(0)
    if run_kwargs.get("is_perf"):
        # One pair, like the quant rows above; the instruction stream is data-independent.
        in0, sc, zp = _quant_byte_sweep_stimuli(mathop, _ELEMENTS_PER_TILE, generator)
        _run_quant_byte(mathop, in0, sc, zp, **run_kwargs)
        return

    in0, sc, zp = _quant_byte_sweep_stimuli(
        mathop, _QUANT_BYTE_PAIRS * _ELEMENTS_PER_TILE, generator
    )
    expected, defined, tie = _quant_byte_golden(mathop, in0, sc, zp)
    assert defined.all() and not tie.any(), "sweep stimuli must be exact and tie-free"
    if mathop in _QUANT_INT8_OUT:
        ends = {_QUANT_INT8_BIAS, _QUANT_INT8_MAX}
    elif mathop in _QUANT_UINT8_OUT:
        ends = {0, _QUANT_UINT8_MAX}
    elif mathop in _QUANT_FP32_OUT:
        ends = set()
    else:
        ends = {_QUANT_INT8_MIN, _QUANT_INT8_MAX}
    assert ends <= set(expected.tolist()), f"{mathop.name} stimuli never saturate"
    result = _run_quant_byte(mathop, in0, sc, zp)
    _assert_quant_byte_lanes(mathop, in0, sc, result, expected, defined, tie, "sweep")


@pytest.mark.parametrize("mathop", _QUANT_BYTE_OPS)
def test_eltwise_binary_sfpu_quant_byte_special(mathop):
    """Special lanes against zero-point 0: +-0, +-inf, NaN payloads, denormals, FLT_MIN/MAX,
    the -128 / 127 / 255 edges and exact .5 ties. A test of its own so --speed-of-light compiles its zero-point.
    """
    zp = 0
    in0, sc, n_real = _quant_byte_special_stimuli(mathop)
    expected, defined, tie = _quant_byte_golden(mathop, in0, sc, zero_point_bits=zp)
    result = _run_quant_byte(mathop, in0, sc, zero_point_bits=zp)
    _assert_quant_byte_lanes(
        mathop, in0, sc, result, expected, defined, tie, "special lanes", n_real
    )
    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and mathop == MathOperation.SfpuQuantInt8
    ):
        # Pinned on Blackhole: every NaN payload and sign reaches the clamp sign-clear, so it packs to 127.
        nan_lanes = np.isnan(_bits_f32(in0))
        nan_lanes[n_real:] = False
        assert nan_lanes.any()
        assert (result[nan_lanes] == _QUANT_INT8_MAX).all(), (
            "Blackhole QUANT_INT8 NaN lanes must keep the baseline result 127\n"
            + "\n".join(
                f"  lane {i}: in0={_hex_words([in0[i]])} got={_hex_words([result[i]])}"
                for i in np.flatnonzero(nan_lanes & (result != _QUANT_INT8_MAX))[:16]
            )
        )


# =============================================================================
# _float_to_int32_positive_, the float -> int32 step of the pow kernels, on raw fp32 words.
# =============================================================================

_FP32_EXP_BIAS = 127
_FP32_MANTISSA_BITS = 23
_FP32_EXP_MASK = 0xFF  # also the inf / NaN exponent code
_FP32_MANTISSA_MASK = (1 << _FP32_MANTISSA_BITS) - 1
_FP32_IMPLICIT_ONE = 1 << _FP32_MANTISSA_BITS
_INT32_MAX = 2**31 - 1
_INT32_SATURATE_EXP = 31  # the helper saturates from 2**31 up
# Largest in-range left shift of the mantissa: an exponent of 30 is 2**30 <= x < 2**31.
_F2I_MAX_LEFT_SHIFT = _INT32_SATURATE_EXP - 1 - _FP32_MANTISSA_BITS


def _float_to_int32_positive_golden(words: np.ndarray) -> np.ndarray:
    """0 below 1.0, INT32_MAX from 2**31 (incl. inf / NaN), the truncated value between."""
    wide = words.astype(np.int64)
    exp = ((wide >> _FP32_MANTISSA_BITS) & _FP32_EXP_MASK) - _FP32_EXP_BIAS
    man = (wide & _FP32_MANTISSA_MASK) | _FP32_IMPLICIT_ONE
    shift = exp - _FP32_MANTISSA_BITS
    in_range = np.where(
        shift >= 0,
        man << np.clip(shift, 0, _F2I_MAX_LEFT_SHIFT),
        man >> np.clip(-shift, 0, _FP32_MANTISSA_BITS),
    )
    saturated = np.where(exp >= _INT32_SATURATE_EXP, _INT32_MAX, in_range)
    return np.where(exp < 0, 0, saturated).astype(np.int32)


@pytest.mark.parametrize("mathop", [MathOperation.SfpuFloatToInt32Positive])
def test_eltwise_binary_sfpu_float_to_int32_positive(mathop):
    """Every biased exponent (sign clear) with eight mantissas, bit-exact."""
    generator = np.random.default_rng(0)
    mantissas = np.concatenate(
        [
            [0, 1, 1 << (_FP32_MANTISSA_BITS - 1), _FP32_MANTISSA_MASK],
            generator.integers(0, _FP32_IMPLICIT_ONE, size=4),
        ]
    )
    biased_exps = np.arange(_FP32_EXP_MASK + 1)
    words = (biased_exps[:, None] << _FP32_MANTISSA_BITS) | mantissas[None, :]
    words = words.reshape(-1)
    words = words.astype(np.int32)
    expected = _float_to_int32_positive_golden(words)
    result = _run_sfpu_binary_word_pairs(
        mathop, words, np.zeros_like(words), 0, DataFormat.Int32
    )
    mismatch = np.flatnonzero(result != expected)
    assert (
        mismatch.size == 0
    ), f"{mismatch.size} of {words.size} lanes differ\n" + "\n".join(
        f"  in={_hex_words([words[i]])} got={_hex_words([result[i]])} "
        f"expected={_hex_words([expected[i]])}"
        for i in mismatch[:16]
    )


# =============================================================================
# Integer ops
# =============================================================================


INT_SWEEP = dict(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwRightShift,
        MathOperation.SfpuElwLeftShift,
        MathOperation.SfpuElwLogicalRightShift,
        MathOperation.SfpuElwLt,
        MathOperation.SfpuElwGt,
        MathOperation.SfpuElwLe,
        MathOperation.SfpuElwGe,
    ],
    dest_acc=[DestAccumulation.Yes],
)


@parametrize(**INT_SWEEP)
def test_eltwise_binary_sfpu_int(
    formats,
    dest_acc,
    mathop,
    **run_kwargs,
):
    # The random half of the Int32 coverage, on the positive-only tie-free integer default --
    # so it cannot tell SfpuElwLe from SfpuElwLt. The two tests below cover the rest.
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        **run_kwargs,
    )


# The four ordered Int32 comparisons. Same MathOperation members as the float comparison
# sweep, routed to the integer kernel. test_sfpu_domains asserts this list stays these four.
_INT_COMPARISON_OPS = [
    MathOperation.SfpuElwLt,
    MathOperation.SfpuElwGt,
    MathOperation.SfpuElwLe,
    MathOperation.SfpuElwGe,
]


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=_INT_COMPARISON_OPS,
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_int_comparison_ties(formats, dest_acc, mathop):
    """The exact-equality input, which the random Int32 sweep never produces.

    `a == b` is the only input on which lt/gt disagree with le/ge, so without it a comparator
    with its tie inverted passes the whole integer sweep. Reuses the float sweep's three-way
    builder, whose +/-1 gaps are exact on an integer axis.
    """
    spec_A, spec_B = _comparison_stimuli_specs()
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B)


def _int_comparison_negative_spec():
    """Paired Int32 stimuli that straddle zero, including a tie at zero.

    Operands of opposite sign are the case the kernel's sign fold exists for, and the random
    draw never produces one. Values stay small so `a - b` cannot overflow, which
    test_eltwise_binary_sfpu_int_extremes drives separately. twos_complement=True is required:
    a sign-magnitude L1 encoding cannot round-trip a negative through Dst.
    """

    def a_face(size, dtype, generator):
        positions = torch.arange(size, dtype=torch.float32)
        # -4..3, so both signs and zero appear.
        return ((positions % 8) - 4.0).to(dtype)

    def b_face(size, dtype, generator):
        positions = torch.arange(size, dtype=torch.float32)
        # Mirror of a_face: pairs run (-4, 4), (-3, 3), ... including (0, 0) as an exact tie
        # at zero, where a sign-magnitude comparator is most likely to disagree.
        return (4.0 - (positions % 8)).to(dtype)

    return _face_spec(a_face), _face_spec(b_face)


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=_INT_COMPARISON_OPS,
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_int_comparison_across_zero(formats, dest_acc, mathop):
    """Negative and mixed-sign Int32 operands, which the positive-only default never reaches.

    Without these a comparator that mishandled operand sign entirely would pass every other
    Int32 test in this file.
    """
    spec_A, spec_B = _int_comparison_negative_spec()
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        spec_A=spec_A,
        spec_B=spec_B,
        twos_complement=True,
    )


def _int_arith_negative_spec():
    """Paired Int32 stimuli where both operands independently take either sign.

    The Int32 add/sub kernels convert each operand out of sign-magnitude before the integer
    add and the result back into it, so three sign folds run per element. The positive-only
    default never sets an operand's sign bit, which leaves the two input-side folds unreached
    by every other Int32 test here -- only the output-side one is exercised, and then only
    because `a - b` on non-negative operands still goes negative.

    The two operands vary at different rates so the face walks the full 8x8 product of
    operand values rather than a single diagonal. A mirrored pair would collapse `a + b` to a
    constant zero and test nothing. Values stay small so neither sum nor difference can
    overflow; test_eltwise_binary_sfpu_int_extremes drives the range ends separately.
    """

    def a_face(size, dtype, generator):
        positions = torch.arange(size, dtype=torch.float32)
        # -4..3, cycling every 8 elements.
        return ((positions % 8) - 4.0).to(dtype)

    def b_face(size, dtype, generator):
        positions = torch.arange(size, dtype=torch.float32)
        # -4..3, cycling every 64, so each a value meets every b value, including (0, 0).
        return (((positions // 8) % 8) - 4.0).to(dtype)

    return _face_spec(a_face), _face_spec(b_face)


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuElwadd, MathOperation.SfpuElwsub],
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_int_arith_across_zero(formats, dest_acc, mathop):
    """Negative and mixed-sign operands for the Int32 add/sub kernels.

    Covers the input-side sign folds in _add_int_ / _sub_int_, which no other Int32 test in
    this file reaches.

    Unlike test_eltwise_binary_sfpu_int_comparison_across_zero above, this one keeps the
    sign-magnitude default rather than passing twos_complement=True. These kernels are
    sign-magnitude end to end -- operands in, result out -- so two's-complement stimuli are
    read back with the sign bit as a magnitude bit and every negative lane diverges. The
    comparison ops can take either encoding on the way out only because they emit 0 or 1.
    """
    spec_A, spec_B = _int_arith_negative_spec()
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B)


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuElwadd, MathOperation.SfpuElwsub],
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_int_arith_wide_signed(formats, dest_acc, mathop):
    """Large-magnitude operands of both signs for the Int32 add/sub kernels.

    across_zero above walks every combination of operand sign but only at magnitude <= 4, and
    the positive-only default reaches wide magnitudes but never sets an operand's sign bit.
    Between them a sign fold that only misbehaves on a wide bit pattern -- the shape a cast
    defect usually takes -- goes unseen.

    Bounds are +-2**29 so neither the sum nor the difference can leave the +-(2**31 - 1) that
    sign-magnitude Dst represents. A and B draw distinct values because the paired face walk
    advances one shared RNG stream across faces, not because of the per-spec seed, which
    generate_face ignores whenever an external generator is supplied.
    """
    bound = float(2**29)
    spec_A = StimuliSpec(
        distribution=DistributionKind.UNIFORM, low=-bound, high=bound, seed=0
    )
    spec_B = StimuliSpec(
        distribution=DistributionKind.UNIFORM, low=-bound, high=bound, seed=1
    )
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B)


BITWISE_SWEEP = dict(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[
        MathOperation.SfpuBitwiseAnd,
        MathOperation.SfpuBitwiseOr,
        MathOperation.SfpuBitwiseXor,
    ],
    dest_acc=[DestAccumulation.Yes],
)


@parametrize(**BITWISE_SWEEP)
def test_eltwise_binary_sfpu_bitwise(formats, dest_acc, mathop, **run_kwargs):
    # int32 bitwise AND/OR/XOR: exact on the full default int range.
    sfpu_binary(formats, dest_acc, mathop, **run_kwargs)


# Ops whose kernel interprets DST as unsigned; run them under UInt32 (the rest are Int32).
_UINT32_BINARY_OPS = {
    MathOperation.SfpuMaxUint32,
    MathOperation.SfpuMinUint32,
    MathOperation.SfpuRemainderUint32,
}

# int/uint binary ops sharing the same driver: dest_acc=Yes, single-format, and a per-op
# uniform positive stimuli range. Ranges keep operands (and results) non-negative and small
# enough to round-trip the sign-magnitude Dst packer plus any int->fp32 reciprocal the
# kernel uses. mathop -> (low, high).
_INT_BINARY_STIMULI = {
    # trunc/floor division < 2**24: exact int->fp32 reciprocal, trunc == floor, and the
    # sign-magnitude pack path can't round-trip the negatives these kernels would emit.
    MathOperation.SfpuDivInt32: (1.0, 8_000_000.0),
    MathOperation.SfpuDivInt32Floor: (1.0, 8_000_000.0),
    # binary-GCD on raw int32 bits (exact): strictly positive within the 31-bit budget.
    MathOperation.SfpuGcd: (1.0, 100_000.0),
    # lcm abs()es both operands and assumes |a|, |b| < 2**15.
    MathOperation.SfpuLcm: (1.0, 20_000.0),
    # int32 multiply low-32: operands < ~46340 so the product stays < 2**31 (non-negative).
    MathOperation.SfpuMulInt32: (1.0, 40_000.0),
    # int32/uint32 max/min via SFPSWAP: non-negative so signed/unsigned agree and round-trip.
    MathOperation.SfpuMaxInt32: (0.0, 1_000_000.0),
    MathOperation.SfpuMinInt32: (0.0, 1_000_000.0),
    MathOperation.SfpuMaxUint32: (0.0, 1_000_000.0),
    MathOperation.SfpuMinUint32: (0.0, 1_000_000.0),
    # remainder/fmod: non-negative operands, divisor >= 1 so every convention agrees;
    # kept < 2**24 for the exact int->fp32 reciprocal the quotient uses.
    MathOperation.SfpuRemainderInt32: (1.0, 10_000.0),
    MathOperation.SfpuFmodInt32: (1.0, 10_000.0),
    MathOperation.SfpuRemainderUint32: (1.0, 10_000.0),
}


INT_UNIFORM_SWEEP = dict(
    mathop=list(_INT_BINARY_STIMULI),
    dest_acc=[DestAccumulation.Yes],
)


@parametrize(**INT_UNIFORM_SWEEP)
def test_eltwise_binary_sfpu_int_uniform(mathop, dest_acc, **run_kwargs):
    _skip_sfpu_lcm_dest_acc_bh(mathop, dest_acc)
    int_format = DataFormat.UInt32 if mathop in _UINT32_BINARY_OPS else DataFormat.Int32
    formats = InputOutputFormat(int_format, int_format)
    low, high = _INT_BINARY_STIMULI[mathop]
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=low, high=high),
        **run_kwargs,
    )


_DIV_INT32_SIGNED_DIVIDENDS = [
    *integer_specials(DataFormat.Int32),
    -7,
    7,
    -100,
    100,
    -(2**24 + 1),
    2**24 + 1,
    -0x55555555,
    0x55555555,
]
_DIV_INT32_SIGNED_DIVISORS = [
    v for v in integer_specials(DataFormat.Int32) if v != 0
] + [-2, 2, -3, 3, -7, 7, -(2**16 + 1), 2**16 + 1]
_DIV_INT32_BOUNDARY_DIVISORS = [-3, 3, -7, 7, -12345, 12345, -(2**16 + 1), 2**16 + 1]
_DIV_INT32_BOUNDARY_QUOTIENTS = [-1000, -1, 1, 1000]


def _build_div_int32_signed_src():
    """Two-tile Int32 signed (dividend, divisor) pairs, including mixed signs the positive sweep misses.

    Also places q*b-1, q*b, and q*b+1 so floor and trunc disagree on one side of each exact quotient.
    """
    pairs = list(
        itertools.product(_DIV_INT32_SIGNED_DIVIDENDS, _DIV_INT32_SIGNED_DIVISORS)
    )
    pairs += [
        (q * b + off, b)
        for b in _DIV_INT32_BOUNDARY_DIVISORS
        for q in _DIV_INT32_BOUNDARY_QUOTIENTS
        for off in (-1, 0, 1)
    ]
    return _build_paired_tile_override(pairs, torch.int32)


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuDivInt32, MathOperation.SfpuDivInt32Floor],
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_div_int32_signed(formats, dest_acc, mathop):
    # twos_complement=True for the same reason as test_eltwise_binary_sfpu_int_extremes: the
    # kernel reads two's-complement Dst, and sign-magnitude stimuli cannot carry INT32_MIN.
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        src_A_override=_build_div_int32_signed_src(),
        twos_complement=True,
    )


RSUB_INT32_SWEEP = dict(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuRsubInt32],
    dest_acc=[DestAccumulation.Yes],
)


@parametrize(**RSUB_INT32_SWEEP)
def test_eltwise_binary_sfpu_rsub_int32(formats, dest_acc, mathop, **run_kwargs):
    sfpu_binary(formats, dest_acc, mathop, twos_complement=True, **run_kwargs)


EQ_NE_INT_SWEEP = dict(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuEqInt, MathOperation.SfpuNeInt],
    dest_acc=[DestAccumulation.Yes],
)


@parametrize(**EQ_NE_INT_SWEEP)
def test_eltwise_binary_sfpu_eq_ne_int(formats, dest_acc, mathop, **run_kwargs):
    # int32 eq/ne via calculate_binary_eq_int (exact 0/1 over the raw INT32 dest bits).
    # Reuse the paired eq/ne stimuli so ~50% of positions compare equal — the equal branch
    # a plain random int sweep would essentially never hit.
    spec_A, spec_B = _eq_ne_stimuli_specs()
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B, **run_kwargs)


# Integer shift edge cases: shift amounts outside [0, 31], arithmetic sign extension, and
# negative operands, INT32_MIN included. These tests pack two's complement, which is the encoding
# native Int32 tiles carry in Dst, so the whole int32 range round-trips -- including 0x80000000,
# which sign-magnitude packing would turn into "negative zero".

_INT32_MIN = -(2**31)

_SHIFT_EDGE_OPS = [
    MathOperation.SfpuElwRightShift,
    MathOperation.SfpuElwLeftShift,
    MathOperation.SfpuElwLogicalRightShift,
]

# Representative Int32 values: zero, small magnitudes of both signs, byte / halfword
# boundaries, the sign bit, an alternating bit pattern and the int32 extremes.
_SHIFT_EDGE_VALUES = [
    0,
    1,
    -1,
    2,
    -2,
    7,
    -8,
    255,
    -256,
    0x0000FFFF,
    -0x00010000,
    0x40000000,
    -0x40000000,
    0x55555555,
    -0x55555555,
    0x7FFFFFFF,  # INT32_MAX
    -0x80000000,  # INT32_MIN
]

# Shift amounts now live in sfpu_domains.SHIFT_EDGE_AMOUNTS, because the unary shift sweep
# drives the same list through a compile-time immediate and a second copy would let
# "interesting shift" mean two different things.
_SHIFT_EDGE_AMOUNTS = list(SHIFT_EDGE_AMOUNTS)


def _shift_reference(mathop, value, shift):
    """Bit-exact reference for one (value, shift) pair: shifts outside [0, 31] -> 0, right
    shift arithmetic, logical right shift unsigned, left shift plain. Mirrors BinarySFPUGolden.
    """
    shift = int(shift)
    if shift < 0 or shift >= 32:
        return 0
    v = torch.tensor(int(value), dtype=torch.int32)
    if mathop == MathOperation.SfpuElwRightShift:
        return int(torch.bitwise_right_shift(v, shift))
    if mathop == MathOperation.SfpuElwLeftShift:
        return int(torch.bitwise_left_shift(v, shift))
    if mathop == MathOperation.SfpuElwLogicalRightShift:
        r = (int(value) & 0xFFFFFFFF) >> shift
        return r - 0x100000000 if r >= 0x80000000 else r
    raise ValueError(f"Unsupported shift op: {mathop}")


def _build_shift_edge_case_src(mathop):
    """Build a deterministic [64, 32] Int32 operand: tile 0 holds values, tile 1 holds
    per-element shift amounts (tilize pairs them by index). Walks the full cartesian product of
    interesting (value, shift) pairs, INT32_MIN included."""
    pairs = list(itertools.product(_SHIFT_EDGE_VALUES, _SHIFT_EDGE_AMOUNTS))
    return _build_paired_tile_override(pairs, torch.int32)


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    mathop=_SHIFT_EDGE_OPS,
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_int_shift_edge_cases(
    formats,
    dest_acc,
    mathop,
):
    # twos_complement=True is required, not decorative, and for the same reason as in
    # test_eltwise_binary_sfpu_int_extremes: the shift kernels operate on Dst bits directly, so
    # sign-magnitude stimuli would hand them 0x80000001 for -1 and every negative lane would
    # diverge from the two's-complement golden. Native Int32 tiles are 2's complement in Dst
    # (see binary_shift.h), so this is also the encoding the kernels see in production.
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        src_A_override=_build_shift_edge_case_src(mathop),
        twos_complement=True,
    )


# Deliberate edge values, generalizing _build_shift_edge_case_src above: walk the cartesian
# product of both operands' interesting values into a two-tile override tensor, which tilize
# pairs by index. The product is the point -- a divisor of 0 against a positive, a negative and
# a zero numerator are three different cases. Values come from sfpu_domains.edge_pair_values(),
# so ops enrol by having a registered singularity rather than by being listed here.


# The edge-pair probe is partitioned before it is driven, and the partition is what makes the
# xfails below mean anything: one tensor holding every pole of an op mixes unrelated failure
# classes, so a single xfail covers all of them and a new mismatch stays invisible for as long
# as any known one survives. Classified by what the golden answers rather than by a predicate
# on the operands, since no reading of the operand signs gets the classes right.
_EDGE_CLASS_BOTH_ZERO = "both_zero"
_EDGE_CLASS_NAN = "nan_golden"
_EDGE_CLASS_NEGATIVE_ZERO = "negative_zero_golden"
_EDGE_CLASS_ORDINARY = "ordinary"

# Cat B: a non-finite operand, as opposed to a non-finite answer. Its own class because the
# other four classify by what the golden answers, which would file IEEE propagation and the
# kernel's own composition in the same bucket.
_EDGE_CLASS_SPECIALS_IN = "specials_in"

# Order matters only for the compile-producer pass: whichever class comes first builds the ELF
# all of them share, and the test body guards against an empty representative starving the
# others of a binary.
_EDGE_CLASSES = (
    _EDGE_CLASS_ORDINARY,
    _EDGE_CLASS_BOTH_ZERO,
    _EDGE_CLASS_NAN,
    _EDGE_CLASS_NEGATIVE_ZERO,
    _EDGE_CLASS_SPECIALS_IN,
)


def _classify_edge_pair(mathop, a, b):
    """Which failure class the pair (*a*, *b*) belongs to for *mathop*."""
    # Tested first because it is a property of the input and the other classes are properties
    # of the output, so otherwise every cat-B pair would be filed as nan_golden.
    if not (math.isfinite(a) and math.isfinite(b)):
        return _EDGE_CLASS_SPECIALS_IN
    if a == 0.0 and b == 0.0:
        return _EDGE_CLASS_BOTH_ZERO

    # Instantiated directly rather than through get_golden_generator: this runs at
    # stimulus-build time, which happens in both build phases, and the --compile-producer stub
    # has no `ops` mapping.
    result = float(BinarySFPUGolden().ops[mathop](torch.tensor(a), torch.tensor(b)))
    if math.isnan(result):
        return _EDGE_CLASS_NAN
    if result == 0.0 and math.copysign(1.0, result) < 0.0:
        return _EDGE_CLASS_NEGATIVE_ZERO
    return _EDGE_CLASS_ORDINARY


def _edge_pairs_for_class(mathop, formats, edge_class, dest_acc, specials=False):
    """The operand pairs of *edge_class* for this op and pipeline.

    Extracted so the override builder and the generated-NaN predicate select from the same
    list. *dest_acc* sizes the ULP steps around each pole, since at dest_acc=No a probe stepped
    by an fp32 ULP would land back on the pole it was straddling.
    """
    return [
        pair
        for pair in edge_pair_values(
            mathop,
            formats.input_format,
            formats.output_format,
            specials=specials,
            dest_acc=dest_acc,
        )
        if _classify_edge_pair(mathop, *pair) == edge_class
    ]


# The ops this suite drives on an integer format. This sweep is float, so an integer op that
# gains a singularity belongs in test_eltwise_binary_sfpu_int_extremes or the shift edge tests
# instead. Assembled from the lists that already drive them, so the two cannot disagree.
_INT_DRIVEN_BINARY_OPS = frozenset(
    set(_INT_BINARY_STIMULI)
    | set(_UINT32_BINARY_OPS)
    | set(_SHIFT_EDGE_OPS)
    | {
        MathOperation.SfpuBitwiseAnd,
        MathOperation.SfpuBitwiseOr,
        MathOperation.SfpuBitwiseXor,
        MathOperation.SfpuEqInt,
        MathOperation.SfpuNeInt,
        MathOperation.SfpuRsubInt32,
    }
)

# Derived rather than listed: an op joins this sweep by gaining either a registered
# singularity or a cat-B entry in sfpu_domains. The cat-B half matters for the float ops with
# no pole -- add, sub, mul, max, min and the comparisons are smooth everywhere, so
# ops_with_singularity() alone could never collect them.
_BINARY_EDGE_OPS = sorted(
    (
        (ops_with_singularity() | set(BINARY_SPECIALS_READY_OPS))
        & _CLASSIFIED_STIMULI_OPS
    )
    - _INT_DRIVEN_BINARY_OPS,
    key=lambda op: op.name,
)

assert _BINARY_EDGE_OPS, (
    "no float binary op reaching sfpu_binary() has an entry in "
    "sfpu_domains._OP_SINGULARITIES or in BINARY_SPECIALS_READY_OPS, so "
    "test_eltwise_binary_sfpu_edges would collect nothing"
)


# Driving the poles found nothing left to tolerate on Wormhole. The negative-zero class used
# to carry a non-strict xfail for div, xlogy, fmod and remainder on the grounds that SFPMAD
# flushes a zero result to positive zero; all 16 cells XPASS, and neither arm of
# passed_test can see the sign of a zero: torch.isclose compares them equal, and the ULP
# arm's value-order index ranks both zeros the same. The indeterminate forms
# are asserted too, now that the golden models the packer substituting an infinity for a NaN
# the pipeline was too narrow to hold; what remains of them on Wormhole is that infinity's
# sign, handled per lane by generated_nan_sign_is_asserted() rather than by an xfail.


@pytest.mark.nightly
@parametrize(
    formats=_BF16_FP32_FORMATS,
    mathop=_BINARY_EDGE_OPS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    # runtime(): the class selects which values go in the override tensor and nothing
    # else, so all four share one ELF instead of compiling the same kernel four times.
    edge_class=runtime(list(_EDGE_CLASSES)),
)
def test_eltwise_binary_sfpu_edges(formats, dest_acc, mathop, edge_class):
    """Drive one class of each binary op's registered pole against its counterparts.

    One variant per (op, class) rather than per op: see the comment above _EDGE_CLASSES for
    why a single tensor holding every pole cannot report which behaviour changed.
    """
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    # Cat B. Two independent gates, both must pass: BINARY_SPECIALS_READY_OPS says the golden
    # defines an answer for a non-finite operand, specials_safe() says the pipeline delivers one
    # intact. dest_acc as passed, which is conservative on Blackhole, where it is promoted later.
    specials = mathop in BINARY_SPECIALS_READY_OPS and specials_safe(
        formats.input_format, formats.output_format, dest_acc
    )

    pairs = _edge_pairs_for_class(
        mathop, formats, edge_class, dest_acc, specials=specials
    )

    if not pairs and TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        # The compile-producer pass must not skip on a runtime-only axis: one item per compile
        # key builds the ELF all classes share, and skipping it leaves the others running against
        # a binary that was never built. The ELF does not depend on which values go in the
        # tensor, so the unfiltered list compiles the right kernel; the consumer still skips.
        pairs = edge_pair_values(
            mathop,
            formats.input_format,
            formats.output_format,
            specials=specials,
            dest_acc=dest_acc,
        )

    if not pairs:
        pytest.skip(
            reason=f"{mathop.name} has no {edge_class} pair among its registered "
            "per-operand edges"
            + ("" if specials else " (cat B is off for this op or this pipeline)")
        )

    # Where the golden's answer is a NaN the op invented, a narrowing pipeline turns its sign
    # into the observable result and Wormhole leaves that sign unspecified, so assert the
    # magnitude there rather than withdrawing the variant. This decides pipeline and arch only;
    # which lanes carry an invented NaN is the golden's own mask, applied by sfpu_binary().
    unspecified_sign = generated_nan_sign_is_asserted(
        formats.input_format,
        formats.output_format,
        dest_acc,
        on_wormhole=TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE,
    )

    dtype = torch.int32 if formats.input_format.is_integer() else torch.float32
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        src_A_override=_build_paired_tile_override(pairs, dtype),
        unspecified_nonfinite_sign=unspecified_sign,
    )


# Integer extremes (cat C). Delivered as a raw override rather than a StimuliSpec, because a
# spec asking for INT32_MIN is silently clamped to INT32_MIN + 1 -- the worst failure mode for
# an edge test. Scope is deliberately narrow: most int binary kernels document a sub-range they
# are valid on (see _INT_BINARY_STIMULI), so feeding them the extremes would produce documented
# limitations rather than findings. The bitwise ops and the exact comparisons document no such
# range, so they are what cat C can honestly cover. INT32_MIN itself is excluded because
# sign-magnitude Dst cannot round-trip it; it has its own xfail below.
_INT_EXTREME_OPS = [
    MathOperation.SfpuBitwiseAnd,
    MathOperation.SfpuBitwiseOr,
    MathOperation.SfpuBitwiseXor,
    MathOperation.SfpuEqInt,
    MathOperation.SfpuNeInt,
    *_INT_COMPARISON_OPS,
]


def _build_int_extremes_src():
    """Two-tile Int32 override walking the product of the int32 extremes, minus INT32_MIN."""
    vals = [v for v in integer_specials(DataFormat.Int32) if v != _INT32_MIN]
    pairs = [(a, b) for a in vals for b in vals]
    return _build_paired_tile_override(pairs, torch.int32)


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=_INT_EXTREME_OPS,
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_int_extremes(formats, dest_acc, mathop):
    # twos_complement=True is required, not decorative: without it the buffer is packed
    # sign-magnitude and the bitwise kernels operate on the wrong bits for negative operands.
    # test_eltwise_binary_sfpu_bitwise cannot catch that -- its stimuli are positive-only.
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        src_A_override=_build_int_extremes_src(),
        twos_complement=True,
    )


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    mathop=_SHIFT_EDGE_OPS,
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_int_shift_int32_min(
    formats,
    dest_acc,
    mathop,
):
    # Every value lane is INT32_MIN shifted by 0, so the golden expects INT32_MIN back. Worth its
    # own test because 0x80000000 is the one value whose round-trip depends entirely on the Dst
    # encoding: two's complement carries it, sign-magnitude reads it as "negative zero".
    num_elements = 32 * 32
    value_grid = [_INT32_MIN] * num_elements
    shift_grid = [0] * num_elements
    src = torch.tensor(value_grid + shift_grid, dtype=torch.int32)
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        src_A_override=src,
        twos_complement=True,
    )


# =============================================================================
# add_top_row
#
# Own inline driver: uses sources/sfpu_binary_test.cpp with a single [64, 32]
# tile pair, disabled format inference, and a golden reshaped to the top-row
# layout — different enough from sfpu_binary() to keep separate.
# =============================================================================

ADD_TOP_ROW_SWEEP = dict(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.UInt32,
        ],
        same=True,
    ),
    mathop=[MathOperation.SfpuAddTopRow],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)


def _run_sfpu_add_top_row(
    formats,
    dest_acc,
    mathop,
    *,
    src_A_override=None,
    twos_complement=False,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
    iterations=32,
    approx_mode=ApproximationMode.No,
):
    input_dimensions = [64, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # The kernel only consumes buffer_A (tile 0 = operand 0, tile 1 = operand 1); the
    # override must be exactly the one [64, 32] tile pair the driver runs.
    if src_A_override is not None:
        src_A = _override_src_A(src_A, src_A_override, repeat=False)

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        0,
        1,
        0,
        1,
        input_dimensions,
        formats.output_format,
    )

    golden_tensor = (
        golden_tensor.view([32, 32])
        if golden_tensor.shape == torch.Size([1024])
        else golden_tensor.view(input_dimensions)
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS
    )

    run_types = _resolve_perf_args(is_perf, perf_report, run_types)

    test_config_kwargs = {
        "test_name": "sources/sfpu_binary_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
            BROADCAST_TYPE(LlkBroadcastType.None_),
            SFPU_BCAST_DIM(LlkBroadcastType.None_),
            SFPU_DST_ROUNDING_MODE(),
            SIGN_MAGNITUDE_FORMAT(False),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(),
            ZERO_POINT(0),
            OUT_ZERO_POINT(0),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            twos_complement=twos_complement,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": formats.input_format.is_32_bit(),
        "disable_format_inference": True,
        "compile_time_formats": True,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).view(input_dimensions)

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    # Without this a row for SfpuAddTopRow would be inert.
    assert_against_contract(
        mathop,
        formats,
        dest_acc,
        golden_tensor,
        res_tensor,
        approx_mode=_APPROX_MODE,
    )


@parametrize(**ADD_TOP_ROW_SWEEP)
def test_eltwise_binary_sfpu_add_top_row(formats, dest_acc, mathop, **run_kwargs):
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    _run_sfpu_add_top_row(formats, dest_acc, mathop, **run_kwargs)


# Both operands of the across-zero pair cycle through the same small signed range.
_ACROSS_ZERO_SPAN = 16  # distinct values per operand
_ACROSS_ZERO_OFFSET = _ACROSS_ZERO_SPAN // 2  # shifts 0..15 to -8..7
# Tile 1 advances once per four lanes, so the 64 written lanes of each face hold distinct pairs.
_ACROSS_ZERO_STRIDE = 4


def _build_add_top_row_int32_across_zero_src():
    """One Int32 tile pair of small values of either sign: tile 0 cycles -8..7 per lane,
    tile 1 cycles the same range once per four lanes, so each face's 64 written lanes hold
    distinct (a, b) pairs. The default sweep draws Int32 from [0, 2**30), where
    sign-magnitude and two's complement are the same bits.
    """
    period = _ACROSS_ZERO_SPAN * _ACROSS_ZERO_STRIDE
    pairs = [
        (
            (i % _ACROSS_ZERO_SPAN) - _ACROSS_ZERO_OFFSET,
            ((i // _ACROSS_ZERO_STRIDE) % _ACROSS_ZERO_SPAN) - _ACROSS_ZERO_OFFSET,
        )
        for i in range(period)
    ]
    return _build_paired_tile_override(pairs, torch.int32)


# mathop and dest_acc stay parametrize axes with one value each: --op selection reads the
# MathOperation from the variant's parameters, and the variant id names the Dest width.
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuAddTopRow],
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_add_top_row_int32_across_zero(formats, dest_acc, mathop):
    """Negative and mixed-sign Int32 operands, which the positive-only sweep never reaches.

    A native Int32 tile reaches Dst as raw two's complement, so a kernel that applied the
    sign-magnitude conversion on load/store fails here on 84 of the 128 written lanes.
    """
    _run_sfpu_add_top_row(
        formats,
        dest_acc,
        mathop,
        src_A_override=_build_add_top_row_int32_across_zero_src(),
        twos_complement=True,
    )


# =============================================================================
# Broadcast kernel
#
# SFPU binary with row/column broadcast (BCAST_COL / BCAST_ROW). Uses its own
# 3-tile kernel source (sources/sfpu_binary_bcast_test.cpp) with a custom init
# and full-tile driver; InstrModLoadStore::DEFAULT works for any float dest
# format (compute is FP32). Unpack is always BroadcastType::NONE; the dim lives
# in dest as SFPU_BCAST_DIM, not unpack BROADCAST_TYPE.
# =============================================================================

_BCAST_BINARY_OPS = {
    MathOperation.SfpuElwadd: torch.add,
    MathOperation.SfpuElwsub: torch.sub,
    MathOperation.SfpuElwmul: torch.mul,
}

BCAST_SWEEP = dict(
    # Only same-format in/out combinations are supported by the broadcast kernel,
    # so `same=True` (a full Cartesian product would also blow past the 100-combo
    # Python test guideline).
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ],
        same=True,
    ),
    bcast_dim=[LlkBroadcastType.Row, LlkBroadcastType.Column],
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)


def _golden_sfpu_binary_bcast(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    bcast_dim: LlkBroadcastType,
    op,
    stimuli_format: DataFormat,
) -> torch.Tensor:
    """Golden for the SFPU bcast kernel (single 32x32 tile): broadcast in row-major space,
    then tilize to the packer's layout. `stimuli_format` drives tilize precision (Float16_b
    for Bfp8_b inputs, since the unpacker converts Bfp8_b -> Float16_b in dest)."""
    a = src_A.flatten()[:1024].reshape(32, 32)
    b = src_B.flatten()[:1024].reshape(32, 32)

    if bcast_dim == LlkBroadcastType.Row:
        b_bcast = b[0].unsqueeze(0).expand_as(b)
    else:
        b_bcast = b[:, 0].unsqueeze(1).expand_as(b)

    golden_rm = op(a, b_bcast.contiguous()).flatten()
    return tilize(golden_rm, stimuli_format=stimuli_format)


def _run_sfpu_binary_bcast(
    formats,
    bcast_dim,
    mathop,
    dest_acc,
    *,
    src_B_override=None,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
    iterations=32,
    approx_mode=ApproximationMode.No,
):
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # The broadcast operand is the one tile in buffer_B; an explicit src_B controls it fully.
    if src_B_override is not None:
        if src_B_override.numel() != src_B.numel():
            raise ValueError("SFPU bcast override must be exactly one tile")
        src_B = src_B_override.to(src_B.dtype).flatten()

    # Bfp8_b stimuli are effectively Float16_b in dest after unpack; golden
    # computes and tilizes at that precision to match.
    golden_format = (
        DataFormat.Float16_b
        if formats.input_format == DataFormat.Bfp8_b
        else formats.input_format
    )
    golden_tensor = _golden_sfpu_binary_bcast(
        src_A, src_B, bcast_dim, _BCAST_BINARY_OPS[mathop], golden_format
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS
    )

    run_types = _resolve_perf_args(is_perf, perf_report, run_types)

    # Only FP32 inputs with dest_acc=Yes take the unpack-to-dest path; all
    # other float formats go through srcA + MATH datacopy into dest.
    unpack_to_dest = unpacks_to_dest(formats.input_format, dest_acc)

    test_config_kwargs = {
        "test_name": "sources/sfpu_binary_bcast_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
            BROADCAST_TYPE(LlkBroadcastType.None_),
            SFPU_BCAST_DIM(bcast_dim),
            # Unread by this kernel. Present so the report columns match sfpu_binary.
            SFPU_DST_ROUNDING_MODE(),
            SIGN_MAGNITUDE_FORMAT(False),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A + tile_cnt_B),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(),
            ZERO_POINT(0),
        ],
        "variant_stimuli": StimuliConfig(
            tilize(src_A, stimuli_format=formats.input_format),
            formats.input_format,
            tilize(src_B, stimuli_format=formats.input_format),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": unpack_to_dest,
        "compile_time_formats": True,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result ({len(res_from_L1)}) and golden ({len(golden_tensor)}) size mismatch"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    # approx_mode unset: this kernel compiles no APPROX_MODE.
    assert_against_contract(mathop, formats, dest_acc, golden_tensor, res_tensor)


@parametrize(**BCAST_SWEEP)
def test_eltwise_binary_sfpu_bcast(
    formats,
    bcast_dim,
    mathop,
    dest_acc,
    **run_kwargs,
):
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    # Mirror sfpu_binary(): on Blackhole, Float16/Float32 inputs require
    # dest_acc=Yes (32-bit dest), so silently upgrade the parametrized value.
    if (
        formats.input_format in [DataFormat.Float16, DataFormat.Float32]
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        dest_acc = DestAccumulation.Yes

    _run_sfpu_binary_bcast(formats, bcast_dim, mathop, dest_acc, **run_kwargs)


_INF = float("inf")
_SIGNED_ZERO_PAIRS = {
    MathOperation.SfpuElwmul: (
        (-3.0, 0.0),
        (3.0, -0.0),
        (-0.0, 1.5),
        (0.0, -1.5),
        (-0.0, -1.5),
        (0.0, 0.0),
        (1e-30, -1e-30),
    ),
    MathOperation.SfpuElwdiv: (
        (0.0, -1.5),
        (-0.0, 1.5),
        (0.0, -_INF),
        (-0.0, _INF),
        (1.5, -_INF),
        (-1.5, _INF),
        (-0.0, -1.5),
        (1.5, _INF),
        (1e-30, -1e30),
    ),
}


@pytest.mark.parametrize("mathop", list(_SIGNED_ZERO_PAIRS), ids=lambda op: op.name)
def test_eltwise_binary_sfpu_fp32_signed_zero(mathop):
    if TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE:
        pytest.skip("the sign of a zero product is kept on Blackhole only")
    src = _build_paired_tile_override(_SIGNED_ZERO_PAIRS[mathop], torch.float32)
    a, b = src[:_ELEMENTS_PER_TILE], src[_ELEMENTS_PER_TILE:]
    res, _ = sfpu_binary(
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        DestAccumulation.Yes,
        mathop,
        src_A_override=src,
        input_dimensions=_EXACT_DIMS,
        check_contract=False,
    )
    got = torch.as_tensor(res, dtype=torch.float32).flatten()[:_ELEMENTS_PER_TILE]
    expected = a * b if mathop == MathOperation.SfpuElwmul else a / b
    assert torch.equal(
        got, expected
    ), f"{got[got != expected][:4].tolist()} != {expected[got != expected][:4].tolist()}"
    wrong_sign = torch.signbit(got) != torch.signbit(expected)
    assert (
        not wrong_sign.any()
    ), f"{mathop.name}: zero sign lost for {list(zip(a[wrong_sign][:4].tolist(), b[wrong_sign][:4].tolist()))}"


# BCAST_COL loads the even columns, so SFPU lanes 1..7 hold the discarded tile columns 2, 4, ..., 14;
# Inf / NaN there must not reach the output. Bfp8_b cannot hold them, so it is not swept.
_BCAST_COL_IGNORED_SWEEP = dict(
    formats=input_output_formats(
        [DataFormat.Float32, DataFormat.Float16_b],
        same=True,
    ),
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)


def _column_vector_with_nonfinite_ignored_columns() -> torch.Tensor:
    """One 32x32 tile: finite column 0, +Inf / NaN / -Inf in discarded columns 2, 10 and 14."""
    gen = torch.Generator().manual_seed(0)
    tile = torch.empty(32, 32, dtype=torch.float32).uniform_(-2.0, 2.0, generator=gen)
    tile[0::3, 2] = float("inf")  # SFPU lane 1
    tile[1::3, 10] = float("nan")  # SFPU lane 5
    tile[2::3, 14] = -float("inf")  # SFPU lane 7
    return tile.flatten()


@parametrize(**_BCAST_COL_IGNORED_SWEEP)
def test_eltwise_binary_sfpu_bcast_col_nonfinite_ignored_columns(
    formats, mathop, dest_acc, **run_kwargs
):
    """Inf / NaN in the ignored columns leaves the output finite; a kernel that zeroes the discarded
    lanes with a multiply turns the whole row NaN (Inf * 0) through the rotate-add broadcast.
    """
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    _run_sfpu_binary_bcast(
        formats,
        LlkBroadcastType.Column,
        mathop,
        dest_acc,
        src_B_override=_column_vector_with_nonfinite_ignored_columns(),
        **run_kwargs,
    )
