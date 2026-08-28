# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict

import pytest
import torch
from conftest import skip_for_quasar
from helpers.chip_architecture import ChipArchitecture
from helpers.data_format_inference import is_format_combination_outlier
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    BinarySFPUGolden,
    BroadcastGolden,
    get_golden_generator,
    quantize_input_to_unpack_format,
)
from helpers.llk_params import BroadcastType as LlkBroadcastType
from helpers.llk_params import DestAccumulation, DestSync, MathOperation, format_dict
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
    runtime,
)
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
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test

# =============================================================================
# Shared skip helpers
# =============================================================================


def _skip_fp32_no_dest_acc(formats, dest_acc):
    """32-bit (Float32) inputs need a 32-bit dest, i.e. dest_acc=Yes."""
    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")


def _skip_bh_float16_no_dest_acc(formats, dest_acc):
    """Blackhole can't run Float16 SFPU input without a 32-bit dest intermediate."""
    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and formats.input_format == DataFormat.Float16
        and dest_acc == DestAccumulation.No
    ):
        pytest.skip(
            "Float16_a isn't supported for SFPU on Blackhole without being converted to 32-bit intermediate format in dest register"
        )


# =============================================================================
# Shared crafted-stimuli helpers
#
# Several predicate/paired ops (mask, isclose, eq/ne, lt/gt/le/ge) need operand
# tiles filled from *different* per-position data, which the default random sweep
# can't express. These builders produce those StimuliSpecs. (logsigmoid also lives
# here, but it is a plain single-distribution spec that never reads in1.)
# =============================================================================

# Number of faces per tile for the [64, 32] two-tile binary harness layout
# (a 32x32 tile is 4 faces of 16x16, and input_dimensions=[64, 32] is 8 faces).
_FACES_PER_TILE = 4
_ELEMENTS_PER_TILE = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM


# Per-op (atol, rtol) overrides for the binary suite, mirroring CUSTOM_TOLERANCES in
# test_eltwise_unary_sfpu.py. `None` keeps the format default (0.05 / 0.05 for the float formats).
#
# Only two ops belong here: their error is a property of the op's own *composition* rather than
# of the stimuli, so it grows with the operands however the domain is drawn. Both were previously
# kept accurate by capping the registry domain instead -- pow at 3, xlogy's x at 4 -- which never
# evaluated the op where it is interesting. Every number below is measured on a Blackhole p150b
# over the widened domains, max across Float16_b and Float32 at dest_acc=Yes, ~32k elements per
# cell. Re-measure before widening either domain further.
#
#   pow -- a**b is exp(b * ln a), so relative error tracks the product handed to the shared
#   exp approximation:
#     A<=3  B<=3  (b*ln a = 3.30)   max_rel  10.00%
#     A<=8  B<=3  (6.24)            max_rel  10.24%
#     A<=8  B<=4  (8.32)            max_rel  13.35%   <- the domain now registered
#     A<=16 B<=4  (11.09)           max_rel  10.34%
#   ~Flat in the operands rather than growing, so the fixed 5% rtol had been capping the domain,
#   not the op; rtol=0.15 clears 13.35% with margin. A<=16 was rejected separately: it drives
#   |golden| to 6.2e4, within 1.06x of Float16's ceiling.
#
#   xlogy -- x * log(y), so *absolute* error scales with x while a fixed atol does not
#   (relative is meaningless: xlogy(0, y) = 0, making any error there infinitely relative):
#     x<=4   max_abs 0.25 (Float16_b) / 0.058 (Float32)
#     x<=8   max_abs 0.50            / 0.116            <- the domain now registered
#     x<=16  max_abs 1.00            / 0.232
#     x<=32  max_abs 2.00            / 0.464
#   Linear in x, matching error ~ x * abs_err(ln y), which is why no fixed atol could hold.
#   Float16_b dominates because at |golden| ~ 72 a bfloat16 ULP is already 0.5 -- mostly output
#   quantization, not the kernel. atol=0.6 covers x<=8 with 20% margin.
BINARY_CUSTOM_TOLERANCES = {
    # Listed per output format only to keep Bfp8_b out of it. pow's error is *relative* -- the
    # measurement above is ~flat in the operands -- so unlike xlogy's absolute error it does not
    # scale with the output format's precision, and the same rtol is the right shape for every
    # float column. Measured: reverting ->Float16 to the 0.05 default fails all 15 of its pow
    # variants on a Wormhole n300, so 0.15 is the op's requirement there too, not a loosening.
    #
    # Bfp8_b is the exception and is deliberately absent: its default rtol is 0.2, so an override
    # of 0.15 would *tighten* it, and at 0.2 a lane that misses the tolerance can still be caught
    # by the block-lattice fallback in passed_test. Falling through keeps both.
    MathOperation.SfpuElwpow: {
        DataFormat.Float32: (None, 0.15),
        DataFormat.Float16_b: (None, 0.15),
        DataFormat.Float16: (None, 0.15),
    },
    # Keyed by *output format*, because the measurement above splits by nearly 5x: applying
    # Float16_b's atol to Float32 would accept five times the error that format was measured to
    # produce. Float16 is measured separately on a Wormhole n300 over x <= 8 and requires 0.0989
    # against a 0.05 format default; 0.12 is that with the same ~20% margin as the other two.
    MathOperation.SfpuXlogy: {
        DataFormat.Float32: (0.14, None),  # 0.116 measured, same ~20% margin as bf16
        DataFormat.Float16_b: (0.6, None),
        DataFormat.Float16: (0.12, None),  # 0.0989 measured
    },
}

# Fallback for an output format the per-format table does not list: no override at all, so
# `helpers/utils.py`'s per-format tolerance applies. Bfp8_b is the only such format left, and
# it does not need one -- its verdict comes from _bfp_block_aware_compare's lattice check
# rather than from a flat atol, and it passes on the default.
#
# Not the widest measured value: an unlisted format sits on its format default (0.05 for Float16,
# 0.1 for Bfp8_b), and handing those 0.6 would loosen them 12x and 6x on columns the measurement
# never covered.
#
# Reading the measurement: passed_test judges with torch.isclose(golden, res, rtol, atol), so the
# bound is atol + rtol * |res| and the atol a format actually needs is max(|g - r| - rtol * |r|),
# not max|g - r| -- which is why the raw figures above sit well above the atols they justify.
_UNLISTED_FORMAT_TOLERANCE = (None, None)


def _custom_tolerances(mathop, output_format):
    """The (atol, rtol) override for *mathop*, per output format where it has one."""
    entry = BINARY_CUSTOM_TOLERANCES.get(mathop)
    if entry is None:
        return (None, None)
    if isinstance(entry, dict):
        return entry.get(output_format, _UNLISTED_FORMAT_TOLERANCE)
    return entry


def _build_paired_tile_override(pairs, dtype):
    """Two-tile raw override from a list of (A, B) pairs: tile 0 holds every A, tile 1
    every B, paired by index (tilize pairs them that way).

    *pairs* is cycled to fill a whole tile rather than zero-padded, so the override
    divides evenly into whatever buffer the driver picks and every element is a pair the
    caller meant to drive. The three edge builders below all need exactly this — an
    interesting pair list is always far shorter than a tile — and differ only in *dtype*.
    """
    if not pairs:
        raise ValueError("_build_paired_tile_override() needs at least one pair")
    a = [pairs[i % len(pairs)][0] for i in range(_ELEMENTS_PER_TILE)]
    b = [pairs[i % len(pairs)][1] for i in range(_ELEMENTS_PER_TILE)]
    return torch.tensor(a + b, dtype=dtype)


def _pair_operand_specs(spec_A, spec_B, input_dimensions):
    """Interleave two per-operand specs across *every* tile pair in the buffer.

    The kernel reads operand 0 from the even tile of each pair and operand 1 from the odd
    one, so per-operand stimuli have to alternate every 4 faces for the whole buffer.
    The list must cover the real tile count: `face_specs` is applied positionally and is
    not cycled, so a short list leaves later pairs with operand 0's distribution on both
    sides. Entries for operand 0's faces stay None to fall through to the base spec.
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

    `size` is whatever the generator passes per face (256 for a 16x16 face), so the
    builders never assume a face size. The ramp repeats 1..8: non-zero everywhere, so
    mask's passthrough is detectable, and of order 1, so the +/-1.0 and +2.0 offsets the
    other operand adds are unambiguous against any rounding.
    """
    positions = torch.arange(size, dtype=torch.float32)
    return positions, 1.0 + (positions % 8)


# =============================================================================
# Which ops take their domain from _OP_DOMAIN_REGISTRY
#
# Every op driven through sfpu_binary() belongs to exactly one of the three sets below,
# and _classify_stimuli_source() enforces that in the driver. The classification is the
# only record of *why* an op is fed what it is fed, so an unclassified op is a coverage
# question nobody asked rather than a default anybody chose.
#
#   _REGISTRY_DOMAIN_OPS            - rerouted onto the op's registered domain.
#   _UNREGISTERED_BINARY_OPS        - no registry entry exists; keeps the format default.
#   _REGISTERED_DEFAULT_STIMULI_OPS - a registry entry exists but is deliberately not
#                                     used here, with the reason recorded per op.
#
# The reroute is float-only on purpose: SfpuElwadd/SfpuElwsub and the shift ops also run
# through test_eltwise_binary_sfpu_int, where a float domain like uniform(-1, 1) would collapse
# to {-1, 0, 1} and gut the int coverage. Ops with crafted stimuli (mask / isclose /
# eq-ne / logsigmoid / shift edge cases) pass their own spec and ignore any default.
# =============================================================================

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
        MathOperation.SfpuLcm,
        MathOperation.SfpuLogsigmoid,
        MathOperation.SfpuMask,
        MathOperation.SfpuMaxInt32,
        MathOperation.SfpuMaxUint32,
        MathOperation.SfpuMinInt32,
        MathOperation.SfpuMinUint32,
        MathOperation.SfpuMulInt32,
        MathOperation.SfpuNeInt,
        MathOperation.SfpuRemainderInt32,
        MathOperation.SfpuRemainderUint32,
        MathOperation.SfpuRsubInt32,
    }
)

# Ops with an _OP_DOMAIN_REGISTRY entry that this suite deliberately does not read, and
# why. Distinct from _UNREGISTERED_BINARY_OPS: there the registry has nothing to offer,
# here it has something we are choosing not to take. Without this set the three shift ops
# sat outside both lists, so the consistency check below passed while saying nothing about
# them — the exact silent-drift hole it exists to close.
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
}


def _assert_domain_sets_consistent():
    """The three stimulus-source sets must be disjoint and consistent with the registry.

    Each half fails quietly otherwise: an op in _REGISTRY_DOMAIN_OPS with no registry
    entry raises deep inside the driver mid-sweep; an op that gains a domain while sitting
    in _UNREGISTERED_BINARY_OPS silently keeps the positive-only default; and an op in no
    set at all keeps the default while looking, to a reader of this file, like it was never
    considered.

    Totality — every op reaching the driver is classified — cannot be asserted here,
    because the set of ops this suite drives is only known once pytest has collected the
    parametrize lists. _classify_stimuli_source() enforces it in the driver instead.

    What *can* be asserted here is family totality: every op sfpu_domains records as a
    binary SFPU op is either classified or declared as not using the shared driver. That
    catches an op added to the family and to no list at all, which the driver-side check
    cannot see until something drives it.
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

    Raises for an op in none of the three declared sets. That is the case the collection
    -time assertion cannot see: adding an op to a parametrize list is enough to drive it,
    and an unclassified op would silently inherit generate_stimuli's positive-only
    uniform(0.1, 1.1) — which is finding #1 of the coverage audit, reintroduced one op at
    a time.
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
    # mask zeroes data (in0) where mask (in1) is 0. Data and mask are separate tiles: keep
    # data strictly non-zero (1..8) and zero ~1/3 of the mask, so a passthrough kernel fails.
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

    Independent random draws land arbitrarily close together, and a near-tie the kernel
    and the total-order golden round differently reads as a failure. The gaps here are
    +/-1.0 against operands of order 1, far wider than any rounding, so every element's
    verdict is unambiguous. The exact-equality third is the point: it is the only input
    where lt/gt and le/ge disagree, and a random sweep essentially never produces it.
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
    # logsigmoid(x) = -softplus(-x). in1 (exp(-x)) is only read in the x > 4 branch, so
    # restrict x to [-8, 3.9] (never uses in1) and sweep the passthrough (x < -4) and
    # polynomial (-4 < x < 4) branches. The distribution is invoked per 16x16 face (size 256).
    def dist(size, dtype, generator):
        return torch.linspace(-8.0, 3.9, size).to(dtype)

    return StimuliSpec(distribution=dist, seed=0)


# =============================================================================
# Shared driver
#
# Every test below (except add_top_row and the separate broadcast kernel) runs
# through sfpu_binary(): it builds the [64, 32] two-tile stimuli, computes the
# golden per tile-pair, and drives sources/sfpu_binary_test.cpp.
# =============================================================================


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
):
    """*unspecified_nonfinite_sign* compares a non-finite result by magnitude only.

    For the one case where the sign genuinely is not specified: a NaN the kernel *generated*,
    packed as a signed infinity through a pipeline too narrow to hold it, on Wormhole, where
    `SFPMAD.md` says that NaN's sign "might or might not be set". Better than withdrawing the
    variant -- the magnitude, the finiteness and every finite lane stay checked, so a kernel
    returning a finite value, a zero or a NaN where an infinity is due still fails.

    Scoped per lane, from the mask the golden records while the NaN is still a NaN, because a
    tensor holds both kinds of non-finite at once: `specials_in` drives `inf - inf`, whose sign
    the ISA leaves open, alongside `0 - (-inf)`, whose `+inf` IEEE fully specifies. Same per-lane
    scoping test_sfpu_reduce uses.
    """

    # Seed the draw so the stimuli are identical run to run. Nothing below sets a seed,
    # and an unseeded redraw makes a variant sitting near its tolerance pass or fail
    # depending on the draw -- an unreproducible failure. eltwise_unary_sfpu seeds too.
    torch.manual_seed(0)

    # FP32 destination tiles occupy twice the register space. Keep four full destination
    # blocks for those formats and four blocks of eight tiles for the remaining formats.
    if input_dimensions is None:
        input_dimensions = (
            [128, 128] if formats.input_format.is_32_bit() else [256, 128]
        )

    # Per-operand domains. Both operands live in buffer_A (even tile = in0, odd tile = in1),
    # so there is no spec_B knob in generate_stimuli — the two specs are interleaved into one
    # spec's face_specs here, where the tile count is known.
    #
    # Ops in _REGISTRY_DOMAIN_OPS take their domain from _OP_DOMAIN_REGISTRY, which is what
    # makes the registered undefined-range holes (SfpuElwdiv divisor, SfpuXlogy B,
    # SfpuElwpow A) reachable. Unlike the unary sweep, most ops here are not registered; a
    # missing entry falls back to generate_stimuli's format default rather than raising.
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
        override = src_A_override.to(src_A.dtype).flatten()
        if src_A.numel() % override.numel() != 0:
            raise ValueError(
                "SFPU binary override must contain a whole number of tile pairs"
            )
        src_A = override.repeat(src_A.numel() // override.numel())

    # generate_stimuli round-trips Bfp4_b and Bfp2_b stimuli through their pack/unpack
    # quantization but not Bfp8_b: the Bfp8_b format default only ever draws values that
    # are already representable (integer 0..2 plus k/16), so nothing needed it. A registry
    # domain -- or an src_A_override -- draws arbitrary values, and then the device sees
    # Bfp8_b-quantized operands while the golden still sees the unrounded bf16 originals.
    # That is the same golden/hardware split Phase 0 fixed inside UnarySFPUGolden. Quantize
    # the golden's copy only, before broadcast: src_A keeps the unrounded values because
    # the packer applies exactly this rounding when it writes the buffer to L1.
    golden_src = src_A
    if formats.input_format == DataFormat.Bfp8_b:
        golden_src = quantize_input_to_unpack_format(golden_src, DataFormat.Bfp8_b)

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

    # ONLY Blackhole needs this for some reason
    #
    # Hoisted above the golden call, which now models the Dest width from the *effective*
    # dest_acc -- computing the golden first would model a 16-bit Dest for a variant that runs
    # with a 32-bit one here. Nothing in between reads dest_acc, so the move is behaviour-
    # preserving for every existing caller.
    if (
        formats.input_format in [DataFormat.Float16, DataFormat.Float32]
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        dest_acc = DestAccumulation.Yes

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_format = (
        DataFormat.Float16_b
        if formats.input_format == DataFormat.Bfp8_b
        else formats.input_format
    )
    elements_per_pair = 2 * 32 * 32
    golden_chunks = []
    generated_nan_chunks = []
    for offset in range(0, golden_src.numel(), elements_per_pair):
        chunk = generate_golden(
            mathop,
            golden_src[offset : offset + elements_per_pair],
            0,
            1,
            0,
            32,
            [64, 32],
            golden_format,
            dest_acc=dest_acc,
            output_format=formats.output_format,
            collect_generated_nan=unspecified_nonfinite_sign,
        )
        # Asked of the return value rather than of the build mode: DummyGoldenGenerator stands in
        # for the golden during --compile-producer and returns a bare tensor whatever it is asked
        # for, and that phase skips before the comparison below anyway.
        if isinstance(chunk, tuple):
            chunk, generated_nan = chunk
            generated_nan_chunks.append(generated_nan)
        golden_chunks.append(chunk.flatten())
    golden_tensor = torch.cat(golden_chunks)

    bcast = broadcast_type if broadcast_type else LlkBroadcastType.None_

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS
    )

    configuration = TestConfig(
        "sources/sfpu_binary_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(),
            BROADCAST_TYPE(bcast),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
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
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    # Per-op tolerances, for the two ops whose error is a property of the op's own
    # composition rather than of the stimuli, and per output format where the error splits by
    # format. See BINARY_CUSTOM_TOLERANCES.
    custom_atol, custom_rtol = _custom_tolerances(mathop, formats.output_format)

    if unspecified_nonfinite_sign and generated_nan_chunks:
        # Clear the sign only on the lanes that held a generated NaN *and* where both sides are
        # non-finite. A golden +inf against a hardware 5.0 still compares +inf vs 5.0 and still
        # fails; a golden +inf against a hardware NaN likewise, because abs() leaves a NaN a NaN
        # and passed_test's both-NaN clause needs both. So this excuses one bit on the lanes the
        # ISA declines to pin, and nothing else anywhere.
        unspecified = (
            torch.cat(generated_nan_chunks)
            & ~torch.isfinite(golden_tensor)
            & ~torch.isfinite(res_tensor)
        )
        golden_tensor = torch.where(unspecified, golden_tensor.abs(), golden_tensor)
        res_tensor = torch.where(unspecified, res_tensor.abs(), res_tensor)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
    ), "Assert against golden failed"


# =============================================================================
# Float ops
# =============================================================================


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
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
        # Eq/Ne and Lt/Gt/Le/Ge are excluded from this *random* sweep: independent draws
        # are never equal (the Eq/Ne golden collapses to a constant) and near-ties that
        # the kernel and the total-order golden round differently read as failures. They
        # are covered with crafted paired stimuli by test_eltwise_binary_sfpu_eq_ne and
        # test_eltwise_binary_sfpu_float_comparison below.
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_float(
    formats,
    dest_acc,
    mathop,
    bcast_dim,
):
    _skip_fp32_no_dest_acc(formats, dest_acc)
    _skip_bh_float16_no_dest_acc(formats, dest_acc)

    # POW/XLOGY are only covered on the float formats: under Bfp8_b the coarse
    # quantization pushes small operands to values that produce -inf/NaN (log/pow),
    # so Bfp8_b coverage for these ops is intentionally skipped.
    if formats.input_format == DataFormat.Bfp8_b and mathop in (
        MathOperation.SfpuElwpow,
        MathOperation.SfpuXlogy,
        MathOperation.SfpuLogaddexp,
        MathOperation.SfpuLogaddexp2,
    ):
        pytest.skip("Bfp8_b is not supported for POW/XLOGY/LOGADDEXP/LOGADDEXP2 coverage")

    if bcast_dim == LlkBroadcastType.Row and (
        dest_acc == DestAccumulation.Yes
        or is_format_combination_outlier(
            formats.input_format, formats.output_format, dest_acc
        )
    ):
        pytest.skip(
            "Row broadcast with FP32 dest: B2D datacopy uses MOVB2D which can't handle FP32 dest format conversion"
        )

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        broadcast_type=bcast_dim,
    )


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
        ]
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_div(formats, dest_acc):
    # DIV routes through the dedicated production kernel (calculate_sfpu_binary_div);
    # split out from the float sweep since the reciprocal path is precision-sensitive.
    _skip_fp32_no_dest_acc(formats, dest_acc)
    _skip_bh_float16_no_dest_acc(formats, dest_acc)

    sfpu_binary(
        formats,
        dest_acc,
        MathOperation.SfpuElwdiv,
        broadcast_type=LlkBroadcastType.None_,
    )


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
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
def test_eltwise_binary_sfpu_float_extended(formats, dest_acc, mathop):
    # max/min (SFPSWAP) and fmod/remainder (fp32 reciprocal) binary kernels with no
    # dedicated production BinaryOp; driven through the same in-DST harness as add/sub.
    _skip_fp32_no_dest_acc(formats, dest_acc)
    _skip_bh_float16_no_dest_acc(formats, dest_acc)

    # fmod/remainder divide by b via a reciprocal; Bfp8_b's coarse quantization blows up
    # the quotient for small divisors (mirrors the pow/xlogy Bfp8_b skip above).
    if formats.input_format == DataFormat.Bfp8_b and mathop in (
        MathOperation.SfpuBinaryFmod,
        MathOperation.SfpuBinaryRemainder,
    ):
        pytest.skip("Bfp8_b is not supported for fmod/remainder coverage")

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        broadcast_type=LlkBroadcastType.None_,
    )


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    mathop=[MathOperation.SfpuMask],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_mask(formats, dest_acc, mathop):
    # float mask: data at tile0, mask at tile1. Output is data where mask != 0, else 0.
    # Crafted stimuli so the mask carries real zeros.
    _skip_fp32_no_dest_acc(formats, dest_acc)

    # One tile pair only, unlike every other op here. calculate_mask hard-codes its
    # operands -- data at dst_reg[0], mask at dst_reg[32], result in place -- and ignores
    # the forwarded dst indices, so only the in0=0/in1=1/out=0 placement computes anything
    # (see calculate_mask_binary in helpers/include/sfpu_test_helpers.h). On a larger
    # buffer the kernel's `tile += 2` loop would mask tile 0 of each block repeatedly and
    # pack tiles 2/4/6 out unmasked, against a golden that masks every pair. [64, 32] is
    # 2 tiles = 1 block = 1 pair, so the only pair driven is the one the adapter supports.
    spec_A, spec_B = _mask_stimuli_specs()
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        broadcast_type=LlkBroadcastType.None_,
        spec_A=spec_A,
        spec_B=spec_B,
        input_dimensions=[64, 32],
    )


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    mathop=[MathOperation.SfpuAtan2],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_atan2(formats, dest_acc, mathop):
    # atan2(y, x): y = tile0, x = tile1. Signed [-5, 5] gives mixed signs so all quadrants
    # (and the |y|>=|x| / x<0 branches) are exercised; minimax approximation matched under PCC.
    _skip_fp32_no_dest_acc(formats, dest_acc)

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0),
    )


@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
    ),
    mathop=[MathOperation.SfpuElwEq, MathOperation.SfpuElwNe],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_eq_ne(formats, dest_acc, mathop):
    # Eq/Ne(a, b) with a = tile0, b = tile1. Crafted paired stimuli give a non-constant 0/1
    # golden so the equal branch is exercised (the default random sweep never is).
    _skip_fp32_no_dest_acc(formats, dest_acc)
    _skip_bh_float16_no_dest_acc(formats, dest_acc)

    spec_A, spec_B = _eq_ne_stimuli_specs()
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B)


@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
    ),
    mathop=[
        MathOperation.SfpuElwLt,
        MathOperation.SfpuElwGt,
        MathOperation.SfpuElwLe,
        MathOperation.SfpuElwGe,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_float_comparison(formats, dest_acc, mathop):
    # lt/gt/le/ge(a, b) with a = tile0, b = tile1. Crafted so a third of the elements are
    # exactly equal and the rest differ by +/-1.0: the tie is what distinguishes the strict
    # comparisons from the non-strict ones, and the wide gaps keep every other element's
    # verdict independent of rounding. See _comparison_stimuli_specs.
    _skip_fp32_no_dest_acc(formats, dest_acc)
    _skip_bh_float16_no_dest_acc(formats, dest_acc)

    spec_A, spec_B = _comparison_stimuli_specs()
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    mathop=[MathOperation.SfpuIsclose],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_isclose(formats, dest_acc, mathop):
    # isclose(a, b) = |a - b| <= atol + rtol*|b|, a = tile0, b = tile1. torch default
    # tolerances (fixed in the C++ dispatch); crafted stimuli give a non-constant 0/1 mix.
    _skip_fp32_no_dest_acc(formats, dest_acc)

    spec_A, spec_B = _isclose_stimuli_specs()
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    mathop=[MathOperation.SfpuLogsigmoid],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_logsigmoid(formats, dest_acc, mathop):
    # logsigmoid(x) with x = tile0. Piecewise poly/passthrough approximation matched under
    # PCC; x swept over [-8, 3.9]. The x > 4 (-exp(-x)) branch needs a device-computed
    # exp(-x) operand the shared harness can't provide, left to a future driver.
    _skip_fp32_no_dest_acc(formats, dest_acc)

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        spec_A=_logsigmoid_stimuli_spec(),
    )


# =============================================================================
# Integer ops
# =============================================================================


@parametrize(
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
def test_eltwise_binary_sfpu_int(
    formats,
    dest_acc,
    mathop,
):
    # The random half of the Int32 coverage. This variant takes generate_stimuli's integer
    # default, uniform(0, INT32_MAX // 2 - 1) -- positive-only and tie-free, so it cannot tell
    # SfpuElwLe from SfpuElwLt. See the two tests below for the rest.
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
    )


# The four ordered Int32 comparisons, which are the same MathOperation members the float
# comparison sweep drives. sfpu_operations.h routes them to calculate_binary_comp_int32 when
# MATH_FORMAT is Int32 -- subtract, fold the sign -- so this is a different kernel from the fp32
# two-vector compare, reached through the same enum entry.
#
# These are also the kernel the five Quasar-only `*Int` members reach; see the alias guard in
# test_sfpu_domains (test_quasar_int_binary_members_alias_covered_kernels) for why coverage audit
# section 4.5 lists them as untested and the kernels are not.
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

    `a == b` is the *only* input on which lt/gt disagree with le/ge, so without it a comparator
    with its tie inverted passes the whole integer sweep. Measured on the default integer spec,
    uniform(0, INT32_MAX // 2 - 1), a 1024-element draw contained 0 ties and 0 negatives.

    Reuses the float sweep's three-way builder: a third of the elements are exactly equal and the
    rest differ by +/-1, an exact gap on an integer axis. Same shape as
    test_eltwise_binary_sfpu_eq_ne_int, which had this for eq/ne already.
    """
    spec_A, spec_B = _comparison_stimuli_specs()
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B)


def _int_comparison_negative_spec():
    """Paired Int32 stimuli that straddle zero, including a tie at zero.

    The kernel normalises to LT(a, b) by computing `a - b` and reading the sign, so operands of
    opposite sign are the case the fold exists for, and the random draw never produces one.
    Values stay small (|v| <= 8) so `a - b` cannot overflow -- overflow is what
    test_eltwise_binary_sfpu_int_extremes drives deliberately, and mixing the two would leave a failure
    with two candidate causes.

    twos_complement=True is required, not incidental: sign-magnitude L1 encoding cannot round-trip
    a negative through Dst -- the same delivery limitation that made the unary RightShift sweep
    positive-only. test_eltwise_binary_sfpu_rsub_int32 takes the same route.
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

    The float sweep was fixed to draw from the op's signed domain; the integer default was left at
    uniform(0, INT32_MAX // 2 - 1), so a comparator that mishandled operand sign entirely would
    pass every other Int32 test in this file.
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


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[
        MathOperation.SfpuBitwiseAnd,
        MathOperation.SfpuBitwiseOr,
        MathOperation.SfpuBitwiseXor,
    ],
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_bitwise(formats, dest_acc, mathop):
    # int32 bitwise AND/OR/XOR: exact on the full default int range.
    sfpu_binary(formats, dest_acc, mathop)


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


@parametrize(
    mathop=list(_INT_BINARY_STIMULI),
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_int_uniform(mathop, dest_acc):
    int_format = DataFormat.UInt32 if mathop in _UINT32_BINARY_OPS else DataFormat.Int32
    formats = InputOutputFormat(int_format, int_format)
    low, high = _INT_BINARY_STIMULI[mathop]
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=low, high=high),
    )


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuRsubInt32],
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_rsub_int32(formats, dest_acc, mathop):
    sfpu_binary(formats, dest_acc, mathop, twos_complement=True)


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuEqInt, MathOperation.SfpuNeInt],
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_eq_ne_int(formats, dest_acc, mathop):
    # int32 eq/ne via calculate_binary_eq_int (exact 0/1 over the raw INT32 dest bits).
    # Reuse the paired eq/ne stimuli so ~50% of positions compare equal — the equal branch
    # a plain random int sweep would essentially never hit.
    spec_A, spec_B = _eq_ne_stimuli_specs()
    sfpu_binary(formats, dest_acc, mathop, spec_A=spec_A, spec_B=spec_B)


# =============================================================================
# Integer shift edge cases
#
# Deterministic edge-case coverage for the integer shift ops: shift amounts
# outside [0, 31] -> 0, arithmetic right-shift sign-extends, negatives shift
# correctly. INT32_MIN is excluded (sign-magnitude Dst can't represent -2^31);
# see the xfail test below and docs/SFPU_INT32_SHIFT.md.
# =============================================================================

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
    -0x80000000,  # INT32_MIN (filtered out per-op; see _build_shift_edge_case_src)
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
    per-element shift amounts (tilize pairs them by index). Walks the cartesian product of
    interesting (value, shift) pairs; pairs touching INT32_MIN are dropped (sign-magnitude
    Dst can't represent -2^31)."""
    pairs = [
        (v, s)
        for v, s in itertools.product(_SHIFT_EDGE_VALUES, _SHIFT_EDGE_AMOUNTS)
        if v != _INT32_MIN and _shift_reference(mathop, v, s) != _INT32_MIN
    ]
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
    request,
    formats,
    dest_acc,
    mathop,
):
    if TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE:
        # xfail rather than skip so a Blackhole kernel port surfaces as XPASS instead of
        # staying silently green-by-omission. The bug is an external dependency of this
        # suite, not a task in it, but a skip cannot tell us when it is fixed.
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Blackhole shift kernels (left / arithmetic right / logical right) "
                "are unmigrated TTI microcode whose predicated out-of-range/sign handling "
                "breaks under INT32_2S_COMP for negative operands, so all three diverge "
                "from the two's-complement golden. See docs/SFPU_INT32_SHIFT.md.",
                strict=False,
            )
        )

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        src_A_override=_build_shift_edge_case_src(mathop),
    )


# =============================================================================
# Deliberate edge values (Phase 4, binary side)
#
# _build_shift_edge_case_src above is the shape this generalizes: walk the cartesian
# product of both operands' interesting values into a two-tile override tensor, which
# tilize pairs by index. The product is the point — a divisor of 0 against a positive, a
# negative and a zero numerator are three different cases, and element-wise pairing would
# test one of them.
#
# Values come from sfpu_domains.edge_pair_values(), so the ops enrol by having a registered
# singularity rather than by being listed here.
# =============================================================================


# The edge-pair probe is partitioned before it is driven, and the partition is what makes
# the xfails below mean anything. One tensor holding every pole of an op mixes failure
# classes that have nothing to do with each other: div's lost zero sign is documented
# Wormhole SFPMAD behaviour that Blackhole is documented to fix, while div(0, 0) returning
# inf is the kernel's own reciprocal composition and fixed nowhere. Bundled into one
# variant, one xfail covers both — so the zero sign improving on Blackhole still reports
# XFAIL rather than the XPASS it was recorded to produce, and a *new* mismatch anywhere in
# the tensor is invisible for as long as either known one survives.
#
# Classified by what the golden says the answer is, rather than by a per-op predicate on
# the operands, because that is where the classes actually live and the golden is the
# authority on it: `fmod(-2, +1/64)` is a negative-zero case and `fmod(+2, -1/64)` is not,
# which no reading of the operand signs gets right.
_EDGE_CLASS_BOTH_ZERO = "both_zero"
_EDGE_CLASS_NAN = "nan_golden"
_EDGE_CLASS_NEGATIVE_ZERO = "negative_zero_golden"
_EDGE_CLASS_ORDINARY = "ordinary"

# Cat B: a non-finite *operand*, as opposed to a non-finite answer. Its own class because the
# existing four classify by what the golden *answers*, and on that axis a NaN input and `x % 0`
# land in the same bucket despite being IEEE propagation and the kernel's own composition
# respectively -- one xfail over two causes.
_EDGE_CLASS_SPECIALS_IN = "specials_in"

# Order is documentation, not mechanism: whichever class comes first builds the shared ELF for the
# compile-producer pass (conftest._collapse_runtime_only_variants keeps one item per compile key),
# and the test body guards against an empty or gated representative starving the others of a
# binary. No class is non-empty for every op, which is what that guard is for.
#
# 0/0, xlogy(0, 0) and 0**0 are indeterminate forms produced by the kernel's own composition (a
# reciprocal, an exp(b·ln a)), a different cause from x % 0 even where both goldens are NaN.
_EDGE_CLASSES = (
    _EDGE_CLASS_ORDINARY,
    _EDGE_CLASS_BOTH_ZERO,
    _EDGE_CLASS_NAN,
    _EDGE_CLASS_NEGATIVE_ZERO,
    _EDGE_CLASS_SPECIALS_IN,
)


def _classify_edge_pair(mathop, a, b):
    """Which failure class the pair (*a*, *b*) belongs to for *mathop*."""
    # Tested before the others because it is a property of the *input*, and the remaining classes
    # are properties of the output. A NaN operand produces a NaN answer, so without this first
    # every cat-B pair would be filed as nan_golden alongside `x % 0`.
    if not (math.isfinite(a) and math.isfinite(b)):
        return _EDGE_CLASS_SPECIALS_IN
    if a == 0.0 and b == 0.0:
        return _EDGE_CLASS_BOTH_ZERO

    # Instantiate BinarySFPUGolden directly rather than through get_golden_generator: the harness
    # swaps in a DummyGoldenGenerator during --compile-producer, and that stub has no `ops`
    # mapping. This runs at *stimulus-build* time, which happens in both phases, so it cannot use
    # the proxy. Same fix as helpers/compressed_utils.py's matmul golden.
    result = float(BinarySFPUGolden().ops[mathop](torch.tensor(a), torch.tensor(b)))
    if math.isnan(result):
        return _EDGE_CLASS_NAN
    if result == 0.0 and math.copysign(1.0, result) < 0.0:
        return _EDGE_CLASS_NEGATIVE_ZERO
    return _EDGE_CLASS_ORDINARY


def _edge_pairs_for_class(mathop, formats, edge_class, dest_acc, specials=False):
    """The operand pairs of *edge_class* for this op and pipeline.

    Extracted so the override builder and the generated-NaN predicate below select from the same
    list -- a second copy of this filter is how the gate and the stimulus would come to disagree
    about which pairs a class contains.

    *dest_acc* sizes the ULP steps around each pole: at dest_acc=No the DEST is 16-bit, so
    a probe stepped by an fp32 ULP lands back on the pole it was straddling. See
    sfpu_domains.probe_spacing_format().
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


# The ops this suite drives on an integer format. This sweep is float — its format axis is
# Float16_b/Float32 and the override built above carries float values — so an integer op
# that gains a singularity belongs in test_eltwise_binary_sfpu_int_extremes or the shift edge tests
# rather than here. Assembled from the lists that already drive them, so the two cannot
# disagree.
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

# Derived rather than listed, which is what the section header above promises: an op joins
# this sweep by gaining an _OP_SINGULARITIES entry in sfpu_domains. Today that resolves to
# div and fmod/remainder (which divide by B), xlogy (log of B) and pow (exp(B·ln A)). The
# two intersections are what this sweep can actually drive — the same table holds the unary
# poles (Reciprocal, Log, Asin, ...), and _CLASSIFIED_STIMULI_OPS is the declared set of ops
# reaching sfpu_binary().
# An op joins by gaining *either* a registered singularity or a cat-B entry, which keeps this a
# derivation rather than a list. The cat-B half matters for the 16 float ops with no pole: `add`,
# `sub`, `mul`, `max`, `min` and the six comparisons are smooth everywhere, so
# ops_with_singularity() alone can never collect them.
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


# What driving the poles found on Wormhole, cross-checked against tt-isa-documentation.
# One of the two causes is documented hardware behaviour; the other is not.
#
# DOCUMENTED, and expected to XPASS on Blackhole:
#
#   The sign of a zero result is lost -- div(0, -x) returns +0.0 where IEEE gives -0.0,
#   fmod/remainder likewise for a negative divisor, as does xlogy(0, tiny). This is SFPMAD,
#   which all of these ops are built on:
#     Wormhole  — "If the output (before rounding) is denormal or negative zero, it'll be
#                  flushed to positive zero."          (WormholeB0/.../SFPMAD.md)
#     Blackhole — "If the output (after rounding) is denormal, it'll be flushed to
#                  sign-preserved zero."               (BlackholeA0/.../SFPMAD.md)
#   and Blackhole's page lists "improved edge-case handling of NaNs and of negative zero"
#   among its upgrades over Wormhole. A documented Wormhole limitation that Blackhole is
#   documented to fix, so the xfails below are non-strict precisely to report XPASS there.
#
# RETRACTED — "0/0 and x%0 return inf where IEEE says nan" was the pack path, not a kernel:
#
#   The kernels return a genuine NaN. BinarySFPUGolden did not model the store to Dest or the pack
#   out of it, so on a pipeline too narrow to hold a NaN the packer's substituted infinity
#   (SFPSTORE: "NaN is also converted to infinity") read as the kernel having produced one.
#   Measured on a Wormhole n150 once the golden modelled both steps: div(0, 0), xlogy(0, 0),
#   fmod(x, 0) and remainder(x, 0) all PASS wherever a NaN reaches L1 and diverge only where
#   nan_survives_to_l1() is False -- a statement about the pipeline, not the arithmetic.
#
#   What is left on the narrowing cells is the *sign* of the substituted infinity: canonical-
#   positive on Blackhole by specification, explicitly unspecified on Wormhole. So these classes
#   are asserted on Blackhole and gated off on Wormhole by generated_nan_sign_is_asserted().
#
#   The finite poles agreed all along (div(-2, ±1/64) = ∓128, every ±inf lines up).
#
# Those groups are the classes the probe partitions into, and _EDGE_CLASS_NEGATIVE_ZERO -- the
# documented one -- is now the only class with an entry left below. _EDGE_CLASS_BOTH_ZERO
# (indeterminate forms, 0**0) and _EDGE_CLASS_NAN (x % 0) were emptied by the retraction above
# and by the pow fix, so they are asserted rather than tolerated. _EDGE_CLASS_ORDINARY holds
# everything that agreed -- ±inf poles, finite quotients, exact remainders -- likewise
# asserted, which is only possible now it does not share a tensor with the others.
#
# Non-strict xfails per Phase 0's approximate-exp precedent, so a case still executes and reports
# XPASS if behaviour changes; enumerated per (input, output, dest_acc) rather than by predicate so
# a combination drifting in or out shows up here. Keyed by (op, edge class): a class XPASSing
# across the board loses its entry, one XPASSing on Blackhole alone becomes arch-gated.
_BINARY_EDGE_COMBINATIONS = {
    MathOperation.SfpuElwdiv: (
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float16_b, DataFormat.Float32, DestAccumulation.No),
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.Yes),
    ),
    MathOperation.SfpuXlogy: (
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float16_b, DataFormat.Float32, DestAccumulation.No),
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.Yes),
    ),
    MathOperation.SfpuBinaryFmod: (
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float16_b, DataFormat.Float32, DestAccumulation.No),
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.Yes),
    ),
    MathOperation.SfpuBinaryRemainder: (
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float16_b, DataFormat.Float32, DestAccumulation.No),
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.Yes),
    ),
}

_ZERO_SIGN_ISA_NOTE = (
    "the lost zero sign is documented Wormhole SFPMAD behaviour ('flushed to positive "
    "zero'); Blackhole is documented to preserve it, so expect XPASS there"
)

# Which classes of each op are expected to diverge, and why — one reason per class, so the
# xfail says what it is waiting for. A class absent from an op's dict is asserted to pass:
# the ±inf poles, the finite quotients and the exact remainders all agreed on Wormhole.
_BINARY_EDGE_REASON = {
    MathOperation.SfpuElwdiv: {
        _EDGE_CLASS_NEGATIVE_ZERO: f"div(0, -x) returns +0.0, not -0.0 "
        f"({_ZERO_SIGN_ISA_NOTE}).",
    },
    MathOperation.SfpuXlogy: {
        _EDGE_CLASS_NEGATIVE_ZERO: f"xlogy(0, tiny) returns +0.0, not -0.0 "
        f"({_ZERO_SIGN_ISA_NOTE}).",
    },
    MathOperation.SfpuBinaryFmod: {
        _EDGE_CLASS_NEGATIVE_ZERO: f"fmod loses the sign of a zero result "
        f"({_ZERO_SIGN_ISA_NOTE}).",
    },
    MathOperation.SfpuBinaryRemainder: {
        _EDGE_CLASS_NEGATIVE_ZERO: f"remainder loses the sign of a zero result "
        f"({_ZERO_SIGN_ISA_NOTE}).",
    },
}

# Deleted rather than kept: the _EDGE_CLASS_BOTH_ZERO entries for SfpuElwdiv, SfpuXlogy,
# SfpuBinaryFmod and SfpuBinaryRemainder, and the _EDGE_CLASS_NAN entries for the latter two. All
# six recorded "returns inf where IEEE says nan", which the retraction above shows to be the pack
# substitution rather than the arithmetic. They are not replaced by xfails on the narrowing cells
# either -- generated_nan_sign_is_asserted() gates those off on Wormhole.
#
# Also deleted: the _EDGE_CLASS_BOTH_ZERO entry for SfpuElwpow, which recorded 0**0 returning 0
# against a golden 1 -- Wormhole's reading; Blackhole returned +inf for it.
# calculate_sfpu_binary_power now ends in an IEEE pow(x, 0) == 1 guard (see the kernel comment for
# the NaN-predicate root cause), so both_zero asserts 0**0 and 0**-0.0 instead. The -0.0 exponent
# is a committed Operand.B edge in sfpu_domains._OP_OPERAND_EDGE_POINTS; without it this class
# would pass against a kernel that dropped setsgn.

# No op may claim a divergence without a combination list to apply it to, and none may
# list combinations with nothing to apply them to.
assert set(_BINARY_EDGE_REASON) == set(_BINARY_EDGE_COMBINATIONS), (
    "_BINARY_EDGE_REASON and _BINARY_EDGE_COMBINATIONS disagree on which ops diverge: "
    f"{set(_BINARY_EDGE_REASON) ^ set(_BINARY_EDGE_COMBINATIONS)}"
)
assert all(
    cls in _EDGE_CLASSES for classes in _BINARY_EDGE_REASON.values() for cls in classes
), "_BINARY_EDGE_REASON names an edge class that _classify_edge_pair never returns"

# Edge classes whose divergence is a *Wormhole* limitation, so on Blackhole the case is
# asserted rather than tolerated.
#
# The SFPMAD negative-zero split quoted above, confirmed: measured on a Blackhole p150b, the
# negative-zero class XPASSed on all 16 cells it is claimed for (div, xlogy, fmod and remainder,
# at both dest_acc values) and nothing else XPASSed. So a zero result's sign is *checked* there
# and a regression fails rather than returning to XFAIL. The indeterminate-form classes
# (both_zero, nan_golden) are not gated here -- see the retraction above
# _BINARY_EDGE_COMBINATIONS; what remains of them on Wormhole is handled per lane by
# generated_nan_sign_is_asserted().
_WORMHOLE_ONLY_EDGE_CLASSES = frozenset({_EDGE_CLASS_NEGATIVE_ZERO})


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    mathop=_BINARY_EDGE_OPS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    # runtime(): the class selects which values go in the override tensor and nothing
    # else, so all four share one ELF instead of compiling the same kernel four times.
    edge_class=runtime(list(_EDGE_CLASSES)),
)
def test_eltwise_binary_sfpu_edges(request, formats, dest_acc, mathop, edge_class):
    """Drive one class of each binary op's registered pole against its counterparts.

    One variant per (op, class) rather than per op: see the comment above _EDGE_CLASSES for
    why a single tensor holding every pole cannot report which behaviour changed.
    """
    _skip_fp32_no_dest_acc(formats, dest_acc)
    _skip_bh_float16_no_dest_acc(formats, dest_acc)

    # A Wormhole-only class is asserted on Blackhole, not tolerated — see
    # _WORMHOLE_ONLY_EDGE_CLASSES for the measurement that established which those are.
    arch_fixed = (
        edge_class in _WORMHOLE_ONLY_EDGE_CLASSES
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    )

    reason = _BINARY_EDGE_REASON.get(mathop, {}).get(edge_class)
    if (
        reason is not None
        and not arch_fixed
        and (
            (formats.input_format, formats.output_format, dest_acc)
            in _BINARY_EDGE_COMBINATIONS[mathop]
        )
    ):
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=False))

    # Cat B. Two independent gates, both must pass: BINARY_SPECIALS_READY_OPS says this op's
    # *golden* defines an answer for a non-finite operand, specials_safe() says this *pipeline*
    # delivers one intact. Neither implies the other.
    #
    # dest_acc as passed, which is right on Wormhole and conservative on Blackhole: sfpu_binary()
    # promotes it to Yes there, and promotion only ever *widens* Dest.
    specials = mathop in BINARY_SPECIALS_READY_OPS and specials_safe(
        formats.input_format, formats.output_format, dest_acc
    )

    pairs = _edge_pairs_for_class(
        mathop, formats, edge_class, dest_acc, specials=specials
    )

    if not pairs and TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        # The compile-producer pass must not skip on a runtime-only axis. `edge_class` is a
        # runtime() axis, so conftest._collapse_runtime_only_variants keeps one item per compile
        # key and *that* item builds the ELF all classes share; a skip here leaves the others
        # running against a binary that was never built, which presents as TENSIX TIMED OUT rather
        # than as a skip.
        #
        # The ELF depends only on the compile-time axes (op, formats, dest_acc), never on which
        # values go in the tensor, so any non-empty pair list compiles the right kernel. Take the
        # unfiltered list; the consumer still applies the class filter and still skips.
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

    # Where the golden's answer is a NaN the op *invented*, a narrowing pipeline turns its sign
    # into the observable result, and Wormhole's SFPMAD leaves that sign unspecified -- so assert
    # the magnitude there rather than withdrawing the variant. Blackhole specifies the canonical
    # NaN and keeps the full assertion.
    #
    # Pipeline and arch only: which *lanes* carry an invented NaN is the golden's own mask, since
    # `specials_in` is classified by an operand being non-finite and so mixes `inf - inf` with
    # `0 - (-inf)`. sfpu_binary() applies it per lane.
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


# Integer extremes (cat C). Delivered as a raw override rather than a StimuliSpec because
# CustomStrategy clamps integers through _get_integer_bounds, which returns info.min + 1 —
# so a spec asking for INT32_MIN silently yields INT32_MIN + 1, the worst failure mode for
# an edge test.
#
# Scope is deliberately narrow, and _INT_BINARY_STIMULI above is why: almost every int
# binary kernel documents a *sub-range* it is valid on (div/fmod < 2**24 for an exact
# int->fp32 reciprocal, mul < ~46340 so the product stays under 2**31, lcm assuming
# |a|,|b| < 2**15, max/min non-negative so signed and unsigned agree). Feeding those the
# int32 extremes would produce failures that are documented limitations rather than
# findings. The bitwise ops are the exception — "exact on the full default int range" — so
# they and the exact eq/ne comparisons are what cat C can honestly cover here.
#
# INT32_MIN itself is excluded: sign-magnitude Dst reads 0x80000000 as "negative zero" and
# cannot round-trip it. That is hardware, not a gap, and it already has a dedicated xfail
# (test_eltwise_binary_sfpu_int_shift_int32_min_unsupported). INT32_MIN + 1 stands in for it.
#
# The four *ordered* comparisons join the exact eq/ne pair on the same reasoning:
# calculate_binary_comp_int32 documents no sub-range, so the extremes are inside what the kernel
# promises and a divergence there is a finding. They also have the most to prove at these values
# -- the kernel normalises by computing `a - b` and folding the sign, and
# `INT32_MAX - (INT32_MIN + 1)` does not fit in int32.
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
    # twos_complement=True is required, not decorative. Without it the buffer is packed
    # sign-magnitude, and the bitwise kernels then operate on the wrong bits for negative
    # operands: (INT32_MIN+1) & -1 came back as -1 instead of INT32_MIN+1. The existing
    # test_eltwise_binary_sfpu_bitwise never caught this because its default stimuli are
    # positive-only, so nothing had established that these kernels need the two's-
    # complement pack path. test_eltwise_binary_sfpu_rsub_int32 already sets the same flag.
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        src_A_override=_build_int_extremes_src(),
        twos_complement=True,
    )


@pytest.mark.xfail(
    reason="Dst stores int32 as sign-magnitude with range +-(2^31 - 1). INT32_MIN "
    "(0x80000000) is 'negative zero' and cannot round-trip through Dst, so shifts that "
    "consume or produce it diverge from the two's-complement golden. This is a hardware "
    "limitation of the Wormhole SFPU load/store path; see docs/SFPU_INT32_SHIFT.md.",
    strict=False,
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
def test_eltwise_binary_sfpu_int_shift_int32_min_unsupported(
    formats,
    dest_acc,
    mathop,
):
    # Every value lane is INT32_MIN shifted by 0: the golden expects INT32_MIN back, but
    # HW loads it as sign-magnitude "negative zero", so this is expected to fail.
    num_elements = 32 * 32
    value_grid = [_INT32_MIN] * num_elements
    shift_grid = [0] * num_elements
    src = torch.tensor(value_grid + shift_grid, dtype=torch.int32)
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        src_A_override=src,
    )


# =============================================================================
# add_top_row
#
# Own inline driver: uses sources/sfpu_binary_test.cpp with a single [64, 32]
# tile pair, disabled format inference, and a golden reshaped to the top-row
# layout — different enough from sfpu_binary() to keep separate.
# =============================================================================


@parametrize(
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
def test_eltwise_binary_sfpu_add_top_row(formats, dest_acc, mathop):
    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip(
            "32-bit integer formats require DestAccumulation.Yes (HW cannot unpack into SrcA/SrcB)"
        )

    input_dimensions = [64, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

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

    configuration = TestConfig(
        "sources/sfpu_binary_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(),
            BROADCAST_TYPE(LlkBroadcastType.None_),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).view(input_dimensions)

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


# =============================================================================
# Broadcast kernel
#
# SFPU binary with row/column broadcast (BCAST_COL / BCAST_ROW). Uses its own
# 3-tile kernel source (sources/sfpu_binary_bcast_test.cpp) with a custom init
# and full-tile driver; InstrModLoadStore::DEFAULT works for any float dest
# format (compute is FP32).
# =============================================================================


class BroadcastType(Enum):
    # Values must match ckernel::BroadcastType in llk_defs.h
    # (NONE=0, COL=1, ROW=2, SCALAR=3) because the kernel does
    # `static_cast<BroadcastType>(BCAST_DIM_VAL)`.
    COL = 1
    ROW = 2


@dataclass
class SFPU_BCAST_DIM(TemplateParameter):
    bcast_dim: BroadcastType

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t BCAST_DIM_VAL = {self.bcast_dim.value};"


@dataclass
class INPUT_TILE_A(TemplateParameter):
    """Base DST tile index for input A.

    The kernel derives the other tile indices from this single value:
      INPUT_TILE_A      -> data tile
      INPUT_TILE_A + 1  -> bcast tile
      INPUT_TILE_A + 2  -> result tile
    """

    tile_index: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t INPUT_TILE_A_VAL = {self.tile_index};"


_BCAST_BINARY_OPS = {
    MathOperation.SfpuElwadd: torch.add,
    MathOperation.SfpuElwsub: torch.sub,
    MathOperation.SfpuElwmul: torch.mul,
}


def _golden_sfpu_binary_bcast(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    bcast_dim: BroadcastType,
    op,
    stimuli_format: DataFormat,
) -> torch.Tensor:
    """Golden for the SFPU bcast kernel (single 32x32 tile): broadcast in row-major space,
    then tilize to the packer's layout. `stimuli_format` drives tilize precision (Float16_b
    for Bfp8_b inputs, since the unpacker converts Bfp8_b -> Float16_b in dest)."""
    a = src_A.flatten()[:1024].reshape(32, 32)
    b = src_B.flatten()[:1024].reshape(32, 32)

    if bcast_dim == BroadcastType.ROW:
        b_bcast = b[0].unsqueeze(0).expand_as(b)
    else:
        b_bcast = b[:, 0].unsqueeze(1).expand_as(b)

    golden_rm = op(a, b_bcast.contiguous()).flatten()
    return tilize(golden_rm, stimuli_format=stimuli_format)


@skip_for_quasar
@parametrize(
    # Only same-format in/out combinations are supported by the broadcast kernel,
    # so `same=True` (a full Cartesian product would also blow past the 100-combo
    # Python test guideline).
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ],
        same=True,
    ),
    bcast_dim=[BroadcastType.ROW, BroadcastType.COL],
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_eltwise_binary_sfpu_bcast(
    formats,
    bcast_dim,
    mathop,
    dest_acc,
):
    _skip_fp32_no_dest_acc(formats, dest_acc)
    _skip_bh_float16_no_dest_acc(formats, dest_acc)

    # Mirror sfpu_binary(): on Blackhole, Float16/Float32 inputs require
    # dest_acc=Yes (32-bit dest), so silently upgrade the parametrized value.
    if (
        formats.input_format in [DataFormat.Float16, DataFormat.Float32]
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        dest_acc = DestAccumulation.Yes

    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

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

    # Only FP32 inputs with dest_acc=Yes take the unpack-to-dest path; all
    # other float formats go through srcA + MATH datacopy into dest.
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        "sources/sfpu_binary_bcast_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop),
            SFPU_BCAST_DIM(bcast_dim),
            INPUT_TILE_A(tile_index=0),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilize(src_A, stimuli_format=formats.input_format),
            formats.input_format,
            tilize(src_B, stimuli_format=formats.input_format),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result ({len(res_from_L1)}) and golden ({len(golden_tensor)}) size mismatch"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
