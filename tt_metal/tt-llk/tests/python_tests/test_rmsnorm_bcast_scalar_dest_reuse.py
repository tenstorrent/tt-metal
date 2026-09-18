# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Coverage for the rmsnorm bcast-scalar dest-reuse family (Blackhole only).

tt-metal #52709 promoted ``rmsnorm_bcast_scalar_reuse_tiles*``
(api/compute/experimental/rmsnorm.h) out of the deepseek_v3_b1 demo tree, taking blaze's
version of the header. Nothing in tt-llk exercised the underlying LLK pair
(``_llk_math_rmsnorm_bcast_scalar_dest_reuse_`` / ``_llk_unpack_A_rmsnorm_init_``) before
this file -- the headers did not even compile under the tt-llk build until the dead locals
they carried over from ``llk_unpack_A.h`` were removed.

Why a new file rather than an extension of an existing binary test: the op is a
``num_tiles``-templated MOP driven from a *single* unpack call, with SrcB sourced from DEST
via ``MOVD2B`` under a ``WAIT_SFPU | SRCB_VLD`` stall rather than from L1. ``test_bcast.py``
does one-tile-per-unpack broadcasts; ``test_eltwise_binary.py`` has neither the
``num_tiles``-as-template-argument plumbing nor a MOP-over-N-tiles axis.

What the driver does (see sources/rmsnorm_bcast_scalar_dest_reuse_test.cpp):

  1. a plain A2D datacopy seeds DEST[0] with one tile -- element [0] of it is the value
     MOVD2B broadcasts, standing in for the 1/RMS that add_rsqrt produces in the real kernel
  2. ``_llk_math_rmsnorm_bcast_scalar_dest_reuse_`` runs with ``src_index == dst_index == 0``,
     which is exactly how unified_kernels/rmsnorm.hpp:146 calls it
  3. ``num_tiles`` tiles are packed out

The seed tile is deliberately non-uniform, so a MOVD2B that picked up the wrong row or face
fails rather than silently agreeing with the golden.

Measured here, and the reason the axes below are shaped the way they are
------------------------------------------------------------------------
**ELWMUL accumulates into DEST; ELWADD overwrites it.** Both MOP branches pass 0 in the
instruction's dest-accumulate slot -- ``TT_OP_ELWADD(0, acc_to_dest, ...)`` with
``acc_to_dest == 0``, and ``TT_OP_ELWMUL(0, 0, ...)`` -- so the two read as if they behaved
alike. On BH p100a they do not: with the ZEROACC suppressed (``clear_dest=False``) the mul
lands ``seed + A*scalar`` while the add lands ``A + scalar``, i.e. only the add discards
what DEST already held.

That is why ``rmsnorm.hpp:146`` passes ``clear_dest=true`` for its mul and why
``clear_dest`` is not a free axis here: for ELWMUL it is a correctness requirement rather
than a preference, and the accumulate behaviour is pinned on its own below
(``test_rmsnorm_bcast_scalar_dest_reuse_mul_accumulates_into_dirty_dest``) so that a future
change making ELWMUL overwrite shows up as a failure instead of passing quietly.

Fidelity is swept for ELWMUL only. In ``rmsnorm_bcast_scalar_dest_reuse_configure_mop`` the
``math_fidelity`` template argument is consumed solely by the ELWMUL branch (via
``is_high_fidelity``), and ``rmsnorm_bcast_scalar_dest_reuse_configure_addrmod`` likewise
only varies its ADDR_MOD programming for ``ELWMUL && high_fidelity``. Sweeping it for
ELWADD would build identical ELFs.

``unpack_to_dest`` is off for every variant: the ``static_assert`` in
``_llk_unpack_A_rmsnorm_mop_config_`` rejects it for this configuration
(SCALAR + acc_to_dest + DEST_TO_SRCB), and driving the seed unpack through it while the op
itself cannot corrupts the fp32 result into alternating datums.
"""

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.golden_generators import EltwiseBinaryGolden, round_to_dest_width
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_FIDELITY,
    MATH_OP,
    RMSNORM_DEST_REUSE,
)
from helpers.utils import passed_test

pytestmark = [skip_for_wormhole, skip_for_quasar]

ELEMENTS_PER_TILE = 1024
ELEMENTS_PER_FACE = 256

# Same-in-same-out: this test is about the MOP and the MOVD2B scalar path, not about format
# conversion, so a mixed pair would only add unpack/pack noise.
FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True)

# The two ops the MOP instantiates distinct branches for. ELWSUB is accepted by
# rmsnorm_bcast_scalar_dest_reuse_configure_mop but the promoted compute API never
# instantiates it, so it is left out rather than covered speculatively.
OPS = [MathOperation.Elwadd, MathOperation.Elwmul]

# DEST half-sync capacity is 8 tiles at bf16 and 4 at fp32, so 4 is the largest count valid
# in every cell. 3 is included because an odd count is what catches an off-by-one in the
# MOP's inner loop (num_tiles * num_faces / 2).
NUM_TILES = [1, 2, 3, 4]

# Fidelity phases the FPU runs for ELWMUL; each masks a different slice of the operand
# mantissas and the results are summed. Mirrors test_eltwise_binary.py.
_FIDELITY_PHASES = {
    MathFidelity.LoFi: 1,
    MathFidelity.HiFi2: 2,
    MathFidelity.HiFi3: 3,
    MathFidelity.HiFi4: 4,
}


def _fidelities(mathop):
    """LoFi only for ELWADD -- see the module docstring.

    All four phase counts for ELWMUL, HiFi3 included. HiFi3 is the odd one: its phase
    count is 3, so unlike HiFi2 and HiFi4 it is not a power of two, and an implementation
    that derived the loop bound by shifting rather than from _FIDELITY_PHASES would pass
    the other three and fail only here. It was missing from every sweep in the suite.
    """
    if mathop == MathOperation.Elwmul:
        return [
            MathFidelity.LoFi,
            MathFidelity.HiFi2,
            MathFidelity.HiFi3,
            MathFidelity.HiFi4,
        ]
    return [MathFidelity.LoFi]


def _clear_dest_values(mathop):
    """ELWMUL accumulates, so only clear_dest=True has a DEST-independent result.

    The clear_dest=False half of the mul matrix is not dropped, it is asserted with the
    accumulating golden in its own test below.
    """
    if mathop == MathOperation.Elwmul:
        return [True]
    return [False, True]


def _skip_unsupported(formats, dest_acc):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")
    if (
        formats.input_format == DataFormat.Float16_b
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip(
            "Float16_b with dest_acc=Yes adds nothing here: the DEST width is not what "
            "this test varies"
        )


def _run(
    formats,
    dest_acc,
    mathop,
    math_fidelity,
    num_tiles,
    num_faces=4,
    clear_dest=False,
    unpack_full_transpose=False,
):
    """Compile+run one variant.

    Returns ``(device, seen_A, seed, scalar)``, all as fp32 tensors/scalars quantised the
    way the device saw them. ``seed`` is the tile the datacopy left in DEST[0] and is what
    the accumulating ELWMUL path adds onto.
    """
    torch.manual_seed(0)

    # src_B carries the broadcast scalar at element [0]. Spread over a wide range so a
    # MOVD2B that fetched the wrong row cannot land on a numerically similar value.
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[num_tiles * 32, 32],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[32, 32],
        spec_A=StimuliSpec.uniform(low=-4.0, high=4.0),
        spec_B=StimuliSpec.uniform(low=-4.0, high=4.0),
    )

    configuration = TestConfig(
        "sources/rmsnorm_bcast_scalar_dest_reuse_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop),
            MATH_FIDELITY(math_fidelity),
            RMSNORM_DEST_REUSE(
                rmsnorm_num_tiles=num_tiles,
                rmsnorm_num_faces=num_faces,
                clear_dest=clear_dest,
                unpack_full_transpose=unpack_full_transpose,
            ),
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
            tile_count_res=num_tiles,
        ),
        # See the module docstring: the op's own static_assert rejects unpack-to-dest.
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )

    span = num_tiles * ELEMENTS_PER_TILE
    res_from_L1 = configuration.run().result[:span]
    torch_format = format_dict[formats.output_format]

    device = torch.tensor(res_from_L1, dtype=torch_format).flatten().to(torch.float32)
    # The operands as the device saw them: quantised to the input format, then to the DEST
    # width the datacopy landed them at.
    seen_A = round_to_dest_width(src_A.flatten()[:span].to(torch.float32), dest_acc)
    seed = round_to_dest_width(
        src_B.flatten()[:ELEMENTS_PER_TILE].to(torch.float32), dest_acc
    )
    scalar = seed[:1].item()
    return device, seen_A, seed, scalar


def _expected(seen_A, scalar, mathop, math_fidelity, output_format, dest_prior=None):
    """Golden for the rows the MOP covers.

    ELWMUL is modelled phase by phase rather than as a single fp32 product: at LoFi the FPU
    keeps only the leading mantissa bits of each operand, which on this sweep costs a few
    percent -- far more than any sane tolerance should absorb. Following
    test_eltwise_binary.py, the truncation is reproduced in the golden (via
    EltwiseBinaryGolden._apply_fidelity_masking) instead of being papered over by widening
    rtol, so the assertion stays exact at every fidelity.

    ``dest_prior`` is what DEST held when the MOP ran, and is only consulted for ELWMUL --
    the one op that accumulates rather than overwrites. Pass ``None`` when the ZEROACC has
    made DEST known-zero.
    """
    if mathop == MathOperation.Elwadd:
        return seen_A + scalar

    # Instantiate directly: get_golden_generator hands back a DummyGoldenGenerator during
    # compile-producer, which has no _apply_fidelity_masking.
    binary_golden = EltwiseBinaryGolden()
    scalar_tensor = torch.full_like(seen_A, scalar)

    product = None
    for fidelity_iteration in range(_FIDELITY_PHASES[math_fidelity]):
        a_masked, b_masked = binary_golden._apply_fidelity_masking(
            output_format, seen_A, scalar_tensor, fidelity_iteration
        )
        phase = a_masked.to(torch.float32) * b_masked.to(torch.float32)
        product = phase if product is None else product + phase

    return product if dest_prior is None else dest_prior + product


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=OPS,
    math_fidelity=lambda mathop: _fidelities(mathop),
    num_tiles=NUM_TILES,
    clear_dest=lambda mathop: _clear_dest_values(mathop),
)
def test_rmsnorm_bcast_scalar_dest_reuse(
    formats, dest_acc, mathop, math_fidelity, num_tiles, clear_dest
):
    """The whole-tile case: every DEST row the pack reads is covered by the MOP.

    This is the shape the production caller uses -- one broadcast scalar applied across
    num_tiles tiles from a single unpack -- swept over both ops, all four fidelities that
    reach distinct ELWMUL code, and the DEST-capacity range of num_tiles.
    """
    _skip_unsupported(formats, dest_acc)

    device, seen_A, _seed, scalar = _run(
        formats, dest_acc, mathop, math_fidelity, num_tiles, clear_dest=clear_dest
    )
    golden = _expected(seen_A, scalar, mathop, math_fidelity, formats.output_format)

    assert passed_test(golden, device, formats.output_format), (
        f"rmsnorm bcast-scalar dest reuse mismatch (op={mathop.name}, "
        f"fidelity={math_fidelity.name}, num_tiles={num_tiles}, "
        f"clear_dest={clear_dest}, scalar={scalar:.6f})"
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    math_fidelity=_fidelities(MathOperation.Elwmul),
)
def test_rmsnorm_bcast_scalar_dest_reuse_mul_accumulates_into_dirty_dest(
    formats, dest_acc, math_fidelity
):
    """ELWMUL adds its product onto whatever DEST already held.

    Measured on BH p100a: with the ZEROACC suppressed the mul lands ``seed + A*scalar``,
    not ``A*scalar``, even though its MOP passes 0 in the instruction's dest-accumulate
    slot exactly as the add's does. ELWADD under the same conditions overwrites -- that
    asymmetry is asserted by the negative control below.

    This is the contract ``rmsnorm.hpp:146`` depends on when it passes ``clear_dest=true``.
    If a change ever makes ELWMUL overwrite, this test fails rather than the production
    caller silently gaining a redundant ZEROACC.

    Restricted to num_tiles=1 on purpose: only DEST[0] is seeded deterministically, so for
    higher counts the tiles past the first would be accumulating onto whatever the previous
    kernel left behind.
    """
    _skip_unsupported(formats, dest_acc)

    device, seen_A, seed, scalar = _run(
        formats,
        dest_acc,
        MathOperation.Elwmul,
        math_fidelity,
        num_tiles=1,
        clear_dest=False,
    )
    accumulating = _expected(
        seen_A,
        scalar,
        MathOperation.Elwmul,
        math_fidelity,
        formats.output_format,
        dest_prior=seed,
    )
    overwriting = _expected(
        seen_A, scalar, MathOperation.Elwmul, math_fidelity, formats.output_format
    )

    assert passed_test(accumulating, device, formats.output_format), (
        f"ELWMUL with clear_dest=False did not accumulate onto the seeded DEST "
        f"(fidelity={math_fidelity.name}, scalar={scalar:.6f}). If it now matches "
        "A*scalar instead, the op has become overwriting and rmsnorm.hpp's clear_dest=true "
        "is no longer load-bearing -- update the module docstring."
    )
    assert not passed_test(
        overwriting, device, formats.output_format, print_errors=False
    ), (
        "ELWMUL with clear_dest=False matched the overwriting golden, so the seeded DEST "
        "made no difference and this test is not measuring the accumulate path"
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    num_tiles=NUM_TILES,
)
def test_rmsnorm_bcast_scalar_dest_reuse_add_overwrites_dirty_dest(
    formats, dest_acc, num_tiles
):
    """Negative control for the test above: ELWADD is insensitive to what DEST held.

    Without this, "ELWMUL accumulates" would be an observation about the driver rather than
    about the op -- it is the contrast with ELWADD under an identical dirty DEST that makes
    it a property of the MOP branch.
    """
    _skip_unsupported(formats, dest_acc)

    dirty, seen_A, _seed, scalar = _run(
        formats,
        dest_acc,
        MathOperation.Elwadd,
        MathFidelity.LoFi,
        num_tiles,
        clear_dest=False,
    )
    golden = _expected(
        seen_A, scalar, MathOperation.Elwadd, MathFidelity.LoFi, formats.output_format
    )

    assert passed_test(golden, dirty, formats.output_format), (
        f"ELWADD with clear_dest=False did not overwrite the seeded DEST "
        f"(num_tiles={num_tiles}, scalar={scalar:.6f}) -- if it has started accumulating, "
        "every rmsnorm add call site that omits clear_dest is now wrong"
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=OPS,
    num_faces=[1, 2],
    num_tiles=[1, 2],
    math_fidelity=lambda mathop: _fidelities(mathop),
)
def test_rmsnorm_bcast_scalar_dest_reuse_partial_faces(
    formats, dest_acc, mathop, num_faces, num_tiles, math_fidelity
):
    """num_faces < 4: the MOP covers only the leading faces of each tile.

    The pack still emits full 4-face tiles, so the trailing (4 - num_faces) * 256 elements
    of every tile are whatever DEST held. With clear_dest=True the ZEROACC wiped them to
    zero, which is what this asserts -- and it is the only configuration in which that flag
    is observable for ELWADD at all, since the add overwrites everything the MOP does cover.

    Why the covered region is per-tile rather than one run from the start: the addr_mod the
    init programs increments DEST by ``8 + (4 - num_faces) * 16``
    (llk_math_rmsnorm_bcast_scalar_dest_reuse.h), i.e. after a tile's covered faces it
    skips the uncovered ones and lands on the next tile's base. The unpack side, by
    contrast, reads ``num_tiles * num_faces`` faces contiguously out of L1. So the k-th
    face of the input goes to tile k's leading face-slot, and input faces the MOP never
    reads simply stay in L1 -- which is why the golden below slices the input contiguously
    while indexing the device output per tile.

    That is also what the num_tiles axis is for. At num_tiles=1 the skip term is
    unobservable: there is no second tile for a wrong stride to land in, so
    ``(4 - num_faces) * 16`` could be any value and the test would still pass. At
    num_tiles=2 it is pinned -- halving the term on BH p100a leaves tile 0's uncovered
    faces non-zero and fails here.
    """
    _skip_unsupported(formats, dest_acc)

    device, seen_A, _seed, scalar = _run(
        formats,
        dest_acc,
        mathop,
        math_fidelity,
        num_tiles=num_tiles,
        num_faces=num_faces,
        clear_dest=True,
    )

    covered = num_faces * ELEMENTS_PER_FACE
    for tile in range(num_tiles):
        source = seen_A[tile * covered : (tile + 1) * covered]
        golden_covered = _expected(
            source, scalar, mathop, math_fidelity, formats.output_format
        )
        base = tile * ELEMENTS_PER_TILE

        assert passed_test(
            golden_covered, device[base : base + covered], formats.output_format
        ), (
            f"the {num_faces} face(s) the MOP covers in tile {tile} are wrong "
            f"(op={mathop.name}, fidelity={math_fidelity.name}, num_tiles={num_tiles}, "
            f"scalar={scalar:.6f})"
        )

        tail = device[base + covered : base + ELEMENTS_PER_TILE]
        nonzero = torch.count_nonzero(tail).item()
        assert nonzero == 0, (
            f"clear_dest=True left {nonzero} non-zero elements in the {4 - num_faces} "
            f"face(s) of tile {tile} the MOP does not cover -- the ZEROACC between the "
            "MOVD2B and the MOP did not clear DEST"
        )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=OPS,
    math_fidelity=lambda mathop: _fidelities(mathop),
)
def test_rmsnorm_bcast_scalar_dest_reuse_unpack_full_transpose(
    formats, dest_acc, mathop, math_fidelity
):
    """The transpose-fold path, which exists only in blaze's version of the header.

    ``rmsnorm_bcast_scalar_reuse_tiles_init_fidelity<..., unpack_full_transpose=true>``
    folds a full 32x32 transpose into the unpack: transpose_of_faces reorders faces via the
    5-instruction replay buffer in ``_llk_unpack_A_rmsnorm_mop_config_``, and
    within_face_16x16_transpose transposes inside each face. The LLK_ASSERTs restrict the
    path to num_tiles == 1 and num_faces == 4.

    This is reachable surface the demo-tree version did not have, so a failure here is a
    reconciliation regression rather than a numerical one.

    Swept over fidelity rather than pinned at LoFi: for ELWMUL the transpose and the
    fidelity phase loop touch the same operands from opposite ends -- the replay buffer
    reorders which face lands in which SrcA bank, and each phase masks a different slice of
    the operand mantissas -- and nothing else in the suite runs the two together. The
    golden composes them in the obvious order (transpose the input, then apply the phase
    sum), so a fidelity-dependent failure here means they do not actually compose.
    """
    _skip_unsupported(formats, dest_acc)

    device, seen_A, _seed, scalar = _run(
        formats,
        dest_acc,
        mathop,
        math_fidelity,
        num_tiles=1,
        clear_dest=True,
        unpack_full_transpose=True,
    )

    # A full 32x32 transpose in face-tiled space: transposing the 2x2 face grid and
    # transposing within each 16x16 face together are exactly the tile transpose.
    faces = seen_A[:ELEMENTS_PER_TILE].reshape(2, 2, 16, 16)
    transposed = faces.permute(1, 0, 3, 2).reshape(-1)

    golden = _expected(transposed, scalar, mathop, math_fidelity, formats.output_format)

    assert passed_test(golden, device, formats.output_format), (
        f"unpack_full_transpose mismatch (op={mathop.name}, "
        f"fidelity={math_fidelity.name}, scalar={scalar:.6f}) -- the transpose-fold path "
        "is the axis blaze's version of rmsnorm.h added"
    )
