# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Top32 row-major sort -- the DeepSeek ``top32_rm`` family, Blackhole only.

Unblocked by tt-metal #52713 merging. Before this file the family had **no tt-llk
coverage at all**: ``_llk_unpack_A_top32_rm_init_``/``_``, ``_llk_math_top32_rm_init_``/``_``,
``_top32_rm_init_``, ``_bitonic_top32_phases_steps_``, ``_bitonic_top32_merge_`` and
``_bitonic_top32_rebuild_`` were uncalled from ``tests/sources``. ``_top32_rm_init_`` looked
covered on a grep and was not -- its only occurrence in the test tree was inside a comment.

What is being asserted
----------------------
One row of ``row_elements`` values plus a parallel row of indices, both **row-major** in L1,
reduced to the 32 largest values in descending order and the index each came from. The
driver mirrors ``tests/tt_metal/tt_metal/test_kernels/compute/top32_rm_dev_compute.cpp``
statement for statement, so what passes here is the sequence the only in-tree consumer runs.

Both operands are the same format, with indices carried as floats holding the integer
itself. That is exact while ``row_elements <= 256`` (bf16 has 8 mantissa bits), which is why
the sweep stops there -- and it keeps the driver free of the srcA/pack format reconfigs the
consumer needs for its uint32 index tile. See the driver header for the trade.

Values are a permutation of **distinct** integers straddling zero. Distinct matters: with
ties, "the top 32" does not determine the indices, and this test asserts indices exactly
rather than tolerating the hardware's tie order (cf. ``test_topk.py``, which has to).
Straddling zero matters because it makes the -inf padding load-bearing: a 32-element tail
chunk fills only two faces, and the other two arrive as -inf from
``_llk_unpack_A_top32_rm_``'s ``CLR_SRC_NEGINF``. Were they zeros, every negative input
would lose to padding and the ``row_elements=160`` case would quietly return the wrong set.

``dest_acc`` is a real axis, not a formality: it selects the index word width inside the
sort (``InstrModLoadStore::INT32`` vs ``LO16``) *and* the Dest-move opcode in
``llk_math_top32_rm_configure_mop`` (ELWADD against a zeroed SrcB at fp32 Dest, MOVA2D
otherwise). The consumer only ever builds fp32 Dest, so the ``dest_acc=No`` cells are the
first exercise the 16-bit half of this family has had.

``test_top32_rm_pre_sorted`` covers the other mode
-------------------------------------------------
At >= 1024 elements the consumer switches to ``top32_rm_dev_compute_v2.cpp`` and a
different set of entry points -- ``_bitonic_top32_of_1024_rm_pre_sorted_{prep,combine,final}_``
-- reaching Dest through ``transpose_tile`` rather than through this family's own unpack, so
each Dest column holds 32 of the row's elements and 16 columns are reduced at once.

That mode carries a contract the plain one does not: **the input must already be sorted into
descending runs of 32**. prep only builds bitonic sequences out of runs that are already
monotone, so unsorted input returns a wrong answer rather than failing, which is why the
stimuli there are generated with the value keyed on ``i % 32`` (the same shape the family's
own dev test uses) plus a per-group tiebreak.

It is Float32-only, and that is forced rather than chosen: at 1024+ elements the indices
leave bf16's exactly-representable range, and so do value tiebreaks fine enough to keep the
group leaders distinct. Float32 also routes the transpose through its 32-bit branch
(unpack-to-dest + ``_llk_math_transpose_dest_``), which is the branch the consumer takes for
its uint32 index tile.

Not covered here, deliberately
------------------------------
* The mixed shape -- whole 1024-element chunks *plus* a 64-element tail, i.e. the Metal dev
  test's row=3232 -- which runs mode 1 and then finishes in mode 0. Both halves are covered
  separately; their composition is not.
* The 8-datum ``bitonic_top32_load8``/``store8`` helpers, which the header itself records
  as referenced by no kernel today.
* The 7 ``llk_math_deepseek_top32_rm_*`` Metal wrappers on main, which still have no caller
  anywhere: they wrap the same primitives this file drives, but through the Metal API layer,
  so covering them needs a metal-side test (same shape as B1).
"""

import numpy as np
import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TOP32_RM
from helpers.utils import passed_test

pytestmark = [skip_for_wormhole, skip_for_quasar]

ELEMENTS_PER_TILE = 1024

# One _llk_unpack_A_top32_rm_ call moves 16 elements per face into a Dest column, 4 faces.
ELEMENTS_PER_CHUNK = 64

TOP_K = 32

# Both widths. They are not the same test: the format decides which branch of the family's
# unpack/copy pair runs, exactly as it does in llk_unpack_A_top32_rm_api.h.
#
#   Float16_b  the unpacker does the within-face transpose and clears SrcA to -infinity
#              first. This is the branch the consumer uses for its VALUE tile.
#   Float32    too wide for SrcA, so the tile goes to Dest via unpack-to-dest and the
#              transpose moves to the math thread. This is the branch the consumer uses for
#              its uint32 INDEX tile.
#
# bf16's 8 mantissa bits are what bound the row at 256, since indices ride in the same format
# as the values; Float32 would not bind, but keeping one sweep makes the two branches
# comparable cell by cell.
FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True)

# 64   one chunk, no merge across tiles at all -- phases_steps + merge + rebuild only
# 128  two full chunks, so the across-tiles merge runs once
# 160  the Metal dev test's tail case: two full chunks plus a 32-element num_faces=2 chunk.
#      **16-bit only** -- see test_top32_rm_32bit_partial_chunk below for why the 32-bit
#      branch cannot do this and what it returns instead.
# 256  four full chunks, and the largest row whose indices stay exact in bf16
ROW_ELEMENTS = [64, 128, 160, 256]


def _stimuli(row_elements):
    """Distinct bf16-exact values straddling zero, plus their row-major indices."""
    torch.manual_seed(0)

    # Distinct integers in [-row/2, row/2), all exact in bf16 for row <= 256, shuffled so
    # neither the top-32 nor the winning indices are contiguous in the input.
    values = torch.arange(row_elements, dtype=torch.float32) - (row_elements // 2)
    values = values[torch.randperm(row_elements)]
    indices = torch.arange(row_elements, dtype=torch.float32)
    return values, indices


def _golden(values, indices):
    order = torch.argsort(values, descending=True, stable=True)[:TOP_K]
    return values[order], indices[order]


def _pad_to_tile(flat):
    remainder = flat.numel() % ELEMENTS_PER_TILE
    if remainder == 0:
        return flat
    return torch.cat([flat, torch.zeros(ELEMENTS_PER_TILE - remainder)])


@parametrize(
    formats=FORMATS,
    row_elements=ROW_ELEMENTS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_top32_rm(formats, row_elements, dest_acc, via_wrappers=False):
    is_32bit = formats.input_format.is_32_bit()
    if is_32bit and dest_acc == DestAccumulation.No:
        pytest.skip("32-bit datums need fp32 Dest")
    if is_32bit and row_elements % ELEMENTS_PER_CHUNK != 0:
        pytest.skip(
            "a partially-filled chunk is broken on the 32-bit branch -- pinned by "
            "test_top32_rm_32bit_partial_chunk"
        )

    values, indices = _stimuli(row_elements)

    torch_format = format_dict[formats.output_format]
    # Round the stimuli to the input format before generating the golden, so the golden is
    # a statement about what the device was actually given. Exact here by construction --
    # every value and index is a small integer -- and asserted below rather than assumed.
    values_in = values.to(torch_format).to(torch.float32)
    indices_in = indices.to(torch_format).to(torch.float32)
    assert torch.equal(values_in, values) and torch.equal(indices_in, indices), (
        f"row_elements={row_elements} is outside the exactly-representable range of "
        f"{formats.input_format.name}; the index encoding this test relies on has broken"
    )

    tiles_per_operand = max(1, -(-row_elements // ELEMENTS_PER_TILE))

    configuration = TestConfig(
        "sources/top32_rm_test.cpp",
        formats,
        templates=[
            TOP32_RM(
                row_elements=row_elements,
                datum_bytes=4 if is_32bit else 2,
                top_min=False,
                top32_mode=0,
                via_wrappers=via_wrappers,
            )
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            _pad_to_tile(values),
            formats.input_format,
            _pad_to_tile(indices),
            formats.input_format,
            formats.output_format,
            tile_count_A=tiles_per_operand,
            tile_count_B=tiles_per_operand,
            # Two result tiles: the top-32 values, then their indices.
            tile_count_res=2,
        ),
        unpack_to_dest=is_32bit,
        dest_acc=dest_acc,
    )

    result = configuration.run().result

    # Each result tile carries its 32 datums at the front: the sort leaves the survivors in
    # the first column of Dest rows 0..31, and the driver narrows the packer to one datum
    # per row before packing, exactly as the consumer does.
    device_values = torch.tensor(result[:TOP_K], dtype=torch_format).to(torch.float32)
    device_indices = torch.tensor(
        result[ELEMENTS_PER_TILE : ELEMENTS_PER_TILE + TOP_K], dtype=torch_format
    ).to(torch.float32)

    golden_values, golden_indices = _golden(values_in, indices_in)

    assert passed_test(
        golden_values, device_values, formats.output_format, print_errors=False
    ), (
        f"top32 values wrong for row_elements={row_elements}, dest_acc={dest_acc.name}\n"
        f"  device: {device_values.tolist()}\n"
        f"  golden: {golden_values.tolist()}"
    )

    # Indices are asserted exactly, which the distinct-values stimuli make well defined:
    # this is what separates a top-32 from a max-32 and is the only check that the sort
    # permutes the index region in step with the value region.
    assert torch.equal(device_indices, golden_indices), (
        f"top32 indices wrong for row_elements={row_elements}, "
        f"dest_acc={dest_acc.name} -- the values may still be right, in which case the "
        f"index region is not being permuted with them\n"
        f"  device: {device_indices.tolist()}\n"
        f"  golden: {golden_indices.tolist()}"
    )


# --- pre-sorted mode (>= 1024 elements) ------------------------------------------------

PRE_SORTED_FORMATS = input_output_formats([DataFormat.Float32], same=True)

# 1024  one tile: prep then final, with no combine at all
# 2048  two tiles, so combine runs and the top-32 has to be assembled across them
# 1088  one tile plus one full 64-element tail chunk
# 1152  one tile plus two full tail chunks, so the tail loop iterates
#
# The tail chunks are all full 64s on purpose. This mode is Float32, i.e. the 32-bit branch,
# and a partially-filled chunk there is broken -- see test_top32_rm_32bit_partial_chunk. The
# Metal dev test's row=3232 has a 32-element tail chunk, so that exact shape is NOT reachable
# today; the two halves either side of it are.
PRE_SORTED_ROW_ELEMENTS = [1024, 2048, 1088, 1152]

# Rank step inside a run of 32. Larger than any tiebreak below, so the runs are strictly
# descending whatever the tiebreaks are -- which is the contract prep depends on.
_RANK_STEP = 1000.0


def _pre_sorted_stimuli(row_elements):
    """Descending runs of 32, with a distinct per-run tiebreak.

    Same shape as the family's own dev test (value keyed on ``i % 32``), but with the random
    jitter replaced by a permutation of the run indices: that keeps every value distinct in
    Float32, so "the top 32" determines the indices and they can be asserted exactly.
    """
    torch.manual_seed(0)

    num_runs = row_elements // TOP_K
    # Which run wins is a permutation, so the top-32 is not the first 32 runs and the
    # answer depends on combine actually merging across tiles.
    tiebreak = torch.randperm(num_runs, dtype=torch.int64).to(torch.float32)

    # For a row with a tail, force the single largest tiebreak into the LAST run, so at least
    # one member of the top-32 can only be found by the tail path. Without this the
    # permutation might put every winner in the whole-tile region and the tail chunks would
    # be dead weight the test could not tell apart from a tail loop that did nothing.
    if row_elements % ELEMENTS_PER_TILE != 0:
        top_run = int(torch.argmax(tiebreak).item())
        last_run = num_runs - 1
        tiebreak[top_run], tiebreak[last_run] = (
            tiebreak[last_run].clone(),
            tiebreak[top_run].clone(),
        )

    rank = torch.arange(row_elements, dtype=torch.float32) % TOP_K
    run = torch.arange(row_elements, dtype=torch.int64) // TOP_K

    values = (TOP_K - rank) * _RANK_STEP + tiebreak[run]
    indices = torch.arange(row_elements, dtype=torch.float32)
    return values, indices


@parametrize(
    formats=PRE_SORTED_FORMATS,
    row_elements=PRE_SORTED_ROW_ELEMENTS,
)
def test_top32_rm_pre_sorted(formats, row_elements, via_wrappers=False):
    """The >= 1024-element path: transpose whole tiles, then prep / combine / final."""
    values, indices = _pre_sorted_stimuli(row_elements)

    # Float32 holds every value and index here exactly, so the golden needs no rounding --
    # asserted rather than assumed, as in the plain mode.
    torch_format = format_dict[formats.output_format]
    assert torch.equal(values.to(torch_format).to(torch.float32), values)
    assert torch.equal(indices.to(torch_format).to(torch.float32), indices)

    # The runs the mode's contract is about.
    runs = values.reshape(-1, TOP_K)
    assert bool((runs[:, :-1] > runs[:, 1:]).all()), (
        "stimuli are not pre-sorted into descending runs of 32, which is the input "
        "contract of _bitonic_top32_of_1024_rm_pre_sorted_prep_"
    )

    # Whole tiles for the pre-sorted path, plus one more to hold the tail elements: the tail
    # is read 64 elements at a time out of the same row-major buffer, so it needs L1 behind it.
    tiles_per_operand = -(-row_elements // ELEMENTS_PER_TILE)

    configuration = TestConfig(
        "sources/top32_rm_test.cpp",
        formats,
        templates=[
            TOP32_RM(
                row_elements=row_elements,
                datum_bytes=4,  # Float32
                top_min=False,
                top32_mode=1,
                via_wrappers=via_wrappers,
            )
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            values,
            formats.input_format,
            indices,
            formats.input_format,
            formats.output_format,
            tile_count_A=tiles_per_operand,
            tile_count_B=tiles_per_operand,
            tile_count_res=2,
        ),
        # 32-bit operands take the unpack-to-dest transpose branch, and fp32 Dest is the
        # only configuration this mode has ever run in.
        unpack_to_dest=True,
        dest_acc=DestAccumulation.Yes,
    )

    result = configuration.run().result

    device_values = torch.tensor(result[:TOP_K], dtype=torch_format).to(torch.float32)
    device_indices = torch.tensor(
        result[ELEMENTS_PER_TILE : ELEMENTS_PER_TILE + TOP_K], dtype=torch_format
    ).to(torch.float32)

    golden_values, golden_indices = _golden(values, indices)

    tail_first = (row_elements // ELEMENTS_PER_TILE) * ELEMENTS_PER_TILE
    if tail_first < row_elements:
        # Guards the point of the mixed-shape cases: if no winner comes from the tail, the
        # test cannot distinguish a working tail loop from one that does nothing.
        assert bool((golden_indices >= tail_first).any()), (
            f"row_elements={row_elements}: no element of the top-32 comes from the tail "
            f"region (index >= {tail_first}), so this case does not exercise the tail path"
        )

    assert passed_test(
        golden_values, device_values, formats.output_format, print_errors=False
    ), (
        f"pre-sorted top32 values wrong for row_elements={row_elements}\n"
        f"  device: {device_values.tolist()}\n"
        f"  golden: {golden_values.tolist()}"
    )

    assert torch.equal(device_indices, golden_indices), (
        f"pre-sorted top32 indices wrong for row_elements={row_elements} -- the values "
        f"may still be right, in which case the index region is not being permuted with "
        f"them\n"
        f"  device: {device_indices.tolist()}\n"
        f"  golden: {golden_indices.tolist()}"
    )


# --- the 32-bit branch cannot take a partially-filled chunk -------------------------------

_PARTIAL_CHUNK_XFAIL = (
    "DEFECT: on the 32-bit (unpack-to-dest) branch of the top32_rm unpack, a chunk that "
    "fills fewer than 4 faces sorts against whatever Dest already held. Measured on BH "
    "p100a: with a 160-element row of values in [-80, 79], the returned top-32 contained "
    "11026.0, 10041.0, 9058.0 and more -- values that are not in the input at all. The "
    "16-bit branch has no such problem: its unpacker clears SrcA to -infinity "
    "(CLR_SRC_NEGINF) before unpacking, so the unfed faces lose every comparison. The "
    "32-bit branch never clears anything, and the ZEROACC in _llk_math_top32_rm_ covers "
    "only `num_faces` faces, leaving the rest of the tile untouched."
)


@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    row_elements=[160],
)
def test_top32_rm_32bit_partial_chunk(formats, row_elements, request):
    """Pins the defect above, and flips to XPASS the moment the 32-bit branch clears its tile.

    Latent in the consumer rather than harmless: `top32_rm_dev_compute.cpp` does call this
    branch with `num_faces=2`, but only for its **uint32 index** tile, and an index slot can
    only be selected if the paired value slot wins -- and the value tile is bf16, so its
    padding is -infinity and never wins. The defect is therefore invisible until someone puts
    *values* through the 32-bit branch, which the family's doc tables permit.

    Marked xfail rather than asserting the wrong answer: what leaks in is whatever Dest held,
    so the failure is real but its contents are not a stable thing to assert on.
    """
    request.node.add_marker(
        pytest.mark.xfail(reason=_PARTIAL_CHUNK_XFAIL, strict=False)
    )

    values, indices = _stimuli(row_elements)
    torch_format = format_dict[formats.output_format]

    configuration = TestConfig(
        "sources/top32_rm_test.cpp",
        formats,
        templates=[
            TOP32_RM(
                row_elements=row_elements, datum_bytes=4, top_min=False, top32_mode=0
            )
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            _pad_to_tile(values),
            formats.input_format,
            _pad_to_tile(indices),
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=2,
        ),
        unpack_to_dest=True,
        dest_acc=DestAccumulation.Yes,
    )

    result = configuration.run().result
    device_values = torch.tensor(result[:TOP_K], dtype=torch_format).to(torch.float32)
    golden_values, _ = _golden(values, indices)

    assert passed_test(
        golden_values, device_values, formats.output_format, print_errors=False
    ), (
        f"32-bit branch, 160-element row (one 32-element chunk)\n"
        f"  device: {device_values.tolist()}\n"
        f"  golden: {golden_values.tolist()}"
    )


# --- the Metal wrapper layer ---------------------------------------------------------------

# `experimental/llk_sfpu/llk_math_deepseek_top32_rm.h` publishes 7 entry points for this family
# and, before these two tests, **nothing in the tree called any of them** -- the in-tree
# consumers drive the `ckernel::sfpu::` primitives directly through SFPU_UNARY_CALL. The
# wrappers are thin (each is the same `_llk_math_eltwise_unary_sfpu_params_` call the driver
# makes itself), so the point is not new arithmetic: it is that the wrapper layer compiles, is
# code-generated, and computes what the primitives compute. A signature drift or a wrong
# template argument inside a wrapper is invisible without this.


@parametrize(
    formats=FORMATS,
    row_elements=[64, 160],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_top32_rm_via_metal_wrappers(formats, row_elements, dest_acc):
    """The plain mode, driven entirely through the Metal wrapper layer.

    Same assertions as ``test_top32_rm`` -- values and indices, exactly -- so a wrapper that
    forwards the wrong argument or instantiates the wrong template shows up as a wrong answer
    rather than as nothing at all. Covers `llk_math_deepseek_top32_rm_init`, `_local_sort`,
    `_merge` and `_rebuild`; the pre-sorted test below covers the other three.
    """
    test_top32_rm(formats, row_elements, dest_acc, via_wrappers=True)


@parametrize(
    formats=PRE_SORTED_FORMATS,
    row_elements=[1024, 1088],
)
def test_top32_rm_pre_sorted_via_metal_wrappers(formats, row_elements):
    """The pre-sorted mode through the wrappers: `_prep`, `_combine`, `_final`.

    1088 rather than only 1024 so the tail path runs too, which is what puts `_rebuild` and
    `_merge` through the wrapper layer in this mode as well.
    """
    test_top32_rm_pre_sorted(formats, row_elements, via_wrappers=True)


# --- edge-value populations: ties, signed zeros, NaN, infinities ----------------------------
#
# Every test above uses distinct finite values, so the order in which the network resolves an
# exact tie, and what it does with the special values, is never observed. SFPSWAP compares
# its operands as sign-magnitude integers -- the total order
#   -NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN
# on the raw bits -- and, with index tracking on, moves the index lanes whenever it moves the
# value lanes, so two bit-equal keys may or may not trade indices depending on the swap mode.
# These tests pin both: the values must be exactly the top 32 in that total order, and every
# returned index must point at an input whose bits equal the value returned beside it (ties
# make the index choice non-unique, so it is checked for validity rather than against one
# answer; bit-identity of the tie order across kernel changes is what the result hash is for).
#
# Subnormals are left to their own population: the 32-bit path may flush them, so that
# population accepts either a raw or a flushed golden there. The 16-bit branch goes through
# SrcA, which (measured on BH, independent of the sort) turns NaN into a same-signed infinity
# and -0 / subnormals into +0 before the sort sees them; its golden is taken on that view.

_F32_NAN_POS = 0x7FC00000
_F32_NAN_NEG = 0xFFC00000
_F32_INF_POS = 0x7F800000
_F32_INF_NEG = 0xFF800000
_F32_ZERO_NEG = 0x80000000


def _sign_mag_key(bits):
    """SFPSWAP's comparison key: sign-magnitude bits as an integer (int64 tensor)."""
    bits = bits.to(torch.int64) & 0xFFFFFFFF
    mag = bits & 0x7FFFFFFF
    return torch.where(bits >= 0x80000000, -mag - 1, mag)


def _bits_to_f32(bits):
    return torch.from_numpy(
        bits.to(torch.int64).numpy().astype(np.uint32).view(np.float32)
    )


def _f32_to_bits(values):
    return torch.from_numpy(
        values.to(torch.float32).numpy().view(np.uint32).astype(np.int64)
    )


def _edge_value_bits(num, population, generator, bf16):
    """Raw fp32 bit patterns for `population`, all exactly representable in bf16 if `bf16`."""
    if population == "ties_specials":
        # A small value set so nearly every comparison is a tie, plus the specials.
        pool = [
            0x00000000,
            _F32_ZERO_NEG,
            _F32_INF_POS,
            _F32_INF_NEG,
            _F32_NAN_POS,
            _F32_NAN_NEG,
            0x3F800000,  # 1.0
            0xBF800000,  # -1.0
            0x40000000,  # 2.0
            0xC0400000,  # -3.0
            0x3F000000,  # 0.5
        ]
        # Weighted so the top 32 is mostly ties at 1.0 / 2.0 with a few +NaN / +inf above.
        weights = torch.tensor([3, 3, 1, 2, 1, 2, 10, 6, 8, 6, 6], dtype=torch.float64)
        pick = torch.multinomial(weights, num, replacement=True, generator=generator)
        bits = torch.tensor(pool, dtype=torch.int64)[pick]
    elif population == "random_ties":
        # Random normals with a coarse mantissa (many exact ties), both signs, wide exponent range.
        sign = torch.randint(0, 2, (num,), generator=generator, dtype=torch.int64) << 31
        exp = (
            torch.randint(100, 150, (num,), generator=generator, dtype=torch.int64)
            << 23
        )
        mant = torch.randint(0, 4, (num,), generator=generator, dtype=torch.int64) << 21
        bits = sign | exp | mant
    elif population == "subnormal":
        # The top 32 lands among subnormals, signed zeros and the smallest normals.
        pool = [
            0x00000000,
            _F32_ZERO_NEG,
            0x00010000,  # subnormals representable in bf16
            0x00400000,
            0x80010000,
            0x80400000,
            0x00800000,  # FLT_MIN
            0x80800000,
            0xBF800000,  # -1.0 filler
            _F32_INF_NEG,
            _F32_NAN_NEG,
        ]
        weights = torch.tensor([3, 3, 3, 3, 3, 3, 1, 3, 20, 3, 3], dtype=torch.float64)
        pick = torch.multinomial(weights, num, replacement=True, generator=generator)
        bits = torch.tensor(pool, dtype=torch.int64)[pick]
    else:
        raise ValueError(population)
    if bf16:
        bits = bits & 0xFFFF0000
    return bits


def _ftz_bits(bits):
    """Flush subnormals to a zero of the same sign."""
    is_sub = ((bits & 0x7F800000) == 0) & ((bits & 0x007FFFFF) != 0)
    return torch.where(is_sub, bits & 0x80000000, bits)


def _srca_bits(bits):
    """What a bf16 operand looks like after the SrcA path of the top32_rm unpack (measured on
    BH): NaN becomes an infinity of the same sign, and -0 and every subnormal become +0.
    """
    is_nan = ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)
    bits = torch.where(is_nan, bits & 0xFF800000, bits)
    return torch.where((bits & 0x7F800000) == 0, torch.zeros_like(bits), bits)


def _check_edge_top32(
    in_bits, idx_in, out_bits, out_idx, allow_ftz, what, srca_path=False
):
    """Values: exactly the top 32 in SFPSWAP's order. Indices: valid, distinct, bit-matching.

    `srca_path`: the input went through SrcA (the 16-bit branch), so the golden is computed on
    what SrcA delivers to Dest rather than on the raw bits; the sort itself is the same.
    """
    if srca_path:
        policies = [("srca", _srca_bits(in_bits))]
    else:
        policies = [("raw", in_bits)] + (
            [("ftz", _ftz_bits(in_bits))] if allow_ftz else []
        )
    errors = []
    for name, view in policies:
        order = torch.argsort(_sign_mag_key(view), descending=True, stable=True)[:TOP_K]
        golden_bits = view[order]
        if not torch.equal(golden_bits, out_bits):
            errors.append(
                f"{name}: values device={[hex(int(b)) for b in out_bits]} golden={[hex(int(b)) for b in golden_bits]}"
            )
            continue
        idx = out_idx.to(torch.int64)
        if (
            not torch.equal(out_idx, idx.to(out_idx.dtype))
            or (idx < 0).any()
            or (idx >= view.numel()).any()
        ):
            errors.append(
                f"{name}: indices out of range or non-integer: {out_idx.tolist()}"
            )
            continue
        pos = torch.searchsorted(idx_in.to(torch.int64), idx)
        if len(set(idx.tolist())) != TOP_K:
            errors.append(f"{name}: duplicate indices {idx.tolist()}")
            continue
        if not torch.equal(view[pos], out_bits):
            errors.append(
                f"{name}: an index does not point at an input equal to its value: {idx.tolist()}"
            )
            continue
        return name
    raise AssertionError(f"{what}\n  " + "\n  ".join(errors))


def _read_top32_bits(configuration, is_32bit):
    """The 32 packed values and 32 packed indices, as raw bits (NaN sign/payload preserved)."""
    raw = configuration.variant_stimuli.collect_raw_result_bytes(
        TestConfig.TENSIX_LOCATION
    )
    ds = 4 if is_32bit else 2
    dtype = np.uint32 if is_32bit else np.uint16
    vals = np.frombuffer(raw[: TOP_K * ds], dtype=dtype).astype(np.int64)
    idxs = np.frombuffer(
        raw[ELEMENTS_PER_TILE * ds : (ELEMENTS_PER_TILE + TOP_K) * ds], dtype=dtype
    ).astype(np.int64)
    if not is_32bit:
        vals, idxs = vals << 16, idxs << 16
    return torch.from_numpy(vals), _bits_to_f32(torch.from_numpy(idxs))


EDGE_POPULATIONS = ["ties_specials", "random_ties", "subnormal"]


@parametrize(
    formats=FORMATS,
    row_elements=[64, 128, 256],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    population=EDGE_POPULATIONS,
)
def test_top32_rm_edge_values(formats, row_elements, dest_acc, population):
    """The plain mode on ties / +-0 / +-NaN / +-inf (and subnormals), bf16 and fp32."""
    is_32bit = formats.input_format.is_32_bit()
    if is_32bit and dest_acc == DestAccumulation.No:
        pytest.skip("32-bit datums need fp32 Dest")

    generator = torch.Generator().manual_seed(row_elements * 7 + len(population))
    in_bits = _edge_value_bits(row_elements, population, generator, bf16=not is_32bit)
    values = _bits_to_f32(in_bits)
    indices = torch.arange(row_elements, dtype=torch.float32)

    configuration = TestConfig(
        "sources/top32_rm_test.cpp",
        formats,
        templates=[
            TOP32_RM(
                row_elements=row_elements,
                datum_bytes=4 if is_32bit else 2,
                top_min=False,
                top32_mode=0,
            )
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            _pad_to_tile(values),
            formats.input_format,
            _pad_to_tile(indices),
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=2,
        ),
        unpack_to_dest=is_32bit,
        dest_acc=dest_acc,
    )
    configuration.run()
    out_bits, out_idx = _read_top32_bits(configuration, is_32bit)
    _check_edge_top32(
        in_bits,
        indices,
        out_bits,
        out_idx,
        allow_ftz=(population == "subnormal"),
        what=f"top32 {population} row={row_elements} dest_acc={dest_acc.name} {formats.input_format.name}",
        srca_path=not is_32bit,
    )


@parametrize(
    formats=PRE_SORTED_FORMATS,
    row_elements=[1024, 2048, 1088],
    population=EDGE_POPULATIONS,
)
def test_top32_rm_pre_sorted_edge_values(formats, row_elements, population):
    """prep / combine / final (and the tail) on ties / +-0 / +-NaN / +-inf (and subnormals).

    The input contract still holds: each run of 32 is sorted descending, here in SFPSWAP's own
    total order, with ties left adjacent.
    """
    generator = torch.Generator().manual_seed(row_elements * 13 + len(population))
    in_bits = _edge_value_bits(row_elements, population, generator, bf16=False)
    runs = in_bits.reshape(-1, TOP_K)
    order = torch.argsort(_sign_mag_key(runs), dim=1, descending=True, stable=True)
    in_bits = torch.gather(runs, 1, order).reshape(-1)
    values = _bits_to_f32(in_bits)
    indices = torch.arange(row_elements, dtype=torch.float32)
    tiles_per_operand = -(-row_elements // ELEMENTS_PER_TILE)

    configuration = TestConfig(
        "sources/top32_rm_test.cpp",
        formats,
        templates=[
            TOP32_RM(
                row_elements=row_elements,
                datum_bytes=4,
                top_min=False,
                top32_mode=1,
            )
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            values,
            formats.input_format,
            indices,
            formats.input_format,
            formats.output_format,
            tile_count_A=tiles_per_operand,
            tile_count_B=tiles_per_operand,
            tile_count_res=2,
        ),
        unpack_to_dest=True,
        dest_acc=DestAccumulation.Yes,
    )
    configuration.run()
    out_bits, out_idx = _read_top32_bits(configuration, True)
    _check_edge_top32(
        in_bits,
        indices,
        out_bits,
        out_idx,
        allow_ftz=(population == "subnormal"),
        what=f"pre-sorted top32 {population} row={row_elements}",
    )
