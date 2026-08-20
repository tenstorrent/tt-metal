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

# Float16_b only: indices ride in the same format as the values, and bf16's 8 mantissa bits
# are what bound the sweep at 256. A wider format would lift that bound but would also send
# both operands down the 32-bit unpack-to-dest path, which is not the one the consumer uses
# for values -- and that path pads with zeros instead of -inf (see the module docstring).
FORMATS = input_output_formats([DataFormat.Float16_b], same=True)

# 64   one chunk, no merge across tiles at all -- phases_steps + merge + rebuild only
# 128  two full chunks, so the across-tiles merge runs once
# 160  the Metal dev test's tail case: two full chunks plus a 32-element num_faces=2 chunk
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
def test_top32_rm(formats, row_elements, dest_acc):
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
                datum_bytes=2,  # Float16_b
                top_min=False,
                mode=0,
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
        unpack_to_dest=False,
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
PRE_SORTED_ROW_ELEMENTS = [1024, 2048]

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

    rank = torch.arange(row_elements, dtype=torch.float32) % TOP_K
    run = torch.arange(row_elements, dtype=torch.int64) // TOP_K

    values = (TOP_K - rank) * _RANK_STEP + tiebreak[run]
    indices = torch.arange(row_elements, dtype=torch.float32)
    return values, indices


@parametrize(
    formats=PRE_SORTED_FORMATS,
    row_elements=PRE_SORTED_ROW_ELEMENTS,
)
def test_top32_rm_pre_sorted(formats, row_elements):
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

    tiles_per_operand = row_elements // ELEMENTS_PER_TILE

    configuration = TestConfig(
        "sources/top32_rm_test.cpp",
        formats,
        templates=[
            TOP32_RM(
                row_elements=row_elements,
                datum_bytes=4,  # Float32
                top_min=False,
                mode=1,
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
