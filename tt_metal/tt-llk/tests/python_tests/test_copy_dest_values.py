# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``copy_dest_values`` (DEST tile -> DEST tile) with the destination slot occupied.

``test_dest_copy.py::test_dump_dest`` covers an L1 -> DEST -> L1 round trip
through the RISC-V debug window with nothing else resident in DEST. This test
covers the case the primitive actually exists for: a real op has just written
both the source and the destination DEST slot, and the copy must overwrite the
destination's existing contents.

The kernel's MATH section is a literal expansion of the public compute API
``copy_dest_values<DATA_FORMAT>(idst_in, idst_out)``
(``api/compute/copy_dest_values.h``) -- same init, same ``SFPU_BINARY_CALL``,
same ``VectorMode::RC`` -- so a failure here is a failure of the shipped API,
not of a re-derivation of it.

Each variant grades two separate claims:

``the source slot holds the source tile`` (self-check)
    Holds whether or not the copy works, so it fails only if the prelude, the
    DEST tile addressing or the pack target something other than what the test
    believes. Without it, a kernel-authoring mistake would be
    indistinguishable from a copy defect. Graded with a tolerance, not
    bit-exactly: the A2D datacopy that fills DEST routes through SrcA, whose
    10-bit mantissa is narrower than fp32, so a host-side fp32 golden is
    unreachable by construction. It only has to be unambiguous about *which*
    tile landed, and the two input tiles have opposite signs.

``the destination slot equals the source slot`` (the contract)
    Graded bit-exactly, and DEST-against-DEST rather than against a host
    golden: both sides went through the same datapath, so ``copy_dest_values``
    owes exact equality regardless of what that datapath did to the values.

``occupancy`` selects whether the destination slot was written by the preceding
datacopy (``occupied``) or left untouched (``untouched``), and ``direction``
runs the copy both ways. Together those separate three failure modes that look
alike from one variant: a store that never lands, a store that lands at the
load's offset, and a copy that only works into a slot no op has claimed.
"""

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, VectorMode, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    COPY_DEST_VALUES_FORMAT,
    SFPU_TILE_INDICES,
    TILE_COUNT,
    VECTOR_MODE,
)

# Input and output share the format: the copy is meant to be value-preserving,
# so any L1 conversion on the way in or out would blur the thing under test.
FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
]

# One 32x32 tile per DEST slot, all four faces -- VectorMode::RC covers them all.
TILE_DIMENSIONS = [32, 32]
ELEMENTS_PER_TILE = TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1]

#: (source DEST slot, destination DEST slot). Both directions, because "the
#: store never lands" and "the store lands at the load's offset" are
#: indistinguishable from a single direction.
DIRECTIONS = [(0, 1), (1, 0)]

#: Tiles the preceding datacopy writes. 2 leaves the destination slot holding
#: another tile's data when the copy runs; 1 leaves it unclaimed.
OCCUPANCY = {"occupied": 2, "untouched": 1}


def _dest_acc(output_format):
    """Native fp32 DEST is required whenever the output is Float32."""
    return (
        DestAccumulation.Yes
        if output_format == DataFormat.Float32
        else DestAccumulation.No
    )


@parametrize(
    formats=FORMATS,
    direction=DIRECTIONS,
    occupancy=list(OCCUPANCY),
)
def test_copy_dest_values(formats, direction, occupancy):
    src_slot, dst_slot = direction
    prefilled_tiles = OCCUPANCY[occupancy]

    if prefilled_tiles <= src_slot:
        pytest.skip(
            f"occupancy={occupancy} writes {prefilled_tiles} tile(s), which does not "
            f"reach the source slot DEST[{src_slot}] -- nothing defined to copy from"
        )

    # Two tiles of stimuli whichever way the copy runs, so the untouched-slot
    # variant reads the same source data as the occupied one and the two are
    # directly comparable. Distinct per-tile ranges make a slot mix-up loud:
    # tile 0 is in [1, 2), tile 1 in [-2, -1), so no value can be mistaken for
    # the other tile's, for zero, or for uninitialised DEST.
    input_dimensions = [2 * TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]]
    src_A, tile_cnt_A, _, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec.uniform(low=1.0, high=2.0),
    )
    src_A = src_A.clone()
    src_A[ELEMENTS_PER_TILE:] = -src_A[ELEMENTS_PER_TILE:]

    tiles_in = [
        src_A[i * ELEMENTS_PER_TILE : (i + 1) * ELEMENTS_PER_TILE] for i in range(2)
    ]

    configuration = TestConfig(
        "sources/copy_dest_values_test.cpp",
        formats,
        templates=[
            # The functor's DataFormat selects its SFPLOAD/SFPSTORE modifier, so
            # it tracks the DEST precision, which here equals the L1 format.
            COPY_DEST_VALUES_FORMAT(formats.output_format),
            VECTOR_MODE(VectorMode.RC),
        ],
        runtimes=[
            TILE_COUNT(prefilled_tiles),
            # copy_dest_value(dst_index_in, dst_index_out, unused): DST_IN0 is
            # the source slot, DST_IN1 the destination, DST_OUT unused.
            SFPU_TILE_INDICES(
                src0_tile_idx=src_slot, src1_tile_idx=dst_slot, dst_tile_idx=0
            ),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_A,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            # The kernel always packs DEST[0] and DEST[1] so either direction
            # can be graded without the host tracking which slot moved.
            tile_count_res=2,
            sfpu=True,
        ),
        dest_acc=_dest_acc(formats.output_format),
    )

    res_from_L1 = configuration.run().result

    assert (
        len(res_from_L1) == 2 * ELEMENTS_PER_TILE
    ), f"expected two {ELEMENTS_PER_TILE}-element output tiles, got {len(res_from_L1)}"

    res = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    tiles_out = [
        res[i * ELEMENTS_PER_TILE : (i + 1) * ELEMENTS_PER_TILE] for i in range(2)
    ]

    host_src = tiles_in[src_slot].to(torch.float32)
    host_dst = tiles_in[dst_slot].to(torch.float32)

    # Self-check first: if the source slot does not hold the source tile, the
    # failure is in this kernel's prelude/addressing and says nothing about
    # copy_dest_values. SrcA rounding is why this is allclose and not equal.
    got_src = tiles_out[src_slot].to(torch.float32)
    assert torch.allclose(got_src, host_src, rtol=1e-2, atol=1e-2), (
        f"source slot DEST[{src_slot}] does not hold input tile {src_slot} -- the "
        f"test's own prelude/addressing is wrong, so the destination check below "
        f"would be meaningless.\n"
        f"  DEST[{src_slot}]      {got_src[:4].tolist()}\n"
        f"  input tile {src_slot}  {host_src[:4].tolist()}"
    )

    # The contract: after the copy, the destination slot must be a bit-exact
    # image of the source slot. Both sides came off the same datapath, so no
    # tolerance is owed here.
    got = tiles_out[dst_slot]
    if torch.equal(got, tiles_out[src_slot]):
        return

    got_f = got.to(torch.float32)
    if prefilled_tiles > dst_slot and torch.allclose(
        got_f, host_dst, rtol=1e-2, atol=1e-2
    ):
        diagnosis = (
            f"DEST[{dst_slot}] still holds what the preceding datacopy left there, "
            f"so the SFPSTORE did not land in DEST[{dst_slot}]"
        )
    elif torch.allclose(got_f, host_src, rtol=1e-2, atol=1e-2):
        diagnosis = (
            f"DEST[{dst_slot}] holds the right tile but not a bit-exact image of "
            f"DEST[{src_slot}], so the copy went through a narrowing conversion"
        )
    else:
        diagnosis = f"DEST[{dst_slot}] holds neither DEST[{src_slot}] nor its own prior contents"

    pytest.fail(
        f"copy_dest_values<{formats.output_format.name}>({src_slot} -> {dst_slot}) "
        f"[occupancy={occupancy}]: {diagnosis}.\n"
        f"  DEST[{dst_slot}] after  {got_f[:4].tolist()}\n"
        f"  DEST[{src_slot}] after  {got_src[:4].tolist()}\n"
        f"  input tile {dst_slot}    {host_dst[:4].tolist()}\n"
        f"  elements differing from DEST[{src_slot}]: "
        f"{int((got != tiles_out[src_slot]).sum())} of {ELEMENTS_PER_TILE}"
    )
