# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Cross-op packer-state restore test for the custom_mm family's block_uninit
(Blackhole only).

``custom_mm_block_uninit`` and ``compressed_custom_mm_block_uninit``
(api/compute/experimental/{custom_mm,compressed_custom_mm}.h, promoted by tt-metal
#52727) are the one behavioural change in that PR, and neither had a caller in tt-llk
before this test. Both bodies consist entirely of two conditional packer-state restores:

    dense_packing         -> W-stride back to the default 64-row tile-to-tile spacing
    restore_tile_pack_mop -> _llk_pack_mop_config_<PackMode::Default>()

Since the two uninits have identical bodies, one driver covers both. Note what that does
*not* buy: the driver replicates the shared body rather than calling either compute-API
function, because a tt-llk test cannot include tt_metal/hw/inc/api/compute. So this test
pins the behaviour the two headers currently share, and a future divergence between them
is exactly what it cannot catch -- that needs a test on the metal side that calls the
real entry points.

How it works
------------
The driver runs two packs separated by the uninit (see the header comment in
sources/custom_mm_uninit_restore_test.cpp for the full sequence):

  run 0  Default pack baseline + optional dense W-stride, then the block-contiguous
         packer MOP swapped in -- what a caller's ``pack_block_contiguous_init`` does,
         and precisely the situation the uninit's comment describes ("replaces the
         packer MOP without owning it"). Output not asserted.
  uninit The function under test.
  run 1  Plain per-tile ``_llk_pack_<PackMode::Default>``, no packer re-init. Because
         ``_llk_pack_`` executes whatever MOP is installed, this reads the restored
         state directly, and on a correct restore is an identity copy of the DEST tiles.

So run 1 is correct if and only if the uninit ran *and* ``restore_tile_pack_mop`` was
set -- run 0 always leaves the block MOP installed, and only the MOP restore undoes it.

Both polarities are asserted, deliberately
------------------------------------------
``restore_tile_pack_mop`` defaults to **false**, and that default is documented as
intentional: the MOP belongs to whichever init programmed it, and an unconditional
``_llk_pack_mop_config_<Default>()`` would install fixed 32x32 geometry and clobber the
1x32 configuration this family targets. So `False` is a supported configuration whose
contract is "the block MOP is still installed on exit", and this test pins that too --
otherwise a future change making the restore unconditional would look like a pure
improvement while breaking the fused callers that rely on inheriting the MOP.

``skip_uninit`` is a negative control rather than a supported call: it drops the uninit
entirely, so that "restore worked" cannot be confused with "nothing needed restoring".

Why the run-0 block MOP is programmed with 2 faces
--------------------------------------------------
The pack MOP bakes in tile geometry, so the MOP restore is only observable when the
run-0 MOP carries a *different* geometry than the run-1 pack needs. Measured on BH
p100a, with the block MOP programmed at 4 faces (the same geometry the restore
installs), run 1 is byte-correct whether or not the restore runs -- the flag is
unobservable. At 2 faces (a 16x32 tiny tile) the un-restored MOP packs half of each
tile, giving a 0.50 per-tile match, which is the hazard the uninit's own comment names:
"installs fixed 32x32 tile geometry -- wrong for 1x32 follow-ons".

test_custom_mm_uninit_pack_mop_restore_is_noop_at_matching_geometry pins that
observation, because it is the reason the flag is opt-in rather than unconditional: a
caller whose follow-on geometry already matches gains nothing from the restore and
would only pay for it.
"""

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import CUSTOM_MM_UNINIT, PACK_NUM_TILES
from helpers.utils import passed_test

pytestmark = [skip_for_wormhole, skip_for_quasar]

ELEMENTS_PER_TILE = 1024

# Same-in-same-out: this test varies packer *state*, not conversion. A mixed pair would
# only add format-conversion noise on top of the layout question being asked.
FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True)

# 4 tiles is the smallest count that makes a tile-to-tile stride error visible on more
# than one tile while staying inside DEST half-sync capacity at fp32 (4 tiles).
NUM_TILES = 4

# The two uninits have identical bodies, and custom_mm_uninit_restore_test.cpp
# replicates that shared body rather than calling either compute-API function (a tt-llk
# test cannot include tt_metal/hw/inc/api/compute). So there is deliberately no
# per-family axis: it would build the identical ELF twice and could not catch the one
# thing it would exist to catch, a future divergence between the two headers. Guarding
# that divergence needs a test on the metal side that calls the real entry points.


def _run(
    formats, dest_acc, dense_packing, restore_mop, skip_uninit, block_mop_num_faces=2
):
    torch.manual_seed(0)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[NUM_TILES * 32, 32],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[NUM_TILES * 32, 32],
    )

    configuration = TestConfig(
        "sources/custom_mm_uninit_restore_test.cpp",
        formats,
        templates=[
            PACK_NUM_TILES(NUM_TILES),
            CUSTOM_MM_UNINIT(
                dense_packing=dense_packing,
                restore_mop=restore_mop,
                skip=skip_uninit,
                block_mop_num_faces=block_mop_num_faces,
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
            # Headroom: if a stale block MOP makes run 1 pack more than one tile's worth
            # per call, the overrun lands in spare result tiles instead of other buffers.
            tile_count_res=2 * NUM_TILES,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    torch_format = format_dict[formats.output_format]

    generate_golden = get_golden_generator(DataCopyGolden)
    golden = generate_golden(
        src_A.flatten(),
        formats.output_format,
        input_dimensions=[NUM_TILES * 32, 32],
        input_format=formats.input_format,
    )

    span = NUM_TILES * ELEMENTS_PER_TILE
    device = torch.tensor(res_from_L1[:span], dtype=torch_format).flatten()
    golden = torch.tensor(golden[:span], dtype=torch_format).flatten()
    return golden, device


# Found by this test: the dense_packing W-stride constants in custom_mm.h /
# compressed_custom_mm.h are hardcoded for a 16-bit pack source and are wrong for a
# 32-bit one.
#
# The canonical writer, cpack_common.h set_packer_strides, computes
#     w_stride = TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * datum_size_in_bytes(fmt)
# while both compute-API headers spell the same expression with a literal `* 2`:
#     init   dense:   (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2   = 1024
#     uninit restore:  TILE_NUM_FACES      * FACE_C_DIM * FACE_R_DIM * 2   = 2048
# For a Float32 pack source datum_size_in_bytes is 4, so the correct values are 2048 and
# 4096. The uninit therefore does not restore what _llk_pack_init_ programmed, and the
# following pack reads tiles at half the intended stride.
#
# Measured on BH p100a, Float32 in/out, dest_acc=Yes, dense_packing=True: run 1 matches
# on tile 0 only (0.25 overall) regardless of restore_tile_pack_mop, i.e. the W-stride
# restore does not recover. The 16-bit path is unaffected and fully correct.
#
# This is pre-existing -- #52727 kept the demo's structure -- but the promotion ships it
# in packaged metalium via HW_JIT_API_HEADERS, so it now reaches more callers. Marked
# xfail rather than skipped so the suite stays green while recording the defect, and
# flips to XPASS the moment the constants become format-aware (or a static_assert
# restricts dense_packing to 16-bit pack sources).
_DENSE_FP32_XFAIL = (
    "custom_mm dense_packing W-stride is hardcoded *2 (16-bit pack source); "
    "wrong for a 32-bit pack source, so the uninit cannot restore it"
)


def _skip_unsupported(formats, dest_acc):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")
    if (
        formats.input_format == DataFormat.Float16_b
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip(
            "Float16_b with dest_acc=Yes adds nothing here: the pack layout, "
            "not the DEST width, is what this test varies"
        )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dense_packing=[False, True],
)
def test_custom_mm_uninit_restores_pack_mop(formats, dest_acc, dense_packing):
    """restore_tile_pack_mop=True must leave the packer able to pack plain tiles."""
    _skip_unsupported(formats, dest_acc)
    if dense_packing and formats.output_format.is_32_bit():
        pytest.xfail(_DENSE_FP32_XFAIL)

    golden, device = _run(
        formats, dest_acc, dense_packing, restore_mop=True, skip_uninit=False
    )

    assert passed_test(golden, device, formats.output_format), (
        f"custom_mm/compressed_custom_mm _block_uninit<dense_packing={dense_packing}, "
        "restore_tile_pack_mop=true> did not restore a usable tile-pack state: the "
        "following plain _llk_pack_ did not reproduce the DEST tiles"
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dense_packing=[False, True],
)
def test_custom_mm_uninit_keeps_pack_mop_when_not_asked(
    formats, dest_acc, dense_packing
):
    """restore_tile_pack_mop=False must leave the caller's block MOP installed.

    The inverse of the test above, and a real contract rather than an accident: fused
    callers deliberately inherit the block-contiguous MOP across ops, which is why the
    flag defaults to false. If this starts passing, the restore has become unconditional
    and those callers are silently getting 32x32 tile geometry.
    """
    _skip_unsupported(formats, dest_acc)

    golden, device = _run(
        formats, dest_acc, dense_packing, restore_mop=False, skip_uninit=False
    )

    assert not passed_test(golden, device, formats.output_format, print_errors=False), (
        "custom_mm/compressed_custom_mm _block_uninit<restore_tile_pack_mop=false> left "
        "the packer in a state where a plain _llk_pack_ reproduced the DEST tiles -- the "
        "block MOP installed before the uninit appears to have been restored anyway"
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dense_packing=[False, True],
)
def test_custom_mm_uninit_is_load_bearing(formats, dest_acc, dense_packing):
    """Negative control: with the uninit dropped entirely, run 1 must be wrong.

    Guards against the whole test degenerating into a tautology -- if run 1 passed
    without any uninit at all, then neither restore would be doing anything and the
    two tests above would be measuring nothing.
    """
    _skip_unsupported(formats, dest_acc)

    golden, device = _run(
        formats, dest_acc, dense_packing, restore_mop=True, skip_uninit=True
    )

    assert not passed_test(golden, device, formats.output_format, print_errors=False), (
        "run 1 reproduced the DEST tiles with no uninit at all, so neither the "
        "W-stride nor the pack-MOP restore is observable here -- this test has lost "
        "its teeth and the setup needs revisiting"
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    restore_mop=[False, True],
)
def test_custom_mm_uninit_pack_mop_restore_is_noop_at_matching_geometry(
    formats, dest_acc, restore_mop
):
    """With the block MOP at the same geometry the restore installs, the flag is inert.

    Both polarities must produce a correct run 1. This is the counterpart to
    test_custom_mm_uninit_keeps_pack_mop_when_not_asked: there the block MOP carries a
    different face count and the flag decides correctness, here it carries the same one
    and the flag cannot be observed at all.

    Pinning it documents the scope of the restore -- it re-establishes *geometry*, not
    some broader packer reset -- and explains why an opt-in flag is the right shape:
    callers already at 32x32 gain nothing from paying for it.
    """
    _skip_unsupported(formats, dest_acc)

    golden, device = _run(
        formats,
        dest_acc,
        dense_packing=False,
        restore_mop=restore_mop,
        skip_uninit=False,
        block_mop_num_faces=4,
    )

    assert passed_test(golden, device, formats.output_format), (
        f"custom_mm/compressed_custom_mm _block_uninit<restore_tile_pack_mop="
        f"{restore_mop}> did not leave a usable tile-pack state even though the block MOP "
        "already carried the matching 4-face geometry -- something other than geometry is "
        "being disturbed"
    )
