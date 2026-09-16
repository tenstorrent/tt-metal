# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Packer W-stride restore test for the custom_mm family's block_uninit (Blackhole only).

``custom_mm_block_uninit`` and ``compressed_custom_mm_block_uninit``
(api/compute/experimental/{custom_mm,compressed_custom_mm}.h) reached main via tt-metal
#52727. **As merged, both do exactly one thing:**

    dense_packing -> W-stride back to the default 64-row tile-to-tile spacing

Scope changed under this test, and that is worth stating plainly. Earlier revisions of
#52727 also restored the tile-pack MOP -- first unconditionally, then behind a
``restore_tile_pack_mop`` flag -- and this file was originally written against that,
sweeping both polarities of the flag. **Neither restore survived review.** Main has no MOP
restore and no such flag; the fused caller is expected to pair ``pack_block_contiguous_init``
with its own uninit instead. The MOP-restore coverage is therefore gone, because the
behaviour is gone, and what replaces it is a test that the uninit does *not* touch the MOP.

Note what the driver does NOT do: it replicates the uninit body rather than calling either
compute-API function, because a tt-llk test cannot include tt_metal/hw/inc/api/compute. So
this pins the behaviour the two headers share, and a divergence between them is what it
cannot catch. ``test_custom_mm_uninit_parity.py`` guards that textually;
catching it properly needs a metal-side test calling the real entry points.

How it works
------------
Three runs, per the header comment in sources/custom_mm_uninit_restore_test.cpp:

  run 0  Default pack baseline + the dense W-stride if selected, then the
         block-contiguous packer MOP swapped in -- what a caller's
         ``pack_block_contiguous_init`` does. Output not asserted.
  uninit The function under test: the conditional W-stride write, and nothing else.
  run 1  Plain per-tile ``_llk_pack_<PackMode::Default>``, no packer re-init, so it packs
         through whatever the uninit left.

``BLOCK_MOP_NUM_FACES`` is what decides which of the two packer states run 1 measures, and
the two tests below use it deliberately:

  4 faces  the geometry run 1 wants, so the MOP is not a confound and run 1 is correct
           exactly when the W-stride was restored. Used by the positive test and by the
           skip-uninit control.
  2 faces  a 16x32 tiny tile, a geometry run 1 does not want. Run 1 can only come back
           correct if something reinstalled the Default MOP -- which main's uninit must
           not do. Used by the leaves-the-caller-MOP-installed test.

``skip_uninit`` is a negative control rather than a supported call: it drops the uninit
entirely, so "the stride was restored" cannot be confused with "nothing needed restoring".
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


def _run(formats, dest_acc, dense_packing, skip_uninit, block_mop_num_faces=4):
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
def test_custom_mm_uninit_restores_dense_wstride(
    request, formats, dest_acc, dense_packing
):
    """The uninit must put the 64-row tile-to-tile W-stride back.

    This is the whole of what ``*_block_uninit`` does as merged. The run-0 block MOP is
    programmed at the geometry run 1 needs (4 faces), so the MOP is not a confound and
    run 1 is correct exactly when the stride was restored. With ``dense_packing=False``
    there is nothing to restore and run 1 must be correct trivially -- that arm is the
    baseline proving the driver itself is sound.
    """
    _skip_unsupported(formats, dest_acc)
    if dense_packing and formats.output_format.is_32_bit():
        request.node.add_marker(
            pytest.mark.xfail(reason=_DENSE_FP32_XFAIL, strict=False)
        )

    golden, device = _run(formats, dest_acc, dense_packing, skip_uninit=False)

    assert passed_test(golden, device, formats.output_format), (
        f"custom_mm/compressed_custom_mm _block_uninit<dense_packing={dense_packing}> "
        "did not leave the packer able to pack plain tiles: the following plain "
        "_llk_pack_ did not reproduce the DEST tiles"
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_custom_mm_uninit_is_load_bearing(formats, dest_acc):
    """Negative control: drop the uninit and the dense stride must still be wrong.

    Without this, "the stride was restored" cannot be told apart from "nothing needed
    restoring". Only ``dense_packing=True`` is driven, because that is the only case in
    which the uninit writes anything at all.
    """
    _skip_unsupported(formats, dest_acc)

    golden, device = _run(formats, dest_acc, dense_packing=True, skip_uninit=True)

    assert not passed_test(golden, device, formats.output_format, print_errors=False), (
        "run 1 reproduced the DEST tiles even though the uninit was skipped and the "
        "dense 32-row W-stride was left in place. Either the stride no longer affects "
        "the pack, or the driver is not actually applying it -- in both cases the "
        "positive test above proves nothing."
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_custom_mm_uninit_leaves_the_caller_mop_installed(formats, dest_acc):
    """The uninit must NOT touch the packer MOP -- that is the merged contract.

    An earlier revision of #52727 restored the tile-pack MOP here, first unconditionally
    and then behind a ``restore_tile_pack_mop`` flag; neither survived review. Main leaves
    the MOP to whichever init programmed it, and the fused caller is expected to pair
    ``pack_block_contiguous_init`` with its own uninit.

    So this pins the *absence* of a restore, which is what a future change re-adding one
    would break. The run-0 block MOP is programmed at 2 faces (a 16x32 tiny tile), a
    geometry run 1 does not want: if the uninit reinstalled the Default 32x32/4-face MOP
    the pack would come back correct, and it must not. ``dense_packing`` is off so the
    stride is not a second reason to fail.
    """
    _skip_unsupported(formats, dest_acc)

    golden, device = _run(
        formats,
        dest_acc,
        dense_packing=False,
        skip_uninit=False,
        block_mop_num_faces=2,
    )

    assert not passed_test(golden, device, formats.output_format, print_errors=False), (
        "run 1 packed correctly through a 2-face block MOP that nothing reinstalled, "
        "which means the uninit restored the Default tile-pack MOP. Main's uninit does "
        "not do that, and callers inheriting the block MOP across ops depend on it not "
        "doing it -- see the note in sources/custom_mm_uninit_restore_test.cpp."
    )
