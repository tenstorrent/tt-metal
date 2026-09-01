# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Behaviour test for ``set_dst_write_addr_offset`` (Blackhole only).

tt-metal #52713 extracted this helper out of ``ckernel_sfpu_topk_xl.h`` and
``ckernel_sfpu_deepseek_top32_rm.h`` into
``sfpu/experimental/ckernel_sfpu_set_dst_write_addr_offset.h``. Its sibling test
``test_sort_headers_coexist.py`` covers the half that motivated the extraction -- both
headers compiling into one translation unit -- and deliberately asserts nothing about
what the helper does, because the datacopy it uses to read DEST back reprograms
``DEST_TARGET_REG_CFG_MATH_Offset_ADDR32`` itself before touching DEST. That left the
helper's actual behaviour untested. This file tests it.

What is asserted
----------------
The helper writes an absolute Dst write address, ``addr + get_dest_buffer_base()``, in
units of Dst ROWS -- and one 32x32 tile is 64 rows, since
``math::set_dst_write_addr<Tile32x32>(tile_index)`` computes ``tile_index << 6``. So at a
multiple of 64 the helper is doing exactly what the LLK's own addressing function does,
which gives an exact, layout-agnostic assertion:

    helper(N * 64) at dst_index 0   ==   no helper at dst_index N

That is ``test_matches_dst_index``, and it is the load-bearing one. ``test_moves_negated_face``
then pins the same thing positionally against a datacopy-only baseline -- the negated face
must appear in tile N and nowhere else -- which is the assertion that would catch a helper
that wrote a plausible-but-wrong address, the failure mode a value-only check cannot see.

Both real call patterns are covered, because both are multiples-of-64 plus a small delta:
``tile_offset`` (``dst_index << 6``) in deepseek_top32_rm, and ``tile_offset + 2`` -- the
column-group flip -- in topk_xl and deepseek both. The ``+ 2`` sub-tile case gets a weaker
claim, see ``test_sub_tile_delta``.

Verified to have teeth, not just to pass
---------------------------------------
Both assertions were checked against deliberately broken helpers on BH p100a:

  * offset discarded (``dst_index = get_dest_buffer_base()``) -> 10 of 14 fail.
  * rows read as datums (``addr * 32 + base``) -> the sub-tile spill check fires.

Note which 4 survive the first mutant: the ``tile=0`` variants. At offset 0 the helper is
a no-op even when broken, so those variants cannot detect a discarded offset -- they are
there to pin that offset 0 does NOT move the write, which is a separate claim. Read a
``tile=0``-only pass as no evidence about the helper.

Not covered: the ``LLK_ASSERT(addr < DEST_REGISTER_HALF_SIZE)``. Nothing in this suite
expects an LLK assert -- conftest reports ``LLKAssertException`` as a failure -- and
tripping one mid-kernel risks leaving the device wedged for whatever runs next.
"""

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DST_WRITE_ADDR_OFFSET, PACK_NUM_TILES
from helpers.utils import passed_test

pytestmark = [skip_for_wormhole, skip_for_quasar]

# Same-in-same-out: this test is about a Dst address, not format conversion.
FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True)

# 3 tiles: enough that a whole-tile rebase to tile 1 has both a tile below it and a tile
# above it that must stay pristine, which is what makes "landed in the right place"
# different from "landed somewhere else". Fits DEST half-sync capacity at fp32 (4 tiles).
NUM_TILES = 3

ELEMENTS_PER_TILE = 1024
# One face, the unit the SFPU body negates.
FACE_ELEMENTS = 256

# Dst rows per 32x32 tile -- DstTileSizeLog2[DstTileShape::Tile32x32] == 6.
ROWS_PER_TILE = 64
# topk_xl's / deepseek's odd-column group flip.
ODD_COL_OFFSET_ROWS = 2


def _skip_unsupported(formats, dest_acc):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")
    if (
        formats.input_format == DataFormat.Float16_b
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip(
            "Float16_b with dest_acc=Yes adds nothing here: the Dst write address, "
            "not the DEST width, is what this test varies"
        )


def _build(
    formats,
    dest_acc,
    *,
    offset_enabled=True,
    offset_rows=0,
    sfpu_dst_index=0,
    sfpu_enabled=True,
):
    """Build (do not run) one variant. Returns (configuration, src_A).

    Strictly positive stimuli: the SFPU body negates, so a negative datum in the result
    can only have come from the body. That is what makes "which datums moved" readable
    without a tolerance.
    """
    torch.manual_seed(0)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[NUM_TILES * 32, 32],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[NUM_TILES * 32, 32],
        spec_A=StimuliSpec.uniform(low=0.5, high=4.0),
    )

    configuration = TestConfig(
        "sources/set_dst_write_addr_offset_test.cpp",
        formats,
        templates=[
            PACK_NUM_TILES(NUM_TILES),
            DST_WRITE_ADDR_OFFSET(
                offset_enabled=offset_enabled,
                offset_rows=offset_rows,
                sfpu_dst_index=sfpu_dst_index,
                sfpu_enabled=sfpu_enabled,
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
            tile_count_res=NUM_TILES,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
    )
    return configuration, src_A


def _finish(configuration, formats):
    """Run a prepared variant, returning the packed result as one flat fp32 tensor."""
    res_from_L1 = configuration.run().result[: NUM_TILES * ELEMENTS_PER_TILE]
    torch_format = format_dict[formats.output_format]
    return torch.tensor(res_from_L1, dtype=torch_format).flatten().to(torch.float32)


def _run_pair(first, second):
    """prepare() both variants before running either.

    ``prepare()`` is the build half of ``run()``, and under ``--compile-producer``
    ``run()`` skips as soon as the first variant is built -- so the second would never
    emit its ELF. Same reason as ``test_topk_xl_rebuild_ascending``.
    """
    for configuration, _ in (first, second):
        configuration.prepare()


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    tile=[0, 1, 2],
)
def test_set_dst_write_addr_offset_matches_dst_index(formats, dest_acc, tile):
    """helper(tile * 64) at dst_index 0 == no helper at dst_index tile.

    The exact assertion, and the reason this test does not need to model DEST layout: at a
    multiple of 64 rows the helper is computing the same address as
    ``math::set_dst_write_addr<Tile32x32>(tile)``, which is what
    ``_llk_math_eltwise_sfpu_start_`` uses. Anything other than bit-equality here means
    the helper's arithmetic (``addr + get_dest_buffer_base()``) or its register write has
    diverged from the LLK's own.

    Bit-equality is the right bar rather than a tolerance: both sides negate the same
    datacopy output on the same hardware, so the results are either identical or the
    helper put the write somewhere else.
    """
    _skip_unsupported(formats, dest_acc)

    via_helper = _build(
        formats,
        dest_acc,
        offset_enabled=True,
        offset_rows=tile * ROWS_PER_TILE,
        sfpu_dst_index=0,
    )
    via_dst_index = _build(formats, dest_acc, offset_enabled=False, sfpu_dst_index=tile)
    _run_pair(via_helper, via_dst_index)

    helper_result = _finish(via_helper[0], formats)
    dst_index_result = _finish(via_dst_index[0], formats)

    mismatches = int((helper_result != dst_index_result).sum())
    assert mismatches == 0, (
        f"set_dst_write_addr_offset({tile * ROWS_PER_TILE}) did not land where "
        f"dst_index={tile} lands: {mismatches}/{helper_result.numel()} datums differ. "
        "The helper's absolute Dst write address has diverged from "
        "math::set_dst_write_addr<Tile32x32>."
    )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    tile=[0, 1, 2],
)
def test_set_dst_write_addr_offset_moves_negated_face(formats, dest_acc, tile):
    """The negated face appears in tile `tile` and nowhere else.

    The positional half. ``test_matches_dst_index`` proves the helper agrees with the LLK's
    addressing, but both could be wrong together; this pins the absolute position against
    a datacopy-only baseline. It is also what distinguishes a correct rebase from no
    rebase at all -- the trap the first version of the coexistence test fell into, where
    the value looked right because the offset had simply been discarded.
    """
    _skip_unsupported(formats, dest_acc)

    baseline = _build(formats, dest_acc, sfpu_enabled=False)
    negated = _build(
        formats,
        dest_acc,
        offset_enabled=True,
        offset_rows=tile * ROWS_PER_TILE,
        sfpu_dst_index=0,
    )
    _run_pair(baseline, negated)

    baseline_result = _finish(baseline[0], formats)
    negated_result = _finish(negated[0], formats)

    # The baseline is a plain datacopy, so it must match the input as the device saw it.
    generate_golden = get_golden_generator(DataCopyGolden)
    golden = generate_golden(
        baseline[1].flatten(),
        formats.output_format,
        input_dimensions=[NUM_TILES * 32, 32],
    )
    assert passed_test(
        torch.tensor(
            golden[: NUM_TILES * ELEMENTS_PER_TILE],
            dtype=format_dict[formats.output_format],
        ).flatten(),
        torch.tensor(
            baseline_result, dtype=format_dict[formats.output_format]
        ).flatten(),
        formats.output_format,
    ), "the SFPU_ENABLED=false baseline is not a clean datacopy; the rest of this test reads from it"

    expected = baseline_result.clone()
    start = tile * ELEMENTS_PER_TILE
    expected[start : start + FACE_ELEMENTS] *= -1

    mismatches = int((negated_result != expected).sum())
    if mismatches:
        moved = (negated_result != baseline_result).nonzero().flatten()
        where = (
            f"datums {int(moved[0])}..{int(moved[-1])}" if moved.numel() else "nothing"
        )
        raise AssertionError(
            f"set_dst_write_addr_offset({tile * ROWS_PER_TILE}) should have negated face 0 "
            f"of tile {tile}, i.e. datums {start}..{start + FACE_ELEMENTS - 1}, leaving the "
            f"other {NUM_TILES - 1} tiles untouched. {mismatches} datums disagree; the "
            f"negate actually touched {where}."
        )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_set_dst_write_addr_offset_sub_tile_delta(formats, dest_acc):
    """A 2-row delta is observable and stays sub-tile.

    ``tile_offset + 2`` is the column-group flip both sort families use. Deliberately a
    weaker claim than the two tests above: 2 Dst rows is 32 datums, and where those land
    in the packed output depends on the face layout the packer walks, which this test does
    not model. What it does assert is that the delta reaches the hardware at all -- the
    result differs from offset 0 -- and that it stays inside the first tile rather than
    spilling into tile 1, which is what a rows-vs-datums units mix-up would do (a 2-datum
    or 2-face reading of the argument both put the write somewhere else).
    """
    _skip_unsupported(formats, dest_acc)

    at_zero = _build(formats, dest_acc, offset_enabled=True, offset_rows=0)
    at_two = _build(
        formats, dest_acc, offset_enabled=True, offset_rows=ODD_COL_OFFSET_ROWS
    )
    _run_pair(at_zero, at_two)

    zero_result = _finish(at_zero[0], formats)
    two_result = _finish(at_two[0], formats)

    assert not torch.equal(zero_result, two_result), (
        f"set_dst_write_addr_offset({ODD_COL_OFFSET_ROWS}) produced the same output as "
        "offset 0, so the sub-tile delta never reached the Dst write pointer"
    )

    # 2 rows into a 64-row tile: the negated face cannot reach tile 1.
    spilled = int((two_result[ELEMENTS_PER_TILE:] < 0).sum())
    assert spilled == 0, (
        f"a {ODD_COL_OFFSET_ROWS}-row offset negated {spilled} datums beyond tile 0, "
        "which a 2-row shift of a one-face write inside a 64-row tile cannot reach -- "
        "the argument is being read in the wrong units"
    )
