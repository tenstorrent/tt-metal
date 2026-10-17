# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ADVANCE TEST: covers experimental LLK compressed_custom_mm (tt-metal#47554 / tt-blaze#1971), promoted into
# tt_llk_blackhole/llk_lib/experimental/ on main by #53295. The compute kernel includes the canonical headers; the
# demo-fork shadow tree this test was first written against no longer exists.
#
# compressed_custom_mm is the BFP-compressed-in1 sibling of custom_mm. Same documented contract as custom_mm
# (llk_math_compressed_custom_mm.h / llk_unpack_AB_compressed_custom_mm.h header banners):
#   in0 tile shape: [{1, 2, 4, 8}, 32]   (partial-row tile -> SrcB, bf16, reused across output width)
#   in1 tile shape: [32, 32]             (full tile -> SrcA, BFP-compressed: Bfp8_b / Bfp4_b / Bfp2_b)
#   rt_dim: 1;  ct_dim: 1..16;  kt_dim: even 2..256;  fidelity: LoFi only;  throttle: not supported
#
# Same two layout facts as custom_mm, and for the same reason -- these are the SAME primitives the silicon-validated
# test_matmul_custom_compressed.py drives, so this test now mirrors its host side:
#   - in0 is kt_dim*2 DENSELY packed in0_rows x 16 faces, not a run of padded 32x32 tiles (the unpacker's SrcB counter
#     stride is one face, datum_size * FACE_C_DIM * face_r_dim). See helpers/custom_mm_utils.pack_in0_faces.
#   - the output is a partial tile [in0_rows, ct_dim*32], packed with dense_packing (DEST tile stride 32 rows; the
#     zero-tile tests also run the 64-row non-dense layout) and read back through
#     helpers/custom_mm_utils.dense_result_rowmajor.
#
# Difference vs custom_mm: BOTH primitives take an extra base_address_meta argument (a buffer of packed 3-bit per-tile
# compression-format codes). We route it through the harness's optional third operand (buffer_C). The codes are CONTROL
# FLOW, not just numerics: code 0 means "zero tile", for which the unpacker emits no UNPACR at all and math only steps
# DEST over the tile's slot; a zero tile in the last column (and the finalize pass) instead waits on SrcA/SrcB valid,
# which is why an all-zero meta buffer hangs Math/Packer forever. The grid tests give every tile the one BFP format
# under test; the zero_tile tests place single zero tiles in a full-width block.
#
# The golden folds in1's BFP quantization in rather than charging it against PCC: in1 is packed to the BFP format and
# then unpacked back to bf16, and that dequantized tensor is what the golden multiplies. This is what
# compressed_utils.run_compressed does.
#
# Blackhole-only (@blackhole_only): the primitive headers live under the Blackhole experimental/ tree.

import pytest
import torch
from conftest import blackhole_only
from helpers.compressed_utils import (
    FMT_CODE,
    FMT_CODE_BY_DATAFORMAT,
    encode_tile_meta,
    pack_bfp_tile,
    unpack_bfp_tile,
)
from helpers.custom_mm_utils import (
    dense_result_rowmajor,
    matmul_grid,
    pack_in0_faces,
)
from helpers.device import BootMode
from helpers.format_config import DataFormat
from helpers.golden_generators import TILE_DIM, MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import generate_combination, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    CUSTOM_MM_DENSE_PACKING,
    IN_FACE_DIMS,
    NUM_FACES,
    TILE_COUNT,
)
from helpers.tile_constants import FACE_C_DIM
from helpers.utils import matmul_acc_atol, passed_test, tolerances

# in0 (SrcB) is bf16; in1 (SrcA) is the BFP-compressed operand. Sweep the three BFP widths the LLK's exponent-section
# MOP config supports.
IN1_COMPRESSED_FORMATS = [DataFormat.Bfp8_b, DataFormat.Bfp4_b, DataFormat.Bfp2_b]

# generate_combination tuple layout (len 7, same_src_format=False):
#   (unpack_A_src, unpack_A_dst, unpack_B_src, unpack_B_dst, pack_src, pack_dst, math)
COMPRESSED_MM_FORMATS = generate_combination(
    [
        (
            DataFormat.Float16_b,  # in0 (SrcB) L1 format
            DataFormat.Float16_b,  # in0 unpack dst
            in1_fmt,  # in1 (SrcA) L1 format -- BFP-compressed
            DataFormat.Float16_b,  # in1 unpack dst (decompressed to bf16 in SrcA)
            DataFormat.Float16_b,  # pack src
            DataFormat.Float16_b,  # pack dst (output)
            DataFormat.Float16_b,  # math format (LoFi bf16)
        )
        for in1_fmt in IN1_COMPRESSED_FORMATS
    ]
)


def _pack_in1_bfp(in1, kt_dim, ct_dim, codes):
    """BFP-pack in1 tile by tile (row-major over the kt x ct grid) and return (bytes, dequantized).

    codes is the per-tile format code, row-major over the same grid. A zero code (bfp0) packs to no
    bytes and dequantizes to a zero tile, which is exactly what the kernel does with it: the tile is
    absent from the stream and the math skips it.

    The dequantized copy is what the golden multiplies, so BFP rounding is folded into the golden
    rather than charged against PCC -- the same trick compressed_utils.run_compressed uses.
    """
    packed = b""
    dequant = torch.zeros_like(in1)
    for k in range(kt_dim):
        for c in range(ct_dim):
            code = codes[k * ct_dim + c]
            blk = in1[
                k * TILE_DIM : (k + 1) * TILE_DIM, c * TILE_DIM : (c + 1) * TILE_DIM
            ]
            full = pack_bfp_tile(blk, code, tile_dim=TILE_DIM)
            packed += full
            dequant[
                k * TILE_DIM : (k + 1) * TILE_DIM, c * TILE_DIM : (c + 1) * TILE_DIM
            ] = unpack_bfp_tile(full, code, tile_dim=TILE_DIM)
    return packed, dequant


# The full sweep is 3 BFP formats x the 40-point (ct_dim, kt_dim, in0_rows) grid == 120 hardware cases, over the
# repo's 100-combination cap for non-nightly parametrizations (.github/instructions/python.instructions.md), and this
# is the only one of the three matmul advance tests that multiplies the grid by a format axis. So the merge-gate test
# below keeps all three compression formats but only the grid corners (ct_dim and in0_rows at their extremes, both
# kt_dim values) == 24 cases, and the full sweep runs nightly. The two are the same body.
CORNER_GRID = matmul_grid(ct_dims=[1, 16], kt_dims=[2, 4], in0_face_r_dims=[1, 8])


@blackhole_only
@parametrize(
    formats=COMPRESSED_MM_FORMATS,
    ct_kt_rows=CORNER_GRID,
)
def test_compressed_custom_mm(
    formats,
    ct_kt_rows,
    boot_mode=BootMode.DEFAULT,
):
    _run_compressed_custom_mm(formats, ct_kt_rows, boot_mode)


@blackhole_only
@pytest.mark.nightly
@parametrize(
    formats=COMPRESSED_MM_FORMATS,
    ct_kt_rows=matmul_grid(),
)
def test_compressed_custom_mm_full_grid(
    formats,
    ct_kt_rows,
    boot_mode=BootMode.DEFAULT,
):
    _run_compressed_custom_mm(formats, ct_kt_rows, boot_mode)


# Zero-code (bfp0) weight tiles. The unpacker emits no UNPACR for them and the math thread takes its skip branch,
# which only has to carry the DEST counter over the tile's slot without multiplying. The full-width case is the one
# that tells: with dense_packing the 16 output tiles of ct_dim=16 fill the whole DEST half (16 x 32 rows), so a skip
# that disturbs any face of the half lands on a live partial. Zero tiles sit in k-iterations >= 1 so the tiles already
# hold partials from the earlier k, and span the inner columns (the last column takes a different branch, SETRWC):
# the skip used to be a ZEROACC whose row select wraps onto the last face of the half, tile 15's second face, which
# it marked zero for a zero tile at column 9 or later -- the k1-c9, k1-c14, k2-c9 and three-zero cases failed on
# that face while the earlier columns passed. Every case must pass, and a wrong stride in the skip would shift every
# later tile, so the passing columns guard the fix as well.
ZERO_TILE_SHAPE = (16, 4, 8)  # (ct_dim, kt_dim, in0_rows)
ZERO_TILE_CASES = [
    pytest.param(((1, 0),), id="k1-c0"),
    pytest.param(((1, 3),), id="k1-c3"),
    pytest.param(((1, 8),), id="k1-c8"),
    pytest.param(((1, 9),), id="k1-c9"),
    pytest.param(((1, 14),), id="k1-c14"),
    pytest.param(((2, 9),), id="k2-c9"),
    pytest.param(((3, 0),), id="k3-c0"),
    pytest.param(((1, 2), (2, 7), (3, 11)), id="k1-c2_k2-c7_k3-c11"),
]


@blackhole_only
@pytest.mark.parametrize("zero_tiles", ZERO_TILE_CASES)
def test_compressed_custom_mm_zero_tile(zero_tiles):
    """Zero-code weight tiles at ct_dim=16 must leave every other tile's partial intact."""
    _run_compressed_custom_mm(
        COMPRESSED_MM_FORMATS[0], ZERO_TILE_SHAPE, zero_tiles=zero_tiles
    )


# The same with dense_packing=False: 64-row DEST tiles, so ct_dim=8 fills the half and the skip has to step the
# non-dense stride (which only this test exercises -- both silicon kernels above pin dense_packing=true). Here the
# last face of the half is the unused lower half of tile 7, so the old ZEROACC skip did not show; the cases guard the
# stride and the layout. Columns span the ZEROACC's observed dense threshold (DEST counter >= 288 rows) both ways.
ZERO_TILE_SHAPE_NON_DENSE = (8, 4, 8)  # (ct_dim, kt_dim, in0_rows)
ZERO_TILE_CASES_NON_DENSE = [
    pytest.param(((1, 3),), id="k1-c3"),
    pytest.param(((1, 6),), id="k1-c6"),
    pytest.param(((2, 5),), id="k2-c5"),
    pytest.param(((1, 1), (2, 4), (3, 6)), id="k1-c1_k2-c4_k3-c6"),
]


@blackhole_only
@pytest.mark.parametrize("zero_tiles", ZERO_TILE_CASES_NON_DENSE)
def test_compressed_custom_mm_zero_tile_non_dense(zero_tiles):
    """Zero-code weight tiles with the 64-row DEST tile stride, ct_dim=8 filling the half."""
    _run_compressed_custom_mm(
        COMPRESSED_MM_FORMATS[0],
        ZERO_TILE_SHAPE_NON_DENSE,
        zero_tiles=zero_tiles,
        dense_packing=False,
    )


def _run_compressed_custom_mm(
    formats,
    ct_kt_rows,
    boot_mode=BootMode.DEFAULT,
    zero_tiles=(),
    dense_packing=True,
):
    ct_dim, kt_dim, in0_rows = ct_kt_rows
    rt_dim = 1  # compressed_custom_mm contract: rt_dim is always 1
    output_tile_cnt = rt_dim * ct_dim

    torch_format = format_dict[formats.output_format]
    in0_format = formats.unpack_A_src  # bf16
    in1_format = formats.unpack_B_src  # BFP-compressed

    in0_dimensions = [in0_rows, kt_dim * TILE_DIM]
    in1_dimensions = [kt_dim * TILE_DIM, ct_dim * TILE_DIM]

    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=in0_format,
        input_dimensions_A=in0_dimensions,
        stimuli_format_B=in0_format,  # generate in1 in bf16; BFP packing happens below
        input_dimensions_B=in1_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    in0 = src_A.reshape(in0_dimensions).to(torch_format)
    in1 = src_B.reshape(in1_dimensions).to(torch_format)

    # Every tile carries the one BFP format under test, except the (k, c) tiles named in zero_tiles, which are
    # zero-code (bfp0): absent from the stream, zero in the golden.
    codes = [FMT_CODE_BY_DATAFORMAT[in1_format]] * (kt_dim * ct_dim)
    for k, c in zero_tiles:
        codes[k * ct_dim + c] = FMT_CODE["bfp0"]
    packed_b, in1_dequant = _pack_in1_bfp(in1, kt_dim, ct_dim, codes)

    # Golden multiplies the DEQUANTIZED in1, so BFP rounding is folded in rather than charged against PCC.
    # LoFi, not a raw torch matmul: the FPU truncates the SrcA/SrcB mantissas before multiplying, which
    # biases a K-deep sum of positive values low by ~2% -- far outside atol if the golden multiplies at full
    # bf16 precision.
    golden_tensor = get_golden_generator(MatmulGolden)(
        in0,
        in1_dequant,
        formats.output_format,
        MathFidelity.LoFi,
        input_A_dimensions=in0_dimensions,
        input_B_dimensions=in1_dimensions,
        tilize=False,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    ).reshape(in0_dimensions[0], in1_dimensions[1])

    in0_faces = pack_in0_faces(in0, kt_dim, in0_format)
    meta_bytes = encode_tile_meta(codes, ct_dim)

    configuration = TestConfig(
        "sources/compressed_custom_mm_test.cpp",
        formats,
        templates=[CUSTOM_MM_DENSE_PACKING(dense_packing)],
        runtimes=[
            # num_faces_A is in0's active face count and num_faces_B is in1's full 4; the kernel CROSSES them into the
            # unpB / unpA slots. num_faces is the pack count.
            NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4),
            TILE_COUNT(output_tile_cnt),
            CRK_TILE_DIMM(ct_dim, rt_dim, kt_dim),
            IN_FACE_DIMS(in0_face_r_dim=in0_rows),
        ],
        # All three inputs are pre-packed L1 images, so they go to L1 verbatim: in0 is the dense
        # partial-face run, in1 the BFP-compressed stream, buffer_C the meta words. The tile counts
        # and formats below are only what reserves each operand's L1 region -- in1 is declared
        # Bfp8_b (the widest BFP tile, 1088 B) so the reservation covers the stream whichever
        # format this variant packs it in, exactly as compressed_utils.CompressedStimuliConfig does.
        variant_stimuli=StimuliConfig(
            buffer_A=in0_faces,
            stimuli_A_format=in0_format,
            tile_count_A=kt_dim,
            buffer_B=packed_b,
            stimuli_B_format=DataFormat.Bfp8_b,
            tile_count_B=kt_dim * ct_dim,
            buffer_C=meta_bytes,
            stimuli_C_format=DataFormat.UInt32,
            tile_count_C=(len(meta_bytes) + 4095) // 4096,
            stimuli_res_format=formats.output_format,
            tile_count_res=output_tile_cnt,
        ),
        dest_acc=DestAccumulation.No,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result
    res_tensor = dense_result_rowmajor(
        torch.tensor(res_from_L1, dtype=torch_format), ct_dim, in0_rows
    )

    assert (
        res_tensor.numel() == golden_tensor.numel()
    ), "Result tensor and golden tensor are not of the same length"

    # Name the output faces that miss the tolerance half of passed_test's gate (same atol/rtol isclose; PCC is
    # whole-tensor and has no per-face reading), so a failure says where in DEST it landed.
    atol = matmul_acc_atol(golden_tensor, kt_dim)
    rtol = tolerances[formats.output_format].rtol
    faces_per_tile = TILE_DIM // FACE_C_DIM
    by_face = lambda t: t.float().reshape(in0_rows, ct_dim, faces_per_tile, FACE_C_DIM)
    res_faces, golden_faces = by_face(res_tensor), by_face(golden_tensor)
    bad_faces = {}
    for c in range(ct_dim):
        for f in range(faces_per_tile):
            res_face, golden_face = res_faces[:, c, f], golden_faces[:, c, f]
            if not torch.isclose(res_face, golden_face, atol=atol, rtol=rtol).all():
                bad_faces[f"tile{c}.F{f}"] = round(
                    (res_face - golden_face).abs().max().item(), 4
                )
    assert passed_test(
        golden_tensor.flatten(),
        res_tensor.flatten(),
        formats.output_format,
        custom_atol=atol,
    ), f"Assert against golden failed; output faces outside atol={atol:.4g}/rtol={rtol} (max abs err): {bad_faces}"
