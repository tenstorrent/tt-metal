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
#   - the output is a partial tile [in0_rows, ct_dim*32], packed with dense_packing (DEST tile stride 32 rows) and read
#     back through helpers/custom_mm_utils.dense_result_rowmajor.
#
# Difference vs custom_mm: BOTH primitives take an extra base_address_meta argument (a buffer of packed 3-bit per-tile
# compression-format codes). We route it through the harness's optional third operand (buffer_C). The codes are CONTROL
# FLOW, not just numerics: code 0 means "zero tile", for which the unpacker emits no UNPACR at all and math takes the
# STALLWAIT(SRCB_VLD) branch, so an all-zero meta buffer hangs Math/Packer forever. Every tile here carries the one BFP
# format under test.
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
    IN_FACE_DIMS,
    NUM_FACES,
    TILE_COUNT,
)
from helpers.utils import matmul_acc_atol, passed_test

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


def _pack_in1_bfp(in1, kt_dim, ct_dim, code):
    """BFP-pack in1 tile by tile (row-major over the kt x ct grid) and return (bytes, dequantized).

    The dequantized copy is what the golden multiplies, so BFP rounding is folded into the golden
    rather than charged against PCC -- the same trick compressed_utils.run_compressed uses.
    """
    packed = b""
    dequant = torch.zeros_like(in1)
    for k in range(kt_dim):
        for c in range(ct_dim):
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


def _run_compressed_custom_mm(
    formats,
    ct_kt_rows,
    boot_mode=BootMode.DEFAULT,
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

    code = FMT_CODE_BY_DATAFORMAT[in1_format]
    packed_b, in1_dequant = _pack_in1_bfp(in1, kt_dim, ct_dim, code)

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
    # Every tile carries the one BFP format under test, so the assignment is uniform.
    meta_bytes = encode_tile_meta([code] * (kt_dim * ct_dim), ct_dim)

    configuration = TestConfig(
        "sources/compressed_custom_mm_test.cpp",
        formats,
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

    assert passed_test(
        golden_tensor.flatten(),
        res_tensor.flatten(),
        formats.output_format,
        custom_atol=matmul_acc_atol(golden_tensor, kt_dim),
    ), "Assert against golden failed"
